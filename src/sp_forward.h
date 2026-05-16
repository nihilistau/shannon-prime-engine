// Shannon-Prime Engine — Theory-First forward pass (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The pure-SP forward pass. Every step is an endomorphism of E^n (or
// the Siegel-variety multi-head generalization, Paper A §3) realized
// as a sequence of sp_ok_tensor operations. ggml is NOT used.
//
// In Phase 1 SKELETON (this file), the implementation delegates to
// existing forward.cpp / forward_native.cpp with weights pre-encoded
// through sp_ok_encode (the Frobenius shim). Phase 1.6 work fills in
// the actual SP-native ops.
//
// Reference: docs/THEORY-FIRST-ENGINE-DESIGN.md §Forward pass.

#pragma once

#include "sp_ok_tensor.h"
#include "sp_kv_cache_ok.h"
#include "engine.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_forward_context — per-request inference state.
//
// The residual stream lives in fp32 (`x_fp32`). At the start of each
// layer we encode it into the O_K mirror `x_ok` for the RMSNorm + Q/K/V
// matmuls. The output projections (Wo, ffn_down) run through
// sp_matmul_ok_to_fp32 so their Frobenius factor is divided out cleanly,
// and the result lands directly in fp32 for residual addition. This is
// the design called out in the Phase 2.2d watchout: every residual-add
// crosses through an explicit fp32 island so scale_recip / frobenius_
// scale mismatches can never silently corrupt the stream.
// -----------------------------------------------------------------------
struct sp_forward_context {
    // fp32 residual stream [n_tokens * n_embd].
    std::vector<float> x_fp32;
    // fp32 buffers for post-projection output (n_tokens * n_embd).
    std::vector<float> proj_out_fp32;
    // fp32 buffer for the final logits [vocab].
    std::vector<float> logits_fp32;

    // Per-layer working tensors (mirrors of x_fp32 in O_K + matmul scratch).
    sp_ok_tensor x_ok;         // encoded residual stream, scale_recip=S, frob=1
    sp_ok_tensor x_norm_ok;    // post-RMSNorm
    sp_ok_tensor q_ok;
    sp_ok_tensor k_ok;
    sp_ok_tensor v_ok;
    sp_ok_tensor attn_out_ok;
    sp_ok_arena  layer_arena;  // reset per layer

    // KV cache (lives across decode steps).
    sp_ok_kv_cache kv_cache;
    sp_ok_arena    kv_arena;

    int     n_layers   = 0;
    int     n_embd     = 0;
    int     n_head     = 0;
    int     n_kv_head  = 0;
    int     head_dim   = 0;
    int     n_ctx      = 0;       // max cache len
    int64_t residual_scale = 0;   // scale_recip for x_ok encoding
    float   rms_eps    = 1e-5f;
    float   rope_base  = 10000.0f;

    // Poncelet adaptive depth tracking (Paper A §7, Theorem 5).
    sp_ok_t poncelet_delta = sp_ok_t{ 0, 0 };
};

// -----------------------------------------------------------------------
// sp_weights — per-model weight tensors in O_K coordinates.
//
// All MATMUL weights are sp_ok_tensors (shim-list, get Frobenius-shimmed).
// All RMSNORM scale vectors stay fp32 (bypass-list, no shim — they are
// the scale-reset valve per Phase 1.7 policy).
// -----------------------------------------------------------------------
struct sp_weights {
    // Shim-list weights (matmul operands, shimmed):
    sp_ok_tensor              tok_embed;       // [n_embd, vocab]
    std::vector<sp_ok_tensor> wq;              // per-layer Q projection [n_embd, d_q]
    std::vector<sp_ok_tensor> wk;              //                        [n_embd, d_kv]
    std::vector<sp_ok_tensor> wv;              //                        [n_embd, d_kv]
    std::vector<sp_ok_tensor> wo;              // attn output proj       [d_q,   n_embd]
    std::vector<sp_ok_tensor> ffn_gate;        //                        [n_embd, d_ff]
    std::vector<sp_ok_tensor> ffn_up;          //                        [n_embd, d_ff]
    std::vector<sp_ok_tensor> ffn_down;        //                        [d_ff,   n_embd]
    sp_ok_tensor              lm_head;         // [n_embd, vocab]

    // Bypass-list (fp32 norms; scale-reset valve per Phase 1.7 policy):
    std::vector<std::vector<float>> attn_norm_w;   // per-layer [n_embd]
    std::vector<std::vector<float>> ffn_norm_w;    // per-layer [n_embd]
    std::vector<float>              final_norm_w;  // [n_embd]

    // Owning storage for every sp_ok_tensor above.
    sp_ok_arena               storage;

    // Model dims (set at alloc time).
    int n_layers  = 0;
    int n_embd    = 0;
    int n_head    = 0;
    int n_kv_head = 0;
    int head_dim  = 0;
    int d_ff      = 0;
    int vocab     = 0;
    int64_t scale_recip = 0;  // common encoding scale
};

// -----------------------------------------------------------------------
// Top-level forward functions.
// -----------------------------------------------------------------------

// Run a single forward step: given a token id, produce logits[vocab].
//
// Phase 2.2d (LIVE):
//   1. Embedding lookup: weights.tok_embed[token_id] → x_fp32 (n_embd)
//   2. For each layer L:
//      a. encode x_fp32 → x_ok (scale_recip=residual_scale, frob=1)
//      b. x_norm_ok = sp_rmsnorm_native(x_ok, attn_norm_w[L])
//      c. q_ok = Wq[L] @ x_norm_ok  (frob=pi^k)
//         k_ok = Wk[L] @ x_norm_ok  (frob=pi^k)
//         v_ok = Wv[L] @ x_norm_ok  (frob=pi^k)
//      d. sp_rope_apply_ok(q_ok); sp_rope_apply_ok(k_ok)   (frob → 1)
//      e. kv_cache.append(L, k_ok, v_ok)
//      f. attn_out_ok = attention(q_ok, K_view, V_view, ...)  (frob=1)
//      g. wo_out_fp32 = sp_matmul_ok_to_fp32(Wo[L], attn_out_ok)
//      h. x_fp32 += wo_out_fp32                            (residual)
//      i. encode x_fp32 → x_ok
//      j. x_norm2_ok = sp_rmsnorm_native(x_ok, ffn_norm_w[L])
//      k. ffn_out_fp32 = sp_ffn_swiglu_to_fp32(x_norm2_ok, gate, up, down)
//      l. x_fp32 += ffn_out_fp32                           (residual)
//   3. encode x_fp32 → x_ok
//   4. x_final_ok = sp_rmsnorm_native(x_ok, final_norm_w)
//   5. logits_fp32 = sp_matmul_ok_to_fp32(lm_head, x_final_ok)
//   6. write logits_fp32 → logits_out
//
// Single-token mode (n_tokens=1). Multi-token prefill lands in 2.2d2.
bool sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out);

// Initialize a forward context for a given model. Allocates KV cache,
// scratch arenas, and the residual-stream buffers. Reads V's expected
// frobenius_scale from weights.wv[0] so the V cache slot matches.
//
// `n_ctx`: maximum KV cache length (typically Config::n_ctx).
// `rope_base`, `rms_eps`: per-model hyperparameters.
bool sp_forward_context_init(sp_forward_context& ctx,
                              const sp_weights&   weights,
                              int                 n_ctx,
                              float               rope_base = 10000.0f,
                              float               rms_eps   = 1e-5f);

// Initialize sp_weights by encoding fp16 weights from a loaded model.
// Returns true on success. Weights remain valid for the lifetime of
// `out` (the arena owns the backing storage).
bool sp_weights_init_from_fp16(sp_weights& out,
                                /* loaded model fp16 weight handle */ const void* loaded_model,
                                const Config& cfg);

// -----------------------------------------------------------------------
// Phase 2.2b unit-test API: build sp_weights from raw fp32 buffers.
//
// Step 1: allocate every slot with the right shape, sets scale_recip.
// Step 2..N: set each slot from an fp32 buffer (encodes to O_K).
// Step (final): apply Frobenius / Sato-Tate shim per config.
// -----------------------------------------------------------------------

// Compute the arena size needed to hold every sp_ok_tensor for the given
// dims. Returns bytes.
size_t sp_weights_required_arena_bytes(int n_layers, int n_embd,
                                         int n_head, int n_kv_head,
                                         int d_ff, int vocab);

// Allocate all slots at the given shapes; data is uninitialised.
bool sp_weights_alloc(sp_weights& out, int n_layers, int n_embd,
                       int n_head, int n_kv_head, int d_ff, int vocab,
                       int64_t scale_recip);

// Per-slot setters; the slot must have been allocated by sp_weights_alloc.
// Source layout (matching the slot shape in sp_weights comments):
//   tok_embed  : [vocab, n_embd]      row-major; src[tok * n_embd + d]
//   wq         : [d_q, n_embd]        row-major; src[i * n_embd + k]
//   wk         : [d_kv, n_embd]                  src[i * n_embd + k]
//   wv         : [d_kv, n_embd]                  src[i * n_embd + k]
//   wo         : [n_embd, d_q]                   src[i * d_q + k]
//   ffn_gate   : [d_ff, n_embd]                  src[i * n_embd + k]
//   ffn_up     : [d_ff, n_embd]                  src[i * n_embd + k]
//   ffn_down   : [n_embd, d_ff]                  src[i * d_ff + k]
//   lm_head    : [vocab, n_embd]                 src[i * n_embd + k]
//
// The src layout matches GGUF row-major: row index = output unit i, col
// index = input dim k. We convert into sp_matmul's W-shape on the fly.
bool sp_weights_set_tok_embed(sp_weights& out, const float* src);
bool sp_weights_set_wq(sp_weights& out, int layer, const float* src);
bool sp_weights_set_wk(sp_weights& out, int layer, const float* src);
bool sp_weights_set_wv(sp_weights& out, int layer, const float* src);
bool sp_weights_set_wo(sp_weights& out, int layer, const float* src);
bool sp_weights_set_ffn_gate(sp_weights& out, int layer, const float* src);
bool sp_weights_set_ffn_up(sp_weights& out, int layer, const float* src);
bool sp_weights_set_ffn_down(sp_weights& out, int layer, const float* src);
bool sp_weights_set_lm_head(sp_weights& out, const float* src);

// Bypass-list (fp32) setters.
bool sp_weights_set_attn_norm(sp_weights& out, int layer, const float* src);
bool sp_weights_set_ffn_norm(sp_weights& out, int layer, const float* src);
bool sp_weights_set_final_norm(sp_weights& out, const float* src);

// Apply the Frobenius / Sato-Tate shim to every shim-list tensor. The
// bypass-list (norms) is NOT touched. Returns the number of tensors that
// were transformed.
int sp_weights_apply_frobenius_shim(sp_weights& out,
                                      bool frobenius_quant,
                                      bool sato_tate_mix,
                                      int64_t p,  int64_t k,
                                      int64_t p1, int64_t k1,
                                      int64_t p2, int64_t k2);

}  // namespace sp::engine
