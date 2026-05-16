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
#include "engine.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_forward_context — per-request inference state. Holds the residual
// stream, KV cache, current layer index, working arena.
// -----------------------------------------------------------------------
struct sp_forward_context {
    sp_ok_tensor x;          // residual stream  [n_embd, batch]
    sp_ok_tensor x_norm;     // post-RMSNorm scratch
    sp_ok_tensor q;          // Q-projection scratch
    sp_ok_tensor k;          // K-projection scratch
    sp_ok_tensor v;          // V-projection scratch
    sp_ok_tensor attn_out;   // attention output scratch
    sp_ok_tensor ffn_out;    // FFN output scratch

    sp_ok_arena  arena;      // per-step scratch arena (reset between layers)

    int          n_layers;
    int          n_embd;
    int          n_head;
    int          n_kv_head;
    int          head_dim;

    // Poncelet adaptive depth tracking (Paper A §7, Theorem 5).
    // Partial sum of layer-endomorphisms in O_K. When the partial sum
    // vanishes modulo the working prime, sp_forward_step exits early.
    sp_ok_t      poncelet_delta;
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
// Phase 1.6 SKELETON: this delegates to forward.cpp with weights that
// have been encoded → Frobenius-applied → decoded back to fp16. The
// "pure" sp_forward (no decode step) is Phase 2 work.
void sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out);

// Initialize a forward context for a given model.
void sp_forward_context_init(sp_forward_context& ctx,
                              const Config&       cfg,
                              int                 n_embd,
                              int                 n_layers,
                              int                 n_head,
                              int                 n_kv_head);

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
