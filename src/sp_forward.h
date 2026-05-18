// Shannon-Prime Engine ? Theory-First forward pass (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The pure-SP forward pass. Every step is an endomorphism of E^n (or
// the Siegel-variety multi-head generalization, Paper A ?3) realized
// as a sequence of sp_ok_tensor operations. ggml is NOT used.
//
// In Phase 1 SKELETON (this file), the implementation delegates to
// existing forward.cpp / forward_native.cpp with weights pre-encoded
// through sp_ok_encode (the Frobenius shim). Phase 1.6 work fills in
// the actual SP-native ops.
//
// Reference: docs/THEORY-FIRST-ENGINE-DESIGN.md ?Forward pass.

#pragma once

/* System headers FIRST -- engine headers below open namespace sp::engine
 * and MSVC will choke on follow-up <cmath>/<algorithm> inclusion with
 * either ADL-pollution errors (sp::engine::std::pair) or UCRT __std_smf_*
 * link-only symbols missing. Including the standard library at root
 * scope BEFORE any engine namespace opens fixes both. */
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "sp_ok_tensor.h"
#include "sp_kv_cache_ok.h"
#include "sp_ffn.h"
#include "sp_rope.h"
#include "engine.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_forward_context ? per-request inference state.
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
// Forward declaration so sp_forward_context can hold a pointer to the
// prefetcher without including the implementation here. Defined below
// after sp_weights so it can refer to the struct.
struct sp_weights;

// -----------------------------------------------------------------------
// Phase 12 Step C: background Q8 prefetch worker.
//
// A dedicated std::thread that races ahead of the forward path, decoding
// each layer's 7 weight tensors from packed Q8 storage into one of two
// double-buffered slots. The forward thread acquires layer L's slot,
// runs all 7 matmuls reading directly from it, then releases. The
// worker is woken by `cv_consumed_` whenever a slot becomes free and
// signals `cv_decoded_` whenever a slot becomes ready.
//
// Producer/consumer ordering invariant: the consumer accesses layers in
// the fixed sequence (0, 1, ..., n_layers-1, 0, 1, ...). The producer
// maintains `next_layer_to_decode_` which monotonically advances mod
// n_layers, always one or two layers ahead of the consumer. The step
// boundary (consumer L=N-1 -> L=0) requires no special handling -- the
// producer's modulo counter wraps naturally.
//
// Memory: each slot holds one layer's decoded weights (~430 MB on
// Gemma3-1B). 2 slots = ~860 MB of live decode workspace. Combined
// with the 1.3 GB Q8 resident storage, peak runtime memory is ~2.16 GB
// versus the original 10.4 GB sp_ok_t arena -- a 4.8x live RAM
// compression while keeping wall time within striking distance of the
// uncompressed Phase 11 path.
//
// Bit-identical to Step B-2's decode-on-demand path: the prefetcher
// just runs the same sp_ok_q8_decode_array on the same packed bytes,
// only ahead of time instead of inline.
// -----------------------------------------------------------------------
class sp_q8_prefetcher {
public:
    sp_q8_prefetcher() = default;
    ~sp_q8_prefetcher() { stop(); }
    sp_q8_prefetcher(const sp_q8_prefetcher&)            = delete;
    sp_q8_prefetcher& operator=(const sp_q8_prefetcher&) = delete;

    // Reserve slot arenas, launch worker. weights.use_q8 must be true.
    // Returns false if weights is empty or not in Q8 mode.
    bool start(const sp_weights& weights);

    // Signal shutdown, join the worker. Safe to call multiple times.
    void stop();

    // Blocks until layer L is fully decoded in some slot. Returns the
    // slot index (0 or 1). Caller MUST call release(slot) after all
    // matmul reads from this slot complete.
    int acquire(int layer);

    // Mark slot as consumed. Worker may now overwrite it with the next
    // layer's decoded weights.
    void release(int slot);

    // Accessors for the 7 decoded tensors in a given slot. Valid
    // between acquire() and release().
    const sp_ok_tensor& wq      (int slot) const { return slots_[slot].wq;       }
    const sp_ok_tensor& wk      (int slot) const { return slots_[slot].wk;       }
    const sp_ok_tensor& wv      (int slot) const { return slots_[slot].wv;       }
    const sp_ok_tensor& wo      (int slot) const { return slots_[slot].wo;       }
    const sp_ok_tensor& ffn_gate(int slot) const { return slots_[slot].ffn_gate; }
    const sp_ok_tensor& ffn_up  (int slot) const { return slots_[slot].ffn_up;   }
    const sp_ok_tensor& ffn_down(int slot) const { return slots_[slot].ffn_down; }

    // Diagnostics: per-call decode time accumulator (microseconds).
    uint64_t decode_us_total() const { return decode_us_total_.load(); }
    uint64_t acquire_wait_us_total() const { return acquire_wait_us_total_.load(); }

private:
    void worker_loop();
    void decode_layer_into(int slot, int layer);

    static constexpr int N_SLOTS = 2;
    struct slot_t {
        sp_ok_arena   arena;
        sp_ok_tensor  wq, wk, wv, wo;
        sp_ok_tensor  ffn_gate, ffn_up, ffn_down;
        int           layer_id = -1;   // -1 = empty
        bool          ready    = false;
    };
    slot_t slots_[N_SLOTS];

    const sp_weights* weights_ = nullptr;

    std::mutex              mu_;
    std::condition_variable cv_decoded_;     // slot became ready
    std::condition_variable cv_consumed_;    // slot became empty

    std::thread        worker_;
    std::atomic<bool>  shutdown_{false};
    int                next_layer_to_decode_ = 0;

    std::atomic<uint64_t> decode_us_total_{0};
    std::atomic<uint64_t> acquire_wait_us_total_{0};
};

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
    float        rope_base  = 10000.0f;
    sp_rope_mode rope_mode  = sp_rope_mode::NORMAL;  // NEOX for qwen/phi/gemma family
    // Phase 2.3b iter 2 ? Gemma family arch knobs:
    float       embd_scale = 1.0f;                 // sqrt(n_embd) for gemma
    sp_ffn_act  ffn_act    = sp_ffn_act::SwiGLU;   // GeGLU_tanh for gemma
    // Phase 2.3b iter 3 ? Gemma3 SWA / softcap knobs (0 = disabled).
    // Gemma3 alternates 5 SWA "local" layers : 1 "global" layer.
    // A layer is SWA-local iff (L + 1) % swa_pattern_period != 0
    // AND swa_window > 0. Local layers use `swa_rope_base` (typically
    // 10000); global layers use `rope_base` (typically 1e6).
    int    swa_window           = 0;
    float  swa_rope_base        = 10000.0f;
    int    swa_pattern_period   = 6;       // gemma3 default
    float  attn_logit_softcap   = 0.0f;    // > 0 caps QK^T/sqrt(d)
    float  final_logit_softcap  = 0.0f;    // > 0 caps LM-head output

    // Phase 3 pivot ? attention dispatch.
    //   0 = standard dot product (Phase 2.2a)
    //   1 = CKKS polynomial-ring (Z[x]/(x^N+1) negacyclic convolution)
    int    attn_mode            = 0;

    // Phase 9b (post Plan C): dual-prime CRT NTT-domain K cache is now
    // the only NTT path the engine calls. SoA layout, two parallel
    // slabs sized [n_layers * n_kv_head * n_ctx * SP_NTT_CRT_N] uint64s.
    // Populated at K-append time, read by sp_attention_poly_ring under
    // SP_ENGINE_POLY_NTT_CRT=1 (default-on when SP_ENGINE_POLY_NTT=1).
    // The 60-bit single-prime sp_ntt path is kept only as the parity
    // test reference (see test_sp_ntt + test_sp_ntt_crt); engine never
    // calls into it. Empty slabs means "scalar O(N^2) fallback".
    std::vector<uint64_t> k_ntt_cache_q1;
    std::vector<uint64_t> k_ntt_cache_q2;
    bool                  k_ntt_crt = false;

    // Poncelet adaptive depth tracking (Paper A ?7, Theorem 5).
    sp_ok_t poncelet_delta = sp_ok_t{ 0, 0 };


    // Phase 12 Step C: background prefetch worker. Decodes packed Q8
    // weights one layer ahead of the forward path into double-buffered
    // slots. Initialized when weights.use_q8 is true at context init,
    // stopped on context destruct.
    sp_q8_prefetcher q8_prefetcher;

    // Phase 13.C: optional long-term memory bank. When `lt_mem` is non-null
    // and `lt_mem_alpha > 0`, the forward path will recall a v_hat from the
    // bank per (layer, head, token) after attention and add `alpha * v_hat`
    // to attn_out_ok before the Wo projection.
    //
    // Pointer-only — the bank is owned by the caller (the CLI), so the
    // forward context doesn't manage its lifetime. Stays null in normal
    // (non-memory) inference.
    struct sp_lt_memory* lt_mem        = nullptr;
    float                lt_mem_alpha  = 0.0f;
    double               lt_mem_norm_thr = 1.0;

    // Phase 13.C recall scratch: per-(layer, head, token) Q decode and
    // v_hat output. Length = head_dim each (lazily sized when lt_mem
    // is engaged).
    std::vector<float> lt_q_decode;
    std::vector<float> lt_v_hat;

    // Phase 12 Step B-2: decode workspace for resident Q8 weights.
    // sp_weights with use_q8=true releases its layer_arenas back to the
    // OS; the matmul kernels still expect sp_ok_t pointers, so we lazily
    // decode each Q8 weight tensor into one of these scratches at the
    // matmul call site. Q/K/V/O matmuls reuse `q8_decode_scratch` (arena
    // reset between calls). The FFN call needs three weights live
    // simultaneously (gate, up, down) so it uses all three scratches
    // without a reset between them; the arena is sized for the sum of
    // those three (~380 MB on Gemma3-1B).
    sp_ok_arena   q8_decode_arena;
    sp_ok_tensor  q8_decode_scratch;
    sp_ok_tensor  q8_decode_scratch_b;
    sp_ok_tensor  q8_decode_scratch_c;
};

// -----------------------------------------------------------------------
// sp_weights ? per-model weight tensors in O_K coordinates.
//
// All MATMUL weights are sp_ok_tensors (shim-list, get Frobenius-shimmed).
// All RMSNORM scale vectors stay fp32 (bypass-list, no shim ? they are
// the scale-reset valve per Phase 1.7 policy).
// -----------------------------------------------------------------------
struct sp_weights {
    // Phase 2.3b iter 5: tok_embed and lm_head are BYPASS-list (Phase 1.7
    // policy). They have frobenius_scale=1 and b=0 by definition, so the
    // 16-B-per-element O_K representation is pure waste. Stored as fp32
    // vectors to save ~9.6 GB on a 1B-parameter model (vocab=262144,
    // n_embd=1152). The matmul path bridges through fp32 directly.
    std::vector<float> tok_embed_fp32;         // [vocab * n_embd]
    std::vector<float> lm_head_fp32;           // [vocab * n_embd]

    // Shim-list matmul operands (per-layer, still in O_K because these
    // are exactly the tensors where Theorem 4 cancellation runs):
    std::vector<sp_ok_tensor> wq;              // per-layer Q projection [n_embd, d_q]
    std::vector<sp_ok_tensor> wk;              //                        [n_embd, d_kv]
    std::vector<sp_ok_tensor> wv;              //                        [n_embd, d_kv]
    std::vector<sp_ok_tensor> wo;              // attn output proj       [d_q,   n_embd]
    std::vector<sp_ok_tensor> ffn_gate;        //                        [n_embd, d_ff]
    std::vector<sp_ok_tensor> ffn_up;          //                        [n_embd, d_ff]
    std::vector<sp_ok_tensor> ffn_down;        //                        [d_ff,   n_embd]

    // Phase 12 Step B-2: resident packed-int8 storage. When use_q8 is true,
    // the shim-list sp_ok_tensors above have their data pointers cleared
    // (their layer arenas are released back to the OS) and the matmul
    // weights live in these q8_* tensors instead. The forward path lazily
    // decodes them into a per-call scratch (sp_forward_context::q8_decode_*).
    bool                              use_q8 = false;
    std::vector<sp_ok_q8_tensor>      q8_wq;       // post-Frobenius packed
    std::vector<sp_ok_q8_tensor>      q8_wk;
    std::vector<sp_ok_q8_tensor>      q8_wv;
    std::vector<sp_ok_q8_tensor>      q8_wo;
    std::vector<sp_ok_q8_tensor>      q8_ffn_gate;
    std::vector<sp_ok_q8_tensor>      q8_ffn_up;
    std::vector<sp_ok_q8_tensor>      q8_ffn_down;
    std::vector<sp_ok_arena>          q8_layer_arenas;  // ~1/8 of layer_arenas

    // Phase 14: same idea as q8 but with 4-bit packed nybble pairs (1 byte
    // per ring element, 16x compression vs raw sp_ok_t). use_q4 implies the
    // sp_ok_tensor data pointers above are nullptr and the fused
    // sp_matmul_ok_q4 dispatch is used in forward. Mutually exclusive with
    // use_q8 — at most one of {use_q8, use_q4} is true at any time.
    bool                              use_q4 = false;
    std::vector<sp_ok_q4_tensor>      q4_wq;
    std::vector<sp_ok_q4_tensor>      q4_wk;
    std::vector<sp_ok_q4_tensor>      q4_wv;
    std::vector<sp_ok_q4_tensor>      q4_wo;
    std::vector<sp_ok_q4_tensor>      q4_ffn_gate;
    std::vector<sp_ok_q4_tensor>      q4_ffn_up;
    std::vector<sp_ok_q4_tensor>      q4_ffn_down;
    std::vector<sp_ok_arena>          q4_layer_arenas;  // ~1/16 of layer_arenas

    // Phase 15: GGUF block-quant storage. Mutually exclusive with
    // use_q8 and use_q4 (only one packed-storage flag is true at a
    // time). When use_block_q8 (or _q4) is set, the standard wq/wk/wv
    // etc. sp_ok_tensor descriptors retain only shape + scale metadata;
    // data ptr is null, and the matmul dispatch picks block_q8 /
    // block_q4 kernels.
    bool                                use_block_q8 = false;
    std::vector<sp_ok_block_q8_tensor>  block_q8_wq;
    std::vector<sp_ok_block_q8_tensor>  block_q8_wk;
    std::vector<sp_ok_block_q8_tensor>  block_q8_wv;
    std::vector<sp_ok_block_q8_tensor>  block_q8_wo;
    std::vector<sp_ok_block_q8_tensor>  block_q8_ffn_gate;
    std::vector<sp_ok_block_q8_tensor>  block_q8_ffn_up;
    std::vector<sp_ok_block_q8_tensor>  block_q8_ffn_down;
    std::vector<sp_ok_arena>            block_q8_layer_arenas;

    bool                                use_block_q4 = false;
    std::vector<sp_ok_block_q4_tensor>  block_q4_wq;
    std::vector<sp_ok_block_q4_tensor>  block_q4_wk;
    std::vector<sp_ok_block_q4_tensor>  block_q4_wv;
    std::vector<sp_ok_block_q4_tensor>  block_q4_wo;
    std::vector<sp_ok_block_q4_tensor>  block_q4_ffn_gate;
    std::vector<sp_ok_block_q4_tensor>  block_q4_ffn_up;
    std::vector<sp_ok_block_q4_tensor>  block_q4_ffn_down;
    std::vector<sp_ok_arena>            block_q4_layer_arenas;

    /* Phase 15b: Q4_1 storage. Per-tensor dispatch — set on a per-LAYER
     * basis via the *_is_q4_1 vectors below, since mixed-quant GGUFs
     * (typical llama-quantize Q4_0 output) put Q4_0 on most tensors and
     * Q4_1 on a subset (typically ffn_down). The forward dispatch
     * checks block_q4_1_*[L].blocks != nullptr to route per-tensor. */
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_wq;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_wk;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_wv;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_wo;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_ffn_gate;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_ffn_up;
    std::vector<sp_ok_block_q4_1_tensor>  block_q4_1_ffn_down;
    std::vector<sp_ok_arena>              block_q4_1_layer_arenas;

    // Bypass-list (fp32 norms; scale-reset valve per Phase 1.7 policy):
    std::vector<std::vector<float>> attn_norm_w;       // per-layer [n_embd]
    std::vector<std::vector<float>> ffn_norm_w;        // per-layer [n_embd]
    std::vector<float>              final_norm_w;      // [n_embd]
    // Phase 2.3b: Gemma3 / Qwen3 per-head Q/K norms (empty = not applied).
    // Sized [head_dim]; shared across all heads (broadcast per head).
    std::vector<std::vector<float>> attn_q_norm_w;     // per-layer [head_dim] or empty
    std::vector<std::vector<float>> attn_k_norm_w;     // per-layer [head_dim] or empty
    // Phase 2.3b: Gemma3 sandwich norms ? applied to projection output
    // BEFORE the residual add. Empty = not applied.
    std::vector<std::vector<float>> attn_post_norm_w;  // per-layer [n_embd] or empty
    std::vector<std::vector<float>> ffn_post_norm_w;   // per-layer [n_embd] or empty

    // Owning storage. Phase 2.3b iter 5: one arena per layer (typically
    // 400 MB - 1 GB each) instead of a single huge contiguous block.
    // Windows heap fragmentation makes a 10 GB+ malloc unreliable even
    // when total free RAM is sufficient; per-layer arenas sidestep
    // that without changing the algebra. layer_arenas[L] owns the
    // backing memory for wq/wk/wv/wo/ffn_gate/ffn_up/ffn_down on layer L.
    std::vector<sp_ok_arena>  layer_arenas;

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

// Phase 12 Step B-2: convert shim-list sp_ok_tensors to resident packed
// int8 storage. Allocates q8_layer_arenas (8x smaller than the live
// arenas), packs every shim-list tensor (post-Frobenius coordinates) into
// the corresponding q8_*[L] descriptor with a per-tensor power-of-2 shift,
// then releases the original layer_arenas[L] back to the OS. After this
// runs:
//   - weights.wq[L].data == nullptr (and friends) -- the unpacked sp_ok_t
//     buffers no longer exist; any matmul that tries to read them
//     directly will crash. Forward code must check weights.use_q8 and
//     dispatch through the decode scratch in sp_forward_context.
//   - weights.use_q8 == true
//   - weights.q8_wq[L].numel == old weights.wq[L].numel
//   - weights.q8_wq[L] carries scale_recip / frobenius_scale / p / k.
// Returns the count of tensors packed. Bypass-list tensors (tok_embed,
// lm_head, norms) are NOT affected. Idempotent: calling on an already-
// converted sp_weights is a no-op.
int sp_weights_convert_to_q8(sp_weights& weights);

// Phase 14: same as sp_weights_convert_to_q8, but packs each shim-list
// tensor's post-Frobenius coordinates into 4-bit nybble pairs (1 byte per
// ring element). After this runs:
//   - weights.wq[L].data == nullptr (etc.)
//   - weights.use_q4 == true, use_q8 == false (mutually exclusive)
//   - weights.q4_wq[L].numel == old wq[L].numel
//   - weights.q4_wq[L] carries q4_shift + scale_recip + frobenius_scale + p + k.
// Returns the count of tensors packed. Idempotent.
int sp_weights_convert_to_q4(sp_weights& weights,
                              uint64_t    prune_threshold = 0);

// Phase 15: ingest GGUF Q8_0 / Q4_0 tensors directly into block-fused
// storage. Caller passes the LlamaWeights walker plus the Frobenius
// (p, k); the function reads each tensor's ggml type, branches to the
// appropriate sp_ok_block_q{8,4}_from_gguf_q{8,4}_0 importer, and
// populates weights.block_q{8,4}_* slot vectors.
//
// Tensors that aren't Q8_0 or Q4_0 (e.g. fp16 norms, fp32 embed) are
// left to the existing fp16 path -- the caller is expected to invoke
// sp_weights_load_from_llama for norms + embeddings first, then this
// function to overlay the block-quant weight storage on top.
//
// Returns count of tensors successfully fused. Sets weights.use_block_q8
// and / or weights.use_block_q4 based on what was actually consumed.
class LlamaWeights;
int sp_weights_ingest_gguf_block_quant(sp_weights&         weights,
                                         const LlamaWeights& src,
                                         int64_t             p,
                                         int64_t             k,
                                         int64_t             scale_recip);

// Run a single forward step: given a token id, produce logits[vocab].
//
// Phase 2.2d (LIVE):
//   1. Embedding lookup: weights.tok_embed[token_id] ? x_fp32 (n_embd)
//   2. For each layer L:
//      a. encode x_fp32 ? x_ok (scale_recip=residual_scale, frob=1)
//      b. x_norm_ok = sp_rmsnorm_native(x_ok, attn_norm_w[L])
//      c. q_ok = Wq[L] @ x_norm_ok  (frob=pi^k)
//         k_ok = Wk[L] @ x_norm_ok  (frob=pi^k)
//         v_ok = Wv[L] @ x_norm_ok  (frob=pi^k)
//      d. sp_rope_apply_ok(q_ok); sp_rope_apply_ok(k_ok)   (frob ? 1)
//      e. kv_cache.append(L, k_ok, v_ok)
//      f. attn_out_ok = attention(q_ok, K_view, V_view, ...)  (frob=1)
//      g. wo_out_fp32 = sp_matmul_ok_to_fp32(Wo[L], attn_out_ok)
//      h. x_fp32 += wo_out_fp32                            (residual)
//      i. encode x_fp32 ? x_ok
//      j. x_norm2_ok = sp_rmsnorm_native(x_ok, ffn_norm_w[L])
//      k. ffn_out_fp32 = sp_ffn_swiglu_to_fp32(x_norm2_ok, gate, up, down)
//      l. x_fp32 += ffn_out_fp32                           (residual)
//   3. encode x_fp32 ? x_ok
//   4. x_final_ok = sp_rmsnorm_native(x_ok, final_norm_w)
//   5. logits_fp32 = sp_matmul_ok_to_fp32(lm_head, x_final_ok)
//   6. write logits_fp32 ? logits_out
//
// Single-token mode (n_tokens=1). Thin wrapper that calls
// sp_forward_step_prefill with n_tokens=1; the two are mathematically
// equivalent and bit-identical (at n_tokens=1 every layout formula
// collapses to its single-token form).
bool sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out);

// Phase 12 Step E: multi-token prefill. Processes n_tokens tokens
// (token_ids[0..n_tokens-1]) at sequential positions [position_base,
// position_base + n_tokens - 1] through the same forward pass as
// sp_forward_step, but each layer's weight reads (W_q/W_k/W_v/W_o/
// W_gate/W_up/W_down) are amortized across all n_tokens query tokens.
// That is the production win for 8B+ models at ctx 4K+: the 1.3 GB
// Q8 weight band streams through DRAM ONCE per layer per chunk,
// not once per token.
//
// Output:
//   logits_out is resized to n_tokens * vocab. The logit row for token
//   t (the t-th token in this prefill call) is at
//     logits_out.data() + t * vocab
//   For perplexity bench's needs, this is the row at position
//   (position_base + t) in the model's output stream.
//
// Constraints:
//   - ctx.x_fp32 / ctx.proj_out_fp32 must be sized at least
//     n_ctx * n_embd at context-init time. sp_forward_context_init
//     allocates n_ctx-sized buffers; the single-token path uses only
//     the first n_embd elements, the prefill path uses n_tokens *
//     n_embd contiguous floats.
//   - KV cache must have room: kv_cache.cur_len + n_tokens <= n_ctx.
//   - 1 <= n_tokens <= n_ctx.
//
// At n_tokens=1 this function is bit-identical to sp_forward_step.
bool sp_forward_step_prefill(sp_forward_context& ctx,
                              const sp_weights&   weights,
                              const int*          token_ids,
                              int                 n_tokens,
                              int                 position_base,
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
//
// `head_dim` is INDEPENDENT of n_embd / n_head ? for Gemma3 (n_embd=640,
// n_head=4, head_dim=256) we have d_q = n_head*head_dim = 1024 which is
// strictly larger than n_embd. For most archs head_dim == n_embd/n_head;
// pass head_dim=0 to use that default.
bool sp_weights_alloc(sp_weights& out, int n_layers, int n_embd,
                       int n_head, int n_kv_head, int d_ff, int vocab,
                       int64_t scale_recip,
                       int head_dim = 0);

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

// Phase 2.3b: optional Gemma3 / Qwen3 norms. Pass null `src` (or skip
// the call entirely) to leave the slot empty (= norm not applied).
// q/k norms are sized [head_dim]; post norms are sized [n_embd].
bool sp_weights_set_attn_q_norm(sp_weights& out, int layer, const float* src);
bool sp_weights_set_attn_k_norm(sp_weights& out, int layer, const float* src);
bool sp_weights_set_attn_post_norm(sp_weights& out, int layer, const float* src);
bool sp_weights_set_ffn_post_norm(sp_weights& out, int layer, const float* src);

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
