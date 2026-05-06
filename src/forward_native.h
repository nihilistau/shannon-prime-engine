// Shannon-Prime Engine — native forward pass (no ggml graph).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 4.4: a Qwen2-architecture layer step composed entirely of
// sp_kernels_cpu calls (RMSNorm, matmul, RoPE, softmax, SiLU·Mul) +
// sp_quant dequant + sp_tensor / sp_arena buffer management.
//
// No ggml_graph, no ggml_backend_sched, no ggml_map_custom hooks, no
// thread fan-out around dispatch sites. The forward pass is a
// straight-line sequence the driver thread walks. CPU kernels
// internally vectorize via NEON; the heterogeneous backends
// (Hexagon, QNN HTP) plug in as direct call replacements at specific
// ops as they land.
//
// Activated by `SP_ENGINE_NATIVE=1` env var inside Engine::generate
// (wiring lands in a follow-up commit). Until then, this is a
// standalone library that callers can drive directly — useful for
// kernel-level benchmarks and for the test harness that validates
// per-layer output against ggml's equivalent forward.

#pragma once

#include "sp_tensor.h"

#include <cstdint>
#include <vector>

namespace sp::engine {

// ─────────────────────────────────────────────────────────────────
// Per-layer weight bundle. Each field points at the GGUF-resident
// weight bytes for that layer; dtype/shape come from the GGUF header.
// We support fp32, fp16, and Q5_K weights (other quants get added as
// we hit models that need them).
//
// Convention: Wq has shape [n_embd, n_head*head_dim] when stored in
// GGUF order — in memory, that's `n_embd` contiguous values per
// output feature, n_head*head_dim output features. Same for Wk/Wv/Wo
// (with appropriate output dims) and the FFN gate/up/down.
// ─────────────────────────────────────────────────────────────────
struct ForwardNativeLayer {
    // RMSNorm scale vectors — each [n_embd] fp32.
    const float* attn_norm = nullptr;
    const float* ffn_norm  = nullptr;

    // Attention projections (any of these may be quantized; their
    // dtype lives in the matching `wX_dtype` field).
    const void*  wq = nullptr;   sp_dtype wq_dtype = sp_dtype::UNDEFINED;
    const void*  wk = nullptr;   sp_dtype wk_dtype = sp_dtype::UNDEFINED;
    const void*  wv = nullptr;   sp_dtype wv_dtype = sp_dtype::UNDEFINED;
    const void*  wo = nullptr;   sp_dtype wo_dtype = sp_dtype::UNDEFINED;

    // Optional Q/K/V/O biases (Qwen2 has bq/bk/bv; Qwen3 omits them
    // by design; Llama variants vary). Always fp32 in GGUF when
    // present. Null = no bias add.
    const float* bq = nullptr;
    const float* bk = nullptr;
    const float* bv = nullptr;
    const float* bo = nullptr;

    // Optional Q/K per-head RMSNorm (Qwen2 doesn't use these; Qwen3
    // does for stability). [head_dim] each. Null when disabled.
    const float* attn_q_norm = nullptr;
    const float* attn_k_norm = nullptr;

    // FFN: gate/up combine (SwiGLU), down projects back to n_embd.
    const void*  ffn_gate = nullptr;  sp_dtype ffn_gate_dtype = sp_dtype::UNDEFINED;
    const void*  ffn_up   = nullptr;  sp_dtype ffn_up_dtype   = sp_dtype::UNDEFINED;
    const void*  ffn_down = nullptr;  sp_dtype ffn_down_dtype = sp_dtype::UNDEFINED;

    // ─── Phase 4.13: HTP weight matmul dispatch ──────────────────────
    // When the QNN HTP backend is active, ForwardNativeContext dequants
    // the seven dense weight matmuls (Q/K/V/O + gate/up/down) from
    // their GGUF dtype to fp16 ONCE at bind time, transposed to QNN's
    // [K, N] expected layout (vs the GGUF-native [N, K] order), and
    // optionally allocates them in rpcmem-ION via sp_qnn_alloc_persistent
    // so HTP reads them without per-call marshal copies. matmul_fp32_lhs
    // prefers mm_dispatch + the matching *_fp16 pointer over the CPU
    // sp_matmul_f32_q5k path.
    //
    // Layout: each *_fp16 buffer is [K, N] row-major fp16. K = input
    // feature dim (n_embd or n_ff), N = output feature dim. N*K*2 bytes.
    const uint16_t* wq_fp16       = nullptr;
    const uint16_t* wk_fp16       = nullptr;
    const uint16_t* wv_fp16       = nullptr;
    const uint16_t* wo_fp16       = nullptr;
    const uint16_t* ffn_gate_fp16 = nullptr;
    const uint16_t* ffn_up_fp16   = nullptr;
    const uint16_t* ffn_down_fp16 = nullptr;

    // ─── Optional backend dispatch hooks ─────────────────────────────
    // The CPU layer step uses sp_matmul_f32 internally for the per-head
    // KQ matmul. When a heterogeneous backend (QNN HTP, Hexagon, etc.)
    // is wired up, set this hook on each layer; the attention loop
    // calls it in place of sp_matmul_f32 for that one op.
    //
    // Contract: caller-supplied function computes
    //   scores[n_seq, n_kv_total] = Q[n_seq, head_dim] @ K^T[n_kv_total, head_dim]
    // (so K is stored row-major as [n_kv_total, head_dim] — same memory
    // forward_native already produces). Returns 0 on success, non-zero
    // to fall through to sp_matmul_f32. fp32 in / fp32 out — backend
    // is responsible for any internal precision conversion.
    typedef int (*kq_dispatch_fn_t)(void* userdata,
                                     const float* Q,        // [n_seq, head_dim]
                                     const float* K,        // [n_kv_total, head_dim]
                                     int n_seq, int head_dim, int n_kv_total,
                                     float* scores);        // [n_seq, n_kv_total]
    kq_dispatch_fn_t kq_dispatch          = nullptr;
    void*            kq_dispatch_userdata = nullptr;

    // Generic weight matmul dispatch. Computes out[M, N] = lhs[M, K] @ W[K, N]
    // where W is the fp16 weight (already transposed to [K, N] layout
    // at bind time, matching QNN's MatMul expectation). Returns 0 on
    // success, non-zero → caller falls through to CPU sp_matmul_f32.
    typedef int (*mm_dispatch_fn_t)(void* userdata,
                                     const float*    lhs,    // [M, K] fp32
                                     const uint16_t* W_fp16, // [K, N] fp16
                                     int M, int K, int N,
                                     float* out);            // [M, N] fp32
    mm_dispatch_fn_t mm_dispatch          = nullptr;
    void*            mm_dispatch_userdata = nullptr;
};

// ─────────────────────────────────────────────────────────────────
// Hyperparameters for the Qwen2-arch layer step.
// ─────────────────────────────────────────────────────────────────
struct ForwardNativeHparams {
    int   n_embd       = 0;     // residual stream width (e.g. 2048 for Qwen2.5-Coder-3B)
    int   n_head       = 0;     // total attention heads
    int   n_head_kv    = 0;     // KV heads (GQA: n_head divisible by n_head_kv)
    int   head_dim     = 0;     // dim per head
    int   n_ff         = 0;     // FFN intermediate width
    int   n_rot        = 0;     // RoPE-rotated dims (typically == head_dim)
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
    float rms_norm_eps    = 1e-5f;
};

// ─────────────────────────────────────────────────────────────────
// KV cache binding for the layer step. Just a thin pointer-pair
// referring to the engine's KvCache (which DOES the actual SP-banded
// compression — VHT2 + Möbius + band quantize on write, inverse on
// read). The forward pass owns nothing; the context owns the
// underlying KvCache.
//
// Phase 4.7: KV storage is now SP-banded packed bytes (~10× smaller
// than fp16) via the engine's KvCache wrapper around sp_shadow_cache_t
// (CPU ship-path compress/decompress). Hexagon backend variant
// (sp_hexagon_cache_t with sp_hexagon_cache_kq_matmul_fused — the
// HVX fused-decompress-matmul kernel from Phase 1.6) plugs in via
// the same KvCache::create call when SP_HEXAGON_FASTRPC=ON.
// ─────────────────────────────────────────────────────────────────
class KvCache;  // forward-declared; full def in kv_cache.h

struct ForwardNativeKv {
    KvCache* kv         = nullptr;   // borrowed; context owns it
    int      layer_idx  = 0;
    int      n_pos_past = 0;         // positions already in the cache

    // First-prefill calibration pass. When true, forward_native_attention
    // feeds each K vector to KvCache::calibrate_feed and runs attention
    // off the local just-computed K/V (no cache write/read). This is how
    // we train the Möbius reorder + per-band variance ranking on real
    // RoPE'd K vectors before the cache holds anything compressed.
    // Caller (ForwardNativeContext::prefill) flips this on for the first
    // pass, calls calibrate_end() between passes, then runs the real
    // prefill with calibrate_pass=false.
    bool     calibrate_pass = false;

    // Optional Hexagon cache pointer + matching ctx. When non-null AND
    // SP_HEXAGON_FASTRPC build, K vectors are mirrored into the
    // sp_hexagon_cache_t (compressed bytes resident on rpcmem-backed
    // pages) and the per-head KQ matmul routes through
    // sp_hexagon_cache_kq_matmul_fused — the HVX fused
    // decompress+matmul kernel from Phase 1.6 (1.79× prefill speedup
    // measured on S22U). Falls back to host CPU matmul on rc != 0.
    // Type kept as void* so this header doesn't pull in the C-only
    // hexagon header into every TU; cast at use site.
    void*    hex_cache  = nullptr;
};

// ─────────────────────────────────────────────────────────────────
// One Qwen2-arch layer step.
//
// Inputs:
//   layer  — weight pointers + dtypes
//   hp     — hyperparameters
//   x      — input residual stream, shape [n_embd, n_seq] fp32
//   pos    — position ids, shape [n_seq] int32 (for RoPE + KV write)
//   kv     — per-layer KV state (read past, append n_seq new)
//   arena  — scratch for intermediates; reset() before each call
//
// Output:
//   out    — output residual stream, shape [n_embd, n_seq] fp32.
//            May alias `x` for in-place residual.
//
// Returns 0 on success, negative on bad shape / arena overflow.
// ─────────────────────────────────────────────────────────────────
int forward_native_layer(const ForwardNativeLayer&  layer,
                         const ForwardNativeHparams& hp,
                         const float*  x,
                         const int32_t* pos,
                         int           n_seq,
                         ForwardNativeKv& kv,
                         sp_arena&     arena,
                         float*        out);

// Sub-passes exposed individually for kernel-level testing /
// progressive validation against ggml. Each reads from + writes into
// caller-managed buffers; does not allocate.
//
// attention_block: applies attn_norm + Q/K/V proj + RoPE + KV
// append + scaled-dot-product attention + wo. Output is the RESIDUAL
// CONTRIBUTION (caller adds to x).
int forward_native_attention(const ForwardNativeLayer&  layer,
                             const ForwardNativeHparams& hp,
                             const float*  x,
                             const int32_t* pos,
                             int           n_seq,
                             ForwardNativeKv& kv,
                             sp_arena&     arena,
                             float*        out_residual);

// ffn_block: applies ffn_norm + gate/up + SiLU·Mul + down. Output is
// the residual contribution (caller adds to x).
int forward_native_ffn(const ForwardNativeLayer&  layer,
                       const ForwardNativeHparams& hp,
                       const float*  x,
                       int           n_seq,
                       sp_arena&     arena,
                       float*        out_residual);

}  // namespace sp::engine
