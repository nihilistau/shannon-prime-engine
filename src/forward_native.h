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

    // Optional Q/K per-head RMSNorm (Qwen2 doesn't use these; Qwen3
    // does for stability). [head_dim] each. Null when disabled.
    const float* attn_q_norm = nullptr;
    const float* attn_k_norm = nullptr;

    // FFN: gate/up combine (SwiGLU), down projects back to n_embd.
    const void*  ffn_gate = nullptr;  sp_dtype ffn_gate_dtype = sp_dtype::UNDEFINED;
    const void*  ffn_up   = nullptr;  sp_dtype ffn_up_dtype   = sp_dtype::UNDEFINED;
    const void*  ffn_down = nullptr;  sp_dtype ffn_down_dtype = sp_dtype::UNDEFINED;
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
// KV cache for one layer. Holds past K/V values that the layer step
// reads from + appends to. Shapes:
//   k_cache: [head_dim, n_head_kv, max_seq] fp16
//   v_cache: [head_dim, n_head_kv, max_seq] fp16
// Storage is contiguous; positions index along the outer dim.
//
// Phase 4.4 first cut: plain fp16 (no SP-banded compression yet).
// SP-banded layers on top later via the existing shannon-prime
// hexagon backend — same buffer pointers, different read/write ops.
// ─────────────────────────────────────────────────────────────────
struct ForwardNativeKv {
    uint16_t* k_cache = nullptr;    // owned by caller
    uint16_t* v_cache = nullptr;
    int       max_seq = 0;
    int       n_pos   = 0;          // number of positions written so far
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
