// Shannon-Prime Engine — native CPU kernels for forward_native.cpp.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Replaces ggml's compute primitives with our own. Kernels take raw
// pointers + dimensions — no tensor abstraction, no graph nodes, no
// scheduler. Layer-step driver in forward_native.cpp calls them
// directly with sp_arena-backed buffers.
//
// All kernels are SINGLE-THREADED on the driver side. Internal
// parallelism via NEON SIMD on arm64; scalar-only fallback for x86 /
// older targets. Multi-thread parallelism (matmul over rows, etc.)
// will land in a separate sp_threadpool when we need it; for now the
// philosophy is "the heterogeneous backends carry the parallel
// dispatch, CPU kernels are the fallback floor".
//
// Each kernel has a clear contract — buffers don't alias unless the
// signature explicitly allows in-place (denoted via comment), and
// row-major layout is assumed: shape = {inner, outer}, stride[0]=1.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sp::engine {

// ─── RMS Norm + scale ──────────────────────────────────────────────
// out[i] = (x[i] / rms) * scale[i],  rms = sqrt(sum(x^2)/n + eps)
//
// Per-row operation. `x` and `out` are [n] fp32. `scale` is [n] fp32
// (the trained per-feature gain). out may alias x for in-place.
void sp_rms_norm_f32(const float* x, const float* scale,
                     int n, float eps, float* out);

// Same, applied row-wise to a [n_rows, n_cols] matrix. Each row is
// normalized independently against its own rms.
void sp_rms_norm_f32_rows(const float* x, const float* scale,
                          int n_cols, int n_rows, float eps,
                          float* out);

// ─── SiLU (Swish-1) ────────────────────────────────────────────────
// out[i] = x[i] / (1 + exp(-x[i]))
// Element-wise, n elements. out may alias x.
void sp_silu_f32(const float* x, int n, float* out);

// SiLU-and-multiply (SwiGLU): out[i] = silu(gate[i]) * up[i]
// One pass over both inputs to avoid an extra cache trip.
void sp_silu_mul_f32(const float* gate, const float* up,
                     int n, float* out);

// ─── Softmax (stable) ──────────────────────────────────────────────
// Row-wise softmax over the inner dim. Uses log-sum-exp for numerical
// stability. Optional `scale` multiplies x BEFORE the softmax (used
// to apply 1/sqrt(d_k) in attention without a separate pass). Optional
// `mask` adds a per-element bias before softmax (used for causal
// attention; mask values are -INF for masked positions, 0 otherwise).
//
// shape: x is [n_cols, n_rows] row-major. out may alias x.
void sp_softmax_f32_rows(const float* x, const float* mask,
                         int n_cols, int n_rows, float scale,
                         float* out);

// ─── RoPE (multi-section / mRoPE for Qwen3) ────────────────────────
// Apply rotary position embedding to a tensor of shape
// [head_dim, n_heads, n_pos]. Each head's first n_rot dims get
// rotated; trailing (head_dim - n_rot) dims pass through unchanged.
//
// Positions are interleaved-pairs: (x[2k], x[2k+1]) → (cos*a - sin*b,
// sin*a + cos*b) with a=x[2k], b=x[2k+1].
//
// freq[i] = 1 / pow(freq_base, 2i/n_rot) * freq_scale, then for
// position p: angle = p * freq[i].
//
// `pos` is [n_pos] int32. For mRoPE (Qwen3), the multi-section split
// is described by `sections[0..3]` which sum to n_rot/2 — currently
// we collapse to standard RoPE (all sections share the same pos);
// dedicated mrope path lands when we hit a model that uses it.
void sp_rope_f32(float* x, int head_dim, int n_heads, int n_pos,
                 const int32_t* pos,
                 int n_rot, float freq_base, float freq_scale);

// ─── Matmul (fp32 LHS × fp32 RHS) ──────────────────────────────────
// out[m, n] = lhs[m, k] @ rhs[n, k]^T   (rhs is row-major and we
// take its rows as columns of the matmul — matches GGUF weight layout
// where weights are stored as [n_in, n_out] but multiplied as
// y = W @ x with W's rows being the output features).
//
// Naive O(m*k*n) loop, NEON-friendly inner kernel. Adequate for the
// CPU fallback path; the heterogeneous backends (Hexagon/QNN HTP)
// take over for the perf-critical matmuls.
void sp_matmul_f32(const float* lhs, const float* rhs,
                   int m, int k, int n, float* out);

// ─── Matmul with Q5_K-packed RHS ───────────────────────────────────
// Same shape as sp_matmul_f32, but `rhs_q5k` is the raw packed bytes
// of (n*k) elements quantized as Q5_K. Dequant is fused into the
// inner loop (we dequant 256 elements at a time into a scratch buffer
// kept in L1, then dot-product against lhs rows). Avoids materializing
// the full dequantized weight matrix.
//
// k MUST be a multiple of 256 (the Q5_K block size). n_blocks_per_row
// = k / 256. Internal scratch buffer is allocated per call (one
// row's worth, 1 KB at k=256).
void sp_matmul_f32_q5k(const float* lhs, const void* rhs_q5k,
                       int m, int k, int n, float* out);

}  // namespace sp::engine
