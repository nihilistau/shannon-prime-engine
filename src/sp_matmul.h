// Shannon-Prime Engine — native O_K matrix multiplication.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.0 — the foundational primitive for the theory-first forward pass.
// Both sp_attention and sp_ffn reduce to "sp_matmul + nonlinearity bridge".
//
// Semantics:
//   Y[i,j] = sum_k W[i,k] * X[k,j]
// where all operands are sp_ok_tensors (int64 (a, b) coordinates) and the
// multiply-accumulate runs in the O_K ring exactly. The output Y stays in
// O_K coordinates with its `scale_recip` and `frobenius_scale` derived
// from the operands (multiplicative composition).
//
// Why a separate file from sp_kernels_cpu:
//   sp_kernels_cpu operates on fp32 / fp16 / Q*K tensors. sp_matmul
//   operates on the integer-coordinate sp_ok_tensor — a different element
//   type. Code paths and rounding semantics differ; better to keep them
//   separate for clarity.
//
// Numerical concerns:
//   - Inputs are sp_ok_t = { int64 a, int64 b }. The product of two such
//     elements multiplies a*a or 41*b*b, so intermediate values can grow
//     by ~6 bits per multiply. For typical Phi-3 / Gemma3 dimensions with
//     scale_recip ~ 2^14, this is safe up to ~32-bit dot-product partials.
//     For larger k or higher scale_recip, we accumulate in __int128.
//   - The fp32-bridge variant decodes to fp32 *at the output* so that
//     softmax / silu can run in fp32 without losing the algebraic
//     structure on the input side.

#pragma once

#include "sp_ok_tensor.h"

#include <cstddef>
#include <cstdint>

namespace sp::engine {

// -----------------------------------------------------------------------
// O_K @ O_K → O_K matmul (the workhorse).
// -----------------------------------------------------------------------
//
// Layout convention (Step E — token-as-row, matches the rest of the engine):
//   W: weight matrix [out_rows M × in_cols K], row-major,  W.data[i*K + k]
//   X: input  matrix [n_cols  N × in_cols K], row-major,  X.data[j*K + k]
//   Y: output matrix [n_cols  N × out_rows M], row-major, Y.data[j*M + i]
//
//   Computes Y[j,i] = sum_k W[i,k] * X[j,k]
//
// This matches the natural token-as-row layout used by embed_lookup,
// rmsnorm, RoPE, KV-cache append, and FFN — every kernel writes/reads X
// as "token j's features start at j*K". The inner k loop is contiguous
// in both W (i*K + k) and X (j*K + k), which is friendly to the hardware
// prefetcher and SIMD.
//
// At N=1 (single-token decode), Y.data[j*M + i] collapses to Y.data[i]
// and X.data[j*K + k] collapses to X.data[k] — bit-identical to the
// pre-Step-E convention. All N=1 PPL numbers remain valid.
//
// All three tensors hold sp_ok_t elements. Y's data is overwritten.
// Y.scale_recip is set to W.scale_recip * X.scale_recip;
// Y.frobenius_scale is set to W.frobenius_scale * X.frobenius_scale.
//
// Returns true on success, false on shape mismatch or null data.
bool sp_matmul_ok(const sp_ok_tensor& W,
                   const sp_ok_tensor& X,
                   sp_ok_tensor&       Y);

// -----------------------------------------------------------------------
// O_K @ O_K → fp32 matmul (bridge to softmax/silu).
// -----------------------------------------------------------------------
//
// Same shapes as above. Y_fp32 is [out_rows, n_cols] caller-allocated
// fp32 buffer. Internally accumulates the O_K dot product into the
// a-coordinate of an sp_ok_t scratch, then decodes:
//
//   Y_fp32[i,j] = (sp_ok_dot_product_a) /
//                 (W.scale_recip * X.scale_recip *
//                  W.frobenius_scale * X.frobenius_scale)
//
// This is the bridge for layers that feed into a non-O_K operation
// (softmax, silu). The Frobenius scale is correctly divided out so the
// downstream nonlinearity sees the original fp32 product (to ULP).
bool sp_matmul_ok_to_fp32(const sp_ok_tensor& W,
                           const sp_ok_tensor& X,
                           float*              Y_fp32,
                           int                 out_rows,
                           int                 n_cols);

// -----------------------------------------------------------------------
// Convenience: matmul where weights are O_K but input is fp32.
// Used at the boundary where fp32 activations from softmax/silu feed
// back into an O_K linear projection.
// -----------------------------------------------------------------------
//
// W: O_K weight matrix [out_rows, in_cols]
// X_fp32: fp32 input matrix [in_cols, n_cols], row-major
// Y: O_K output [out_rows, n_cols]
//
// Y.scale_recip carries forward from W; Y.frobenius_scale carries from W.
// Internally re-encodes X_fp32 to a per-call scale matching W's scale.
bool sp_matmul_fp32_input_to_ok(const sp_ok_tensor& W,
                                  const float*        X_fp32,
                                  int                 in_cols,
                                  int                 n_cols,
                                  sp_ok_tensor&       Y);

// -----------------------------------------------------------------------
// Phase 12 Step D: fused packed-Q8 weight matmul (the production endgame).
//
// Same semantics as sp_matmul_ok, but the W operand is provided as a
// 2 B/element packed sp_ok_q8_tensor plus a per-tensor shift. The
// kernel sign-extends each int8 pair to int64 and applies the shift
// inline in the inner loop, then performs the sp_ok ring multiply.
//
// API contract:
//   W_shape  -- sp_ok_tensor with shape[], scale_recip, frobenius_scale
//               populated (its .data may be nullptr; only metadata is used).
//   W_q8     -- the packed bytes (numel = W_shape.numel()) + q8_shift.
//   X        -- input as a full sp_ok_tensor (same as sp_matmul_ok).
//   Y        -- output sp_ok_tensor (same as sp_matmul_ok).
//
// The fused path eliminates the 430 MB decoded-buffer write that Step C's
// prefetcher introduced; weights are streamed through the matmul kernel
// at 2 B/element with the shift applied per-lane, dropping DRAM pressure
// by 8x and keeping the working set resident in L2/L3 across an entire
// layer's matmuls.
// -----------------------------------------------------------------------
bool sp_matmul_ok_q8(const sp_ok_tensor&    W_shape,
                     const sp_ok_q8_tensor& W_q8,
                     const sp_ok_tensor&    X,
                     sp_ok_tensor&          Y);

// fp32-output variant for the Wo projection / LM head bridge.
// Same divisor calculation as sp_matmul_ok_to_fp32 (W_shape provides
// scale_recip + frobenius_scale).
bool sp_matmul_ok_q8_to_fp32(const sp_ok_tensor&    W_shape,
                             const sp_ok_q8_tensor& W_q8,
                             const sp_ok_tensor&    X,
                             float*                 Y_fp32,
                             int                    out_rows,
                             int                    n_cols);

// -----------------------------------------------------------------------
// Phase 14: fused packed-Q4 weight matmul.
//
// Same semantics as sp_matmul_ok_q8 with the W codebook halved to 1
// byte/element. Sign-extends each 4-bit nybble pair to int64 (via the
// arithmetic-shift idiom in sp_ok_q4_decode_one) and applies q4_shift
// inline in the inner loop, then performs the sp_ok ring multiply.
//
// At Phi-3 / Qwen weight scales the q4_shift is ~4 bits larger than
// q8_shift on the same tensor, so absolute quantization noise per
// coordinate is ~16x larger. Whether Theorem 2's projective cancellation
// absorbs that noise is a forward-pass question; the storage + dispatch
// contract is well-defined either way.
// -----------------------------------------------------------------------
bool sp_matmul_ok_q4(const sp_ok_tensor&    W_shape,
                     const sp_ok_q4_tensor& W_q4,
                     const sp_ok_tensor&    X,
                     sp_ok_tensor&          Y);

bool sp_matmul_ok_q4_to_fp32(const sp_ok_tensor&    W_shape,
                             const sp_ok_q4_tensor& W_q4,
                             const sp_ok_tensor&    X,
                             float*                 Y_fp32,
                             int                    out_rows,
                             int                    n_cols);

}  // namespace sp::engine
