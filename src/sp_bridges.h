// Shannon-Prime Engine — Phase 2.1 fp32 bridges for the native O_K pipeline.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Three bridge utilities that connect the O_K-coordinate matmul primitive
// (sp_matmul) to the transcendental non-linearities that don't have exact
// O_K representations:
//
//   sp_rmsnorm_native    O_K state -> O_K state (fp32 RMS island in middle)
//                         RMSNorm is the per-layer "scale-reset operator":
//                         output frobenius_scale is reset to 1.
//
//   sp_softmax_bridge    fp32 logits -> fp32 weights
//                         Numerically-stable max-subtraction softmax.
//                         Stays fp32 throughout — downstream feeds back
//                         into sp_matmul_fp32_input_to_ok.
//
//   sp_silu_bridge       fp32 gate, fp32 up -> fp32 product
//                         silu(g) = g / (1 + exp(-g)). Output is
//                         silu(gate) * up, the SwiGLU activation.
//
// Pipeline shape per layer:
//   x (O_K)
//     -> sp_rmsnorm_native -> x_norm (O_K, scale=1 reset)
//     -> sp_matmul_ok       -> Q, K, V (O_K)
//     -> sp_matmul_ok_to_fp32(Q, K^T) -> scores (fp32)
//     -> sp_softmax_bridge -> weights (fp32)
//     -> sp_matmul_fp32_input_to_ok(weights, V) -> attn (O_K)
//     -> sp_matmul_ok      -> attn_out (O_K)
//     -> residual add (O_K += O_K)
//     -> sp_rmsnorm_native -> x_norm2 (O_K, scale=1 reset)
//     -> sp_matmul_ok_to_fp32 (gate proj, up proj) -> g, u (fp32)
//     -> sp_silu_bridge    -> activated (fp32)
//     -> sp_matmul_fp32_input_to_ok(activated, W_down) -> ffn_out (O_K)
//     -> residual add (O_K += O_K)

#pragma once

#include "sp_ok_tensor.h"

#include <cstddef>
#include <cstdint>

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_rmsnorm_native — RMSNorm on O_K state, fp32 RMS island.
// -----------------------------------------------------------------------
//
// x:        input  O_K state, shape [n_embd] (or [n_embd, n_tokens] for
//           batched; pass n_tokens via the second dim).
// scale_fp32: bypass-mode RMSNorm scale (per Phase 1.7 policy, RMSNorm
//             scales stay native fp16/fp32). Caller decodes fp16 -> fp32
//             once at load.
// out:      output O_K state, same shape as x. out.scale_recip is set to
//           a fresh value (caller decides; we default to x.scale_recip).
//           out.frobenius_scale is RESET to 1 (the scale-reset valve).
// eps:      RMSNorm epsilon for numerical stability.
// n_embd, n_tokens: shape parameters (we normalize over n_embd per token).
//
// Semantics:
//   x_decoded[i] = x[i].a / (x.scale_recip * x.frobenius_scale)
//   rms          = sqrt(mean(x_decoded^2) + eps)
//   out_fp[i]    = (x_decoded[i] / rms) * scale_fp32[i]
//   out[i].a     = round(out_fp[i] * out.scale_recip)
//   out[i].b     = 0
//
// Returns true on success, false on bad shapes / null data.
bool sp_rmsnorm_native(const sp_ok_tensor& x,
                        const float*        scale_fp32,
                        sp_ok_tensor&       out,
                        float               eps,
                        int                 n_embd,
                        int                 n_tokens);

// -----------------------------------------------------------------------
// sp_softmax_bridge — standard numerically-stable softmax in fp32.
// -----------------------------------------------------------------------
//
// in:       [n] fp32 logits.
// out:      [n] fp32 normalized weights (sum to 1).
// In-place is fine (out can alias in).
void sp_softmax_bridge(const float* in, int n, float* out);

// Batched variant: applies softmax across the innermost dim of length n,
// across n_rows. Each row is independently normalized.
void sp_softmax_bridge_rows(const float* in, int n_cols, int n_rows, float* out);

// Causal-mask variant: zeros out positions >= valid_len before softmax.
// Used inside attention with KV-cache.
void sp_softmax_bridge_causal(const float* in, int n, int valid_len, float* out);

// -----------------------------------------------------------------------
// sp_silu_bridge — SwiGLU activation: silu(gate) * up, elementwise.
// -----------------------------------------------------------------------
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// gate, up: [n] fp32 tensors (typically the gate-proj and up-proj outputs).
// out:      [n] fp32 product. Can alias gate or up.
void sp_silu_bridge(const float* gate, const float* up, int n, float* out);

// Plain silu without the gating multiply (for non-SwiGLU activation paths).
void sp_silu_inplace(float* x, int n);

}  // namespace sp::engine
