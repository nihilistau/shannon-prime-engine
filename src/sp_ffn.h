// Shannon-Prime Engine — FFN (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// SwiGLU FFN in O_K coordinates: out = (silu(gate_proj(x)) * up_proj(x)) @ down_proj.

#pragma once

#include "sp_ok_tensor.h"

namespace sp::engine {

// SwiGLU FFN: out = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// O_K-output variant. The output `out` carries down_w's frobenius_scale
// (pi^k after the shim), so the caller cannot directly residual-add
// against a frobenius_scale=1 stream. Prefer sp_ffn_swiglu_to_fp32 in
// the main sp_forward_step path.
void sp_ffn_swiglu(const sp_ok_tensor& x,
                    const sp_ok_tensor& gate_w,
                    const sp_ok_tensor& up_w,
                    const sp_ok_tensor& down_w,
                    sp_ok_tensor&       out);

// SwiGLU FFN with fp32 output — the down projection runs as
// sp_matmul_ok_to_fp32 so down_w's Frobenius factor is divided out at the
// matmul boundary. This is the variant sp_forward_step uses when adding
// FFN output back into the residual stream.
//
// out_fp32: [n_embd, n_tokens] (or just [n_embd] for single-token).
// scratch_arena: caller-provided arena used to encode the post-silu
//                activations to O_K for the final matmul. Must have
//                room for ~d_ff sp_ok_t elements.
//
// Returns true on success.
bool sp_ffn_swiglu_to_fp32(const sp_ok_tensor& x,
                            const sp_ok_tensor& gate_w,
                            const sp_ok_tensor& up_w,
                            const sp_ok_tensor& down_w,
                            float*              out_fp32,
                            int                 n_tokens,
                            sp_ok_arena&        scratch_arena);

}  // namespace sp::engine
