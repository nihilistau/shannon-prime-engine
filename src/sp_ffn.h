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

// Gated-MLP activation kind for sp_ffn_swiglu_to_fp32:
//   SwiGLU      = silu(gate) * up         (Llama / Qwen / Mistral / Phi)
//   GeGLU_tanh  = gelu_tanh(gate) * up    (Gemma / Gemma2 / Gemma3)
enum class sp_ffn_act {
    SwiGLU      = 0,
    GeGLU_tanh  = 1,
};

// Gated-MLP FFN with fp32 output — the down projection runs as
// sp_matmul_ok_to_fp32 so down_w's Frobenius factor is divided out at the
// matmul boundary. This is the variant sp_forward_step uses when adding
// FFN output back into the residual stream.
//
// `act` selects the gate activation. Default SwiGLU preserves Phase 2.2d
// behavior.
//
// out_fp32: [n_embd, n_tokens] (or just [n_embd] for single-token).
// scratch_arena: caller-provided arena used to encode the post-act
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
                            sp_ok_arena&        scratch_arena,
                            sp_ffn_act          act = sp_ffn_act::SwiGLU);

// -----------------------------------------------------------------------
// Phase 12 Step D: fused-Q8 variant. Each of the three weight tensors
// (gate, up, down) is supplied as a SHAPE descriptor (data may be null)
// + a PACKED Q8 descriptor. The matmuls all run through sp_matmul_ok_q8*
// with the shift inlined per-lane.
// -----------------------------------------------------------------------
bool sp_ffn_swiglu_to_fp32_q8(const sp_ok_tensor&    x,
                              const sp_ok_tensor&    gate_w_shape,
                              const sp_ok_q8_tensor& gate_w_q8,
                              const sp_ok_tensor&    up_w_shape,
                              const sp_ok_q8_tensor& up_w_q8,
                              const sp_ok_tensor&    down_w_shape,
                              const sp_ok_q8_tensor& down_w_q8,
                              float*                 out_fp32,
                              int                    n_tokens,
                              sp_ok_arena&           scratch_arena,
                              sp_ffn_act             act = sp_ffn_act::SwiGLU);

// -----------------------------------------------------------------------
// Phase 14: fused-Q4 variant. Same as the Q8 helper above with the
// packed weight tensors swapped for 4-bit nybble-pair storage.
// -----------------------------------------------------------------------
bool sp_ffn_swiglu_to_fp32_q4(const sp_ok_tensor&    x,
                              const sp_ok_tensor&    gate_w_shape,
                              const sp_ok_q4_tensor& gate_w_q4,
                              const sp_ok_tensor&    up_w_shape,
                              const sp_ok_q4_tensor& up_w_q4,
                              const sp_ok_tensor&    down_w_shape,
                              const sp_ok_q4_tensor& down_w_q4,
                              float*                 out_fp32,
                              int                    n_tokens,
                              sp_ok_arena&           scratch_arena,
                              sp_ffn_act             act = sp_ffn_act::SwiGLU);

// -----------------------------------------------------------------------
// Phase 15: GGUF block-quant fused FFN variants.
// -----------------------------------------------------------------------
bool sp_ffn_swiglu_to_fp32_block_q8(const sp_ok_tensor&          x,
                                     const sp_ok_tensor&          gate_w_shape,
                                     const sp_ok_block_q8_tensor& gate_w_blk,
                                     const sp_ok_tensor&          up_w_shape,
                                     const sp_ok_block_q8_tensor& up_w_blk,
                                     const sp_ok_tensor&          down_w_shape,
                                     const sp_ok_block_q8_tensor& down_w_blk,
                                     float*                       out_fp32,
                                     int                          n_tokens,
                                     sp_ok_arena&                 scratch_arena,
                                     sp_ffn_act                   act = sp_ffn_act::SwiGLU);

bool sp_ffn_swiglu_to_fp32_block_q4(const sp_ok_tensor&          x,
                                     const sp_ok_tensor&          gate_w_shape,
                                     const sp_ok_block_q4_tensor& gate_w_blk,
                                     const sp_ok_tensor&          up_w_shape,
                                     const sp_ok_block_q4_tensor& up_w_blk,
                                     const sp_ok_tensor&          down_w_shape,
                                     const sp_ok_block_q4_tensor& down_w_blk,
                                     float*                       out_fp32,
                                     int                          n_tokens,
                                     sp_ok_arena&                 scratch_arena,
                                     sp_ffn_act                   act = sp_ffn_act::SwiGLU);

/* Phase 15b: mixed Q4_0 / Q4_1 FFN. Either of the three weight tensors
 * (gate, up, down) may be Q4_0 or Q4_1; pass the matching block pointer
 * for the populated one and nullptr for the other. Used for GGUF Q4_0
 * files where ffn_down is upgraded to Q4_1 by llama-quantize. */
bool sp_ffn_swiglu_to_fp32_block_q4_mixed(
    const sp_ok_tensor&              x,
    const sp_ok_tensor&              gate_shape,
    const sp_ok_block_q4_tensor*     gate_q4_0,    /* one of these two non-null */
    const sp_ok_block_q4_1_tensor*   gate_q4_1,
    const sp_ok_tensor&              up_shape,
    const sp_ok_block_q4_tensor*     up_q4_0,
    const sp_ok_block_q4_1_tensor*   up_q4_1,
    const sp_ok_tensor&              down_shape,
    const sp_ok_block_q4_tensor*     down_q4_0,
    const sp_ok_block_q4_1_tensor*   down_q4_1,
    float*                           out_fp32,
    int                              n_tokens,
    sp_ok_arena&                     scratch_arena,
    sp_ffn_act                       act = sp_ffn_act::SwiGLU);

}  // namespace sp::engine
