// Shannon-Prime Engine — FFN (Phase 2.2a — SwiGLU).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Implements sp_ffn_swiglu: y = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// Pipeline:
//   1. gate_fp32 = gate_w · x           [O_K · O_K → fp32 via matmul_ok_to_fp32]
//   2. up_fp32   = up_w · x             [O_K · O_K → fp32]
//   3. act_fp32  = silu(gate_fp32) * up_fp32   [sp_silu_bridge]
//   4. out       = down_w · act_fp32    [fp32 × O_K → O_K via matmul_fp32_input_to_ok]
//
// SHAPES (single-token decode):
//   x:        shape={n_embd, 1}                 row-major: x[i]
//   gate_w:   shape={n_embd, d_ff}              W·x → [d_ff, 1]
//   up_w:     shape={n_embd, d_ff}
//   down_w:   shape={d_ff,   n_embd}
//   out:      shape={n_embd, 1}

#include "sp_ffn.h"
#include "sp_matmul.h"
#include "sp_bridges.h"
#include "sp_ok_encode.h"

#include <cstdio>
#include <vector>

namespace sp::engine {

void sp_ffn_swiglu(const sp_ok_tensor& x,
                    const sp_ok_tensor& gate_w,
                    const sp_ok_tensor& up_w,
                    const sp_ok_tensor& down_w,
                    sp_ok_tensor&       out) {
    if (x.data == nullptr || gate_w.data == nullptr || up_w.data == nullptr ||
        down_w.data == nullptr || out.data == nullptr) return;

    // Derive shapes from the operands. sp_matmul convention: shape[0] is
    // the innermost (K dim for weight matrices, N dim for activations).
    //   gate_w / up_w shape = {n_embd, d_ff}  -> K=n_embd, M=d_ff
    //   down_w shape       = {d_ff, n_embd}  -> K=d_ff,  M=n_embd
    //   x  shape           = {1, n_embd}     -> N=1, K=n_embd (single token)
    //   out shape          = {1, n_embd}     -> N=1, M=n_embd
    const int n_embd = (int)gate_w.shape[0];
    const int d_ff   = (int)gate_w.shape[1];
    if (n_embd <= 0 || d_ff <= 0) return;
    if (up_w.shape[0] != n_embd || up_w.shape[1] != d_ff) return;
    if (down_w.shape[0] != d_ff || down_w.shape[1] != n_embd) return;
    if (x.shape[0] != 1 || x.shape[1] != n_embd) return;
    if (out.shape[0] != 1 || out.shape[1] != n_embd) return;

    // 1) Gate projection -> fp32 directly (avoid an intermediate O_K tensor).
    std::vector<float> gate_fp32(d_ff);
    if (!sp_matmul_ok_to_fp32(gate_w, x, gate_fp32.data(), d_ff, 1)) {
        std::fprintf(stderr, "[sp_ffn] gate matmul failed\n");
        return;
    }

    // 2) Up projection -> fp32.
    std::vector<float> up_fp32(d_ff);
    if (!sp_matmul_ok_to_fp32(up_w, x, up_fp32.data(), d_ff, 1)) {
        std::fprintf(stderr, "[sp_ffn] up matmul failed\n");
        return;
    }

    // 3) SwiGLU: silu(gate) * up, elementwise.
    std::vector<float> act_fp32(d_ff);
    sp_silu_bridge(gate_fp32.data(), up_fp32.data(), d_ff, act_fp32.data());

    // 4) Down projection: fp32 activations × O_K weights → O_K out.
    if (!sp_matmul_fp32_input_to_ok(down_w, act_fp32.data(), d_ff, 1, out)) {
        std::fprintf(stderr, "[sp_ffn] down matmul failed\n");
        return;
    }
}

// =========================================================================
// sp_ffn_swiglu_to_fp32 — Phase 2.2d residual-bound path.
//
// Same as sp_ffn_swiglu but the final down-projection writes to fp32 via
// sp_matmul_ok_to_fp32 so down_w's Frobenius factor is divided out at
// the matmul boundary. The output is in original (un-shimmed) fp32 units
// — caller can residual-add directly against the fp32 residual stream.
// =========================================================================

bool sp_ffn_swiglu_to_fp32(const sp_ok_tensor& x,
                            const sp_ok_tensor& gate_w,
                            const sp_ok_tensor& up_w,
                            const sp_ok_tensor& down_w,
                            float*              out_fp32,
                            int                 n_tokens,
                            sp_ok_arena&        scratch_arena,
                            sp_ffn_act          act) {
    if (x.data == nullptr || gate_w.data == nullptr || up_w.data == nullptr ||
        down_w.data == nullptr || out_fp32 == nullptr) return false;
    const int n_embd = (int)gate_w.shape[0];
    const int d_ff   = (int)gate_w.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;
    if (up_w.shape[0] != n_embd || up_w.shape[1] != d_ff) return false;
    if (down_w.shape[0] != d_ff || down_w.shape[1] != n_embd) return false;
    if (x.shape[0] != n_tokens || x.shape[1] != n_embd) return false;

    // For each token we run the gated activation independently; the
    // down-projection then absorbs all tokens at once via matmul.
    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_matmul_ok_to_fp32(gate_w, x, gate_all.data(), d_ff, n_tokens)) {
        return false;
    }
    if (!sp_matmul_ok_to_fp32(up_w, x, up_all.data(), d_ff, n_tokens)) {
        return false;
    }
    // Gated MLP per element across the whole (d_ff * n_tokens) block.
    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(), d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(), d_ff * n_tokens, act_all.data());
        break;
    }

    // Encode the post-silu activation as O_K so we can run
    // sp_matmul_ok_to_fp32(down_w, act_ok, ...) → fp32 out with Frobenius
    // automatically absorbed.
    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_w.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    // act_ok.frobenius_scale = 1.
    // sp_matmul_ok_to_fp32 divides by down_w.scale_recip * act_ok.scale_recip
    //                              * down_w.frobenius_scale * 1
    // which gives us fp32 in the original units.
    if (!sp_matmul_ok_to_fp32(down_w, act_ok, out_fp32, n_embd, n_tokens)) {
        return false;
    }
    return true;
}

// =========================================================================
// Phase 12 Step D: fused-Q8 FFN
// =========================================================================

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
                              sp_ffn_act             act) {
    if (x.data == nullptr || out_fp32 == nullptr) return false;
    if (gate_w_q8.data == nullptr || up_w_q8.data == nullptr ||
        down_w_q8.data == nullptr) return false;
    const int n_embd = (int)gate_w_shape.shape[0];
    const int d_ff   = (int)gate_w_shape.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;
    if (up_w_shape.shape[0] != n_embd || up_w_shape.shape[1] != d_ff) return false;
    if (down_w_shape.shape[0] != d_ff || down_w_shape.shape[1] != n_embd) return false;
    if (x.shape[0] != n_tokens || x.shape[1] != n_embd) return false;

    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_matmul_ok_q8_to_fp32(gate_w_shape, gate_w_q8, x,
                                  gate_all.data(), d_ff, n_tokens)) return false;
    if (!sp_matmul_ok_q8_to_fp32(up_w_shape, up_w_q8, x,
                                  up_all.data(),   d_ff, n_tokens)) return false;

    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(),
                       d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(),
                            d_ff * n_tokens, act_all.data());
        break;
    }

    /* Encode post-activation into O_K at down_w's scale_recip, then run
     * the fused-Q8 down matmul. */
    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_w_shape.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    if (!sp_matmul_ok_q8_to_fp32(down_w_shape, down_w_q8, act_ok,
                                  out_fp32, n_embd, n_tokens)) {
        return false;
    }
    return true;
}

// =========================================================================
// Phase 14: fused-Q4 FFN
// =========================================================================

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
                              sp_ffn_act             act) {
    if (x.data == nullptr || out_fp32 == nullptr) return false;
    if (gate_w_q4.data == nullptr || up_w_q4.data == nullptr ||
        down_w_q4.data == nullptr) return false;
    const int n_embd = (int)gate_w_shape.shape[0];
    const int d_ff   = (int)gate_w_shape.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;
    if (up_w_shape.shape[0] != n_embd || up_w_shape.shape[1] != d_ff) return false;
    if (down_w_shape.shape[0] != d_ff || down_w_shape.shape[1] != n_embd) return false;
    if (x.shape[0] != n_tokens || x.shape[1] != n_embd) return false;

    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_matmul_ok_q4_to_fp32(gate_w_shape, gate_w_q4, x,
                                  gate_all.data(), d_ff, n_tokens)) return false;
    if (!sp_matmul_ok_q4_to_fp32(up_w_shape, up_w_q4, x,
                                  up_all.data(),   d_ff, n_tokens)) return false;

    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(),
                       d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(),
                            d_ff * n_tokens, act_all.data());
        break;
    }

    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_w_shape.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    if (!sp_matmul_ok_q4_to_fp32(down_w_shape, down_w_q4, act_ok,
                                  out_fp32, n_embd, n_tokens)) {
        return false;
    }
    return true;
}

// =========================================================================
// Phase 15: block-quant FFN variants
// =========================================================================

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
                                     sp_ffn_act                   act) {
    if (x.data == nullptr || out_fp32 == nullptr) return false;
    if (gate_w_blk.blocks == nullptr || up_w_blk.blocks == nullptr ||
        down_w_blk.blocks == nullptr) return false;
    const int n_embd = (int)gate_w_shape.shape[0];
    const int d_ff   = (int)gate_w_shape.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;

    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_matmul_ok_block_q8_to_fp32(gate_w_shape, gate_w_blk, x,
                                        gate_all.data(), d_ff, n_tokens)) return false;
    if (!sp_matmul_ok_block_q8_to_fp32(up_w_shape, up_w_blk, x,
                                        up_all.data(),   d_ff, n_tokens)) return false;

    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(),
                       d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(),
                            d_ff * n_tokens, act_all.data());
        break;
    }

    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_w_shape.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    return sp_matmul_ok_block_q8_to_fp32(down_w_shape, down_w_blk, act_ok,
                                          out_fp32, n_embd, n_tokens);
}

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
                                     sp_ffn_act                   act) {
    if (x.data == nullptr || out_fp32 == nullptr) return false;
    if (gate_w_blk.blocks == nullptr || up_w_blk.blocks == nullptr ||
        down_w_blk.blocks == nullptr) return false;
    const int n_embd = (int)gate_w_shape.shape[0];
    const int d_ff   = (int)gate_w_shape.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;

    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_matmul_ok_block_q4_to_fp32(gate_w_shape, gate_w_blk, x,
                                        gate_all.data(), d_ff, n_tokens)) return false;
    if (!sp_matmul_ok_block_q4_to_fp32(up_w_shape, up_w_blk, x,
                                        up_all.data(),   d_ff, n_tokens)) return false;

    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(),
                       d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(),
                            d_ff * n_tokens, act_all.data());
        break;
    }

    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_w_shape.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    return sp_matmul_ok_block_q4_to_fp32(down_w_shape, down_w_blk, act_ok,
                                          out_fp32, n_embd, n_tokens);
}

// Helper: route a single matmul to whichever block_q4 variant has a
// non-null tensor for that slot.
static inline bool sp_ffn_blk_matmul_mixed_to_fp32(
    const sp_ok_tensor&            W_shape,
    const sp_ok_block_q4_tensor*   W_q4_0,
    const sp_ok_block_q4_1_tensor* W_q4_1,
    const sp_ok_tensor&            X,
    float*                         Y_fp32,
    int                            out_rows,
    int                            n_cols)
{
    if (W_q4_0 && W_q4_0->blocks) {
        return sp_matmul_ok_block_q4_to_fp32(
            W_shape, *W_q4_0, X, Y_fp32, out_rows, n_cols);
    }
    if (W_q4_1 && W_q4_1->blocks) {
        return sp_matmul_ok_block_q4_1_to_fp32(
            W_shape, *W_q4_1, X, Y_fp32, out_rows, n_cols);
    }
    /* Phase 15d fallback: when no block storage is populated for this
     * tensor (e.g. Q5_0 or Q6_K that dequanted into raw sp_ok_t), the
     * caller passes W_shape with .data populated. Use the raw matmul. */
    if (W_shape.data) {
        return sp_matmul_ok_to_fp32(W_shape, X, Y_fp32, out_rows, n_cols);
    }
    return false;
}

bool sp_ffn_swiglu_to_fp32_block_q4_mixed(
    const sp_ok_tensor&              x,
    const sp_ok_tensor&              gate_shape,
    const sp_ok_block_q4_tensor*     gate_q4_0,
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
    sp_ffn_act                       act)
{
    if (x.data == nullptr || out_fp32 == nullptr) return false;
    const int n_embd = (int)gate_shape.shape[0];
    const int d_ff   = (int)gate_shape.shape[1];
    if (n_embd <= 0 || d_ff <= 0 || n_tokens <= 0) return false;

    std::vector<float> gate_all(d_ff * n_tokens);
    std::vector<float> up_all(d_ff * n_tokens);
    std::vector<float> act_all(d_ff * n_tokens);

    if (!sp_ffn_blk_matmul_mixed_to_fp32(
            gate_shape, gate_q4_0, gate_q4_1, x,
            gate_all.data(), d_ff, n_tokens)) return false;
    if (!sp_ffn_blk_matmul_mixed_to_fp32(
            up_shape, up_q4_0, up_q4_1, x,
            up_all.data(), d_ff, n_tokens)) return false;

    switch (act) {
    case sp_ffn_act::SwiGLU:
        sp_silu_bridge(gate_all.data(), up_all.data(),
                       d_ff * n_tokens, act_all.data());
        break;
    case sp_ffn_act::GeGLU_tanh:
        sp_gelu_tanh_bridge(gate_all.data(), up_all.data(),
                            d_ff * n_tokens, act_all.data());
        break;
    }

    sp_ok_tensor act_ok;
    int64_t act_shape[4] = { n_tokens, d_ff, 1, 1 };
    if (!sp_ok_encode_from_fp32(act_ok, act_all.data(), 2, act_shape,
                                  /*scale*/ down_shape.scale_recip,
                                  scratch_arena)) {
        return false;
    }
    return sp_ffn_blk_matmul_mixed_to_fp32(
        down_shape, down_q4_0, down_q4_1, act_ok,
        out_fp32, n_embd, n_tokens);
}

}  // namespace sp::engine
