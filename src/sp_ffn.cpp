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

}  // namespace sp::engine
