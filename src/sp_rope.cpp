// Shannon-Prime Engine — RoPE on O_K tensors (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_rope.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace sp::engine {

bool sp_rope_apply_ok(sp_ok_tensor&      qk,
                       int                n_heads,
                       int                head_dim,
                       int                n_tokens,
                       const int32_t*     positions,
                       float              freq_base,
                       float              freq_scale,
                       sp_rope_mode       mode) {
    if (qk.data == nullptr || positions == nullptr) return false;
    if (n_heads <= 0 || head_dim <= 0 || n_tokens <= 0) return false;
    if ((head_dim & 1) != 0) return false;  // pairs of 2
    if (qk.n_dims < 2) return false;
    const int64_t T_inner = qk.shape[0];
    const int64_t F_outer = qk.shape[1];
    if ((int64_t)n_tokens != T_inner) return false;
    if ((int64_t)(n_heads * head_dim) != F_outer) return false;

    // Combined divisor for decode: scale_recip * frobenius_scale.
    const double divisor = (double)qk.scale_recip * (double)qk.frobenius_scale;
    if (divisor == 0.0) return false;
    const int64_t S = qk.scale_recip;  // re-encode at scale_recip alone.

    const int n_pairs = head_dim / 2;

    // Pre-compute per-pair frequencies (independent of token).
    // Per ggml RoPE convention, freq[k] = freq_scale * base^(-2k/head_dim)
    // for both NORMAL and NEOX modes — only the pair-element layout differs.
    std::vector<float> freqs(n_pairs);
    for (int k = 0; k < n_pairs; ++k) {
        const float exp_arg = -(float)(2 * k) / (float)head_dim;
        freqs[k] = freq_scale * std::pow(freq_base, exp_arg);
    }

    // Step E row-major-by-token layout: qk.data[t * F_outer + i].
    // At n_tokens=1 this collapses to qk.data[i] — bit-identical to the
    // pre-Step-E column-major-by-token formula qk.data[i * T_inner + t].
    (void)T_inner;  // shape array still says {n_tokens, F}; numel matches.

    // Iterate tokens, heads, pairs.
    for (int t = 0; t < n_tokens; ++t) {
        const float pp = (float)positions[t];
        sp_ok_t* row = qk.data + (int64_t)t * F_outer;
        for (int h = 0; h < n_heads; ++h) {
            for (int k = 0; k < n_pairs; ++k) {
                const float ang = pp * freqs[k];
                const float c   = std::cos(ang);
                const float s   = std::sin(ang);

                // NORMAL: pair (2k, 2k+1).
                // NEOX:   pair (k, k + n_pairs)  i.e. (k, k + head_dim/2).
                int64_t i_even, i_odd;
                if (mode == sp_rope_mode::NEOX) {
                    i_even = (int64_t)h * head_dim + k;
                    i_odd  = (int64_t)h * head_dim + k + n_pairs;
                } else {
                    i_even = (int64_t)h * head_dim + 2 * k;
                    i_odd  = i_even + 1;
                }
                sp_ok_t& e_even = row[i_even];
                sp_ok_t& e_odd  = row[i_odd];

                const double a = (double)e_even.a / divisor;
                const double b = (double)e_odd.a  / divisor;

                const double a_rot = (double)c * a - (double)s * b;
                const double b_rot = (double)s * a + (double)c * b;

                e_even.a = (int64_t)std::llrint(a_rot * (double)S);
                e_even.b = 0;
                e_odd.a  = (int64_t)std::llrint(b_rot * (double)S);
                e_odd.b  = 0;
            }
        }
    }
    qk.frobenius_scale = 1;
    return true;
}

bool sp_rope_apply_ok_contig(sp_ok_tensor& qk,
                              int           n_heads,
                              int           head_dim,
                              int           n_tokens,
                              int           start_pos,
                              float         freq_base,
                              float         freq_scale,
                              sp_rope_mode  mode) {
    std::vector<int32_t> pos(n_tokens);
    for (int t = 0; t < n_tokens; ++t) pos[t] = start_pos + t;
    return sp_rope_apply_ok(qk, n_heads, head_dim, n_tokens,
                             pos.data(), freq_base, freq_scale, mode);
}

}  // namespace sp::engine
