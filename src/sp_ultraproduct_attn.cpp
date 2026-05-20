// Shannon-Prime Engine — Phase 7: Ultraproduct attention implementation.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// See sp_ultraproduct_attn.h for the design.  This file implements the
// PRINCIPAL case only — argmax-based hard attention.  The score-compute
// pipeline matches sp_attention_dot_product exactly so the same SWA
// window, softcap, and sieve mask apply.

#include "sp_ultraproduct_attn.h"
#include "sp_attention.h"
#include "sp_ok_encode.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

namespace sp::engine {

void sp_ultraproduct_attn_principal(const sp_ok_tensor& q,
                                      const sp_ok_tensor& k,
                                      const sp_ok_tensor& v,
                                      sp_ok_tensor&       out,
                                      int n_head, int n_kv_head, int head_dim,
                                      int   t_valid_arg,
                                      int   t_stride_arg,
                                      int   pos_offset_arg,
                                      int   swa_window,
                                      float attn_logit_softcap,
                                      const uint8_t* evicted_mask,
                                      float evicted_gamma,
                                      int32_t* selected_pos)
{
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr ||
        out.data == nullptr) return;
    if (n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) return;
    if (n_head % n_kv_head != 0) return;

    const int64_t T_stride = (t_stride_arg < 0) ? k.shape[0] : (int64_t)t_stride_arg;
    const int64_t T_valid  = (t_valid_arg  < 0) ? k.shape[0] : (int64_t)t_valid_arg;
    if (T_stride <= 0 || T_valid <= 0) return;
    if (T_valid > T_stride) return;
    if (v.shape[0] != T_stride) return;

    const int64_t n_q = q.shape[0];
    if (n_q <= 0) return;
    if (q.shape[1] != n_head * head_dim) return;
    if (out.shape[0] != n_q) return;
    if (out.shape[1] != n_head * head_dim) return;

    const int pos_offset = (pos_offset_arg < 0)
                             ? (int)(T_valid - n_q) : pos_offset_arg;

    if (q.scale_recip == 0 || q.frobenius_scale == 0 ||
        k.scale_recip == 0 || k.frobenius_scale == 0) return;

    const double v_divisor = (double)v.scale_recip * (double)v.frobenius_scale;
    if (v_divisor == 0.0) return;

    const int64_t S_out = out.scale_recip;
    if (S_out == 0) return;

    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::vector<float> scores(T_valid);

    const int64_t d_q_total = (int64_t)n_head * head_dim;

    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + (int)qi;
            const int64_t q_row_off   = qi * d_q_total + (int64_t)h * head_dim;
            const int64_t out_row_off = q_row_off;

            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            const double q_div_inner = (double)q.scale_recip *
                                          (double)q.frobenius_scale;
            const double k_div_inner = (double)k.scale_recip *
                                          (double)k.frobenius_scale;
            const int64_t t_lo = swa_lo;
            const int64_t t_hi = std::min<int64_t>((int64_t)q_pos + 1, T_valid);

            for (int64_t t = 0; t < t_lo; ++t)       scores[t] = NEG_INF;
            for (int64_t t = t_hi; t < T_valid; ++t) scores[t] = NEG_INF;

            // 1) Score compute — identical to sp_attention_dot_product.
            for (int64_t t = t_lo; t < t_hi; ++t) {
                double acc = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    const sp_ok_t& k_dt =
                        k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    const sp_ok_t& q_d  = q.data[q_row_off + d];
                    double k_val = (double)k_dt.a / k_div_inner;
                    double q_val = (double)q_d.a  / q_div_inner;
                    acc += q_val * k_val;
                    if (k_dt.b != 0 || q_d.b != 0) {
                        double k_b = (double)k_dt.b / k_div_inner;
                        double q_b = (double)q_d.b  / q_div_inner;
                        acc -= (double)SP_OK_OMEGA_NORM * k_b * q_b;
                    }
                }
                scores[t] = (float)(acc * (double)inv_sqrt_d);
            }

            // 2) Softcap on the valid range.
            if (attn_logit_softcap > 0.0f) {
                const float cap = attn_logit_softcap;
                const float inv_cap = 1.0f / cap;
                for (int64_t t = t_lo; t < t_hi; ++t) {
                    scores[t] = std::tanh(scores[t] * inv_cap) * cap;
                }
            }

            // 3) Sieve mask — same semantics as sp_attention_dot_product.
            //    The mask still influences which key wins the argmax;
            //    soft γ pulls down the score by exactly γ instead of
            //    going to NEG_INF.
            if (evicted_mask) {
                if (evicted_gamma > 0.0f) {
                    for (int64_t t = 0; t < T_valid; ++t)
                        if (evicted_mask[t]) scores[t] -= evicted_gamma;
                } else {
                    for (int64_t t = 0; t < T_valid; ++t)
                        if (evicted_mask[t]) scores[t] = NEG_INF;
                }
            }

            // 4) Principal-ultrafilter reduction: pick argmax_t scores[t]
            //    over [t_lo, t_hi). Break ties by lower index (defines
            //    the principal ultrafilter deterministically).
            //
            //    If the entire valid window has been NEG_INF-masked
            //    (sieve evicted every reachable key — unlikely but
            //    possible) we fall back to attending to the query
            //    position itself, i.e. p* = q_pos. This is the
            //    "empty-ultrafilter" degeneracy and matches the
            //    Paper III §5.4 finite-window convention: every
            //    bounded cache has at least one principal ultrafilter,
            //    namely U_{q_pos}.
            int64_t p_star = -1;
            float   best   = -std::numeric_limits<float>::infinity();
            for (int64_t t = t_lo; t < t_hi; ++t) {
                if (scores[t] > best) {
                    best   = scores[t];
                    p_star = t;
                }
            }
            if (p_star < 0) {
                p_star = (int64_t)q_pos;
            }

            if (selected_pos) {
                selected_pos[qi * n_head + h] = (int32_t)p_star;
            }

            // 5) out_h[d] = V_h[d, p*] re-encoded at S_out.
            //    ult_{U_{p*}}(V_t) = V_{p*} (Łoś / principal case).
            for (int d = 0; d < head_dim; ++d) {
                const sp_ok_t& v_dp =
                    v.data[((int64_t)kv_h * head_dim + d) * T_stride + p_star];
                double v_val = (double)v_dp.a / v_divisor;
                int64_t a_out = (int64_t)std::llrint(v_val * (double)S_out);
                out.data[out_row_off + d] = sp_ok_t{ a_out, 0 };
            }
        }
    }
    out.frobenius_scale = 1;
}

}  // namespace sp::engine
