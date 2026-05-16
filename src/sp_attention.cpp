// Shannon-Prime Engine — Attention (Phase 2.2b — multi-token + causal mask).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Scaled dot-product attention for either:
//   (a) a single query token over a history of length T_valid, or
//   (b) a batch of n_q query tokens (prefill) over a history of length T_valid
//       with an applied causal mask.
//
// Pipeline per head h, per query q:
//   1. scores[t]  = (K_h^T · q_h_q) / sqrt(head_dim)
//                   for t in [0, t_valid)
//   2. mask: scores[t > pos_q] = -inf  (causal mask, pos_q = pos_offset + q)
//   3. weights  = softmax(scores)
//   4. attn[d]  = sum_t V_h_t,d * weights[t]
//
// KV layout:
//   k.data[(kv_h*head_dim + d) * t_stride + t]
//   v.data[(kv_h*head_dim + d) * t_stride + t]
//
// `t_stride` may be larger than `t_valid` (the KV cache view case, where
// shape[0] reflects the cache's max_len but only [0, cur_len) is valid).
//
// Phase 2.2b adds: multi-token prefill (n_q > 1), causal mask, stride/valid
// split. Phase 2.2a's single-token n_q=1 behavior is preserved as the
// default for callers that don't pass the new args.

#include "sp_attention.h"
#include "sp_matmul.h"
#include "sp_bridges.h"
#include "sp_ok_encode.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_poly_ring.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace sp::engine {

void sp_attention_dot_product(const sp_ok_tensor& q,
                                const sp_ok_tensor& k,
                                const sp_ok_tensor& v,
                                sp_ok_tensor&       out,
                                int n_head, int n_kv_head, int head_dim,
                                int   t_valid_arg,
                                int   t_stride_arg,
                                int   pos_offset_arg,
                                int   swa_window,
                                float attn_logit_softcap) {
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr ||
        out.data == nullptr) return;
    if (n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) return;
    if (n_head % n_kv_head != 0) return;

    // Recover stride and valid length.
    const int64_t T_stride = (t_stride_arg < 0) ? k.shape[0] : (int64_t)t_stride_arg;
    const int64_t T_valid  = (t_valid_arg  < 0) ? k.shape[0] : (int64_t)t_valid_arg;
    if (T_stride <= 0 || T_valid <= 0) return;
    if (T_valid > T_stride) return;
    if (v.shape[0] != T_stride) return;

    // Number of query tokens.
    const int64_t n_q = q.shape[0];
    if (n_q <= 0) return;
    if (q.shape[1] != n_head * head_dim) return;
    if (out.shape[0] != n_q) return;
    if (out.shape[1] != n_head * head_dim) return;

    const int pos_offset = (pos_offset_arg < 0)
                             ? (int)(T_valid - n_q) : pos_offset_arg;

    // Combined divisor for QK^T dot product.
    const double qk_divisor = (double)k.scale_recip * (double)q.scale_recip *
                               (double)k.frobenius_scale * (double)q.frobenius_scale;
    if (qk_divisor == 0.0) return;

    // V divisor.
    const double v_divisor = (double)v.scale_recip * (double)v.frobenius_scale;
    if (v_divisor == 0.0) return;

    const int64_t S_out = out.scale_recip;
    if (S_out == 0) return;

    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::vector<float> scores(T_valid);
    std::vector<float> weights(T_valid);

    // Layout reminders (matches sp_matmul output shape = {N, M}):
    //   q.data[(h*head_dim + d) * n_q + qi]   for qi in [0,n_q), d in [0,head_dim)
    //   k.data[(kv_h*head_dim + d) * T_stride + t]
    //   v.data[(kv_h*head_dim + d) * T_stride + t]
    //   out.data[(h*head_dim + d) * n_q + qi]

    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + (int)qi;

            // SWA range: each query at position q_pos sees positions
            //   [swa_lo, q_pos]. For a global layer (swa_window=0) this
            //   degenerates to [0, q_pos], i.e. ordinary causal attention.
            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            // 1) scores[t] = (K_h[t]^T · q_h[qi]) / sqrt(head_dim)
            for (int64_t t = 0; t < T_valid; ++t) {
                int64_t acc_a = 0;
                for (int d = 0; d < head_dim; ++d) {
                    const sp_ok_t& k_dt =
                        k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    const sp_ok_t& q_d  =
                        q.data[((int64_t)h    * head_dim + d) * n_q + qi];
                    acc_a += k_dt.a * q_d.a
                           - SP_OK_OMEGA_NORM * k_dt.b * q_d.b;
                }
                scores[t] = (float)(((double)acc_a / qk_divisor)
                                    * (double)inv_sqrt_d);
            }

            // 2a) Apply logit softcap BEFORE masking so the masked positions
            //     stay -inf (tanh(-inf/cap)*cap == -cap which would corrupt
            //     the softmax denominator). Pre-cap is the same shape as
            //     ggml_flash_attn_ext's softcap (applied to QK^T/sqrt(d)).
            if (attn_logit_softcap > 0.0f) {
                const float cap = attn_logit_softcap;
                const float inv_cap = 1.0f / cap;
                for (int64_t t = 0; t < T_valid; ++t) {
                    scores[t] = std::tanh(scores[t] * inv_cap) * cap;
                }
            }

            // 2b) Causal mask + SWA mask: positions t > q_pos OR
            //     t < swa_lo are -inf.
            for (int64_t t = 0; t < T_valid; ++t) {
                if ((int)t > q_pos || (int)t < swa_lo) scores[t] = NEG_INF;
            }

            // 3) softmax
            sp_softmax_bridge(scores.data(), (int)T_valid, weights.data());

            // 4) attn[d] = sum_t V_h,d,t * weights[t]; re-encode at S_out.
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int64_t t = 0; t < T_valid; ++t) {
                    const sp_ok_t& v_dt =
                        v.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    double v_val = (double)v_dt.a / v_divisor;
                    acc += v_val * (double)weights[t];
                }
                int64_t a_out = (int64_t)std::llrint(acc * (double)S_out);
                out.data[((int64_t)h * head_dim + d) * n_q + qi] =
                    sp_ok_t{ a_out, 0 };
            }
        }
    }
    out.frobenius_scale = 1;
}

// -----------------------------------------------------------------------
// Weil-pairing attention — Phase 4 (still stub).
// -----------------------------------------------------------------------

void sp_attention_weil_pairing(const sp_ok_tensor& q,
                                 const sp_ok_tensor& k,
                                 const sp_ok_tensor& v,
                                 sp_ok_tensor&       out,
                                 int n) {
    (void)q; (void)k; (void)v; (void)out; (void)n;
    // Phase 4. Reference: test-suite/src/weil_pairing.py and Paper A §9.2.
}

// -----------------------------------------------------------------------
// Phase 3 pivot — CKKS polynomial-ring attention.
//
// Drop-in for sp_attention_dot_product. Same masking + softmax + V
// weighting. The score function is replaced: instead of decoding the
// O_K dot product through fp32 directly, we encode the per-(qi, t)
// head-vector pair as integer polynomials in Z[x]/(x^N+1) and recover
// Σ q_d k_d from the x^{d-1} coefficient of Q(x) * K_rev(x).
//
// All Q/K/V reads stay in the same O_K representation as the standard
// path; only the score function changes. The encoder takes fp32 (we
// decode from O_K once per dot product), so the existing scale_recip /
// frobenius_scale dance is preserved.
// -----------------------------------------------------------------------

// Smallest power of 2 ≥ x.
static inline int next_pow2_ge(int x) {
    int n = 1;
    while (n < x) n <<= 1;
    return n;
}

void sp_attention_poly_ring(const sp_ok_tensor& q,
                              const sp_ok_tensor& k,
                              const sp_ok_tensor& v,
                              sp_ok_tensor&       out,
                              int n_head, int n_kv_head, int head_dim,
                              int   t_valid_arg,
                              int   t_stride_arg,
                              int   pos_offset_arg,
                              int   swa_window,
                              float attn_logit_softcap) {
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

    // O_K decode divisors. Q and K are presumed to come from the same
    // scale/frob pipeline as the standard dot-product attention, so
    // their divisors are the same combined factor.
    const double q_div = (double)q.scale_recip * (double)q.frobenius_scale;
    const double k_div = (double)k.scale_recip * (double)k.frobenius_scale;
    if (q_div == 0.0 || k_div == 0.0) return;

    const double v_divisor = (double)v.scale_recip * (double)v.frobenius_scale;
    if (v_divisor == 0.0) return;
    const int64_t S_out = out.scale_recip;
    if (S_out == 0) return;

    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    // Polynomial-ring setup.
    const int N = next_pow2_ge(head_dim);  // ring degree
    // Encoder scale Δ. Per the unit-test results:
    //   d=16  delta=2^14 → err ~1e-5
    //   d=64  delta=2^14 → err ~1e-5
    //   d=256 delta=2^10 → err ~7e-5
    // For arbitrary head_dim pick Δ so that (Δ * |value|)² * head_dim
    // stays well under 2^62 (single-multiply int64 headroom).
    // Empirically: Δ = 2^min(14, 22 - log2(head_dim) - 1).
    int log2_d = 0;
    while ((1 << log2_d) < head_dim) ++log2_d;
    int delta_bits = 14;
    if (22 - log2_d - 1 < delta_bits) delta_bits = 22 - log2_d - 1;
    if (delta_bits < 8) delta_bits = 8;
    const double delta = (double)(1LL << delta_bits);

    // Decode buffers + polynomial scratch.
    std::vector<float> q_vec(head_dim);
    std::vector<float> k_vec(head_dim);
    std::vector<sp_poly_coeff> poly_scratch(3 * N);
    std::vector<float> scores(T_valid);
    std::vector<float> weights(T_valid);

    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + (int)qi;

            // Decode Q for this (h, qi) into fp32.
            for (int d = 0; d < head_dim; ++d) {
                const sp_ok_t& q_d =
                    q.data[((int64_t)h * head_dim + d) * n_q + qi];
                q_vec[d] = (float)((double)q_d.a / q_div);
            }

            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            // Per-t scores via polynomial-ring dot product.
            for (int64_t t = 0; t < T_valid; ++t) {
                // Decode K for this (kv_h, t).
                for (int d = 0; d < head_dim; ++d) {
                    const sp_ok_t& k_dt =
                        k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    k_vec[d] = (float)((double)k_dt.a / k_div);
                }
                float dot = sp_poly_dot_product(
                    q_vec.data(), k_vec.data(), head_dim, N, delta,
                    poly_scratch.data());
                scores[t] = dot * inv_sqrt_d;
            }

            // Softcap before masking.
            if (attn_logit_softcap > 0.0f) {
                const float cap = attn_logit_softcap;
                const float inv_cap = 1.0f / cap;
                for (int64_t t = 0; t < T_valid; ++t) {
                    scores[t] = std::tanh(scores[t] * inv_cap) * cap;
                }
            }

            // Causal + SWA mask.
            for (int64_t t = 0; t < T_valid; ++t) {
                if ((int)t > q_pos || (int)t < swa_lo) scores[t] = NEG_INF;
            }

            // Softmax.
            sp_softmax_bridge(scores.data(), (int)T_valid, weights.data());

            // Weighted V sum, re-encode at S_out.
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int64_t t = 0; t < T_valid; ++t) {
                    const sp_ok_t& v_dt =
                        v.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    double v_val = (double)v_dt.a / v_divisor;
                    acc += v_val * (double)weights[t];
                }
                int64_t a_out = (int64_t)std::llrint(acc * (double)S_out);
                out.data[((int64_t)h * head_dim + d) * n_q + qi] =
                    sp_ok_t{ a_out, 0 };
            }
        }
    }
    out.frobenius_scale = 1;
}

}  // namespace sp::engine
