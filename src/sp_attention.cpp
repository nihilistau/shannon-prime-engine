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
#include "../lib/shannon-prime/core/sp_ntt_crt.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>   // getenv
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
                                float attn_logit_softcap,
                                 const uint8_t* evicted_mask)
{
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

    // Sanity check on the per-side divisors (per-element decode happens
    // inside the inner loop now — the old single qk_divisor would have
    // overflowed double for head_dim>=256).
    if (q.scale_recip == 0 || q.frobenius_scale == 0 ||
        k.scale_recip == 0 || k.frobenius_scale == 0) return;

    // V divisor.
    const double v_divisor = (double)v.scale_recip * (double)v.frobenius_scale;
    if (v_divisor == 0.0) return;

    const int64_t S_out = out.scale_recip;
    if (S_out == 0) return;

    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::vector<float> scores(T_valid);
    std::vector<float> weights(T_valid);

    // Layout reminders (Step E — Q/out row-major-by-token, K/V cache-strided):
    //   q.data[qi * d_q   + (h*head_dim    + d)]   d_q   = n_head    * head_dim
    //   k.data[(kv_h*head_dim + d) * T_stride + t]
    //   v.data[(kv_h*head_dim + d) * T_stride + t]
    //   out.data[qi * d_q + (h*head_dim    + d)]
    //
    // At n_q=1 the qi*d_q offset is 0 and these collapse to the previous
    // single-token formulas — bit-identical to pre-Step-E semantics.
    const int64_t d_q_total = (int64_t)n_head * head_dim;

    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + (int)qi;
            const int64_t q_row_off   = qi * d_q_total + (int64_t)h * head_dim;
            const int64_t out_row_off = q_row_off;

            // SWA range: each query at position q_pos sees positions
            //   [swa_lo, q_pos]. For a global layer (swa_window=0) this
            //   degenerates to [0, q_pos], i.e. ordinary causal attention.
            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            // 1) scores[t] = (K_h[t]^T · q_h[qi]) / sqrt(head_dim).
            //
            // Phase 3.x bugfix: the previous int64 accumulator
            // overflowed at head_dim ≥ 256 because q.a * k.a is ~2^56
            // per element, summed over 256 entries → 2^64 wraps. By
            // the time attention runs, Q and K already have
            // frobenius_scale = 1 (RoPE reset), so there's no
            // Theorem-4 reason to keep the multiplication in the
            // integer domain — just decode to fp64 per element and
            // accumulate. Matches what sp_attention_poly_ring does
            // and what the unit test always assumed.
            //
            // Step E-3 causal/SWA shortcut: only compute scores in the
            // valid range [t_lo, t_hi). Positions outside this window
            // are masked to -inf for softmax (zero contribution) and
            // contribute 0 to the V sum. At single-token decode (n_q=1,
            // q_pos = T_valid - 1, swa_lo = 0 globally) this is the
            // full [0, T_valid) range — bit-identical to the unshortened
            // version. At N-token prefill the saved work is ~N/2.
            const double q_div_inner = (double)q.scale_recip *
                                          (double)q.frobenius_scale;
            const double k_div_inner = (double)k.scale_recip *
                                          (double)k.frobenius_scale;
            const int64_t t_lo = swa_lo;
            const int64_t t_hi = std::min<int64_t>((int64_t)q_pos + 1, T_valid);

            // Initialize out-of-range scores to -inf so softmax treats
            // them as zero contribution. No need for a separate mask pass.
            for (int64_t t = 0; t < t_lo; ++t)      scores[t] = NEG_INF;
            for (int64_t t = t_hi; t < T_valid; ++t) scores[t] = NEG_INF;

            for (int64_t t = t_lo; t < t_hi; ++t) {
                double acc = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    const sp_ok_t& k_dt =
                        k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    const sp_ok_t& q_d  = q.data[q_row_off + d];
                    double k_val = (double)k_dt.a / k_div_inner;
                    double q_val = (double)q_d.a  / q_div_inner;
                    acc += q_val * k_val;
                    // b-component coupling (sp_ok_mul). After RoPE both
                    // b's are 0 so this is normally a no-op, but keep
                    // the path for correctness in pre-RoPE flows.
                    if (k_dt.b != 0 || q_d.b != 0) {
                        double k_b = (double)k_dt.b / k_div_inner;
                        double q_b = (double)q_d.b  / q_div_inner;
                        acc -= (double)SP_OK_OMEGA_NORM * k_b * q_b;
                    }
                }
                scores[t] = (float)(acc * (double)inv_sqrt_d);
            }

            // 2) Apply logit softcap to the valid range only (the masked
            //    tails stay at NEG_INF — tanh(-inf/cap)*cap = -cap would
            //    corrupt the softmax denominator).
            if (attn_logit_softcap > 0.0f) {
                const float cap = attn_logit_softcap;
                const float inv_cap = 1.0f / cap;
                for (int64_t t = t_lo; t < t_hi; ++t) {
                    scores[t] = std::tanh(scores[t] * inv_cap) * cap;
                }
            }

            // 3) softmax over the full T_valid window — the NEG_INF tails
            //    softmax to 0 and contribute nothing to the normalization.
            // Phase 4d: Friedman sieve POLICY mask � final NEG_INF pass on
            // positions the sieve flagged as structurally subsumed.  Runs
            // after the in-window score compute + softcap, before softmax.
            if (evicted_mask) {
                for (int64_t t = 0; t < T_valid; ++t)
                    if (evicted_mask[t]) scores[t] = NEG_INF;
            }
            sp_softmax_bridge(scores.data(), (int)T_valid, weights.data());

            // 4) attn[d] = sum_t V_h,d,t * weights[t]; re-encode at S_out.
            //    Sum is restricted to t in [t_lo, t_hi); outside positions
            //    have weights[t] = 0 and contribute nothing.
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int64_t t = t_lo; t < t_hi; ++t) {
                    const sp_ok_t& v_dt =
                        v.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    double v_val = (double)v_dt.a / v_divisor;
                    acc += v_val * (double)weights[t];
                }
                int64_t a_out = (int64_t)std::llrint(acc * (double)S_out);
                out.data[out_row_off + d] = sp_ok_t{ a_out, 0 };
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
                              float attn_logit_softcap,
                              const uint64_t* k_ntt_slab_q1,
                              const uint64_t* k_ntt_slab_q2,
                                 const uint8_t* evicted_mask)
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

    // Phase 9b (post Plan C): NTT path is the CRT pipeline, gated by
    // SP_ENGINE_POLY_NTT=1 (defaulted on by main.cpp when attn_mode==1).
    // Falls back to scalar sp_poly_dot_product when off or when the
    // dual K-cache slabs are not supplied.
    static const bool g_use_ntt = []() {
        const char* env = std::getenv("SP_ENGINE_POLY_NTT");
        bool enabled = (env == nullptr) || (env[0] && env[0] != '0');
        if (enabled) {
            std::fprintf(stderr,
                "[sp-attention] POLY_RING CRT path ENABLED "
                "(N=%d, q1=%llu, q2=%llu)\n",
                (int)SP_NTT_CRT_N,
                (unsigned long long)SP_NTT_CRT_Q1,
                (unsigned long long)SP_NTT_CRT_Q2);
        }
        return enabled;
    }();
    // Phase 9b (post Plan C): the CRT NTT path is the only NTT path
    // the engine calls. When both dual slabs are supplied AND head_dim
    // fits the ring, route through CRT. Otherwise fall back to scalar
    // sp_poly_dot_product (the O(N^2) baseline).
    const bool use_crt_here =
        g_use_ntt && (head_dim <= SP_NTT_CRT_N)
        && (k_ntt_slab_q1 != nullptr) && (k_ntt_slab_q2 != nullptr);
    std::vector<int64_t>  crt_int_scratch;
    std::vector<uint64_t> crt_Q_q1;
    std::vector<uint64_t> crt_Q_q2;
    std::vector<uint64_t> crt_c_q1;
    std::vector<uint64_t> crt_c_q2;
    if (use_crt_here) {
        crt_int_scratch.assign(SP_NTT_CRT_N, 0);
        crt_Q_q1.assign(SP_NTT_CRT_N, 0);
        crt_Q_q2.assign(SP_NTT_CRT_N, 0);
        crt_c_q1.assign(SP_NTT_CRT_N, 0);
        crt_c_q2.assign(SP_NTT_CRT_N, 0);
    }

    // Step E layout (row-major-by-token for Q and out, K/V keep their
    // cache-strided layout):
    //   q.data[qi * d_q_total + (h*head_dim + d)]
    //   out.data[qi * d_q_total + (h*head_dim + d)]
    // At n_q=1 the qi*d_q_total offset is zero and these reduce to the
    // previous single-token formulas (bit-identical).
    const int64_t d_q_total = (int64_t)n_head * head_dim;

    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + (int)qi;
            const int64_t q_row_off   = qi * d_q_total + (int64_t)h * head_dim;
            const int64_t out_row_off = q_row_off;

            // Decode Q for this (h, qi) into fp32.
            for (int d = 0; d < head_dim; ++d) {
                const sp_ok_t& q_d = q.data[q_row_off + d];
                q_vec[d] = (float)((double)q_d.a / q_div);
            }

            // Phase 9b (post Plan C): dual-universe Q-encode + forward
            // NTT once per (h, qi). Reused for every t in the inner loop.
            if (use_crt_here) {
                sp_poly_encode_ntt_q_crt(crt_Q_q1.data(), crt_Q_q2.data(),
                                          q_vec.data(), head_dim, delta,
                                          crt_int_scratch.data());
            }

            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            // Step E-3 causal/SWA shortcut: only do the NTT-pointwise
            // dot product for t in [t_lo, t_hi). Outside positions stay
            // at NEG_INF -> 0 softmax weight, contribute nothing to V sum.
            // Bit-identical to the unshortened version (which masked the
            // same positions post-hoc).
            const int64_t t_lo = swa_lo;
            const int64_t t_hi = std::min<int64_t>((int64_t)q_pos + 1, T_valid);

            for (int64_t t = 0; t < t_lo; ++t)       scores[t] = NEG_INF;
            for (int64_t t = t_hi; t < T_valid; ++t) scores[t] = NEG_INF;

            // Per-t scores via polynomial-ring dot product, valid range only.
            for (int64_t t = t_lo; t < t_hi; ++t) {
                float dot;
                if (use_crt_here) {
                    // Phase 9b: dual-prime cached K slabs, pure
                    // pointwise + inverse + CRT stitch inner loop.
                    const size_t k_off =
                        ((size_t)kv_h * (size_t)T_stride + (size_t)t)
                            * (size_t)SP_NTT_CRT_N;
                    int ok = 0;
                    dot = sp_poly_dot_product_ntt_crt_qk_cached(
                        crt_Q_q1.data(), crt_Q_q2.data(),
                        k_ntt_slab_q1 + k_off,
                        k_ntt_slab_q2 + k_off,
                        head_dim, delta,
                        crt_c_q1.data(), crt_c_q2.data(),
                        &ok);
                    if (!ok) {
                        // Defensive fall-back, should never fire (head_dim
                        // <= SP_NTT_CRT_N is checked at use_crt_here).
                        dot = sp_poly_dot_product(
                            q_vec.data(), k_vec.data(), head_dim, N, delta,
                            poly_scratch.data());
                    }
                } else {
                    // Decode K for this (kv_h, t) only on the fallback path.
                    for (int d = 0; d < head_dim; ++d) {
                        const sp_ok_t& k_dt =
                            k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                        k_vec[d] = (float)((double)k_dt.a / k_div);
                    }
                    dot = sp_poly_dot_product(
                        q_vec.data(), k_vec.data(), head_dim, N, delta,
                        poly_scratch.data());
                }
                scores[t] = dot * inv_sqrt_d;
            }

            // Softcap on the valid range only (NEG_INF tails preserved).
            if (attn_logit_softcap > 0.0f) {
                const float cap = attn_logit_softcap;
                const float inv_cap = 1.0f / cap;
                for (int64_t t = t_lo; t < t_hi; ++t) {
                    scores[t] = std::tanh(scores[t] * inv_cap) * cap;
                }
            }

            // Softmax over the full window (NEG_INF tails softmax to 0).
            // Phase 4d: Friedman sieve POLICY mask � final NEG_INF pass on
            // positions the sieve flagged as structurally subsumed.  Runs
            // after the in-window score compute + softcap, before softmax.
            if (evicted_mask) {
                for (int64_t t = 0; t < T_valid; ++t)
                    if (evicted_mask[t]) scores[t] = NEG_INF;
            }
            sp_softmax_bridge(scores.data(), (int)T_valid, weights.data());

            // Weighted V sum over the valid range only.
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int64_t t = t_lo; t < t_hi; ++t) {
                    const sp_ok_t& v_dt =
                        v.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    double v_val = (double)v_dt.a / v_divisor;
                    acc += v_val * (double)weights[t];
                }
                int64_t a_out = (int64_t)std::llrint(acc * (double)S_out);
                out.data[out_row_off + d] = sp_ok_t{ a_out, 0 };
            }
        }
    }
    out.frobenius_scale = 1;
}

}  // namespace sp::engine
