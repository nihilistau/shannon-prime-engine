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
#include "../lib/shannon-prime/core/sp_ntt.h"
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
            const double q_div_inner = (double)q.scale_recip *
                                          (double)q.frobenius_scale;
            const double k_div_inner = (double)k.scale_recip *
                                          (double)k.frobenius_scale;
            for (int64_t t = 0; t < T_valid; ++t) {
                double acc = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    const sp_ok_t& k_dt =
                        k.data[((int64_t)kv_h * head_dim + d) * T_stride + t];
                    const sp_ok_t& q_d  =
                        q.data[((int64_t)h    * head_dim + d) * n_q + qi];
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
                              float attn_logit_softcap,
                              const uint64_t* k_ntt_slab,
                              const uint64_t* k_ntt_slab_q1,
                              const uint64_t* k_ntt_slab_q2) {
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

    // Phase 4: NTT-accelerated polynomial multiply when N matches the
    // pre-computed constant tables (SP_NTT_N = 256, q ≈ 2^60). Opt-in
    // via SP_ENGINE_POLY_NTT=1; falls back to O(N^2) sp_poly_dot_product
    // otherwise. Logged once per process.
    static const bool g_use_ntt = []() {
        const char* env = std::getenv("SP_ENGINE_POLY_NTT");
        bool enabled = env && env[0] && env[0] != '0';
        if (enabled) {
            std::fprintf(stderr, "[sp-attention] POLY_RING NTT path ENABLED "
                "(SP_NTT_N=%d, q=%llu)\n",
                (int)SP_NTT_N, (unsigned long long)SP_NTT_Q);
        }
        return enabled;
    }();
    const bool use_ntt_here = g_use_ntt && (N == SP_NTT_N);
    // Phase 9b: route through the CRT NTT path when both dual slabs are
    // supplied AND the ring size matches. The CRT path is preferred over
    // the 60-bit single-prime path when active because it has no
    // __int128 dependency.
    const bool use_crt_here =
        g_use_ntt && (head_dim <= SP_NTT_CRT_N)
        && (k_ntt_slab_q1 != nullptr) && (k_ntt_slab_q2 != nullptr);
    // Phase 5b: hoist NTT(Q) out of the per-t inner loop. Phase 6: also
    // pre-NTT every K once per call into K_ntt_cache so the (qi, t) inner
    // loop is just pointwise multiply + inverse + extract. K's NTT only
    // depends on (kv_h, t), not on (h, qi), so each K is transformed once
    // and reused across all queries in this attention call.
    std::vector<int64_t>  ntt_Q_int_scratch;       // [SP_NTT_N]
    std::vector<uint64_t> ntt_Q_buf;               // [SP_NTT_N]
    std::vector<int64_t>  ntt_K_int_scratch;       // [SP_NTT_N]
    std::vector<uint64_t> ntt_C_buf;               // [SP_NTT_N]
    std::vector<uint64_t> K_ntt_cache;             // [n_kv_head * T_valid * SP_NTT_N]
    std::vector<float>    k_decode_buf;            // [head_dim]
    // Phase 9b CRT scratch (dual-universe Q + per-call inverse buffers).
    std::vector<int64_t>  crt_int_scratch;         // [SP_NTT_CRT_N] shared encoder workspace
    std::vector<uint64_t> crt_Q_q1;                // [SP_NTT_CRT_N]
    std::vector<uint64_t> crt_Q_q2;                // [SP_NTT_CRT_N]
    std::vector<uint64_t> crt_c_q1;                // [SP_NTT_CRT_N]
    std::vector<uint64_t> crt_c_q2;                // [SP_NTT_CRT_N]
    if (use_crt_here) {
        crt_int_scratch.assign(SP_NTT_CRT_N, 0);
        crt_Q_q1.assign(SP_NTT_CRT_N, 0);
        crt_Q_q2.assign(SP_NTT_CRT_N, 0);
        crt_c_q1.assign(SP_NTT_CRT_N, 0);
        crt_c_q2.assign(SP_NTT_CRT_N, 0);
    }
    if (use_ntt_here && !use_crt_here) {
        ntt_Q_int_scratch.assign(SP_NTT_N, 0);
        ntt_Q_buf.assign(SP_NTT_N, 0);
        ntt_K_int_scratch.assign(SP_NTT_N, 0);
        ntt_C_buf.assign(SP_NTT_N, 0);
        if (k_ntt_slab == nullptr) {
            // Phase 6: in-call build of the K-NTT cache.
            K_ntt_cache.assign((size_t)n_kv_head * (size_t)T_valid * (size_t)SP_NTT_N, 0);
            k_decode_buf.assign(head_dim, 0.0f);
            for (int kvh = 0; kvh < n_kv_head; ++kvh) {
                for (int64_t t = 0; t < T_valid; ++t) {
                    for (int d = 0; d < head_dim; ++d) {
                        const sp_ok_t& k_dt =
                            k.data[((int64_t)kvh * head_dim + d) * T_stride + t];
                        k_decode_buf[d] = (float)((double)k_dt.a / k_div);
                    }
                    uint64_t* slot = K_ntt_cache.data() +
                        ((size_t)kvh * (size_t)T_valid + (size_t)t) * (size_t)SP_NTT_N;
                    sp_poly_encode_ntt_k_reversed(slot, k_decode_buf.data(),
                                                  head_dim, delta,
                                                  ntt_K_int_scratch.data());
                }
            }
        }
        // Else: Phase 7 — caller supplied a persistent slab. We index into it
        // directly at (kvh * T_stride + t) * SP_NTT_N during the inner loop.
    }

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

            // Phase 5b: hoist NTT(Q) once per (h, qi).
            if (use_ntt_here && !use_crt_here) {
                sp_poly_encode_ntt_q(ntt_Q_buf.data(),
                                     q_vec.data(), head_dim, delta,
                                     ntt_Q_int_scratch.data());
            }
            // Phase 9b: dual-universe Q-encode + forward NTT once per (h, qi).
            if (use_crt_here) {
                sp_poly_encode_ntt_q_crt(crt_Q_q1.data(), crt_Q_q2.data(),
                                          q_vec.data(), head_dim, delta,
                                          crt_int_scratch.data());
            }

            const int swa_lo = (swa_window > 0)
                ? std::max(0, q_pos - swa_window + 1)
                : 0;

            // Per-t scores via polynomial-ring dot product.
            for (int64_t t = 0; t < T_valid; ++t) {
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
                } else if (use_ntt_here) {
                    const uint64_t* K_ntt = (k_ntt_slab != nullptr)
                        // Phase 7: persistent slab, indexed by t_stride.
                        ? (k_ntt_slab +
                            ((size_t)kv_h * (size_t)T_stride + (size_t)t) * (size_t)SP_NTT_N)
                        // Phase 6: in-call cache, indexed by T_valid.
                        : (K_ntt_cache.data() +
                            ((size_t)kv_h * (size_t)T_valid + (size_t)t) * (size_t)SP_NTT_N);
                    dot = sp_poly_dot_product_ntt_qk_cached(
                        ntt_Q_buf.data(), K_ntt, head_dim, delta,
                        ntt_C_buf.data());
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
