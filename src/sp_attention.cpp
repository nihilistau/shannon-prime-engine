// Shannon-Prime Engine — Attention (Phase 2.2a — single-token multi-head).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Implements sp_attention_dot_product: classical scaled dot-product attention
// for a single query token attending over a history (KV stored as sp_ok_tensors).
//
// Pipeline per head h:
//   1. scores_h = (K_h^T · q_h) / sqrt(head_dim)    [fp32 via O_K-to-fp32 matmul]
//   2. weights_h = softmax(scores_h)                [fp32 bridge]
//   3. attn_h = V_h · weights_h                     [fp32 × O_K → O_K]
//
// Multi-head: loops h = 0..n_head-1, slicing q, k, v at h * head_dim offsets.
// GQA: n_kv_head may be < n_head; head h reads from kv slot (h * n_kv_head /
// n_head). Equal heads case (n_kv_head = n_head) collapses to standard MHA.
//
// SHAPES (single-token decode):
//   q:   shape={n_head * head_dim, 1}     row-major: q[h * head_dim + d]
//   k:   shape={n_kv_head * head_dim, T}  row-major: k[(kv_h * head_dim + d) * T + t]
//   v:   shape={n_kv_head * head_dim, T}  same layout as k
//   out: shape={n_head * head_dim, 1}     row-major: out[h * head_dim + d]
//
// where T = history_len = number of context positions including the current.
//
// Phase 2.2b will add: RoPE on Q and K, KV-cache write-back (the K and V
// passed here are already the full history including the new token), and
// multi-token prefill batching.

#include "sp_attention.h"
#include "sp_matmul.h"
#include "sp_bridges.h"
#include "sp_ok_encode.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace sp::engine {

void sp_attention_dot_product(const sp_ok_tensor& q,
                                const sp_ok_tensor& k,
                                const sp_ok_tensor& v,
                                sp_ok_tensor&       out,
                                int n_head, int n_kv_head, int head_dim) {
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr ||
        out.data == nullptr) return;
    if (n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) return;
    if (n_head % n_kv_head != 0) return;  // GQA requires divisibility

    // Recover history length T from k's INNER dim (sp_ok_tensor convention:
    // shape[0] is innermost / contiguous). k is laid out as d_q rows of
    // length T (one row per (kv_h, head_dim_d) pair), so k.data flat index
    // for d, t is (kv_h*head_dim + d) * T + t. That makes shape[0]=T.
    const int64_t T = k.shape[0];
    if (T <= 0) return;
    if (v.shape[0] != T) return;
    if (q.shape[1] != 1) return;  // single-token Phase 2.2a constraint

    // Combined scale for decode of (K^T · q): this is the divisor that
    // sp_matmul_ok_to_fp32 applies internally — k.scale_recip * q.scale_recip
    // * k.frobenius_scale * q.frobenius_scale. We just use the matmul; it
    // handles the divisor.

    // Output: per-head head_dim values, concatenated across heads.
    // The output sp_ok_tensor is at the same scale as v (since attn = V * w).
    // out.scale_recip and frobenius_scale carry from v.

    // Scratch buffers.
    std::vector<float> scores(T);       // per-head fp32 scores buffer
    std::vector<float> weights(T);      // post-softmax

    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

    // Per-head loop.
    for (int h = 0; h < n_head; ++h) {
        const int kv_h = (h * n_kv_head) / n_head;

        // 1) Compute scores = (K_h^T · q_h) / sqrt(head_dim).
        // K_h is a sub-tensor of k at the kv_h-th head: rows kv_h*head_dim
        // .. (kv_h+1)*head_dim - 1, all T columns.
        // q_h is rows h*head_dim..(h+1)*head_dim - 1, column 0.
        //
        // scores[t] = sum_d K[(kv_h*head_dim + d) * T + t] * q[h*head_dim + d]
        //           / (k.scale_recip * q.scale_recip * k.frob * q.frob *
        //              sqrt(head_dim))
        //
        // We do this inline rather than constructing sub-tensor descriptors;
        // it's cleaner for Phase 2.2a. Phase 2.2b can switch to view-based.
        const double divisor = (double)k.scale_recip * (double)q.scale_recip *
                                (double)k.frobenius_scale * (double)q.frobenius_scale;
        if (divisor == 0.0) continue;

        for (int t = 0; t < T; ++t) {
            int64_t acc_a = 0;
            for (int d = 0; d < head_dim; ++d) {
                const sp_ok_t& k_dt = k.data[((int64_t)kv_h * head_dim + d) * T + t];
                const sp_ok_t& q_d  = q.data[(int64_t)h    * head_dim + d];
                acc_a += k_dt.a * q_d.a - SP_OK_OMEGA_NORM * k_dt.b * q_d.b;
            }
            scores[t] = (float)(((double)acc_a / divisor) * (double)inv_sqrt_d);
        }

        // 2) Softmax over the T scores (causal mask already implicit since
        //    caller passes only valid history positions — Phase 2.2b's
        //    sp_forward_step will manage that).
        sp_softmax_bridge(scores.data(), T, weights.data());

        // 3) attn_h[d] = sum_t V[(kv_h*head_dim + d) * T + t] * weights[t].
        // V stays in O_K, weights are fp32. This is sp_matmul_fp32_input_to_ok
        // semantics, but on a slice — we do it inline.
        const double v_divisor = (double)v.scale_recip * (double)v.frobenius_scale;
        const int64_t S_out = out.scale_recip;
        if (v_divisor == 0.0 || S_out == 0) continue;
        // The decoded V·weights value gets re-encoded at out.scale_recip.
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) {
                const sp_ok_t& v_dt = v.data[((int64_t)kv_h * head_dim + d) * T + t];
                double v_val = (double)v_dt.a / v_divisor;
                acc += v_val * (double)weights[t];
            }
            int64_t a_out = (int64_t)std::llrint(acc * (double)S_out);
            out.data[(int64_t)h * head_dim + d] = sp_ok_t{ a_out, 0 };
        }
    }
    out.frobenius_scale = 1;  // attention output is at fp32-encoded scale 1
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

}  // namespace sp::engine
