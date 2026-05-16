// Shannon-Prime Engine — native O_K matrix multiplication (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_matmul.h"

#include <cmath>
#include <cstring>

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
}

namespace sp::engine {

// ---------- O_K @ O_K → O_K ------------------------------------------------

bool sp_matmul_ok(const sp_ok_tensor& W,
                   const sp_ok_tensor& X,
                   sp_ok_tensor&       Y) {
    // Shape contract:
    //   W: [out_rows, in_cols] -> shape[0]=in_cols (inner), shape[1]=out_rows
    //   X: [in_cols, n_cols]   -> shape[0]=n_cols  (inner), shape[1]=in_cols
    // Wait — our convention (sp_ok_tensor::reset) is shape[0] is innermost
    // and contiguous. For row-major matmul of W [M, K] * X [K, N]:
    //   shape[0]=K (innermost), shape[1]=M for W
    //   shape[0]=N (innermost), shape[1]=K for X
    //   Y: shape[0]=N, shape[1]=M
    if (W.data == nullptr || X.data == nullptr) return false;
    if (W.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M = W.shape[1];       // out_rows
    const int64_t K = W.shape[0];       // in_cols
    const int64_t K2 = X.shape[1];      // X's row count
    const int64_t N = X.shape[0];       // n_cols
    if (K != K2) return false;
    if (Y.data == nullptr) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;

    // Y[i,j] = sum_k W[i,k] * X[k,j]
    // For each output element, accumulate the O_K dot product into a
    // running (a, b) total. The a-coordinate sum stays in int64 for
    // typical dims; for very large K we should switch to __int128.
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t acc_a = 0;
            int64_t acc_b = 0;
            for (int64_t k = 0; k < K; ++k) {
                const sp_ok_t& w_ik = W.data[i * K + k];
                const sp_ok_t& x_kj = X.data[k * N + j];
                // sp_ok_mul:
                //   r.a = w.a * x.a - 41 * w.b * x.b
                //   r.b = w.a * x.b + x.a * w.b + w.b * x.b
                acc_a += w_ik.a * x_kj.a - SP_OK_OMEGA_NORM * w_ik.b * x_kj.b;
                acc_b += w_ik.a * x_kj.b + x_kj.a * w_ik.b + w_ik.b * x_kj.b;
            }
            Y.data[i * N + j] = sp_ok_t{ acc_a, acc_b };
        }
    }

    Y.scale_recip = W.scale_recip * X.scale_recip;
    Y.frobenius_scale = W.frobenius_scale * X.frobenius_scale;
    return true;
}

// ---------- O_K @ O_K → fp32 (softmax/silu bridge) ------------------------

bool sp_matmul_ok_to_fp32(const sp_ok_tensor& W,
                           const sp_ok_tensor& X,
                           float*              Y_fp32,
                           int                 out_rows,
                           int                 n_cols) {
    if (W.data == nullptr || X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M = W.shape[1];
    const int64_t K = W.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;

    // Same combined scale as the all-O_K path; we just decode at the end.
    const double divisor = (double)W.scale_recip * (double)X.scale_recip *
                           (double)W.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t acc_a = 0;
            for (int64_t k = 0; k < K; ++k) {
                const sp_ok_t& w_ik = W.data[i * K + k];
                const sp_ok_t& x_kj = X.data[k * N + j];
                // Only need acc_a for fp32 bridge — the b-coordinate is
                // dropped at fp32 boundaries (caller's nonlinearity isn't
                // O_K-aware).
                acc_a += w_ik.a * x_kj.a - SP_OK_OMEGA_NORM * w_ik.b * x_kj.b;
            }
            Y_fp32[i * N + j] = (float)((double)acc_a / divisor);
        }
    }
    return true;
}

// ---------- fp32 input × O_K weights → O_K output -------------------------

bool sp_matmul_fp32_input_to_ok(const sp_ok_tensor& W,
                                  const float*        X_fp32,
                                  int                 in_cols,
                                  int                 n_cols,
                                  sp_ok_tensor&       Y) {
    if (W.data == nullptr || X_fp32 == nullptr) return false;
    if (W.n_dims < 2) return false;
    const int64_t M = W.shape[1];
    const int64_t K = W.shape[0];
    if ((int64_t)in_cols != K) return false;
    const int64_t N = n_cols;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if (Y.data == nullptr) return false;

    // Re-encode X_fp32 at W's scale so the int64 product stays accurate.
    // Y inherits the combined scale.
    const int64_t S = W.scale_recip;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t acc_a = 0;
            int64_t acc_b = 0;
            for (int64_t k = 0; k < K; ++k) {
                const sp_ok_t& w_ik = W.data[i * K + k];
                // Encode X_fp32 element on the fly.
                int64_t x_a = (int64_t)std::llrint((double)X_fp32[k * N + j] * (double)S);
                // x has b=0 (fp32 input is "scalar in omega direction").
                acc_a += w_ik.a * x_a;
                acc_b += w_ik.b * x_a;
            }
            Y.data[i * N + j] = sp_ok_t{ acc_a, acc_b };
        }
    }

    Y.scale_recip = W.scale_recip * S;   // S * S = scale_recip squared
    Y.frobenius_scale = W.frobenius_scale;
    return true;
}

}  // namespace sp::engine
