// Shannon-Prime Engine — native O_K matrix multiplication (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.3c: outer-M parallelization via sp_threadpool. The M axis
// (output rows) is embarrassingly parallel — each output row's
// accumulator is independent. Partition rows across threads; each
// worker handles its slice with the same inner loop.

#include "sp_matmul.h"
#include "sp_threadpool.h"

#include <cmath>
#include <cstring>

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
}

namespace sp::engine {

// Compute [m_start, m_end) given thread_id and total work M.
static inline void split_rows(int64_t M, int n_threads, int thread_id,
                                int64_t& m_start, int64_t& m_end) {
    const int64_t per = (M + n_threads - 1) / n_threads;
    m_start = (int64_t)thread_id * per;
    m_end   = std::min(m_start + per, M);
}

// ---------- O_K @ O_K → O_K ------------------------------------------------

bool sp_matmul_ok(const sp_ok_tensor& W,
                   const sp_ok_tensor& X,
                   sp_ok_tensor&       Y) {
    // Shape contract (sp_ok_tensor convention: shape[0] is innermost):
    //   W: shape[0]=K (in_cols, inner), shape[1]=M (out_rows)
    //   X: shape[0]=N (n_cols, inner),  shape[1]=K (in_cols)
    //   Y: shape[0]=N (inner),          shape[1]=M
    if (W.data == nullptr || X.data == nullptr) return false;
    if (W.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M = W.shape[1];
    const int64_t K = W.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N = X.shape[0];
    if (K != K2) return false;
    if (Y.data == nullptr) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    // Y[i,j] = sum_k W[i,k] * X[k,j]
    // Each output row i is independent → partition M across threads.
    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_t& w_ik = W.data[i * K + k];
                    const sp_ok_t& x_kj = X.data[k * N + j];
                    acc_a += w_ik.a * x_kj.a - SP_OK_OMEGA_NORM * w_ik.b * x_kj.b;
                    acc_b += w_ik.a * x_kj.b + x_kj.a * w_ik.b + w_ik.b * x_kj.b;
                }
                Y.data[i * N + j] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

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

    const double divisor = (double)W.scale_recip * (double)X.scale_recip *
                           (double)W.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_t& w_ik = W.data[i * K + k];
                    const sp_ok_t& x_kj = X.data[k * N + j];
                    acc_a += w_ik.a * x_kj.a - SP_OK_OMEGA_NORM * w_ik.b * x_kj.b;
                }
                Y_fp32[i * N + j] = (float)((double)acc_a / divisor);
            }
        }
    });
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

    const int64_t S = W.scale_recip;
    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_t& w_ik = W.data[i * K + k];
                    int64_t x_a =
                        (int64_t)std::llrint((double)X_fp32[k * N + j] * (double)S);
                    acc_a += w_ik.a * x_a;
                    acc_b += w_ik.b * x_a;
                }
                Y.data[i * N + j] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip = W.scale_recip * S;
    Y.frobenius_scale = W.frobenius_scale;
    return true;
}

}  // namespace sp::engine
