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
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) || defined(__AVX512F__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__)))
#  include <immintrin.h>
#  define SP_MATMUL_HAVE_SIMD 1
#endif

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
    // Shape contract (Step E layout flip — matches the rest of the engine's
    // token-as-row physical layout used by rmsnorm/embed/RoPE/KV-append):
    //   W: out_rows M × in_cols K, row-major as W.data[i*K + k]
    //   X: n_cols N × in_cols K,   row-major as X.data[j*K + k]
    //              (token j's features are contiguous; embed_lookup writes
    //              this layout, rmsnorm reads/writes this layout)
    //   Y: n_cols N × out_rows M,  row-major as Y.data[j*M + i]
    //
    // At N=1 this collapses to flat[k] / flat[i] — bit-identical to the
    // pre-Step-E column-major-by-token convention, so all N=1 PPL
    // measurements remain valid.
    //
    // shape[] validation kept as {N, K}/{K, M}/{N, M} for numel/error
    // checking; the "shape[0] is innermost" sp_ok_tensor convention is a
    // soft convention not enforced by the kernel.
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
    //
    // Phase 8 GEMV fast-path: when N==1 (autoregressive decoding), W[i,k]
    // and X[k] are both contiguous over k. We vectorize the inner-k
    // reduction with two independent vector accumulators sum_m1 / sum_m2
    // (no in-loop blends), horizontal-sum once at the end.
    //
    // Lane math (interpreting each pair of int64 sp_ok_t as [a, b]):
    //   _mm256_mul_epi32(w, x)        -> [w0.a*x0.a, w0.b*x0.b, w1.a*x1.a, w1.b*x1.b]
    //   _mm256_mul_epi32(w, swap(x))  -> [w0.a*x0.b, w0.b*x0.a, w1.a*x1.b, w1.b*x1.a]
    // Reduces to:
    //   acc_a = Σ w.a*x.a - 41 * Σ w.b*x.b
    //   acc_b = Σ w.a*x.b + Σ w.b*x.a + Σ w.b*x.b
    //
    // Safety: _mm256_mul_epi32 reads the LOW 32 bits as signed int32.
    // Caller must ensure |W.data[*].a|, |W.data[*].b|, |X.data[*].a|,
    // |X.data[*].b| all fit in int32 (i.e., < 2^31). For no-shim weights
    // at scale_recip=2^14, values are O(2^14) and safely within range.
    // Safety gate for the int32-lane GEMV fast-path: when the inputs have
    // been Frobenius-shimmed (frobenius_scale > 1), .a/.b values can reach
    // O(scale_recip * 41^k) ~ 2^43+ which exceeds int32 range and would be
    // silently truncated by _mm256_mul_epi32. Only engage when both inputs
    // are in the unshimmed regime.
    const bool fast_path_safe =
        (W.frobenius_scale == 1) && (X.frobenius_scale == 1);

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            if (N == 1 && fast_path_safe) {
                // ----- GEMV fast-path (N==1, generation step) -----
                int64_t k = 0;
                int64_t acc_a = 0, acc_b = 0;
#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
                __m512i sum_m1 = _mm512_setzero_si512();
                __m512i sum_m2 = _mm512_setzero_si512();
                for (; k <= K - 4; k += 4) {
                    __m512i w = _mm512_loadu_si512(
                        (const __m512i*)&W.data[i * K + k]);
                    __m512i x = _mm512_loadu_si512(
                        (const __m512i*)&X.data[k]);
                    __m512i x_swp = _mm512_shuffle_epi32(x,
                        (_MM_PERM_ENUM)0x4E);  // swap .a/.b within each pair
                    sum_m1 = _mm512_add_epi64(sum_m1,
                        _mm512_mul_epi32(w, x));
                    sum_m2 = _mm512_add_epi64(sum_m2,
                        _mm512_mul_epi32(w, x_swp));
                }
                // Reduce AVX-512 to AVX2 form for unified horizontal sum.
                __m256i acc_m1 = _mm256_add_epi64(
                    _mm512_castsi512_si256(sum_m1),
                    _mm512_extracti64x4_epi64(sum_m1, 1));
                __m256i acc_m2 = _mm256_add_epi64(
                    _mm512_castsi512_si256(sum_m2),
                    _mm512_extracti64x4_epi64(sum_m2, 1));
#elif defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX2__)
                __m256i acc_m1 = _mm256_setzero_si256();
                __m256i acc_m2 = _mm256_setzero_si256();
                for (; k <= K - 2; k += 2) {
                    __m256i w = _mm256_loadu_si256(
                        (const __m256i*)&W.data[i * K + k]);
                    __m256i x = _mm256_loadu_si256(
                        (const __m256i*)&X.data[k]);
                    __m256i x_swp = _mm256_shuffle_epi32(x, 0x4E);
                    acc_m1 = _mm256_add_epi64(acc_m1,
                        _mm256_mul_epi32(w, x));
                    acc_m2 = _mm256_add_epi64(acc_m2,
                        _mm256_mul_epi32(w, x_swp));
                }
#endif
#if defined(SP_MATMUL_HAVE_SIMD)
                // Horizontal reduce once per output row.
                alignas(32) int64_t arr_m1[4];
                alignas(32) int64_t arr_m2[4];
                _mm256_store_si256((__m256i*)arr_m1, acc_m1);
                _mm256_store_si256((__m256i*)arr_m2, acc_m2);
                const int64_t S_aa = arr_m1[0] + arr_m1[2];  // Σ w.a*x.a
                const int64_t S_bb = arr_m1[1] + arr_m1[3];  // Σ w.b*x.b
                const int64_t S_ab = arr_m2[0] + arr_m2[2];  // Σ w.a*x.b
                const int64_t S_ba = arr_m2[1] + arr_m2[3];  // Σ w.b*x.a
                acc_a = S_aa - (int64_t)SP_OK_OMEGA_NORM * S_bb;
                acc_b = S_ab + S_ba + S_bb;
#endif
                // Scalar tail (and the entire loop when SIMD unavailable).
                for (; k < K; ++k) {
                    const sp_ok_t& w_ik = W.data[i * K + k];
                    const sp_ok_t& x_k0 = X.data[k];
                    acc_a += w_ik.a * x_k0.a
                           - (int64_t)SP_OK_OMEGA_NORM * w_ik.b * x_k0.b;
                    acc_b += w_ik.a * x_k0.b
                           + x_k0.a * w_ik.b
                           + w_ik.b * x_k0.b;
                }
                Y.data[i] = sp_ok_t{ acc_a, acc_b };
            } else {
                // ----- General GEMM path (N > 1, prefill) -----
                // Row-major-by-token: both W row (W.data[i*K..]) and X row
                // (X.data[j*K..]) are contiguous in the inner k loop —
                // friendly to the hardware prefetcher and SIMD.
                for (int64_t j = 0; j < N; ++j) {
                    int64_t acc_a = 0;
                    int64_t acc_b = 0;
                    const sp_ok_t* w_row = W.data + i * K;
                    const sp_ok_t* x_row = X.data + j * K;
                    for (int64_t k = 0; k < K; ++k) {
                        const sp_ok_t& w_ik = w_row[k];
                        const sp_ok_t& x_jk = x_row[k];
                        acc_a += w_ik.a * x_jk.a
                               - (int64_t)SP_OK_OMEGA_NORM * w_ik.b * x_jk.b;
                        acc_b += w_ik.a * x_jk.b
                               + x_jk.a * w_ik.b
                               + w_ik.b * x_jk.b;
                    }
                    Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
                }
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
            const sp_ok_t* w_row = W.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                const sp_ok_t* x_row = X.data + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_t& w_ik = w_row[k];
                    const sp_ok_t& x_jk = x_row[k];
                    acc_a += w_ik.a * x_jk.a - SP_OK_OMEGA_NORM * w_ik.b * x_jk.b;
                }
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
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
            const sp_ok_t* w_row = W.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                const float* x_row = X_fp32 + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_t& w_ik = w_row[k];
                    int64_t x_a =
                        (int64_t)std::llrint((double)x_row[k] * (double)S);
                    acc_a += w_ik.a * x_a;
                    acc_b += w_ik.b * x_a;
                }
                Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip = W.scale_recip * S;
    Y.frobenius_scale = W.frobenius_scale;
    return true;
}

// ---------- Phase 12 Step D: fused packed-Q8 matmul ------------------------

bool sp_matmul_ok_q8(const sp_ok_tensor&    W_shape,
                     const sp_ok_q8_tensor& W_q8,
                     const sp_ok_tensor&    X,
                     sp_ok_tensor&          Y) {
    /* Shape contract pulled from W_shape (data may be null, that's the
     * point of Step B-2's resident packed storage). */
    if (X.data == nullptr || Y.data == nullptr) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    if (W_q8.data == nullptr || W_q8.numel == 0) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if ((size_t)(M * K) != W_q8.numel) return false;

    const int8_t shift = W_q8.q8_shift;
    const int nt = std::max(1, sp_threadpool_n_threads());

    /* Output row i is independent -- partition M across threads. Each
     * worker reads its slice of W_q8 packed bytes (2 B/elem) and decodes
     * lane-by-lane into the sp_ok ring multiply.
     *
     * Cache behaviour: at 2 B/elem, an entire weight ROW for K=6912 fits
     * in 13.5 KB -- well under L1 (typically 48 KB). For Gemma3-1B's
     * ffn_down (M=1152, K=6912) the full matrix is 16 MB packed, fits in
     * L3 on most workstations. The decode is a per-element sign-extend +
     * shift, completed inside the inner k loop with no scratch buffer. */
    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q8_t* w_row = W_q8.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    /* Decode-and-multiply, fused. The shift inlines as
                     * two sll's; the ring multiply matches sp_matmul_ok. */
                    const sp_ok_q8_t& w_q = w_row[k];
                    const int64_t w_a = ((int64_t)w_q.a) << shift;
                    const int64_t w_b = ((int64_t)w_q.b) << shift;
                    const sp_ok_t& x_jk = x_row[k];
                    acc_a += w_a * x_jk.a
                           - (int64_t)SP_OK_OMEGA_NORM * w_b * x_jk.b;
                    acc_b += w_a * x_jk.b
                           + x_jk.a * w_b
                           + w_b * x_jk.b;
                }
                Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip     = W_shape.scale_recip     * X.scale_recip;
    Y.frobenius_scale = W_shape.frobenius_scale * X.frobenius_scale;
    return true;
}

bool sp_matmul_ok_q8_to_fp32(const sp_ok_tensor&    W_shape,
                             const sp_ok_q8_tensor& W_q8,
                             const sp_ok_tensor&    X,
                             float*                 Y_fp32,
                             int                    out_rows,
                             int                    n_cols) {
    if (X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    if (W_q8.data == nullptr || W_q8.numel == 0) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;
    if ((size_t)(M * K) != W_q8.numel) return false;

    /* The decode multiplies by 2^shift; that becomes part of W's effective
     * scale_recip when bridging to fp32. The divisor in the original path
     * is W.scale_recip * X.scale_recip * W.frobenius * X.frobenius. For
     * the q8 path we additionally multiply by 2^shift since each w_a is
     * (int8 << shift). Matches what sp_matmul_ok_q8 produces in the int64
     * Y output before the fp32 conversion. */
    const double divisor =
        (double)W_shape.scale_recip * (double)X.scale_recip *
        (double)W_shape.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int8_t shift = W_q8.q8_shift;
    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q8_t* w_row = W_q8.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                const sp_ok_t* x_row = X.data + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    const sp_ok_q8_t& w_q = w_row[k];
                    const int64_t w_a = ((int64_t)w_q.a) << shift;
                    const int64_t w_b = ((int64_t)w_q.b) << shift;
                    const sp_ok_t& x_jk = x_row[k];
                    acc_a += w_a * x_jk.a
                           - (int64_t)SP_OK_OMEGA_NORM * w_b * x_jk.b;
                }
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
            }
        }
    });
    return true;
}

}  // namespace sp::engine
