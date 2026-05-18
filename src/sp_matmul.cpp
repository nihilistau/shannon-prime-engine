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
//
// Phase 14c (this commit): AVX-512 inner loop for the fused decode-and-multiply.
// Process 8 ring elements per iter; 4 _mm512_mullo_epi64 (full int64×int64)
// across 8 lanes in parallel, vs ~32 scalar int64 mults per 8 elements.
// Hoists the 41 (= SP_OK_OMEGA_NORM) factor out of the hot loop by
// accumulating sum(w_b * x.b) as a separate lane-parallel sum and applying *41
// once at horizontal-reduce time.
//
// We use mullo_epi64 (not mul_epi32) because production Q8 + Frobenius k=8
// hits shifts ~26-29 on Gemma3-1B: (int8 << shift) doesn't fit in int32.
// mullo_epi64 is 3-cycle latency / 1-cycle throughput on Tiger Lake — slower
// per-mul than mul_epi32's 1/1, but still ~4-5x faster than the scalar inner
// loop across the 8-way parallelism.
//
// Overflow profile matches the scalar path: per-product is up to
// (w_a<<shift) * x_a which in practice clusters near 2^39 (W centred near 0
// after shift selection, X bounded by post-RMSNorm ~2^28). Accumulator over
// K=6912 stays under 2^52 — plenty of int64 headroom.

#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
/* Inner-K accumulator for one (i, j) output cell. Processes K elements,
 * delivers (acc_a, acc_b) via out params. Caller writes Y. */
static inline void sp_matmul_ok_q8_inner_avx512(
    const sp_ok_q8_t* w_row,
    const sp_ok_t*    x_row,
    int64_t           K,
    int               shift,
    int64_t&          out_acc_a,
    int64_t&          out_acc_b)
{
    /* Three lane-parallel accumulators (8x int64 each). Algebra:
     *   For each k:
     *     acc_a_part   += w_a(k) * x_a(k)
     *     acc_b_xb_sum += w_b(k) * x_b(k)
     *     acc_b_cross  += w_a(k) * x_b(k) + w_b(k) * x_a(k)
     *   After the K loop:
     *     acc_a = acc_a_part  - SP_OK_OMEGA_NORM * acc_b_xb_sum
     *     acc_b = acc_b_cross + acc_b_xb_sum
     * Hoists the *41 out of the inner loop. */
    __m512i acc_a_part   = _mm512_setzero_si512();
    __m512i acc_b_xb_sum = _mm512_setzero_si512();
    __m512i acc_b_cross  = _mm512_setzero_si512();

    /* Constants for the deinterleave shuffles. Defined once per fn-call. */
    const __m512i idx_a_i32 = _mm512_setr_epi32( 0, 2, 4, 6, 8,10,12,14,
                                                  0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx_b_i32 = _mm512_setr_epi32( 1, 3, 5, 7, 9,11,13,15,
                                                  0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx_xa_i64 = _mm512_setr_epi64(0, 2, 4, 6,  8,10,12,14);
    const __m512i idx_xb_i64 = _mm512_setr_epi64(1, 3, 5, 7,  9,11,13,15);

    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        /* ---- Load 16 packed Q8 bytes = 8 ring elements (a0,b0,...,a7,b7) ---- */
        __m128i wq_packed = _mm_loadu_si128((const __m128i*)(w_row + k));

        /* Sign-extend 16 int8 -> 16 int32 in one __m512i. */
        __m512i wq_32 = _mm512_cvtepi8_epi32(wq_packed);

        /* Deinterleave a-coords and b-coords into the low 256 bits of two
         * temporaries. The high 256 bits are don't-care; we discard via
         * cast-to-256. */
        __m256i wa_32 = _mm512_castsi512_si256(
            _mm512_permutexvar_epi32(idx_a_i32, wq_32));
        __m256i wb_32 = _mm512_castsi512_si256(
            _mm512_permutexvar_epi32(idx_b_i32, wq_32));

        /* Promote int32 -> int64 (8 lanes each), then apply per-tensor shift. */
        __m512i wa_64 = _mm512_slli_epi64(_mm512_cvtepi32_epi64(wa_32), shift);
        __m512i wb_64 = _mm512_slli_epi64(_mm512_cvtepi32_epi64(wb_32), shift);

        /* ---- Load 8 sp_ok_t (16 int64) for X ---- */
        __m512i x_lo = _mm512_loadu_si512((const __m512i*)(x_row + k));     /* x[k..k+3] */
        __m512i x_hi = _mm512_loadu_si512((const __m512i*)(x_row + k + 4)); /* x[k+4..k+7] */

        /* Deinterleave x_a and x_b across the two halves via permutex2var. */
        __m512i xa = _mm512_permutex2var_epi64(x_lo, idx_xa_i64, x_hi);
        __m512i xb = _mm512_permutex2var_epi64(x_lo, idx_xb_i64, x_hi);

        /* ---- Four int64×int64 multiplies (8 lanes parallel) per chunk ----
         * mullo_epi64 = full 64×64→64 low product per lane. Slower than
         * mul_epi32 but works for any shift; required when (int8<<shift)
         * doesn't fit in int32 (i.e. Frobenius k=8 production). */
        __m512i p_wa_xa = _mm512_mullo_epi64(wa_64, xa);
        __m512i p_wb_xb = _mm512_mullo_epi64(wb_64, xb);
        __m512i p_wa_xb = _mm512_mullo_epi64(wa_64, xb);
        __m512i p_wb_xa = _mm512_mullo_epi64(wb_64, xa);

        acc_a_part   = _mm512_add_epi64(acc_a_part,   p_wa_xa);
        acc_b_xb_sum = _mm512_add_epi64(acc_b_xb_sum, p_wb_xb);
        acc_b_cross  = _mm512_add_epi64(acc_b_cross,
                                         _mm512_add_epi64(p_wa_xb, p_wb_xa));
    }

    /* Horizontal-reduce to scalars and combine into the omega form. */
    int64_t s_a_part   = _mm512_reduce_add_epi64(acc_a_part);
    int64_t s_b_xb_sum = _mm512_reduce_add_epi64(acc_b_xb_sum);
    int64_t s_b_cross  = _mm512_reduce_add_epi64(acc_b_cross);

    int64_t acc_a = s_a_part  - (int64_t)SP_OK_OMEGA_NORM * s_b_xb_sum;
    int64_t acc_b = s_b_cross + s_b_xb_sum;

    /* Scalar tail for K not multiple of 8. */
    for (; k < K; ++k) {
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

    out_acc_a = acc_a;
    out_acc_b = acc_b;
}
#endif  /* AVX512F */

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

    /* SIMD gate: AVX-512 inner uses _mm512_mullo_epi64 (full int64×int64),
     * works for any non-negative shift. Same overflow profile as scalar. */
#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
    const bool use_simd = (shift >= 0);
#else
    const bool use_simd = false;
#endif

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
#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
                if (use_simd) {
                    sp_matmul_ok_q8_inner_avx512(
                        w_row, x_row, K, (int)shift, acc_a, acc_b);
                } else
#endif
                {
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

#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
    const bool use_simd = (shift >= 0) && (shift <= 24);
#else
    const bool use_simd = false;
#endif

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q8_t* w_row = W_q8.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                const sp_ok_t* x_row = X.data + j * K;
#if defined(SP_MATMUL_HAVE_SIMD) && defined(__AVX512F__)
                if (use_simd) {
                    /* fp32 variant only needs the a-component; drop the
                     * b-cross accumulators since they'd be discarded. */
                    __m512i acc_a_part   = _mm512_setzero_si512();
                    __m512i acc_b_xb_sum = _mm512_setzero_si512();

                    const __m512i idx_a_i32 = _mm512_setr_epi32(
                         0, 2, 4, 6, 8,10,12,14,  0, 0, 0, 0, 0, 0, 0, 0);
                    const __m512i idx_b_i32 = _mm512_setr_epi32(
                         1, 3, 5, 7, 9,11,13,15,  0, 0, 0, 0, 0, 0, 0, 0);
                    const __m512i idx_xa_i64 = _mm512_setr_epi64(
                         0, 2, 4, 6,  8,10,12,14);
                    const __m512i idx_xb_i64 = _mm512_setr_epi64(
                         1, 3, 5, 7,  9,11,13,15);

                    int64_t k = 0;
                    for (; k + 8 <= K; k += 8) {
                        __m128i wq_packed = _mm_loadu_si128(
                            (const __m128i*)(w_row + k));
                        __m512i wq_32 = _mm512_cvtepi8_epi32(wq_packed);
                        __m256i wa_32 = _mm512_castsi512_si256(
                            _mm512_permutexvar_epi32(idx_a_i32, wq_32));
                        __m256i wb_32 = _mm512_castsi512_si256(
                            _mm512_permutexvar_epi32(idx_b_i32, wq_32));
                        __m512i wa_64 = _mm512_slli_epi64(
                            _mm512_cvtepi32_epi64(wa_32), shift);
                        __m512i wb_64 = _mm512_slli_epi64(
                            _mm512_cvtepi32_epi64(wb_32), shift);
                        __m512i x_lo = _mm512_loadu_si512(
                            (const __m512i*)(x_row + k));
                        __m512i x_hi = _mm512_loadu_si512(
                            (const __m512i*)(x_row + k + 4));
                        __m512i xa = _mm512_permutex2var_epi64(
                            x_lo, idx_xa_i64, x_hi);
                        __m512i xb = _mm512_permutex2var_epi64(
                            x_lo, idx_xb_i64, x_hi);
                        acc_a_part   = _mm512_add_epi64(acc_a_part,
                            _mm512_mullo_epi64(wa_64, xa));
                        acc_b_xb_sum = _mm512_add_epi64(acc_b_xb_sum,
                            _mm512_mullo_epi64(wb_64, xb));
                    }
                    int64_t s_a_part   = _mm512_reduce_add_epi64(acc_a_part);
                    int64_t s_b_xb_sum = _mm512_reduce_add_epi64(acc_b_xb_sum);
                    acc_a = s_a_part
                          - (int64_t)SP_OK_OMEGA_NORM * s_b_xb_sum;
                    /* Scalar tail */
                    for (; k < K; ++k) {
                        const sp_ok_q8_t& w_q = w_row[k];
                        const int64_t w_a = ((int64_t)w_q.a) << shift;
                        const int64_t w_b = ((int64_t)w_q.b) << shift;
                        const sp_ok_t& x_jk = x_row[k];
                        acc_a += w_a * x_jk.a
                               - (int64_t)SP_OK_OMEGA_NORM * w_b * x_jk.b;
                    }
                } else
#endif
                {
                    for (int64_t k = 0; k < K; ++k) {
                        const sp_ok_q8_t& w_q = w_row[k];
                        const int64_t w_a = ((int64_t)w_q.a) << shift;
                        const int64_t w_b = ((int64_t)w_q.b) << shift;
                        const sp_ok_t& x_jk = x_row[k];
                        acc_a += w_a * x_jk.a
                               - (int64_t)SP_OK_OMEGA_NORM * w_b * x_jk.b;
                    }
                }
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
            }
        }
    });
    return true;
}

// ---------- Phase 14: fused packed-Q4 matmul -----------------------------

/* Decode one packed nybble-pair to a pair of int64 coordinates with the
 * per-tensor shift inlined. Mirrors sp_ok_q4_decode_one but expands inline
 * here so the compiler can keep the shifts in registers across the inner
 * loop. The arithmetic-shift idiom sign-extends each 4-bit field into
 * 32 bits without a mask table. */
static inline void sp_q4_decode_pair(uint8_t packed, int8_t shift,
                                      int64_t& w_a, int64_t& w_b) {
    int32_t a4 = ((int32_t)((uint32_t)packed << 28)) >> 28;
    int32_t b4 = ((int32_t)((uint32_t)packed << 24)) >> 28;
    w_a = ((int64_t)a4) << shift;
    w_b = ((int64_t)b4) << shift;
}

bool sp_matmul_ok_q4(const sp_ok_tensor&    W_shape,
                     const sp_ok_q4_tensor& W_q4,
                     const sp_ok_tensor&    X,
                     sp_ok_tensor&          Y) {
    if (X.data == nullptr || Y.data == nullptr) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    if (W_q4.data == nullptr || W_q4.numel == 0) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if ((size_t)(M * K) != W_q4.numel) return false;

    const int8_t shift = W_q4.q4_shift;
    const int nt = std::max(1, sp_threadpool_n_threads());

    /* Output row i is independent -- partition M across threads. At
     * 1 B/elem the entire weight ROW for K=6912 fits in 6.75 KB (well
     * under L1). For Gemma3-1B's ffn_down (M=1152, K=6912) the full
     * matrix is 8 MB packed, comfortably in L2 on most workstations. */
    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_t* w_row = W_q4.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t w_a, w_b;
                    sp_q4_decode_pair(w_row[k].packed, shift, w_a, w_b);
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

bool sp_matmul_ok_q4_to_fp32(const sp_ok_tensor&    W_shape,
                             const sp_ok_q4_tensor& W_q4,
                             const sp_ok_tensor&    X,
                             float*                 Y_fp32,
                             int                    out_rows,
                             int                    n_cols) {
    if (X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    if (W_q4.data == nullptr || W_q4.numel == 0) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;
    if ((size_t)(M * K) != W_q4.numel) return false;

    const double divisor =
        (double)W_shape.scale_recip * (double)X.scale_recip *
        (double)W_shape.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int8_t shift = W_q4.q4_shift;
    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_t* w_row = W_q4.data + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                const sp_ok_t* x_row = X.data + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t w_a, w_b;
                    sp_q4_decode_pair(w_row[k].packed, shift, w_a, w_b);
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

// ---------- Phase 15: block-scale fused matmul (Q8_0 / Q4_0 ingest) -----

/* Helper: scalar inner loop for one (i, j) output cell against a slice
 * of W blocks. Used in the OK->OK path and as the scalar fallback when
 * AVX-512 is unavailable.
 *
 * For each block of 32 elements in K:
 *   Load B_a, B_b from the block.
 *   For each k in [0..32):
 *     int8 w = block.packed[k]                          (Q8 path)
 *           or sp_ok_block_q4_decode_codepoint(packed, k) (Q4 path, +8-biased)
 *     F_a = B_a * x.a[k] − 41 * B_b * x.b[k]
 *     F_b = B_a * x.b[k] + B_b * x.a[k] + B_b * x.b[k]
 *     acc_a += w * F_a
 *     acc_b += w * F_b
 */

template <bool IS_Q4, bool A_ONLY>
static inline void sp_matmul_ok_block_inner_scalar(
    const void*    w_blocks,         // sp_ok_q8_block_t* or sp_ok_q4_block_t*
    const sp_ok_t* x_row,
    size_t         n_blocks,
    int64_t&       out_acc_a,
    int64_t&       out_acc_b)
{
    constexpr int64_t W41 = (int64_t)SP_OK_OMEGA_NORM;
    int64_t acc_a = 0;
    int64_t acc_b = 0;

    for (size_t b = 0; b < n_blocks; ++b) {
        int64_t B_a, B_b;
        const uint8_t* packed;
        const int8_t*  packed_i8 = nullptr;

        if constexpr (IS_Q4) {
            const sp_ok_q4_block_t* blk =
                (const sp_ok_q4_block_t*)w_blocks + b;
            B_a    = blk->B_a;
            B_b    = blk->B_b;
            packed = blk->packed;
        } else {
            const sp_ok_q8_block_t* blk =
                (const sp_ok_q8_block_t*)w_blocks + b;
            B_a       = blk->B_a;
            B_b       = blk->B_b;
            packed_i8 = blk->packed;
        }

        const size_t k_base = b * (size_t)SP_OK_BLOCK_SIZE;
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            int64_t w_int;
            if constexpr (IS_Q4) {
                w_int = (int64_t)sp_ok_block_q4_decode_codepoint(packed, k);
            } else {
                w_int = (int64_t)packed_i8[k];
            }
            const sp_ok_t& x_jk = x_row[k_base + k];
            const int64_t F_a = B_a * x_jk.a - W41 * B_b * x_jk.b;
            acc_a += w_int * F_a;
            if constexpr (!A_ONLY) {
                const int64_t F_b = B_a * x_jk.b
                                   + B_b * x_jk.a
                                   + B_b * x_jk.b;
                acc_b += w_int * F_b;
            }
        }
    }

    out_acc_a = acc_a;
    if constexpr (!A_ONLY) out_acc_b = acc_b;
}

bool sp_matmul_ok_block_q8(const sp_ok_tensor&          W_shape,
                            const sp_ok_block_q8_tensor& W_blk,
                            const sp_ok_tensor&          X,
                            sp_ok_tensor&                Y) {
    if (X.data == nullptr || Y.data == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;
    if (W_blk.n_blocks != (size_t)M * blocks_per_row) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q8_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0;
                int64_t acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_inner_scalar<false, false>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip     = W_shape.scale_recip     * X.scale_recip;
    Y.frobenius_scale = W_shape.frobenius_scale * X.frobenius_scale;
    return true;
}

bool sp_matmul_ok_block_q8_to_fp32(const sp_ok_tensor&          W_shape,
                                    const sp_ok_block_q8_tensor& W_blk,
                                    const sp_ok_tensor&          X,
                                    float*                       Y_fp32,
                                    int                          out_rows,
                                    int                          n_cols) {
    if (X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;

    const double divisor =
        (double)W_shape.scale_recip * (double)X.scale_recip *
        (double)W_shape.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q8_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0, acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_inner_scalar<false, true>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
            }
        }
    });
    return true;
}

bool sp_matmul_ok_block_q4(const sp_ok_tensor&          W_shape,
                            const sp_ok_block_q4_tensor& W_blk,
                            const sp_ok_tensor&          X,
                            sp_ok_tensor&                Y) {
    if (X.data == nullptr || Y.data == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;
    if (W_blk.n_blocks != (size_t)M * blocks_per_row) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0, acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_inner_scalar<true, false>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip     = W_shape.scale_recip     * X.scale_recip;
    Y.frobenius_scale = W_shape.frobenius_scale * X.frobenius_scale;
    return true;
}

bool sp_matmul_ok_block_q4_to_fp32(const sp_ok_tensor&          W_shape,
                                    const sp_ok_block_q4_tensor& W_blk,
                                    const sp_ok_tensor&          X,
                                    float*                       Y_fp32,
                                    int                          out_rows,
                                    int                          n_cols) {
    if (X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;

    const double divisor =
        (double)W_shape.scale_recip * (double)X.scale_recip *
        (double)W_shape.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0, acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_inner_scalar<true, true>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
            }
        }
    });
    return true;
}

// ---------- Phase 15b: Q4_1 matmul (asymmetric block_min) ----------------
//
// Math: W[k] = d·x_int[k] + m, where x_int[k] is UNSIGNED [0, 15] and
// (d, m) is fused into per-block (B, M) integer pairs at load time.
// The ring-multiplied accumulator becomes
//
//   acc_a = Σ_k (x_int[k]·B_a + M_a)·x.a[k] − 41·(x_int[k]·B_b + M_b)·x.b[k]
//         = Σ_k x_int[k] · F_a[k]  +  Σ_k (M_a · x.a[k] − 41·M_b · x.b[k])
//         = Σ_k x_int[k] · F_a[k]  +  M_a · Sx_a − 41·M_b · Sx_b
//
// where Sx_a = Σ_k x.a[k] over the 32 elements of the block, Sx_b
// similar. Sx_a / Sx_b depend only on X, so the M-contribution is two
// mults per block per token — negligible vs the K-loop cost.

template <bool A_ONLY>
static inline void sp_matmul_ok_block_q4_1_inner(
    const sp_ok_q4_1_block_t* w_blocks,
    const sp_ok_t*            x_row,
    size_t                    n_blocks,
    int64_t&                  out_acc_a,
    int64_t&                  out_acc_b)
{
    constexpr int64_t W41 = (int64_t)SP_OK_OMEGA_NORM;
    int64_t acc_a = 0;
    int64_t acc_b = 0;

    for (size_t b = 0; b < n_blocks; ++b) {
        const sp_ok_q4_1_block_t& blk = w_blocks[b];
        const size_t k_base = b * (size_t)SP_OK_BLOCK_SIZE;

        /* Sx_a / Sx_b over the 32 elements of this block (depend on X only). */
        int64_t Sx_a = 0;
        int64_t Sx_b = 0;
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            Sx_a += x_row[k_base + k].a;
            Sx_b += x_row[k_base + k].b;
        }

        /* Σ_k x_int[k] · F_a / F_b */
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            const int64_t w_int = (int64_t)
                sp_ok_block_q4_1_decode_codepoint(blk.packed, k);
            const sp_ok_t& x_jk = x_row[k_base + k];
            const int64_t F_a = blk.B_a * x_jk.a - W41 * blk.B_b * x_jk.b;
            acc_a += w_int * F_a;
            if constexpr (!A_ONLY) {
                const int64_t F_b = blk.B_a * x_jk.b
                                   + blk.B_b * x_jk.a
                                   + blk.B_b * x_jk.b;
                acc_b += w_int * F_b;
            }
        }

        /* M-contribution: (M_a, M_b) · (Sx_a, Sx_b) once per block. */
        acc_a += blk.M_a * Sx_a - W41 * blk.M_b * Sx_b;
        if constexpr (!A_ONLY) {
            acc_b += blk.M_a * Sx_b + blk.M_b * Sx_a + blk.M_b * Sx_b;
        }
    }

    out_acc_a = acc_a;
    if constexpr (!A_ONLY) out_acc_b = acc_b;
}

bool sp_matmul_ok_block_q4_1(const sp_ok_tensor&            W_shape,
                              const sp_ok_block_q4_1_tensor& W_blk,
                              const sp_ok_tensor&            X,
                              sp_ok_tensor&                  Y) {
    if (X.data == nullptr || Y.data == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2) return false;
    if (Y.n_dims < 2 || Y.shape[0] != N || Y.shape[1] != M) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_1_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0, acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_q4_1_inner<false>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y.data[j * M + i] = sp_ok_t{ acc_a, acc_b };
            }
        }
    });

    Y.scale_recip     = W_shape.scale_recip     * X.scale_recip;
    Y.frobenius_scale = W_shape.frobenius_scale * X.frobenius_scale;
    return true;
}

bool sp_matmul_ok_block_q4_1_to_fp32(const sp_ok_tensor&            W_shape,
                                      const sp_ok_block_q4_1_tensor& W_blk,
                                      const sp_ok_tensor&            X,
                                      float*                         Y_fp32,
                                      int                            out_rows,
                                      int                            n_cols) {
    if (X.data == nullptr || Y_fp32 == nullptr) return false;
    if (W_blk.blocks == nullptr || W_blk.n_blocks == 0) return false;
    if (W_shape.n_dims < 2 || X.n_dims < 2) return false;
    const int64_t M  = W_shape.shape[1];
    const int64_t K  = W_shape.shape[0];
    const int64_t K2 = X.shape[1];
    const int64_t N  = X.shape[0];
    if (K != K2 || (int64_t)out_rows != M || (int64_t)n_cols != N) return false;
    if ((size_t)(M * K) != W_blk.numel) return false;
    if ((K % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t blocks_per_row = (size_t)K / SP_OK_BLOCK_SIZE;

    const double divisor =
        (double)W_shape.scale_recip * (double)X.scale_recip *
        (double)W_shape.frobenius_scale * (double)X.frobenius_scale;
    if (divisor == 0.0) return false;

    const int nt = std::max(1, sp_threadpool_n_threads());

    sp_parallel_for([&](int thread_id) {
        int64_t i0, i1;
        split_rows(M, nt, thread_id, i0, i1);
        for (int64_t i = i0; i < i1; ++i) {
            const sp_ok_q4_1_block_t* w_row =
                W_blk.blocks + (size_t)i * blocks_per_row;
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc_a = 0, acc_b = 0;
                const sp_ok_t* x_row = X.data + j * K;
                sp_matmul_ok_block_q4_1_inner<true>(
                    w_row, x_row, blocks_per_row, acc_a, acc_b);
                Y_fp32[j * M + i] = (float)((double)acc_a / divisor);
            }
        }
    });
    return true;
}

}  // namespace sp::engine
