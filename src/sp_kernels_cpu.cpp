// sp_kernels_cpu — CPU compute primitives for forward_native.cpp.
// See sp_kernels_cpu.h.

#include "sp_kernels_cpu.h"
#include "sp_quant.h"
#include "sp_threadpool.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
  #include <arm_neon.h>
  #define SP_HAS_NEON 1
#else
  #define SP_HAS_NEON 0
#endif

namespace sp::engine {

// ──────────────────────────────────────────────────────────────────
// RMS Norm
// ──────────────────────────────────────────────────────────────────

void sp_rms_norm_f32(const float* x, const float* scale,
                     int n, float eps, float* out) {
    // sum of squares — accumulate in fp64 to match ggml's behavior
    // for stability on long vectors.
    double ssq = 0.0;
#if SP_HAS_NEON
    // 4-way SIMD reduction. Tail loop handles any non-multiple-of-4.
    int i = 0;
    float32x4_t v_acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v_acc = vfmaq_f32(v_acc, v, v);
    }
    float partial[4];
    vst1q_f32(partial, v_acc);
    ssq = (double)(partial[0] + partial[1] + partial[2] + partial[3]);
    for (; i < n; ++i) ssq += (double)x[i] * (double)x[i];
#else
    for (int i = 0; i < n; ++i) ssq += (double)x[i] * (double)x[i];
#endif
    const float mean    = (float)(ssq / (double)n);
    const float inv_rms = 1.0f / std::sqrt(mean + eps);

#if SP_HAS_NEON
    int j = 0;
    const float32x4_t v_inv = vdupq_n_f32(inv_rms);
    for (; j + 4 <= n; j += 4) {
        float32x4_t v_x = vld1q_f32(x + j);
        float32x4_t v_s = vld1q_f32(scale + j);
        vst1q_f32(out + j, vmulq_f32(vmulq_f32(v_x, v_inv), v_s));
    }
    for (; j < n; ++j) out[j] = (x[j] * inv_rms) * scale[j];
#else
    for (int i = 0; i < n; ++i) out[i] = (x[i] * inv_rms) * scale[i];
#endif
}

void sp_rms_norm_f32_rows(const float* x, const float* scale,
                          int n_cols, int n_rows, float eps,
                          float* out) {
    for (int r = 0; r < n_rows; ++r) {
        sp_rms_norm_f32(x + (size_t)r * n_cols, scale,
                         n_cols, eps,
                         out + (size_t)r * n_cols);
    }
}

// ──────────────────────────────────────────────────────────────────
// Bias add (broadcast)
// ──────────────────────────────────────────────────────────────────

void sp_bias_add_f32_rows(float* out, const float* bias,
                           int n_cols, int n_rows) {
    if (!bias) return;
    for (int r = 0; r < n_rows; ++r) {
        float* row = out + (size_t)r * n_cols;
#if SP_HAS_NEON
        int c = 0;
        for (; c + 4 <= n_cols; c += 4) {
            float32x4_t v_o = vld1q_f32(row + c);
            float32x4_t v_b = vld1q_f32(bias + c);
            vst1q_f32(row + c, vaddq_f32(v_o, v_b));
        }
        for (; c < n_cols; ++c) row[c] += bias[c];
#else
        for (int c = 0; c < n_cols; ++c) row[c] += bias[c];
#endif
    }
}

// ──────────────────────────────────────────────────────────────────
// SiLU
// ──────────────────────────────────────────────────────────────────
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Use std::expf — bit-exact match to ggml on most platforms (both
// route through libc's expf). NEON doesn't have a direct exp; we
// fall through to scalar for the exp call but vectorize the surround.

static inline float sp_silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

void sp_silu_f32(const float* x, int n, float* out) {
    for (int i = 0; i < n; ++i) out[i] = sp_silu_scalar(x[i]);
}

void sp_silu_mul_f32(const float* gate, const float* up, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = sp_silu_scalar(gate[i]) * up[i];
    }
}

// ──────────────────────────────────────────────────────────────────
// Softmax
// ──────────────────────────────────────────────────────────────────

void sp_softmax_f32_rows(const float* x, const float* mask,
                         int n_cols, int n_rows, float scale,
                         float* out) {
    for (int r = 0; r < n_rows; ++r) {
        const float* xr   = x   + (size_t)r * n_cols;
        const float* mr   = mask ? (mask + (size_t)r * n_cols) : nullptr;
        float*       outr = out + (size_t)r * n_cols;

        // Pass 1: find max for numerical stability.
        float mx = -INFINITY;
        for (int j = 0; j < n_cols; ++j) {
            float v = xr[j] * scale + (mr ? mr[j] : 0.0f);
            if (v > mx) mx = v;
        }

        // Pass 2: exp and sum.
        double sum = 0.0;
        for (int j = 0; j < n_cols; ++j) {
            float v = xr[j] * scale + (mr ? mr[j] : 0.0f);
            float e = std::exp(v - mx);
            outr[j] = e;
            sum += (double)e;
        }

        // Pass 3: normalize.
        const float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
        for (int j = 0; j < n_cols; ++j) outr[j] *= inv;
    }
}

// ──────────────────────────────────────────────────────────────────
// RoPE (interleaved-pair, scalar; mRoPE-section variant TBD)
// ──────────────────────────────────────────────────────────────────

void sp_rope_f32(float* x, int head_dim, int n_heads, int n_pos,
                 const int32_t* pos,
                 int n_rot, float freq_base, float freq_scale) {
    // Half-pairs over n_rot dims: index k in [0, n_rot/2).
    // freq[k] = freq_scale / pow(freq_base, 2k / n_rot)
    const int n_pairs = n_rot / 2;
    for (int p = 0; p < n_pos; ++p) {
        const float pp = (float)pos[p];
        for (int h = 0; h < n_heads; ++h) {
            float* xv = x
                + (size_t)p * n_heads * head_dim
                + (size_t)h * head_dim;
            for (int k = 0; k < n_pairs; ++k) {
                const float exp_arg = -(float)(2 * k) / (float)n_rot;
                const float freq    = freq_scale * std::pow(freq_base, exp_arg);
                const float ang     = pp * freq;
                const float c       = std::cos(ang);
                const float s       = std::sin(ang);
                const float a       = xv[2 * k];
                const float b       = xv[2 * k + 1];
                xv[2 * k]     = c * a - s * b;
                xv[2 * k + 1] = s * a + c * b;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Matmul (fp32 × fp32)
// ──────────────────────────────────────────────────────────────────
//
// Convention matches ggml_mul_mat:
//   y = W @ x
//   W is [k, n] (k input features, n output features), stored row-major
//   so W[i, j] is at index i*n + j. Per ggml convention this is laid
//   out as [n_out=n, n_in=k] from a model perspective, but in memory
//   the inner stride is over n_in.
//
// Our signature:
//   lhs[m, k]  — m rows of activations (m = batch * seq * heads)
//   rhs[n, k]  — weight matrix, n output rows, each k wide
//   out[m, n]
//
// out[i, j] = dot(lhs[i, :], rhs[j, :])

// Single-row matmul slice (used as the inner kernel of the
// threaded path and the single-thread fallback).
static inline void matmul_f32_row(const float* lh, const float* rhs,
                                   int k, int n, float* op) {
    for (int j = 0; j < n; ++j) {
        const float* rh = rhs + (size_t)j * k;
        double acc = 0.0;
#if SP_HAS_NEON
        int p = 0;
        float32x4_t v_acc = vdupq_n_f32(0.0f);
        for (; p + 4 <= k; p += 4) {
            float32x4_t v_l = vld1q_f32(lh + p);
            float32x4_t v_r = vld1q_f32(rh + p);
            v_acc = vfmaq_f32(v_acc, v_l, v_r);
        }
        float partial[4];
        vst1q_f32(partial, v_acc);
        acc = (double)(partial[0] + partial[1] + partial[2] + partial[3]);
        for (; p < k; ++p) acc += (double)lh[p] * (double)rh[p];
#else
        for (int p = 0; p < k; ++p) acc += (double)lh[p] * (double)rh[p];
#endif
        op[j] = (float)acc;
    }
}

void sp_matmul_f32(const float* lhs, const float* rhs,
                   int m, int k, int n, float* out) {
    const int nt = sp_threadpool_n_threads();
    if (nt <= 1 || m < nt) {
        // Single-threaded: skip the parallel_for plumbing entirely.
        for (int i = 0; i < m; ++i) {
            matmul_f32_row(lhs + (size_t)i * k, rhs, k, n,
                           out + (size_t)i * n);
        }
        return;
    }
    // Partition output ROWS [0, m) across threads. Each thread writes
    // a contiguous slice — no false sharing on output cache lines as
    // long as `n * sizeof(float)` is comfortably > 64B (typically true).
    sp_parallel_for([&](int tid) {
        const int rows_per = (m + nt - 1) / nt;
        const int i_start = tid * rows_per;
        const int i_end   = (i_start + rows_per > m) ? m : i_start + rows_per;
        for (int i = i_start; i < i_end; ++i) {
            matmul_f32_row(lhs + (size_t)i * k, rhs, k, n,
                           out + (size_t)i * n);
        }
    });
}

// ──────────────────────────────────────────────────────────────────
// Matmul with Q5_K-packed RHS (fused dequant)
// ──────────────────────────────────────────────────────────────────
//
// Memory pattern: dequant one weight ROW (k floats = k/256 Q5_K
// blocks) into a scratch buffer, then dot against all m activation
// rows. That gives us O(m) reuse of every dequant — much better than
// dequanting per-(m,j) pair.

// Per-output-column slice: dequant row j of W, dot against all m
// LHS rows. Used as the inner kernel for both threaded and
// single-threaded paths.
static inline void matmul_q5k_col(const float* lhs,
                                   const sp_block_q5_K* row_blocks,
                                   int m, int k, int n, int j,
                                   int blocks_per_row,
                                   float* row_scratch,
                                   float* out) {
    sp_dequant_q5_K_to_f32(row_blocks, row_scratch, (size_t)blocks_per_row);
    for (int i = 0; i < m; ++i) {
        const float* lh = lhs + (size_t)i * k;
        double acc = 0.0;
#if SP_HAS_NEON
        int p = 0;
        float32x4_t v_acc = vdupq_n_f32(0.0f);
        for (; p + 4 <= k; p += 4) {
            float32x4_t v_l = vld1q_f32(lh + p);
            float32x4_t v_r = vld1q_f32(row_scratch + p);
            v_acc = vfmaq_f32(v_acc, v_l, v_r);
        }
        float partial[4];
        vst1q_f32(partial, v_acc);
        acc = (double)(partial[0] + partial[1] + partial[2] + partial[3]);
        for (; p < k; ++p) acc += (double)lh[p] * (double)row_scratch[p];
#else
        for (int p = 0; p < k; ++p) acc += (double)lh[p] * (double)row_scratch[p];
#endif
        out[(size_t)i * n + j] = (float)acc;
    }
}

void sp_matmul_f32_q5k(const float* lhs, const void* rhs_q5k,
                       int m, int k, int n, float* out) {
    if (k % SP_QK_K != 0) {
        std::fprintf(stderr,
            "[sp_matmul_q5k] k=%d not multiple of QK_K=%d\n", k, SP_QK_K);
        return;
    }
    const int blocks_per_row = k / SP_QK_K;
    const auto* blocks_base  = (const sp_block_q5_K*)rhs_q5k;
    const int nt = sp_threadpool_n_threads();

    if (nt <= 1 || n < nt) {
        // Single-threaded: one shared scratch row.
        std::vector<float> row_scratch((size_t)k);
        for (int j = 0; j < n; ++j) {
            matmul_q5k_col(lhs,
                           blocks_base + (size_t)j * blocks_per_row,
                           m, k, n, j, blocks_per_row,
                           row_scratch.data(), out);
        }
        return;
    }
    // Partition output COLUMNS [0, n) across threads. Each thread
    // owns its own dequant scratch row (one fp32 row per thread —
    // ~16 KB at k=4096 × 4 threads = 64 KB total, well under L1
    // pressure even per-core).
    sp_parallel_for([&](int tid) {
        std::vector<float> row_scratch((size_t)k);
        const int cols_per = (n + nt - 1) / nt;
        const int j_start = tid * cols_per;
        const int j_end   = (j_start + cols_per > n) ? n : j_start + cols_per;
        for (int j = j_start; j < j_end; ++j) {
            matmul_q5k_col(lhs,
                           blocks_base + (size_t)j * blocks_per_row,
                           m, k, n, j, blocks_per_row,
                           row_scratch.data(), out);
        }
    });
}

}  // namespace sp::engine
