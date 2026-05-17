// Shannon-Prime Engine — Phase 2.1 bridges (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_bridges.h"
#include "sp_ok_encode.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace sp::engine {

// =========================================================================
// sp_rmsnorm_native
//
// Decode O_K input to fp32 (one pass), compute RMS, normalize, multiply by
// per-feature scale, re-encode at the output's scale_recip with
// frobenius_scale RESET to 1 (the scale-reset valve).
// =========================================================================

bool sp_rmsnorm_native(const sp_ok_tensor& x,
                        const float*        scale_fp32,
                        sp_ok_tensor&       out,
                        float               eps,
                        int                 n_embd,
                        int                 n_tokens) {
    if (x.data == nullptr || scale_fp32 == nullptr || out.data == nullptr) return false;
    if (n_embd <= 0 || n_tokens <= 0) return false;
    const int64_t total = (int64_t)n_embd * (int64_t)n_tokens;
    if (x.numel() != total || out.numel() != total) return false;

    const double in_divisor = (double)x.scale_recip * (double)x.frobenius_scale;
    if (in_divisor == 0.0) return false;

    // Choose output scale: caller may have set it; if zero, default to a
    // safe value matching the input scale.
    if (out.scale_recip <= 0) out.scale_recip = x.scale_recip;
    out.frobenius_scale = 1;  // SCALE RESET VALVE — see Phase 1.7 bypass policy

    const int64_t S_out = out.scale_recip;
    const double out_scale = (double)S_out;

    // Per-token row of length n_embd.
    for (int t = 0; t < n_tokens; ++t) {
        const sp_ok_t* row_in  = x.data   + (int64_t)t * n_embd;
        sp_ok_t*       row_out = out.data + (int64_t)t * n_embd;

        // Pass 1: decode and compute sum-of-squares for rms.
        double sum_sq = 0.0;
        // To avoid an explicit fp32 buffer, we can decode twice (once for
        // rms, once for the output) — or pass once and store fp32 to the
        // output's a-component temporarily. For clarity and minimal alloc,
        // we decode in-line: read from O_K, compute fp32 value, square,
        // then in a second loop normalize and re-encode.
        for (int i = 0; i < n_embd; ++i) {
            double v = (double)row_in[i].a / in_divisor;
            sum_sq += v * v;
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / (double)n_embd + (double)eps);

        // Pass 2: normalize, multiply by per-feature scale, re-encode.
        for (int i = 0; i < n_embd; ++i) {
            double v        = (double)row_in[i].a / in_divisor;
            double v_norm   = v * inv_rms * (double)scale_fp32[i];
            int64_t a_out   = (int64_t)std::llrint(v_norm * out_scale);
            row_out[i] = sp_ok_t{ a_out, 0 };
        }
    }
    return true;
}

// =========================================================================
// sp_softmax_bridge
// =========================================================================

void sp_softmax_bridge(const float* in, int n, float* out) {
    if (n <= 0) return;
    float mx = in[0];
    for (int i = 1; i < n; ++i) if (in[i] > mx) mx = in[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float e = std::exp(in[i] - mx);
        out[i] = e;
        sum += (double)e;
    }
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

void sp_softmax_bridge_rows(const float* in, int n_cols, int n_rows, float* out) {
    for (int r = 0; r < n_rows; ++r) {
        sp_softmax_bridge(in + (size_t)r * n_cols, n_cols, out + (size_t)r * n_cols);
    }
}

void sp_softmax_bridge_causal(const float* in, int n, int valid_len, float* out) {
    if (n <= 0) return;
    if (valid_len <= 0) {
        for (int i = 0; i < n; ++i) out[i] = 0.0f;
        return;
    }
    if (valid_len > n) valid_len = n;
    // Compute max only over valid positions.
    float mx = in[0];
    for (int i = 1; i < valid_len; ++i) if (in[i] > mx) mx = in[i];
    double sum = 0.0;
    for (int i = 0; i < valid_len; ++i) {
        float e = std::exp(in[i] - mx);
        out[i] = e;
        sum += (double)e;
    }
    for (int i = valid_len; i < n; ++i) out[i] = 0.0f;
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < valid_len; ++i) out[i] *= inv;
}

// =========================================================================
// sp_silu_bridge
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// =========================================================================

static inline float silu_scalar(float x) {
    // Two equivalent forms; the second is slightly more stable for large
    // positive x (avoids exp(positive) overflow).
    if (x >= 0.0f) {
        return x / (1.0f + std::exp(-x));
    } else {
        float e = std::exp(x);
        return x * e / (1.0f + e);
    }
}

void sp_silu_bridge(const float* gate, const float* up, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

void sp_silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = silu_scalar(x[i]);
}

// =========================================================================
// GELU (tanh approximation) — matches ggml_gelu / gelu_pytorch_tanh / gemma.
//
//   c0 = sqrt(2/pi) ≈ 0.7978845608028654
//   c1 = 0.044715
//   gelu(x) = 0.5 * x * (1 + tanh(c0 * (x + c1 * x^3)))
// =========================================================================

static inline float gelu_tanh_scalar(float x) {
    constexpr float C0 = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float C1 = 0.044715f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(C0 * (x + C1 * x3)));
}

void sp_gelu_tanh_bridge(const float* gate, const float* up, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = gelu_tanh_scalar(gate[i]) * up[i];
    }
}

void sp_gelu_tanh_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = gelu_tanh_scalar(x[i]);
}

// =========================================================================
// Phase 2.3b helpers
// =========================================================================

bool sp_per_head_rmsnorm_native(sp_ok_tensor& qk,
                                  const float* scale_fp32,
                                  float eps,
                                  int n_heads, int head_dim, int n_tokens) {
    if (qk.data == nullptr || scale_fp32 == nullptr) return false;
    if (n_heads <= 0 || head_dim <= 0 || n_tokens <= 0) return false;
    if (qk.n_dims < 2) return false;
    const int64_t T = qk.shape[0];
    const int64_t F = qk.shape[1];
    if (T != (int64_t)n_tokens) return false;
    if (F != (int64_t)n_heads * head_dim) return false;

    const double in_divisor = (double)qk.scale_recip * (double)qk.frobenius_scale;
    if (in_divisor == 0.0) return false;
    const int64_t S = qk.scale_recip;
    const double  S_d = (double)S;

    // Layout (Step E row-major-by-token): qk.data[t * F + (h*head_dim + d)]
    // At n_tokens=1 this collapses to qk.data[h*head_dim + d] — bit-identical
    // to the pre-Step-E column-major-by-token formula qk.data[f * T + t].
    const int64_t d_total = F;  // n_heads * head_dim
    for (int t = 0; t < n_tokens; ++t) {
        sp_ok_t* row = qk.data + (int64_t)t * d_total;
        for (int h = 0; h < n_heads; ++h) {
            sp_ok_t* head_ptr = row + (int64_t)h * head_dim;
            // Pass 1: sum of squares over head_dim for (h, t).
            double sum_sq = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                double v = (double)head_ptr[d].a / in_divisor;
                sum_sq += v * v;
            }
            const double inv_rms =
                1.0 / std::sqrt(sum_sq / (double)head_dim + (double)eps);

            // Pass 2: normalize, multiply by per-feature scale, re-encode.
            for (int d = 0; d < head_dim; ++d) {
                double v   = (double)head_ptr[d].a / in_divisor;
                double y   = v * inv_rms * (double)scale_fp32[d];
                int64_t a  = (int64_t)std::llrint(y * S_d);
                head_ptr[d] = sp_ok_t{ a, 0 };
            }
        }
    }
    qk.frobenius_scale = 1;
    return true;
}

void sp_rmsnorm_fp32(const float* x, const float* scale_fp32, float* out,
                       int n_embd, int n_tokens, float eps) {
    for (int t = 0; t < n_tokens; ++t) {
        const float* row_in  = x   + (size_t)t * n_embd;
        float*       row_out = out + (size_t)t * n_embd;
        double ss = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            ss += (double)row_in[i] * (double)row_in[i];
        }
        const float inv_rms =
            1.0f / std::sqrt((float)(ss / (double)n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) {
            row_out[i] = row_in[i] * inv_rms * scale_fp32[i];
        }
    }
}

}  // namespace sp::engine
