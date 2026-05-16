// Shannon-Prime Engine — fp16 <-> O_K encoding + Frobenius shim (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_ok_encode.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_frobenius.h"
}

#include <cmath>
#include <cstring>

namespace sp::engine {

// =========================================================================
// IEEE half <-> float helpers (header-only inline equivalents are in
// sp_quant; we reproduce the minimal version here to keep this TU
// self-contained).
// =========================================================================

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

static inline uint16_t fp32_to_fp16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;            // flush to zero
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00); // inf
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}

// =========================================================================
// Encode
// =========================================================================

bool sp_ok_encode_from_fp32(sp_ok_tensor& out, const float* w,
                             int n_dims, const int64_t shape[4],
                             int64_t scale, sp_ok_arena& arena) {
    out.reset(n_dims, shape);
    out.scale_recip = scale;
    out.frobenius_scale = 1;
    if (!arena.alloc_tensor(out)) return false;
    const int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) {
        double v = (double)w[i] * (double)scale;
        // Round half-to-even via std::lrint
        int64_t a = (int64_t)std::llrint(v);
        out.data[i] = sp_ok_t{ a, 0 };
    }
    return true;
}

bool sp_ok_encode_from_fp16(sp_ok_tensor& out, const uint16_t* w_fp16,
                             int n_dims, const int64_t shape[4],
                             int64_t scale, sp_ok_arena& arena) {
    out.reset(n_dims, shape);
    out.scale_recip = scale;
    out.frobenius_scale = 1;
    if (!arena.alloc_tensor(out)) return false;
    const int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) {
        float v = fp16_to_fp32(w_fp16[i]) * (float)scale;
        int64_t a = (int64_t)std::llrint(v);
        out.data[i] = sp_ok_t{ a, 0 };
    }
    return true;
}

// =========================================================================
// Decode
// =========================================================================

void sp_ok_decode_to_fp32(float* out, const sp_ok_tensor& t) {
    const int64_t n = t.numel();
    const double divisor = (double)t.scale_recip * (double)t.frobenius_scale;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (float)((double)t.data[i].a / divisor);
    }
}

void sp_ok_decode_to_fp16(uint16_t* out_fp16, const sp_ok_tensor& t) {
    const int64_t n = t.numel();
    const double divisor = (double)t.scale_recip * (double)t.frobenius_scale;
    for (int64_t i = 0; i < n; ++i) {
        float v = (float)((double)t.data[i].a / divisor);
        out_fp16[i] = fp32_to_fp16(v);
    }
}

// =========================================================================
// Frobenius shim
// =========================================================================

void sp_ok_encode_apply_frobenius_quant(sp_ok_tensor& t,
                                          int64_t p, int64_t k) {
    if (k == 0) return;
    const int64_t n = t.numel();
    // Use the C-side tensor apply, which handles inert/split internally.
    sp_frobenius_quant_tensor(t.data, (size_t)n, p, k);

    // Update frobenius_scale so decode produces the original fp32.
    //
    // For inert p: phi_p^(2m) acts as scalar (-p)^m on (a,b). The
    // effective scale on the a-component is |(-p)^m| = p^m. So
    // frobenius_scale *= p^m.
    //
    // For split p: phi_p^k = pi^k where N(pi) = p. The "scale" in the
    // a-direction depends on Re(pi^k). For our purposes, we track the
    // NORM contribution (multiplicative): N(pi^k) = p^k. The decode
    // recovers w only up to a sign / rotation factor; for the shim
    // we use sqrt(N(pi^k)) = p^(k/2) as the scale.
    //
    // NOTE: this is a single-channel-scalar approximation. The full
    // Theorem 4 cancellation requires that the same scale factor be
    // applied to QK^T attention scores (where it cancels exactly).
    // For now the decode just produces a uniformly-scaled fp32; the
    // forward pass must handle the residual scale.
    if (sp_is_inert(p)) {
        int64_t m = k / 2;
        int64_t scale_growth = 1;
        for (int64_t i = 0; i < m; ++i) scale_growth *= p;
        t.frobenius_scale *= scale_growth;
    } else if (sp_is_split(p)) {
        // sqrt of N(pi^k) = p^(k/2). For odd k this is irrational, so
        // we approximate by p^k and absorb the sqrt into a sampler-side
        // factor. For even k it's exact: p^(k/2).
        if (k % 2 == 0) {
            int64_t half = k / 2;
            int64_t sc = 1;
            for (int64_t i = 0; i < half; ++i) sc *= p;
            t.frobenius_scale *= sc;
        } else {
            // For odd k, the proper scale is p^(k/2) which is irrational.
            // We track p^k and let the caller compensate.
            int64_t sc = 1;
            for (int64_t i = 0; i < k; ++i) sc *= p;
            t.frobenius_scale *= sc;
        }
    }
}

void sp_ok_encode_apply_sato_tate_mix(sp_ok_tensor& t,
                                        int64_t p1, int64_t k1,
                                        int64_t p2, int64_t k2) {
    sp_ok_encode_apply_frobenius_quant(t, p1, k1);
    sp_ok_encode_apply_frobenius_quant(t, p2, k2);
}

// =========================================================================
// Round-trip helpers — encode → frobenius → decode in one call (in place
// on a fp16 buffer). Useful as the load-time shim before forward.cpp.
// =========================================================================

double sp_ok_apply_frobenius_quant_inplace_fp16(uint16_t* fp16_buf,
                                                  size_t numel,
                                                  int64_t p, int64_t k,
                                                  int64_t scale) {
    // Allocate a temporary sp_ok_tensor backed by fresh memory.
    sp_ok_arena tmp;
    tmp.reserve(numel * sizeof(sp_ok_t) + 64);
    sp_ok_tensor t;
    int64_t shape[4] = { (int64_t)numel, 1, 1, 1 };
    if (!sp_ok_encode_from_fp16(t, fp16_buf, 1, shape, scale, tmp)) {
        return 0.0;
    }
    sp_ok_encode_apply_frobenius_quant(t, p, k);
    sp_ok_decode_to_fp16(fp16_buf, t);
    return (double)t.frobenius_scale;
}

double sp_ok_apply_sato_tate_mix_inplace_fp16(uint16_t* fp16_buf,
                                                size_t numel,
                                                int64_t p1, int64_t k1,
                                                int64_t p2, int64_t k2,
                                                int64_t scale) {
    sp_ok_arena tmp;
    tmp.reserve(numel * sizeof(sp_ok_t) + 64);
    sp_ok_tensor t;
    int64_t shape[4] = { (int64_t)numel, 1, 1, 1 };
    if (!sp_ok_encode_from_fp16(t, fp16_buf, 1, shape, scale, tmp)) {
        return 0.0;
    }
    sp_ok_encode_apply_sato_tate_mix(t, p1, k1, p2, k2);
    sp_ok_decode_to_fp16(fp16_buf, t);
    return (double)t.frobenius_scale;
}

}  // namespace sp::engine
