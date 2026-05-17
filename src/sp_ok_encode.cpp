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
        // phi_p^(2m) = (-p)^m as SIGNED scalar on (a, b). For odd m the
        // scalar is negative (e.g. φ_2^2 = -2), and the SIGN MATTERS:
        // decoding by the absolute value alone produces sign-flipped
        // weights, which breaks the SwiGLU FFN (silu(-x) ≠ -silu(x))
        // and explodes PPL in compose paths (Config E).
        //
        // We therefore accumulate a SIGNED frobenius_scale = (-p)^m,
        // and decode divides by (scale_recip * frobenius_scale) where
        // frobenius_scale is signed. The negative-divisor / negative-
        // numerator cancellation recovers the original w correctly.
        int64_t m = k / 2;
        int64_t signed_scale = 1;
        for (int64_t i = 0; i < m; ++i) signed_scale *= (-p);
        t.frobenius_scale *= signed_scale;
    } else if (sp_is_split(p)) {
        // CRITICAL FIX (Phase 1.8): for a state (a, 0), multiplication by
        // pi^k = (pi_a, pi_b) produces (a*pi_a, a*pi_b). The a-component
        // scaling factor is exactly pi^k.a, NOT p^(k/2)=|pi^k|.
        //
        // The earlier formula (p^(k/2)) was the *norm*-based scaling,
        // which is correct only when pi^k = (sqrt(N), 0) — i.e. only
        // when pi^k is real-valued in our basis. For our canonical pi,
        // pi^k has nonzero b for any k that doesn't divide the order,
        // and the real-component scaling is strictly less than p^(k/2).
        //
        // Empirical signature of the bug it caused: PPL ~ 49 at Config B
        // vs baseline 19 because every weight got multiplied by
        // pi^k.a / p^(k/2) = pi^k.a / |pi^k| = cos(theta_k) < 1.
        sp_ok_t pi;
        if (sp_find_element_of_norm(p, &pi)) {
            sp_ok_t pi_pow = sp_ok_pow(pi, k);
            if (pi_pow.a != 0) {
                // Keep the SIGN of pi^k.a so decode correctly inverts the
                // (a, 0) -> (a * pi_pow.a, a * pi_pow.b) mapping that the
                // shim applies to weights.
                t.frobenius_scale *= pi_pow.a;
            } else {
                // pi^k has zero real part (rare; happens when k * theta_pi
                // crosses pi/2). Fall back to the norm-based scale and
                // accept the residual sign/rotation as a sampler factor.
                int64_t sc = 1;
                for (int64_t i = 0; i < k; ++i) sc *= p;
                t.frobenius_scale *= sc;
            }
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


// =========================================================================
// Phase 12 Step B: packed-int8 encoder
// =========================================================================

bool sp_ok_encode_q8_from_fp16_with_frobenius(sp_ok_q8_tensor& out,
                                              const uint16_t* w_fp16,
                                              size_t numel,
                                              int64_t scale,
                                              int64_t p, int64_t k,
                                              sp_ok_arena& arena,
                                              sp_ok_t* scratch) {
    if (numel == 0 || !w_fp16) return false;

    /* Allocate packed output from arena (2 * numel bytes). */
    if (!arena.alloc_tensor_q8(out, numel)) return false;

    /* Acquire scratch buffer. If caller didn't supply one, use a local
     * vector; this matches the existing in-place-fp16 path which also
     * allocates a temporary arena per call. */
    std::vector<sp_ok_t> local_scratch;
    sp_ok_t* sok = scratch;
    if (!sok) {
        local_scratch.resize(numel);
        sok = local_scratch.data();
    }

    /* Step 1: fp16 -> sp_ok_t (a = round(fp16 * scale), b = 0). */
    for (size_t i = 0; i < numel; ++i) {
        float v = fp16_to_fp32(w_fp16[i]) * (float)scale;
        sok[i].a = (int64_t)std::llrint(v);
        sok[i].b = 0;
    }

    /* Step 2: apply Frobenius phi_p^k in place on the int64 buffer. */
    int64_t frob_scale = 1;
    if (k != 0) {
        sp_frobenius_quant_tensor(sok, numel, p, k);
        /* Replicate the frobenius_scale tracking from
         * sp_ok_encode_apply_frobenius_quant: split prime uses pi^k.a as
         * the signed scale, inert uses (-p)^(k/2). */
        if (sp_is_inert(p)) {
            int64_t m = k / 2;
            int64_t ss = 1;
            for (int64_t i = 0; i < m; ++i) ss *= (-p);
            frob_scale = ss;
        } else if (sp_is_split(p)) {
            sp_ok_t pi;
            if (sp_find_element_of_norm(p, &pi)) {
                sp_ok_t pi_pow = sp_ok_pow(pi, k);
                if (pi_pow.a != 0) {
                    frob_scale = pi_pow.a;
                } else {
                    int64_t ss = 1;
                    for (int64_t i = 0; i < k; ++i) ss *= p;
                    frob_scale = ss;
                }
            }
        }
    }

    /* Step 3: pack post-Frobenius (a, b) -> packed int8 pair with per-tensor
     * shift. sp_ok_q8_encode_array computes absmax internally and returns
     * the chosen shift. */
    int8_t shift = sp_ok_q8_encode_array(out.data, sok, numel);

    /* Step 4: populate metadata. */
    out.q8_shift        = shift;
    out.scale_recip     = scale;
    out.frobenius_scale = frob_scale;
    out.frobenius_p     = (int16_t)p;
    out.frobenius_k     = (int16_t)k;
    return true;
}

}  // namespace sp::engine
