// sp_quant — native K-quant dequant. See sp_quant.h.

#include "sp_quant.h"

#include <cstring>

namespace sp::engine {

// ---------------------------------------------------------------------
// fp16 ↔ fp32
// ---------------------------------------------------------------------

#if defined(__aarch64__) || defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 2)
// Native fp16 path on arm64 — the compiler emits a direct conversion.
float sp_fp16_to_fp32(uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, sizeof(v));
    return (float)v;
}
uint16_t sp_fp32_to_fp16(float f) {
    __fp16 v = (__fp16)f;
    uint16_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}
#else
// Portable bit-twiddling fallback for x86 desktop / older ARM.
// IEEE 754 binary16: 1 sign | 5 exp (bias 15) | 10 mantissa.
float sp_fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FF;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;  // signed zero
        } else {
            // Subnormal fp16 → normal fp32. Shift m left until bit 10
            // (the implicit-one position) is set; each shift consumes
            // one unit of exponent. fp16 subnormals have effective
            // exponent -14 (as if e==1) and we additionally subtract
            // (shifts - 1) for the implicit-bit normalization, hence
            // bias = 127 + (-14 - (shifts - 1)) = 114 - shifts.
            uint32_t mm = m, shifts = 1;
            while (!(mm & 0x400)) { mm <<= 1; ++shifts; }
            const uint32_t bias_exp = 114u - shifts;
            f = s | (bias_exp << 23) | ((mm & 0x3FF) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000 | (m << 13);  // inf or NaN
    } else {
        f = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}
uint16_t sp_fp32_to_fp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t s = (u >> 16) & 0x8000;
    int32_t  e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFF;

    if (e <= 0) {
        // Underflow → subnormal or zero, with round-to-nearest-even.
        if (e < -10) return (uint16_t)s;
        m |= 0x800000;
        const uint32_t shift = (uint32_t)(14 - e);
        const uint32_t r = m & ((1u << shift) - 1u);
        m >>= shift;
        if (r > (1u << (shift - 1)) ||
            (r == (1u << (shift - 1)) && (m & 1))) {
            ++m;
        }
        return (uint16_t)(s | m);
    }
    if (e >= 31) {
        // Overflow → inf (or NaN preserved).
        return (uint16_t)(s | 0x7C00 | (m ? 1 : 0));
    }
    // Normal: round-to-nearest-even on the trailing 13 bits.
    const uint32_t r = m & 0x1FFF;
    m >>= 13;
    if (r > 0x1000 || (r == 0x1000 && (m & 1))) {
        ++m;
        if (m == 0x400) { m = 0; ++e; }
        if (e >= 31) return (uint16_t)(s | 0x7C00);
    }
    return (uint16_t)(s | (e << 10) | m);
}
#endif

// ---------------------------------------------------------------------
// Q5_K dequant
// ---------------------------------------------------------------------
//
// Reproduces dequantize_row_q5_K from ggml-quants.c (line 1669) bit-
// for-bit. The trick is the 6+6-bit scale/min packing in `scales[12]`.
// ggml's helper get_scale_min_k4 has two cases keyed on j<4 vs j>=4:
//
//   j < 4:                                           // sub-blocks 0..3
//     scale = scales[j]   & 63
//     min   = scales[j+4] & 63
//   j >= 4:                                          // sub-blocks 4..7
//     scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//     min   = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
//
// The 5-bit quants are reconstructed as low4 (from qs) + high1 (from qh).
// The high-bit-plane mask shifts left by 2 every TWO sub-blocks (because
// each pair of sub-blocks shares 32 bytes of qs but uses different qh
// bit positions).

static inline void sp_q5_K_get_scale_min(int j,
                                         const uint8_t* q,
                                         uint8_t& sc,
                                         uint8_t& mn) {
    if (j < 4) {
        sc = q[j]     & 0x3F;
        mn = q[j + 4] & 0x3F;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >>   4) | ((q[j]     >> 6) << 4);
    }
}

void sp_dequant_q5_K_to_f32(const sp_block_q5_K* blocks,
                             float* out,
                             size_t nblocks) {
    for (size_t i = 0; i < nblocks; ++i) {
        const sp_block_q5_K& b = blocks[i];
        const float d    = sp_fp16_to_fp32(b.d);
        const float dmin = sp_fp16_to_fp32(b.dmin);

        const uint8_t* qh = b.qh;
        const uint8_t* ql = b.qs;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < SP_QK_K; j += 64) {
            uint8_t sc, mn;
            sp_q5_K_get_scale_min(is + 0, b.scales, sc, mn);
            const float d1 = d * sc;
            const float m1 = dmin * mn;
            sp_q5_K_get_scale_min(is + 1, b.scales, sc, mn);
            const float d2 = d * sc;
            const float m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) {
                const int q = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                *out++ = d1 * (float)q - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const int q = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                *out++ = d2 * (float)q - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void sp_dequant_q5_K_to_f16(const sp_block_q5_K* blocks,
                             uint16_t* out,
                             size_t nblocks) {
    // Two-pass: dequant to fp32 in a small stack buffer, then narrow.
    // 256 floats × sizeof(float) = 1 KB stack per block — fine.
    for (size_t i = 0; i < nblocks; ++i) {
        float tmp[SP_QK_K];
        sp_dequant_q5_K_to_f32(&blocks[i], tmp, 1);
        for (int k = 0; k < SP_QK_K; ++k) {
            out[k] = sp_fp32_to_fp16(tmp[k]);
        }
        out += SP_QK_K;
    }
}

}  // namespace sp::engine
