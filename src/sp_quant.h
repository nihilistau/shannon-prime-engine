// Shannon-Prime Engine — native dequant for GGUF K-quants.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Reproduces ggml's K-quant block layout + dequant math, bit-for-bit.
// Used by forward_native.cpp so we don't have to drag in ggml's
// graph compute pipeline just to read GGUF weights.
//
// Block layouts match `block_q*_K` structs in ggml-common.h. We use
// our own structs of identical size so the GGUF weight bytes can be
// memcpy'd / mmap'd directly without re-encoding.
//
// Phase 4 priority: Q5_K_M (Qwen2.5-Coder-3B is shipped at this quant).
// Q4_K, Q6_K, Q8_0, F16 dequant added as we hit models that need them.

#pragma once

#include <cstdint>
#include <cstddef>

namespace sp::engine {

// QK_K = 256 — every K-quant block packs 256 elements.
inline constexpr int SP_QK_K        = 256;
// 12 bytes of packed 6-bit scales + 6-bit mins (8 sub-blocks of 32).
inline constexpr int SP_K_SCALE_SZ  = 12;

// fp16 helper. Native conversion via __fp16 on arm64; portable bit-shift
// fallback elsewhere. Defined in sp_quant.cpp.
float    sp_fp16_to_fp32(uint16_t h);
uint16_t sp_fp32_to_fp16(float f);

// ---------------------------------------------------------------------
// Q5_K block — 176 bytes for 256 elements (5.5 effective bits/element)
// ---------------------------------------------------------------------
//
// fp16 d:     super-block scale for the 8 sub-block scales
// fp16 dmin:  super-block scale for the 8 sub-block mins
// scales[12]: 8 sub-blocks × (6-bit scale + 6-bit min) packed
// qh[32]:     high bit (1 bit per quant, 256 bits = 32 bytes)
// qs[128]:    low 4 bits (1 nibble per quant, 256 nibbles = 128 bytes)
//
// Sub-block i (i in [0,8)):
//   scale s_i (6 bits), min m_i (6 bits) extracted from `scales[]`
//   For each of 32 elements x_j:
//     q_j = (low4 from qs) | (high1 from qh << 4)   // 5-bit quant
//     y_j = d * s_i * q_j - dmin * m_i
//
// Total bytes: 4 + 12 + 32 + 128 = 176 ✓
struct alignas(2) sp_block_q5_K {
    uint16_t d;                          // fp16 super-scale
    uint16_t dmin;                       // fp16 super-min-scale
    uint8_t  scales[SP_K_SCALE_SZ];      // 6+6 bit packed scales/mins
    uint8_t  qh[SP_QK_K / 8];            // 32 bytes — high bit plane
    uint8_t  qs[SP_QK_K / 2];            // 128 bytes — low 4-bit nibbles
};
static_assert(sizeof(sp_block_q5_K) == 176, "sp_block_q5_K must be 176 bytes");

// Dequantize `nblocks` Q5_K blocks (= nblocks * 256 fp32 elements) into
// `out`. Bit-exact with ggml's dequantize_row_q5_K.
void sp_dequant_q5_K_to_f32(const sp_block_q5_K* blocks,
                             float* out,
                             size_t nblocks);

// Same, but writes fp16 (saves a pass when downstream consumer is fp16).
// Internal compute is still fp32 for accuracy; only the store narrows.
void sp_dequant_q5_K_to_f16(const sp_block_q5_K* blocks,
                             uint16_t* out,
                             size_t nblocks);

}  // namespace sp::engine
