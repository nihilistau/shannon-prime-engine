// qnn_bin_quant_table.h — per-tensor UFIXED_POINT_16 scale/offset table for
// AI Hub-compiled Qwen3-4B V69 .bins.
//
// AI Hub bakes quantization parameters into compiled graph ops rather than
// exposing them via QNN tensor metadata (schema dump always returns
// quant_encoding=0 for dtype=1046 tensors). This table captures the
// effective (scale, offset) for each tensor class, recovered via the
// multi-point probe script at tools/qnn_bin_probe.py.
//
// QNN canonical codec (QnnTypes.h Qnn_ScaleOffset_t convention):
//   decode:  fp32 = (uint16 + offset) * scale
//   encode:  uint16 = round(fp32 / scale) - offset
//
// Note: the empirical probe uses fp32 = scale * uint16 + offset
// (sign-absorbed into offset), which is equivalent when the stored
// offset absorbs the term. Both columns below follow the QNN convention.
//
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#pragma once

#include <cstdint>

namespace sp::engine {

struct Ufixed16Params {
    float   scale;
    int32_t offset;   // QNN sign convention: fp32 = (uint16 + offset) * scale
};

// ── Output tensors (recovered from qnn-net-run fp32 vs native probe) ──────

// Split 1 embedding output: [1, 128, 2560]
// Empirical anchor from Phase 5.7 handoff. Probe cross-checks this.
// fp32 ≈ 7.2e-6 * uint16 - 0.222  →  scale=7.2e-6, offset = -0.222/7.2e-6 ≈ -30833
// Run tools/qnn_bin_probe.py to get exact values; these are the pre-probe estimates.
static constexpr Ufixed16Params QNN_QUANT_EMBEDDING = { 7.2e-6f, -30833 };

// Inter-split residual streams: [1, 128, 2560] — same hidden dim as embedding.
// Expect similar range to embedding; probe will refine.
// Placeholder: use embedding params until probe runs.
static constexpr Ufixed16Params QNN_QUANT_RESIDUAL = { 7.2e-6f, -30833 };

// Logits output (split 4): [1, 128, 151936]
// Range covers raw logit scores (typically -30..+30 for Qwen3 w4a16).
// Placeholder; argmax over uint16 is valid regardless of scale (scale>0 preserves order).
// Probe will give exact values for dequant-based sampling.
static constexpr Ufixed16Params QNN_QUANT_LOGITS = { 1.0e-3f, -32768 };

// ── Input tensors (derived from calibration range properties) ─────────────

// attention_mask: values ∈ {0.0, mask_neg ≈ -65504}.
// AIMET w4a16 export calibrates over [−65504, 0]:
//   QNN decode: fp32 = (uint16 + offset) * scale
//   For uint16=65535 → fp32=0:  (65535 + offset)*scale = 0  → offset = -65535
//   For uint16=0     → fp32=-65504: (0-65535)*scale = -65504 → scale = 65504/65535
//   encode(0.0)     → uint16 = 65535  (attended, maximum)
//   encode(-65504)  → uint16 = 0      (blocked, minimum)
// Use --use_native_input_files probe to confirm; or override at runtime via
// SP_QNN_BIN_MASK_ATTENDED_U16 / SP_QNN_BIN_MASK_BLOCKED_U16 env vars.
static constexpr Ufixed16Params QNN_QUANT_MASK = { 65504.0f / 65535.0f, -65535 };

// position_ids_cos / position_ids_sin: values ∈ [-1.0, +1.0].
// UFIXED_16 over [-1, 1]:
//   scale = 2.0 / 65535 ≈ 3.0518e-5, offset = -32768
//   encode(-1.0) → uint16 = 0,     encode(0.0) → uint16 = 32768,
//   encode(+1.0) → uint16 = 65535
static constexpr Ufixed16Params QNN_QUANT_COS_SIN = { 3.0518e-5f, -32768 };

// past_key / past_value initial zeros.
// Residual stream dtype — same params as QNN_QUANT_RESIDUAL.
// "Quantized zero" is uint16 = round(0 / scale) - offset = 30833.
static constexpr Ufixed16Params QNN_QUANT_PAST_KV = { 7.2e-6f, -30833 };

// ── Encode / decode helpers ───────────────────────────────────────────────

// Encode fp32 → uint16, clamped to [0, 65535].
// QNN convention: uint16 = round(fp32 / scale) - offset
inline uint16_t sp_ufixed16_encode(float v, Ufixed16Params p) {
    // q = fp32/scale − offset
    float q = v / p.scale - (float)p.offset;
    if (q < 0.0f)     q = 0.0f;
    if (q > 65535.0f) q = 65535.0f;
    return (uint16_t)(q + 0.5f);
}

// Decode uint16 → fp32.
// QNN convention: fp32 = (uint16 + offset) * scale
inline float sp_ufixed16_decode(uint16_t q, Ufixed16Params p) {
    return ((float)q + (float)p.offset) * p.scale;
}

// The uint16 value that encodes fp32=0.0 for a given tensor.
// Used to zero-fill past_key/past_value buffers correctly.
inline uint16_t sp_ufixed16_zero(Ufixed16Params p) {
    return sp_ufixed16_encode(0.0f, p);
}

}  // namespace sp::engine
