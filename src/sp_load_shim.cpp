// Shannon-Prime Engine — load-time Frobenius weight shim (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_load_shim.h"
#include "sp_ok_encode.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sp::engine {

// ---------- fp16 absmax (for per-tensor scale derivation) -----------------

static inline float fp16_to_fp32_local(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

static float buffer_absmax_fp16(const uint16_t* buf, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float v = std::fabs(fp16_to_fp32_local(buf[i]));
        if (v > mx) mx = v;
    }
    return mx;
}

// ---------- name matching --------------------------------------------------

// Case-insensitive substring search.
static bool name_contains(const std::string& s, const char* sub) {
    const size_t n = s.size();
    const size_t m = std::strlen(sub);
    if (m == 0 || m > n) return m == 0;
    for (size_t i = 0; i + m <= n; ++i) {
        bool ok = true;
        for (size_t j = 0; j < m; ++j) {
            char a = s[i + j], b = sub[j];
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
            if (a != b) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}

// Tensors that must STAY native fp16.
static bool is_bypass_tensor(const std::string& n) {
    // RMSNorm / LayerNorm scales (any variant)
    if (name_contains(n, "norm.weight")) return true;
    if (name_contains(n, "_norm"))       return true;
    if (name_contains(n, "ln_"))         return true;
    // LM head — keep logit scale at native
    if (name_contains(n, "output.weight"))   return true;
    if (name_contains(n, "lm_head"))         return true;
    if (name_contains(n, "output_norm"))     return true;  // belt-and-braces
    // Token embedding — first residual stream native
    if (name_contains(n, "tok_embd"))        return true;
    if (name_contains(n, "embed_tokens"))    return true;
    if (name_contains(n, "token_embd"))      return true;
    // Bias-like (we don't expect biases in Phi-3, but Llama/Qwen do have them)
    if (name_contains(n, ".bias"))           return true;
    return false;
}

// Tensors that are SHIM candidates (linear projections).
static bool is_shim_tensor(const std::string& n) {
    if (name_contains(n, "attn_q"))        return true;
    if (name_contains(n, "attn_k"))        return true;
    if (name_contains(n, "attn_v"))        return true;
    if (name_contains(n, "attn_output"))   return true;
    if (name_contains(n, "attn_out"))      return true;
    if (name_contains(n, "wq.weight"))     return true;
    if (name_contains(n, "wk.weight"))     return true;
    if (name_contains(n, "wv.weight"))     return true;
    if (name_contains(n, "wo.weight"))     return true;
    if (name_contains(n, "ffn_gate"))      return true;
    if (name_contains(n, "ffn_up"))        return true;
    if (name_contains(n, "ffn_down"))      return true;
    if (name_contains(n, "mlp.gate"))      return true;
    if (name_contains(n, "mlp.up"))        return true;
    if (name_contains(n, "mlp.down"))      return true;
    return false;
}

// ---------- decision -------------------------------------------------------

sp_shim_decision sp_shim_decide(const std::string& tensor_name,
                                  bool frobenius_quant,
                                  bool sato_tate_mix) {
    // Biases get bypassed unconditionally — they sit in the residual
    // stream and shouldn't be Frobenius-scaled. Check this first because
    // ".bias" can appear in a tensor named like "blk.0.attn_q.bias" which
    // would otherwise hit the shim list.
    if (name_contains(tensor_name, ".bias")) {
        return { sp_shim_mode::Bypass, "bias tensor — bypass" };
    }
    // Norm scales bypass — same reasoning, scale-reset operator.
    if (name_contains(tensor_name, "norm.weight") ||
        name_contains(tensor_name, "_norm") ||
        name_contains(tensor_name, "ln_")) {
        return { sp_shim_mode::Bypass, "RMSNorm/LayerNorm bypass" };
    }
    // Token embedding bypass.
    if (name_contains(tensor_name, "tok_embd")     ||
        name_contains(tensor_name, "embed_tokens") ||
        name_contains(tensor_name, "token_embd")) {
        return { sp_shim_mode::Bypass, "token embedding bypass" };
    }
    // Check SHIM list before falling to LM head bypass, so that
    // "attn_output.weight" matches Q/K/V/O before "output.weight" matches
    // the LM head pattern.
    if (is_shim_tensor(tensor_name)) {
        if (sato_tate_mix) {
            return { sp_shim_mode::SatoTateMix, "linear projection — Sato-Tate mix" };
        }
        if (frobenius_quant) {
            return { sp_shim_mode::FrobeniusQuant, "linear projection — Frobenius" };
        }
        return { sp_shim_mode::Bypass, "linear proj, but no quant flag set" };
    }
    // LM head bypass — checked LAST among bypass categories so that the
    // "output.weight" substring doesn't false-positive against the shim
    // tensors (e.g. blk.N.attn_output.weight, which is a shim target).
    if (name_contains(tensor_name, "output.weight") ||
        name_contains(tensor_name, "lm_head")) {
        return { sp_shim_mode::Bypass, "LM head bypass" };
    }
    // Unknown — bypass conservatively. Caller should log.
    return { sp_shim_mode::Bypass, "unknown tensor — conservative bypass" };
}

// ---------- apply ----------------------------------------------------------

// Derive per-tensor scale_recip from absmax so |encoded a| stays below 2^30.
// Conservative: pick scale such that absmax * scale = 2^24 (leaving 6 bits
// for the Frobenius power).
static int64_t pick_scale_for_buffer(float absmax) {
    if (absmax <= 0.0f) return 1;
    // Target |encoded| ≈ 2^24 = 16777216
    double target = 16777216.0;
    double s_d = target / (double)absmax;
    if (s_d > 1.0e15) s_d = 1.0e15;  // sanity ceiling
    if (s_d < 1.0)    s_d = 1.0;
    return (int64_t)s_d;
}

int sp_load_shim_apply_frobenius(uint16_t* fp16_buf, size_t numel,
                                   int64_t p, int64_t k,
                                   int64_t* out_scale) {
    if (!fp16_buf || numel == 0) return -1;
    float absmax = buffer_absmax_fp16(fp16_buf, numel);
    int64_t scale = pick_scale_for_buffer(absmax);
    double frob_scale = sp_ok_apply_frobenius_quant_inplace_fp16(
        fp16_buf, numel, p, k, scale);
    if (out_scale) *out_scale = scale;
    return (frob_scale > 0.0) ? 0 : -2;
}

int sp_load_shim_apply_sato_tate(uint16_t* fp16_buf, size_t numel,
                                   int64_t p1, int64_t k1,
                                   int64_t p2, int64_t k2,
                                   int64_t* out_scale) {
    if (!fp16_buf || numel == 0) return -1;
    float absmax = buffer_absmax_fp16(fp16_buf, numel);
    int64_t scale = pick_scale_for_buffer(absmax);
    double frob_scale = sp_ok_apply_sato_tate_mix_inplace_fp16(
        fp16_buf, numel, p1, k1, p2, k2, scale);
    if (out_scale) *out_scale = scale;
    return (frob_scale > 0.0) ? 0 : -2;
}

}  // namespace sp::engine
