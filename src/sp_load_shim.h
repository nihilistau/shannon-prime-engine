// Shannon-Prime Engine — load-time Frobenius weight shim.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 1.7 — intercepts fp16 weight tensors at model load and runs them
// through the Frobenius (or Sato-Tate) shim from sp_ok_encode. The forward
// pass then sees fp16 weights that have been algebraically transformed
// per Paper D Config B or Config E.
//
// BYPASS POLICY (load-time decision per tensor name):
//
//   BYPASS (stay native fp16):
//     - *.norm.weight / *.rmsnorm.weight    — RMSNorm scale-reset operator
//     - output.weight / lm_head.weight      — LM head (logit scale ≠ softmax temp)
//     - tok_embd.weight / embed_tokens.weight — first residual stream stays native
//
//   SHIM (apply Frobenius):
//     - attn_q.weight / attn_k.weight / attn_v.weight  — Q/K/V projections
//     - attn_output.weight / attn_out.weight           — attention output proj
//     - ffn_gate.weight / ffn_up.weight / ffn_down.weight — SwiGLU FFN
//
// The bypass list is conservative: when in doubt, bypass. Tensor names
// not matched by either list are LOGGED AND BYPASSED so we don't silently
// quantize something we don't understand.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sp::engine {

// Mode applied to a single tensor.
enum class sp_shim_mode {
    Bypass,                  // leave fp16 unchanged
    FrobeniusQuant,          // apply phi_p^k (Config B)
    SatoTateMix,             // apply phi_p1^k1 ∘ phi_p2^k2 (Config E)
};

// Decide what to do for a given tensor name, given the engine's quant
// config (from src/engine.h Config struct).
//
// Returns one of {Bypass, FrobeniusQuant, SatoTateMix}.
// If cfg.frobenius_quant is true and the name is a SHIM candidate,
// returns FrobeniusQuant. Same for sato_tate_mix.
// If neither flag is set, always returns Bypass (legacy load path).
struct sp_shim_decision {
    sp_shim_mode mode;
    const char*  reason;     // diagnostic string ("RMSNorm bypass", "linear proj", etc.)
};

sp_shim_decision sp_shim_decide(const std::string& tensor_name,
                                  bool frobenius_quant,
                                  bool sato_tate_mix);

// Apply the chosen shim to a fp16 buffer in place.
//
// fp16_buf:    in-place buffer of `numel` fp16 elements
// p, k, etc.:  Frobenius parameters from cfg
// returns:     0 on success, non-zero on parameter / sanity error
//
// The per-tensor scale_recip is computed automatically from the buffer's
// absmax to keep |encoded_int64| < 2^30 with headroom for k applications
// of phi_p. Returns the chosen scale_recip in *out_scale (caller may want
// to log it; the decoded fp16 incorporates this scale already).
int sp_load_shim_apply_frobenius(uint16_t* fp16_buf, size_t numel,
                                   int64_t p, int64_t k,
                                   int64_t* out_scale);

int sp_load_shim_apply_sato_tate(uint16_t* fp16_buf, size_t numel,
                                   int64_t p1, int64_t k1,
                                   int64_t p2, int64_t k2,
                                   int64_t* out_scale);

// Diagnostic summary of one load pass. Emitted by sp_load_shim_run().
struct sp_load_shim_stats {
    int n_tensors_total       = 0;
    int n_tensors_bypassed    = 0;
    int n_tensors_shimmed     = 0;
    int n_tensors_unknown     = 0;   // not in either list — treat as bypass
    int64_t total_fp16_elems  = 0;
    int64_t shimmed_fp16_elems = 0;
};

}  // namespace sp::engine
