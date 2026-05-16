// Shannon-Prime Engine — fp16 <-> O_K encoding + Frobenius shim.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 1.3 of the Theory-First engine. Bridges fp16 weight tensors
// (which is what GGUF gives us) to sp_ok_tensor (O_K-coordinate state)
// and back. The pipeline:
//
//    fp16 tensor               sp_ok_tensor
//    ----------                ------------
//    w[i] = float              data[i] = { a = round(w*S), b = 0 }
//
// where S = `scale_recip` is per-tensor, chosen so that the rounded
// integer fits comfortably in int64 (target: |a| <= 2^30 to leave
// 32 bits of Frobenius headroom; S is typically 2^16 to 2^24).
//
// THE FROBENIUS SHIM (Paper D Config B + Config E):
//
// Once weights are encoded, sp_ok_encode_apply_frobenius_quant() or
// sp_ok_encode_apply_sato_tate_mix() runs the framework's quantization
// in place. The KEY PROPERTY (Theorem 4 of Paper A, verified by
// test_sp_frobenius bit-exactly):
//
//    decode(phi_p^k(encode(w))) == w * (rational scale factor)
//
// where the rational scale factor is exactly N(pi)^(k/2) for split p
// or p^(k/2) for inert p. Because the scale factor is *known and
// rational*, decoding produces the original w up to that constant —
// and because the same constant applies uniformly to every weight in
// every layer, attention scores and softmax outputs are INVARIANT
// (the constant cancels in QK^T after scaling, and in V*attention
// after the final divide-by-sqrt(d)).
//
// Practical effect on inference:
//   - load weights → encode → apply Frobenius → decode → write back
//   - the forward pass sees fp16 weights that are bit-shifted /
//     algebraically transformed but produce identical attention
//     outputs to the unshifted weights (Theorem 4)
//   - the FIRST-ORDER drift is the Frobenius trace |a_p|, which is
//     0 for inert primes (Deuring) and bounded by 2*sqrt(p) for
//     split primes (Hasse-Weil). At p=41 this gives |a_41|=1 — the
//     smallest possible non-zero drift.
//
// This is the SHIM path: existing forward.cpp still runs, but the
// weights it sees are Theory-First quantized. The pure-SP sp_forward
// (Phase 1.6+) will avoid the decode step entirely.

#pragma once

#include "sp_ok_tensor.h"

#include <cstdint>
#include <vector>

namespace sp::engine {

// -----------------------------------------------------------------------
// Encoding parameters.
// -----------------------------------------------------------------------

// Per-tensor scale recommendation. Chosen so that the maximum
// |round(w*S)| stays well below 2^30, leaving headroom for an 8-power
// of Frobenius at p=41 (max growth factor N(omega)^8 ≈ 7.98e12 — needs
// the scale to start no larger than ~2^17 to fit in int64).
//
// For Phi-3 / Qwen weights with absmax ~ 8.0, scale_recip = 1<<16 = 65536
// keeps |encoded| ≤ 8*65536 ≈ 524288 ≈ 2^19. After phi_41^8 this becomes
// 2^19 * 2^43 = 2^62 — fits in int64 with 1 bit of headroom. Tight but
// workable. For larger k or smaller p, increase the divisor.
inline int64_t sp_ok_encode_recommend_scale(int64_t k_max_frobenius = 8,
                                              int64_t p_split        = 41) {
    // Conservative: pick S such that S * absmax(w) <= 2^30, leaving
    // 2^32 of int64 headroom for k_max applications of phi_p.
    // For now use a fixed value tuned to Phi-3 absmax ~ 8.0.
    (void)k_max_frobenius; (void)p_split;
    return (int64_t)1 << 14;  // 16384 — conservative
}

// -----------------------------------------------------------------------
// Encode a contiguous fp16 / fp32 buffer into an sp_ok_tensor.
// -----------------------------------------------------------------------

// Encode fp32 weight tensor `w` (numel elements) into `out`. Allocates
// from `arena`. Sets out.scale_recip to `scale`. b-component is zero
// (weights are "scalar in omega-direction" — Möbius-trivial).
//
// Returns true on success, false if the arena can't hold the result.
bool sp_ok_encode_from_fp32(sp_ok_tensor& out, const float* w,
                             int n_dims, const int64_t shape[4],
                             int64_t scale, sp_ok_arena& arena);

// Same but reads from fp16 (uint16_t IEEE half).
bool sp_ok_encode_from_fp16(sp_ok_tensor& out, const uint16_t* w_fp16,
                             int n_dims, const int64_t shape[4],
                             int64_t scale, sp_ok_arena& arena);

// -----------------------------------------------------------------------
// Decode back to fp32 / fp16.
//
// Accounts for any Frobenius applications stored in t.frobenius_scale:
//   w_decoded = (float) t.data[i].a / (t.scale_recip * t.frobenius_scale)
//
// The b-component is dropped. If b is non-zero (the state has departed
// from the "scalar in omega-direction" subspace), the caller may want
// to apply a custom projection — for the Phase 1 shim, dropping b is
// the right behavior because Frobenius application keeps b proportional
// to the b-direction component, which started at zero.
// -----------------------------------------------------------------------
void sp_ok_decode_to_fp32(float* out, const sp_ok_tensor& t);
void sp_ok_decode_to_fp16(uint16_t* out_fp16, const sp_ok_tensor& t);

// -----------------------------------------------------------------------
// Frobenius shim — apply the quant tier to an in-flight sp_ok_tensor.
// -----------------------------------------------------------------------

// Config B: --frobenius-quant. Applies phi_p^k in place.
// Updates t.frobenius_scale so subsequent decode produces the correct
// fp32 values.
void sp_ok_encode_apply_frobenius_quant(sp_ok_tensor& t,
                                          int64_t p, int64_t k);

// Config E: --sato-tate-mix. Applies phi_p1^k1 ∘ phi_p2^k2 in place.
void sp_ok_encode_apply_sato_tate_mix(sp_ok_tensor& t,
                                        int64_t p1, int64_t k1,
                                        int64_t p2, int64_t k2);

// -----------------------------------------------------------------------
// Round-trip helpers — encode → frobenius → decode in one call.
// Useful as the load-time shim before forward.cpp runs.
// -----------------------------------------------------------------------

// fp16 -> O_K -> phi^k -> O_K -> fp16 (in place on a fp16 buffer).
// Returns the scale factor applied (for diagnostics).
double sp_ok_apply_frobenius_quant_inplace_fp16(uint16_t* fp16_buf,
                                                  size_t numel,
                                                  int64_t p, int64_t k,
                                                  int64_t scale);

double sp_ok_apply_sato_tate_mix_inplace_fp16(uint16_t* fp16_buf,
                                                size_t numel,
                                                int64_t p1, int64_t k1,
                                                int64_t p2, int64_t k2,
                                                int64_t scale);

}  // namespace sp::engine
