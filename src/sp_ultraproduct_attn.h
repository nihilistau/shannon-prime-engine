// Shannon-Prime Engine — Phase 7: Ultraproduct attention (Paper III §5.3).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Standard softmax attention computes
//     Attn(Q, K, V) = Σ_t σ_t · V_t,    σ_t = softmax_t(QK^T / sqrt(d_k)).
//
// Ultraproduct attention replaces the soft mixture with the ultraproduct
// limit along an ultrafilter U on key positions:
//     UltraAttn(Q, K, V; U) = ult_U(V_t).
//
// Łoś's theorem: a first-order property holds on UltraAttn iff it holds
// for a U-large set of positions.  On a finite cache every ultrafilter
// is principal — U = U_{p*} for some position p* — and
//     ult_{U_{p*}}(V_t) = V_{p*}.
// This is the "top-1 attention" the paper refers to (§5.3).
//
// Phase 7 ships ONLY the principal case.  p* is chosen as the argmax of
// the (scaled, softcap'd, sieve-masked) attention scores — exactly the
// score field the softmax variant would have consumed — so the sieve's
// per-(layer, position) eviction mask, the SWA window, and the soft-γ
// attenuation all continue to influence which key wins.
//
// This is hard attention; gradient flow through the argmax is undefined.
// The path is INFERENCE-ONLY.  Calling this kernel during training is a
// no-op for the soft mixture path it replaces.
//
// API mirrors sp_attention_dot_product so sp_forward.cpp can dispatch by
// flag without changing call sites.

#pragma once

#include <cstdint>

#include "sp_ok_tensor.h"

namespace sp::engine {

// Modes — kept here so engine.h doesn't pull this header.
//   NONE         — standard softmax attention (the existing path).
//   PRINCIPAL    — argmax-based hard attention; V_{p*} dispatched per
//                  (query, head).  Shipping in Phase 7.
//   NONPRINCIPAL — the §5.3 non-principal limit; deferred to Phase 7+.
enum class sp_ultraproduct_mode : int {
    NONE         = 0,
    PRINCIPAL    = 1,
    NONPRINCIPAL = 2,
};

// Principal-case ultraproduct attention.
//
// Score compute, softcap, SWA window, and sieve mask (soft-γ or hard
// NEG_INF) follow sp_attention_dot_product exactly.  Only the final
// reduction differs:
//   - sp_attention_dot_product:  out_h = Σ_t softmax(scores)_t · V_h[t]
//   - sp_ultraproduct_attn_principal:  out_h = V_h[p*],
//     where p* = argmax_{t ∈ [t_lo, t_hi)} scores[t]
//                with evicted positions excluded if they would tie.
//
// Optional Phase 7 instrumentation: pass selected_pos != nullptr to
// receive the chosen position per (qi, h).  Layout is [n_q * n_head].
void sp_ultraproduct_attn_principal(const sp_ok_tensor& q,
                                      const sp_ok_tensor& k,
                                      const sp_ok_tensor& v,
                                      sp_ok_tensor&       out,
                                      int n_head, int n_kv_head, int head_dim,
                                      int   t_valid_arg           = -1,
                                      int   t_stride_arg          = -1,
                                      int   pos_offset_arg        = -1,
                                      int   swa_window            = 0,
                                      float attn_logit_softcap    = 0.0f,
                                      const uint8_t* evicted_mask = nullptr,
                                      float evicted_gamma         = 0.0f,
                                      int32_t* selected_pos       = nullptr);

}  // namespace sp::engine
