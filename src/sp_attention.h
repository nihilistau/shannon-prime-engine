// Shannon-Prime Engine — Attention (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Two paths reserved here:
//   - sp_attention_dot_product: classical softmax(QK^T/sqrt(d)) * V,
//     ported to operate on O_K-coordinate Q/K/V.
//   - sp_attention_weil_pairing: Paper A §9.2 — replace softmax-of-dot-
//     product with e_n(Q, K) (Weil pairing). Phase 4 work. Verified in
//     the test suite via test_E92_weil_pairing_miller on E[3] over F_7.

#pragma once

#include "sp_ok_tensor.h"

namespace sp::engine {

// Classical multi-head attention in O_K coordinates.
// Q [n_head * head_dim], K [n_kv_head * head_dim], V [n_kv_head * head_dim].
// Writes the attention output to `out` [n_head * head_dim].
//
// The mask is causal (lower-triangular) by default.
//
// SKELETON: signature only. Phase 1.6 implementation fills in.
void sp_attention_dot_product(const sp_ok_tensor& q,
                                const sp_ok_tensor& k,
                                const sp_ok_tensor& v,
                                sp_ok_tensor&       out,
                                int n_head, int n_kv_head, int head_dim);

// Weil-pairing attention (Paper A §9.2). Phase 4 work.
//
// The Q and K vectors are projected onto E[n] for the chosen prime n
// dividing the layer order, then the pairing e_n(Q, K) replaces the
// scaled dot product. Linear in sequence length.
//
// NOT YET IMPLEMENTED. Stub returns the identity for now.
void sp_attention_weil_pairing(const sp_ok_tensor& q,
                                 const sp_ok_tensor& k,
                                 const sp_ok_tensor& v,
                                 sp_ok_tensor&       out,
                                 int n);

}  // namespace sp::engine
