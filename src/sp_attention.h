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
//
// Single-token mode (Phase 2.2a):
//   q.shape  = { d_q, 1 }                  d_q = n_head * head_dim
//   k.shape  = { T, d_kv }                 d_kv = n_kv_head * head_dim
//   v.shape  = { T, d_kv }
//   out.shape= { d_q, 1 }
//
// Multi-token prefill mode (Phase 2.2b):
//   q.shape  = { n_q,    d_q }             n_q queries
//   k.shape  = { T_kv,   d_kv }            full history including the
//                                          n_q new tokens
//   v.shape  = { T_kv,   d_kv }
//   out.shape= { n_q,    d_q }
//
// Causal mask is applied: query q[i] attends only to k[0..pos_i] where
// pos_i = pos_offset + i. If `pos_offset` is -1 (default), it's set to
// T_kv - n_q (i.e. the new tokens are at the end of the history — the
// usual case).
//
// `t_stride` lets callers pass a KV cache view whose feature stride is
// larger than the valid length. If t_stride < 0, uses k.shape[0] as the
// stride. `t_valid` is the number of valid history positions to attend
// over; if < 0, uses k.shape[0].
void sp_attention_dot_product(const sp_ok_tensor& q,
                                const sp_ok_tensor& k,
                                const sp_ok_tensor& v,
                                sp_ok_tensor&       out,
                                int n_head, int n_kv_head, int head_dim,
                                int t_valid   = -1,
                                int t_stride  = -1,
                                int pos_offset = -1);

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
