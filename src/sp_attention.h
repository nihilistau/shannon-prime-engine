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

#include <cstdint>

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
// Optional Phase 2.3b iter 3 args:
//   swa_window: > 0 enables sliding-window attention. Query at position
//               p attends only to history positions in
//               [max(0, p - swa_window + 1), p].
//   attn_logit_softcap: > 0 applies tanh(score / cap) * cap to the
//                       (scaled) scores before softmax — matches Gemma3's
//                       attention.logit_softcapping.
void sp_attention_dot_product(const sp_ok_tensor& q,
                                const sp_ok_tensor& k,
                                const sp_ok_tensor& v,
                                sp_ok_tensor&       out,
                                int n_head, int n_kv_head, int head_dim,
                                int   t_valid              = -1,
                                int   t_stride             = -1,
                                int   pos_offset           = -1,
                                int   swa_window           = 0,
                                float attn_logit_softcap   = 0.0f);

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

// =========================================================================
// Phase 3 pivot — CKKS-style polynomial-ring attention.
//
// Replaces the fp32 dot-product bridge in sp_attention_dot_product with
// integer polynomial multiplication in Z[x] / (x^N + 1). The score
// Σ q_i k_i is recovered exactly (to fp32 ULP) at coefficient x^{d-1}
// of Q(x) * K_rev(x). No metric topology destruction; KL=0 vs softmax
// in test_sp_poly_attention.
//
// Same call signature as sp_attention_dot_product so the forward step
// can dispatch on env var without changing call sites. Q/K/V are read
// from the same O_K representation; the polynomial encoding happens
// inside per (qi, t) pair, with N picked as the smallest power of 2 ≥
// head_dim (typically N = 2*head_dim).
// =========================================================================
// Phase 9b (post Plan C) — the CRT NTT path is the only NTT path the
// engine calls. When BOTH k_ntt_slab_q1 and k_ntt_slab_q2 are non-null,
// attention routes through the CRT pipeline (no __int128, portable to
// any 64-bit ALU). Their slot offset is (kv_h * t_stride + t) *
// SP_NTT_CRT_N within each slab. When the slabs are null, attention
// falls back to scalar sp_poly_dot_product (O(N^2), correct but slow).
void sp_attention_poly_ring(const sp_ok_tensor& q,
                              const sp_ok_tensor& k,
                              const sp_ok_tensor& v,
                              sp_ok_tensor&       out,
                              int n_head, int n_kv_head, int head_dim,
                              int   t_valid              = -1,
                              int   t_stride             = -1,
                              int   pos_offset           = -1,
                              int   swa_window           = 0,
                              float attn_logit_softcap   = 0.0f,
                              const uint64_t* k_ntt_slab_q1 = nullptr,
                              const uint64_t* k_ntt_slab_q2 = nullptr);

}  // namespace sp::engine
