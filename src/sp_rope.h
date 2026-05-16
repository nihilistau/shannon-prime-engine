// Shannon-Prime Engine — Rotary Position Embedding on O_K-coordinate tensors.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.2b. RoPE is a pure rotation in fp32 — there is no closed-form O_K
// representation for cos/sin at arbitrary positions, so we decode to fp32,
// rotate, and re-encode. Theorem 4 (Paper A) is preserved as long as Q and
// K carry the SAME Frobenius factor going in: post-RoPE we absorb that
// factor into the value and reset `frobenius_scale` to 1, so downstream
// attention's combined divisor stays correct.
//
// Layout convention (matches sp_matmul output of wq @ x):
//   qk.shape = { n_tokens, n_heads * head_dim }
//   qk.data[feature * n_tokens + token]  // shape[0]=n_tokens innermost
//
// For a single-token decode step (n_tokens=1) this collapses to
// qk.data[feature], which is what sp_attention expects.
//
// RoPE acts on pairs (2k, 2k+1) within each head, rotating by
//   angle = position[token] * freq[k]
// where freq[k] = freq_scale / freq_base^(2k / n_rot), n_rot=head_dim.

#pragma once

#include "sp_ok_tensor.h"

#include <cstddef>
#include <cstdint>

namespace sp::engine {

// RoPE rotation layout. Matches the ggml convention:
//   NORMAL — adjacent pairs (2k, 2k+1) get rotated together. Used by
//            Llama, Mistral, Granite.
//   NEOX   — half-pairs (k, k + n_rot/2) get rotated together. Used by
//            Qwen, Phi, Gemma 1/2/3, Phi-3. Picking the wrong mode
//            scrambles Q/K relative phase and destroys long-context
//            attention.
enum class sp_rope_mode {
    NORMAL = 0,
    NEOX   = 1,
};

// Apply RoPE in place to an O_K-coordinate Q or K tensor.
//
// qk:        shape = { n_tokens, n_heads * head_dim }. Modified in place.
// n_heads:   number of heads (n_head for Q, n_kv_head for K).
// head_dim:  dimensionality per head; must be even.
// n_tokens:  number of token positions (= qk.shape[0]).
// positions: per-token absolute position [n_tokens] (int32).
// freq_base: RoPE theta base (e.g. 10000 or 1000000 depending on model).
// freq_scale: RoPE freq scale multiplier (1.0 standard; <1 for extended ctx).
// mode:      NORMAL (default) or NEOX (Qwen/Phi/Gemma family).
//
// Returns false on shape mismatch / null data.
//
// On success:
//   qk.scale_recip unchanged
//   qk.frobenius_scale RESET to 1 (post-RoPE value absorbs the Frobenius
//                                   factor; Theorem 4 cancellation still
//                                   holds since K is also reset).
bool sp_rope_apply_ok(sp_ok_tensor&      qk,
                       int                n_heads,
                       int                head_dim,
                       int                n_tokens,
                       const int32_t*     positions,
                       float              freq_base,
                       float              freq_scale,
                       sp_rope_mode       mode = sp_rope_mode::NORMAL);

// Convenience overload: same positions buffer for a contiguous run
// [start_pos, start_pos + n_tokens). Common case for prefill.
bool sp_rope_apply_ok_contig(sp_ok_tensor& qk,
                              int           n_heads,
                              int           head_dim,
                              int           n_tokens,
                              int           start_pos,
                              float         freq_base,
                              float         freq_scale,
                              sp_rope_mode  mode = sp_rope_mode::NORMAL);

}  // namespace sp::engine
