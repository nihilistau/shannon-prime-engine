// Shannon-Prime Engine — PrimePE-RoPE-ALiBi
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Builds per-frequency RoPE factors and ALiBi slopes for the engine's
// positional-encoding family. The canonical name is **PrimePE-RoPE-ALiBi**
// — lattice-drawn integer frequencies (replacing the pure-geometric RoPE
// base), blended at `alpha` with the standard geometric schedule,
// optionally composed with per-head ALiBi biases.
//
// The feed-forward engine treats the output of this module as a constant
// over the forward pass; it's cheap to rebuild each graph.

#pragma once

#include "engine.h"   // PeMode enum lives on Config
#include <vector>

namespace sp::engine {

// Returns a length-(n_rot/2) fp32 tensor of per-frequency divisors to
// feed into ggml_rope_ext's `freq_factors` input. The i-th element
// scales the standard geometric freq: effective_freq_i = base_i / f[i].
//
// The underlying lattice is the canonical three-tier
// composite/prime allocation from `prime_rope.h`:
//   - local tier (dims 0..n/3):   small numbers (2..64)   — high freq
//   - mid tier   (dims n/3..2n/3): medium (64..1024)
//   - long tier  (dims 2n/3..n):  large (1024..8192)      — low freq
// Factors are then blended with identity at `pe_alpha`
// (alpha=0 ⇒ every factor = 1 ⇒ pure geometric RoPE).
//
// Empty return value means "pass nullptr to ggml_rope_ext" — i.e.,
// use the standard geometric schedule unchanged. Returned when
// pe_mode is Standard or AlibiOnly, or when pe_alpha is exactly 0.
std::vector<float> prime_pe_freq_factors(Config::PeMode pe_mode,
                                         float  pe_alpha,
                                         int    pe_tier,
                                         int    n_rot,
                                         float  freq_base);

// Returns the `max_bias` scalar to hand ggml_soft_max_ext. When >0,
// ggml generates per-head ALiBi slopes internally on the pattern
// m_h = 2^(-max_bias * h / n_head) and adds them to the attention
// scores. Returns 0 in Standard / PrimePe modes (no ALiBi).
float prime_pe_alibi_max_bias(Config::PeMode pe_mode, float pe_alpha);

// Human-readable one-liner for the CLI (e.g., "PrimePE-RoPE (tier=1, α=0.17)").
std::string prime_pe_describe(Config::PeMode pe_mode, float pe_alpha, int pe_tier);

} // namespace sp::engine
