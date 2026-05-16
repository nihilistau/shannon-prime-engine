// Shannon-Prime Engine — Sampler (Phase 2 prep).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Standard sampling primitives: top-k, top-p (nucleus), temperature,
// repetition penalty, mirostat. Standalone — operates on a logits array.
// No ggml dependency.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

struct sp_sampler_params {
    float temperature      = 1.0f;
    int   top_k            = 40;
    float top_p            = 0.95f;
    float repetition_penalty = 1.1f;
    int   repetition_window = 64;
    uint64_t seed          = 0;     // 0 = deterministic-seedless
};

// Apply temperature in place: logits[i] /= T (T > 0).
void sp_sampler_apply_temperature(float* logits, int n, float T);

// Top-k filter: zero out (set to -inf) all but the top-k logits.
void sp_sampler_apply_top_k(float* logits, int n, int k);

// Top-p (nucleus) filter: zero out logits whose cumulative softmax
// probability mass exceeds p.
void sp_sampler_apply_top_p(float* logits, int n, float p);

// Repetition penalty: divide logits at recently-emitted token positions
// by `penalty`.
void sp_sampler_apply_repetition_penalty(float* logits,
                                           const std::vector<int32_t>& recent,
                                           float penalty);

// Softmax into a probability buffer (sized n; reused across calls).
void sp_sampler_softmax(const float* logits, int n, float* probs_out);

// Sample a token id from `probs` using a 64-bit Xorshift PRNG seeded
// by `seed_state` (mutated in place). probs must sum to 1.
int32_t sp_sampler_sample(const float* probs, int n, uint64_t* seed_state);

// One-shot: applies all filters then samples a token.
int32_t sp_sampler_step(float* logits, int n,
                         const std::vector<int32_t>& recent,
                         const sp_sampler_params& params,
                         uint64_t* seed_state);

}  // namespace sp::engine
