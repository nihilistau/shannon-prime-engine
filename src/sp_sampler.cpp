// Shannon-Prime Engine — Sampler (Phase 2 prep, impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_sampler.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace sp::engine {

// ---------- Xorshift64 PRNG -----------------------------------------------
static inline uint64_t xs64_next(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}
static inline uint64_t xs64_seed(uint64_t s) {
    // splitmix-style mix to avoid pathological zero seed
    if (s == 0) s = 0x9E3779B97F4A7C15ULL;
    s ^= s >> 30;  s *= 0xBF58476D1CE4E5B9ULL;
    s ^= s >> 27;  s *= 0x94D049BB133111EBULL;
    s ^= s >> 31;
    return s;
}

// ---------- Temperature ----------------------------------------------------
void sp_sampler_apply_temperature(float* logits, int n, float T) {
    if (T == 1.0f || T <= 0.0f) return;
    const float inv = 1.0f / T;
    for (int i = 0; i < n; ++i) logits[i] *= inv;
}

// ---------- Top-k ----------------------------------------------------------
void sp_sampler_apply_top_k(float* logits, int n, int k) {
    if (k <= 0 || k >= n) return;
    // Partial-sort indices by descending logit; cut at k.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    const float keep = logits[idx[k - 1]];
    for (int i = 0; i < n; ++i) {
        if (logits[i] < keep) logits[i] = -FLT_MAX;
    }
}

// ---------- Top-p (nucleus) -----------------------------------------------
void sp_sampler_apply_top_p(float* logits, int n, float p) {
    if (p >= 1.0f || p <= 0.0f) return;
    // Convert logits → softmax probs (numerically stable).
    std::vector<float> probs(n);
    sp_sampler_softmax(logits, n, probs.data());

    // Index sort by descending probability.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    float cum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cum += probs[idx[i]];
        if (cum >= p) { cutoff = i + 1; break; }
    }
    // Mask out everything past the cutoff.
    std::unordered_set<int> keep(idx.begin(), idx.begin() + cutoff);
    for (int i = 0; i < n; ++i) {
        if (keep.find(i) == keep.end()) logits[i] = -FLT_MAX;
    }
}

// ---------- Repetition penalty --------------------------------------------
void sp_sampler_apply_repetition_penalty(float* logits,
                                           const std::vector<int32_t>& recent,
                                           float penalty) {
    if (penalty == 1.0f) return;
    for (int32_t t : recent) {
        if (t < 0) continue;
        float v = logits[t];
        // If logit > 0, divide; if logit < 0, multiply (per HF convention).
        logits[t] = (v > 0.0f) ? (v / penalty) : (v * penalty);
    }
}

// ---------- Softmax --------------------------------------------------------
void sp_sampler_softmax(const float* logits, int n, float* probs_out) {
    // Find max for numerical stability.
    float mx = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float e = std::exp(logits[i] - mx);
        probs_out[i] = e;
        sum += e;
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) probs_out[i] *= inv;
}

// ---------- Sampling -------------------------------------------------------
int32_t sp_sampler_sample(const float* probs, int n, uint64_t* seed_state) {
    uint64_t r = xs64_next(seed_state);
    // Uniform in [0, 1).
    double u = (double)r / (double)UINT64_MAX;
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        if (u < cum) return (int32_t)i;
    }
    return n - 1;
}

int32_t sp_sampler_step(float* logits, int n,
                         const std::vector<int32_t>& recent,
                         const sp_sampler_params& params,
                         uint64_t* seed_state) {
    sp_sampler_apply_repetition_penalty(logits, recent, params.repetition_penalty);
    sp_sampler_apply_temperature(logits, n, params.temperature);
    sp_sampler_apply_top_k(logits, n, params.top_k);
    sp_sampler_apply_top_p(logits, n, params.top_p);

    std::vector<float> probs(n);
    sp_sampler_softmax(logits, n, probs.data());

    if (!seed_state || *seed_state == 0) {
        uint64_t local_seed = xs64_seed(params.seed ? params.seed : 0xC0DECAFEULL);
        if (seed_state) *seed_state = local_seed;
        return sp_sampler_sample(probs.data(), n, seed_state ? seed_state : &local_seed);
    }
    return sp_sampler_sample(probs.data(), n, seed_state);
}

}  // namespace sp::engine
