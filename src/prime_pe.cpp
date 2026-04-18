// Shannon-Prime Engine — PrimePE-RoPE-ALiBi
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// The lattice math here is the canonical three-tier allocation from
// `prime_rope.h` (Prime Harmonic RoPE) — local / mid / long tiers of
// composite (or prime) numbers, log-linearly mapped into the geometric
// frequency envelope the pre-trained RoPE expects. Factors are then
// blended with the identity at `pe_alpha`, so alpha=0 → pure geometric
// RoPE and alpha=1 → full lattice replacement.

#include "prime_pe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sp::engine {

// ── Sieve / pools ────────────────────────────────────────────────────

static std::vector<int> sieve_primes_up_to(int n) {
    std::vector<char> is_p((size_t)n + 1, 1);
    if (n >= 0) is_p[0] = 0;
    if (n >= 1) is_p[1] = 0;
    for (int i = 2; (long long)i * i <= n; ++i) {
        if (is_p[(size_t)i]) {
            for (long long j = (long long)i * i; j <= n; j += i) is_p[(size_t)j] = 0;
        }
    }
    std::vector<int> out;
    out.reserve((size_t)n / 4 + 8);
    for (int i = 2; i <= n; ++i) if (is_p[(size_t)i]) out.push_back(i);
    return out;
}

static std::vector<int> get_composites(int lo, int hi) {
    auto primes = sieve_primes_up_to(hi);
    std::vector<char> is_p((size_t)hi + 1, 0);
    for (int p : primes) is_p[(size_t)p] = 1;
    std::vector<int> comp;
    for (int n = std::max(lo, 4); n <= hi; ++n) {
        if (!is_p[(size_t)n]) comp.push_back(n);
    }
    return comp;
}

// Pick n_freqs values from pool, evenly spaced (cycle if pool too small).
static std::vector<int> pick_from_pool(const std::vector<int>& pool, int n_freqs) {
    std::vector<int> picked;
    if (pool.empty() || n_freqs <= 0) return picked;
    if ((int)pool.size() <= n_freqs) {
        picked = pool;
        while ((int)picked.size() < n_freqs) {
            picked.push_back(pool[picked.size() % pool.size()]);
        }
    } else {
        float step = (float)pool.size() / (float)n_freqs;
        for (int i = 0; i < n_freqs; ++i) {
            picked.push_back(pool[(size_t)((int)(i * step))]);
        }
    }
    return picked;
}

// ── Core factor computation (from prime_rope.h) ──────────────────────

static std::vector<float> compute_raw_factors(int n_dims, float freq_base, bool use_primes) {
    const int n_freqs = n_dims / 2;
    if (n_freqs <= 0) return {};

    // Three-tier split: 0.34 / 0.33 / 0.33.
    const int n_local = (int)(n_freqs * 0.34f);
    const int n_mid   = (int)(n_freqs * 0.33f);
    const int n_long  = n_freqs - n_local - n_mid;

    const int local_lo = 2,    local_hi = 64;
    const int mid_lo   = 64,   mid_hi   = 1024;
    const int long_lo  = 1024, long_hi  = 8192;

    std::vector<int> pool_local, pool_mid, pool_long;
    if (!use_primes) {
        pool_local = get_composites(local_lo, local_hi);
        pool_mid   = get_composites(mid_lo,   mid_hi);
        pool_long  = get_composites(long_lo,  long_hi);
    } else {
        auto all_primes = sieve_primes_up_to(long_hi);
        for (int p : all_primes) {
            if (p >= local_lo && p <= local_hi) pool_local.push_back(p);
            if (p >= mid_lo   && p <= mid_hi)   pool_mid.push_back(p);
            if (p >= long_lo  && p <= long_hi)  pool_long.push_back(p);
        }
    }

    auto picked_local = pick_from_pool(pool_local, n_local);
    auto picked_mid   = pick_from_pool(pool_mid,   n_mid);
    auto picked_long  = pick_from_pool(pool_long,  n_long);

    std::vector<int> all_picked;
    all_picked.reserve((size_t)n_freqs);
    all_picked.insert(all_picked.end(), picked_local.begin(), picked_local.end());
    all_picked.insert(all_picked.end(), picked_mid.begin(),   picked_mid.end());
    all_picked.insert(all_picked.end(), picked_long.begin(),  picked_long.end());

    // Geometric envelope the pretrained model expects.
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    std::vector<float> geo_freqs((size_t)n_freqs);
    {
        float gf = 1.0f;
        for (int i = 0; i < n_freqs; ++i) { geo_freqs[(size_t)i] = gf; gf *= theta_scale; }
    }
    const float geo_max = geo_freqs[0];
    const float geo_min = geo_freqs[(size_t)(n_freqs - 1)];

    // Raw composite/prime angular freqs: 2π/n. Sort descending so the
    // smallest numbers (= highest freq) land in the low-dim slots.
    std::vector<float> comp_freqs((size_t)n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        comp_freqs[(size_t)i] = 2.0f * (float)M_PI / (float)all_picked[(size_t)i];
    }
    std::sort(comp_freqs.begin(), comp_freqs.end(), std::greater<float>());
    const float comp_max = comp_freqs[0];
    const float comp_min = comp_freqs[(size_t)(n_freqs - 1)];

    // Log-linear map comp_freqs → geometric range.
    std::vector<float> norm_freqs((size_t)n_freqs);
    const float log_comp_min = logf(comp_min);
    const float log_comp_max = logf(comp_max);
    const float log_geo_min  = logf(geo_min);
    const float log_geo_max  = logf(geo_max);
    const float log_span     = log_comp_max - log_comp_min;
    for (int i = 0; i < n_freqs; ++i) {
        float t = (log_span > 1e-10f)
            ? (logf(comp_freqs[(size_t)i]) - log_comp_min) / log_span
            : (float)i / (float)(n_freqs - 1);
        norm_freqs[(size_t)i] = expf(log_geo_min + t * (log_geo_max - log_geo_min));
    }

    // freq_factors = geometric / normalized. These divide the pretrained
    // freqs inside ggml_rope_ext — small perturbations around 1.0 that
    // replace the geometric spacing with arithmetic-lattice spacing.
    std::vector<float> factors((size_t)n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        factors[(size_t)i] = geo_freqs[(size_t)i] / norm_freqs[(size_t)i];
    }
    return factors;
}

// ── Public API ───────────────────────────────────────────────────────

std::vector<float> prime_pe_freq_factors(Config::PeMode pe_mode,
                                         float  pe_alpha,
                                         int    pe_tier,
                                         int    n_rot,
                                         float  freq_base) {
    if (n_rot <= 0) return {};
    if (pe_mode == Config::PeMode::Standard || pe_mode == Config::PeMode::AlibiOnly) return {};
    if (pe_alpha <= 0.0f) return {};

    const bool use_primes = (pe_tier == 1);
    std::vector<float> raw = compute_raw_factors(n_rot, freq_base, use_primes);
    if (raw.empty()) return {};

    // Blend with identity so alpha=0 is a no-op and alpha=1 is full
    // lattice substitution. Factors from compute_raw_factors are already
    // close to 1.0, so small alpha gives small perturbations.
    const float a = pe_alpha;
    for (auto& f : raw) f = (1.0f - a) + a * f;
    return raw;
}

float prime_pe_alibi_max_bias(Config::PeMode pe_mode, float pe_alpha) {
    if (pe_mode == Config::PeMode::AlibiOnly || pe_mode == Config::PeMode::PrimePeAlibi) {
        return 8.0f * (pe_alpha > 0.0f ? pe_alpha : 0.0f);
    }
    return 0.0f;
}

std::string prime_pe_describe(Config::PeMode pe_mode, float pe_alpha, int pe_tier) {
    char buf[96];
    const char* name = "Standard";
    switch (pe_mode) {
        case Config::PeMode::Standard:     name = "Standard";             break;
        case Config::PeMode::PrimePe:      name = "PrimePE-RoPE";         break;
        case Config::PeMode::PrimePeAlibi: name = "PrimePE-RoPE-ALiBi";   break;
        case Config::PeMode::AlibiOnly:    name = "ALiBi-only";           break;
    }
    if (pe_mode == Config::PeMode::Standard) {
        std::snprintf(buf, sizeof(buf), "%s", name);
    } else {
        const char* tier_name = (pe_tier == 1) ? "prime-tiered" : "composite-tiered";
        std::snprintf(buf, sizeof(buf), "%s (%s, alpha=%.2f)", name, tier_name, pe_alpha);
    }
    return std::string(buf);
}

} // namespace sp::engine
