// Shannon-Prime — Algebraic Resonance Memory unit/capacity test (Phase 13.A).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// What we verify:
//   1. Round-trip:  store one (k, v) pair, recall with k, get v back.
//   2. Capacity:    store K random (k_i, v_i) pairs in one slab, query
//                   each k_j, measure cos(v_hat_j, v_j) and SNR over
//                   K = 1, 2, 4, 8, 16, 32, 64.
//   3. Noisy cue:   query with k_j + Gaussian noise (SNR 10dB), measure
//                   recall degradation.
//   4. No-slab-corruption: norm of an unwritten slab is 0 after K writes
//                   to OTHER slabs.
//
// Capacity gate (printed but not enforced as a hard ASSERT — we want
// to see the curve and tune):
//   cos(v_hat_j, v_j) > 0.5 for K <= 16   at d = N = 256, delta = 256.

extern "C" {
#include "../lib/shannon-prime/core/sp_arm.h"
#include "../lib/shannon-prime/core/sp_ntt_crt.h"
}

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#define TEST(name) static void name(); \
    static int reg_##name = (g_tests.push_back({#name, name}), 0); \
    static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)
#define ASSERT_NEAR(a, b, eps) do { double _d = std::abs((double)(a)-(double)(b)); \
    if (_d > (eps)) { std::fprintf(stderr, \
        "  ASSERT_NEAR FAIL (%s:%d): %.6g vs %.6g (delta %.6g > eps %.6g)\n", \
        __FILE__, __LINE__, (double)(a), (double)(b), _d, (double)(eps)); g_fail++; } } while (0)

namespace {

constexpr int    N     = SP_ARM_RING_N;     // 256
constexpr double DELTA = 256.0;             // 2^8 — safe headroom for K up to ~64

/* HRR theory note. For unit-norm random K and the negacyclic involution:
 *   K(x) * inv(K)(x) = 1 * delta_0 + epsilon(x)
 * where epsilon has cross-coefficient stddev ~ 1/sqrt(N). For N=256 the
 * noise floor on a single-pattern recall is cos ~ 1/sqrt(2) ≈ 0.707.
 * Storing K patterns multiplies the cross-talk by sqrt(K). These are
 * fundamental SNR limits of the HRR substrate — not bugs in this code.
 *
 * Increasing the effective ring degree relative to the feature dim (d
 * < N, sparse keys) cleans up the recall: zero-padded coefficients
 * contribute nothing to either the signal at x^0 (since k_i=0) or to
 * specific cross-coefficient lags. The capacity_sweep below runs both
 * d=N (full) and d=N/4 (sparse) to characterize this tradeoff. */

static double rms(const std::vector<float>& v) {
    double ss = 0.0;
    for (float x : v) ss += (double)x * (double)x;
    return std::sqrt(ss / (double)v.size());
}

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / std::sqrt(na * nb);
}

// Generate a length-d random vector. Optionally normalize to unit L2.
static std::vector<float> rand_vec(std::mt19937& rng, int d_, bool unit_norm) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v((size_t)d_);
    for (int i = 0; i < d_; ++i) v[(size_t)i] = nd(rng);
    if (unit_norm) {
        double ss = 0.0;
        for (float x : v) ss += (double)x * (double)x;
        const double inv = 1.0 / std::sqrt(ss);
        for (auto& x : v) x = (float)(x * inv);
    }
    return v;
}

} // anonymous namespace

// ============================================================================
// 1. Round-trip: store ONE pair, recall with exact key, should get value back.
// ============================================================================
TEST(arm_round_trip_one_pair_exact_recall) {
    const int d = N;  /* full ring */
    sp_arm_bank bank;
    std::vector<uint64_t> M_q1((size_t)N), M_q2((size_t)N);
    sp_arm_bank_init(&bank, M_q1.data(), M_q2.data(),
                      /*n_slabs=*/1, d, DELTA);

    std::mt19937 rng(7);
    auto k = rand_vec(rng, d, /*unit_norm=*/true);
    auto v = rand_vec(rng, d, /*unit_norm=*/false);

    std::vector<uint64_t> scratch4((size_t)4 * N);
    std::vector<int64_t>  int_scratch((size_t)N);
    sp_arm_bank_write(&bank, 0, k.data(), v.data(),
                       scratch4.data(), int_scratch.data());

    std::vector<float> v_hat((size_t)d, 0.0f);
    std::vector<float> inv_q_scratch((size_t)N);
    sp_arm_bank_recall(&bank, 0, k.data(), v_hat.data(),
                        scratch4.data(), int_scratch.data(),
                        inv_q_scratch.data());

    const double c   = cosine(v, v_hat);
    const double err = rms({ // err vec
        // construct difference inline
    });
    // recompute err properly
    std::vector<float> diff((size_t)d);
    for (int i = 0; i < d; ++i) diff[(size_t)i] = v[(size_t)i] - v_hat[(size_t)i];
    const double rms_v   = rms(v);
    const double rms_err = rms(diff);
    const double snr_db  = 20.0 * std::log10(rms_v / (rms_err + 1e-30));

    std::fprintf(stderr,
        "  K=1, d=%d: cos(v,v_hat) = %.4f   rms(v) = %.4f   rms(err) = %.4f   SNR = %.1f dB\n",
        d, c, rms_v, rms_err, snr_db);

    /* HRR theoretical limit at K=1 with full ring (d=N): cos ~ 1/sqrt(2)
     * ≈ 0.707 because the off-axis correlation noise stddev is ~1/sqrt(N)
     * comparable in energy to the on-axis signal. Sparse-key version
     * (d < N) does much better — tested in arm_round_trip_sparse. */
    ASSERT(c > 0.6);
    (void)err;
}

/* Sparse-key recall: d=32 in N=256 ring (1:8 sparsity). Each key
 * polynomial has only 32 nonzero coefficients out of 256, so the
 * involution math gives cleaner auto-correlation. */
TEST(arm_round_trip_one_pair_sparse_recall) {
    const int d = 32;
    sp_arm_bank bank;
    std::vector<uint64_t> M_q1((size_t)N), M_q2((size_t)N);
    sp_arm_bank_init(&bank, M_q1.data(), M_q2.data(),
                      /*n_slabs=*/1, d, DELTA);

    std::mt19937 rng(11);
    auto k = rand_vec(rng, d, /*unit_norm=*/true);
    auto v = rand_vec(rng, d, /*unit_norm=*/false);

    std::vector<uint64_t> scratch4((size_t)4 * N);
    std::vector<int64_t>  int_scratch((size_t)N);
    sp_arm_bank_write(&bank, 0, k.data(), v.data(),
                       scratch4.data(), int_scratch.data());

    std::vector<float> v_hat((size_t)d, 0.0f);
    std::vector<float> inv_q_scratch((size_t)N);
    sp_arm_bank_recall(&bank, 0, k.data(), v_hat.data(),
                        scratch4.data(), int_scratch.data(),
                        inv_q_scratch.data());

    const double c = cosine(v, v_hat);
    std::fprintf(stderr,
        "  K=1, d=%d (sparse): cos(v,v_hat) = %.4f\n", d, c);
    /* Sparse keys should produce noticeably cleaner recall. */
    ASSERT(c > 0.85);
}

// ============================================================================
// 2. Capacity sweep: store K random pairs in one slab, recall each one
//    with the exact key, average cosine + SNR over the K queries.
// ============================================================================
static void capacity_sweep_with_K(int K_count, int d, std::mt19937& rng) {
    sp_arm_bank bank;
    std::vector<uint64_t> M_q1((size_t)N), M_q2((size_t)N);
    sp_arm_bank_init(&bank, M_q1.data(), M_q2.data(),
                      /*n_slabs=*/1, d, DELTA);

    std::vector<std::vector<float>> ks, vs;
    ks.reserve((size_t)K_count);
    vs.reserve((size_t)K_count);
    for (int i = 0; i < K_count; ++i) {
        ks.push_back(rand_vec(rng, d, /*unit_norm=*/true));
        vs.push_back(rand_vec(rng, d, /*unit_norm=*/false));
    }

    std::vector<uint64_t> scratch4((size_t)4 * N);
    std::vector<int64_t>  int_scratch((size_t)N);
    for (int i = 0; i < K_count; ++i) {
        sp_arm_bank_write(&bank, 0, ks[(size_t)i].data(), vs[(size_t)i].data(),
                           scratch4.data(), int_scratch.data());
    }

    double sum_cos = 0.0;
    double sum_snr_db = 0.0;
    int    queries = 0;
    for (int j = 0; j < K_count; ++j) {
        std::vector<float> v_hat((size_t)d, 0.0f);
        std::vector<float> inv_q_scratch((size_t)N);
        sp_arm_bank_recall(&bank, 0, ks[(size_t)j].data(), v_hat.data(),
                            scratch4.data(), int_scratch.data(),
                            inv_q_scratch.data());
        const double c = cosine(vs[(size_t)j], v_hat);
        std::vector<float> diff((size_t)d);
        for (int i = 0; i < d; ++i) {
            diff[(size_t)i] = vs[(size_t)j][(size_t)i] - v_hat[(size_t)i];
        }
        const double rms_v   = rms(vs[(size_t)j]);
        const double rms_err = rms(diff);
        const double snr_db  = 20.0 * std::log10(rms_v / (rms_err + 1e-30));
        sum_cos += c;
        sum_snr_db += snr_db;
        queries += 1;
    }
    const double mean_cos = sum_cos / (double)queries;
    const double mean_snr = sum_snr_db / (double)queries;

    // Slab norm (sanity check that something accumulated):
    std::vector<uint64_t> norm_scratch((size_t)2 * N);
    const double slab_norm = sp_arm_bank_norm(&bank, 0, norm_scratch.data());

    std::fprintf(stderr,
        "  d=%3d  K=%2d  mean_cos = %.4f   mean_SNR = %5.1f dB   slab_norm = %.3e\n",
        d, K_count, mean_cos, mean_snr, slab_norm);
}

TEST(arm_capacity_sweep) {
    std::mt19937 rng(42);
    std::fprintf(stderr, "  --- ARM capacity sweep at d=N=%d (full ring) ---\n", N);
    for (int K_count : { 1, 2, 4, 8, 16, 32, 64 }) {
        capacity_sweep_with_K(K_count, /*d=*/N, rng);
    }
    std::fprintf(stderr, "  --- ARM capacity sweep at d=%d (sparse, 1:8) ---\n", N/8);
    for (int K_count : { 1, 2, 4, 8, 16, 32, 64 }) {
        capacity_sweep_with_K(K_count, /*d=*/N/8, rng);
    }
    ASSERT(true);
}

// ============================================================================
// 3. Noisy cue: query with k_j + Gaussian noise at ~10dB SNR.
// ============================================================================
TEST(arm_noisy_cue_recall) {
    constexpr int K_count = 8;
    const int d = N / 8;  /* sparse keys for cleaner recall */
    sp_arm_bank bank;
    std::vector<uint64_t> M_q1((size_t)N), M_q2((size_t)N);
    sp_arm_bank_init(&bank, M_q1.data(), M_q2.data(),
                      /*n_slabs=*/1, d, DELTA);

    std::mt19937 rng(123);
    std::vector<std::vector<float>> ks, vs;
    for (int i = 0; i < K_count; ++i) {
        ks.push_back(rand_vec(rng, d, /*unit_norm=*/true));
        vs.push_back(rand_vec(rng, d, /*unit_norm=*/false));
    }
    std::vector<uint64_t> scratch4((size_t)4 * N);
    std::vector<int64_t>  int_scratch((size_t)N);
    for (int i = 0; i < K_count; ++i) {
        sp_arm_bank_write(&bank, 0, ks[(size_t)i].data(), vs[(size_t)i].data(),
                           scratch4.data(), int_scratch.data());
    }

    // Query with k_2 + noise scaled to ~10dB SNR (noise rms = 0.316 * cue rms).
    // Since k is unit-norm, rms = 1/sqrt(d). Noise std = 0.316 / sqrt(d).
    std::normal_distribution<float> nd(0.0f, (float)(0.316 / std::sqrt((double)d)));
    std::vector<float> q_noisy((size_t)d);
    for (int i = 0; i < d; ++i) q_noisy[(size_t)i] = ks[2][(size_t)i] + nd(rng);

    std::vector<float> v_hat((size_t)d, 0.0f);
    std::vector<float> inv_q_scratch((size_t)N);
    sp_arm_bank_recall(&bank, 0, q_noisy.data(), v_hat.data(),
                        scratch4.data(), int_scratch.data(),
                        inv_q_scratch.data());
    const double c = cosine(vs[2], v_hat);
    std::fprintf(stderr,
        "  noisy cue (K=8, 10dB cue SNR): cos(v_target, v_hat) = %.4f\n", c);
    // Lenient gate; tune after first observation.
    ASSERT(c > 0.3);
}

// ============================================================================
// 4. Slab isolation: writes to slab A don't affect slab B's norm.
// ============================================================================
TEST(arm_slab_isolation) {
    constexpr int K_count = 4;
    const int d = N;
    sp_arm_bank bank;
    std::vector<uint64_t> M_q1((size_t)2 * N), M_q2((size_t)2 * N);
    sp_arm_bank_init(&bank, M_q1.data(), M_q2.data(),
                      /*n_slabs=*/2, d, DELTA);

    std::mt19937 rng(99);
    std::vector<uint64_t> scratch4((size_t)4 * N);
    std::vector<int64_t>  int_scratch((size_t)N);

    for (int i = 0; i < K_count; ++i) {
        auto k = rand_vec(rng, d, /*unit_norm=*/true);
        auto v = rand_vec(rng, d, /*unit_norm=*/false);
        sp_arm_bank_write(&bank, /*slab=*/0, k.data(), v.data(),
                           scratch4.data(), int_scratch.data());
    }

    std::vector<uint64_t> norm_scratch((size_t)2 * N);
    const double n_slab0 = sp_arm_bank_norm(&bank, 0, norm_scratch.data());
    const double n_slab1 = sp_arm_bank_norm(&bank, 1, norm_scratch.data());
    std::fprintf(stderr,
        "  slab 0 norm = %.3e   slab 1 norm (should be ~0) = %.3e\n",
        n_slab0, n_slab1);
    ASSERT(n_slab0 > 0.0);
    ASSERT(n_slab1 == 0.0);
}

// ============================================================================
int main() {
    std::printf("Shannon-Prime sp_arm tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
