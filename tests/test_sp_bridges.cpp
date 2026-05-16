// Shannon-Prime Engine — Phase 2.1 bridge unit tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "../src/sp_bridges.h"
#include "../src/sp_ok_encode.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)
#define ASSERT_NEAR(a, b, eps) do { double _d = std::abs((double)(a)-(double)(b)); \
    if (_d > (eps)) { std::fprintf(stderr, "  ASSERT_NEAR FAIL (%s:%d): %.6g vs %.6g (delta %.6g > eps %.6g)\n", \
        __FILE__, __LINE__, (double)(a), (double)(b), _d, (double)(eps)); g_fail++; } } while (0)

using namespace sp::engine;

// =========================================================================
// fp32 reference implementations
// =========================================================================

static void rmsnorm_fp32_ref(const float* x, const float* scale, int n,
                              float eps, float* out) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += (double)x[i] * (double)x[i];
    const double inv_rms = 1.0 / std::sqrt(sum_sq / (double)n + (double)eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (float)((double)x[i] * inv_rms * (double)scale[i]);
    }
}

static void softmax_fp32_ref(const float* x, int n, float* out) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float e = std::exp(x[i] - mx);
        out[i] = e;
        sum += (double)e;
    }
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

static inline float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

// =========================================================================
// sp_rmsnorm_native
// =========================================================================

TEST(rmsnorm_native_matches_fp32_reference) {
    constexpr int N = 64;
    std::vector<float> x_fp32(N), scale(N), ref(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) { x_fp32[i] = d(rng); scale[i] = 0.5f + d(rng) * 0.5f; }
    rmsnorm_fp32_ref(x_fp32.data(), scale.data(), N, 1e-6f, ref.data());

    sp_ok_arena arena(16 * 1024);
    sp_ok_tensor X, Y;
    int64_t shape[4] = { N, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 1, shape, 1 << 14, arena));
    Y.reset(1, shape);
    ASSERT(arena.alloc_tensor(Y));
    Y.scale_recip = 1 << 14;
    ASSERT(sp_rmsnorm_native(X, scale.data(), Y, 1e-6f, N, 1));
    ASSERT(Y.frobenius_scale == 1);  // scale-reset valve

    std::vector<float> got(N);
    sp_ok_decode_to_fp32(got.data(), Y);
    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(got[i], ref[i], 0.01);
    }
}

TEST(rmsnorm_native_resets_frobenius_scale) {
    // Even if the input came in with a large frobenius_scale (post-Config-B),
    // rmsnorm output must have frobenius_scale = 1 so downstream layers
    // re-establish their own.
    constexpr int N = 32;
    std::vector<float> x_fp32(N, 0.5f), scale(N, 1.0f);
    sp_ok_arena arena(8 * 1024);
    sp_ok_tensor X, Y;
    int64_t shape[4] = { N, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 1, shape, 1 << 12, arena));
    X.frobenius_scale = 41 * 41 * 41 * 41;  // simulate Config B post-shim state
    Y.reset(1, shape); ASSERT(arena.alloc_tensor(Y));
    Y.scale_recip = 1 << 12;
    Y.frobenius_scale = 999;  // junk; rmsnorm must clobber to 1
    ASSERT(sp_rmsnorm_native(X, scale.data(), Y, 1e-6f, N, 1));
    ASSERT(Y.frobenius_scale == 1);
}

TEST(rmsnorm_native_batched_two_tokens) {
    constexpr int N = 16, T = 2;
    std::vector<float> x_fp32(N * T), scale(N), ref(N * T);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : x_fp32) v = d(rng);
    for (auto& v : scale)  v = 1.0f + d(rng) * 0.2f;
    for (int t = 0; t < T; ++t) {
        rmsnorm_fp32_ref(x_fp32.data() + t * N, scale.data(), N, 1e-6f,
                          ref.data() + t * N);
    }
    sp_ok_arena arena(16 * 1024);
    sp_ok_tensor X, Y;
    int64_t shape[4] = { N, T, 1, 1 };  // shape[0]=N inner, shape[1]=T outer
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, shape, 1 << 13, arena));
    Y.reset(2, shape); ASSERT(arena.alloc_tensor(Y));
    Y.scale_recip = 1 << 13;
    ASSERT(sp_rmsnorm_native(X, scale.data(), Y, 1e-6f, N, T));

    std::vector<float> got(N * T);
    sp_ok_decode_to_fp32(got.data(), Y);
    for (int i = 0; i < N * T; ++i) {
        ASSERT_NEAR(got[i], ref[i], 0.01);
    }
}

// =========================================================================
// sp_softmax_bridge
// =========================================================================

TEST(softmax_bridge_matches_reference) {
    constexpr int N = 50;
    std::vector<float> x(N), ref(N), got(N);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    for (auto& v : x) v = d(rng);
    softmax_fp32_ref(x.data(), N, ref.data());
    sp_softmax_bridge(x.data(), N, got.data());
    for (int i = 0; i < N; ++i) ASSERT_NEAR(got[i], ref[i], 1e-6);
    double s = 0; for (auto v : got) s += v;
    ASSERT_NEAR(s, 1.0, 1e-5);
}

TEST(softmax_bridge_stable_with_large_logits) {
    // Without max-subtraction this would overflow exp.
    float logits[3] = { 1000.0f, 999.0f, 998.0f };
    float out[3];
    sp_softmax_bridge(logits, 3, out);
    ASSERT(out[0] > out[1]);
    ASSERT(out[1] > out[2]);
    ASSERT(std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]));
    ASSERT_NEAR(out[0] + out[1] + out[2], 1.0, 1e-5);
}

TEST(softmax_bridge_causal_zeros_invalid) {
    constexpr int N = 8;
    float logits[N] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float out[N];
    sp_softmax_bridge_causal(logits, N, 4, out);
    // First 4 elements form a valid softmax distribution
    double s = 0; for (int i = 0; i < 4; ++i) s += out[i];
    ASSERT_NEAR(s, 1.0, 1e-5);
    // Last 4 elements are masked to 0
    for (int i = 4; i < N; ++i) ASSERT(out[i] == 0.0f);
}

TEST(softmax_bridge_rows_per_row_normalization) {
    constexpr int N = 4, R = 3;
    std::vector<float> in(N * R), out(N * R);
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (auto& v : in) v = d(rng);
    sp_softmax_bridge_rows(in.data(), N, R, out.data());
    for (int r = 0; r < R; ++r) {
        double s = 0; for (int j = 0; j < N; ++j) s += out[r * N + j];
        ASSERT_NEAR(s, 1.0, 1e-5);
    }
}

// =========================================================================
// sp_silu_bridge
// =========================================================================

TEST(silu_bridge_matches_reference) {
    constexpr int N = 32;
    std::vector<float> gate(N), up(N), ref(N), got(N);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    for (auto& v : gate) v = d(rng);
    for (auto& v : up)   v = d(rng);
    for (int i = 0; i < N; ++i) ref[i] = silu_ref(gate[i]) * up[i];
    sp_silu_bridge(gate.data(), up.data(), N, got.data());
    for (int i = 0; i < N; ++i) ASSERT_NEAR(got[i], ref[i], 1e-5);
}

TEST(silu_bridge_stable_for_large_negative) {
    // For large negative x, silu(x) -> 0; must not produce nan from
    // numerator dominated by exp(positive).
    float gate[3] = { -100.0f, -50.0f, -1.0f };
    float up[3]   = { 1.0f, 1.0f, 1.0f };
    float out[3];
    sp_silu_bridge(gate, up, 3, out);
    for (int i = 0; i < 3; ++i) {
        ASSERT(std::isfinite(out[i]));
        if (i < 2) ASSERT(std::abs(out[i]) < 1e-10);  // ~0 for very negative
    }
}

TEST(silu_bridge_stable_for_large_positive) {
    // For large positive x, silu(x) -> x; must not overflow.
    float gate[3] = { 100.0f, 50.0f, 10.0f };
    float up[3]   = { 0.5f, 0.5f, 0.5f };
    float out[3];
    sp_silu_bridge(gate, up, 3, out);
    for (int i = 0; i < 3; ++i) ASSERT(std::isfinite(out[i]));
    ASSERT_NEAR(out[0], 50.0f, 0.001);  // silu(100) ≈ 100, times 0.5
    ASSERT_NEAR(out[1], 25.0f, 0.001);
}

TEST(silu_inplace_smoke) {
    float x[5] = { 0.5f, -0.5f, 1.0f, -1.0f, 0.0f };
    sp_silu_inplace(x, 5);
    ASSERT_NEAR(x[0], silu_ref(0.5f), 1e-6);
    ASSERT_NEAR(x[1], silu_ref(-0.5f), 1e-6);
    ASSERT_NEAR(x[2], silu_ref(1.0f), 1e-6);
    ASSERT_NEAR(x[3], silu_ref(-1.0f), 1e-6);
    ASSERT_NEAR(x[4], 0.0f, 1e-6);
}

// =========================================================================
// Driver
// =========================================================================

int main() {
    std::printf("Shannon-Prime sp_bridges unit tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
