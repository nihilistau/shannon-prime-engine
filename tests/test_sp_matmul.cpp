// Shannon-Prime Engine — sp_matmul unit tests (Phase 2.0).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify that the O_K-native matmul produces results matching fp32
// reference matmul to a tolerance set by the per-tensor scale.

#include "../src/sp_matmul.h"
#include "../src/sp_ok_tensor.h"
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

// fp32 reference matmul: Y[i,j] = sum_k W[i,k] * X[k,j]
static void fp32_matmul(const float* W, int M, int K,
                         const float* X, int N,
                         float* Y) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += (double)W[i * K + k] * (double)X[k * N + j];
            }
            Y[i * N + j] = (float)s;
        }
    }
}

// =========================================================================
// O_K @ O_K → O_K — verify decoded output matches fp32 matmul of decoded inputs.
// =========================================================================

TEST(matmul_ok_2x3_times_3x4_matches_fp32_reference) {
    // W: 2x3, X: 3x4, Y: 2x4. Random fp32 inputs.
    constexpr int M = 2, K = 3, N = 4;
    std::vector<float> w_fp32(M * K), x_fp32(K * N), y_ref(M * N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : w_fp32) v = d(rng);
    for (auto& v : x_fp32) v = d(rng);
    fp32_matmul(w_fp32.data(), M, K, x_fp32.data(), N, y_ref.data());

    // Encode W and X into O_K with a chosen scale (kept modest so int64
    // accumulator stays well within range).
    sp_ok_arena arena(16 * 1024);
    sp_ok_tensor W, X, Y;
    int64_t w_shape[4] = { K, M, 1, 1 };  // shape[0]=K innermost, shape[1]=M
    int64_t x_shape[4] = { N, K, 1, 1 };
    int64_t y_shape[4] = { N, M, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(W, w_fp32.data(), 2, w_shape, 1 << 12, arena));
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 12, arena));
    Y.reset(2, y_shape);
    ASSERT(arena.alloc_tensor(Y));
    ASSERT(sp_matmul_ok(W, X, Y));

    // Decode Y and compare to fp32 reference.
    std::vector<float> y_decoded(M * N);
    sp_ok_decode_to_fp32(y_decoded.data(), Y);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ASSERT_NEAR(y_decoded[i * N + j], y_ref[i * N + j], 0.01);
        }
    }
}

TEST(matmul_ok_8x16_times_16x4_larger_dims) {
    constexpr int M = 8, K = 16, N = 4;
    std::vector<float> w_fp32(M * K), x_fp32(K * N), y_ref(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-0.3f, 0.3f);
    for (auto& v : w_fp32) v = d(rng);
    for (auto& v : x_fp32) v = d(rng);
    fp32_matmul(w_fp32.data(), M, K, x_fp32.data(), N, y_ref.data());

    sp_ok_arena arena(128 * 1024);
    sp_ok_tensor W, X, Y;
    int64_t w_shape[4] = { K, M, 1, 1 };
    int64_t x_shape[4] = { N, K, 1, 1 };
    int64_t y_shape[4] = { N, M, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(W, w_fp32.data(), 2, w_shape, 1 << 11, arena));
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 11, arena));
    Y.reset(2, y_shape);
    ASSERT(arena.alloc_tensor(Y));
    ASSERT(sp_matmul_ok(W, X, Y));

    std::vector<float> y_decoded(M * N);
    sp_ok_decode_to_fp32(y_decoded.data(), Y);
    int near = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(y_decoded[i] - y_ref[i]) < 0.02) ++near;
    }
    ASSERT(near >= M * N - 1);  // allow one outlier from rounding accumulation
}

// =========================================================================
// O_K @ O_K → fp32 bridge — verify decoded fp32 matches reference.
// =========================================================================

TEST(matmul_ok_to_fp32_bridge_matches_reference) {
    constexpr int M = 4, K = 6, N = 3;
    std::vector<float> w_fp32(M * K), x_fp32(K * N), y_ref(M * N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-0.4f, 0.4f);
    for (auto& v : w_fp32) v = d(rng);
    for (auto& v : x_fp32) v = d(rng);
    fp32_matmul(w_fp32.data(), M, K, x_fp32.data(), N, y_ref.data());

    sp_ok_arena arena(32 * 1024);
    sp_ok_tensor W, X;
    int64_t w_shape[4] = { K, M, 1, 1 };
    int64_t x_shape[4] = { N, K, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(W, w_fp32.data(), 2, w_shape, 1 << 12, arena));
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 12, arena));

    std::vector<float> y_fp32(M * N);
    ASSERT(sp_matmul_ok_to_fp32(W, X, y_fp32.data(), M, N));
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(y_fp32[i], y_ref[i], 0.005);
    }
}

// =========================================================================
// fp32 × O_K → O_K — caller has fp32 activations, weights stay O_K.
// =========================================================================

TEST(matmul_fp32_input_to_ok_matches_reference) {
    constexpr int M = 3, K = 4, N = 2;
    std::vector<float> w_fp32(M * K), x_fp32(K * N), y_ref(M * N);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-0.4f, 0.4f);
    for (auto& v : w_fp32) v = d(rng);
    for (auto& v : x_fp32) v = d(rng);
    fp32_matmul(w_fp32.data(), M, K, x_fp32.data(), N, y_ref.data());

    sp_ok_arena arena(32 * 1024);
    sp_ok_tensor W, Y;
    int64_t w_shape[4] = { K, M, 1, 1 };
    int64_t y_shape[4] = { N, M, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(W, w_fp32.data(), 2, w_shape, 1 << 12, arena));
    Y.reset(2, y_shape);
    ASSERT(arena.alloc_tensor(Y));
    ASSERT(sp_matmul_fp32_input_to_ok(W, x_fp32.data(), K, N, Y));

    std::vector<float> y_decoded(M * N);
    sp_ok_decode_to_fp32(y_decoded.data(), Y);
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(y_decoded[i], y_ref[i], 0.01);
    }
}

// =========================================================================
// Frobenius pre-shimmed weights — verify the cancellation works in matmul.
// =========================================================================

TEST(matmul_after_frobenius_shim_preserves_product) {
    // Encode W, apply phi_41^8 (Theorem 4 stress), then matmul. Decoded
    // output should equal the un-shimmed fp32 reference matmul.
    constexpr int M = 4, K = 4, N = 2;
    std::vector<float> w_fp32(M * K), x_fp32(K * N), y_ref(M * N);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> d(-0.3f, 0.3f);
    for (auto& v : w_fp32) v = d(rng);
    for (auto& v : x_fp32) v = d(rng);
    fp32_matmul(w_fp32.data(), M, K, x_fp32.data(), N, y_ref.data());

    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor W, X, Y;
    int64_t w_shape[4] = { K, M, 1, 1 };
    int64_t x_shape[4] = { N, K, 1, 1 };
    int64_t y_shape[4] = { N, M, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(W, w_fp32.data(), 2, w_shape, 1 << 10, arena));
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 10, arena));

    // Apply Theorem 4 stress: phi_41^8 on W only (X stays vanilla — same
    // as Paper D Config B where activations are not transformed).
    sp_ok_encode_apply_frobenius_quant(W, 41, 8);

    Y.reset(2, y_shape);
    ASSERT(arena.alloc_tensor(Y));
    ASSERT(sp_matmul_ok(W, X, Y));

    std::vector<float> y_decoded(M * N);
    sp_ok_decode_to_fp32(y_decoded.data(), Y);

    // The Frobenius factor cancels in decode because Y.frobenius_scale
    // = W.frobenius_scale * X.frobenius_scale, and Y.scale_recip
    // = W.scale_recip * X.scale_recip. So decoded Y[i,j] = sum_k w*x.
    int near = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(y_decoded[i] - y_ref[i]) < 0.05) ++near;
    }
    ASSERT(near >= M * N - 1);
}

// =========================================================================
// Driver
// =========================================================================

int main() {
    std::printf("Shannon-Prime sp_matmul unit tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
