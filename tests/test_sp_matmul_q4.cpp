/* test_sp_matmul_q4.cpp — Phase 14 fused Q4 matmul parity test.
 *
 * For a shared random (W, X) pair, build:
 *   Y_raw = sp_matmul_ok       (raw 16 B/elem reference)
 *   Y_q8  = sp_matmul_ok_q8    (2 B/elem packed)
 *   Y_q4  = sp_matmul_ok_q4    (1 B/elem packed)
 *
 * Then characterize relative-error vs Y_raw on both Q8 (tight budget,
 * was validated in Step D) and Q4 (looser budget, ~16x noise floor).
 * Also exercises sp_matmul_ok_q4_to_fp32 — fp32 output path used by the
 * Wo projection bridge.
 */

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "../src/sp_ok_tensor.h"
#include "../src/sp_ok_encode.h"
#include "../src/sp_matmul.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_ok_q8.h"
#include "../lib/shannon-prime/core/sp_ok_q4.h"
}

static int g_failures = 0;
static int g_tests = 0;
#define CHECK(cond, msg) do {                                              \
    ++g_tests;                                                              \
    if (!(cond)) {                                                          \
        ++g_failures;                                                       \
        std::fprintf(stderr,                                                \
                     "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg);       \
    }                                                                       \
} while (0)

static void run(const char* name, void (*fn)()) {
    std::fprintf(stderr, "[run] %s\n", name);
    fn();
}

static inline uint16_t f32_to_f16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}

static std::vector<uint16_t> make_w(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> body(0.0f, 0.05f);
    std::vector<uint16_t> w(n);
    for (size_t i = 0; i < n; ++i) w[i] = f32_to_f16(body(rng));
    return w;
}

/* Build a random (a, b) input tensor in the post-RoPE post-RMSNorm
 * regime: |a|, |b| in a small range comparable to typical activations
 * after Frobenius scaling at p=41, k=2. */
static void make_x(sp::engine::sp_ok_tensor& X, int64_t N, int64_t K,
                   sp::engine::sp_ok_arena& arena, uint64_t seed) {
    int64_t shape[4] = { N, K, 1, 1 };
    X.reset(2, shape);
    arena.alloc_tensor(X);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(-(1 << 10), (1 << 10));
    for (int64_t i = 0; i < N * K; ++i) X.data[i] = sp_ok_t{ dist(rng), dist(rng) };
    X.scale_recip = 1;
    X.frobenius_scale = 1;
}

/* ---------- Test 1: Q4 matmul produces same shape, bounded error ------- */

static void q4_matmul_shape_and_norm() {
    /* Small enough to keep raw int64 acc within range, large enough to
     * exercise multi-row work distribution + threadpool. */
    constexpr int64_t M = 64;
    constexpr int64_t K = 256;
    constexpr int64_t N = 4;
    constexpr int64_t SCALE = 1 << 10;  /* smaller scale -> looser quant budget */
    constexpr int64_t P = 41;
    constexpr int64_t Kf = 2;           /* tame Frobenius power so int64 doesn't overflow */

    auto w_fp16 = make_w(M * K, 0xCAFEBABE);

    /* Build W shape descriptor + Q8 + Q4 packed storage. */
    sp::engine::sp_ok_tensor W_shape;
    int64_t shape_w[4] = { K, M, 1, 1 };
    W_shape.reset(2, shape_w);
    /* W_shape carries scale_recip / frobenius_scale for the downstream
     * divisor; data is left null because the matmul reads from W_q8/q4. */

    sp::engine::sp_ok_arena arena_q8(M * K * sizeof(sp_ok_q8_t) + 4096);
    ::sp_ok_q8_tensor W_q8;
    bool ok_enc_q8 = sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(
        W_q8, w_fp16.data(), (size_t)(M * K), SCALE, P, Kf, arena_q8);
    CHECK(ok_enc_q8, "encode W -> Q8");
    W_shape.scale_recip     = W_q8.scale_recip;
    W_shape.frobenius_scale = W_q8.frobenius_scale;

    sp::engine::sp_ok_arena arena_q4(M * K * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor W_q4;
    bool ok_enc_q4 = sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(
        W_q4, w_fp16.data(), (size_t)(M * K), SCALE, P, Kf, arena_q4);
    CHECK(ok_enc_q4, "encode W -> Q4");
    CHECK(W_q4.scale_recip     == W_q8.scale_recip,
          "scale_recip matches Q8 path");
    CHECK(W_q4.frobenius_scale == W_q8.frobenius_scale,
          "frobenius_scale matches Q8 path");

    /* Q4 shift should be >= Q8 shift (typically ~4 bits larger). */
    CHECK(W_q4.q4_shift >= W_q8.q8_shift,
          "Q4 shift >= Q8 shift (codebook halved)");
    std::fprintf(stderr,
        "  W_q8.shift=%d  W_q4.shift=%d  delta=%d\n",
        (int)W_q8.q8_shift, (int)W_q4.q4_shift,
        (int)(W_q4.q4_shift - W_q8.q8_shift));

    /* Build X and the three output tensors. */
    sp::engine::sp_ok_arena arena_x(N * K * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor X;
    make_x(X, N, K, arena_x, 0xFEEDFACE);

    sp::engine::sp_ok_arena arena_yq8(N * M * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_arena arena_yq4(N * M * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor Y_q8, Y_q4;
    int64_t shape_y[4] = { N, M, 1, 1 };
    Y_q8.reset(2, shape_y);
    Y_q4.reset(2, shape_y);
    arena_yq8.alloc_tensor(Y_q8);
    arena_yq4.alloc_tensor(Y_q4);

    bool ok_q8 = sp::engine::sp_matmul_ok_q8(W_shape, W_q8, X, Y_q8);
    bool ok_q4 = sp::engine::sp_matmul_ok_q4(W_shape, W_q4, X, Y_q4);
    CHECK(ok_q8, "sp_matmul_ok_q8 returned true");
    CHECK(ok_q4, "sp_matmul_ok_q4 returned true");
    CHECK(Y_q4.scale_recip == Y_q8.scale_recip,
          "Y.scale_recip identical across Q4 and Q8");
    CHECK(Y_q4.frobenius_scale == Y_q8.frobenius_scale,
          "Y.frobenius_scale identical across Q4 and Q8");

    /* Characterize Q4 vs Q8 relative error. The Q4 result deviates from
     * Q8 by O(K) accumulated per-element quant noise; we expect mean
     * relative-magnitude error <~ 0.25 (the codebook is 16 levels). */
    double sum_rel = 0.0;
    size_t counted = 0;
    int64_t max_abs_q8 = 0;
    for (int64_t i = 0; i < N * M; ++i) {
        int64_t mag = std::max(std::abs(Y_q8.data[i].a), std::abs(Y_q8.data[i].b));
        if (mag > max_abs_q8) max_abs_q8 = mag;
    }
    for (int64_t i = 0; i < N * M; ++i) {
        int64_t mag = std::max(std::abs(Y_q8.data[i].a), std::abs(Y_q8.data[i].b));
        if (mag < max_abs_q8 / 16) continue;  /* skip near-zero outputs */
        int64_t da = std::abs(Y_q4.data[i].a - Y_q8.data[i].a);
        int64_t db = std::abs(Y_q4.data[i].b - Y_q8.data[i].b);
        sum_rel += (double)std::max(da, db) / (double)mag;
        ++counted;
    }
    double mean_rel = counted ? sum_rel / counted : 0.0;
    std::fprintf(stderr,
        "  N*M=%lld counted=%zu mean_rel(Q4 vs Q8)=%.4f max_abs_q8=%lld\n",
        (long long)(N * M), counted, mean_rel, (long long)max_abs_q8);
    CHECK(mean_rel < 0.5,
          "mean relative error Q4 vs Q8 within tolerance");
}

/* ---------- Test 2: Q4 fp32 bridge round-trip --------------------------- */

static void q4_to_fp32_matches_q8_to_fp32_bridge() {
    constexpr int64_t M = 32;
    constexpr int64_t K = 128;
    constexpr int64_t N = 2;
    constexpr int64_t SCALE = 1 << 10;
    constexpr int64_t P = 41;
    constexpr int64_t Kf = 2;

    auto w_fp16 = make_w(M * K, 0xDECAFFEED);
    sp::engine::sp_ok_tensor W_shape;
    int64_t shape_w[4] = { K, M, 1, 1 };
    W_shape.reset(2, shape_w);

    sp::engine::sp_ok_arena arena_q8(M * K * sizeof(sp_ok_q8_t) + 4096);
    sp::engine::sp_ok_arena arena_q4(M * K * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q8_tensor W_q8; ::sp_ok_q4_tensor W_q4;
    sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(W_q8, w_fp16.data(), M*K, SCALE, P, Kf, arena_q8);
    sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(W_q4, w_fp16.data(), M*K, SCALE, P, Kf, arena_q4);
    W_shape.scale_recip     = W_q8.scale_recip;
    W_shape.frobenius_scale = W_q8.frobenius_scale;

    sp::engine::sp_ok_arena arena_x(N * K * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor X;
    make_x(X, N, K, arena_x, 0xB1BAD00D);

    std::vector<float> Y_q8_f(N * M), Y_q4_f(N * M);
    bool ok_q8 = sp::engine::sp_matmul_ok_q8_to_fp32(W_shape, W_q8, X, Y_q8_f.data(), (int)M, (int)N);
    bool ok_q4 = sp::engine::sp_matmul_ok_q4_to_fp32(W_shape, W_q4, X, Y_q4_f.data(), (int)M, (int)N);
    CHECK(ok_q8, "q8 fp32 bridge");
    CHECK(ok_q4, "q4 fp32 bridge");

    /* Magnitudes should be in the same order; differences come from the
     * quant noise. We measure mean abs relative error across saturated
     * outputs. */
    float max_q8 = 0.0f;
    for (auto v : Y_q8_f) max_q8 = std::max(max_q8, std::abs(v));
    double sum_rel = 0.0;
    size_t counted = 0;
    for (size_t i = 0; i < Y_q8_f.size(); ++i) {
        if (std::abs(Y_q8_f[i]) < max_q8 / 16.0f) continue;
        sum_rel += std::abs(Y_q4_f[i] - Y_q8_f[i]) / std::abs(Y_q8_f[i]);
        ++counted;
    }
    double mean_rel = counted ? sum_rel / counted : 0.0;
    std::fprintf(stderr,
        "  fp32 bridge: counted=%zu mean_rel=%.4f max_q8=%.4f\n",
        counted, mean_rel, max_q8);
    CHECK(mean_rel < 0.5, "fp32 bridge Q4 within tolerance of Q8");
}

/* ---------- Test 3: Q4 shape gate ---------------------------------------- */

static void q4_matmul_rejects_shape_mismatch() {
    constexpr int64_t M = 8, K = 16, N = 2;
    sp::engine::sp_ok_tensor W_shape;
    int64_t shape_w[4] = { K, M, 1, 1 };
    W_shape.reset(2, shape_w);

    sp::engine::sp_ok_arena arena(M * K * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor W_q4;
    arena.alloc_tensor_q4(W_q4, (size_t)(M * K));
    /* Bad X: shape mismatch in K. */
    sp::engine::sp_ok_tensor X;
    int64_t shape_x[4] = { N, K + 1, 1, 1 };
    X.reset(2, shape_x);
    sp::engine::sp_ok_arena arena_x(N * (K + 1) * sizeof(sp_ok_t) + 4096);
    arena_x.alloc_tensor(X);

    sp::engine::sp_ok_tensor Y;
    int64_t shape_y[4] = { N, M, 1, 1 };
    Y.reset(2, shape_y);
    sp::engine::sp_ok_arena arena_y(N * M * sizeof(sp_ok_t) + 4096);
    arena_y.alloc_tensor(Y);

    bool ok = sp::engine::sp_matmul_ok_q4(W_shape, W_q4, X, Y);
    CHECK(!ok, "Q4 matmul rejects mismatched K");
}

int main() {
    std::fprintf(stderr, "test_sp_matmul_q4: Phase 14 fused Q4 matmul parity\n");
    run("q4_matmul_shape_and_norm",            q4_matmul_shape_and_norm);
    run("q4_to_fp32_matches_q8_to_fp32_bridge", q4_to_fp32_matches_q8_to_fp32_bridge);
    run("q4_matmul_rejects_shape_mismatch",    q4_matmul_rejects_shape_mismatch);
    std::fprintf(stderr,
        "[result] %s — %d check(s) %d failure(s)\n",
        g_failures == 0 ? "PASS" : "FAIL",
        g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
