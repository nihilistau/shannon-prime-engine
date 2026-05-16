// Shannon-Prime — sp_poly_ring unit tests (Phase 3 pivot, part 1).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Locks in three properties before we plumb the ring into attention:
//   1. Naive polynomial multiplication closes correctly in R = Z[x]/(x^N+1).
//   2. The reversed-encoding identity: Q(x) * K_rev(x) puts Σ q_i k_i at
//      coefficient x^{d-1}.
//   3. CKKS-style fp32 → int → fp32 dot product round-trips within ULP.

extern "C" {
#include "../lib/shannon-prime/core/sp_poly_ring.h"
}

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
    if (_d > (eps)) { std::fprintf(stderr, \
        "  ASSERT_NEAR FAIL (%s:%d): %.6g vs %.6g (delta %.6g > eps %.6g)\n", \
        __FILE__, __LINE__, (double)(a), (double)(b), _d, (double)(eps)); g_fail++; } } while (0)

// =========================================================================
// 1. Negacyclic ring closure.
//
//    (x^N + 1) = 0  ⟹  x^N = -1
//
//    Verify: encode P(x) = x^{N-1}, multiply by Q(x) = x, expect
//    P*Q = x^N ≡ -1 in the ring, i.e. coeff[0] == -1, rest == 0.
// =========================================================================

TEST(ring_negacyclic_xN_eq_neg1) {
    constexpr int N = 8;
    std::vector<sp_poly_coeff> a_buf(N), b_buf(N), c_buf(N);
    sp_poly A = { a_buf.data(), N };
    sp_poly B = { b_buf.data(), N };
    sp_poly C = { c_buf.data(), N };

    sp_poly_zero(&A);
    sp_poly_zero(&B);
    A.coeffs[N - 1] = 1;  // P = x^{N-1}
    B.coeffs[1]     = 1;  // Q = x

    sp_poly_mul(&C, &A, &B);

    // Expect C = x^N = -1 in this ring → coeff[0] == -1, rest 0.
    ASSERT(C.coeffs[0] == -1);
    for (int i = 1; i < N; ++i) ASSERT(C.coeffs[i] == 0);
}

// =========================================================================
// 2. Constant polynomial multiplication.
//    P(x) = 3, Q(x) = 5  ⟹  P*Q = 15 (coeff[0]).
// =========================================================================

TEST(ring_constant_multiply) {
    constexpr int N = 8;
    std::vector<sp_poly_coeff> a_buf(N), b_buf(N), c_buf(N);
    sp_poly A = { a_buf.data(), N };
    sp_poly B = { b_buf.data(), N };
    sp_poly C = { c_buf.data(), N };
    sp_poly_zero(&A); sp_poly_zero(&B);
    A.coeffs[0] = 3; B.coeffs[0] = 5;
    sp_poly_mul(&C, &A, &B);
    ASSERT(C.coeffs[0] == 15);
    for (int i = 1; i < N; ++i) ASSERT(C.coeffs[i] == 0);
}

// =========================================================================
// 3. The CKKS dot product identity, integer-only.
//
//    Q(x)     = q_0 + q_1 x + q_2 x^2 + ... + q_{d-1} x^{d-1}
//    K_rev(x) = k_{d-1} + k_{d-2} x + ... + k_0 x^{d-1}
//
//    Coeff of x^{d-1} in Q*K_rev = Σ_{i+j == d-1} q_i · k_{d-1-j}
//                                = Σ_{i} q_i · k_i
//
//    Verify: hand-built q, k integer vectors, recovered dot matches.
// =========================================================================

TEST(reversed_encoding_dot_product_identity) {
    constexpr int d = 4;
    constexpr int N = 8;
    int64_t q[d] = { 1, 2, 3, 4 };
    int64_t k[d] = { 5, 6, 7, 8 };
    int64_t expected = 1*5 + 2*6 + 3*7 + 4*8;  // = 70

    std::vector<sp_poly_coeff> a_buf(N), b_buf(N), c_buf(N);
    sp_poly Q = { a_buf.data(), N };
    sp_poly K = { b_buf.data(), N };
    sp_poly C = { c_buf.data(), N };
    sp_poly_zero(&Q); sp_poly_zero(&K);
    // Q forward, K reversed.
    for (int i = 0; i < d; ++i) Q.coeffs[i] = q[i];
    for (int i = 0; i < d; ++i) K.coeffs[i] = k[d - 1 - i];
    sp_poly_mul(&C, &Q, &K);
    // Σ q_i k_i lives at coefficient x^{d-1}.
    ASSERT(C.coeffs[d - 1] == expected);
}

// =========================================================================
// 4. CKKS scale-encode + dot product round-trip for fp32 vectors.
//
//    Generate random q, k ∈ R^d. Compute reference dot product. Run
//    through sp_poly_dot_product. Compare.
// =========================================================================

static double ref_dot(const std::vector<float>& q, const std::vector<float>& k) {
    double s = 0;
    for (size_t i = 0; i < q.size(); ++i) s += (double)q[i] * (double)k[i];
    return s;
}

TEST(ckks_dot_product_recovers_fp32_baseline_d16) {
    constexpr int d = 16;
    constexpr int N = 32;
    constexpr double delta = 1 << 14;  // 16384
    std::vector<float> q(d), k(d);
    std::mt19937 rng(101);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : q) v = nd(rng);
    for (auto& v : k) v = nd(rng);
    double expected = ref_dot(q, k);

    std::vector<sp_poly_coeff> scratch(3 * N);
    float got = sp_poly_dot_product(q.data(), k.data(), d, N, delta, scratch.data());

    std::printf("  d=16:  expected=%.6f  got=%.6f  err=%.2e\n",
                expected, (double)got, std::abs(expected - got));
    // delta = 2^14 → quantization error per coefficient ~ 1/delta = 6.1e-5.
    // Summing d=16 such errors: worst case ~ 16 * 6.1e-5 * |max(q_i)|*delta /delta^2
    // ~ 16 * 6.1e-5 ~ 1e-3. Allow generous 1e-2.
    ASSERT_NEAR(got, expected, 1e-2);
}

TEST(ckks_dot_product_recovers_fp32_baseline_d64) {
    constexpr int d = 64;
    constexpr int N = 128;
    constexpr double delta = 1 << 14;
    std::vector<float> q(d), k(d);
    std::mt19937 rng(202);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : q) v = nd(rng);
    for (auto& v : k) v = nd(rng);
    double expected = ref_dot(q, k);

    std::vector<sp_poly_coeff> scratch(3 * N);
    float got = sp_poly_dot_product(q.data(), k.data(), d, N, delta, scratch.data());

    std::printf("  d=64:  expected=%.6f  got=%.6f  err=%.2e\n",
                expected, (double)got, std::abs(expected - got));
    ASSERT_NEAR(got, expected, 5e-2);
}

TEST(ckks_dot_product_d256_typical_attention_size) {
    // Gemma3 head_dim = 256. Use a smaller delta to keep int64 from
    // overflowing on 256 quadratic terms.
    constexpr int d = 256;
    constexpr int N = 512;
    constexpr double delta = 1 << 10;  // smaller scale for larger d
    std::vector<float> q(d), k(d);
    std::mt19937 rng(303);
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)d));
    for (auto& v : q) v = nd(rng);
    for (auto& v : k) v = nd(rng);
    double expected = ref_dot(q, k);

    std::vector<sp_poly_coeff> scratch(3 * N);
    float got = sp_poly_dot_product(q.data(), k.data(), d, N, delta, scratch.data());

    std::printf("  d=256: expected=%.6f  got=%.6f  err=%.2e\n",
                expected, (double)got, std::abs(expected - got));
    // Smaller delta → larger ULP. Expect ~1e-3 absolute.
    ASSERT_NEAR(got, expected, 1e-2);
}

int main() {
    std::printf("Shannon-Prime sp_poly_ring tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
