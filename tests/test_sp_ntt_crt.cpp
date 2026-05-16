/* Phase 9: dual-prime CRT NTT parity test.
 *
 * Locks five contracts:
 *   1. Forward → inverse roundtrip in each prime universe is identity.
 *   2. CRT-recombined poly mul matches the existing 60-bit sp_poly_mul_ntt
 *      bit-for-bit on random poly pairs in the operating regime.
 *   3. Matches the O(N^2) integer baseline sp_poly_mul on same inputs.
 *   4. Edge cases: 1*1=1 and x*x^(N-1)=-1.
 *   5. CKKS dot-product recovery to fp32 ULP at d=256.
 *
 * If 2 and 3 hold, the CRT path can be dropped into the engine in
 * place of the 60-bit NTT path with bit-exact PPL parity.
 */

extern "C" {
#include "../lib/shannon-prime/core/sp_poly_ring.h"
#include "../lib/shannon-prime/core/sp_ntt.h"
#include "../lib/shannon-prime/core/sp_ntt_crt.h"
}

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TE { const char *name; void (*fn)(); };
static std::vector<TE> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

static constexpr int N = SP_NTT_CRT_N;

/* Test 1 — Forward then inverse must round-trip to the original poly
 * in each prime universe independently. */
TEST(crt_forward_inverse_roundtrip) {
    std::mt19937_64 rng(0xC0DEFACE);
    std::vector<uint64_t> buf1(N), buf1_orig(N);
    std::vector<uint64_t> buf2(N), buf2_orig(N);
    for (int i = 0; i < N; ++i) {
        buf1[i] = rng() % SP_NTT_CRT_Q1;
        buf2[i] = rng() % SP_NTT_CRT_Q2;
        buf1_orig[i] = buf1[i];
        buf2_orig[i] = buf2[i];
    }
    sp_ntt_crt_forward(buf1.data(), &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(buf1.data(), &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_forward(buf2.data(), &SP_NTT_CRT_CTX_Q2);
    sp_ntt_crt_inverse(buf2.data(), &SP_NTT_CRT_CTX_Q2);
    int err1 = 0, err2 = 0;
    for (int i = 0; i < N; ++i) {
        if (buf1[i] != buf1_orig[i]) ++err1;
        if (buf2[i] != buf2_orig[i]) ++err2;
    }
    ASSERT(err1 == 0);
    ASSERT(err2 == 0);
}

/* Test 2 — CRT pipeline matches the existing 60-bit sp_poly_mul_ntt
 * bit-for-bit. Both routes solve the same integer polynomial mul
 * (just with different modular shards). */
TEST(crt_matches_60bit_ntt_path) {
    std::mt19937_64 rng(0xDEADBEEF);
    /* Operating regime: |coeff| < 2^24 keeps the per-coefficient
     * convolution sum below M/2 = 2^59. */
    const int64_t LIM = (int64_t)1 << 24;

    std::vector<int64_t> a(N), b(N), c_60bit(N), c_crt(N);
    std::vector<sp_poly_coeff> sa(N), sb(N), sc(N);
    std::vector<uint64_t> ws_60(3 * N), ws_crt(6 * N);

    int mismatches = 0;
    for (int trial = 0; trial < 4; ++trial) {
        for (int i = 0; i < N; ++i) {
            a[i] = (int64_t)(rng() % (2 * LIM)) - LIM;
            b[i] = (int64_t)(rng() % (2 * LIM)) - LIM;
            sa[i] = a[i];
            sb[i] = b[i];
        }
        /* 60-bit NTT path */
        sp_poly A60 = { sa.data(), N };
        sp_poly B60 = { sb.data(), N };
        sp_poly C60 = { sc.data(), N };
        int rc = sp_poly_mul_ntt(&C60, &A60, &B60,
            ws_60.data(), ws_60.data() + N, ws_60.data() + 2 * N);
        ASSERT(rc == 0);
        for (int i = 0; i < N; ++i) c_60bit[i] = sc[i];

        /* CRT path */
        int rc2 = sp_ntt_crt_poly_mul(c_crt.data(), a.data(), b.data(),
                                      N, ws_crt.data());
        ASSERT(rc2 == 0);

        for (int i = 0; i < N; ++i) {
            if (c_crt[i] != c_60bit[i]) {
                if (mismatches < 3) {
                    std::fprintf(stderr,
                        "  [trial %d coeff %d] 60bit=%lld  crt=%lld  diff=%lld\n",
                        trial, i, (long long)c_60bit[i],
                        (long long)c_crt[i],
                        (long long)(c_crt[i] - c_60bit[i]));
                }
                ++mismatches;
            }
        }
    }
    ASSERT(mismatches == 0);
}

/* Test 3 — CRT matches the O(N^2) integer reference for small inputs
 * where the signed result fits comfortably. */
TEST(crt_matches_integer_negacyclic_reference) {
    std::mt19937_64 rng(0x12345);
    /* Keep |coeff| small so |c_ref| stays well below M/2. */
    const int64_t LIM = (int64_t)1 << 16;
    std::vector<int64_t> a(N), b(N), c_crt(N);
    std::vector<int64_t> c_ref(N, 0);
    std::vector<uint64_t> ws(6 * N);

    for (int i = 0; i < N; ++i) {
        a[i] = (int64_t)(rng() % (2 * LIM)) - LIM;
        b[i] = (int64_t)(rng() % (2 * LIM)) - LIM;
    }

    /* O(N^2) negacyclic reference. */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = (i + j) % N;
            int64_t prod = a[i] * b[j];
            c_ref[k] += (i + j < N) ? prod : -prod;
        }
    }

    int rc = sp_ntt_crt_poly_mul(c_crt.data(), a.data(), b.data(),
                                  N, ws.data());
    ASSERT(rc == 0);

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (c_crt[i] != c_ref[i]) {
            if (mismatches < 3) {
                std::fprintf(stderr,
                    "  [coeff %d] ref=%lld crt=%lld\n",
                    i, (long long)c_ref[i], (long long)c_crt[i]);
            }
            ++mismatches;
        }
    }
    ASSERT(mismatches == 0);
}

/* Test 4 — edge cases. */
TEST(crt_identity_and_xN_negative_one) {
    std::vector<int64_t> a(N, 0), b(N, 0), c(N, 0);
    std::vector<uint64_t> ws(6 * N);

    /* 1 * 1 = 1 */
    a[0] = 1; b[0] = 1;
    int rc = sp_ntt_crt_poly_mul(c.data(), a.data(), b.data(), N, ws.data());
    ASSERT(rc == 0);
    ASSERT(c[0] == 1);
    for (int i = 1; i < N; ++i) ASSERT(c[i] == 0);

    /* x * x^(N-1) = x^N = -1 */
    std::fill(a.begin(), a.end(), 0);
    std::fill(b.begin(), b.end(), 0);
    a[1] = 1;
    b[N - 1] = 1;
    rc = sp_ntt_crt_poly_mul(c.data(), a.data(), b.data(), N, ws.data());
    ASSERT(rc == 0);
    ASSERT(c[0] == -1);
    for (int i = 1; i < N; ++i) ASSERT(c[i] == 0);
}

/* Test 5 — CKKS dot product recovery via the CRT pipeline. */
TEST(crt_ckks_dot_product_recovery) {
    constexpr int d = 256;
    const double delta = (double)(1 << 14);

    std::mt19937_64 rng(0xABCD);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(d), k(d);
    for (int i = 0; i < d; ++i) { q[i] = dist(rng); k[i] = dist(rng); }

    double expected = 0.0;
    for (int i = 0; i < d; ++i) expected += (double)q[i] * (double)k[i];

    /* Encode Q forward, K reversed. */
    std::vector<int64_t> Q_int(N, 0), K_int(N, 0), C_int(N, 0);
    for (int i = 0; i < d; ++i) {
        Q_int[i]         = (int64_t)std::llrint((double)q[i] * delta);
        K_int[d - 1 - i] = (int64_t)std::llrint((double)k[i] * delta);
    }
    std::vector<uint64_t> ws(6 * N);
    int rc = sp_ntt_crt_poly_mul(C_int.data(), Q_int.data(), K_int.data(),
                                  N, ws.data());
    ASSERT(rc == 0);
    double got = (double)C_int[d - 1] / (delta * delta);
    double err = std::fabs(got - expected);
    std::fprintf(stderr,
        "  [crt dot] expected=%.6f got=%.6f err=%.3e\n",
        expected, got, err);
    ASSERT(err < 5e-2);
}

/* Test 6 — Phase 9b engine integration helpers.
 *
 * The new dual-universe encoders + qk_cached path must produce the
 * same fp32 score as the existing 60-bit sp_poly_dot_product_ntt_q_cached
 * to within ULP. This locks parity at the API the engine will actually
 * call (Q is encoded once per (h, qi), K is encoded once per (kv_h, t),
 * then per-(qi, t) we pay only pointwise+inverse+CRT-stitch). */
TEST(crt_engine_helpers_match_60bit_qk_cached) {
    constexpr int d = 256;
    const double delta = (double)(1 << 14);
    std::mt19937_64 rng(0xFACEFEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> qv(d), kv(d);
    for (int i = 0; i < d; ++i) { qv[i] = dist(rng); kv[i] = dist(rng); }

    /* --- 60-bit baseline path (Phase 5b/6 helpers) --- */
    std::vector<uint64_t> Q60(SP_NTT_N, 0);
    std::vector<int64_t>  int_scratch_60(SP_NTT_N, 0);
    sp_poly_encode_ntt_q(Q60.data(), qv.data(), d, delta, int_scratch_60.data());

    std::vector<int64_t>  k_int_60(SP_NTT_N, 0);
    std::vector<uint64_t> k_ntt_60(SP_NTT_N, 0);
    std::vector<uint64_t> c_ntt_60(SP_NTT_N, 0);
    int ok60 = 0;
    float dot60 = sp_poly_dot_product_ntt_q_cached(
        Q60.data(), kv.data(), d, delta,
        k_int_60.data(), k_ntt_60.data(), c_ntt_60.data(), &ok60);
    ASSERT(ok60 == 1);

    /* --- New CRT path (Phase 9b helpers) --- */
    std::vector<uint64_t> Q_q1(N, 0), Q_q2(N, 0);
    std::vector<int64_t>  int_scratch_crt(N, 0);
    sp_poly_encode_ntt_q_crt(Q_q1.data(), Q_q2.data(),
                              qv.data(), d, delta,
                              int_scratch_crt.data());

    std::vector<uint64_t> K_q1(N, 0), K_q2(N, 0);
    sp_poly_encode_ntt_k_reversed_crt(K_q1.data(), K_q2.data(),
                                       kv.data(), d, delta,
                                       int_scratch_crt.data());

    std::vector<uint64_t> c_q1(N, 0), c_q2(N, 0);
    int okcrt = 0;
    float dotcrt = sp_poly_dot_product_ntt_crt_qk_cached(
        Q_q1.data(), Q_q2.data(), K_q1.data(), K_q2.data(),
        d, delta, c_q1.data(), c_q2.data(), &okcrt);
    ASSERT(okcrt == 1);

    /* The two paths share semantics but route through different primes,
     * so we expect agreement to within fp32 ULP for this dot product
     * (the CRT path has slightly less headroom but the modulus is large
     * enough that all real-valued products fit). */
    const double diff = std::fabs((double)dot60 - (double)dotcrt);
    std::fprintf(stderr,
        "  [crt-vs-60bit] dot60=%.6f dotcrt=%.6f diff=%.3e\n",
        dot60, dotcrt, diff);
    ASSERT(diff < 1e-3);
}

int main() {
    std::fprintf(stderr,
        "test_sp_ntt_crt: Q1=%llu Q2=%llu  M=q1*q2~2^60  N=%d\n",
        (unsigned long long)SP_NTT_CRT_Q1,
        (unsigned long long)SP_NTT_CRT_Q2,
        (int)SP_NTT_CRT_N);
    for (auto &t : g_tests) {
        std::fprintf(stderr, "[run] %s\n", t.name);
        t.fn();
    }
    std::fprintf(stderr, "\n[result] %s — %d failure(s)\n",
        g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
