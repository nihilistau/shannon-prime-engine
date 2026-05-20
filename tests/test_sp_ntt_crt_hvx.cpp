/* Phase HVX-1: HVX-Barrett-Proth NTT parity test.
 *
 * Locks four contracts:
 *   1. sp_ntt_crt_hvx_forward output equals sp_ntt_crt_forward output
 *      bit-for-bit on random input, per prime context.
 *   2. sp_ntt_crt_hvx_inverse output equals sp_ntt_crt_inverse bit-for-bit.
 *   3. sp_ntt_crt_hvx_pointwise_mul equals sp_ntt_crt_pointwise_mul bit-for-bit.
 *   4. CRT-stitched dot product via the HVX wrapper matches the scalar
 *      wrapper to fp32 ULP (no precision loss in the inner ops).
 *
 * On x86 / CI the HVX kernel runs through the scalar fallback (no
 * SP_HEXAGON_ENABLED). The test still fires — it validates the
 * algorithm against the AVX-512 reference. The on-device run picks
 * up real Q6_* intrinsics; if Tests 1-3 still pass there with
 * SP_HEXAGON_ENABLED=1, the intrinsics are wired correctly.
 */

extern "C" {
#include "../lib/shannon-prime/core/sp_poly_ring.h"
#include "../lib/shannon-prime/core/sp_ntt_crt.h"
#include "../lib/shannon-prime/core/sp_ntt_crt_hvx.h"
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

/* Test 1 — HVX forward output equals AVX-512 (scalar) forward output. */
TEST(hvx_forward_matches_reference) {
    std::mt19937_64 rng(0xA1B2C3D4);
    for (auto ctx_ptr : { &SP_NTT_CRT_CTX_Q1, &SP_NTT_CRT_CTX_Q2 }) {
        const uint64_t q = ctx_ptr->q;
        std::vector<uint64_t> in_ref(N), in_hvx(N);
        for (int i = 0; i < N; ++i) {
            uint64_t v = rng() % q;
            in_ref[i] = v;
            in_hvx[i] = v;
        }
        sp_ntt_crt_forward(in_ref.data(), ctx_ptr);
        sp_ntt_crt_hvx_forward(in_hvx.data(), ctx_ptr);
        int diffs = 0;
        for (int i = 0; i < N; ++i) {
            if (in_ref[i] != in_hvx[i]) ++diffs;
        }
        ASSERT(diffs == 0);
    }
}

/* Test 2 — HVX inverse output equals reference inverse output. */
TEST(hvx_inverse_matches_reference) {
    std::mt19937_64 rng(0xBEEFCAFE);
    for (auto ctx_ptr : { &SP_NTT_CRT_CTX_Q1, &SP_NTT_CRT_CTX_Q2 }) {
        const uint64_t q = ctx_ptr->q;
        /* Seed with NTT-domain values, i.e. already-forward-transformed. */
        std::vector<uint64_t> in_ref(N), in_hvx(N);
        for (int i = 0; i < N; ++i) {
            uint64_t v = rng() % q;
            in_ref[i] = v;
            in_hvx[i] = v;
        }
        sp_ntt_crt_inverse(in_ref.data(), ctx_ptr);
        sp_ntt_crt_hvx_inverse(in_hvx.data(), ctx_ptr);
        int diffs = 0;
        for (int i = 0; i < N; ++i) {
            if (in_ref[i] != in_hvx[i]) ++diffs;
        }
        ASSERT(diffs == 0);
    }
}

/* Test 3 — HVX pointwise_mul output equals reference pointwise_mul output. */
TEST(hvx_pointwise_mul_matches_reference) {
    std::mt19937_64 rng(0x12345678);
    for (auto ctx_ptr : { &SP_NTT_CRT_CTX_Q1, &SP_NTT_CRT_CTX_Q2 }) {
        const uint64_t q = ctx_ptr->q;
        std::vector<uint64_t> a(N), b(N), c_ref(N), c_hvx(N);
        for (int i = 0; i < N; ++i) {
            a[i] = rng() % q;
            b[i] = rng() % q;
        }
        sp_ntt_crt_pointwise_mul(c_ref.data(), a.data(), b.data(), ctx_ptr);
        sp_ntt_crt_hvx_pointwise_mul(c_hvx.data(), a.data(), b.data(), ctx_ptr);
        int diffs = 0;
        for (int i = 0; i < N; ++i) {
            if (c_ref[i] != c_hvx[i]) ++diffs;
        }
        ASSERT(diffs == 0);
    }
}

/* Test 4 — HVX forward→inverse round-trip is identity (self-consistency
 * check on the HVX path alone; if Tests 1 & 2 pass, this is implied
 * but the redundancy catches algorithm drift quickly). */
TEST(hvx_forward_inverse_roundtrip) {
    std::mt19937_64 rng(0xFEEDFACE);
    for (auto ctx_ptr : { &SP_NTT_CRT_CTX_Q1, &SP_NTT_CRT_CTX_Q2 }) {
        const uint64_t q = ctx_ptr->q;
        std::vector<uint64_t> a(N), a_orig(N);
        for (int i = 0; i < N; ++i) {
            a[i] = rng() % q;
            a_orig[i] = a[i];
        }
        sp_ntt_crt_hvx_forward(a.data(), ctx_ptr);
        sp_ntt_crt_hvx_inverse(a.data(), ctx_ptr);
        int diffs = 0;
        for (int i = 0; i < N; ++i) {
            if (a[i] != a_orig[i]) ++diffs;
        }
        ASSERT(diffs == 0);
    }
}

/* Test 5 — CRT-stitched dot product via the HVX wrapper matches the
 * scalar wrapper. Both build their inputs from sp_poly_encode_ntt_*_crt
 * then call their respective dot product. The fp32 output must be
 * bit-equal because all inner ops are bit-equal. */
TEST(hvx_dot_product_matches_reference) {
    std::mt19937_64 rng(0xC0DEC0DE);
    const int d = 256;
    const double delta = 1024.0;  /* same fixed-point delta used in production */

    std::vector<float> q_vec(d), k_vec(d);
    for (int i = 0; i < d; ++i) {
        q_vec[i] = (float)((rng() % 2001) - 1000) * 0.001f;
        k_vec[i] = (float)((rng() % 2001) - 1000) * 0.001f;
    }
    std::vector<int64_t> int_scratch(N);
    std::vector<uint64_t> Q_q1(N), Q_q2(N), K_q1(N), K_q2(N);
    std::vector<uint64_t> ws_q1_ref(N), ws_q2_ref(N), ws_q1_hvx(N), ws_q2_hvx(N);

    sp_poly_encode_ntt_q_crt(Q_q1.data(), Q_q2.data(), q_vec.data(), d, delta, int_scratch.data());
    sp_poly_encode_ntt_k_reversed_crt(K_q1.data(), K_q2.data(), k_vec.data(), d, delta, int_scratch.data());

    int ok_ref = 0, ok_hvx = 0;
    float ref = sp_poly_dot_product_ntt_crt_qk_cached(
        Q_q1.data(), Q_q2.data(), K_q1.data(), K_q2.data(),
        d, delta, ws_q1_ref.data(), ws_q2_ref.data(), &ok_ref);
    float hvx = sp_poly_dot_product_ntt_crt_qk_cached_hvx(
        Q_q1.data(), Q_q2.data(), K_q1.data(), K_q2.data(),
        d, delta, ws_q1_hvx.data(), ws_q2_hvx.data(), &ok_hvx);

    ASSERT(ok_ref == 1);
    ASSERT(ok_hvx == 1);
    /* Bit-equality is the stronger contract — they must produce exactly
     * the same float because every inner op is bit-equal. */
    union { float f; uint32_t u; } a, b;
    a.f = ref; b.f = hvx;
    ASSERT(a.u == b.u);
}

/* Test 6 — availability flag. Informational; logs which path the kernel
 * actually took so the wider test harness can record it. */
TEST(hvx_availability_flag) {
    int avail = sp_ntt_crt_hvx_available();
    std::fprintf(stderr, "  [info] sp_ntt_crt_hvx_available() = %d (1=HVX intrinsics, 0=scalar fallback)\n", avail);
    /* Either value is acceptable on the host; this just records it. */
    ASSERT(avail == 0 || avail == 1);
}

int main() {
    std::fprintf(stderr, "test_sp_ntt_crt_hvx: running %zu tests\n", g_tests.size());
    for (auto& t : g_tests) {
        std::fprintf(stderr, "  %s ...\n", t.name);
        t.fn();
    }
    if (g_fail) {
        std::fprintf(stderr, "FAILED %d assertions\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "all tests passed\n");
    return 0;
}
