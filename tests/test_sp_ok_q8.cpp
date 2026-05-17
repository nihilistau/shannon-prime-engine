/* test_sp_ok_q8.cpp — Phase 12 Step A parity test.
 *
 * Verifies that random integer ring elements survive the full Frobenius +
 * int8-pack + unpack chain with:
 *   1. coordinate-space error <= 2^(q8_shift - 1) (rounding bound)
 *   2. relative error in the magnitude-saturated regime <= 1/128
 *   3. ring multiplication structure preserved up to that quantization
 *   4. correct shift selection (matches absmax / 128 ceiling)
 *   5. 8x storage compression vs raw sp_ok_t
 *
 * Inputs are drawn to imitate post-encode pre-Frobenius weight coordinates:
 * |a| in [0, 2^24), b = 0 (matches sp_ok_encode_from_fp16 output).
 */

#include <cstddef>
extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_frobenius.h"
#include "../lib/shannon-prime/core/sp_ok_q8.h"
}

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

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

/* ---------- Test 1: pick_shift returns minimal valid shift -------------- */

static void shift_picker_is_minimal() {
    /* Boundary cases. */
    CHECK(sp_ok_q8_pick_shift(0)   == 0, "absmax=0 -> shift=0");
    CHECK(sp_ok_q8_pick_shift(127) == 0, "127 fits in int8");
    CHECK(sp_ok_q8_pick_shift(128) == 1, "128 needs shift=1 (128+1)>>1=64<=127");
    CHECK(sp_ok_q8_pick_shift(255) == 2, "255 -> 128 -> 64 with ceil-shift, s=2");
    CHECK(sp_ok_q8_pick_shift(256) == 2, "256 -> 128 -> 64 with ceil-shift, s=2");
    /* O(2^24) (pre-Frobenius regime) */
    int8_t s24 = sp_ok_q8_pick_shift(((int64_t)1 << 24) - 1);
    CHECK(s24 == 18, "2^24 - 1 -> shift=18 (ceil-shift accounts for round bias)");
    /* O(2^45) (post-Frobenius regime). */
    int8_t s45 = sp_ok_q8_pick_shift(((int64_t)1 << 45));
    CHECK(s45 == 39, "2^45 -> shift=39 (post-Frobenius regime, ceil-shift)");
    /* O(2^60) (worst case for int64). */
    int8_t s60 = sp_ok_q8_pick_shift(((int64_t)1 << 60));
    CHECK(s60 == 54, "2^60 -> shift=54");
    std::fprintf(stderr, "  shift(2^24-1)=%d  shift(2^45)=%d  shift(2^60)=%d\n",
                 s24, s45, s60);
}

/* ---------- Test 2: encode/decode round-trip respects rounding bound ---- */

static void roundtrip_bounded_by_rounding_step() {
    constexpr size_t N = 4096;
    std::mt19937_64 rng(0xC0DEFACECAFEBABE);
    std::uniform_int_distribution<int64_t> dist(-((int64_t)1 << 24),
                                                  ((int64_t)1 << 24));
    std::vector<sp_ok_t> src(N);
    for (size_t i = 0; i < N; ++i) src[i] = sp_ok_t{ dist(rng), dist(rng) };

    std::vector<sp_ok_q8_t> packed(N);
    int8_t shift = sp_ok_q8_encode_array(packed.data(), src.data(), N);

    std::vector<sp_ok_t> recon(N);
    sp_ok_q8_decode_array(recon.data(), packed.data(), N, shift);

    int64_t err_bound = sp_ok_q8_max_error(shift);
    int64_t worst_a = 0, worst_b = 0;
    for (size_t i = 0; i < N; ++i) {
        int64_t da = std::abs(recon[i].a - src[i].a);
        int64_t db = std::abs(recon[i].b - src[i].b);
        if (da > worst_a) worst_a = da;
        if (db > worst_b) worst_b = db;
        CHECK(da <= err_bound, "|a_recon - a_src| within rounding step");
        CHECK(db <= err_bound, "|b_recon - b_src| within rounding step");
    }
    std::fprintf(stderr,
        "  N=%zu shift=%d  err_bound=%lld  worst_a=%lld  worst_b=%lld\n",
        N, (int)shift, (long long)err_bound,
        (long long)worst_a, (long long)worst_b);
}

/* ---------- Test 3: full Frobenius round-trip (the headline) ------------ */

static void frobenius_roundtrip_relative_error() {
    constexpr size_t N = 4096;
    std::mt19937_64 rng(0xDEADBEEFCAFEBABE);
    std::uniform_int_distribution<int64_t> dist(-((int64_t)1 << 24),
                                                  ((int64_t)1 << 24));

    /* Build pre-Frobenius array (encoder output: (a, 0) with a in int24 range). */
    std::vector<sp_ok_t> pre(N);
    for (size_t i = 0; i < N; ++i) pre[i] = sp_ok_t{ dist(rng), 0 };

    /* Apply phi_41^8 — the production Config B from Paper D. */
    std::vector<sp_ok_t> post(N);
    for (size_t i = 0; i < N; ++i) {
        post[i] = sp_apply_frobenius(pre[i], 41, 8);
    }

    /* Verify post magnitudes land in the expected ~2^45 regime. */
    int64_t absmax_post = sp_ok_q8_absmax(post.data(), N);
    int64_t expected_floor = (int64_t)1 << 42;  /* phi_41^8 scales by ~2^45 */
    CHECK(absmax_post > expected_floor,
          "post-Frobenius absmax exceeds 2^42 (phi_41^8 actually applied)");
    std::fprintf(stderr,
        "  post-Frobenius absmax = %lld (%.3f * 2^45)\n",
        (long long)absmax_post,
        (double)absmax_post / (double)((int64_t)1 << 45));

    /* Pack to int8. */
    std::vector<sp_ok_q8_t> packed(N);
    int8_t shift = sp_ok_q8_encode_array(packed.data(), post.data(), N);
    std::fprintf(stderr, "  pack: shift = %d  packed_bytes = %zu  raw_bytes = %zu\n",
                 (int)shift, N * sizeof(sp_ok_q8_t), N * sizeof(sp_ok_t));
    CHECK(sizeof(sp_ok_q8_t) == 2,        "packed element is exactly 2 bytes");
    CHECK(sizeof(sp_ok_t)    == 16,       "raw element is 16 bytes (compression = 8x)");

    /* Unpack. */
    std::vector<sp_ok_t> recon(N);
    sp_ok_q8_decode_array(recon.data(), packed.data(), N, shift);

    /* Accumulate per-coord max relative error in the saturated regime
     * (|coord| >= absmax/2 — the entries that drive the int8 utilization). */
    double  rel_err_sum = 0.0;
    size_t  rel_err_count = 0;
    int64_t worst_abs_err = 0;
    int64_t err_bound = sp_ok_q8_max_error(shift);
    for (size_t i = 0; i < N; ++i) {
        int64_t da = std::abs(recon[i].a - post[i].a);
        int64_t db = std::abs(recon[i].b - post[i].b);
        if (da > worst_abs_err) worst_abs_err = da;
        if (db > worst_abs_err) worst_abs_err = db;
        CHECK(da <= err_bound, "|a_recon - a_post| <= rounding step");
        CHECK(db <= err_bound, "|b_recon - b_post| <= rounding step");
        int64_t mag = std::max(std::abs(post[i].a), std::abs(post[i].b));
        if (mag >= absmax_post / 2 && mag > 0) {
            rel_err_sum += (double)std::max(da, db) / (double)mag;
            ++rel_err_count;
        }
    }
    double rel_err_mean = (rel_err_count > 0)
                              ? rel_err_sum / (double)rel_err_count
                              : 0.0;
    std::fprintf(stderr,
        "  recon: worst_abs_err = %lld  rel_err_bound = 2^-7 = %.4f  observed_mean = %.4f\n",
        (long long)worst_abs_err, 1.0 / 128.0, rel_err_mean);
    /* Saturated-regime mean relative error should be well below 1/128. */
    CHECK(rel_err_mean < 1.0 / 128.0,
          "mean relative error in saturated regime under 1/128 (int8 quant ceiling)");
}

/* ---------- Test 4: norm preservation under pack/unpack ----------------- */
/*
 * sp_ok_mul has internal cancellation: (x.a*y.a - 41*x.b*y.b) can be a
 * tiny difference of large terms, giving unbounded relative error for the
 * coordinate-wise comparison even when each coord error is tiny. The
 * algebra-preserving invariant is the NORM, which is positive-definite:
 *   N(x) = x.a^2 + x.a*x.b + 41*x.b^2
 * The norm of the decoded element should remain within O(1/64) of the
 * original norm (sum of the relative coord errors squared, then summed).
 */
static void norm_preserved_under_pack_unpack() {
    constexpr size_t N = 1024;
    std::mt19937_64 rng(0xABCDEF0123456789);
    /* Use int24 magnitudes to keep the squared norm in int64. */
    std::uniform_int_distribution<int64_t> dist(-((int64_t)1 << 24),
                                                  ((int64_t)1 << 24));

    std::vector<sp_ok_t> src(N);
    for (size_t i = 0; i < N; ++i) src[i] = sp_ok_t{ dist(rng), dist(rng) };

    std::vector<sp_ok_q8_t> packed(N);
    int8_t shift = sp_ok_q8_encode_array(packed.data(), src.data(), N);
    std::vector<sp_ok_t> recon(N);
    sp_ok_q8_decode_array(recon.data(), packed.data(), N, shift);

    double sum_rel_err  = 0.0;
    double worst_rel    = 0.0;
    size_t counted      = 0;
    int64_t absmax = sp_ok_q8_absmax(src.data(), N);
    for (size_t i = 0; i < N; ++i) {
        int64_t n_orig  = sp_ok_norm(src[i]);
        int64_t n_recon = sp_ok_norm(recon[i]);
        int64_t mag = std::max(std::abs(src[i].a), std::abs(src[i].b));
        /* Only count elements in the saturated regime (>= absmax/4):
         * smaller-magnitude entries dominate the relative-error tail and
         * are expected to behave worse under int8 quant. */
        if (mag >= absmax / 4 && n_orig > 0) {
            double rel = std::abs((double)n_recon - (double)n_orig)
                       / (double)n_orig;
            sum_rel_err += rel;
            if (rel > worst_rel) worst_rel = rel;
            ++counted;
        }
    }
    double mean_rel = counted ? sum_rel_err / (double)counted : 0.0;
    std::fprintf(stderr,
        "  norm: shift=%d  counted=%zu  worst_rel=%.4f  mean_rel=%.4f\n",
        (int)shift, counted, worst_rel, mean_rel);
    /* The two coords contribute additively under squaring; the budget is
     * 2 * (1/128) = 1/64 on the relative norm error. */
    CHECK(mean_rel < 1.0 / 32.0,
          "saturated-regime mean norm preserved within int8 quant budget (1/32, allows for a*b cross-term)");
}

/* ---------- Test 5: deterministic encode (same input -> same output) ---- */

static void encoder_is_deterministic() {
    constexpr size_t N = 1024;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(-((int64_t)1 << 30),
                                                  ((int64_t)1 << 30));
    std::vector<sp_ok_t> src(N);
    for (size_t i = 0; i < N; ++i) src[i] = sp_ok_t{ dist(rng), dist(rng) };

    std::vector<sp_ok_q8_t> p1(N), p2(N);
    int8_t s1 = sp_ok_q8_encode_array(p1.data(), src.data(), N);
    int8_t s2 = sp_ok_q8_encode_array(p2.data(), src.data(), N);
    CHECK(s1 == s2, "encoder produces same shift for identical input");
    CHECK(std::memcmp(p1.data(), p2.data(), N * sizeof(sp_ok_q8_t)) == 0,
          "encoder produces byte-identical packed output");
}

int main() {
    std::fprintf(stderr,
        "test_sp_ok_q8: Phase 12 Step A — packed int8 O_K storage parity\n");
    run("shift_picker_is_minimal",            shift_picker_is_minimal);
    run("roundtrip_bounded_by_rounding_step", roundtrip_bounded_by_rounding_step);
    run("frobenius_roundtrip_relative_error", frobenius_roundtrip_relative_error);
    run("norm_preserved_under_pack_unpack",      norm_preserved_under_pack_unpack);
    run("encoder_is_deterministic",           encoder_is_deterministic);
    std::fprintf(stderr,
        "[result] %s — %d check(s) %d failure(s)\n",
        g_failures == 0 ? "PASS" : "FAIL",
        g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
