/* test_sp_ok_q8_load_path.cpp — Phase 12 Step B-1 parity test.
 *
 * Verifies the combined fp16 -> Frobenius -> pack pipeline that the live
 * load shim will use. Specifically, the new
 * sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius() must produce the same packed
 * output as calling the existing pieces (sp_ok_encode_from_fp16 +
 * sp_ok_encode_apply_frobenius_quant + sp_ok_q8_encode_array) by hand.
 *
 * Inputs are fp16 weights matching the Gemma3-1B distribution profile
 * (absmax ~ 8.0, std ~ 0.05) so the scale_recip = 16384 default is in
 * its normal operating regime.
 */

/* System headers FIRST: subsequent engine headers open namespace sp::engine,
 * and gcc-11 will resolve std::sqrt in random distribution templates through
 * that namespace's ADL context if the system includes are pulled in after. */
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

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_frobenius.h"
#include "../lib/shannon-prime/core/sp_ok_q8.h"
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

/* fp32 -> fp16 helper, matches the one in sp_ok_encode.cpp. */
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

/* Generate a Gemma3-flavoured fp16 weight buffer. */
static std::vector<uint16_t> make_weight_buffer(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    /* Empirical Gemma3 weight profile: zero mean, std ~ 0.05, occasional
     * outlier reaching |w| ~ 8.0 (the LM head spike). We mix two
     * distributions to capture both regimes. */
    std::normal_distribution<float> body(0.0f, 0.05f);
    std::uniform_real_distribution<float> spike(-8.0f, 8.0f);
    std::bernoulli_distribution is_spike(1.0 / 256.0);
    std::vector<uint16_t> w(n);
    for (size_t i = 0; i < n; ++i) {
        float v = is_spike(rng) ? spike(rng) : body(rng);
        w[i] = f32_to_f16(v);
    }
    return w;
}

/* ---------- Test 1: arena allocates q8 storage at expected size --------- */

static void arena_allocates_q8_at_2_bytes_per_element() {
    constexpr size_t N = 16384;
    sp::engine::sp_ok_arena arena(N * sizeof(sp_ok_q8_t) + 4096 /* slack */);
    ::sp_ok_q8_tensor t;
    CHECK(arena.alloc_tensor_q8(t, N), "alloc_tensor_q8 succeeded");
    CHECK(t.numel == N,        "numel set");
    CHECK(t.data  != nullptr,  "data pointer set");
    CHECK(t.q8_shift == 0,     "shift initialised to 0");
    CHECK(t.scale_recip == 1,  "scale_recip initialised to 1");
    CHECK(t.frobenius_scale == 1, "frobenius_scale initialised to 1");
    /* Arena should have consumed at least N * 2 bytes. */
    CHECK(arena.used() >= N * sizeof(sp_ok_q8_t),
          "arena consumed >= N * 2 bytes");
    /* Sanity-check the storage by writing + reading. */
    for (size_t i = 0; i < N; ++i) t.data[i] = sp_ok_q8_t{(int8_t)(i & 127), (int8_t)((-(int)i) & 127)};
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (t.data[i].a != (int8_t)(i & 127)) { ok = false; break; }
    }
    CHECK(ok, "round-trip read of packed bytes");
}

/* ---------- Test 2: combined encoder matches manual pipeline ------------ */

static void combined_encoder_matches_manual_pipeline() {
    constexpr size_t N = 4096;
    constexpr int64_t SCALE = 1 << 14;  /* 16384 = engine default */
    constexpr int64_t P = 41;
    constexpr int64_t K = 8;

    auto w = make_weight_buffer(N, 0xC0DECAFEBABE5EED);

    /* Path A: combined encoder. */
    sp::engine::sp_ok_arena arena_a(N * sizeof(sp_ok_q8_t) + 4096);
    ::sp_ok_q8_tensor q8_a;
    bool ok_a = sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(
        q8_a, w.data(), N, SCALE, P, K, arena_a, /*scratch=*/nullptr);
    CHECK(ok_a, "combined encoder returned true");

    /* Path B: manual reproduction. */
    sp::engine::sp_ok_arena arena_b(N * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor mid;
    int64_t shape[4] = { (int64_t)N, 1, 1, 1 };
    bool enc_ok = sp::engine::sp_ok_encode_from_fp16(mid, w.data(), 1, shape, SCALE, arena_b);
    CHECK(enc_ok, "manual fp16 -> sp_ok_t encode");
    sp::engine::sp_ok_encode_apply_frobenius_quant(mid, P, K);

    std::vector<sp_ok_q8_t> packed_b(N);
    int8_t shift_b = sp_ok_q8_encode_array(packed_b.data(), mid.data, N);

    /* Compare: same shift, byte-identical packed data, matching metadata. */
    CHECK(q8_a.q8_shift == shift_b, "shift matches manual pipeline");
    CHECK(q8_a.scale_recip == SCALE, "scale_recip propagated");
    CHECK(q8_a.frobenius_scale == mid.frobenius_scale,
          "frobenius_scale propagated");
    CHECK(q8_a.frobenius_p == (int16_t)P, "frobenius_p stored");
    CHECK(q8_a.frobenius_k == (int16_t)K, "frobenius_k stored");

    size_t mismatches = 0;
    for (size_t i = 0; i < N; ++i) {
        if (q8_a.data[i].a != packed_b[i].a) ++mismatches;
        if (q8_a.data[i].b != packed_b[i].b) ++mismatches;
    }
    std::fprintf(stderr,
        "  N=%zu shift=%d mismatches=%zu (out of %zu coord cells)\n",
        N, (int)shift_b, mismatches, 2 * N);
    CHECK(mismatches == 0, "combined encoder byte-identical to manual pipeline");
}

/* ---------- Test 3: decode recovers the post-Frobenius coordinates ------ */

static void packed_storage_decodes_to_post_frobenius() {
    constexpr size_t N = 2048;
    constexpr int64_t SCALE = 1 << 14;
    constexpr int64_t P = 41;
    constexpr int64_t K = 8;

    auto w = make_weight_buffer(N, 0xDEADBEEFFEEDFACE);

    /* Ground truth: build the post-Frobenius sp_ok_t array by hand. */
    sp::engine::sp_ok_arena truth_arena(N * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor truth;
    int64_t shape[4] = { (int64_t)N, 1, 1, 1 };
    bool ok = sp::engine::sp_ok_encode_from_fp16(truth, w.data(), 1, shape, SCALE, truth_arena);
    CHECK(ok, "encoded truth tensor");
    sp::engine::sp_ok_encode_apply_frobenius_quant(truth, P, K);

    /* Q8 pipeline output. */
    sp::engine::sp_ok_arena q8_arena(N * sizeof(sp_ok_q8_t) + 4096);
    ::sp_ok_q8_tensor q8;
    bool ok2 = sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(
        q8, w.data(), N, SCALE, P, K, q8_arena, nullptr);
    CHECK(ok2, "q8 encoder");

    /* Decode and compare against the truth post-Frobenius coords. */
    int64_t err_bound = sp_ok_q8_max_error(q8.q8_shift);
    int64_t worst_a = 0, worst_b = 0;
    double sum_rel = 0.0;
    size_t counted = 0;
    int64_t absmax = sp_ok_q8_absmax(truth.data, N);
    for (size_t i = 0; i < N; ++i) {
        sp_ok_t r = sp_ok_q8_decode_one(q8.data[i], q8.q8_shift);
        int64_t da = std::abs(r.a - truth.data[i].a);
        int64_t db = std::abs(r.b - truth.data[i].b);
        if (da > worst_a) worst_a = da;
        if (db > worst_b) worst_b = db;
        CHECK(da <= err_bound, "decoded.a within rounding step of truth");
        CHECK(db <= err_bound, "decoded.b within rounding step of truth");
        int64_t mag = std::max(std::abs(truth.data[i].a), std::abs(truth.data[i].b));
        if (mag >= absmax / 2 && mag > 0) {
            sum_rel += (double)std::max(da, db) / (double)mag;
            ++counted;
        }
    }
    double mean_rel = counted ? sum_rel / (double)counted : 0.0;
    std::fprintf(stderr,
        "  shift=%d  err_bound=%lld  worst_a=%lld worst_b=%lld  mean_rel(sat)=%.4f\n",
        (int)q8.q8_shift, (long long)err_bound,
        (long long)worst_a, (long long)worst_b, mean_rel);
    CHECK(mean_rel < 1.0 / 64.0,
          "post-Frobenius coords decoded within int8 quant budget");
}

/* ---------- Test 4: compression ratio is 8x vs raw sp_ok_t -------------- */

static void compression_ratio_is_eight() {
    constexpr size_t N = 1024;
    /* raw sp_ok_t = 16 B / element; packed = 2 B / element. */
    size_t raw    = N * sizeof(sp_ok_t);
    size_t packed = N * sizeof(sp_ok_q8_t);
    CHECK(packed * 8 == raw, "raw = 8 * packed (16 B / 2 B = 8x)");
    std::fprintf(stderr, "  raw=%zu B  packed=%zu B  ratio=%.1fx\n",
                 raw, packed, (double)raw / (double)packed);
}

/* ---------- Test 5: deterministic across runs --------------------------- */

static void encoder_is_deterministic() {
    constexpr size_t N = 4096;
    constexpr int64_t SCALE = 1 << 14;
    auto w = make_weight_buffer(N, 0x123456789ABCDEF0ULL);
    sp::engine::sp_ok_arena arena1(N * sizeof(sp_ok_q8_t) + 4096);
    sp::engine::sp_ok_arena arena2(N * sizeof(sp_ok_q8_t) + 4096);
    ::sp_ok_q8_tensor t1, t2;
    sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(t1, w.data(), N, SCALE, 41, 8, arena1, nullptr);
    sp::engine::sp_ok_encode_q8_from_fp16_with_frobenius(t2, w.data(), N, SCALE, 41, 8, arena2, nullptr);
    CHECK(t1.q8_shift == t2.q8_shift, "shift deterministic");
    CHECK(t1.scale_recip == t2.scale_recip, "scale deterministic");
    CHECK(t1.frobenius_scale == t2.frobenius_scale, "frobenius_scale deterministic");
    CHECK(std::memcmp(t1.data, t2.data, N * sizeof(sp_ok_q8_t)) == 0,
          "packed bytes deterministic");
}

int main() {
    std::fprintf(stderr,
        "test_sp_ok_q8_load_path: Phase 12 Step B-1 — arena + encoder parity\n");
    run("arena_allocates_q8_at_2_bytes_per_element",  arena_allocates_q8_at_2_bytes_per_element);
    run("combined_encoder_matches_manual_pipeline",   combined_encoder_matches_manual_pipeline);
    run("packed_storage_decodes_to_post_frobenius",   packed_storage_decodes_to_post_frobenius);
    run("compression_ratio_is_eight",                 compression_ratio_is_eight);
    run("encoder_is_deterministic",                   encoder_is_deterministic);
    std::fprintf(stderr,
        "[result] %s — %d check(s) %d failure(s)\n",
        g_failures == 0 ? "PASS" : "FAIL",
        g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
