/* test_sp_ok_q4_load_path.cpp — Phase 14 Q4 load-path parity test.
 *
 * Verifies the combined fp16 -> Frobenius -> 4-bit-pack pipeline that the
 * Q4 disk-shrink load shim will use. Mirrors test_sp_ok_q8_load_path with
 * the codebook halved.
 *
 * Specifically: sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius()
 * must produce the same packed output as calling the existing pieces
 * (sp_ok_encode_from_fp16 + sp_ok_encode_apply_frobenius_quant +
 * sp_ok_q4_encode_array) by hand.
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

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_frobenius.h"
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

static std::vector<uint16_t> make_weight_buffer(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
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

/* ---------- Test 1: arena allocates q4 storage at 1 byte/element -------- */

static void arena_allocates_q4_at_1_byte_per_element() {
    constexpr size_t N = 16384;
    sp::engine::sp_ok_arena arena(N * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor t;
    CHECK(arena.alloc_tensor_q4(t, N), "alloc_tensor_q4 succeeded");
    CHECK(t.numel == N,        "numel set");
    CHECK(t.data  != nullptr,  "data pointer set");
    CHECK(t.q4_shift == 0,     "shift initialised to 0");
    CHECK(t.scale_recip == 1,  "scale_recip initialised to 1");
    CHECK(t.frobenius_scale == 1, "frobenius_scale initialised to 1");
    CHECK(arena.used() >= N * sizeof(sp_ok_q4_t),
          "arena consumed >= N * 1 bytes");
    /* Sanity-check the storage by writing + reading. */
    for (size_t i = 0; i < N; ++i) t.data[i].packed = (uint8_t)(i & 0xFF);
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (t.data[i].packed != (uint8_t)(i & 0xFF)) { ok = false; break; }
    }
    CHECK(ok, "round-trip read of packed bytes");
}

/* ---------- Test 2: combined encoder matches manual pipeline ------------ */

static void combined_encoder_matches_manual_pipeline() {
    constexpr size_t N = 4096;
    constexpr int64_t SCALE = 1 << 14;
    constexpr int64_t P = 41;
    constexpr int64_t K = 8;

    auto w = make_weight_buffer(N, 0xC0DECAFEBABE5EED);

    /* Path A: combined encoder. */
    sp::engine::sp_ok_arena arena_a(N * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor q4_a;
    bool ok_a = sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(
        q4_a, w.data(), N, SCALE, P, K, arena_a, /*scratch=*/nullptr);
    CHECK(ok_a, "combined encoder returned true");

    /* Path B: manual reproduction. */
    sp::engine::sp_ok_arena arena_b(N * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor mid;
    int64_t shape[4] = { (int64_t)N, 1, 1, 1 };
    bool enc_ok = sp::engine::sp_ok_encode_from_fp16(mid, w.data(), 1, shape, SCALE, arena_b);
    CHECK(enc_ok, "manual fp16 -> sp_ok_t encode");
    sp::engine::sp_ok_encode_apply_frobenius_quant(mid, P, K);

    std::vector<sp_ok_q4_t> packed_b(N);
    int8_t shift_b = sp_ok_q4_encode_array(packed_b.data(), mid.data, N);

    CHECK(q4_a.q4_shift == shift_b, "shift matches manual pipeline");
    CHECK(q4_a.scale_recip == SCALE, "scale_recip propagated");
    CHECK(q4_a.frobenius_scale == mid.frobenius_scale,
          "frobenius_scale propagated");
    CHECK(q4_a.frobenius_p == (int16_t)P, "frobenius_p stored");
    CHECK(q4_a.frobenius_k == (int16_t)K, "frobenius_k stored");

    size_t mismatches = 0;
    for (size_t i = 0; i < N; ++i) {
        if (q4_a.data[i].packed != packed_b[i].packed) ++mismatches;
    }
    std::fprintf(stderr,
        "  N=%zu shift=%d mismatches=%zu (out of %zu packed bytes)\n",
        N, (int)shift_b, mismatches, N);
    CHECK(mismatches == 0, "combined encoder byte-identical to manual pipeline");
}

/* ---------- Test 3: decode recovers post-Frobenius coords --------------- */

static void packed_storage_decodes_to_post_frobenius() {
    constexpr size_t N = 2048;
    constexpr int64_t SCALE = 1 << 14;
    constexpr int64_t P = 41;
    constexpr int64_t K = 8;

    auto w = make_weight_buffer(N, 0xDEADBEEFFEEDFACE);

    sp::engine::sp_ok_arena truth_arena(N * sizeof(sp_ok_t) + 4096);
    sp::engine::sp_ok_tensor truth;
    int64_t shape[4] = { (int64_t)N, 1, 1, 1 };
    bool ok = sp::engine::sp_ok_encode_from_fp16(truth, w.data(), 1, shape, SCALE, truth_arena);
    CHECK(ok, "encoded truth tensor");
    sp::engine::sp_ok_encode_apply_frobenius_quant(truth, P, K);

    sp::engine::sp_ok_arena q4_arena(N * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor q4;
    bool ok2 = sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(
        q4, w.data(), N, SCALE, P, K, q4_arena, nullptr);
    CHECK(ok2, "q4 encoder");

    int64_t err_bound = sp_ok_q4_max_error(q4.q4_shift);
    int64_t worst_a = 0, worst_b = 0;
    for (size_t i = 0; i < N; ++i) {
        sp_ok_t r = sp_ok_q4_decode_one(q4.data[i], q4.q4_shift);
        int64_t da = std::abs(r.a - truth.data[i].a);
        int64_t db = std::abs(r.b - truth.data[i].b);
        if (da > worst_a) worst_a = da;
        if (db > worst_b) worst_b = db;
        CHECK(da <= err_bound, "decoded.a within rounding step of truth");
        CHECK(db <= err_bound, "decoded.b within rounding step of truth");
    }
    std::fprintf(stderr,
        "  shift=%d  err_bound=%lld  worst_a=%lld worst_b=%lld\n",
        (int)q4.q4_shift, (long long)err_bound,
        (long long)worst_a, (long long)worst_b);
}

/* ---------- Test 4: 16x compression ratio vs raw sp_ok_t ---------------- */

static void compression_ratio_is_sixteen() {
    constexpr size_t N = 1024;
    size_t raw    = N * sizeof(sp_ok_t);
    size_t packed = N * sizeof(sp_ok_q4_t);
    CHECK(packed * 16 == raw, "raw = 16 * packed (16 B / 1 B = 16x)");
    std::fprintf(stderr, "  raw=%zu B  packed=%zu B  ratio=%.1fx\n",
                 raw, packed, (double)raw / (double)packed);
}

/* ---------- Test 5: deterministic across runs --------------------------- */

static void encoder_is_deterministic() {
    constexpr size_t N = 4096;
    constexpr int64_t SCALE = 1 << 14;
    auto w = make_weight_buffer(N, 0x123456789ABCDEF0ULL);
    sp::engine::sp_ok_arena arena1(N * sizeof(sp_ok_q4_t) + 4096);
    sp::engine::sp_ok_arena arena2(N * sizeof(sp_ok_q4_t) + 4096);
    ::sp_ok_q4_tensor t1, t2;
    sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(t1, w.data(), N, SCALE, 41, 8, arena1, nullptr);
    sp::engine::sp_ok_encode_q4_from_fp16_with_frobenius(t2, w.data(), N, SCALE, 41, 8, arena2, nullptr);
    CHECK(t1.q4_shift == t2.q4_shift, "shift deterministic");
    CHECK(t1.scale_recip == t2.scale_recip, "scale deterministic");
    CHECK(t1.frobenius_scale == t2.frobenius_scale, "frobenius_scale deterministic");
    CHECK(std::memcmp(t1.data, t2.data, N * sizeof(sp_ok_q4_t)) == 0,
          "packed bytes deterministic");
}

int main() {
    std::fprintf(stderr,
        "test_sp_ok_q4_load_path: Phase 14 — Q4 arena + encoder parity\n");
    run("arena_allocates_q4_at_1_byte_per_element",  arena_allocates_q4_at_1_byte_per_element);
    run("combined_encoder_matches_manual_pipeline",  combined_encoder_matches_manual_pipeline);
    run("packed_storage_decodes_to_post_frobenius",  packed_storage_decodes_to_post_frobenius);
    run("compression_ratio_is_sixteen",              compression_ratio_is_sixteen);
    run("encoder_is_deterministic",                  encoder_is_deterministic);
    std::fprintf(stderr,
        "[result] %s — %d check(s) %d failure(s)\n",
        g_failures == 0 ? "PASS" : "FAIL",
        g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
