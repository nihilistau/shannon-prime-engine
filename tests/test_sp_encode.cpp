// Shannon-Prime Engine — unit tests for sp_ok_tensor + sp_ok_encode + sp_sampler.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Tests Phase 1.2 + Phase 1.3 + Phase 2-prep deliverables.

#include "../src/sp_ok_tensor.h"
#include "../src/sp_ok_encode.h"
#include "../src/sp_sampler.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_frobenius.h"
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
// sp_ok_tensor + sp_ok_arena
// =========================================================================

TEST(tensor_reset_strides_contiguous) {
    sp_ok_tensor t;
    int64_t shape[4] = { 4, 8, 1, 1 };
    t.reset(2, shape);
    ASSERT(t.n_dims == 2);
    ASSERT(t.shape[0] == 4);
    ASSERT(t.shape[1] == 8);
    ASSERT(t.strides[0] == sizeof(sp_ok_t));
    ASSERT(t.strides[1] == sizeof(sp_ok_t) * 4);
    ASSERT(t.is_contiguous());
}

TEST(arena_alloc_tensor) {
    sp_ok_arena arena(1024 * 1024);
    sp_ok_tensor t;
    int64_t shape[4] = { 100, 1, 1, 1 };
    t.reset(1, shape);
    ASSERT(arena.alloc_tensor(t));
    ASSERT(t.data != nullptr);
    // Write + read
    for (int i = 0; i < 100; ++i) t.data[i] = sp_ok_t{ i, -i };
    for (int i = 0; i < 100; ++i) {
        ASSERT(t.data[i].a == (int64_t)i);
        ASSERT(t.data[i].b == (int64_t)-i);
    }
}

TEST(arena_reset_rewinds_without_freeing) {
    sp_ok_arena arena(4096);
    sp_ok_tensor t;
    int64_t shape[4] = { 16, 1, 1, 1 };
    t.reset(1, shape);
    ASSERT(arena.alloc_tensor(t));
    size_t used_before = arena.used();
    ASSERT(used_before > 0);
    arena.reset();
    ASSERT(arena.used() == 0);
    ASSERT(arena.capacity() >= 4096);
}

// =========================================================================
// fp16 / fp32 <-> O_K round-trip
// =========================================================================

TEST(encode_fp32_decode_fp32_roundtrip) {
    sp_ok_arena arena(1024 * 1024);
    sp_ok_tensor t;
    constexpr int N = 128;
    float w[N];
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) w[i] = d(rng);
    int64_t shape[4] = { N, 1, 1, 1 };
    int64_t scale = 1LL << 18;  // 2^18 — fp32 has ~24 bits of mantissa
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, scale, arena));
    float decoded[N];
    sp_ok_decode_to_fp32(decoded, t);
    // Round-trip error bound: max_input * 2 / scale (one ULP each way).
    const float tol = 2.0f / (float)scale;
    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(decoded[i], w[i], tol);
    }
}

TEST(encode_b_component_is_zero) {
    sp_ok_arena arena(8192);
    sp_ok_tensor t;
    float w[16] = { 0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                    1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f, 0.125f, -0.125f };
    int64_t shape[4] = { 16, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, 1024, arena));
    for (int i = 0; i < 16; ++i) {
        ASSERT(t.data[i].b == 0);
    }
}

// =========================================================================
// Frobenius shim
// =========================================================================

TEST(shim_inert_phi_2_squared_scales_by_minus_2) {
    // Phase 1.8 signed convention: phi_p^(2m) = (-p)^m as a SIGNED scalar.
    // For inert p=2, k=2: m=1, so phi_2^2 = (-2)^1 = -2. The a-component
    // is multiplied by -2 AND frobenius_scale tracks the signed factor -2,
    // so decode (a / (scale_recip * frobenius_scale)) recovers +w exactly.
    // Pre-fix, frobenius_scale was unsigned (= 2) and decode produced -w,
    // breaking SwiGLU since silu(-x) != -silu(x).
    sp_ok_arena arena(8192);
    sp_ok_tensor t;
    float w[8] = { 0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f };
    int64_t shape[4] = { 8, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, 1024, arena));
    int64_t pre[8];
    for (int i = 0; i < 8; ++i) pre[i] = t.data[i].a;

    sp_ok_encode_apply_frobenius_quant(t, 2, 2);

    // a-components scaled by -2 (the shim applies the signed Frobenius).
    for (int i = 0; i < 8; ++i) {
        ASSERT(t.data[i].a == -2 * pre[i]);
    }
    // frobenius_scale tracks the SIGNED factor (Phase 1.8 fix).
    ASSERT(t.frobenius_scale == -2);

    // Decode: a_new / (scale_recip * (-2)) = (-2 * a_orig) / (scale_recip * (-2))
    //       = a_orig / scale_recip = +w. Sign-flip cancellation.
    float decoded[8];
    sp_ok_decode_to_fp32(decoded, t);
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decoded[i], +w[i], 2.0f / 1024.0f);
    }
}

TEST(shim_split_p41_k8_norm_growth) {
    // Phase 1.8 signed-pi^k.a convention: for split p, the shim applies
    // phi_p^k = pi^k to each O_K element. Norm growth is still p^k
    // (algebraic invariant of Frobenius), but frobenius_scale tracks
    // pi^k.a — the SIGNED real-component scaling that actually applies
    // to (a, 0) inputs — NOT the norm-based p^(k/2).
    //
    // Pre-fix: frobenius_scale = p^(k/2) = 41^4 over-corrected by a
    //          factor of pi^k.a / |pi^k| = cos(theta_k) < 1, shrinking
    //          decoded weights and exploding PPL ~49 at Config B.
    sp_ok_arena arena(1024 * 64);
    sp_ok_tensor t;
    constexpr int N = 32;
    float w[N];
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (int i = 0; i < N; ++i) w[i] = d(rng);
    int64_t shape[4] = { N, 1, 1, 1 };
    int64_t scale = 1LL << 12;
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, scale, arena));

    int64_t pre_norm_sum = sp_ok_tensor_sum_norms(t);
    sp_ok_encode_apply_frobenius_quant(t, 41, 8);

    int64_t post_norm_sum = sp_ok_tensor_sum_norms(t);
    int64_t expected_growth = 1;
    for (int i = 0; i < 8; ++i) expected_growth *= 41;
    ASSERT(post_norm_sum == pre_norm_sum * expected_growth);

    // Expected frobenius_scale = pi^8.a (signed real component).
    sp_ok_t pi;
    ASSERT(sp_find_element_of_norm(41, &pi));
    sp_ok_t pi_pow = sp_ok_pow(pi, 8);
    ASSERT(t.frobenius_scale == pi_pow.a);

    // Sanity: pi^8.a must be strictly less than 41^4 (the old wrong scale)
    // — this is the bug's empirical signature.
    int64_t old_wrong_scale = 1;
    for (int i = 0; i < 4; ++i) old_wrong_scale *= 41;
    ASSERT(std::llabs(pi_pow.a) < old_wrong_scale);
}

TEST(shim_sato_tate_mix_norm_growth_2_2_41_4) {
    sp_ok_arena arena(1024 * 16);
    sp_ok_tensor t;
    constexpr int N = 16;
    float w[N];
    for (int i = 0; i < N; ++i) w[i] = 0.1f * (i - 8);
    int64_t shape[4] = { N, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, 1024, arena));

    int64_t pre_norm_sum = sp_ok_tensor_sum_norms(t);
    sp_ok_encode_apply_sato_tate_mix(t, 2, 2, 41, 4);

    // Expected norm growth: 2^2 * 41^4 (inert contributes p1^k1, split p2^k2).
    int64_t exp = 4;
    for (int i = 0; i < 4; ++i) exp *= 41;
    int64_t post_norm_sum = sp_ok_tensor_sum_norms(t);
    ASSERT(post_norm_sum == pre_norm_sum * exp);
}

// =========================================================================
// In-place fp16 round-trip via shim
// =========================================================================

// fp16 helpers identical to those in sp_ok_encode.cpp
static inline uint16_t fp32_to_fp16(float v) {
    uint32_t f; std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

TEST(roundtrip_fp16_inplace_no_frobenius_recovers_orig) {
    // Pure round-trip (k=0 means no Frobenius application): encode →
    // decode must recover the original fp16 buffer to one ULP.
    //
    // NOTE: a *non-trivial* Frobenius (e.g. p=41,k=8) produces decoded
    // fp16 values scaled by N(pi)^(k/2) = 41^4 ≈ 2.8M — way out of fp16
    // range. Theorem 4 says the cancellation happens *inside QK·V*,
    // not in a single-tensor round-trip. So the only meaningful fp16
    // round-trip test for the shim is with k=0.
    constexpr size_t N = 64;
    std::vector<uint16_t> buf(N);
    std::vector<float> orig(N);
    for (size_t i = 0; i < N; ++i) {
        orig[i] = 0.01f * (float)((int)i - 32);
        buf[i] = fp32_to_fp16(orig[i]);
    }
    std::vector<uint16_t> buf_copy = buf;
    double scale_applied = sp_ok_apply_frobenius_quant_inplace_fp16(
        buf.data(), N, 41, 0, 1LL << 16);
    ASSERT(scale_applied == 1.0);
    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        // Allow 1-ULP fp16 difference (the round-trip goes via int64).
        int diff = (int)buf[i] - (int)buf_copy[i];
        if (diff >= -1 && diff <= 1) matched++;
    }
    ASSERT(matched >= (int)N - 2);
}

TEST(roundtrip_fp16_with_frobenius_norm_grows_predictably) {
    // With non-trivial Frobenius, fp16 values are no longer recoverable
    // by a single-tensor decode (the Theorem 4 cancellation requires
    // attention's QK·V structure). But the underlying sp_ok_tensor's
    // norm sum must grow by exactly p^k. We verify that property here
    // by encoding to fp32, applying the shim, and checking the
    // *norm-summed* int64 values pre/post.
    constexpr size_t N = 16;
    sp_ok_arena arena(1024 * 8);
    sp_ok_tensor t;
    float w[N];
    for (size_t i = 0; i < N; ++i) w[i] = 0.1f * (float)((int)i - 8);
    int64_t shape[4] = { (int64_t)N, 1, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(t, w, 1, shape, 1024, arena));
    int64_t pre = sp_ok_tensor_sum_norms(t);
    sp_ok_encode_apply_frobenius_quant(t, 41, 8);
    int64_t post = sp_ok_tensor_sum_norms(t);
    int64_t exp = 1;
    for (int i = 0; i < 8; ++i) exp *= 41;
    ASSERT(post == pre * exp);
}

// =========================================================================
// sp_sampler
// =========================================================================

TEST(sampler_softmax_sums_to_1) {
    float logits[5] = { 1.0f, 2.0f, 0.5f, -1.0f, 0.0f };
    float probs[5];
    sp_sampler_softmax(logits, 5, probs);
    float s = 0.0f;
    for (int i = 0; i < 5; ++i) s += probs[i];
    ASSERT_NEAR(s, 1.0f, 1e-5f);
}

TEST(sampler_top_k_masks_lower) {
    float logits[6] = { 5.0f, 3.0f, 4.0f, 1.0f, 2.0f, 0.0f };
    sp_sampler_apply_top_k(logits, 6, 3);
    // Top-3 are indices 0, 2, 1 (values 5, 4, 3). Others masked.
    ASSERT(logits[0] == 5.0f);
    ASSERT(logits[2] == 4.0f);
    ASSERT(logits[1] == 3.0f);
    ASSERT(logits[3] == -FLT_MAX);
    ASSERT(logits[4] == -FLT_MAX);
    ASSERT(logits[5] == -FLT_MAX);
}

TEST(sampler_deterministic_seed) {
    float logits[4] = { 1.0f, 2.0f, 0.5f, 0.0f };
    sp_sampler_params p;
    p.seed = 42;
    uint64_t s1 = p.seed, s2 = p.seed;
    float l1[4]; std::memcpy(l1, logits, sizeof(logits));
    float l2[4]; std::memcpy(l2, logits, sizeof(logits));
    int32_t t1 = sp_sampler_step(l1, 4, {}, p, &s1);
    int32_t t2 = sp_sampler_step(l2, 4, {}, p, &s2);
    ASSERT(t1 == t2);
}

// =========================================================================
// Driver
// =========================================================================

#include <cfloat>

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("Shannon-Prime Encode + Sampler unit tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
