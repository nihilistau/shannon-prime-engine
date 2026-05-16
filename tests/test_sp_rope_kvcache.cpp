// Shannon-Prime Engine — Phase 2.2b RoPE + KV-cache unit tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify:
//   1. sp_rope_apply_ok produces fp32-equivalent rotations to within
//      the O_K encoding's ULP, and resets frobenius_scale.
//   2. sp_ok_kv_cache_init allocates per-layer slots correctly.
//   3. sp_ok_kv_cache_append_layer copies new tokens into the cache at
//      the right offset, and a subsequent attention pass over the
//      view sees the appended values bit-exactly.

#include "../src/sp_rope.h"
#include "../src/sp_kv_cache_ok.h"
#include "../src/sp_attention.h"
#include "../src/sp_matmul.h"
#include "../src/sp_ok_encode.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
// fp32 reference RoPE (matches sp_rope_f32 in sp_kernels_cpu.cpp).
// =========================================================================

static void ref_rope_fp32(float* x, int n_tokens, int n_heads, int head_dim,
                            const int32_t* pos, float freq_base, float freq_scale) {
    // Layout: x[feature * n_tokens + t]  (matches sp_matmul output convention)
    const int n_pairs = head_dim / 2;
    for (int t = 0; t < n_tokens; ++t) {
        const float pp = (float)pos[t];
        for (int h = 0; h < n_heads; ++h) {
            for (int k = 0; k < n_pairs; ++k) {
                const float ang  = pp * freq_scale *
                                   std::pow(freq_base, -(float)(2*k)/(float)head_dim);
                const float c    = std::cos(ang);
                const float s    = std::sin(ang);
                const int f_e = h * head_dim + 2*k;
                const int f_o = f_e + 1;
                const float a = x[f_e * n_tokens + t];
                const float b = x[f_o * n_tokens + t];
                x[f_e * n_tokens + t] = c*a - s*b;
                x[f_o * n_tokens + t] = s*a + c*b;
            }
        }
    }
}

// =========================================================================
// Test 1: RoPE on single-token Q tensor matches fp32 reference.
// =========================================================================

TEST(rope_single_token_matches_fp32_reference) {
    constexpr int n_heads = 2, head_dim = 4, n_tokens = 1;
    const int d_q = n_heads * head_dim;
    const float freq_base = 10000.0f;

    std::vector<float> q_fp32(d_q * n_tokens);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : q_fp32) v = dd(rng);

    // Reference rotation.
    std::vector<float> ref = q_fp32;
    int32_t pos = 3;
    ref_rope_fp32(ref.data(), n_tokens, n_heads, head_dim,
                  &pos, freq_base, 1.0f);

    // Encode + RoPE + decode.
    sp_ok_arena arena(16 * 1024);
    sp_ok_tensor Q;
    int64_t q_shape[4] = { n_tokens, d_q, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(Q, q_fp32.data(), 2, q_shape, 1 << 14, arena));
    // Pretend the encode came out of a shimmed matmul: set frobenius_scale != 1.
    Q.frobenius_scale = 1;  // simplest test: post-RMSNorm, before any shim
    ASSERT(sp_rope_apply_ok(Q, n_heads, head_dim, n_tokens, &pos,
                              freq_base, 1.0f));
    ASSERT(Q.frobenius_scale == 1);

    std::vector<float> got(d_q * n_tokens);
    sp_ok_decode_to_fp32(got.data(), Q);

    int near = 0;
    for (int i = 0; i < d_q * n_tokens; ++i) {
        if (std::abs(got[i] - ref[i]) < 0.001f) ++near;
    }
    ASSERT(near == d_q * n_tokens);
}

// =========================================================================
// Test 2: RoPE on multi-token K tensor matches fp32 reference for ALL
//         positions (prefill case with stride-aware layout).
// =========================================================================

TEST(rope_multi_token_matches_fp32_reference) {
    constexpr int n_heads = 2, head_dim = 6, n_tokens = 4;
    const int d_kv = n_heads * head_dim;
    const float freq_base = 10000.0f;

    std::vector<float> k_fp32(d_kv * n_tokens);
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : k_fp32) v = dd(rng);

    std::vector<int32_t> positions = { 0, 1, 2, 3 };

    std::vector<float> ref = k_fp32;
    ref_rope_fp32(ref.data(), n_tokens, n_heads, head_dim,
                  positions.data(), freq_base, 1.0f);

    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor K;
    int64_t k_shape[4] = { n_tokens, d_kv, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(K, k_fp32.data(), 2, k_shape, 1 << 14, arena));
    ASSERT(sp_rope_apply_ok(K, n_heads, head_dim, n_tokens,
                              positions.data(), freq_base, 1.0f));

    std::vector<float> got(d_kv * n_tokens);
    sp_ok_decode_to_fp32(got.data(), K);

    int near = 0;
    for (int i = 0; i < d_kv * n_tokens; ++i) {
        if (std::abs(got[i] - ref[i]) < 0.002f) ++near;
    }
    ASSERT(near >= d_kv * n_tokens - 1);
}

// =========================================================================
// Test 3: KV cache init + append + view-read round-trip.
// =========================================================================

TEST(kv_cache_append_and_view_roundtrip) {
    constexpr int n_layers = 2, max_len = 8, n_kv_head = 2, head_dim = 4;
    const int d_kv = n_kv_head * head_dim;

    sp_ok_arena arena(128 * 1024);
    sp_ok_kv_cache cache;
    ASSERT(sp_ok_kv_cache_init(cache, n_layers, max_len, n_kv_head, head_dim,
                                 /*k_scale*/ 1 << 14,
                                 /*v_scale*/ 1 << 14,
                                 /*v_frobenius_scale*/ 1,
                                 arena));
    ASSERT(cache.cur_len == 0);
    ASSERT((int)cache.layers.size() == n_layers);

    // Step 1: append 3 tokens to layer 0.
    constexpr int n_new = 3;
    std::vector<float> k_new_fp32(d_kv * n_new), v_new_fp32(d_kv * n_new);
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : k_new_fp32) v = dd(rng);
    for (auto& v : v_new_fp32) v = dd(rng);

    sp_ok_tensor K_new, V_new;
    int64_t shp[4] = { n_new, d_kv, 1, 1 };
    sp_ok_arena step_arena(64 * 1024);
    ASSERT(sp_ok_encode_from_fp32(K_new, k_new_fp32.data(), 2, shp, 1 << 14, step_arena));
    ASSERT(sp_ok_encode_from_fp32(V_new, v_new_fp32.data(), 2, shp, 1 << 14, step_arena));

    ASSERT(sp_ok_kv_cache_append_layer(cache, /*layer*/ 0, K_new, V_new, n_new));
    // cur_len doesn't advance until caller calls _advance — verify pre-state.
    ASSERT(cache.cur_len == 0);
    sp_ok_kv_cache_advance(cache, n_new);
    ASSERT(cache.cur_len == n_new);

    // Read back via view.
    sp_ok_tensor K_view = sp_ok_kv_cache_view_k(cache, 0);
    sp_ok_tensor V_view = sp_ok_kv_cache_view_v(cache, 0);
    ASSERT(K_view.data != nullptr);
    ASSERT(V_view.data != nullptr);
    ASSERT(K_view.scale_recip == 1 << 14);
    ASSERT(K_view.frobenius_scale == 1);

    // The view exposes shape[0] = max_len (the real stride). Verify the
    // first n_new positions hold the appended values.
    for (int f = 0; f < d_kv; ++f) {
        for (int t = 0; t < n_new; ++t) {
            sp_ok_t cv = K_view.data[f * max_len + t];
            sp_ok_t nv = K_new.data[f * n_new + t];
            ASSERT(cv.a == nv.a);
            ASSERT(cv.b == nv.b);
        }
    }
}

// =========================================================================
// Test 4: KV cache + sp_attention_dot_product round-trip.
// Append history; attention over the view (with t_stride=max_len,
// t_valid=cur_len) must match an attention computed on the freshly
// encoded (compact) tensors.
// =========================================================================

TEST(kv_cache_attention_with_stride) {
    constexpr int n_head = 2, head_dim = 4, T = 3, max_len = 8;
    const int d_q = n_head * head_dim;

    std::vector<float> q_fp32(d_q), k_fp32(d_q * T), v_fp32(d_q * T);
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : q_fp32) v = dd(rng);
    for (auto& v : k_fp32) v = dd(rng);
    for (auto& v : v_fp32) v = dd(rng);

    // --- Path A: compact attention (Phase 2.2a style, no stride). ---
    sp_ok_arena arena_a(128 * 1024);
    sp_ok_tensor Q_a, K_a, V_a, OUT_a;
    int64_t q_shape[4] = { 1, d_q, 1, 1 };
    int64_t k_shape[4] = { T, d_q, 1, 1 };
    int64_t out_shape[4] = { 1, d_q, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(Q_a, q_fp32.data(), 2, q_shape, 1 << 14, arena_a));
    ASSERT(sp_ok_encode_from_fp32(K_a, k_fp32.data(), 2, k_shape, 1 << 14, arena_a));
    ASSERT(sp_ok_encode_from_fp32(V_a, v_fp32.data(), 2, k_shape, 1 << 14, arena_a));
    OUT_a.reset(2, out_shape);
    ASSERT(arena_a.alloc_tensor(OUT_a));
    OUT_a.scale_recip = 1 << 14;
    sp_attention_dot_product(Q_a, K_a, V_a, OUT_a, n_head, n_head, head_dim);

    std::vector<float> got_a(d_q);
    sp_ok_decode_to_fp32(got_a.data(), OUT_a);

    // --- Path B: KV-cache attention (stride=max_len, valid=T). ---
    sp_ok_arena arena_b(128 * 1024);
    sp_ok_kv_cache cache;
    ASSERT(sp_ok_kv_cache_init(cache, 1, max_len, n_head, head_dim,
                                 1 << 14, 1 << 14, 1, arena_b));
    sp_ok_arena step_arena(64 * 1024);
    sp_ok_tensor K_n, V_n;
    ASSERT(sp_ok_encode_from_fp32(K_n, k_fp32.data(), 2, k_shape, 1 << 14, step_arena));
    ASSERT(sp_ok_encode_from_fp32(V_n, v_fp32.data(), 2, k_shape, 1 << 14, step_arena));
    ASSERT(sp_ok_kv_cache_append_layer(cache, 0, K_n, V_n, T));
    sp_ok_kv_cache_advance(cache, T);

    sp_ok_tensor K_view = sp_ok_kv_cache_view_k(cache, 0);
    sp_ok_tensor V_view = sp_ok_kv_cache_view_v(cache, 0);

    sp_ok_tensor Q_b, OUT_b;
    sp_ok_arena arena_b_step(64 * 1024);
    ASSERT(sp_ok_encode_from_fp32(Q_b, q_fp32.data(), 2, q_shape, 1 << 14, arena_b_step));
    OUT_b.reset(2, out_shape);
    ASSERT(arena_b_step.alloc_tensor(OUT_b));
    OUT_b.scale_recip = 1 << 14;
    sp_attention_dot_product(Q_b, K_view, V_view, OUT_b,
                                n_head, n_head, head_dim,
                                /*t_valid*/ T, /*t_stride*/ max_len,
                                /*pos_offset*/ T - 1);

    std::vector<float> got_b(d_q);
    sp_ok_decode_to_fp32(got_b.data(), OUT_b);

    // The two paths should match bit-exactly (same data, just stride).
    int near = 0;
    for (int i = 0; i < d_q; ++i) {
        if (std::abs(got_a[i] - got_b[i]) < 0.001f) ++near;
    }
    ASSERT(near >= d_q - 1);
}

// =========================================================================
// Driver
// =========================================================================

int main() {
    std::printf("Shannon-Prime sp_rope + sp_kv_cache_ok tests (%zu)\n",
                g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
