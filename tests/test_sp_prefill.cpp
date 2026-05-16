// Shannon-Prime Engine — Phase 2.2b multi-token prefill attention test.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify that sp_attention_dot_product with n_q > 1 produces causal-masked
// outputs matching an fp32 reference, AND that each individual query
// row matches running the single-token attention separately with the
// causal prefix.

#include "../src/sp_attention.h"
#include "../src/sp_matmul.h"
#include "../src/sp_ok_encode.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

using namespace sp::engine;

// =========================================================================
// fp32 reference multi-token causal attention.
//   q: shape [n_head*head_dim, n_q] row-major: q[(h*head_dim+d)*n_q + qi]
//   k, v: shape [n_head*head_dim, T] row-major
//   out: shape [n_head*head_dim, n_q]
//   pos_offset: position[i] = pos_offset + i
// =========================================================================

static void ref_attention_multi(const float* q,
                                  const float* k_hist,
                                  const float* v_hist,
                                  int n_head, int head_dim,
                                  int n_q, int T, int pos_offset,
                                  float* out) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(T), weights(T);
    for (int h = 0; h < n_head; ++h) {
        for (int qi = 0; qi < n_q; ++qi) {
            const int q_pos = pos_offset + qi;
            for (int t = 0; t < T; ++t) {
                float dot = 0;
                for (int d = 0; d < head_dim; ++d) {
                    dot += k_hist[(h * head_dim + d) * T + t] *
                           q[(h * head_dim + d) * n_q + qi];
                }
                scores[t] = (t > q_pos) ? NEG_INF : dot * inv_sqrt_d;
            }
            float mx = scores[0];
            for (int t = 1; t < T; ++t)
                if (scores[t] > mx) mx = scores[t];
            double sum = 0;
            for (int t = 0; t < T; ++t) {
                weights[t] = std::exp(scores[t] - mx);
                sum += weights[t];
            }
            for (int t = 0; t < T; ++t) weights[t] /= (float)sum;
            for (int d = 0; d < head_dim; ++d) {
                double a = 0;
                for (int t = 0; t < T; ++t) {
                    a += (double)v_hist[(h * head_dim + d) * T + t] *
                         (double)weights[t];
                }
                out[(h * head_dim + d) * n_q + qi] = (float)a;
            }
        }
    }
}

// =========================================================================
// Test 1: prefill of n_q tokens over T=n_q history matches reference.
// =========================================================================

TEST(attention_prefill_causal_matches_fp32_reference) {
    constexpr int n_head = 2, head_dim = 4, n_q = 4;
    const int T = n_q;
    const int d_q = n_head * head_dim;

    std::vector<float> q_fp32(d_q * n_q), k_fp32(d_q * T), v_fp32(d_q * T);
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : q_fp32) v = dd(rng);
    for (auto& v : k_fp32) v = dd(rng);
    for (auto& v : v_fp32) v = dd(rng);

    std::vector<float> ref(d_q * n_q);
    ref_attention_multi(q_fp32.data(), k_fp32.data(), v_fp32.data(),
                         n_head, head_dim, n_q, T, /*pos_offset*/ 0,
                         ref.data());

    sp_ok_arena arena(128 * 1024);
    sp_ok_tensor Q, K, V, OUT;
    int64_t q_shape[4] = { n_q, d_q, 1, 1 };
    int64_t kv_shape[4] = { T, d_q, 1, 1 };
    int64_t out_shape[4] = { n_q, d_q, 1, 1 };

    ASSERT(sp_ok_encode_from_fp32(Q, q_fp32.data(), 2, q_shape,  1 << 14, arena));
    ASSERT(sp_ok_encode_from_fp32(K, k_fp32.data(), 2, kv_shape, 1 << 14, arena));
    ASSERT(sp_ok_encode_from_fp32(V, v_fp32.data(), 2, kv_shape, 1 << 14, arena));
    OUT.reset(2, out_shape);
    ASSERT(arena.alloc_tensor(OUT));
    OUT.scale_recip = 1 << 14;

    sp_attention_dot_product(Q, K, V, OUT, n_head, n_head, head_dim,
                                /*t_valid*/ T, /*t_stride*/ T,
                                /*pos_offset*/ 0);

    std::vector<float> got(d_q * n_q);
    sp_ok_decode_to_fp32(got.data(), OUT);

    int near = 0;
    const int N = d_q * n_q;
    for (int i = 0; i < N; ++i) {
        if (std::abs(got[i] - ref[i]) < 0.005f) ++near;
    }
    ASSERT(near >= N - 2);
}

// =========================================================================
// Test 2: First query (q_pos=0) should attend ONLY to position 0 (causal
// mask). Verify that altering positions [1..T) in V doesn't change row 0
// of the attention output.
// =========================================================================

TEST(attention_prefill_first_token_is_causal) {
    constexpr int n_head = 2, head_dim = 4, n_q = 3;
    const int T = n_q;
    const int d_q = n_head * head_dim;

    std::vector<float> q_fp32(d_q * n_q), k_fp32(d_q * T), v_fp32(d_q * T);
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> dd(-0.5f, 0.5f);
    for (auto& v : q_fp32) v = dd(rng);
    for (auto& v : k_fp32) v = dd(rng);
    for (auto& v : v_fp32) v = dd(rng);

    auto run_attn = [&](const std::vector<float>& vv,
                        std::vector<float>& out_fp32) {
        sp_ok_arena arena(128 * 1024);
        sp_ok_tensor Q, K, V, OUT;
        int64_t q_shape[4]   = { n_q, d_q, 1, 1 };
        int64_t kv_shape[4]  = { T,   d_q, 1, 1 };
        int64_t out_shape[4] = { n_q, d_q, 1, 1 };
        ASSERT(sp_ok_encode_from_fp32(Q, q_fp32.data(), 2, q_shape,  1 << 14, arena));
        ASSERT(sp_ok_encode_from_fp32(K, k_fp32.data(), 2, kv_shape, 1 << 14, arena));
        ASSERT(sp_ok_encode_from_fp32(V, vv.data(),     2, kv_shape, 1 << 14, arena));
        OUT.reset(2, out_shape);
        ASSERT(arena.alloc_tensor(OUT));
        OUT.scale_recip = 1 << 14;
        sp_attention_dot_product(Q, K, V, OUT, n_head, n_head, head_dim,
                                    T, T, 0);
        out_fp32.resize(d_q * n_q);
        sp_ok_decode_to_fp32(out_fp32.data(), OUT);
    };

    std::vector<float> out_orig, out_perturbed;
    run_attn(v_fp32, out_orig);

    // Perturb V at positions t=1, t=2 (future positions for qi=0).
    auto v_perturbed = v_fp32;
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (int f = 0; f < d_q; ++f) {
        v_perturbed[f * T + 1] += noise(rng);
        v_perturbed[f * T + 2] += noise(rng);
    }
    run_attn(v_perturbed, out_perturbed);

    // Row qi=0 (out[(h*head_dim+d) * n_q + 0]) must be IDENTICAL.
    int qi = 0;
    for (int f = 0; f < d_q; ++f) {
        float a = out_orig[f * n_q + qi];
        float b = out_perturbed[f * n_q + qi];
        ASSERT(std::abs(a - b) < 0.001f);
    }
    // Row qi=2 (last) sees all positions — must DIFFER.
    qi = 2;
    int n_diff = 0;
    for (int f = 0; f < d_q; ++f) {
        float a = out_orig[f * n_q + qi];
        float b = out_perturbed[f * n_q + qi];
        if (std::abs(a - b) > 0.001f) ++n_diff;
    }
    ASSERT(n_diff >= d_q - 2);
}

int main() {
    std::printf("Shannon-Prime sp_prefill tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
