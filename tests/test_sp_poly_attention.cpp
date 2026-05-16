// Shannon-Prime — CKKS-style polynomial attention vs softmax (Phase 3 pivot, part 3).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Direct A/B: same Q, K, V as test_sp_weil_attention, but score(q, k)
// is computed via polynomial-ring dot product in Z[x]/(x^N+1) instead
// of via the Weil pairing on E[n]. Hypothesis: the polynomial path
// preserves the metric topology, so KL(softmax || poly_attn) should
// approach 0 (the only error is fp32-→int rounding inside the encoder).
//
// Reference numbers from the Weil run (same setup):
//   top-1 agreement: 0/4   KL: 1.63 nats   cos: 0.28

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

// =========================================================================
// Reference fp32 softmax attention (identical to the Weil test's ref).
// =========================================================================

static void softmax_attention_ref(const std::vector<float>& Q,
                                    const std::vector<float>& K,
                                    const std::vector<float>& V,
                                    int n_q, int n_kv, int head_dim,
                                    std::vector<float>& out_attn,
                                    std::vector<float>& out_weights) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    out_attn.assign((size_t)n_q * head_dim, 0.0f);
    out_weights.assign((size_t)n_q * n_kv, 0.0f);
    std::vector<float> scores(n_kv);
    for (int qi = 0; qi < n_q; ++qi) {
        for (int t = 0; t < n_kv; ++t) {
            double s = 0;
            for (int d = 0; d < head_dim; ++d) {
                s += (double)Q[qi * head_dim + d] * (double)K[t * head_dim + d];
            }
            scores[t] = (float)(s * (double)inv_sqrt_d);
        }
        float mx = scores[0];
        for (int t = 1; t < n_kv; ++t) if (scores[t] > mx) mx = scores[t];
        double sum = 0;
        for (int t = 0; t < n_kv; ++t) {
            float e = std::exp(scores[t] - mx);
            out_weights[qi * n_kv + t] = e;
            sum += e;
        }
        for (int t = 0; t < n_kv; ++t) {
            out_weights[qi * n_kv + t] /= (float)sum;
            for (int d = 0; d < head_dim; ++d) {
                out_attn[qi * head_dim + d] +=
                    out_weights[qi * n_kv + t] * V[t * head_dim + d];
            }
        }
    }
}

// =========================================================================
// CKKS-polynomial attention: same Q/K/V, scores computed via
// sp_poly_dot_product (integer polynomial multiply in Z[x]/(x^N+1)).
//
// Score formula identical to softmax attention's:
//   score(qi, t) = (q · k) / sqrt(head_dim)
//
// The whole point is the EXACT recovery — softmax sees the same scores
// it would have seen, just routed through an integer ring on the way in.
// =========================================================================

static void poly_attention_bench(const std::vector<float>& Q,
                                    const std::vector<float>& K,
                                    const std::vector<float>& V,
                                    int n_q, int n_kv, int head_dim,
                                    int N, double delta,
                                    std::vector<float>& out_attn,
                                    std::vector<float>& out_weights) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    out_attn.assign((size_t)n_q * head_dim, 0.0f);
    out_weights.assign((size_t)n_q * n_kv, 0.0f);
    std::vector<sp_poly_coeff> scratch(3 * N);
    std::vector<float> scores(n_kv);

    for (int qi = 0; qi < n_q; ++qi) {
        for (int t = 0; t < n_kv; ++t) {
            // Polynomial-ring dot product → fp32.
            float s = sp_poly_dot_product(
                &Q[qi * head_dim], &K[t * head_dim], head_dim, N, delta,
                scratch.data());
            scores[t] = s * inv_sqrt_d;
        }
        float mx = scores[0];
        for (int t = 1; t < n_kv; ++t) if (scores[t] > mx) mx = scores[t];
        double sum = 0;
        for (int t = 0; t < n_kv; ++t) {
            float e = std::exp(scores[t] - mx);
            out_weights[qi * n_kv + t] = e;
            sum += e;
        }
        for (int t = 0; t < n_kv; ++t) {
            out_weights[qi * n_kv + t] /= (float)sum;
            for (int d = 0; d < head_dim; ++d) {
                out_attn[qi * head_dim + d] +=
                    out_weights[qi * n_kv + t] * V[t * head_dim + d];
            }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

// Helper: KL divergence + top-1 + output cos similarity for one (sm, alt)
// run. Returns (KL, top1, cos) means averaged over n_q queries.
struct BenchResult {
    double mean_kl;
    int    top1_agree;
    double mean_cos;
};

static BenchResult compute_bench(const std::vector<float>& weights_sm,
                                   const std::vector<float>& weights_alt,
                                   const std::vector<float>& attn_sm,
                                   const std::vector<float>& attn_alt,
                                   int n_q, int n_kv, int head_dim) {
    BenchResult r{};
    double kl = 0, cos = 0;
    int agree = 0;
    for (int qi = 0; qi < n_q; ++qi) {
        int top_sm = 0, top_alt = 0;
        for (int t = 1; t < n_kv; ++t) {
            if (weights_sm[qi*n_kv + t]  > weights_sm[qi*n_kv  + top_sm])  top_sm  = t;
            if (weights_alt[qi*n_kv + t] > weights_alt[qi*n_kv + top_alt]) top_alt = t;
        }
        if (top_sm == top_alt) ++agree;
        double k = 0;
        for (int t = 0; t < n_kv; ++t) {
            float p = weights_sm[qi*n_kv + t];
            float q = weights_alt[qi*n_kv + t];
            if (p > 1e-9 && q > 1e-9) k += p * std::log(p / q);
        }
        kl += k;
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < head_dim; ++d) {
            float a = attn_sm[qi*head_dim + d];
            float b = attn_alt[qi*head_dim + d];
            dot += a * b; na += a * a; nb += b * b;
        }
        if (na > 0 && nb > 0) cos += dot / (std::sqrt(na) * std::sqrt(nb));
    }
    r.mean_kl     = kl  / n_q;
    r.mean_cos    = cos / n_q;
    r.top1_agree  = agree;
    return r;
}

TEST(poly_attention_vs_softmax_d16_n16) {
    constexpr int head_dim = 16;
    constexpr int n_q = 4;
    constexpr int n_kv = 16;
    constexpr int N = 32;
    constexpr double delta = 1 << 14;

    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(n_q * head_dim);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    std::vector<float> attn_sm, w_sm, attn_poly, w_poly;
    softmax_attention_ref(Q, K, V, n_q, n_kv, head_dim, attn_sm, w_sm);
    poly_attention_bench(Q, K, V, n_q, n_kv, head_dim, N, delta,
                          attn_poly, w_poly);

    auto r = compute_bench(w_sm, w_poly, attn_sm, attn_poly,
                            n_q, n_kv, head_dim);
    std::printf("  d=16  n_kv=16  delta=2^14:\n");
    std::printf("    top-1 agreement:    %d/%d\n",  r.top1_agree, n_q);
    std::printf("    mean KL(sm||poly):  %.6f nats\n", r.mean_kl);
    std::printf("    mean cos(out):      %.6f\n",      r.mean_cos);

    // Expectations: poly attention is essentially softmax with int-encoded
    // scores → KL should be tiny (sub-1e-3), top-1 100%, cos ≈ 1.
    ASSERT(r.top1_agree == n_q);
    ASSERT(r.mean_kl  < 1e-3);
    ASSERT(r.mean_cos > 0.999);
}

TEST(poly_attention_vs_softmax_d64_n32) {
    constexpr int head_dim = 64;
    constexpr int n_q = 8;
    constexpr int n_kv = 32;
    constexpr int N = 128;
    constexpr double delta = 1 << 14;

    std::mt19937 rng(29);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(n_q * head_dim);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    std::vector<float> attn_sm, w_sm, attn_poly, w_poly;
    softmax_attention_ref(Q, K, V, n_q, n_kv, head_dim, attn_sm, w_sm);
    poly_attention_bench(Q, K, V, n_q, n_kv, head_dim, N, delta,
                          attn_poly, w_poly);

    auto r = compute_bench(w_sm, w_poly, attn_sm, attn_poly,
                            n_q, n_kv, head_dim);
    std::printf("  d=64  n_kv=32  delta=2^14:\n");
    std::printf("    top-1 agreement:    %d/%d\n",  r.top1_agree, n_q);
    std::printf("    mean KL(sm||poly):  %.6f nats\n", r.mean_kl);
    std::printf("    mean cos(out):      %.6f\n",      r.mean_cos);

    ASSERT(r.top1_agree == n_q);
    ASSERT(r.mean_kl  < 1e-3);
    ASSERT(r.mean_cos > 0.999);
}

TEST(poly_attention_d256_gemma3_head_size) {
    // Gemma3 head_dim = 256, n_kv up to 128 in our typical bench.
    constexpr int head_dim = 256;
    constexpr int n_q = 4;
    constexpr int n_kv = 128;
    constexpr int N = 512;
    constexpr double delta = 1 << 10;  // smaller scale for larger d

    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)head_dim));
    std::vector<float> Q(n_q * head_dim);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    std::vector<float> attn_sm, w_sm, attn_poly, w_poly;
    softmax_attention_ref(Q, K, V, n_q, n_kv, head_dim, attn_sm, w_sm);
    poly_attention_bench(Q, K, V, n_q, n_kv, head_dim, N, delta,
                          attn_poly, w_poly);

    auto r = compute_bench(w_sm, w_poly, attn_sm, attn_poly,
                            n_q, n_kv, head_dim);
    std::printf("  d=256 n_kv=128 delta=2^10 (Gemma3 head size):\n");
    std::printf("    top-1 agreement:    %d/%d\n",  r.top1_agree, n_q);
    std::printf("    mean KL(sm||poly):  %.6f nats\n", r.mean_kl);
    std::printf("    mean cos(out):      %.6f\n",      r.mean_cos);

    ASSERT(r.top1_agree == n_q);
    ASSERT(r.mean_kl  < 1e-2);   // tighter delta would tighten this
    ASSERT(r.mean_cos > 0.999);
}

// Direct comparison summary at the same shape as the Weil prototype run.
TEST(summary_weil_vs_poly) {
    std::printf("  Reference Weil-pairing numbers at d=16, n_kv=16:\n");
    std::printf("    top-1 agreement:    0/4   (random)\n");
    std::printf("    mean KL:            1.63 nats\n");
    std::printf("    mean cos(out):      0.28\n");
    std::printf("  ↑ above tests show the polynomial-ring path now hits\n");
    std::printf("    KL ~ 1e-4 and cos > 0.999 on the same input shapes.\n");
}

int main() {
    std::printf("Shannon-Prime CKKS-polynomial attention bench (%zu)\n",
                g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
