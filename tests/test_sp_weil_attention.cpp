// Shannon-Prime — Weil-pairing attention prototype (Phase 3 parts 2+3).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Replaces the QK^T/sqrt(d) softmax bridge with an integer Weil pairing
// on E[n] torsion. From Paper A §9.2:
//
//   - Curve  E: y^2 = x^3 + 1 over F_p, p prime with p ≡ 2 (mod 3) so E
//     is supersingular and |E(F_p)| = p + 1. Then E[n] is 2-dimensional
//     for any n | (p+1) coprime to p.
//   - Pick two independent generators P_0, P_1 ∈ E[n].
//   - Map each head vector to a torsion point via a fixed linear hash:
//        h = head_vec → (a, b) ∈ (Z/n)^2 → P = a P_0 + b P_1
//   - Score(q, k) = log_zeta(e_n(P_q, P_k))  where ζ is a primitive
//     n-th root of unity in F_p. The discrete log lifts the multi-
//     plicative pairing into an additive real-valued score.
//
// This replaces softmax(QK^T/sqrt(d)) entirely. The score function is
// O(log n) pure integer arithmetic, no transcendentals.
//
// Goals:
//   1. Demonstrate the pipeline runs end-to-end on real data
//   2. Compare output distributions vs standard softmax attention
//   3. Measure how well the pairing-based scores correlate with
//      dot-product scores on the same Q, K, V
//
// What we will NOT claim: that this is a drop-in replacement for
// softmax that preserves model quality. That's a Phase 3+ research
// question and depends on retraining the network for the new score
// function.

extern "C" {
#include "../lib/shannon-prime/core/sp_ec_weil.h"
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
// Curve setup: E: y^2 = x^3 + 1 over F_23. |E(F_23)| = 24 = 2^3 * 3.
// E[3] is full 2-dim with 9 points. Two independent generators give
// every E[3] element as a * P_0 + b * P_1 for (a, b) ∈ (Z/3)^2.
// =========================================================================

struct WeilParams {
    sp_ec_curve E;
    int64_t     n;        // torsion order (e.g. 3)
    sp_ec_point P_0;      // first generator
    sp_ec_point P_1;      // second generator (independent of P_0)
    int64_t     zeta;     // primitive n-th root of unity in F_p (for log)
    std::vector<int64_t> root_table;  // root_table[i] = zeta^i mod p, size n
};

// Search for a (curve, n, p) with FULL 2-dim E[n] over F_p. Brute-force
// over small primes p (10..200) and curve b ∈ [1, p-1]:
//   - Enumerate E(F_p).
//   - Compute torsion E[n] for n in {2, 3, 5}.
//   - If |E[n]| == n², we have 2-dim torsion.
// Returns the first working combo, or sets E.p = -1 if none found.
static bool find_2dim_torsion(WeilParams& W) {
    static const int64_t test_primes[] = {
        11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
        149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199
    };
    // Prefer LARGER n (richer score range). Iterate n outermost.
    for (int64_t n : { (int64_t)7, (int64_t)5, (int64_t)3, (int64_t)2 }) {
    for (int64_t p : test_primes) {
        if (p % n == 0) continue;
        for (int64_t b = 1; b < p; ++b) {
            sp_ec_curve E = { 0, b, p };
            // Enumerate.
            std::vector<sp_ec_point> all_pts;
            all_pts.push_back(SP_EC_INFINITY);
            for (int64_t x = 0; x < p; ++x) {
                int64_t rhs = (((x * x) % p) * x + b) % p;
                for (int64_t y = 0; y < p; ++y) {
                    if ((y * y) % p == rhs) {
                        sp_ec_point P = { x, y };
                        all_pts.push_back(P);
                    }
                }
            }
            // Find n-torsion.
            {
                std::vector<sp_ec_point> torsion;
                for (auto T : all_pts) {
                    if (sp_ec_is_infinity(sp_ec_mul(&E, n, T))) torsion.push_back(T);
                }
                if ((int64_t)torsion.size() == n * n) {
                    // FULL 2-dim torsion. Pick generators.
                    sp_ec_point P_0 = SP_EC_INFINITY;
                    for (auto T : torsion) {
                        if (!sp_ec_is_infinity(T)) { P_0 = T; break; }
                    }
                    sp_ec_point P_1 = SP_EC_INFINITY;
                    for (auto T : torsion) {
                        if (sp_ec_is_infinity(T)) continue;
                        // T not in <P_0> = {k*P_0 : k in [0, n)}
                        bool in_span = false;
                        sp_ec_point cur = SP_EC_INFINITY;
                        for (int64_t k = 0; k < n; ++k) {
                            if (sp_ec_eq(T, cur)) { in_span = true; break; }
                            cur = sp_ec_add(&E, cur, P_0);
                        }
                        if (!in_span) { P_1 = T; break; }
                    }
                    if (!sp_ec_is_infinity(P_1)) {
                        int64_t zeta = sp_ec_weil_pairing(&E, n, P_0, P_1);
                        if (zeta != 1 && zeta > 0) {
                            W.E = E;
                            W.n = n;
                            W.P_0 = P_0;
                            W.P_1 = P_1;
                            W.zeta = zeta;
                            return true;
                        }
                    }
                }
            }
        }
    }
    }
    W.E.p = -1;
    return false;
}

static WeilParams make_weil_params() {
    WeilParams W;
    if (!find_2dim_torsion(W)) {
        std::fprintf(stderr,
            "  ERROR: no (curve, n, p) with full 2-dim E[n] in search range\n");
        return W;
    }
    std::fprintf(stderr,
        "  Found: E: y^2 = x^3 + %lld over F_%lld, n=%lld\n"
        "         |E[%lld]| = n^2 = %lld   P_0 = (%lld, %lld)  P_1 = (%lld, %lld)\n",
        (long long)W.E.b, (long long)W.E.p, (long long)W.n,
        (long long)W.n, (long long)(W.n * W.n),
        (long long)W.P_0.x, (long long)W.P_0.y,
        (long long)W.P_1.x, (long long)W.P_1.y);
    std::fprintf(stderr, "  zeta = e_n(P_0, P_1) = %lld\n", (long long)W.zeta);

    // Primitive n-th root of unity: zeta = e_n(P_0, P_1).
    W.zeta = sp_ec_weil_pairing(&W.E, W.n, W.P_0, W.P_1);
    std::fprintf(stderr, "  zeta = e_3(P_0, P_1) = %lld\n", (long long)W.zeta);

    // Precompute root table: zeta^0, zeta^1, ..., zeta^(n-1).
    W.root_table.resize(W.n);
    for (int64_t i = 0; i < W.n; ++i) {
        W.root_table[i] = sp_ec_mod_pow(W.zeta, i, W.E.p);
    }
    return W;
}

// Reset hash quantization bin count based on n (so vectors map across
// the full (Z/n)^2).
static int64_t hash_bin_count_for(int64_t n) {
    if (n <= 4) return 8;
    return 4 * n;
}

// Map a head-dim vector to (a, b) ∈ (Z/n)^2 via two random linear hashes.
// The hash projections are fixed across the test.
struct VecHasher {
    std::vector<float> w_a;   // [head_dim] for first projection
    std::vector<float> w_b;   // [head_dim] for second projection
    float bias_a, bias_b;
    int64_t n;

    void init(int head_dim, int64_t n_, uint64_t seed) {
        n = n_;
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)head_dim));
        w_a.resize(head_dim);
        w_b.resize(head_dim);
        for (auto& v : w_a) v = nd(rng);
        for (auto& v : w_b) v = nd(rng);
        bias_a = 0.0f;
        bias_b = 0.0f;
    }

    void hash(const float* vec, int head_dim, int64_t& a_out, int64_t& b_out) const {
        double a = bias_a, b = bias_b;
        for (int i = 0; i < head_dim; ++i) {
            a += (double)vec[i] * (double)w_a[i];
            b += (double)vec[i] * (double)w_b[i];
        }
        // Quantize via a wide bin then mod n. The bin count is set to
        // hash_bin_count_for(n) so that typical unit-norm vectors
        // distribute uniformly across (Z/n)^2.
        const int64_t Q = hash_bin_count_for(n);
        int64_t ai = ((int64_t)std::llrint(a * (double)Q) % n + n) % n;
        int64_t bi = ((int64_t)std::llrint(b * (double)Q) % n + n) % n;
        a_out = ai;
        b_out = bi;
    }
};

// Vector → E[n] torsion point: P_q = a * P_0 + b * P_1.
static sp_ec_point vec_to_torsion(const WeilParams& W, const VecHasher& H,
                                    const float* vec, int head_dim) {
    int64_t a, b;
    H.hash(vec, head_dim, a, b);
    sp_ec_point aP = sp_ec_mul(&W.E, a, W.P_0);
    sp_ec_point bQ = sp_ec_mul(&W.E, b, W.P_1);
    return sp_ec_add(&W.E, aP, bQ);
}

// Pairing → real-valued score via discrete log (returned as int in [0, n)).
// If pairing == 1, log = 0. Otherwise look up in root_table.
static double pairing_score(const WeilParams& W, int64_t pairing_val) {
    if (pairing_val == 1) return 0.0;
    for (int64_t i = 0; i < W.n; ++i) {
        if (W.root_table[i] == pairing_val) return (double)i;
    }
    return 0.0;  // shouldn't happen for valid n-th roots
}

// =========================================================================
// Reference fp32 softmax attention: scores = q·k / sqrt(d), softmax, weight V.
// =========================================================================

static void softmax_attention_ref(const std::vector<float>& Q,  // [n_q, head_dim]
                                    const std::vector<float>& K,  // [n_kv, head_dim]
                                    const std::vector<float>& V,  // [n_kv, head_dim]
                                    int n_q, int n_kv, int head_dim,
                                    std::vector<float>& out_attn,    // [n_q, head_dim]
                                    std::vector<float>& out_weights  // [n_q, n_kv]
                                    ) {
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
// Weil-pairing attention prototype: score = log_ζ(e_n(P_q, P_k)).
// Then softmax over those discrete scores, weight V.
// =========================================================================

static void weil_attention_ref(const WeilParams& W, const VecHasher& H,
                                  const std::vector<float>& Q,
                                  const std::vector<float>& K,
                                  const std::vector<float>& V,
                                  int n_q, int n_kv, int head_dim,
                                  std::vector<float>& out_attn,
                                  std::vector<float>& out_weights) {
    out_attn.assign((size_t)n_q * head_dim, 0.0f);
    out_weights.assign((size_t)n_q * n_kv, 0.0f);

    // Precompute all K → torsion points once.
    std::vector<sp_ec_point> K_pts(n_kv);
    for (int t = 0; t < n_kv; ++t) {
        K_pts[t] = vec_to_torsion(W, H, &K[t * head_dim], head_dim);
    }

    std::vector<double> scores(n_kv);
    for (int qi = 0; qi < n_q; ++qi) {
        sp_ec_point P_q = vec_to_torsion(W, H, &Q[qi * head_dim], head_dim);
        for (int t = 0; t < n_kv; ++t) {
            int64_t pair_val = sp_ec_weil_pairing(&W.E, W.n, P_q, K_pts[t]);
            scores[t] = pairing_score(W, pair_val);
        }
        double mx = scores[0];
        for (int t = 1; t < n_kv; ++t) if (scores[t] > mx) mx = scores[t];
        double sum = 0;
        for (int t = 0; t < n_kv; ++t) {
            double e = std::exp(scores[t] - mx);
            out_weights[qi * n_kv + t] = (float)e;
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

TEST(weil_params_setup) {
    WeilParams W = make_weil_params();
    ASSERT(!sp_ec_is_infinity(W.P_0));
    ASSERT(!sp_ec_is_infinity(W.P_1));
    // e_n(P_0, P_1) must be a primitive n-th root of unity in F_p.
    int64_t zeta = sp_ec_weil_pairing(&W.E, W.n, W.P_0, W.P_1);
    ASSERT(zeta != 1);
    int64_t zeta_n = sp_ec_mod_pow(zeta, W.n, W.E.p);
    ASSERT(zeta_n == 1);
    // root_table size matches the search result.
    ASSERT((int64_t)W.root_table.size() == W.n);
    ASSERT(W.root_table[0] == 1);
    std::printf("  search result: E over F_%lld, n=%lld, %d possible scores\n",
                (long long)W.E.p, (long long)W.n, (int)W.root_table.size());
}

TEST(weil_attention_pipeline_runs) {
    // Single-head, small dims — just verify the pipeline executes
    // and produces a normalized distribution.
    WeilParams W = make_weil_params();
    VecHasher H;
    constexpr int head_dim = 8;
    constexpr int n_q = 1;
    constexpr int n_kv = 6;
    H.init(head_dim, W.n, /*seed*/ 101);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(n_q * head_dim);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    std::vector<float> attn_weil, weights_weil;
    weil_attention_ref(W, H, Q, K, V, n_q, n_kv, head_dim,
                        attn_weil, weights_weil);

    // Weights sum to 1.
    double s = 0;
    for (int t = 0; t < n_kv; ++t) s += weights_weil[t];
    ASSERT(std::abs(s - 1.0) < 1e-5);

    // Output is finite.
    for (auto x : attn_weil) ASSERT(std::isfinite(x));
}

TEST(weil_vs_softmax_bench) {
    // Single-head, larger n_kv — see how the two attention distributions
    // compare on random data. Print rank correlation and KL divergence.
    WeilParams W = make_weil_params();
    VecHasher H;
    constexpr int head_dim = 16;
    constexpr int n_q = 4;
    constexpr int n_kv = 16;
    H.init(head_dim, W.n, /*seed*/ 202);

    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(n_q * head_dim);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    std::vector<float> attn_sm, weights_sm;
    std::vector<float> attn_weil, weights_weil;

    softmax_attention_ref(Q, K, V, n_q, n_kv, head_dim, attn_sm, weights_sm);
    weil_attention_ref(W, H, Q, K, V, n_q, n_kv, head_dim, attn_weil, weights_weil);

    // For each query, compute:
    //   - Max-weight position picked by softmax vs Weil
    //   - KL divergence: D(sm || weil)
    //   - Cosine similarity of the attention outputs
    int agree_top1 = 0;
    double mean_kl = 0;
    double mean_cos = 0;
    for (int qi = 0; qi < n_q; ++qi) {
        // Top-1 agreement.
        int top_sm = 0, top_weil = 0;
        for (int t = 1; t < n_kv; ++t) {
            if (weights_sm[qi*n_kv + t] > weights_sm[qi*n_kv + top_sm]) top_sm = t;
            if (weights_weil[qi*n_kv + t] > weights_weil[qi*n_kv + top_weil]) top_weil = t;
        }
        if (top_sm == top_weil) ++agree_top1;
        // KL(sm || weil).
        double kl = 0;
        for (int t = 0; t < n_kv; ++t) {
            float p = weights_sm[qi*n_kv + t];
            float q = weights_weil[qi*n_kv + t];
            if (p > 1e-9 && q > 1e-9) kl += p * std::log(p / q);
        }
        mean_kl += kl;
        // Cos sim of attn outputs.
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < head_dim; ++d) {
            float a = attn_sm[qi*head_dim + d];
            float b = attn_weil[qi*head_dim + d];
            dot += a * b; na += a * a; nb += b * b;
        }
        if (na > 0 && nb > 0) mean_cos += dot / (std::sqrt(na) * std::sqrt(nb));
    }
    mean_kl /= n_q;
    mean_cos /= n_q;

    std::printf("  top-1 agreement:    %d/%d\n", agree_top1, n_q);
    std::printf("  mean KL(sm||weil):  %.4f nats\n", mean_kl);
    std::printf("  mean cos(out):      %.4f\n", mean_cos);

    // We don't ASSERT specific values — the goal is to learn what the
    // numbers actually are. If the pairing-based scores are completely
    // uncorrelated with the dot product, KL will be huge and cos will
    // be near zero. If they correlate, we'll see structure.

    // Sanity: weights sum to 1, outputs finite.
    for (int qi = 0; qi < n_q; ++qi) {
        double s_sm = 0, s_weil = 0;
        for (int t = 0; t < n_kv; ++t) {
            s_sm   += weights_sm[qi*n_kv + t];
            s_weil += weights_weil[qi*n_kv + t];
        }
        ASSERT(std::abs(s_sm - 1.0) < 1e-4);
        ASSERT(std::abs(s_weil - 1.0) < 1e-4);
    }
}

TEST(weil_attention_changes_with_query) {
    // Verify the pairing attention DOES change when we perturb Q.
    // (Catches the bug where the hasher collapses many vectors to the
    // same torsion point — would give identical weights everywhere.)
    WeilParams W = make_weil_params();
    VecHasher H;
    constexpr int head_dim = 16;
    constexpr int n_kv = 16;
    H.init(head_dim, W.n, /*seed*/ 303);

    std::mt19937 rng(19);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> K(n_kv * head_dim);
    std::vector<float> V(n_kv * head_dim);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    // Two different queries.
    std::vector<float> Q1(head_dim), Q2(head_dim);
    for (auto& x : Q1) x = nd(rng);
    for (auto& x : Q2) x = nd(rng);

    std::vector<float> a1, w1, a2, w2;
    weil_attention_ref(W, H, Q1, K, V, 1, n_kv, head_dim, a1, w1);
    weil_attention_ref(W, H, Q2, K, V, 1, n_kv, head_dim, a2, w2);

    // The two weight distributions must not be identical.
    bool any_diff = false;
    for (int t = 0; t < n_kv; ++t) {
        if (std::abs(w1[t] - w2[t]) > 1e-4) { any_diff = true; break; }
    }
    ASSERT(any_diff);
}

int main() {
    std::printf("Shannon-Prime Weil-pairing attention prototype (%zu)\n",
                g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
