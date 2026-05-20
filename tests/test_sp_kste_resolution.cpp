/* test_sp_kste_resolution.cpp — Phase-4 KSTE semantic resolution probe.
 *
 * Question (the existential gate for T2.3): does a 60-node, 3-label
 * tree retain enough discriminative resolution to distinguish close-
 * but-distinct natural-language meanings?
 *
 * We can't answer "natural-language" in a sandbox without Gemma3-1B,
 * but we CAN answer the structural prerequisite: when fed clustered
 * K-vectors (a synthetic stand-in for "tokens with similar attention
 * patterns"), does the sieve evict within-cluster duplicates while
 * admitting cross-cluster novelty?  And how cleanly does the filter's
 * dominance signal correlate with cosine similarity of the underlying
 * vectors?
 *
 * If intra-cluster subsumption is high and inter-cluster subsumption
 * stays low, the encoder retains semantic resolution and T2.3 is
 * within reach on the real model.  If the inter rate matches the
 * intra rate, the encoder over-compresses and T2.3 will tank.
 *
 * Output: a single JSON report at tests/results/T4_RES_PROBE.json
 * with per-cluster subsumption rates, an overall ROC AUC of cosine
 * sim vs filter survival, and a verdict line.
 *
 * No model weights, no GGUF, no inference engine — runs anywhere
 * a C++17 compiler builds.
 */

extern "C" {
#include "../lib/shannon-prime/core/shannon_prime.h"
#include "../lib/shannon-prime/core/sp_kste.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(_WIN32)
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/types.h>
#  define MKDIR(p) mkdir(p, 0755)
#endif

/* ---------- IO helpers (same as other tests) --------------------------- */

static void write_json_file(const char *id, const std::string &json)
{
    MKDIR("tests"); MKDIR("tests/results");
    MKDIR("../tests"); MKDIR("../tests/results");
    MKDIR("../../tests"); MKDIR("../../tests/results");
    const char *candidates[] = {
        "../../tests/results/", "../tests/results/", "tests/results/", "./",
    };
    for (const char *dir : candidates) {
        char path[256];
        std::snprintf(path, sizeof(path), "%s%s.json", dir, id);
        FILE *fp = std::fopen(path, "w");
        if (fp) {
            std::fwrite(json.data(), 1, json.size(), fp);
            std::fclose(fp);
            std::fprintf(stderr, "    -> %s\n", path);
            return;
        }
    }
}

/* ---------- Synthetic clustered K-vector corpus ------------------------ */

/* Per-cluster: a unit-norm direction in R^128 + Gaussian noise around it
 * with variance sigma.  This is the simplest non-trivial structure that
 * mimics "semantically similar token activations": each cluster is a
 * concept, samples within it are token variants of that concept. */

static void make_cluster_vector(std::mt19937 &rng,
                                const std::vector<float> &center,
                                float sigma,
                                std::vector<float> &out)
{
    std::normal_distribution<float> N(0.0f, sigma);
    for (size_t i = 0; i < center.size(); ++i) {
        out[i] = center[i] + N(rng);
    }
}

static void make_random_unit(std::mt19937 &rng, std::vector<float> &out)
{
    std::normal_distribution<float> N(0.0f, 1.0f);
    double s = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = N(rng);
        s += (double)out[i] * (double)out[i];
    }
    s = std::sqrt(s);
    if (s == 0.0) s = 1.0;
    for (size_t i = 0; i < out.size(); ++i) out[i] /= (float)s;
}

static float cos_sim(const std::vector<float> &a, const std::vector<float> &b)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / std::sqrt(na * nb));
}

/* ---------- Sieve-style subsumption decision -------------------------- */

/* For two encoded trees, "subsumed" means: T_Q ⪯ T_K under the full
 * Kruskal-Friedman embedding test.  We also report the Tier-0+Tier-1
 * filter decision for the same pair, to compute filter precision vs
 * a known-correct embed reference. */

struct PairProbe {
    int    cluster_a;
    int    cluster_b;
    float  cos;
    int    filter_ok;     /* 1 iff Tier-0 + Tier-1 dominance both pass */
    int    embed_ok;      /* 1 iff full embed returns 1                */
};

/* ---------- Stats helpers --------------------------------------------- */

static double mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0; for (double x : v) s += x; return s / (double)v.size();
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::floor(q * (v.size() - 1));
    return v[i];
}

/* Tier-0-signature L1 distance.  Pure structural fingerprint distance
 * on (A, B, C, depth, node_count) — semantic-similarity probe under
 * Paper III §11.4 fuzzy-radius decisions. */
static int tier0_l1_distance(sp_kste_signature_t a, sp_kste_signature_t b)
{
    int total = 0;
    for (int i = 0; i < 5; ++i) {
        int xa = (int)((a >> (8 * i)) & 0xFF);
        int xb = (int)((b >> (8 * i)) & 0xFF);
        total += (xa > xb) ? (xa - xb) : (xb - xa);
    }
    return total;
}

/* Simple ROC AUC: probabilities are cos_sim, positives are pairs where
 * the full embed returns 1.  AUC via Mann–Whitney U statistic. */
static double roc_auc(const std::vector<PairProbe> &pairs) {
    /* Separate positives (embed_ok=1) and negatives. */
    std::vector<float> pos, neg;
    for (const auto &p : pairs) {
        if (p.embed_ok) pos.push_back(p.cos);
        else            neg.push_back(p.cos);
    }
    if (pos.empty() || neg.empty()) return 0.5;
    /* Mann-Whitney U: count (cos_pos > cos_neg) pairs. */
    double wins = 0.0, ties = 0.0;
    for (float p : pos) for (float n : neg) {
        if (p > n) wins += 1.0;
        else if (p == n) ties += 1.0;
    }
    return (wins + 0.5 * ties) / ((double)pos.size() * (double)neg.size());
}

/* ---------- Main probe routine ---------------------------------------- */

int main()
{
    const int HD          = 128;
    const int N_CLUSTERS  = 50;
    const int PER_CLUSTER = 20;
    const float SIGMA_SAMPLES[] = { 0.005f, 0.01f, 0.02f, 0.05f, 0.10f };
    const int N_SIGMAS = (int)(sizeof(SIGMA_SAMPLES) / sizeof(SIGMA_SAMPLES[0]));

    std::fprintf(stderr, "Phase 4 — KSTE semantic resolution probe\n");

    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::fprintf(stderr, "  FATAL: sp_kste_ctx_init failed\n");
        return 1;
    }

    /* Build cluster centers (unit-norm, well-separated). */
    std::mt19937 rng_centers(0xC1C1);
    std::vector<std::vector<float>> centers(N_CLUSTERS, std::vector<float>(HD));
    for (int c = 0; c < N_CLUSTERS; ++c) {
        make_random_unit(rng_centers, centers[c]);
    }

    /* Per-sigma sweep: how cleanly does the encoder discriminate? */
    std::string per_sigma_json;
    bool overall_pass = true;

    for (int si = 0; si < N_SIGMAS; ++si) {
        float sigma = SIGMA_SAMPLES[si];
        std::fprintf(stderr, "  sigma = %.3f ...\n", sigma);

        /* Encode every sample. */
        std::mt19937 rng((uint32_t)(0x5A11 + si * 7));
        std::vector<std::vector<float>> samples(N_CLUSTERS * PER_CLUSTER,
                                                std::vector<float>(HD));
        std::vector<sp_kste_tree>       trees (N_CLUSTERS * PER_CLUSTER);
        std::vector<float>              scratch(3 * HD);
        for (int c = 0; c < N_CLUSTERS; ++c) {
            for (int j = 0; j < PER_CLUSTER; ++j) {
                int idx = c * PER_CLUSTER + j;
                make_cluster_vector(rng, centers[c], sigma, samples[idx]);
                sp_kste_encode(&trees[idx], samples[idx].data(),
                               &ctx, scratch.data());
            }
        }

        /* Sample a few thousand pairs.  Build the four populations
         * by quadrant: (intra-cluster random pairs) and (inter-cluster
         * random pairs) crossed with (filter_ok, embed_ok). */
        std::mt19937 sel((uint32_t)(0xD0DE + si));
        std::uniform_int_distribution<int> pick_cluster(0, N_CLUSTERS - 1);
        std::uniform_int_distribution<int> pick_within(0, PER_CLUSTER - 1);

        const int N_INTRA = 4000;
        const int N_INTER = 4000;

        long long intra_filter_ok = 0, intra_embed_ok = 0;
        long long inter_filter_ok = 0, inter_embed_ok = 0;
        std::vector<double> intra_cos, inter_cos;
        std::vector<PairProbe> pairs;
        pairs.reserve(N_INTRA + N_INTER);

        for (int p = 0; p < N_INTRA; ++p) {
            int c = pick_cluster(sel);
            int a = pick_within(sel), b = pick_within(sel);
            if (a == b) continue;
            int ia = c * PER_CLUSTER + a, ib = c * PER_CLUSTER + b;
            float cs = cos_sim(samples[ia], samples[ib]);
            sp_kste_signature_t sa = sp_kste_compute_signature(&trees[ia]);
            sp_kste_signature_t sb = sp_kste_compute_signature(&trees[ib]);
            sp_kste_anc_sig_t   aa, ab;
            sp_kste_compute_anc_sig(&trees[ia], &aa);
            sp_kste_compute_anc_sig(&trees[ib], &ab);
            int f_ok = sp_kste_sig_dominates(sb, sa) &&
                       sp_kste_anc_sig_dominates(&ab, &aa);
            int e_ok = sp_kste_embed_unordered(&trees[ia], &trees[ib]);
            intra_filter_ok += f_ok;
            intra_embed_ok  += e_ok;
            intra_cos.push_back(cs);
            pairs.push_back({c, c, cs, f_ok, e_ok});
        }
        for (int p = 0; p < N_INTER; ++p) {
            int ca = pick_cluster(sel), cb = pick_cluster(sel);
            if (ca == cb) continue;
            int a = pick_within(sel), b = pick_within(sel);
            int ia = ca * PER_CLUSTER + a, ib = cb * PER_CLUSTER + b;
            float cs = cos_sim(samples[ia], samples[ib]);
            sp_kste_signature_t sa = sp_kste_compute_signature(&trees[ia]);
            sp_kste_signature_t sb = sp_kste_compute_signature(&trees[ib]);
            sp_kste_anc_sig_t   aa, ab;
            sp_kste_compute_anc_sig(&trees[ia], &aa);
            sp_kste_compute_anc_sig(&trees[ib], &ab);
            int f_ok = sp_kste_sig_dominates(sb, sa) &&
                       sp_kste_anc_sig_dominates(&ab, &aa);
            int e_ok = sp_kste_embed_unordered(&trees[ia], &trees[ib]);
            inter_filter_ok += f_ok;
            inter_embed_ok  += e_ok;
            inter_cos.push_back(cs);
            pairs.push_back({ca, cb, cs, f_ok, e_ok});
        }

        /* --- Path D experiment: fuzzy-radius decisions on Tier-0 signatures. */
        int fuzzy_radii[] = { 1, 2, 4, 8 };
        const int N_R = (int)(sizeof(fuzzy_radii) / sizeof(int));
        long long fuzzy_intra[4] = {0};
        long long fuzzy_inter[4] = {0};
        for (int ip = 0; ip < (int)pairs.size(); ++ip) {
            int ia = ip < N_INTRA ? (pairs[ip].cluster_a * PER_CLUSTER) : 0;
            (void)ia;
        }
        /* Direct recompute on per-pair sigs.  Faster: recompute per pair. */
        std::vector<sp_kste_signature_t> sig_cache(N_CLUSTERS * PER_CLUSTER);
        for (int j = 0; j < (int)trees.size(); ++j) {
            sig_cache[j] = sp_kste_compute_signature(&trees[j]);
        }
        std::mt19937 sel2((uint32_t)(0xD0DE + si));
        std::uniform_int_distribution<int> pick_c(0, N_CLUSTERS - 1);
        std::uniform_int_distribution<int> pick_w(0, PER_CLUSTER - 1);
        for (int kp = 0; kp < 4000; ++kp) {
            int c = pick_c(sel2), a = pick_w(sel2), b = pick_w(sel2);
            if (a == b) continue;
            int ia = c * PER_CLUSTER + a, ib = c * PER_CLUSTER + b;
            int dist = tier0_l1_distance(sig_cache[ia], sig_cache[ib]);
            for (int r = 0; r < N_R; ++r) if (dist <= fuzzy_radii[r]) fuzzy_intra[r]++;
        }
        for (int kp = 0; kp < 4000; ++kp) {
            int ca = pick_c(sel2), cb = pick_c(sel2);
            if (ca == cb) continue;
            int a = pick_w(sel2), b = pick_w(sel2);
            int ia = ca * PER_CLUSTER + a, ib = cb * PER_CLUSTER + b;
            int dist = tier0_l1_distance(sig_cache[ia], sig_cache[ib]);
            for (int r = 0; r < N_R; ++r) if (dist <= fuzzy_radii[r]) fuzzy_inter[r]++;
        }
        std::fprintf(stderr, "    fuzzy r=1: intra=%.3f inter=%.3f  r=2: intra=%.3f inter=%.3f  "
                             "r=4: intra=%.3f inter=%.3f  r=8: intra=%.3f inter=%.3f\n",
            fuzzy_intra[0]/4000.0, fuzzy_inter[0]/4000.0,
            fuzzy_intra[1]/4000.0, fuzzy_inter[1]/4000.0,
            fuzzy_intra[2]/4000.0, fuzzy_inter[2]/4000.0,
            fuzzy_intra[3]/4000.0, fuzzy_inter[3]/4000.0);

        double intra_filter_rate = (double)intra_filter_ok / (double)intra_cos.size();
        double intra_embed_rate  = (double)intra_embed_ok  / (double)intra_cos.size();
        double inter_filter_rate = (double)inter_filter_ok / (double)inter_cos.size();
        double inter_embed_rate  = (double)inter_embed_ok  / (double)inter_cos.size();
        double auc = roc_auc(pairs);

        /* Pass per sigma: intra subsumption (embed) at least 2x inter. */
        double intra_inter_ratio = (inter_embed_rate > 0.0)
            ? intra_embed_rate / inter_embed_rate
            : 1e9;
        bool sigma_pass = (intra_embed_rate > inter_embed_rate) &&
                          (intra_inter_ratio >= 2.0 || inter_embed_rate < 0.01);
        if (!sigma_pass) overall_pass = false;

        std::fprintf(stderr, "    intra: filter=%.3f embed=%.3f cos=[%.3f .. %.3f]\n",
            intra_filter_rate, intra_embed_rate,
            *std::min_element(intra_cos.begin(), intra_cos.end()),
            *std::max_element(intra_cos.begin(), intra_cos.end()));
        std::fprintf(stderr, "    inter: filter=%.3f embed=%.3f cos=[%.3f .. %.3f]  AUC=%.3f\n",
            inter_filter_rate, inter_embed_rate,
            *std::min_element(inter_cos.begin(), inter_cos.end()),
            *std::max_element(inter_cos.begin(), inter_cos.end()),
            auc);

        char block[1024];
        std::snprintf(block, sizeof(block),
            "    {\n"
            "      \"sigma\": %.3f,\n"
            "      \"clusters\": %d, \"per_cluster\": %d,\n"
            "      \"intra_filter_rate\": %.4f, \"intra_embed_rate\": %.4f,\n"
            "      \"inter_filter_rate\": %.4f, \"inter_embed_rate\": %.4f,\n"
            "      \"intra_inter_embed_ratio\": %.4f,\n"
            "      \"roc_auc_cos_vs_embed\": %.4f,\n"
            "      \"sigma_pass\": %s\n"
            "    }%s\n",
            sigma, N_CLUSTERS, PER_CLUSTER,
            intra_filter_rate, intra_embed_rate,
            inter_filter_rate, inter_embed_rate,
            intra_inter_ratio, auc,
            sigma_pass ? "true" : "false",
            (si + 1 == N_SIGMAS) ? "" : ",");
        per_sigma_json += block;
    }

    sp_kste_ctx_destroy(&ctx);

    char header[512];
    std::snprintf(header, sizeof(header),
        "{\n"
        "  \"test_id\": \"T4_RES_PROBE\", \"phase\": 4,\n"
        "  \"name\": \"KSTE semantic resolution probe\",\n"
        "  \"config\": { \"head_dim\": %d, \"n_clusters\": %d, \"per_cluster\": %d },\n"
        "  \"per_sigma\": [\n",
        HD, N_CLUSTERS, PER_CLUSTER);

    std::string json = header + per_sigma_json
        + "  ],\n"
        + "  \"verdict\": \"" + (overall_pass ? "PASS" : "FAIL") + "\"\n"
        + "}\n";

    write_json_file("T4_RES_PROBE", json);

    std::fprintf(stderr, "\nResolution probe verdict: %s\n",
                 overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
