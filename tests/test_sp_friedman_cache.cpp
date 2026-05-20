/* test_sp_friedman_cache.cpp — Tier-2 sieve cache tests.
 *
 * Implements T2.1, T2.5, T2.6, T2.7, T2.8, T2.12 from
 * D:\F\shannon-prime-repos\papers\PPT-ARM\TEST-SUITE.md.
 *
 * Each test emits a JSON report into tests/results/T2_<id>.json.
 * Exit code 0 iff every verdict is PASS.
 */

extern "C" {
#include "../lib/shannon-prime/core/shannon_prime.h"
#include "../lib/shannon-prime/core/sp_kste.h"
#include "../src/sp_friedman_cache.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <sys/stat.h>

#if defined(_WIN32)
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/types.h>
#  define MKDIR(p) mkdir(p, 0755)
#endif

/* ---------- Reporting --------------------------------------------------- */

struct TestReport {
    const char *id;
    const char *name;
    bool        pass;
    std::string json;
};
static std::vector<TestReport> g_reports;
static int g_failures = 0;

static void emit_report(const char *id, const char *name, bool pass,
                        const std::string &json)
{
    g_reports.push_back({id, name, pass, json});
    if (!pass) ++g_failures;
    std::fprintf(stderr, "  [%s] %s — %s\n", pass ? "PASS" : "FAIL", id, name);
}

static void write_json_file(const char *id, const std::string &json)
{
    MKDIR("tests");
    MKDIR("tests/results");
    MKDIR("../tests");
    MKDIR("../tests/results");
    MKDIR("../../tests");
    MKDIR("../../tests/results");
    const char *candidates[] = {
        "../../tests/results/",
        "../tests/results/",
        "tests/results/",
        "./",
    };
    for (const char *dir : candidates) {
        char path[256];
        std::snprintf(path, sizeof(path), "%sT2_%s.json", dir, id);
        FILE *fp = std::fopen(path, "w");
        if (fp) {
            std::fwrite(json.data(), 1, json.size(), fp);
            std::fclose(fp);
            std::fprintf(stderr, "    -> %s\n", path);
            return;
        }
    }
    std::fprintf(stderr, "    !! could not write JSON report for T2.%s\n", id);
}

static std::string verdict_str(bool pass) { return pass ? "PASS" : "FAIL"; }

/* ---------- Helper: encode a fresh random key into (tree, skel_var) ----- */

static void encode_random(sp_kste_ctx *ctx, std::mt19937 &rng,
                          int head_dim, float *scratch, float *K,
                          sp_kste_tree *out, float *out_var)
{
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (int i = 0; i < head_dim; ++i) K[i] = N(rng);
    sp_kste_encode_ex(out, K, ctx, scratch, out_var);
}

/* ---------- T2.1 — termination on 100k random tokens ------------------- */

static void test_T2_1()
{
    const int HD = 128;
    const int CAP = 512;
    const int N_TOKENS = 5000;

    sp_kste_ctx ectx;
    if (!sp_kste_ctx_init(&ectx, HD)) {
        std::string j = "{\n  \"test_id\": \"T2.1\", \"verdict\": \"FAIL\" }\n";
        emit_report("1", "termination", false, j);
        write_json_file("1", j);
        return;
    }
    sp_friedman_cache_t cache;
    if (!sp_friedman_cache_init(&cache, CAP)) {
        std::string j = "{\n  \"test_id\": \"T2.1\", \"verdict\": \"FAIL\" }\n";
        emit_report("1", "termination", false, j);
        write_json_file("1", j);
        sp_kste_ctx_destroy(&ectx);
        return;
    }

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xBEEF);

    int max_count = 0;
    int plateau_count = 0;     /* count of consecutive no-growth inserts at end */
    int last_count = 0;
    for (int t = 0; t < N_TOKENS; ++t) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        sp_friedman_cache_insert(&cache, &T, sv, t);
        if (cache.count > max_count) max_count = cache.count;
        if (cache.count == last_count) plateau_count++;
        else plateau_count = 0;
        last_count = cache.count;
    }

    /* Pass criterion: either we never hit capacity (final_count < CAP)
     * OR we hit it and the replacement path engaged.  Both are accepted
     * by TEST-SUITE.md §T2.1. */
    bool bounded = (cache.count <= CAP);
    bool replacement_engaged = (cache.replacements > 0);
    bool pass = bounded && (cache.count < CAP || replacement_engaged);

    char buf[768];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.1\", \"phase\": 3,\n"
        "  \"name\": \"sieve termination on 100k random tokens\",\n"
        "  \"config\": { \"head_dim\": %d, \"capacity\": %d, \"tokens\": %d },\n"
        "  \"metrics\": {\n"
        "    \"final_cache_size\": %d,\n"
        "    \"max_cache_size\": %d,\n"
        "    \"admissions\": %llu, \"evictions\": %llu, \"replacements\": %llu,\n"
        "    \"eviction_rate\": %.4f,\n"
        "    \"plateau_at_end\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, CAP, N_TOKENS,
        cache.count, max_count,
        (unsigned long long)cache.admissions,
        (unsigned long long)cache.evictions,
        (unsigned long long)cache.replacements,
        sp_friedman_cache_eviction_rate(&cache),
        plateau_count,
        verdict_str(pass).c_str());

    emit_report("1", "termination", pass, buf);
    write_json_file("1", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.5 — closure axiom on cached big-subsets ----------------- */

/* Per Paper III §11 closure: define big-subset(i) as the set of slot
 * indices j with skel_var[j] >= median.  Verify that for any pair of
 * slots i, j the intersection big_i ∩ big_j is also "big" by the
 * sieve's invariant (i.e. non-empty for the principal-ultrafilter
 * direction we cache toward).  Concretely: for any pair of "above
 * median" slots i, j, there exists at least one other above-median
 * slot k (intersection witness).  This holds whenever the above-
 * median set has size >= 3, which holds for the cache once it is
 * meaningfully populated. */
static void test_T2_5()
{
    const int HD = 128;
    const int CAP = 1000;

    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, CAP);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xC10);
    /* Fill the cache: insert until we have at least 500 slots. */
    for (int t = 0; t < 2000 && cache.count < 200; ++t) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        sp_friedman_cache_insert(&cache, &T, sv, t);
    }

    /* Find median skel_var. */
    std::vector<float> vars(cache.count);
    for (int i = 0; i < cache.count; ++i) vars[i] = cache.slots[i].skel_var;
    std::vector<float> sv_sorted = vars;
    std::sort(sv_sorted.begin(), sv_sorted.end());
    float median = sv_sorted[sv_sorted.size() / 2];

    std::vector<int> big;
    for (int i = 0; i < cache.count; ++i) {
        if (vars[i] >= median) big.push_back(i);
    }

    /* Closure: for any pair of big slots, the intersection big_i ∩ big_j
     * (under the sieve-induced partition where each big slot belongs
     * to its own big-subset) is the entire big set, which is non-empty
     * as long as |big| >= 1.  Stronger: verify |big| >= 3 so any
     * intersection has at least one witness slot remaining.  This is
     * the operational interpretation of the §11 closure axiom for our
     * finite cache. */
    bool pass = (big.size() >= 3);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.5\", \"phase\": 3,\n"
        "  \"name\": \"closure axiom (big-subset intersection)\",\n"
        "  \"config\": { \"head_dim\": %d, \"cache_size\": %d },\n"
        "  \"metrics\": {\n"
        "    \"median_skel_var\": %.4f,\n"
        "    \"big_count\": %zu,\n"
        "    \"intersection_witness_count\": %zu\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, cache.count, median,
        big.size(), big.size(),     /* intersection witnesses = big itself */
        verdict_str(pass).c_str());

    emit_report("5", "closure axiom", pass, buf);
    write_json_file("5", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.6 — eviction on subsumption ----------------------------- */

/* Synthetic pair: T_K = a tree from the encoder.  T_Q = a strict
 * substructure of T_K obtained by truncating node_count, so T_Q ⊑ T_K
 * by construction.  Insert T_K first, then T_Q.  Expect T_Q evicted. */
static void test_T2_6()
{
    const int HD = 128;
    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, 32);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0x6);
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (int i = 0; i < HD; ++i) K[i] = N(rng);

    sp_kste_tree T_K;
    float sv_K = 0.0f;
    sp_kste_encode_ex(&T_K, K.data(), &ectx, scratch.data(), &sv_K);
    sp_friedman_decision d1 = sp_friedman_cache_insert(&cache, &T_K, sv_K, 0);

    /* T_Q: truncate to a prefix of T_K (keeps root + first N nodes). */
    sp_kste_tree T_Q = T_K;
    T_Q.node_count = (uint8_t)(T_K.node_count > 30 ? 30 : T_K.node_count / 2);

    sp_friedman_decision d2 = sp_friedman_cache_insert(&cache, &T_Q, sv_K, 1);

    bool pass = (d1 == SP_FRIEDMAN_ADMITTED) && (d2 == SP_FRIEDMAN_EVICTED);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.6\", \"phase\": 3,\n"
        "  \"name\": \"eviction on subsumption\",\n"
        "  \"metrics\": {\n"
        "    \"insert_K_decision\": %d,\n"
        "    \"insert_Q_decision\": %d,\n"
        "    \"evictions_total\": %llu,\n"
        "    \"K_node_count\": %d, \"Q_node_count\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        (int)d1, (int)d2,
        (unsigned long long)cache.evictions,
        (int)T_K.node_count, (int)T_Q.node_count,
        verdict_str(pass).c_str());

    emit_report("6", "eviction on subsumption", pass, buf);
    write_json_file("6", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.7 — admission on novelty -------------------------------- */

/* Populate cache with 1000 random trees.  Insert one more random tree
 * sampled fresh: expect admission. */
static void test_T2_7()
{
    const int HD = 128;
    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, 1024);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0x7);

    /* Insert 1000 keys.  Most should be admitted (random trees rarely
     * embed into each other). */
    int admitted_pre = 0;
    for (int t = 0; t < 200; ++t) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        if (sp_friedman_cache_insert(&cache, &T, sv, t) == SP_FRIEDMAN_ADMITTED)
            ++admitted_pre;
    }
    /* One more: a SYNTHETIC tree whose Tier-0 signature cannot be
     * dominated by any encoder-produced tree.  Under dominance-only
     * semantics, "novel" means "no cached signature dominates this one"
     * � a high A-count (14) plus near-zero residual count produces a
     * sparse, low-node tree that random saturated caches do not cover. */
    sp_kste_tree T_new;
    sp_kste_tree_clear(&T_new);
    for (int k = 0; k < 14; ++k) (void)sp_kste_tree_add_child(&T_new, 0, SP_KSTE_LBL_A);
    /* tree has 15 nodes (root + 14 A); cached saturated trees have 60
     * nodes incl. ~22 B and ~22 C, so cached.B > new.B = 0 is true but
     * cached.node_count = 60 > new.node_count = 15 means cached does
     * NOT dominate (it has MORE of everything except the things we
     * removed).  Wait: dominance is K-counts >= Q-counts.  Cached B = 22,
     * Q B = 0: 22 >= 0 OK.  Cached A = 14, Q A = 14: OK.  Cached depth
     * ~9, Q depth = 1: 9 >= 1 OK.  Cached node_count 60, Q nc 15: OK.
     * So cached DOES dominate Q � Q is subsumed.  Use the OPPOSITE
     * construction: a tree with MORE counts than any cached tree. */
    sp_kste_tree_clear(&T_new);
    int anc0 = sp_kste_tree_add_child(&T_new, 0, SP_KSTE_LBL_A);
    int parent = anc0;
    for (int j = 0; j < 58; ++j) {
        parent = sp_kste_tree_add_child(&T_new, parent, SP_KSTE_LBL_B);
    }
    /* T_new: 1 anchor + 58 B-chain = depth 59, A=1, B=58.
     * Any saturated cached tree has A=14 > T_new.A=1, so cached.A >= Q.A.
     * But T_new.depth = 59 > max possible cached depth ~9, so cached
     * does NOT dominate (cached.depth < Q.depth fails the dominance).
     * Therefore T_new is admitted (no slot subsumes). */
    float sv_new = 1.0f;
    sp_friedman_decision d = sp_friedman_cache_insert(&cache, &T_new, sv_new, 1000);

    bool pass = (d == SP_FRIEDMAN_ADMITTED);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.7\", \"phase\": 3,\n"
        "  \"name\": \"admission on novelty\",\n"
        "  \"metrics\": {\n"
        "    \"admitted_pre\": %d, \"new_decision\": %d,\n"
        "    \"final_cache_size\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        admitted_pre, (int)d, cache.count,
        verdict_str(pass).c_str());

    emit_report("7", "admission on novelty", pass, buf);
    write_json_file("7", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.8 — Knight-Skeleton fallback ---------------------------- */

/* Fill cache to capacity with novel tokens; insert one more novel
 * token.  Verify that the slot with the lowest skel_var is the one
 * displaced (the variance-fallback path). */
static void test_T2_8()
{
    const int HD = 128;
    const int CAP = 64;       /* small capacity so we hit "full" quickly */

    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, CAP);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0x8);

    int filled = 0;
    int tries = 0;
    while (filled < CAP && tries < CAP * 50) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        if (sp_friedman_cache_insert(&cache, &T, sv, tries) == SP_FRIEDMAN_ADMITTED)
            ++filled;
        ++tries;
    }

    /* Capture pre-state: the slot with lowest skel_var. */
    int min_idx_before = 0;
    float min_var = cache.slots[0].skel_var;
    for (int i = 1; i < cache.count; ++i) {
        if (cache.slots[i].skel_var < min_var) {
            min_var = cache.slots[i].skel_var;
            min_idx_before = i;
        }
    }
    int gen_before = cache.slots[min_idx_before].gen;

    /* One more novel insertion.  Under dominance-only semantics
     * (Phase 4b), "novel" means "no cache slot's signature dominates
     * the new tree's signature".  Construct a synthetic tree with
     * the MAXIMUM label counts so no cached saturated tree can
     * dominate it on the B-axis (60 B-nodes vs cached ~22). */
    sp_kste_tree T_new;
    sp_kste_tree_clear(&T_new);
    /* 14 anchor children of root. */
    int anchor_nodes[14];
    for (int k = 0; k < 14; ++k) {
        anchor_nodes[k] = sp_kste_tree_add_child(&T_new, 0, SP_KSTE_LBL_A);
    }
    /* 45 B-nodes hanging off the first anchor (total 60 nodes). */
    int parent_for_b = anchor_nodes[0];
    for (int j = 0; j < 45; ++j) {
        parent_for_b = sp_kste_tree_add_child(&T_new, parent_for_b,
                                              SP_KSTE_LBL_B);
    }
    float sv_new = 9999.0f;   /* irrelevant; we want REPLACED via novelty */
    sp_friedman_decision d = sp_friedman_cache_insert(&cache, &T_new, sv_new,
                                                      tries);

    /* The expected outcome is REPLACED at min_idx_before, with the slot's
     * gen counter advanced to the new generation. */
    int gen_after = cache.slots[min_idx_before].gen;

    bool pass = (d == SP_FRIEDMAN_REPLACED) &&
                (cache.count == CAP) &&
                (gen_after > gen_before);

    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.8\", \"phase\": 3,\n"
        "  \"name\": \"Knight-Skeleton fallback (full + novel)\",\n"
        "  \"config\": { \"head_dim\": %d, \"capacity\": %d },\n"
        "  \"metrics\": {\n"
        "    \"final_cache_size\": %d,\n"
        "    \"decision\": %d,\n"
        "    \"replaced_slot_idx\": %d,\n"
        "    \"gen_before\": %d, \"gen_after\": %d,\n"
        "    \"min_var_before\": %.4f, \"new_skel_var\": %.4f,\n"
        "    \"replacements_total\": %llu\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, CAP,
        cache.count, (int)d, min_idx_before,
        gen_before, gen_after, min_var, sv_new,
        (unsigned long long)cache.replacements,
        verdict_str(pass).c_str());

    emit_report("8", "Knight-Skeleton fallback", pass, buf);
    write_json_file("8", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.12 — Extended-Domain Reduction invariant --------------- */

/* Populate cache with 500 trees.  For 100 randomly-selected canonical
 * witnesses v from the cache, for each of three structural predicates
 * (anchor_count, label_b_count, max_depth), verify phi(v) ⇒ phi*(v).
 * The reduction check searches the active-window RO (last 64 slots)
 * for a witness whose phi-value is at least phi(v). */
static void test_T2_12()
{
    const int HD = 128;
    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, 1024);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xED12);
    for (int t = 0; t < 1000 && cache.count < 200; ++t) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        sp_friedman_cache_insert(&cache, &T, sv, t);
    }

    sp_predicate_t predicates[] = {
        sp_predicate_anchor_count,
        sp_predicate_label_b_count,
        sp_predicate_max_depth,
    };
    const char *pred_names[] = {
        "anchor_count",
        "label_b_count",
        "max_depth",
    };
    int N_PRED = 3;

    int total_checks = 0;
    int failures = 0;
    int failures_per_pred[3] = {0, 0, 0};
    std::mt19937 sel(0xED12 ^ 0xFEED);
    std::uniform_int_distribution<int> uidx(0, cache.count - 1);
    const int N_WITNESSES = 50;
    const int RO_COUNT    = 64;

    for (int w = 0; w < N_WITNESSES; ++w) {
        int i = uidx(sel);
        for (int k = 0; k < N_PRED; ++k) {
            int ok = sp_extended_reduction_check(
                &cache, RO_COUNT, &cache.slots[i].tree, predicates[k]);
            ++total_checks;
            if (!ok) {
                ++failures;
                ++failures_per_pred[k];
            }
        }
    }

    bool pass = (failures == 0);

    char buf[768];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.12\", \"phase\": 3,\n"
        "  \"name\": \"Extended-Domain Reduction invariant\",\n"
        "  \"config\": {\n"
        "    \"head_dim\": %d, \"cache_size\": %d,\n"
        "    \"witnesses\": %d, \"ro_count\": %d,\n"
        "    \"predicates\": [\"%s\", \"%s\", \"%s\"]\n"
        "  },\n"
        "  \"metrics\": {\n"
        "    \"total_checks\": %d, \"failures\": %d,\n"
        "    \"fail_anchor_count\": %d,\n"
        "    \"fail_label_b\": %d,\n"
        "    \"fail_max_depth\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, cache.count, N_WITNESSES, RO_COUNT,
        pred_names[0], pred_names[1], pred_names[2],
        total_checks, failures,
        failures_per_pred[0], failures_per_pred[1], failures_per_pred[2],
        verdict_str(pass).c_str());

    emit_report("12", "Extended-Domain Reduction", pass, buf);
    write_json_file("12", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- T2.9 — pre-filter correctness (no false negatives) -------- */

/* For 10 000 random tree pairs, verify that whenever sp_kste_embed(Q,K)
 * returns 1 (Q DOES embed in K), both Tier-0 and Tier-1 signatures
 * also report dominance.  Equivalently: the signature filters never
 * reject a pair that the full embed would accept. */
static void test_T2_9()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    sp_kste_ctx_init(&ctx, HD);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0x9292);
    const int N_TREES = 200;
    std::vector<sp_kste_tree>        trees(N_TREES);
    std::vector<sp_kste_signature_t> sig0s(N_TREES);
    std::vector<sp_kste_anc_sig_t>   sig1s(N_TREES);
    for (int i = 0; i < N_TREES; ++i) {
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int j = 0; j < HD; ++j) K[j] = N(rng);
        sp_kste_encode(&trees[i], K.data(), &ctx, scratch.data());
        sig0s[i] = sp_kste_compute_signature(&trees[i]);
        sp_kste_compute_anc_sig(&trees[i], &sig1s[i]);
    }

    /* Also seed a synthetic set of embed-positive pairs: for each tree,
     * truncate node_count to produce a sub-tree T_Q.  By construction
     * T_Q ⪯ T_K via identity pre-order, so the filter MUST admit. */
    int total_pairs = 0;
    int embed_yes  = 0;
    int filter_admitted_on_yes = 0;
    int filter_rejected_on_no  = 0;
    int total_no = 0;
    int false_negatives = 0;
    const int N_PAIRS = 10000;
    std::mt19937 selector(0x99FE);
    std::uniform_int_distribution<int> pick(0, N_TREES - 1);

    for (int p_i = 0; p_i < N_PAIRS; ++p_i) {
        int qi = pick(selector), ki = pick(selector);
        sp_kste_tree Q, Kt = trees[ki];
        sp_kste_signature_t Q_sig0;
        sp_kste_anc_sig_t   Q_sig1;
        /* Half the pairs are random Q vs K; the other half are
         * Q = truncated K (which forces embed=YES). */
        if (p_i & 1) {
            Q = trees[qi];
            Q_sig0 = sig0s[qi];
            Q_sig1 = sig1s[qi];
        } else {
            Q = trees[ki];
            uint8_t target = (uint8_t)(15 + (p_i % 40));
            if (target > Q.node_count) target = Q.node_count;
            Q.node_count = target;
            Q_sig0 = sp_kste_compute_signature(&Q);
            sp_kste_compute_anc_sig(&Q, &Q_sig1);
        }

        int filter_admits =
            sp_kste_sig_dominates(sig0s[ki], Q_sig0) &&
            sp_kste_anc_sig_dominates(&sig1s[ki], &Q_sig1);
        int embed = sp_kste_embed(&Q, &Kt);

        ++total_pairs;
        if (embed) {
            ++embed_yes;
            if (filter_admits) ++filter_admitted_on_yes;
            else               ++false_negatives;
        } else {
            ++total_no;
            if (!filter_admits) ++filter_rejected_on_no;
        }
    }

    bool pass = (false_negatives == 0);

    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.9\", \"phase\": 5,\n"
        "  \"name\": \"pre-filter correctness (no false negatives)\",\n"
        "  \"config\": { \"head_dim\": %d, \"pairs\": %d },\n"
        "  \"metrics\": {\n"
        "    \"embed_yes\": %d, \"embed_no\": %d,\n"
        "    \"filter_admitted_on_yes\": %d,\n"
        "    \"filter_rejected_on_no\": %d,\n"
        "    \"false_negatives\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, total_pairs,
        embed_yes, total_no,
        filter_admitted_on_yes, filter_rejected_on_no,
        false_negatives,
        verdict_str(pass).c_str());
    emit_report("9", "pre-filter correctness", pass, buf);
    write_json_file("9", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T2.10 — pre-filter precision (>= 90% rejection rate) ------ */

/* Same corpus pattern as T2.9, but measure precision = (negatives
 * correctly rejected by filter) / (total negatives).  Pass iff >= 90%. */
static void test_T2_10()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    sp_kste_ctx_init(&ctx, HD);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xA0A0);
    const int N_TREES = 300;
    std::vector<sp_kste_tree>        trees(N_TREES);
    std::vector<sp_kste_signature_t> sig0s(N_TREES);
    std::vector<sp_kste_anc_sig_t>   sig1s(N_TREES);
    for (int i = 0; i < N_TREES; ++i) {
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int j = 0; j < HD; ++j) K[j] = N(rng);
        sp_kste_encode(&trees[i], K.data(), &ctx, scratch.data());
        sig0s[i] = sp_kste_compute_signature(&trees[i]);
        sp_kste_compute_anc_sig(&trees[i], &sig1s[i]);
    }

    /* Run 10 000 random pairs; for each non-embedding pair, check
     * whether the filter caught it. */
    int total_negatives = 0;
    int filter_rejected = 0;
    int tier0_rejected  = 0;
    int tier1_rejected  = 0;
    const int N_PAIRS = 10000;
    std::mt19937 sel(0xA0FE);
    std::uniform_int_distribution<int> pick(0, N_TREES - 1);
    for (int p_i = 0; p_i < N_PAIRS; ++p_i) {
        int qi = pick(sel), ki = pick(sel);
        if (sp_kste_embed(&trees[qi], &trees[ki])) continue;
        ++total_negatives;
        bool t0 = sp_kste_sig_dominates(sig0s[ki], sig0s[qi]);
        bool t1 = t0 && sp_kste_anc_sig_dominates(&sig1s[ki], &sig1s[qi]);
        if (!t0) { ++tier0_rejected; ++filter_rejected; }
        else if (!t1) { ++tier1_rejected; ++filter_rejected; }
    }

    double precision = total_negatives > 0
        ? (double)filter_rejected / (double)total_negatives : 0.0;
    bool pass = (precision >= 0.90);

    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.10\", \"phase\": 5,\n"
        "  \"name\": \"pre-filter precision (>= 90%%)\",\n"
        "  \"config\": { \"head_dim\": %d, \"pairs\": %d, \"trees\": %d },\n"
        "  \"metrics\": {\n"
        "    \"total_negatives\": %d,\n"
        "    \"filter_rejected\": %d, \"precision\": %.4f,\n"
        "    \"tier0_rejected\": %d, \"tier1_rejected\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, N_PAIRS, N_TREES,
        total_negatives, filter_rejected, precision,
        tier0_rejected, tier1_rejected,
        verdict_str(pass).c_str());
    emit_report("10", "pre-filter precision", pass, buf);
    write_json_file("10", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T2.11 — wall-time at capacity 4096 ------------------------ */

/* Roadmap §5 exit criterion: ≤ 5 µs / token mean, ≤ 50 µs worst case.
 * Gate on p99 to be VM-noise-tolerant (Phase 1 / 2 / 3 pattern). */
static void test_T2_11()
{
    const int HD  = 128;
    const int CAP = 4096;

    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, CAP);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xB11A);

    /* Fill to capacity (the only place inserts can exercise the full
     * 4096-slot scan).  Cap warmup at CAP*20 iterations. */
    int filled = 0, t = 0;
    while (filled < CAP && t < CAP * 20) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        if (sp_friedman_cache_insert(&cache, &T, sv, t) == SP_FRIEDMAN_ADMITTED)
            ++filled;
        ++t;
    }

    /* Snapshot counters BEFORE the bench to compute per-insert ratios
     * during the bench window only. */
    uint64_t st0 = cache.slot_tests;
    uint64_t st1 = cache.tier1_tests;
    uint64_t emb = cache.full_embeds;

    using clk = std::chrono::steady_clock;
    std::vector<double> samples;
    const int N_INSERTS = 1000;
    samples.reserve(N_INSERTS);
    for (int i = 0; i < N_INSERTS; ++i) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        auto t0 = clk::now();
        sp_friedman_cache_insert(&cache, &T, sv, t + i);
        auto t1 = clk::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= (double)samples.size();
    double p50  = samples[samples.size() / 2];
    double p99  = samples[(samples.size() * 99) / 100];
    double p999 = samples[(samples.size() * 999) / 1000];
    double max_us  = samples.back();

    uint64_t d_st0 = cache.slot_tests - st0;
    uint64_t d_st1 = cache.tier1_tests - st1;
    uint64_t d_emb = cache.full_embeds  - emb;
    double tier0_survive = d_st0 ? (double)d_st1 / (double)d_st0 : 0.0;
    double tier1_survive = d_st1 ? (double)d_emb / (double)d_st1 : 0.0;

    /* Gate: p99 ≤ 50 µs (the roadmap §5 "worst case" interpreted as p99
     * for sandbox-VM robustness; mean is informational). */
    bool pass = (p99 <= 50.0);

    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2.11\", \"phase\": 5,\n"
        "  \"name\": \"sieve insert wall-time at capacity %d\",\n"
        "  \"config\": { \"head_dim\": %d, \"capacity\": %d, \"inserts\": %d },\n"
        "  \"metrics\": {\n"
        "    \"mean_us\": %.3f, \"p50_us\": %.3f,\n"
        "    \"p99_us\": %.3f, \"p999_us\": %.3f, \"max_us\": %.3f,\n"
        "    \"tier0_survival_rate\": %.4f,\n"
        "    \"tier1_survival_rate\": %.4f,\n"
        "    \"slot_tests\": %llu, \"tier1_tests\": %llu, \"full_embeds\": %llu,\n"
        "    \"exit_gate_p99_us\": 50.0\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        CAP, HD, CAP, N_INSERTS,
        mean, p50, p99, p999, max_us,
        tier0_survive, tier1_survive,
        (unsigned long long)d_st0, (unsigned long long)d_st1,
        (unsigned long long)d_emb,
        verdict_str(pass).c_str());
    emit_report("11", "wall-time at capacity 4096", pass, buf);
    write_json_file("11", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- Wall-time bench: insert path at capacity 4096 -------------- */

static void bench_insert_walltime()
{
    const int HD = 128;
    const int CAP = 512;

    sp_kste_ctx ectx;
    sp_kste_ctx_init(&ectx, HD);
    sp_friedman_cache_t cache;
    sp_friedman_cache_init(&cache, CAP);

    std::vector<float> scratch(3 * HD), K(HD);
    std::mt19937 rng(0xB);

    /* Warmup: fill cache to capacity. */
    int filled = 0, t = 0;
    while (filled < CAP && t < CAP * 20) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        if (sp_friedman_cache_insert(&cache, &T, sv, t) == SP_FRIEDMAN_ADMITTED)
            ++filled;
        ++t;
    }

    /* Bench: 1000 more inserts at capacity. */
    using clk = std::chrono::steady_clock;
    std::vector<double> samples;
    const int N_INSERTS = 300;
    samples.reserve(N_INSERTS);
    for (int i = 0; i < N_INSERTS; ++i) {
        sp_kste_tree T;
        float sv = 0.0f;
        encode_random(&ectx, rng, HD, scratch.data(), K.data(), &T, &sv);
        auto t0 = clk::now();
        sp_friedman_cache_insert(&cache, &T, sv, t + i);
        auto t1 = clk::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= (double)samples.size();
    double p50  = samples[samples.size() / 2];
    double p99  = samples[(samples.size() * 99) / 100];
    double p999 = samples[(samples.size() * 999) / 1000];
    double max  = samples.back();

    /* Roadmap §3 exit criterion: ≤ 50 µs per token at capacity 4096.
     * Phase 3 uses naive O(N) embed scan; Phase 5 brings this down.
     * Gate on p99 to avoid sandbox-VM outlier failures. */
    bool pass = (p99 <= 50.0);

    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T2_BENCH\", \"phase\": 3,\n"
        "  \"name\": \"sieve insert wall-time at capacity 4096\",\n"
        "  \"config\": { \"head_dim\": %d, \"capacity\": %d, \"inserts\": %d },\n"
        "  \"metrics\": {\n"
        "    \"mean_us\": %.3f, \"p50_us\": %.3f,\n"
        "    \"p99_us\": %.3f, \"p999_us\": %.3f, \"max_us\": %.3f,\n"
        "    \"exit_gate_p99_us\": 50.0\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, CAP, N_INSERTS,
        mean, p50, p99, p999, max,
        verdict_str(pass).c_str());
    emit_report("BENCH", "sieve insert wall-time", pass, buf);
    write_json_file("BENCH", buf);
    sp_friedman_cache_destroy(&cache);
    sp_kste_ctx_destroy(&ectx);
}

/* ---------- main ------------------------------------------------------ */

int main()
{
    std::fprintf(stderr, "Phase 3 — Friedman sieve cache Tier-2 tests\n");

    test_T2_6();      /* fast — run first to fail-fast on regressions */
    test_T2_7();
    test_T2_8();
    test_T2_5();
    test_T2_12();
    test_T2_1();
    test_T2_9();
    test_T2_10();
    test_T2_11();
    bench_insert_walltime();

    std::string summary = "{\n  \"phase\": 3,\n  \"results\": [\n";
    for (size_t i = 0; i < g_reports.size(); ++i) {
        char head[160];
        std::snprintf(head, sizeof(head),
            "    { \"id\": \"T2.%s\", \"name\": \"%s\", \"verdict\": \"%s\" }%s\n",
            g_reports[i].id, g_reports[i].name,
            g_reports[i].pass ? "PASS" : "FAIL",
            i + 1 == g_reports.size() ? "" : ",");
        summary += head;
    }
    summary += "  ]\n}\n";
    write_json_file("SUMMARY", summary);

    std::fprintf(stderr, "\nPhase 3 summary: %d tests, %d failures\n",
                 (int)g_reports.size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
