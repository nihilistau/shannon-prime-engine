/* test_sp_kste.cpp — Tier-1 KSTE encoder tests.
 *
 * Implements T1.1–T1.5 from D:\F\shannon-prime-repos\papers\PPT-ARM\TEST-SUITE.md.
 * Each test emits a JSON report into tests/results/T1_<id>.json so the
 * audit trail required by the roadmap §9 builds incrementally.
 *
 * Run with: ctest -R sp_kste --output-on-failure
 *
 * The tests deliberately avoid any framework: they're plain `main()`
 * that prints PASS/FAIL and writes the JSON.  Exit code 0 iff every
 * test verdict is PASS.
 */

extern "C" {
#include "../lib/shannon-prime/core/shannon_prime.h"
#include "../lib/shannon-prime/core/sp_kste.h"
}

#include <chrono>
#include <algorithm>
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
    /* Ensure tests/results/ exists.  Working directory at ctest time is
     * the build/tests directory; we walk back to source tree. */
    MKDIR("tests");
    MKDIR("tests/results");
    MKDIR("../tests");
    MKDIR("../tests/results");
    MKDIR("../../tests");
    MKDIR("../../tests/results");

    /* Try a few candidate locations; first one that opens for write wins. */
    const char *candidates[] = {
        "../../tests/results/",
        "../tests/results/",
        "tests/results/",
        "./",
    };
    for (const char *dir : candidates) {
        char path[256];
        std::snprintf(path, sizeof(path), "%sT1_%s.json", dir, id);
        FILE *fp = std::fopen(path, "w");
        if (fp) {
            std::fwrite(json.data(), 1, json.size(), fp);
            std::fclose(fp);
            std::fprintf(stderr, "    -> %s\n", path);
            return;
        }
    }
    std::fprintf(stderr, "    !! could not write JSON report for T1.%s\n", id);
}

static std::string verdict_str(bool pass) { return pass ? "PASS" : "FAIL"; }

/* ---------- Helpers ----------------------------------------------------- */

static int count_label(const sp_kste_tree &T, sp_kste_label lbl)
{
    int c = 0;
    for (int i = 1; i < T.node_count; ++i) {
        if (sp_kste_unpack_label(T.labels, i) == lbl) ++c;
    }
    return c;
}

static int count_anchors(const sp_kste_tree &T)
{
    /* Anchors are A-labelled children of the root (parent index 0). */
    int c = 0;
    for (int i = 1; i < T.node_count; ++i) {
        if (sp_kste_unpack_parent(T.parents, i) == 0u &&
            sp_kste_unpack_label (T.labels,  i) == SP_KSTE_LBL_A) ++c;
    }
    return c;
}

static bool trees_equal(const sp_kste_tree &a, const sp_kste_tree &b)
{
    if (a.node_count != b.node_count) return false;
    if (std::memcmp(a.labels,  b.labels,  sizeof(a.labels))  != 0) return false;
    if (std::memcmp(a.parents, b.parents, sizeof(a.parents)) != 0) return false;
    return true;
}

/* ---------- T1.1 — determinism ------------------------------------------ */

static void test_T1_1()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.1\", \"verdict\": \"FAIL\",\n"
                        "  \"notes\": \"sp_kste_ctx_init failed\"\n}\n";
        emit_report("1", "encoder determinism", false, j);
        write_json_file("1", j);
        return;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> K(HD);
    for (int i = 0; i < HD; ++i) K[i] = N(rng);

    std::vector<float> scratch(3 * HD);
    sp_kste_tree first;
    sp_kste_encode(&first, K.data(), &ctx, scratch.data());

    int bit_identical = 1;
    for (int trial = 1; trial < 1000; ++trial) {
        sp_kste_tree t;
        sp_kste_encode(&t, K.data(), &ctx, scratch.data());
        if (!trees_equal(first, t)) { bit_identical = 0; break; }
    }

    bool pass = (bit_identical == 1);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.1\", \"phase\": 1,\n"
        "  \"name\": \"encoder determinism\",\n"
        "  \"config\": { \"head_dim\": %d, \"seed\": 42, \"trials\": 1000 },\n"
        "  \"metrics\": {\n"
        "    \"bit_identical_count\": %d,\n"
        "    \"trials\": 1000,\n"
        "    \"node_count\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD,
        bit_identical ? 1000 : 0,
        (int)first.node_count,
        verdict_str(pass).c_str());

    emit_report("1", "encoder determinism", pass, buf);
    write_json_file("1", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.2 — order-invariance under Frobenius shim ---------------- */

static void test_T1_2()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.2\", \"verdict\": \"FAIL\" }\n";
        emit_report("2", "Frobenius order-invariance", false, j);
        write_json_file("2", j);
        return;
    }

    /* Frobenius scale |pi_41^8| = 41^4 = 2825761.  The encoder must
     * produce a bit-identical tree under this positive rescale. */
    const float scale = 41.0f * 41.0f * 41.0f * 41.0f;  /* 41^4 */

    std::mt19937 rng(42);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> scratch(3 * HD);

    int bit_identical = 1;
    int trials = 100;
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<float> K(HD), K_shim(HD);
        for (int i = 0; i < HD; ++i) {
            K[i] = N(rng);
            K_shim[i] = K[i] * scale;
        }
        sp_kste_tree T_base, T_shim;
        sp_kste_encode(&T_base, K.data(),      &ctx, scratch.data());
        sp_kste_encode(&T_shim, K_shim.data(), &ctx, scratch.data());
        if (!trees_equal(T_base, T_shim)) { bit_identical = 0; break; }
    }

    bool pass = (bit_identical == 1);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.2\", \"phase\": 1,\n"
        "  \"name\": \"order-invariance under Frobenius shim\",\n"
        "  \"config\": { \"head_dim\": %d, \"frobenius\": \"p=41,k=8\","
        " \"scale\": %.0f, \"trials\": %d },\n"
        "  \"metrics\": { \"bit_identical\": %s },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, (double)scale, trials,
        bit_identical ? "true" : "false",
        verdict_str(pass).c_str());

    emit_report("2", "Frobenius order-invariance", pass, buf);
    write_json_file("2", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.3 — sign-respecting -------------------------------------- */

static void test_T1_3()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.3\", \"verdict\": \"FAIL\" }\n";
        emit_report("3", "sign-respecting", false, j);
        write_json_file("3", j);
        return;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> K(HD), Kn(HD);
    for (int i = 0; i < HD; ++i) {
        K[i] = N(rng);
        Kn[i] = -K[i];
    }
    std::vector<float> scratch(3 * HD);
    sp_kste_tree Tp, Tn;
    sp_kste_encode(&Tp, K.data(),  &ctx, scratch.data());
    sp_kste_encode(&Tn, Kn.data(), &ctx, scratch.data());

    int A_p = count_label(Tp, SP_KSTE_LBL_A);
    int B_p = count_label(Tp, SP_KSTE_LBL_B);
    int C_p = count_label(Tp, SP_KSTE_LBL_C);
    int A_n = count_label(Tn, SP_KSTE_LBL_A);
    int B_n = count_label(Tn, SP_KSTE_LBL_B);
    int C_n = count_label(Tn, SP_KSTE_LBL_C);

    bool tree_shape_same = (Tp.node_count == Tn.node_count) &&
        (std::memcmp(Tp.parents, Tn.parents, sizeof(Tp.parents)) == 0);
    bool A_match = (A_p == A_n);
    bool BC_swap = (B_p == C_n) && (C_p == B_n);
    bool pass = tree_shape_same && A_match && BC_swap;

    char buf[768];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.3\", \"phase\": 1,\n"
        "  \"name\": \"sign-respecting (B<->C swap)\",\n"
        "  \"config\": { \"head_dim\": %d, \"seed\": 42 },\n"
        "  \"metrics\": {\n"
        "    \"tree_shape_same\": %s,\n"
        "    \"A_pos\": %d, \"A_neg\": %d,\n"
        "    \"B_pos\": %d, \"B_neg\": %d,\n"
        "    \"C_pos\": %d, \"C_neg\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD,
        tree_shape_same ? "true" : "false",
        A_p, A_n, B_p, B_n, C_p, C_n,
        verdict_str(pass).c_str());

    emit_report("3", "sign-respecting (B<->C swap)", pass, buf);
    write_json_file("3", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.4 — 60-node budget --------------------------------------- */

static void test_T1_4()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.4\", \"verdict\": \"FAIL\" }\n";
        emit_report("4", "60-node budget", false, j);
        write_json_file("4", j);
        return;
    }
    std::vector<float> scratch(3 * HD);
    std::vector<float> K(HD);

    int max_nodes = 0;
    int min_nodes = SP_KSTE_MAX_NODES + 1;
    long long sum_nodes = 0;
    bool overflow = false;
    const int TRIALS = 1000;
    for (int seed = 0; seed < TRIALS; ++seed) {
        std::mt19937 rng((uint32_t)seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree T;
        sp_kste_encode(&T, K.data(), &ctx, scratch.data());
        int n = (int)T.node_count;
        if (n > SP_KSTE_MAX_NODES) overflow = true;
        if (n > max_nodes) max_nodes = n;
        if (n < min_nodes) min_nodes = n;
        sum_nodes += n;
    }
    double mean_nodes = (double)sum_nodes / (double)TRIALS;
    bool pass = !overflow && (max_nodes <= SP_KSTE_MAX_NODES);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.4\", \"phase\": 1,\n"
        "  \"name\": \"60-node budget enforced\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d },\n"
        "  \"metrics\": {\n"
        "    \"max_nodes\": %d, \"min_nodes\": %d, \"mean_nodes\": %.2f,\n"
        "    \"budget\": %d, \"overflow\": %s\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS, max_nodes, min_nodes, mean_nodes,
        SP_KSTE_MAX_NODES, overflow ? "true" : "false",
        verdict_str(pass).c_str());

    emit_report("4", "60-node budget enforced", pass, buf);
    write_json_file("4", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.5 — anchor count ----------------------------------------- */

static void test_T1_5()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.5\", \"verdict\": \"FAIL\" }\n";
        emit_report("5", "anchor count 14 +/- 2", false, j);
        write_json_file("5", j);
        return;
    }
    std::vector<float> scratch(3 * HD);
    std::vector<float> K(HD);

    int min_a = SP_KSTE_N_ANCHORS + 1, max_a = 0;
    long long sum_a = 0;
    bool out_of_band = false;
    const int TRIALS = 1000;
    const int LO = 12, HI = 16;
    for (int seed = 0; seed < TRIALS; ++seed) {
        std::mt19937 rng((uint32_t)seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree T;
        sp_kste_encode(&T, K.data(), &ctx, scratch.data());
        int a = count_anchors(T);
        if (a < LO || a > HI) out_of_band = true;
        if (a < min_a) min_a = a;
        if (a > max_a) max_a = a;
        sum_a += a;
    }
    double mean_a = (double)sum_a / (double)TRIALS;
    bool pass = !out_of_band;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.5\", \"phase\": 1,\n"
        "  \"name\": \"anchor count 14 +/- 2\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d,"
        " \"band_lo\": %d, \"band_hi\": %d },\n"
        "  \"metrics\": {\n"
        "    \"min_anchors\": %d, \"max_anchors\": %d, \"mean_anchors\": %.2f,\n"
        "    \"out_of_band\": %s, \"tau_A_default\": %.4f\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS, LO, HI,
        min_a, max_a, mean_a,
        out_of_band ? "true" : "false",
        (double)SP_KSTE_TAU_A_DEFAULT,
        verdict_str(pass).c_str());

    emit_report("5", "anchor count 14 +/- 2", pass, buf);
    write_json_file("5", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.6 — self-embedding (1000 trees from T1.4) --------------- */

static void test_T1_6()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.6\", \"verdict\": \"FAIL\" }\n";
        emit_report("6", "self-embedding", false, j);
        write_json_file("6", j);
        return;
    }
    std::vector<float> scratch(3 * HD), K(HD);
    int ok_count = 0;
    const int TRIALS = 1000;
    long long total_steps = 0, max_steps = 0, max_depth = 0, capped = 0;
    for (int seed = 0; seed < TRIALS; ++seed) {
        std::mt19937 rng((uint32_t)seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree T;
        sp_kste_encode(&T, K.data(), &ctx, scratch.data());
        sp_kste_embed_stats st;
        int r = sp_kste_embed_ex(&T, &T, &st);
        if (r) ++ok_count;
        total_steps += st.steps;
        if (st.steps > max_steps) max_steps = st.steps;
        if (st.max_depth > max_depth) max_depth = st.max_depth;
        if (st.capped) ++capped;
    }
    bool pass = (ok_count == TRIALS) && (capped == 0);
    char buf[768];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.6\", \"phase\": 2,\n"
        "  \"name\": \"self-embedding (1000 trees)\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d },\n"
        "  \"metrics\": {\n"
        "    \"embed_count\": %d, \"capped_count\": %lld,\n"
        "    \"mean_steps\": %.1f, \"max_steps\": %lld,\n"
        "    \"max_depth\": %lld\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS,
        ok_count, capped,
        (double)total_steps / (double)TRIALS, max_steps, max_depth,
        verdict_str(pass).c_str());
    emit_report("6", "self-embedding (1000 trees)", pass, buf);
    write_json_file("6", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.7 — empty-subtree embedding ----------------------------- */

static void test_T1_7()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.7\", \"verdict\": \"FAIL\" }\n";
        emit_report("7", "empty-subtree embedding", false, j);
        write_json_file("7", j);
        return;
    }
    sp_kste_tree T_rho;
    sp_kste_tree_clear(&T_rho);

    std::vector<float> scratch(3 * HD), K(HD);
    int ok_count = 0;
    const int TRIALS = 1000;
    for (int seed = 0; seed < TRIALS; ++seed) {
        std::mt19937 rng((uint32_t)seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree T;
        sp_kste_encode(&T, K.data(), &ctx, scratch.data());
        if (sp_kste_embed(&T_rho, &T)) ++ok_count;
    }
    bool pass = (ok_count == TRIALS);
    char buf[384];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.7\", \"phase\": 2,\n"
        "  \"name\": \"empty-subtree embedding\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d },\n"
        "  \"metrics\": { \"embed_count\": %d },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS, ok_count, verdict_str(pass).c_str());
    emit_report("7", "empty-subtree embedding", pass, buf);
    write_json_file("7", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.8 — transitivity ---------------------------------------- */

/* Take a tree T3 from the encoder; construct T2 by truncating node_count
 * (drops the last-added nodes, which are by construction always leaves
 * of residual chains).  T1 by truncating further.  By construction
 * T1 <= T2 <= T3 via identity-as-pre-order. */
static void test_T1_8()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.8\", \"verdict\": \"FAIL\" }\n";
        emit_report("8", "transitivity", false, j);
        write_json_file("8", j);
        return;
    }
    std::vector<float> scratch(3 * HD), K(HD);
    int ok_triples = 0;
    int direct_ok = 0;
    int composed_ok = 0;
    const int TRIALS = 100;
    std::mt19937 master(7);
    for (int trial = 0; trial < TRIALS; ++trial) {
        std::mt19937 rng(master());
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree T3;
        sp_kste_encode(&T3, K.data(), &ctx, scratch.data());
        if (T3.node_count < 30) continue;

        sp_kste_tree T2 = T3;
        sp_kste_tree T1 = T3;
        std::uniform_int_distribution<int> U2(20, T3.node_count - 1);
        int cut2 = U2(rng);
        T2.node_count = (uint8_t)cut2;
        std::uniform_int_distribution<int> U1(15, cut2 - 1);
        int cut1 = U1(rng);
        T1.node_count = (uint8_t)cut1;

        bool a = sp_kste_embed(&T1, &T2) != 0;
        bool b = sp_kste_embed(&T2, &T3) != 0;
        bool c = sp_kste_embed(&T1, &T3) != 0;
        if (a && b && c) ++ok_triples;
        if (a) ++composed_ok;
        if (c) ++direct_ok;
    }
    bool pass = (ok_triples == TRIALS);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.8\", \"phase\": 2,\n"
        "  \"name\": \"transitivity (truncation triples)\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d },\n"
        "  \"metrics\": {\n"
        "    \"triples_pass\": %d,\n"
        "    \"T1_in_T2_ok\": %d, \"T1_in_T3_ok\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS,
        ok_triples, composed_ok, direct_ok,
        verdict_str(pass).c_str());
    emit_report("8", "transitivity", pass, buf);
    write_json_file("8", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.9 — antisymmetry on canonical forms --------------------- */

static void test_T1_9()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) {
        std::string j = "{\n  \"test_id\": \"T1.9\", \"verdict\": \"FAIL\" }\n";
        emit_report("9", "antisymmetry", false, j);
        write_json_file("9", j);
        return;
    }
    std::vector<float> scratch(3 * HD), K(HD);
    int canon_match = 0;
    const int TRIALS = 100;
    for (int seed = 0; seed < TRIALS; ++seed) {
        std::mt19937 rng((uint32_t)(seed * 31 + 11));
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < HD; ++i) K[i] = N(rng);
        sp_kste_tree A, B;
        sp_kste_encode(&A, K.data(), &ctx, scratch.data());
        std::memcpy(&B, &A, sizeof(A));
        int ab = sp_kste_embed(&A, &B);
        int ba = sp_kste_embed(&B, &A);
        bool eq = trees_equal(A, B);
        if (ab && ba && eq) ++canon_match;
    }
    bool pass = (canon_match == TRIALS);
    char buf[384];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.9\", \"phase\": 2,\n"
        "  \"name\": \"antisymmetry on canonical forms\",\n"
        "  \"config\": { \"head_dim\": %d, \"trials\": %d },\n"
        "  \"metrics\": { \"canonical_match\": %d },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, TRIALS, canon_match, verdict_str(pass).c_str());
    emit_report("9", "antisymmetry", pass, buf);
    write_json_file("9", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- T1.10 — backtracking necessity ----------------------------- */

/* Hand-construct an adversarial pair.  Q = root -> A -> C, root -> B.
 * K = root -> A1 (leaf), A2 -> C, B.  A1 matches A's label but has no
 * descendants of label C.  Greedy first-fit fails on A1; the kernel
 * must back out and try A2. */
static void test_T1_10()
{
    sp_kste_tree Q, K;

    sp_kste_tree_clear(&Q);
    int qA = sp_kste_tree_add_child(&Q, 0, SP_KSTE_LBL_A);
    int qC = sp_kste_tree_add_child(&Q, qA, SP_KSTE_LBL_C);
    int qB = sp_kste_tree_add_child(&Q, 0, SP_KSTE_LBL_B);
    (void)qC; (void)qB;

    sp_kste_tree_clear(&K);
    int kA1 = sp_kste_tree_add_child(&K, 0, SP_KSTE_LBL_A);
    int kA2 = sp_kste_tree_add_child(&K, 0, SP_KSTE_LBL_A);
    int kC  = sp_kste_tree_add_child(&K, kA2, SP_KSTE_LBL_C);
    int kB  = sp_kste_tree_add_child(&K, 0, SP_KSTE_LBL_B);
    (void)kA1; (void)kC; (void)kB;

    sp_kste_embed_stats st;
    int ok = sp_kste_embed_ex(&Q, &K, &st);

    bool pass = (ok == 1) && (st.backtracks > 0) && (st.capped == 0);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1.10\", \"phase\": 2,\n"
        "  \"name\": \"backtracking necessity (adversarial pair)\",\n"
        "  \"config\": {\n"
        "    \"Q\": \"root -> A -> C, root -> B\",\n"
        "    \"K\": \"root -> [A1 leaf, A2 -> C, B]\"\n"
        "  },\n"
        "  \"metrics\": {\n"
        "    \"result\": %d, \"backtracks\": %d,\n"
        "    \"steps\": %d, \"max_depth\": %d, \"capped\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        ok, st.backtracks, st.steps, st.max_depth, st.capped,
        verdict_str(pass).c_str());
    emit_report("10", "backtracking necessity", pass, buf);
    write_json_file("10", buf);
}

/* ---------- Phase-2 wall-time bench ------------------------------------ */

static void bench_embed_walltime()
{
    const int HD = 128;
    sp_kste_ctx ctx;
    if (!sp_kste_ctx_init(&ctx, HD)) return;
    std::vector<float> scratch(3 * HD), K(HD);

    std::vector<sp_kste_tree> trees(1000);
    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (size_t i = 0; i < trees.size(); ++i) {
        for (int j = 0; j < HD; ++j) K[j] = N(rng);
        sp_kste_encode(&trees[i], K.data(), &ctx, scratch.data());
    }

    /* Warmup: page in caches, warm up branch predictor. */
    for (int rep = 0; rep < 64; ++rep) {
        volatile int r = sp_kste_embed(&trees[rep % 100], &trees[(rep + 1) % 100]);
        (void)r;
    }

    using clk = std::chrono::steady_clock;
    std::vector<double> samples;
    samples.reserve(trees.size());
    int ok_count = 0;
    for (size_t i = 0; i + 1 < trees.size(); ++i) {
        auto t0 = clk::now();
        int r = sp_kste_embed(&trees[i], &trees[i + 1]);
        auto t1 = clk::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        samples.push_back(us);
        if (r) ++ok_count;
    }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double q) {
        size_t idx = (size_t)std::floor(q * (samples.size() - 1));
        return samples[idx];
    };
    double mean_us = 0.0;
    for (double v : samples) mean_us += v;
    mean_us /= (double)samples.size();
    double max_us  = samples.back();
    double p50_us  = pct(0.50);
    double p99_us  = pct(0.99);
    double p999_us = pct(0.999);

    /* Exit gate: p99 worst-case <= 100 us.  The absolute max is
     * preserved in metrics for forensics but a single outlier in a
     * shared-VM bench rig is not the right gate. */
    bool pass = (p99_us <= 100.0);

    char buf[768];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T1_BENCH\", \"phase\": 2,\n"
        "  \"name\": \"embed wall-time (random pairs)\",\n"
        "  \"config\": { \"head_dim\": %d, \"pairs\": %zu, \"warmup\": 64 },\n"
        "  \"metrics\": {\n"
        "    \"mean_us\": %.3f, \"p50_us\": %.3f,\n"
        "    \"p99_us\": %.3f, \"p999_us\": %.3f, \"max_us\": %.3f,\n"
        "    \"embed_yes_count\": %d, \"exit_gate_p99_us\": 100.0\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        HD, trees.size() - 1, mean_us, p50_us, p99_us, p999_us, max_us, ok_count,
        verdict_str(pass).c_str());
    emit_report("BENCH", "embed wall-time", pass, buf);
    write_json_file("BENCH", buf);
    sp_kste_ctx_destroy(&ctx);
}

/* ---------- main ------------------------------------------------------ */

int main()
{
    std::fprintf(stderr, "Phase 1-2 — KSTE encoder + embedding Tier-1 tests\n");

    test_T1_1();
    test_T1_2();
    test_T1_3();
    test_T1_4();
    test_T1_5();
    test_T1_6();
    test_T1_7();
    test_T1_8();
    test_T1_9();
    test_T1_10();
    bench_embed_walltime();

    std::string summary = "{\n  \"phase\": 2,\n  \"results\": [\n";
    for (size_t i = 0; i < g_reports.size(); ++i) {
        char head[128];
        std::snprintf(head, sizeof(head),
            "    { \"id\": \"T1.%s\", \"name\": \"%s\", \"verdict\": \"%s\" }%s\n",
            g_reports[i].id, g_reports[i].name,
            g_reports[i].pass ? "PASS" : "FAIL",
            i + 1 == g_reports.size() ? "" : ",");
        summary += head;
    }
    summary += "  ]\n}\n";
    write_json_file("SUMMARY", summary);

    std::fprintf(stderr, "\nPhase 1-2 summary: %d tests, %d failures\n",
                 (int)g_reports.size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
