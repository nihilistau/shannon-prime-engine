// Shannon-Prime Engine — Phase 7: Ultraproduct attention tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Implements three tests from TEST-SUITE.md / IMPLEMENTATION-ROADMAP.md:
//   T3.1 — principal ultrafilter ⇒ Top-1 attention. Hand-crafted scores
//           with a unique argmax position p*; verify that the kernel's
//           output equals V_{p*} bit-identically through the encode /
//           decode round-trip.
//   T3.2 — Łoś property on a toy. Pick a first-order property φ on the
//           V vectors ("V[0] > 5.0"). Verify φ(UltraAttn) == φ(V_{p*})
//           — on a finite cache every ultrafilter is principal, so this
//           reduces to checking the top-1 selection respects the
//           property when the U-large set is precisely {p*}.
//   T3.6 — Choice operator F canonicality. Generate 100 random sp_kste_
//           tree objects, shuffle the pointer array 1000 times, and
//           verify that sp_kste_select_canonical returns a pointer
//           whose byte-content is identical to the first call's result
//           on every shuffle.
//
// Each test writes a JSON report to ../../tests/results/T3_<id>.json.

#include "../src/sp_ultraproduct_attn.h"
#include "../src/sp_ok_encode.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_kste.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  define MKDIR(p) mkdir(p, 0755)
#endif

using namespace sp::engine;

struct TestReport {
    const char *id;
    const char *name;
    bool        pass;
    std::string json;
};
static std::vector<TestReport> g_reports;
static int g_failures = 0;

static void emit(const char *id, const char *name, bool pass, const std::string &json)
{
    g_reports.push_back({id, name, pass, json});
    if (!pass) ++g_failures;
    std::printf("  [%s] %s — %s\n", pass ? "PASS" : "FAIL", id, name);
}

static void write_json_file(const char *id, const std::string &body)
{
    MKDIR("../../tests");
    MKDIR("../../tests/results");
    char path[256];
    std::snprintf(path, sizeof(path), "../../tests/results/T3_%s.json", id);
    FILE *fp = std::fopen(path, "wb");
    if (fp) {
        std::fwrite(body.data(), 1, body.size(), fp);
        std::fclose(fp);
        std::printf("    -> %s\n", path);
    }
}

// =========================================================================
// T3.1 — Principal ⇒ Top-1 attention
// =========================================================================
//
// Setup: 1 head, head_dim=4, T=8.  Q and K crafted so that QK^T peaks at
// position p* = 5.  V is set so that V[:, p*] = [10, 20, 30, 40] which
// is unique and far from any other column.  The ultraproduct kernel
// must return exactly [10, 20, 30, 40] (up to encode/decode rounding).
static void test_T3_1()
{
    constexpr int n_head = 1, head_dim = 4, T = 8;
    constexpr int d_q = n_head * head_dim;
    constexpr int p_star = 5;

    std::vector<float> q_fp(d_q, 0.0f);
    std::vector<float> k_fp(d_q * T, 0.0f);
    std::vector<float> v_fp(d_q * T, 0.0f);

    // Make Q a fixed unit-ish vector.
    q_fp[0] = 1.0f;
    q_fp[1] = 1.0f;
    q_fp[2] = 1.0f;
    q_fp[3] = 1.0f;

    // Storage: k_fp[(h*head_dim + d) * T + t] = K_h[d, t].
    // Set K[:, p*] = [1, 1, 1, 1] (matches Q) and K[:, other] = small noise.
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < head_dim; ++d) {
            float val = (t == p_star) ? 1.0f : 0.01f * ((d & 1) ? -1.0f : 1.0f);
            k_fp[(0 * head_dim + d) * T + t] = val;
        }
    }

    // Set V[:, p*] = [10, 20, 30, 40].  Other columns small.
    const float v_target[head_dim] = { 10.0f, 20.0f, 30.0f, 40.0f };
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < head_dim; ++d) {
            v_fp[(0 * head_dim + d) * T + t] =
                (t == p_star) ? v_target[d] : 0.01f * (float)(d - t);
        }
    }

    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor Q, K, V, OUT;
    int64_t q_shape[4] = { 1, d_q, 1, 1 };
    int64_t k_shape[4] = { T, d_q, 1, 1 };
    int64_t v_shape[4] = { T, d_q, 1, 1 };
    int64_t out_shape[4] = { 1, d_q, 1, 1 };
    sp_ok_encode_from_fp32(Q, q_fp.data(), 2, q_shape, 1 << 14, arena);
    sp_ok_encode_from_fp32(K, k_fp.data(), 2, k_shape, 1 << 14, arena);
    sp_ok_encode_from_fp32(V, v_fp.data(), 2, v_shape, 1 << 14, arena);
    OUT.reset(2, out_shape);
    arena.alloc_tensor(OUT);
    OUT.scale_recip = 1 << 14;

    int32_t selected[1] = { -1 };
    sp_ultraproduct_attn_principal(Q, K, V, OUT,
                                    n_head, /*n_kv_head*/ 1, head_dim,
                                    /*t_valid*/ T,
                                    /*t_stride*/ T,
                                    /*pos_offset*/ T - 1, // single query is the last position
                                    /*swa*/ 0,
                                    /*softcap*/ 0.0f,
                                    /*evicted_mask*/ nullptr,
                                    /*evicted_gamma*/ 0.0f,
                                    selected);

    std::vector<float> got(d_q);
    sp_ok_decode_to_fp32(got.data(), OUT);

    bool pass_pos = (selected[0] == p_star);
    bool pass_val = true;
    for (int d = 0; d < head_dim; ++d) {
        if (std::abs(got[d] - v_target[d]) > 0.01f) pass_val = false;
    }
    bool pass = pass_pos && pass_val;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T3.1\", \"phase\": 7,\n"
        "  \"name\": \"principal ultrafilter implies top-1\",\n"
        "  \"metrics\": {\n"
        "    \"selected_p_star\": %d,\n"
        "    \"expected_p_star\": %d,\n"
        "    \"out_v\": [%.4f, %.4f, %.4f, %.4f],\n"
        "    \"expected_v\": [%.4f, %.4f, %.4f, %.4f]\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        (int)selected[0], p_star,
        got[0], got[1], got[2], got[3],
        v_target[0], v_target[1], v_target[2], v_target[3],
        pass ? "PASS" : "FAIL");
    emit("1", "principal ultrafilter implies top-1", pass, buf);
    write_json_file("1", buf);
}

// =========================================================================
// T3.2 — Łoś property on toy
// =========================================================================
//
// First-order property φ(V) := "V[0] > 5.0".  Setup T=4 with
//   V[:, 0] = [ 1, 0, 0, 0]  → φ false
//   V[:, 1] = [10, 0, 0, 0]  → φ true
//   V[:, 2] = [ 2, 0, 0, 0]  → φ false
//   V[:, 3] = [ 8, 0, 0, 0]  → φ true
// Configure Q/K so QK^T peaks at t = 1 (φ true).  By Łoś:
//   φ(ult_{U_1} V) ⇔ φ(V_1) = true.
// Then run again with peak at t = 0 (φ false).  By Łoś:
//   φ(ult_{U_0} V) ⇔ φ(V_0) = false.
// Both assertions must hold for the test to PASS.
static void test_T3_2()
{
    constexpr int n_head = 1, head_dim = 4, T = 4;
    constexpr int d_q = n_head * head_dim;

    auto run_one = [&](int want_p, float& got_v0) {
        std::vector<float> q_fp(d_q, 0.0f);
        std::vector<float> k_fp(d_q * T, 0.0f);
        std::vector<float> v_fp(d_q * T, 0.0f);

        q_fp[0] = 1.0f;
        // Drive argmax to want_p by giving K[:, want_p] a large positive
        // component aligned with Q.
        for (int t = 0; t < T; ++t) {
            float val = (t == want_p) ? 1.0f : -0.5f;
            k_fp[0 * T + t] = val;
        }
        // V[0, :] = [1, 10, 2, 8].
        const float v0_col[4] = { 1.0f, 10.0f, 2.0f, 8.0f };
        for (int t = 0; t < T; ++t) {
            v_fp[0 * T + t] = v0_col[t];
        }

        sp_ok_arena arena(64 * 1024);
        sp_ok_tensor Q, K, V, OUT;
        int64_t q_shape[4] = { 1, d_q, 1, 1 };
        int64_t kv_shape[4] = { T, d_q, 1, 1 };
        int64_t out_shape[4] = { 1, d_q, 1, 1 };
        sp_ok_encode_from_fp32(Q, q_fp.data(), 2, q_shape, 1 << 14, arena);
        sp_ok_encode_from_fp32(K, k_fp.data(), 2, kv_shape, 1 << 14, arena);
        sp_ok_encode_from_fp32(V, v_fp.data(), 2, kv_shape, 1 << 14, arena);
        OUT.reset(2, out_shape);
        arena.alloc_tensor(OUT);
        OUT.scale_recip = 1 << 14;

        int32_t selected[1] = { -1 };
        sp_ultraproduct_attn_principal(Q, K, V, OUT,
                                        n_head, 1, head_dim,
                                        T, T, T - 1, 0, 0.0f,
                                        nullptr, 0.0f, selected);
        std::vector<float> got(d_q);
        sp_ok_decode_to_fp32(got.data(), OUT);
        got_v0 = got[0];
        return (int)selected[0];
    };

    float got_v0_high = 0, got_v0_low = 0;
    int sel_high = run_one(/*want_p=*/1, got_v0_high);
    int sel_low  = run_one(/*want_p=*/0, got_v0_low);

    bool phi_high = got_v0_high > 5.0f;
    bool phi_low  = got_v0_low  > 5.0f;

    bool pass = (sel_high == 1) && phi_high && (sel_low == 0) && !phi_low;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T3.2\", \"phase\": 7,\n"
        "  \"name\": \"Los property on toy\",\n"
        "  \"metrics\": {\n"
        "    \"peak_at_1\": {\"selected\": %d, \"v0\": %.4f, \"phi\": %s},\n"
        "    \"peak_at_0\": {\"selected\": %d, \"v0\": %.4f, \"phi\": %s}\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        sel_high, got_v0_high, phi_high ? "true" : "false",
        sel_low,  got_v0_low,  phi_low  ? "true" : "false",
        pass ? "PASS" : "FAIL");
    emit("2", "Los property on toy", pass, buf);
    write_json_file("2", buf);
}

// =========================================================================
// T3.6 — Choice operator F canonicality (1000 shuffled invocations)
// =========================================================================
static void test_T3_6()
{
    constexpr int N = 100;
    std::vector<sp_kste_tree> trees(N);
    std::vector<const sp_kste_tree *> ptrs(N);

    std::mt19937 rng(0x5eed);

    // Build N random valid trees by repeated add_child.
    for (int i = 0; i < N; ++i) {
        sp_kste_tree_clear(&trees[i]);
        int target_nodes = 5 + (int)(rng() % 50); // 5..54 nodes
        for (int j = 1; j < target_nodes; ++j) {
            int parent = (int)(rng() % j);          // any prior node
            int lbl_pick = (int)(rng() % 3);
            sp_kste_label lbl = (lbl_pick == 0) ? SP_KSTE_LBL_A
                              : (lbl_pick == 1) ? SP_KSTE_LBL_B
                                                : SP_KSTE_LBL_C;
            sp_kste_tree_add_child(&trees[i], parent, lbl);
        }
        ptrs[i] = &trees[i];
    }

    // First invocation establishes the canonical reference.
    const sp_kste_tree *ref = sp_kste_select_canonical(ptrs.data(), N);
    if (!ref) {
        emit("6", "choice operator canonicality", false,
             "{\n  \"test_id\": \"T3.6\",\n  \"verdict\": \"FAIL\",\n"
             "  \"reason\": \"ref is NULL\"\n}\n");
        write_json_file("6",
             "{\n  \"test_id\": \"T3.6\",\n  \"verdict\": \"FAIL\",\n"
             "  \"reason\": \"ref is NULL\"\n}\n");
        return;
    }

    // 1000 random shuffles of the pointer array.  Each invocation must
    // return a pointer whose byte-content equals ref.
    bool pass = true;
    int  mismatches = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        std::shuffle(ptrs.begin(), ptrs.end(), rng);
        const sp_kste_tree *got = sp_kste_select_canonical(ptrs.data(), N);
        if (!got || std::memcmp(got, ref, sizeof(sp_kste_tree)) != 0) {
            ++mismatches;
            pass = false;
        }
    }

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"test_id\": \"T3.6\", \"phase\": 7,\n"
        "  \"name\": \"choice operator canonicality\",\n"
        "  \"metrics\": {\n"
        "    \"n_trees\": %d, \"shuffles\": 1000, \"mismatches\": %d\n"
        "  },\n"
        "  \"verdict\": \"%s\"\n"
        "}\n",
        N, mismatches, pass ? "PASS" : "FAIL");
    emit("6", "choice operator canonicality", pass, buf);
    write_json_file("6", buf);
}

// =========================================================================
// Driver
// =========================================================================
int main()
{
    std::printf("Phase 7 — Ultraproduct attention tests\n");
    test_T3_1();
    test_T3_2();
    test_T3_6();

    // Summary file
    std::string sum = "{\n  \"phase\": 7,\n  \"tests\": [\n";
    for (size_t i = 0; i < g_reports.size(); ++i) {
        char line[256];
        std::snprintf(line, sizeof(line),
            "    {\"id\": \"T3.%s\", \"name\": \"%s\", \"verdict\": \"%s\"}%s\n",
            g_reports[i].id, g_reports[i].name,
            g_reports[i].pass ? "PASS" : "FAIL",
            (i + 1 < g_reports.size()) ? "," : "");
        sum += line;
    }
    sum += "  ]\n}\n";
    write_json_file("SUMMARY", sum);

    std::printf("\nPhase 7 summary: %zu tests, %d failures\n",
                g_reports.size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
