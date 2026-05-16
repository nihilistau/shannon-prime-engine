// Shannon-Prime Engine — unit tests for sp_ok_arith + sp_frobenius.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Bit-exact mirror of the Python oracle at:
//   test-suite/src/sp_algebra.py
//   test-suite/src/engine_hooks2.py
//
// To run cross-validation against the Python golden file:
//   python3 ../../test-suite/scripts/make_golden.py > sato_tate_golden.json
//   ./test_sp_frobenius
//
// The test reads sato_tate_golden.json (when present) and asserts that
// the C engine's sp_sato_tate_mix output matches byte-for-byte.

#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_frobenius.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;

static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

#define ASSERT_OK_EQ(x, y) do { \
    sp_ok_t _x = (x); sp_ok_t _y = (y); \
    if (_x.a != _y.a || _x.b != _y.b) { \
        std::fprintf(stderr, "  ASSERT_OK_EQ FAIL (%s:%d): (%lld,%lld) != (%lld,%lld)\n", \
            __FILE__, __LINE__, (long long)_x.a, (long long)_x.b, (long long)_y.a, (long long)_y.b); \
        g_fail++; \
    } } while (0)

// =========================================================================
// O_K arithmetic
// =========================================================================

TEST(omega_squared_equals_omega_minus_41) {
    sp_ok_t w = SP_OK_OMEGA;
    sp_ok_t w2 = sp_ok_mul(w, w);
    sp_ok_t expected = sp_ok_sub(w, sp_ok_t{41, 0});
    ASSERT_OK_EQ(w2, expected);
}

TEST(omega_norm_is_41) {
    ASSERT(sp_ok_norm(SP_OK_OMEGA) == 41);
}

TEST(omega_times_omega_bar_is_norm) {
    sp_ok_t wbar = sp_ok_conjugate(SP_OK_OMEGA);
    sp_ok_t prod = sp_ok_mul(SP_OK_OMEGA, wbar);
    ASSERT_OK_EQ(prod, (sp_ok_t{41, 0}));
}

TEST(commutativity_random) {
    std::mt19937_64 rng(0xC0FFEE);
    for (int i = 0; i < 100; i++) {
        sp_ok_t x = { (int64_t)(rng() % 1000) - 500, (int64_t)(rng() % 1000) - 500 };
        sp_ok_t y = { (int64_t)(rng() % 1000) - 500, (int64_t)(rng() % 1000) - 500 };
        sp_ok_t xy = sp_ok_mul(x, y);
        sp_ok_t yx = sp_ok_mul(y, x);
        ASSERT_OK_EQ(xy, yx);
    }
}

// =========================================================================
// Prime classification
// =========================================================================

TEST(p2_is_inert) {
    ASSERT(sp_is_inert(2));
    ASSERT(!sp_is_split(2));
}

TEST(p11_is_inert_paper_D_fix) {
    // Paper D v0.2 wrongly claimed p=11 is split. Test suite caught it.
    ASSERT(sp_is_inert(11));
    ASSERT(!sp_is_split(11));
}

TEST(p41_is_split) {
    ASSERT(sp_is_split(41));
    ASSERT(!sp_is_inert(41));
}

TEST(p43_is_split_eulers_polynomial) {
    // n^2 + n + 41 at n=1
    ASSERT(sp_is_split(43));
}

TEST(p163_is_ramified) {
    ASSERT(sp_is_ramified(163));
    ASSERT(!sp_is_inert(163));
    ASSERT(!sp_is_split(163));
}

// =========================================================================
// Frobenius
// =========================================================================

TEST(phi_2_squared_equals_minus_2) {
    sp_ok_t s = { 7, 11 };
    sp_ok_t out = sp_apply_frobenius(s, 2, 2);
    sp_ok_t expected = sp_ok_scalar_mul(s, -2);
    ASSERT_OK_EQ(out, expected);
}

TEST(find_element_of_norm_41) {
    sp_ok_t pi;
    bool ok = sp_find_element_of_norm(41, &pi);
    ASSERT(ok);
    ASSERT(sp_ok_norm(pi) == 41);
}

TEST(frobenius_split_norm_invariant) {
    // The bit-exact contract: N(phi_p^k(state)) = N(state) * p^k,
    // regardless of which representative (pi or pi_bar) is chosen.
    sp_ok_t state = { 123, -45 };
    int64_t k = 8;
    sp_ok_t out = sp_apply_frobenius(state, 41, k);
    int64_t expected_norm = sp_ok_norm(state);
    for (int i = 0; i < k; i++) expected_norm *= 41;
    ASSERT(sp_ok_norm(out) == expected_norm);
}

TEST(sato_tate_commutativity) {
    sp_ok_t state = { 7, 11 };
    sp_ok_t ab = sp_apply_frobenius(sp_apply_frobenius(state, 2, 2), 41, 3);
    sp_ok_t ba = sp_apply_frobenius(sp_apply_frobenius(state, 41, 3), 2, 2);
    ASSERT_OK_EQ(ab, ba);
}

TEST(sato_tate_random_commutativity) {
    std::mt19937_64 rng(2026);
    for (int i = 0; i < 50; i++) {
        sp_ok_t s = { (int64_t)(rng() % 201) - 100, (int64_t)(rng() % 201) - 100 };
        sp_ok_t ab = sp_apply_frobenius(sp_apply_frobenius(s, 2, 2), 41, 3);
        sp_ok_t ba = sp_apply_frobenius(sp_apply_frobenius(s, 41, 3), 2, 2);
        ASSERT_OK_EQ(ab, ba);
    }
}

// =========================================================================
// Tensor-level
// =========================================================================

TEST(sp_frobenius_quant_tensor_norm) {
    const size_t N = 64;
    std::vector<sp_ok_t> state(N);
    std::mt19937_64 rng(0xDEADBEEF);
    std::vector<int64_t> initial_norms(N);
    for (size_t i = 0; i < N; i++) {
        state[i] = { (int64_t)(rng() % 41) - 20, (int64_t)(rng() % 41) - 20 };
        initial_norms[i] = sp_ok_norm(state[i]);
    }
    sp_frobenius_quant_tensor(state.data(), N, 41, 8);
    int64_t p8 = 1;
    for (int i = 0; i < 8; i++) p8 *= 41;
    for (size_t i = 0; i < N; i++) {
        ASSERT(sp_ok_norm(state[i]) == initial_norms[i] * p8);
    }
}

TEST(sp_sato_tate_mix_tensor_norm) {
    const size_t N = 32;
    std::vector<sp_ok_t> state(N);
    std::mt19937_64 rng(0xCAFE);
    std::vector<int64_t> initial_norms(N);
    for (size_t i = 0; i < N; i++) {
        state[i] = { (int64_t)(rng() % 21) - 10, (int64_t)(rng() % 21) - 10 };
        initial_norms[i] = sp_ok_norm(state[i]);
    }
    sp_sato_tate_mix_tensor(state.data(), N, 2, 2, 41, 4);
    int64_t expected_factor = 4;  // 2^2 from inert channel
    for (int i = 0; i < 4; i++) expected_factor *= 41;
    for (size_t i = 0; i < N; i++) {
        ASSERT(sp_ok_norm(state[i]) == initial_norms[i] * expected_factor);
    }
}

// =========================================================================
// Cross-validation against Python golden file (if present).
// =========================================================================

static bool parse_golden(const std::string &path, std::vector<std::pair<sp_ok_t, sp_ok_t>> &out) {
    // Minimal JSON parser for our specific golden file format.
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    size_t pos = 0;
    while (true) {
        size_t in_pos = s.find("\"in\"", pos);
        if (in_pos == std::string::npos) break;
        size_t lb1 = s.find('[', in_pos);
        size_t rb1 = s.find(']', lb1);
        std::string in_arr = s.substr(lb1 + 1, rb1 - lb1 - 1);
        long long a_in = 0, b_in = 0;
        std::sscanf(in_arr.c_str(), " %lld , %lld", &a_in, &b_in);

        size_t out_pos = s.find("\"out\"", rb1);
        size_t lb2 = s.find('[', out_pos);
        size_t rb2 = s.find(']', lb2);
        std::string out_arr = s.substr(lb2 + 1, rb2 - lb2 - 1);
        long long a_out = 0, b_out = 0;
        std::sscanf(out_arr.c_str(), " %lld , %lld", &a_out, &b_out);

        out.push_back({{(int64_t)a_in, (int64_t)b_in}, {(int64_t)a_out, (int64_t)b_out}});
        pos = rb2 + 1;
    }
    return !out.empty();
}

TEST(cross_validate_python_oracle_optional) {
    std::vector<std::pair<sp_ok_t, sp_ok_t>> golden;
    if (!parse_golden("sato_tate_golden.json", &golden[0] ? golden : golden)) {
        std::printf("  (skipped — sato_tate_golden.json not present)\n");
        return;
    }
    int mismatches = 0;
    for (auto &kv : golden) {
        sp_ok_t s = kv.first;
        sp_ok_t expected = kv.second;
        sp_sato_tate_mix_tensor(&s, 1, 2, 2, 41, 8);
        if (s.a != expected.a || s.b != expected.b) {
            mismatches++;
            if (mismatches <= 3) {
                std::fprintf(stderr, "    mismatch: (%lld,%lld) -> got (%lld,%lld) expected (%lld,%lld)\n",
                    (long long)kv.first.a, (long long)kv.first.b,
                    (long long)s.a, (long long)s.b,
                    (long long)expected.a, (long long)expected.b);
            }
        }
    }
    ASSERT(mismatches == 0);
    std::printf("  cross-val: %zu states, %d mismatches\n", golden.size(), mismatches);
}

// =========================================================================
// Driver
// =========================================================================

int main(int argc, char **argv) {
    std::printf("Shannon-Prime Frobenius unit tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
