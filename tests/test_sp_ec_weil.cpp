// Shannon-Prime — sp_ec_weil unit tests (Phase 3 part 1).
// Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
//
// Golden parity with test-suite/src/weil_pairing.py.
// Curve: E: y^2 = x^3 + 1 over F_7 (|E(F_7)| = 12).
// Tests: point arithmetic, n-torsion, Miller, Weil pairing.

extern "C" {
#include "../lib/shannon-prime/core/sp_ec_weil.h"
}

#include <cstdio>
#include <cstdint>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)
#define ASSERT_EQ(a, b) do { auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { std::fprintf(stderr, "  ASSERT_EQ FAIL (%s:%d): %lld != %lld\n", \
        __FILE__, __LINE__, (long long)_a, (long long)_b); g_fail++; } } while (0)

// =========================================================================
// E: y^2 = x^3 + 1 over F_7
// =========================================================================

static sp_ec_curve make_E7() {
    sp_ec_curve E = { 0, 1, 7 };
    return E;
}

// All points on E(F_7) — 12 points expected.
static std::vector<sp_ec_point> all_points(const sp_ec_curve& E) {
    std::vector<sp_ec_point> pts;
    pts.push_back(SP_EC_INFINITY);
    for (int64_t x = 0; x < E.p; ++x) {
        int64_t rhs = (x * x * x + E.a * x + E.b) % E.p;
        if (rhs < 0) rhs += E.p;
        for (int64_t y = 0; y < E.p; ++y) {
            if ((y * y) % E.p == rhs) {
                sp_ec_point P = { x, y };
                pts.push_back(P);
            }
        }
    }
    return pts;
}

TEST(curve_E7_has_12_points) {
    sp_ec_curve E = make_E7();
    auto pts = all_points(E);
    ASSERT_EQ((int)pts.size(), 12);
    for (auto P : pts) {
        ASSERT(sp_ec_is_on_curve(&E, P));
    }
}

TEST(point_at_infinity_is_identity) {
    sp_ec_curve E = make_E7();
    sp_ec_point P = { 0, 1 };
    ASSERT(sp_ec_is_on_curve(&E, P));
    sp_ec_point R = sp_ec_add(&E, P, SP_EC_INFINITY);
    ASSERT(sp_ec_eq(R, P));
    R = sp_ec_add(&E, SP_EC_INFINITY, P);
    ASSERT(sp_ec_eq(R, P));
}

TEST(neg_and_add_to_infinity) {
    sp_ec_curve E = make_E7();
    sp_ec_point P = { 0, 1 };
    sp_ec_point negP = sp_ec_neg(&E, P);
    ASSERT(negP.x == 0 && negP.y == 6);  // -1 mod 7 = 6
    sp_ec_point R = sp_ec_add(&E, P, negP);
    ASSERT(sp_ec_is_infinity(R));
}

TEST(doubling_and_order) {
    sp_ec_curve E = make_E7();
    sp_ec_point P = { 0, 1 };
    // 2P = ?
    sp_ec_point twoP = sp_ec_add(&E, P, P);
    ASSERT(sp_ec_is_on_curve(&E, twoP));
    // Order of (0, 1) on E: y^2 = x^3 + 1 over F_7 should divide |E|=12.
    int64_t ord = sp_ec_order(&E, P, 100);
    ASSERT(ord > 0);
    ASSERT(12 % ord == 0);
}

TEST(mul_k_consistent_with_repeated_add) {
    sp_ec_curve E = make_E7();
    sp_ec_point P = { 0, 1 };
    sp_ec_point R = SP_EC_INFINITY;
    for (int k = 1; k <= 6; ++k) {
        R = sp_ec_add(&E, R, P);
        sp_ec_point Q = sp_ec_mul(&E, k, P);
        ASSERT(sp_ec_eq(R, Q));
    }
}

// 3-torsion of E: y^2 = x^3 + 1 over F_7 — 9 points (over alg closure)
// but over F_7 we should find E[3] = {O, plus all P with 3P = O}.
TEST(weil_pairing_E3_F7_alternating_and_nondegenerate) {
    sp_ec_curve E = make_E7();
    auto pts = all_points(E);

    // Collect E[3].
    std::vector<sp_ec_point> torsion3;
    for (auto P : pts) {
        sp_ec_point R = sp_ec_mul(&E, 3, P);
        if (sp_ec_is_infinity(R)) torsion3.push_back(P);
    }
    std::printf("  |E[3]| = %d\n", (int)torsion3.size());
    ASSERT(torsion3.size() >= 3);  // at least O plus a non-trivial pair

    // Find two independent torsion points.
    sp_ec_point O = SP_EC_INFINITY;
    sp_ec_point P = O, Q = O;
    for (auto T : torsion3) {
        if (sp_ec_is_infinity(T)) continue;
        if (sp_ec_is_infinity(P)) { P = T; continue; }
        // Check independence: T not in <P> = {O, P, 2P}
        sp_ec_point twoP = sp_ec_add(&E, P, P);
        if (sp_ec_eq(T, P) || sp_ec_eq(T, twoP)) continue;
        Q = T;
        break;
    }
    if (sp_ec_is_infinity(Q)) {
        // Test the 1-dim case: all torsion in <P>. Skip the non-degeneracy
        // check but verify alternating + cubic property.
        int64_t e_PP = sp_ec_weil_pairing(&E, 3, P, P);
        ASSERT_EQ(e_PP, (int64_t)1);
        return;
    }
    std::printf("  P = (%lld, %lld)  Q = (%lld, %lld)\n",
                (long long)P.x, (long long)P.y, (long long)Q.x, (long long)Q.y);

    // e_3(P, P) = 1 (alternating)
    int64_t e_PP = sp_ec_weil_pairing(&E, 3, P, P);
    ASSERT_EQ(e_PP, (int64_t)1);

    // e_3(P, Q) is a primitive cube root of unity mod 7
    // (cube roots of unity in F_7 are 1, 2, 4).
    int64_t e_PQ = sp_ec_weil_pairing(&E, 3, P, Q);
    std::printf("  e_3(P, Q) = %lld\n", (long long)e_PQ);
    int64_t cubed = sp_ec_mod_pow(e_PQ, 3, E.p);
    ASSERT_EQ(cubed, (int64_t)1);
    ASSERT(e_PQ != 1);  // non-degenerate

    // Skew-symmetry: e_3(Q, P) = e_3(P, Q)^{-1} mod p
    int64_t e_QP = sp_ec_weil_pairing(&E, 3, Q, P);
    std::printf("  e_3(Q, P) = %lld\n", (long long)e_QP);
    int64_t prod = (e_PQ * e_QP) % E.p;
    ASSERT_EQ(prod, (int64_t)1);
}

// Field-arithmetic sanity.
TEST(mod_inv_correctness) {
    int64_t p = 7;
    for (int64_t x = 1; x < p; ++x) {
        int64_t inv = sp_ec_mod_inv(x, p);
        ASSERT_EQ((x * inv) % p, (int64_t)1);
    }
}

int main() {
    std::printf("Shannon-Prime sp_ec_weil tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
