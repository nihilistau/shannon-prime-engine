/* Strike 5 parity test: HVX two-phase matmul kernel.
 *
 * Locks the math contract: sp_hex_matmul_ok_block_q8_inner produces
 * bit-identical int64 (acc_a, acc_b) to the engine's reference
 * sp_matmul_ok_block_inner_scalar from src/sp_matmul.cpp (the desktop
 * AVX-512 baseline). If this passes on host, the algebra of the
 * two-phase HVX formulation is correct, and the on-device HVX kernel
 * only needs to verify the intrinsic-level mechanics — the math
 * itself is locked.
 *
 * Both kernels read the same packed sp_ok_q8_block_t storage and the
 * same sp_ok_t activation row, so on-device the kernel will consume
 * exactly the bytes that the .sp_ok file's prefetcher loads into ION
 * pages. No format translation, no marshal copy.
 *
 * Coverage:
 *   1. Random packed Q8 blocks + random activation → bit-equal acc.
 *   2. A_ONLY mode (out_acc_b=NULL) matches the engine's A_ONLY=true path.
 *   3. Various block counts (1, 4, 16, 64) — kernel must scale.
 *   4. Edge: all-zero codepoints → zero accumulators.
 *   5. Edge: all-zero activations → zero accumulators.
 */

extern "C" {
#include "../lib/shannon-prime/backends/hexagon/sp_hex_matmul_block_q8.h"
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_ok_block_quant.h"
}

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TE { const char *name; void (*fn)(); };
static std::vector<TE> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

/* Engine reference, manually expanded from sp_matmul.cpp's
 * sp_matmul_ok_block_inner_scalar<IS_Q4=false, A_ONLY=template>.
 * Kept here so the test has no dependency on the engine's C++
 * template internals — we replicate the exact direct-form math. */
static void engine_reference_inner(
    const sp_ok_q8_block_t* w_blocks,
    const sp_ok_t*          x_row,
    size_t                  blocks_per_row,
    int64_t*                out_acc_a,
    int64_t*                out_acc_b)
{
    constexpr int64_t W41 = 41;
    int64_t acc_a = 0;
    int64_t acc_b = 0;
    const bool need_b = (out_acc_b != nullptr);

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const sp_ok_q8_block_t* blk = w_blocks + b;
        const int64_t B_a = blk->B_a;
        const int64_t B_b = blk->B_b;
        const sp_ok_t* x_tile = x_row + b * SP_OK_BLOCK_SIZE;

        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            const int64_t w_int = (int64_t)blk->packed[k];
            const sp_ok_t& x = x_tile[k];
            /* Direct form, matches engine reference exactly. */
            const int64_t F_a = B_a * x.a - W41 * B_b * x.b;
            acc_a += w_int * F_a;
            if (need_b) {
                const int64_t F_b = B_a * x.b + B_b * x.a + B_b * x.b;
                acc_b += w_int * F_b;
            }
        }
    }
    *out_acc_a = acc_a;
    if (need_b) *out_acc_b = acc_b;
}

/* Build synthetic packed blocks + activation row for testing.
 * Coordinates are bounded so the engine's overflow invariants hold. */
static void build_random_inputs(
    size_t blocks_per_row,
    std::mt19937_64& rng,
    std::vector<sp_ok_q8_block_t>& w_blocks,
    std::vector<sp_ok_t>& x_row)
{
    w_blocks.resize(blocks_per_row);
    x_row.resize(blocks_per_row * SP_OK_BLOCK_SIZE);

    /* Frobenius (B_a, B_b) typical range from sp_ok_pack output:
     * production scale_recip=16384 × π² × fp16_scale ~ ±1e4. */
    std::uniform_int_distribution<int64_t> B_dist(-20000, 20000);
    /* Activation sp_ok_t coords bounded by engine's Q-format. */
    std::uniform_int_distribution<int64_t> X_dist(-1 << 18, 1 << 18);
    /* int8 codepoints clamped to engine's [-127, +127] range. */
    std::uniform_int_distribution<int>     w_dist(-127, 127);

    for (auto& blk : w_blocks) {
        blk.B_a = B_dist(rng);
        blk.B_b = B_dist(rng);
        blk.reserved_block_min_a = 0;
        blk.reserved_block_min_b = 0;
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            blk.packed[k] = (int8_t)w_dist(rng);
        }
    }
    for (auto& x : x_row) {
        x.a = X_dist(rng);
        x.b = X_dist(rng);
    }
}

/* Test 1 — random inputs, both acc_a and acc_b paths, bit-equal. */
TEST(parity_random_inputs_full) {
    std::mt19937_64 rng(0xABCD1234ULL);
    for (size_t blocks : { (size_t)1, (size_t)4, (size_t)16, (size_t)64 }) {
        std::vector<sp_ok_q8_block_t> W;
        std::vector<sp_ok_t> X;
        build_random_inputs(blocks, rng, W, X);

        int64_t hex_a = 0, hex_b = 0;
        int64_t ref_a = 0, ref_b = 0;
        sp_hex_matmul_ok_block_q8_inner(W.data(), X.data(), blocks, &hex_a, &hex_b);
        engine_reference_inner          (W.data(), X.data(), blocks, &ref_a, &ref_b);

        if (hex_a != ref_a || hex_b != ref_b) {
            std::fprintf(stderr, "  blocks=%zu hex=(%lld,%lld) ref=(%lld,%lld)\n",
                         blocks, (long long)hex_a, (long long)hex_b,
                                 (long long)ref_a, (long long)ref_b);
        }
        ASSERT(hex_a == ref_a);
        ASSERT(hex_b == ref_b);
    }
    std::fprintf(stderr, "  [info] hvx_active=%d\n",
                 sp_hex_matmul_block_q8_uses_hvx());
}

/* Test 2 — A_ONLY mode (out_acc_b=NULL). Matches engine's A_ONLY=true
 * specialization used by sp_matmul_ok_block_q8_to_fp32. */
TEST(parity_a_only_mode) {
    std::mt19937_64 rng(0xC0FFEE99ULL);
    for (size_t blocks : { (size_t)1, (size_t)8, (size_t)32 }) {
        std::vector<sp_ok_q8_block_t> W;
        std::vector<sp_ok_t> X;
        build_random_inputs(blocks, rng, W, X);

        int64_t hex_a = 0, ref_a = 0;
        sp_hex_matmul_ok_block_q8_inner(W.data(), X.data(), blocks, &hex_a, nullptr);
        engine_reference_inner          (W.data(), X.data(), blocks, &ref_a, nullptr);
        ASSERT(hex_a == ref_a);
    }
}

/* Test 3 — zero codepoints. acc_a and acc_b must be zero. */
TEST(zero_codepoints) {
    std::vector<sp_ok_q8_block_t> W(4);
    std::vector<sp_ok_t> X(4 * SP_OK_BLOCK_SIZE);
    /* Random Frobenius and activations, but all codepoints zero. */
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int64_t> Bd(-1000, 1000);
    std::uniform_int_distribution<int64_t> Xd(-1000, 1000);
    for (auto& blk : W) {
        blk.B_a = Bd(rng); blk.B_b = Bd(rng);
        blk.reserved_block_min_a = 0; blk.reserved_block_min_b = 0;
        std::memset(blk.packed, 0, SP_OK_BLOCK_SIZE);
    }
    for (auto& x : X) { x.a = Xd(rng); x.b = Xd(rng); }

    int64_t a = 99, b = 99;
    sp_hex_matmul_ok_block_q8_inner(W.data(), X.data(), W.size(), &a, &b);
    ASSERT(a == 0);
    ASSERT(b == 0);
}

/* Test 4 — zero activations. acc_a and acc_b must be zero. */
TEST(zero_activations) {
    std::vector<sp_ok_q8_block_t> W(4);
    std::vector<sp_ok_t> X(4 * SP_OK_BLOCK_SIZE);
    std::mt19937_64 rng(13);
    std::uniform_int_distribution<int64_t> Bd(-1000, 1000);
    std::uniform_int_distribution<int>     wd(-127, 127);
    for (auto& blk : W) {
        blk.B_a = Bd(rng); blk.B_b = Bd(rng);
        blk.reserved_block_min_a = 0; blk.reserved_block_min_b = 0;
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) blk.packed[k] = (int8_t)wd(rng);
    }
    /* X already default-zero from sp_ok_t{0,0}. */
    for (auto& x : X) { x.a = 0; x.b = 0; }

    int64_t a = 99, b = 99;
    sp_hex_matmul_ok_block_q8_inner(W.data(), X.data(), W.size(), &a, &b);
    ASSERT(a == 0);
    ASSERT(b == 0);
}

/* Test 5 — known-value check. Build one block with (B_a=1, B_b=0)
 * and codepoints all = 1, activation all (a=1, b=0). The HVX kernel
 * should return acc_a = 32 * (1 * 1 - 41 * 0 * 0) = 32, acc_b = 0. */
TEST(known_value_simple) {
    std::vector<sp_ok_q8_block_t> W(1);
    std::vector<sp_ok_t>          X(SP_OK_BLOCK_SIZE);
    W[0].B_a = 1; W[0].B_b = 0;
    W[0].reserved_block_min_a = 0; W[0].reserved_block_min_b = 0;
    for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) W[0].packed[k] = 1;
    for (auto& x : X) { x.a = 1; x.b = 0; }

    int64_t a = -1, b = -1;
    sp_hex_matmul_ok_block_q8_inner(W.data(), X.data(), 1, &a, &b);
    ASSERT(a == 32);
    ASSERT(b == 0);
}

int main() {
    std::fprintf(stderr, "test_sp_hex_matmul_block_q8: %zu tests\n", g_tests.size());
    for (auto& t : g_tests) {
        std::fprintf(stderr, "  %s ...\n", t.name);
        t.fn();
    }
    if (g_fail) {
        std::fprintf(stderr, "FAILED %d assertions\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "all tests passed\n");
    return 0;
}
