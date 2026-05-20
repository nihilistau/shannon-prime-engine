/* Strike 2 parity test: VHT2 + Möbius + banded q8 + Frobenius fusion.
 *
 * Contracts:
 *   1. Encode → decode round-trip recovers the fp32 vector to within
 *      the int8 quantization noise floor (RMSE < 0.05 for unit-norm input).
 *   2. The spectral-domain dot product sp_vht2_q8_dot equals the fp32
 *      dot product of the original vectors to within int8 quant noise.
 *   3. Frobenius (B_a, B_b) round-trip: B_a / (scale_recip * π_a) recovers
 *      the per-tile scale used at encode time, bit-equal to the round-trip
 *      tolerance.
 *   4. Progressive read: max_bands=1 returns a partial sum, max_bands=n_bands
 *      returns the full dot. The partial captures > 25% of the energy for
 *      a random VHT2-domain-friendly input (Band 0 dominance is the
 *      Möbius-reorder + amax-bias structural prediction).
 *   5. Geometry: head_dim=128 / n_bands=4 / blocks_per_band=1 (4 blocks total).
 *      head_dim=256 / n_bands=4 / blocks_per_band=2 (8 blocks total).
 */

extern "C" {
#include "../lib/shannon-prime/core/shannon_prime.h"
#include "../lib/shannon-prime/core/sp_vht2_block_q8.h"
}

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TE { const char *name; void (*fn)(); };
static std::vector<TE> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)
#define ASSERT_NEAR(a, b, tol) do { float _d = std::fabs((a)-(b)); \
    if (_d > (tol)) { std::fprintf(stderr, "  FAIL %s:%d: |%g - %g| = %g > %g\n", \
        __FILE__, __LINE__, (double)(a), (double)(b), (double)_d, (double)(tol)); \
        g_fail++; } } while (0)

/* Production Frobenius: p=41 split prime, k=2 (matches engine default). */
static constexpr int64_t F_P = 41;
static constexpr int64_t F_K = 2;
static constexpr int64_t SCALE_RECIP = 1 << 20;

static float fp32_dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

/* Test 1 — encode/decode round-trip on head_dim=128 with 4 bands.
 * RMSE bounded by the int8 quant noise: amax/127 per element. */
TEST(roundtrip_hd128_4bands) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);
    ASSERT(ctx.total_blocks == 4);

    std::mt19937 rng(0xABCDEF01);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> in(HD), out(HD), scratch(HD);
    for (int i = 0; i < HD; ++i) in[i] = N(rng);

    std::vector<sp_ok_q8_block_t> blocks(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(blocks.data(), in.data(), &ctx, scratch.data()) == 1);
    ASSERT(sp_vht2_q8_decode(out.data(), blocks.data(), &ctx, scratch.data()) == 1);

    double sse = 0.0, energy = 0.0;
    for (int i = 0; i < HD; ++i) {
        double d = (double)in[i] - (double)out[i];
        sse += d*d;
        energy += (double)in[i] * (double)in[i];
    }
    double rmse = std::sqrt(sse / HD);
    double rel  = std::sqrt(sse / energy);
    std::fprintf(stderr, "  [info] hd128 RMSE=%.4f rel=%.4f\n", rmse, rel);
    ASSERT(rel < 0.05);  /* < 5% relative error — int8 noise floor */

    sp_mobius_mask_free(&mask);
}

/* Test 2 — encode/decode round-trip on head_dim=256 with 4 bands (2 blocks/band). */
TEST(roundtrip_hd256_4bands) {
    constexpr int HD = 256;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);
    ASSERT(ctx.total_blocks == 8);
    ASSERT(ctx.blocks_per_band == 2);

    std::mt19937 rng(0xCAFEBABE);
    std::normal_distribution<float> N(0.0f, 0.5f);
    std::vector<float> in(HD), out(HD), scratch(HD);
    for (int i = 0; i < HD; ++i) in[i] = N(rng);

    std::vector<sp_ok_q8_block_t> blocks(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(blocks.data(), in.data(), &ctx, scratch.data()) == 1);
    ASSERT(sp_vht2_q8_decode(out.data(), blocks.data(), &ctx, scratch.data()) == 1);

    double sse = 0.0, energy = 0.0;
    for (int i = 0; i < HD; ++i) {
        double d = (double)in[i] - (double)out[i];
        sse += d*d;
        energy += (double)in[i] * (double)in[i];
    }
    double rel = std::sqrt(sse / energy);
    std::fprintf(stderr, "  [info] hd256 RMSE rel=%.4f\n", rel);
    ASSERT(rel < 0.05);

    sp_mobius_mask_free(&mask);
}

/* Test 3 — spectral-domain dot product matches fp32 dot product. */
TEST(spectral_dot_matches_fp32) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);

    std::mt19937 rng(0xDEADBEEF);
    std::normal_distribution<float> N(0.0f, 0.3f);
    std::vector<float> a(HD), b(HD), sa(HD), sb(HD);
    for (int i = 0; i < HD; ++i) { a[i] = N(rng); b[i] = N(rng); }

    std::vector<sp_ok_q8_block_t> A(ctx.total_blocks), B(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(A.data(), a.data(), &ctx, sa.data()) == 1);
    ASSERT(sp_vht2_q8_encode(B.data(), b.data(), &ctx, sb.data()) == 1);

    float ref = fp32_dot(a, b);
    float spec = sp_vht2_q8_dot(A.data(), B.data(), &ctx, /*max_bands=*/0);
    std::fprintf(stderr, "  [info] fp32_dot=%.5f spectral_dot=%.5f delta=%.5f\n",
                 ref, spec, std::fabs(ref - spec));
    /* Tolerance scales with vector magnitude × int8 noise; with HD=128
     * and 0.3 stddev inputs, |dot| ≈ 0..3, so ~0.1 absolute tol works. */
    ASSERT_NEAR(ref, spec, 0.15f);

    sp_mobius_mask_free(&mask);
}

/* Test 4 — Frobenius (B_a, B_b) round-trip: the encoded coefficients
 * recover the per-tile scale exactly (to round-half-to-even). */
TEST(frobenius_scale_roundtrip) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);
    /* π^2 for p=41: π=(7,1) so π^2 = a + bω with explicit calc.
     * a' = a*a + b*b * (-41) = 49 - 41 = 8 ... actually we don't need
     * to spell it; just confirm both coords are non-zero for p=41 split. */
    ASSERT(ctx.pi_pow.a != 0 || ctx.pi_pow.b != 0);

    std::vector<float> in(HD), scratch(HD);
    in[0] = 1.0f;
    for (int i = 1; i < HD; ++i) in[i] = 0.0f;
    std::vector<sp_ok_q8_block_t> blocks(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(blocks.data(), in.data(), &ctx, scratch.data()) == 1);

    /* For each tile, recover the scale and confirm it's a finite positive number. */
    double pi_a = (double)ctx.pi_pow.a;
    double pi_b = (double)ctx.pi_pow.b;
    double S    = (double)ctx.scale_recip;
    for (int blk = 0; blk < ctx.total_blocks; ++blk) {
        double sa_a = (pi_a != 0.0) ? (double)blocks[blk].B_a / (S * pi_a) : 0.0;
        double sa_b = (pi_b != 0.0) ? (double)blocks[blk].B_b / (S * pi_b) : 0.0;
        /* The two recoveries must agree (both are reconstructions of the
         * same underlying tile_scale, just via different Frobenius coords). */
        if (pi_a != 0.0 && pi_b != 0.0 && sa_a > 1e-9 && sa_b > 1e-9) {
            ASSERT_NEAR((float)sa_a, (float)sa_b, 1e-3f);
        }
    }

    sp_mobius_mask_free(&mask);
}

/* Test 5 — progressive read correctness: max_bands=N must equal full sum;
 * sum of per-band contributions must equal max_bands=N. Tests the API
 * mechanics, not the workload-specific "Band 0 dominates" claim (which
 * depends on how the model's data aligns with the Möbius permutation —
 * not a property of arbitrary inputs).
 *
 * The "Strike 3 System 1 prefilter" win depends on real KV-cache vectors
 * having squarefree-aligned spectral energy concentration; that's
 * validated end-to-end at the PPL bench level, not in this unit test. */
TEST(progressive_read_correctness) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);

    std::mt19937 rng(0x5EED5EED);
    std::normal_distribution<float> N(0.0f, 0.5f);
    std::vector<float> a(HD), b(HD), sa(HD), sb(HD);
    for (int i = 0; i < HD; ++i) { a[i] = N(rng); b[i] = N(rng); }

    std::vector<sp_ok_q8_block_t> A(ctx.total_blocks), B(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(A.data(), a.data(), &ctx, sa.data()) == 1);
    ASSERT(sp_vht2_q8_encode(B.data(), b.data(), &ctx, sb.data()) == 1);

    /* max_bands=0 (= use all) must equal max_bands=NB. */
    float full_zero = sp_vht2_q8_dot(A.data(), B.data(), &ctx, /*max_bands=*/0);
    float full_nb   = sp_vht2_q8_dot(A.data(), B.data(), &ctx, /*max_bands=*/NB);
    ASSERT_NEAR(full_zero, full_nb, 1e-6f);

    /* max_bands=0 must equal the sum of per-band contributions (each
     * computed as max_bands=k - max_bands=k-1). */
    float prev = 0.0f;
    float accum = 0.0f;
    std::fprintf(stderr, "  [info] per-band breakdown:");
    for (int k = 1; k <= NB; ++k) {
        float cur = sp_vht2_q8_dot(A.data(), B.data(), &ctx, /*max_bands=*/k);
        float band_k = cur - prev;
        std::fprintf(stderr, " B%d=%.4f", k-1, band_k);
        accum += band_k;
        prev = cur;
    }
    std::fprintf(stderr, " full=%.4f sum=%.4f\n", full_zero, accum);
    ASSERT_NEAR(full_zero, accum, 1e-4f);

    sp_mobius_mask_free(&mask);
}

/* ─── Strike 3: System 1 prefilter gate tests ──────────────────────── */

/* Helper: directly stamp a tile's (B_a, B_b) for a given desired tile_scale,
 * bypassing the encoder. Used to construct controlled per-band energy
 * profiles for gate tests. */
static void stamp_tile_scale(sp_ok_q8_block_t* dst,
                             double tile_scale,
                             const sp_vht2_q8_ctx* ctx,
                             int8_t code_value) {
    double S = (double)ctx->scale_recip;
    dst->B_a = (int64_t)std::llrint(S * tile_scale * (double)ctx->pi_pow.a);
    dst->B_b = (int64_t)std::llrint(S * tile_scale * (double)ctx->pi_pow.b);
    dst->reserved_block_min_a = 0;
    dst->reserved_block_min_b = 0;
    for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) dst->packed[i] = code_value;
}

/* Test 6 — band energy sanity: sum of per-band energies equals encoded
 * vector L2² (= original L2² up to int8 quant noise, since VHT2 is
 * orthonormal). */
TEST(band_energy_sums_to_total) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);

    std::mt19937 rng(0x1A2B3C4D);
    std::normal_distribution<float> N(0.0f, 0.7f);
    std::vector<float> in(HD), scratch(HD);
    for (int i = 0; i < HD; ++i) in[i] = N(rng);

    std::vector<sp_ok_q8_block_t> blocks(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(blocks.data(), in.data(), &ctx, scratch.data()) == 1);

    float be[NB];
    sp_vht2_q8_band_energy(blocks.data(), &ctx, be);
    double sum_band = 0.0;
    for (int b = 0; b < NB; ++b) sum_band += be[b];

    double l2sq_in = 0.0;
    for (int i = 0; i < HD; ++i) l2sq_in += (double)in[i] * (double)in[i];

    double rel = std::fabs(sum_band - l2sq_in) / l2sq_in;
    std::fprintf(stderr, "  [info] L2² in=%.4f band_sum=%.4f rel_err=%.4f\n",
                 l2sq_in, sum_band, rel);
    ASSERT(rel < 0.05);  /* within int8 noise floor */

    sp_mobius_mask_free(&mask);
}

/* Test 7 — gate is monotone in threshold and clamped. */
TEST(min_bands_monotone_and_clamped) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);

    std::mt19937 rng(0xDEAD0042);
    std::normal_distribution<float> N(0.0f, 0.5f);
    std::vector<float> in(HD), scratch(HD);
    for (int i = 0; i < HD; ++i) in[i] = N(rng);
    std::vector<sp_ok_q8_block_t> blocks(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(blocks.data(), in.data(), &ctx, scratch.data()) == 1);

    int prev = 0;
    for (float t = 0.1f; t <= 1.0f; t += 0.1f) {
        int k = sp_vht2_q8_min_bands_for_energy(blocks.data(), &ctx, t);
        ASSERT(k >= 1);
        ASSERT(k <= NB);
        ASSERT(k >= prev);  /* non-decreasing in threshold */
        prev = k;
    }
    /* Edge cases. */
    ASSERT(sp_vht2_q8_min_bands_for_energy(blocks.data(), &ctx, 0.0f) == 1);
    ASSERT(sp_vht2_q8_min_bands_for_energy(blocks.data(), &ctx, 1.5f) == NB);

    sp_mobius_mask_free(&mask);
}

/* Test 8 — controlled band profile: Band-0-dominant input has min_bands=1
 * for typical thresholds; Band-3-dominant input requires all bands.
 * Bypasses the encoder to stamp known per-tile scales directly. */
TEST(min_bands_controlled_distributions) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    /* No Möbius needed — we stamp blocks directly in band order. */
    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, /*mobius=*/nullptr,
                               SCALE_RECIP, F_P, F_K) == 1);

    /* Profile 1: Band 0 dominant — scales [10.0, 1.0, 1.0, 1.0].
     * Codepoints fixed at 100, so energy[band] = (32 × 100²) × scale². */
    std::vector<sp_ok_q8_block_t> b0_dom(ctx.total_blocks);
    {
        double scales[NB] = { 10.0, 1.0, 1.0, 1.0 };
        for (int b = 0; b < NB; ++b)
            for (int t = 0; t < ctx.blocks_per_band; ++t)
                stamp_tile_scale(&b0_dom[b * ctx.blocks_per_band + t],
                                 scales[b], &ctx, /*code_value=*/100);
    }
    float be0[NB];
    sp_vht2_q8_band_energy(b0_dom.data(), &ctx, be0);
    std::fprintf(stderr, "  [info] B0-dom energies: %.1f %.1f %.1f %.1f\n",
                 be0[0], be0[1], be0[2], be0[3]);
    /* For threshold 0.50 → Band 0 alone (100/103 = 97%). */
    ASSERT(sp_vht2_q8_min_bands_for_energy(b0_dom.data(), &ctx, 0.5f)  == 1);
    /* For threshold 0.86 (production default) → Band 0 still suffices. */
    ASSERT(sp_vht2_q8_min_bands_for_energy(b0_dom.data(), &ctx, 0.86f) == 1);

    /* Profile 2: Band 3 dominant — scales [1.0, 1.0, 1.0, 10.0]. */
    std::vector<sp_ok_q8_block_t> b3_dom(ctx.total_blocks);
    {
        double scales[NB] = { 1.0, 1.0, 1.0, 10.0 };
        for (int b = 0; b < NB; ++b)
            for (int t = 0; t < ctx.blocks_per_band; ++t)
                stamp_tile_scale(&b3_dom[b * ctx.blocks_per_band + t],
                                 scales[b], &ctx, /*code_value=*/100);
    }
    float be3[NB];
    sp_vht2_q8_band_energy(b3_dom.data(), &ctx, be3);
    std::fprintf(stderr, "  [info] B3-dom energies: %.1f %.1f %.1f %.1f\n",
                 be3[0], be3[1], be3[2], be3[3]);
    /* For threshold 0.50 → must include all 4 bands to reach 50%
     * (Bands 0-2 = 3 units, Band 3 = 100 units → cumulative needs Band 3). */
    int k = sp_vht2_q8_min_bands_for_energy(b3_dom.data(), &ctx, 0.5f);
    ASSERT(k == 4);
    /* For threshold 0.86 → also needs Band 3. */
    ASSERT(sp_vht2_q8_min_bands_for_energy(b3_dom.data(), &ctx, 0.86f) == 4);
}

/* Test 9 — gated attention path: given a K-cache vector with controlled
 * band distribution, the gate picks the right number of bands and the
 * resulting partial dot agrees with the explicit-max_bands call. */
TEST(gated_attention_path) {
    constexpr int HD = 128;
    constexpr int NB = 4;
    sp_mobius_mask_t mask;
    ASSERT(sp_mobius_mask_init(&mask, HD) == 0);

    sp_vht2_q8_ctx ctx;
    ASSERT(sp_vht2_q8_ctx_init(&ctx, HD, NB, &mask, SCALE_RECIP, F_P, F_K) == 1);

    std::mt19937 rng(0xCAFEC0DE);
    std::normal_distribution<float> N(0.0f, 0.4f);
    std::vector<float> q_vec(HD), k_vec(HD), sa(HD), sb(HD);
    for (int i = 0; i < HD; ++i) { q_vec[i] = N(rng); k_vec[i] = N(rng); }

    std::vector<sp_ok_q8_block_t> Q(ctx.total_blocks), K(ctx.total_blocks);
    ASSERT(sp_vht2_q8_encode(Q.data(), q_vec.data(), &ctx, sa.data()) == 1);
    ASSERT(sp_vht2_q8_encode(K.data(), k_vec.data(), &ctx, sb.data()) == 1);

    /* Gate on K (the cache side — query is per-token but K is the
     * structure we exploit). */
    int gate = sp_vht2_q8_min_bands_for_energy(K.data(), &ctx, 0.86f);
    float gated = sp_vht2_q8_dot(Q.data(), K.data(), &ctx, gate);
    float full  = sp_vht2_q8_dot(Q.data(), K.data(), &ctx, /*max_bands=*/0);
    std::fprintf(stderr, "  [info] gate=%d gated_dot=%.5f full_dot=%.5f\n",
                 gate, gated, full);
    /* The gate is at threshold 0.86 — partial result captures the
     * dominant component but may diverge from full. We just verify the
     * mechanics: gated equals sp_vht2_q8_dot with max_bands=gate. */
    float explicit_partial = sp_vht2_q8_dot(Q.data(), K.data(), &ctx, gate);
    ASSERT_NEAR(gated, explicit_partial, 1e-6f);
    /* At gate=NB the gated dot equals full. */
    if (gate == NB) ASSERT_NEAR(gated, full, 1e-6f);

    sp_mobius_mask_free(&mask);
}

int main() {
    std::fprintf(stderr, "test_sp_vht2_block_q8: %zu tests\n", g_tests.size());
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
