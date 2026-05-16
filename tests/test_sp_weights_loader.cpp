// Shannon-Prime Engine — Phase 2.2c sp_weights loader test.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify the fp16-source loader builds an sp_weights end-to-end and that
// every slot decodes back to (approximately) the original fp32 source
// values. Also verify Theorem 4 cancellation across the full load path
// when --frobenius-quant is enabled.

#include "../src/sp_forward.h"
#include "../src/sp_weights_loader.h"
#include "../src/sp_ok_encode.h"
#include "../src/sp_matmul.h"
#include "../src/sp_ok_tensor.h"
#include "../src/engine.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

using namespace sp::engine;

// =========================================================================
// fp32 ↔ fp16 helpers (identical to sp_ok_encode.cpp's local copy).
// =========================================================================

static inline uint16_t fp32_to_fp16(float v) {
    uint32_t f; std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}

static std::vector<uint16_t> to_fp16_vec(const std::vector<float>& src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = fp32_to_fp16(src[i]);
    return dst;
}

// =========================================================================
// Build a synthetic 2-layer model's worth of random weights, run them
// through sp_weights_load_from_fp16_source, and check round-trip.
// =========================================================================

namespace {
struct SyntheticModel {
    int n_layers, n_embd, n_head, n_kv_head, d_ff, vocab, head_dim;
    int d_q, d_kv;
    std::vector<float> tok_embd_fp32, lm_head_fp32, final_norm_fp32;
    struct Layer {
        std::vector<float> wq, wk, wv, wo;
        std::vector<float> ffn_gate, ffn_up, ffn_down;
        std::vector<float> attn_norm, ffn_norm;
    };
    std::vector<Layer> layers;

    // fp16 mirrors (owned for the duration of the load).
    std::vector<uint16_t> tok_embd_fp16, lm_head_fp16;
    struct Layer16 {
        std::vector<uint16_t> wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down;
    };
    std::vector<Layer16> layers16;
};

static SyntheticModel build_synth(int n_layers, int n_embd, int n_head,
                                    int n_kv_head, int d_ff, int vocab,
                                    uint64_t seed) {
    SyntheticModel m;
    m.n_layers = n_layers;
    m.n_embd = n_embd;
    m.n_head = n_head;
    m.n_kv_head = n_kv_head;
    m.d_ff = d_ff;
    m.vocab = vocab;
    m.head_dim = n_embd / n_head;
    m.d_q = n_head * m.head_dim;
    m.d_kv = n_kv_head * m.head_dim;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dd(-0.3f, 0.3f);
    auto fill = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        for (auto& x : v) x = dd(rng);
    };

    fill(m.tok_embd_fp32,   (size_t)vocab * n_embd);
    fill(m.lm_head_fp32,    (size_t)vocab * n_embd);
    fill(m.final_norm_fp32, n_embd);
    for (auto& v : m.final_norm_fp32) v = 1.0f + 0.1f * v;  // norms ~ 1.0

    m.layers.resize(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        auto& lyr = m.layers[L];
        fill(lyr.wq,       (size_t)m.d_q  * n_embd);
        fill(lyr.wk,       (size_t)m.d_kv * n_embd);
        fill(lyr.wv,       (size_t)m.d_kv * n_embd);
        fill(lyr.wo,       (size_t)n_embd * m.d_q);
        fill(lyr.ffn_gate, (size_t)d_ff  * n_embd);
        fill(lyr.ffn_up,   (size_t)d_ff  * n_embd);
        fill(lyr.ffn_down, (size_t)n_embd * d_ff);
        fill(lyr.attn_norm, n_embd);
        fill(lyr.ffn_norm,  n_embd);
        for (auto& v : lyr.attn_norm) v = 1.0f + 0.1f * v;
        for (auto& v : lyr.ffn_norm)  v = 1.0f + 0.1f * v;
    }

    m.tok_embd_fp16 = to_fp16_vec(m.tok_embd_fp32);
    m.lm_head_fp16  = to_fp16_vec(m.lm_head_fp32);
    m.layers16.resize(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        auto& l16 = m.layers16[L];
        const auto& l  = m.layers[L];
        l16.wq       = to_fp16_vec(l.wq);
        l16.wk       = to_fp16_vec(l.wk);
        l16.wv       = to_fp16_vec(l.wv);
        l16.wo       = to_fp16_vec(l.wo);
        l16.ffn_gate = to_fp16_vec(l.ffn_gate);
        l16.ffn_up   = to_fp16_vec(l.ffn_up);
        l16.ffn_down = to_fp16_vec(l.ffn_down);
    }
    return m;
}

static sp_weights_fp16_source make_source(
    const SyntheticModel& m,
    std::vector<sp_weights_layer_fp16_source>& layer_srcs)
{
    layer_srcs.resize(m.n_layers);
    for (int L = 0; L < m.n_layers; ++L) {
        auto& s = layer_srcs[L];
        s.wq        = m.layers16[L].wq.data();
        s.wk        = m.layers16[L].wk.data();
        s.wv        = m.layers16[L].wv.data();
        s.wo        = m.layers16[L].wo.data();
        s.ffn_gate  = m.layers16[L].ffn_gate.data();
        s.ffn_up    = m.layers16[L].ffn_up.data();
        s.ffn_down  = m.layers16[L].ffn_down.data();
        s.attn_norm = m.layers[L].attn_norm.data();
        s.ffn_norm  = m.layers[L].ffn_norm.data();
    }
    sp_weights_fp16_source src;
    src.n_layers   = m.n_layers;
    src.n_embd     = m.n_embd;
    src.n_head     = m.n_head;
    src.n_kv_head  = m.n_kv_head;
    src.d_ff       = m.d_ff;
    src.vocab      = m.vocab;
    src.tok_embd   = m.tok_embd_fp16.data();
    src.lm_head    = m.lm_head_fp16.data();
    src.final_norm = m.final_norm_fp32.data();
    src.layers     = layer_srcs.data();
    return src;
}
}  // anon namespace

// =========================================================================
// Test 1: loader builds sp_weights with correct shapes and decode round-trips.
// =========================================================================

TEST(loader_no_shim_roundtrip) {
    auto m = build_synth(/*n_layers*/ 2, /*n_embd*/ 16, /*n_head*/ 4,
                          /*n_kv_head*/ 2, /*d_ff*/ 32, /*vocab*/ 8,
                          /*seed*/ 51);
    std::vector<sp_weights_layer_fp16_source> layer_srcs;
    auto src = make_source(m, layer_srcs);

    Config cfg;
    cfg.frobenius_quant = false;
    cfg.sato_tate_mix   = false;

    sp_weights W;
    ASSERT(sp_weights_load_from_fp16_source(W, src, cfg, /*scale*/ 1 << 14));
    ASSERT(W.n_layers == 2);
    ASSERT(W.n_embd   == 16);
    ASSERT(W.n_head   == 4);
    ASSERT(W.n_kv_head == 2);
    ASSERT(W.d_ff     == 32);
    ASSERT(W.vocab    == 8);

    // Decode wq[0] and verify it matches the source fp32 (within fp16
    // rounding noise + scale_recip ULP).
    std::vector<float> got((size_t)m.d_q * m.n_embd);
    sp_ok_decode_to_fp32(got.data(), W.wq[0]);
    int near = 0;
    const int N = (int)got.size();
    for (int i = 0; i < N; ++i) {
        if (std::abs(got[i] - m.layers[0].wq[i]) < 0.001f) ++near;
    }
    ASSERT(near >= N - 5);

    // Norms must match exactly (fp32 → fp32 copy).
    for (int i = 0; i < m.n_embd; ++i) {
        ASSERT(W.attn_norm_w[0][i] == m.layers[0].attn_norm[i]);
        ASSERT(W.ffn_norm_w[1][i]  == m.layers[1].ffn_norm[i]);
        ASSERT(W.final_norm_w[i]   == m.final_norm_fp32[i]);
    }
}

// =========================================================================
// Test 2: with --frobenius-quant the shim is applied to the shim-list
// (every projection in every layer) and NOT to tok_embed / lm_head /
// norms. Verify Theorem 4 cancellation: decoded(W·X) ≈ raw fp32 W·X.
// =========================================================================

TEST(loader_with_frobenius_shim_preserves_matmul) {
    auto m = build_synth(1, 8, 2, 2, 16, 4, /*seed*/ 53);
    std::vector<sp_weights_layer_fp16_source> layer_srcs;
    auto src = make_source(m, layer_srcs);

    Config cfg;
    cfg.frobenius_quant = true;
    cfg.frobenius_p     = 41;
    cfg.frobenius_k     = 8;

    sp_weights W;
    ASSERT(sp_weights_load_from_fp16_source(W, src, cfg, 1 << 14));

    // Verify the shim hit wq (a SHIM-list tensor). Phase 2.3b iter 5:
    // tok_embed and lm_head are fp32 vectors (no frobenius_scale field),
    // so the bypass verification reduces to "they aren't in the
    // sp_ok_tensor arena at all."
    ASSERT(W.wq[0].frobenius_scale != 1);     // shimmed
    ASSERT(!W.tok_embed_fp32.empty());        // populated as fp32
    ASSERT(!W.lm_head_fp32.empty());          // populated as fp32

    // Theorem 4 round-trip: compute fp32 W·x reference using the source
    // fp32 weights; compute the same via sp_matmul_ok_to_fp32 on the
    // shimmed sp_weights. The two must match within the combined fp16-
    // rounding + ULP envelope.
    std::vector<float> x_fp32(m.n_embd);
    std::mt19937 rng(59);
    std::uniform_real_distribution<float> dd(-0.3f, 0.3f);
    for (auto& v : x_fp32) v = dd(rng);

    // Reference: y_ref[i] = sum_k wq_fp32[i*n_embd + k] * x_fp32[k]
    // (where wq_fp32 came from the ORIGINAL fp32 source — note fp16
    // round-trip injects a small per-element error).
    std::vector<float> y_ref(m.d_q, 0.0f);
    for (int i = 0; i < m.d_q; ++i) {
        double a = 0;
        for (int k = 0; k < m.n_embd; ++k) {
            a += (double)m.layers[0].wq[i * m.n_embd + k] * (double)x_fp32[k];
        }
        y_ref[i] = (float)a;
    }

    // Encode x at the same scale (no shim).
    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor X;
    int64_t x_shape[4] = { 1, m.n_embd, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 14, arena));

    std::vector<float> y_fp32(m.d_q);
    ASSERT(sp_matmul_ok_to_fp32(W.wq[0], X, y_fp32.data(), m.d_q, 1));

    // Acceptance: max_abs_err < 0.01 over d_q=8 elements. With fp16-loaded
    // weights + 1<<14 scale + φ_41^8 Frobenius cancellation, this gives
    // headroom for ~1e-3 per-element error (fp16 ULP * sum-of-products).
    int near = 0;
    float max_err = 0;
    for (int i = 0; i < m.d_q; ++i) {
        float e = std::abs(y_fp32[i] - y_ref[i]);
        if (e > max_err) max_err = e;
        if (e < 0.01f) ++near;
    }
    std::printf("  loader_shim_preserves: max_abs_err=%.6g near=%d/%d\n",
                (double)max_err, near, m.d_q);
    ASSERT(near >= m.d_q - 1);
}

// =========================================================================
// Test 3: with --sato-tate-mix (Config E) the shim composes phi_2^2 ∘ phi_41^8
// and decode still recovers the original matmul to ULP.
// =========================================================================

TEST(loader_with_sato_tate_mix_preserves_matmul) {
    auto m = build_synth(1, 8, 2, 2, 16, 4, /*seed*/ 67);
    std::vector<sp_weights_layer_fp16_source> layer_srcs;
    auto src = make_source(m, layer_srcs);

    Config cfg;
    cfg.sato_tate_mix = true;
    cfg.st_p1 = 2;  cfg.st_k1 = 2;
    cfg.st_p2 = 41; cfg.st_k2 = 4;  // smaller k to leave int64 headroom

    sp_weights W;
    ASSERT(sp_weights_load_from_fp16_source(W, src, cfg, 1 << 14));
    ASSERT(W.wq[0].frobenius_scale != 1);

    // Same reference / shimmed comparison as Test 2.
    std::vector<float> x_fp32(m.n_embd);
    std::mt19937 rng(71);
    std::uniform_real_distribution<float> dd(-0.3f, 0.3f);
    for (auto& v : x_fp32) v = dd(rng);
    std::vector<float> y_ref(m.d_q, 0.0f);
    for (int i = 0; i < m.d_q; ++i) {
        double a = 0;
        for (int k = 0; k < m.n_embd; ++k) {
            a += (double)m.layers[0].wq[i * m.n_embd + k] * (double)x_fp32[k];
        }
        y_ref[i] = (float)a;
    }
    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor X;
    int64_t x_shape[4] = { 1, m.n_embd, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, 1 << 14, arena));
    std::vector<float> y_fp32(m.d_q);
    ASSERT(sp_matmul_ok_to_fp32(W.wq[0], X, y_fp32.data(), m.d_q, 1));
    int near = 0;
    for (int i = 0; i < m.d_q; ++i) {
        if (std::abs(y_fp32[i] - y_ref[i]) < 0.01f) ++near;
    }
    ASSERT(near >= m.d_q - 1);
}

int main() {
    std::printf("Shannon-Prime sp_weights_loader tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
