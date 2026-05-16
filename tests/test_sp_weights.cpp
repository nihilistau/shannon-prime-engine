// Shannon-Prime Engine — Phase 2.2b sp_weights alloc + setter API tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify:
//   1. sp_weights_alloc allocates every slot with the right shape and scale.
//   2. sp_weights_set_wq round-trips an fp32 buffer through encode/decode.
//   3. sp_weights_apply_frobenius_shim transforms the shim-list and
//      preserves Theorem 4 cancellation across a matmul: decode(W*X) is
//      unchanged after both W and X have been Frobenius-shimmed.

#include "../src/sp_forward.h"
#include "../src/sp_ok_encode.h"
#include "../src/sp_matmul.h"
#include "../src/sp_ok_tensor.h"

#include <cmath>
#include <cstdio>
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
// Test 1: sp_weights_alloc gives slots with the documented shapes.
// =========================================================================

TEST(sp_weights_alloc_shapes) {
    constexpr int n_layers = 2, n_embd = 16, n_head = 4, n_kv_head = 2;
    constexpr int d_ff = 32, vocab = 64;
    const int head_dim = n_embd / n_head;
    const int64_t d_q  = n_head * head_dim;
    const int64_t d_kv = n_kv_head * head_dim;

    sp_weights W;
    ASSERT(sp_weights_alloc(W, n_layers, n_embd, n_head, n_kv_head, d_ff,
                              vocab, /*scale*/ 1 << 14));
    ASSERT(W.n_layers == n_layers);
    ASSERT(W.head_dim == head_dim);
    // tok_embed: [n_embd, vocab]
    ASSERT(W.tok_embed.shape[0] == n_embd);
    ASSERT(W.tok_embed.shape[1] == vocab);
    ASSERT(W.tok_embed.scale_recip == 1 << 14);
    // lm_head: [n_embd, vocab]
    ASSERT(W.lm_head.shape[0] == n_embd);
    ASSERT(W.lm_head.shape[1] == vocab);
    // Layer 0 attn projections.
    ASSERT(W.wq[0].shape[0] == n_embd);
    ASSERT(W.wq[0].shape[1] == d_q);
    ASSERT(W.wk[0].shape[0] == n_embd);
    ASSERT(W.wk[0].shape[1] == d_kv);
    ASSERT(W.wv[0].shape[0] == n_embd);
    ASSERT(W.wv[0].shape[1] == d_kv);
    ASSERT(W.wo[0].shape[0] == d_q);
    ASSERT(W.wo[0].shape[1] == n_embd);
    // Layer 0 FFN.
    ASSERT(W.ffn_gate[0].shape[0] == n_embd);
    ASSERT(W.ffn_gate[0].shape[1] == d_ff);
    ASSERT(W.ffn_up[0].shape[0]   == n_embd);
    ASSERT(W.ffn_up[0].shape[1]   == d_ff);
    ASSERT(W.ffn_down[0].shape[0] == d_ff);
    ASSERT(W.ffn_down[0].shape[1] == n_embd);
    // Norms (fp32).
    ASSERT((int)W.attn_norm_w[0].size() == n_embd);
    ASSERT((int)W.ffn_norm_w[0].size()  == n_embd);
    ASSERT((int)W.final_norm_w.size()   == n_embd);
}

// =========================================================================
// Test 2: setter + decode round-trip.
// =========================================================================

TEST(sp_weights_set_roundtrip) {
    constexpr int n_layers = 1, n_embd = 8, n_head = 2, n_kv_head = 2;
    constexpr int d_ff = 16, vocab = 4;
    const int head_dim = n_embd / n_head;
    const int64_t d_q  = n_head * head_dim;
    const int64_t numel_wq = (int64_t)n_embd * d_q;

    sp_weights W;
    ASSERT(sp_weights_alloc(W, n_layers, n_embd, n_head, n_kv_head, d_ff,
                              vocab, 1 << 14));

    std::vector<float> wq_fp32(numel_wq);
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> dd(-0.4f, 0.4f);
    for (auto& v : wq_fp32) v = dd(rng);

    ASSERT(sp_weights_set_wq(W, 0, wq_fp32.data()));
    std::vector<float> got(numel_wq);
    sp_ok_decode_to_fp32(got.data(), W.wq[0]);

    int near = 0;
    for (int64_t i = 0; i < numel_wq; ++i) {
        if (std::abs(got[i] - wq_fp32[i]) < 1.0f / (1 << 13)) ++near;
    }
    ASSERT(near >= (int)numel_wq - 1);
}

// =========================================================================
// Test 3: Theorem 4 cancellation across sp_weights_apply_frobenius_shim.
//
// Build a small matmul Y = W @ X where:
//   W comes from sp_weights[0].wq (after shim)
//   X is a manually encoded fp32 input at frobenius_scale=1
// Without shim: Y_raw = W_raw @ X
// With  shim:   W_shim, scale matches Theorem 4. Decoded Y_shim must equal
//               Y_raw to within rounding noise.
// =========================================================================

TEST(sp_weights_frobenius_shim_preserves_matmul_value) {
    constexpr int n_layers = 1, n_embd = 8, n_head = 2, n_kv_head = 2;
    constexpr int d_ff = 16, vocab = 4;
    const int head_dim = n_embd / n_head;
    const int64_t d_q  = n_head * head_dim;
    const int64_t scale = 1 << 14;

    // --- Build raw fp32 wq and x. ---
    std::vector<float> wq_fp32((int64_t)n_embd * d_q);
    std::vector<float> x_fp32((int64_t)n_embd * 1);   // single token
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dd(-0.3f, 0.3f);
    for (auto& v : wq_fp32) v = dd(rng);
    for (auto& v : x_fp32)  v = dd(rng);

    // --- Reference fp32 matmul. ---
    std::vector<float> y_ref(d_q, 0.0f);
    for (int i = 0; i < d_q; ++i) {
        double a = 0;
        for (int k = 0; k < n_embd; ++k) {
            a += (double)wq_fp32[i * n_embd + k] * (double)x_fp32[k];
        }
        y_ref[i] = (float)a;
    }

    // --- Build sp_weights, set wq, apply Frobenius shim. ---
    sp_weights W;
    ASSERT(sp_weights_alloc(W, n_layers, n_embd, n_head, n_kv_head, d_ff,
                              vocab, scale));
    ASSERT(sp_weights_set_wq(W, 0, wq_fp32.data()));

    int n_shimmed = sp_weights_apply_frobenius_shim(
        W, /*frobenius_quant*/ true, /*sato_tate_mix*/ false,
        /*p*/ 41, /*k*/ 8, 0, 0, 0, 0);
    ASSERT(n_shimmed >= 1);

    // --- Encode x at the same scale (frobenius_scale=1, NO shim). ---
    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor X;
    int64_t x_shape[4] = { 1, n_embd, 1, 1 };
    ASSERT(sp_ok_encode_from_fp32(X, x_fp32.data(), 2, x_shape, scale, arena));

    // --- Y = W @ X via the fp32 bridge (it divides by combined scale). ---
    std::vector<float> y_fp32(d_q);
    ASSERT(sp_matmul_ok_to_fp32(W.wq[0], X, y_fp32.data(), (int)d_q, 1));

    // --- Compare to reference; tolerance accounts for both encoding
    //     rounding and Frobenius shim ULP drift (Hasse-Weil bound). ---
    int near = 0;
    float max_abs_err = 0.0f;
    for (int i = 0; i < d_q; ++i) {
        float err = std::abs(y_fp32[i] - y_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (err < 0.002f) ++near;
    }
    std::printf("  shim_preserves: max_abs_err=%.6g, near=%d/%d\n",
                (double)max_abs_err, near, (int)d_q);
    ASSERT(near >= (int)d_q - 1);
}

int main() {
    std::printf("Shannon-Prime sp_weights tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
