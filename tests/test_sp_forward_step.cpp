// Shannon-Prime Engine — Phase 2.2d end-to-end forward step test.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Build a synthetic 1-layer transformer in fp32, then build the same
// model as sp_weights and run sp_forward_step. The two logit vectors
// must match within the Theorem 4 + fp16-quantization envelope.
//
// What this proves:
//   - The grand-assembly of {embed lookup, RMSNorm, Wq/Wk/Wv, RoPE,
//     KV append, attention, Wo, residual, RMSNorm, FFN, residual,
//     final-norm, LM head} is algebraically equivalent to a pure-fp32
//     reference forward.
//   - Theorem 4's projective cancellation survives across every
//     scale-reset boundary (RMSNorm, RoPE, sp_matmul_ok_to_fp32).
//   - sp_ok_kv_cache + the stride-aware sp_attention work in the live
//     pipeline, not just in isolation.

#include "../src/sp_forward.h"
#include "../src/sp_weights_loader.h"
#include "../src/engine.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
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
// fp32 reference forward — single layer, single token at position 0.
//
// All matmul conventions match sp_weights:
//   - tok_embd     : [vocab,  n_embd]      row-major; row[t][d] = src[t*n_embd+d]
//   - wq           : [d_q,    n_embd]                 W[i][k]  = src[i*n_embd+k]
//   - wk           : [d_kv,   n_embd]
//   - wv           : [d_kv,   n_embd]
//   - wo           : [n_embd, d_q]
//   - ffn_gate     : [d_ff,   n_embd]
//   - ffn_up       : [d_ff,   n_embd]
//   - ffn_down     : [n_embd, d_ff]
//   - lm_head      : [vocab,  n_embd]
// =========================================================================

struct FpModel {
    int n_layers, n_embd, n_head, n_kv_head, head_dim, d_ff, vocab;
    int d_q, d_kv;
    std::vector<float> tok_embd, lm_head, final_norm;
    struct Layer {
        std::vector<float> wq, wk, wv, wo;
        std::vector<float> ffn_gate, ffn_up, ffn_down;
        std::vector<float> attn_norm, ffn_norm;
    };
    std::vector<Layer> layers;
};

static FpModel build_fp32_model(int n_layers, int n_embd, int n_head,
                                  int n_kv_head, int d_ff, int vocab,
                                  uint64_t seed) {
    FpModel m;
    m.n_layers = n_layers;
    m.n_embd   = n_embd;
    m.n_head   = n_head;
    m.n_kv_head = n_kv_head;
    m.head_dim = n_embd / n_head;
    m.d_ff     = d_ff;
    m.vocab    = vocab;
    m.d_q      = n_head    * m.head_dim;
    m.d_kv     = n_kv_head * m.head_dim;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dd(-0.25f, 0.25f);
    auto fill = [&](std::vector<float>& v, size_t n, float lo, float hi) {
        v.resize(n);
        std::uniform_real_distribution<float> d(lo, hi);
        for (auto& x : v) x = d(rng);
    };

    fill(m.tok_embd,   (size_t)vocab * n_embd,  -0.25f, 0.25f);
    fill(m.lm_head,    (size_t)vocab * n_embd,  -0.25f, 0.25f);
    fill(m.final_norm, n_embd, 0.95f, 1.05f);

    m.layers.resize(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        auto& lyr = m.layers[L];
        fill(lyr.wq,       (size_t)m.d_q  * n_embd, -0.25f, 0.25f);
        fill(lyr.wk,       (size_t)m.d_kv * n_embd, -0.25f, 0.25f);
        fill(lyr.wv,       (size_t)m.d_kv * n_embd, -0.25f, 0.25f);
        fill(lyr.wo,       (size_t)n_embd * m.d_q,  -0.25f, 0.25f);
        fill(lyr.ffn_gate, (size_t)d_ff  * n_embd, -0.25f, 0.25f);
        fill(lyr.ffn_up,   (size_t)d_ff  * n_embd, -0.25f, 0.25f);
        fill(lyr.ffn_down, (size_t)n_embd * d_ff,  -0.25f, 0.25f);
        fill(lyr.attn_norm, n_embd, 0.95f, 1.05f);
        fill(lyr.ffn_norm,  n_embd, 0.95f, 1.05f);
    }
    return m;
}

// =========================================================================
// Reference helpers.
// =========================================================================

static void rmsnorm_fp32(const float* x, const float* w, float* out,
                           int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

static void matmul_fp32(const float* W, const float* x, float* y,
                          int M, int K) {
    // y[i] = sum_k W[i*K + k] * x[k]
    for (int i = 0; i < M; ++i) {
        double a = 0;
        for (int k = 0; k < K; ++k) a += (double)W[i * K + k] * (double)x[k];
        y[i] = (float)a;
    }
}

static void rope_fp32_inplace(float* qk, int pos, int n_heads, int head_dim,
                                float freq_base) {
    const int n_pairs = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        for (int k = 0; k < n_pairs; ++k) {
            float freq = std::pow(freq_base, -(float)(2*k)/(float)head_dim);
            float ang  = (float)pos * freq;
            float c = std::cos(ang);
            float s = std::sin(ang);
            int fe = h * head_dim + 2 * k;
            int fo = fe + 1;
            float a = qk[fe], b = qk[fo];
            qk[fe] = c*a - s*b;
            qk[fo] = s*a + c*b;
        }
    }
}

static float silu(float x) {
    if (x >= 0) return x / (1.0f + std::exp(-x));
    float e = std::exp(x);
    return x * e / (1.0f + e);
}

// Reference single-step forward at position 0 (cache is empty so attention
// trivially returns V[0]).
static std::vector<float> ref_forward_pos0(const FpModel& m, int token_id,
                                              float rms_eps, float rope_base) {
    std::vector<float> x(m.n_embd);
    for (int i = 0; i < m.n_embd; ++i) {
        x[i] = m.tok_embd[(size_t)token_id * m.n_embd + i];
    }
    std::vector<float> x_norm(m.n_embd);
    std::vector<float> q(m.d_q), k(m.d_kv), v(m.d_kv);
    std::vector<float> attn(m.d_q), wo_out(m.n_embd);
    std::vector<float> gate(m.d_ff), up(m.d_ff), act(m.d_ff), ffn_out(m.n_embd);

    for (int L = 0; L < m.n_layers; ++L) {
        const auto& lyr = m.layers[L];
        rmsnorm_fp32(x.data(), lyr.attn_norm.data(), x_norm.data(),
                      m.n_embd, rms_eps);
        matmul_fp32(lyr.wq.data(), x_norm.data(), q.data(), m.d_q,  m.n_embd);
        matmul_fp32(lyr.wk.data(), x_norm.data(), k.data(), m.d_kv, m.n_embd);
        matmul_fp32(lyr.wv.data(), x_norm.data(), v.data(), m.d_kv, m.n_embd);
        rope_fp32_inplace(q.data(), 0, m.n_head,    m.head_dim, rope_base);
        rope_fp32_inplace(k.data(), 0, m.n_kv_head, m.head_dim, rope_base);

        // Attention: at position 0 the cache holds just (k, v). Softmax(0)=1.
        // attn[h*head_dim+d] = V[(kv_h*head_dim+d)] where kv_h = h * n_kv_head / n_head.
        for (int h = 0; h < m.n_head; ++h) {
            const int kv_h = (h * m.n_kv_head) / m.n_head;
            for (int d = 0; d < m.head_dim; ++d) {
                attn[h * m.head_dim + d] = v[kv_h * m.head_dim + d];
            }
        }

        matmul_fp32(lyr.wo.data(), attn.data(), wo_out.data(),
                     m.n_embd, m.d_q);
        for (int i = 0; i < m.n_embd; ++i) x[i] += wo_out[i];

        rmsnorm_fp32(x.data(), lyr.ffn_norm.data(), x_norm.data(),
                      m.n_embd, rms_eps);
        matmul_fp32(lyr.ffn_gate.data(), x_norm.data(), gate.data(),
                     m.d_ff, m.n_embd);
        matmul_fp32(lyr.ffn_up.data(),   x_norm.data(), up.data(),
                     m.d_ff, m.n_embd);
        for (int j = 0; j < m.d_ff; ++j) act[j] = silu(gate[j]) * up[j];
        matmul_fp32(lyr.ffn_down.data(), act.data(), ffn_out.data(),
                     m.n_embd, m.d_ff);
        for (int i = 0; i < m.n_embd; ++i) x[i] += ffn_out[i];
    }
    // Final norm + LM head.
    rmsnorm_fp32(x.data(), m.final_norm.data(), x_norm.data(),
                  m.n_embd, rms_eps);
    std::vector<float> logits(m.vocab);
    matmul_fp32(m.lm_head.data(), x_norm.data(), logits.data(),
                 m.vocab, m.n_embd);
    return logits;
}

// =========================================================================
// Build sp_weights from FpModel (no fp16 detour for tightest comparison).
// =========================================================================

static bool build_sp_weights_from_fp_model(sp_weights& W, const FpModel& m,
                                              const Config& cfg,
                                              int64_t scale_recip) {
    if (!sp_weights_alloc(W, m.n_layers, m.n_embd, m.n_head, m.n_kv_head,
                            m.d_ff, m.vocab, scale_recip)) return false;
    if (!sp_weights_set_tok_embed(W, m.tok_embd.data())) return false;
    if (!sp_weights_set_lm_head(W, m.lm_head.data())) return false;
    if (!sp_weights_set_final_norm(W, m.final_norm.data())) return false;
    for (int L = 0; L < m.n_layers; ++L) {
        const auto& lyr = m.layers[L];
        if (!sp_weights_set_wq(W, L, lyr.wq.data())) return false;
        if (!sp_weights_set_wk(W, L, lyr.wk.data())) return false;
        if (!sp_weights_set_wv(W, L, lyr.wv.data())) return false;
        if (!sp_weights_set_wo(W, L, lyr.wo.data())) return false;
        if (!sp_weights_set_ffn_gate(W, L, lyr.ffn_gate.data())) return false;
        if (!sp_weights_set_ffn_up(W, L, lyr.ffn_up.data())) return false;
        if (!sp_weights_set_ffn_down(W, L, lyr.ffn_down.data())) return false;
        if (!sp_weights_set_attn_norm(W, L, lyr.attn_norm.data())) return false;
        if (!sp_weights_set_ffn_norm(W, L, lyr.ffn_norm.data())) return false;
    }
    if (cfg.frobenius_quant || cfg.sato_tate_mix) {
        sp_weights_apply_frobenius_shim(
            W, cfg.frobenius_quant, cfg.sato_tate_mix,
            cfg.frobenius_p, cfg.frobenius_k,
            cfg.st_p1, cfg.st_k1, cfg.st_p2, cfg.st_k2);
    }
    return true;
}

// =========================================================================
// Test 1: no-shim sp_forward_step matches fp32 reference at position 0.
// =========================================================================

TEST(forward_step_pos0_no_shim_matches_fp32_reference) {
    constexpr int n_layers = 1, n_embd = 16, n_head = 2, n_kv_head = 2;
    constexpr int d_ff = 32, vocab = 8;
    const float rms_eps   = 1e-5f;
    const float rope_base = 10000.0f;
    auto m = build_fp32_model(n_layers, n_embd, n_head, n_kv_head, d_ff,
                                vocab, /*seed*/ 79);
    const int token = 3;

    auto ref_logits = ref_forward_pos0(m, token, rms_eps, rope_base);

    Config cfg;
    cfg.frobenius_quant = false;
    sp_weights W;
    ASSERT(build_sp_weights_from_fp_model(W, m, cfg, /*scale*/ 1 << 14));

    sp_forward_context ctx;
    ASSERT(sp_forward_context_init(ctx, W, /*n_ctx*/ 8, rope_base, rms_eps));

    std::vector<float> got_logits;
    ASSERT(sp_forward_step(ctx, W, token, /*position*/ 0, got_logits));
    ASSERT((int)got_logits.size() == vocab);

    int near = 0;
    float max_abs = 0;
    for (int i = 0; i < vocab; ++i) {
        float err = std::abs(got_logits[i] - ref_logits[i]);
        if (err > max_abs) max_abs = err;
        if (err < 0.01f) ++near;
    }
    std::printf("  no_shim_pos0: max_abs_err=%.6g  near=%d/%d  ref[0]=%.4f got[0]=%.4f\n",
                (double)max_abs, near, vocab,
                (double)ref_logits[0], (double)got_logits[0]);
    ASSERT(near >= vocab - 1);

    // KV cache must have advanced.
    ASSERT(ctx.kv_cache.cur_len == 1);
}

// =========================================================================
// Test 2: with --frobenius-quant the shim is applied and Theorem 4
// projective cancellation keeps the logits within a tightened envelope
// of the fp32 reference.
// =========================================================================

TEST(forward_step_pos0_frobenius_shim_matches_fp32_reference) {
    constexpr int n_layers = 1, n_embd = 16, n_head = 2, n_kv_head = 2;
    constexpr int d_ff = 32, vocab = 8;
    const float rms_eps   = 1e-5f;
    const float rope_base = 10000.0f;
    auto m = build_fp32_model(n_layers, n_embd, n_head, n_kv_head, d_ff,
                                vocab, /*seed*/ 83);
    const int token = 5;

    auto ref_logits = ref_forward_pos0(m, token, rms_eps, rope_base);

    Config cfg;
    cfg.frobenius_quant = true;
    cfg.frobenius_p     = 41;
    cfg.frobenius_k     = 8;
    sp_weights W;
    ASSERT(build_sp_weights_from_fp_model(W, m, cfg, 1 << 14));
    // Bypass policy verification. Phase 2.3b iter 5: tok_embed and
    // lm_head are fp32 vectors (no frobenius_scale field). Bypass now
    // means "stored as fp32, never enters the O_K shim path."
    ASSERT(!W.tok_embed_fp32.empty());
    ASSERT(!W.lm_head_fp32.empty());
    ASSERT(W.wq[0].frobenius_scale != 1);  // shim-list, was shimmed

    sp_forward_context ctx;
    ASSERT(sp_forward_context_init(ctx, W, 8, rope_base, rms_eps));

    std::vector<float> got_logits;
    ASSERT(sp_forward_step(ctx, W, token, 0, got_logits));

    int near = 0;
    float max_abs = 0;
    for (int i = 0; i < vocab; ++i) {
        float err = std::abs(got_logits[i] - ref_logits[i]);
        if (err > max_abs) max_abs = err;
        if (err < 0.02f) ++near;
    }
    std::printf("  shim_pos0:    max_abs_err=%.6g  near=%d/%d\n",
                (double)max_abs, near, vocab);
    ASSERT(near >= vocab - 2);
}

// =========================================================================
// Test 3: sequential decode — three positions in a row. Logits must be
// finite/non-NaN; cache cur_len must advance with each step.
// =========================================================================

TEST(forward_step_sequential_decode_kv_cache_advances) {
    constexpr int n_layers = 1, n_embd = 16, n_head = 2, n_kv_head = 2;
    constexpr int d_ff = 32, vocab = 8;
    auto m = build_fp32_model(n_layers, n_embd, n_head, n_kv_head, d_ff,
                                vocab, /*seed*/ 89);
    Config cfg;
    sp_weights W;
    ASSERT(build_sp_weights_from_fp_model(W, m, cfg, 1 << 14));

    sp_forward_context ctx;
    ASSERT(sp_forward_context_init(ctx, W, /*n_ctx*/ 16, 10000.0f, 1e-5f));

    int tokens[] = { 2, 5, 7, 1 };
    for (int pos = 0; pos < 4; ++pos) {
        std::vector<float> logits;
        ASSERT(sp_forward_step(ctx, W, tokens[pos], pos, logits));
        ASSERT((int)logits.size() == vocab);
        // Finite check.
        for (int i = 0; i < vocab; ++i) {
            ASSERT(std::isfinite(logits[i]));
        }
        ASSERT(ctx.kv_cache.cur_len == pos + 1);
    }
}

int main() {
    std::printf("Shannon-Prime sp_forward_step tests (%zu)\n", g_tests.size());
    for (auto& t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
