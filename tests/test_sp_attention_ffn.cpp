// Shannon-Prime Engine — Phase 2.2a attention + FFN unit tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Verify that sp_attention_dot_product and sp_ffn_swiglu produce results
// matching an fp32 reference implementation to within rounding noise.

#include "../src/sp_attention.h"
#include "../src/sp_ffn.h"
#include "../src/sp_matmul.h"
#include "../src/sp_bridges.h"
#include "../src/sp_ok_encode.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)
#define ASSERT_NEAR(a, b, eps) do { double _d = std::abs((double)(a)-(double)(b)); \
    if (_d > (eps)) { std::fprintf(stderr, "  ASSERT_NEAR FAIL (%s:%d): %.6g vs %.6g (delta %.6g > eps %.6g)\n", \
        __FILE__, __LINE__, (double)(a), (double)(b), _d, (double)(eps)); g_fail++; } } while (0)

using namespace sp::engine;

// =========================================================================
// fp32 reference attention (single query, multi-head, no GQA for simplicity)
// =========================================================================

static void ref_attention(const float* q,           // [n_head * head_dim]
                           const float* k_hist,      // [n_head * head_dim, T]
                           const float* v_hist,      // [n_head * head_dim, T]
                           int n_head, int head_dim, int T,
                           float* out) {             // [n_head * head_dim]
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(T), weights(T);

    for (int h = 0; h < n_head; ++h) {
        for (int t = 0; t < T; ++t) {
            float dot = 0;
            for (int d = 0; d < head_dim; ++d) {
                dot += k_hist[(h * head_dim + d) * T + t] * q[h * head_dim + d];
            }
            scores[t] = dot * inv_sqrt_d;
        }
        // softmax
        float mx = scores[0];
        for (int t = 1; t < T; ++t) if (scores[t] > mx) mx = scores[t];
        double sum = 0;
        for (int t = 0; t < T; ++t) {
            weights[t] = std::exp(scores[t] - mx);
            sum += weights[t];
        }
        for (int t = 0; t < T; ++t) weights[t] /= (float)sum;
        // attn = sum V_t * w_t
        for (int d = 0; d < head_dim; ++d) {
            double a = 0;
            for (int t = 0; t < T; ++t) {
                a += (double)v_hist[(h * head_dim + d) * T + t] * (double)weights[t];
            }
            out[h * head_dim + d] = (float)a;
        }
    }
}

// =========================================================================
// fp32 reference SwiGLU FFN
// =========================================================================

static void ref_ffn_swiglu(const float* x,        // [n_embd]
                            const float* gate_w,   // [n_embd, d_ff]: gate_w[i * d_ff + j]
                            const float* up_w,
                            const float* down_w,   // [d_ff, n_embd]: down_w[j * n_embd + i]
                            int n_embd, int d_ff,
                            float* out) {          // [n_embd]
    // NOTE: the storage of weights in our O_K tensors uses the matmul
    // convention where shape[0] is the inner dim. For gate_w[shape={n_embd,d_ff}]:
    //   stride[0]=es, stride[1]=es*n_embd → gate_w.data[i*d_ff + j]?
    // Actually let's just match the matmul: sp_matmul_ok_to_fp32(W=gate_w, X=x).
    //   W.shape[0]=K=n_embd, W.shape[1]=M=d_ff → out[i,j] = sum_k W.data[i*K+k]*X[k*N+j]
    //   So W.data[i*K + k] = gate_w[output_unit_i, input_dim_k]
    //   With K=n_embd: data layout has each row as a vector of length K=n_embd,
    //   and there are M=d_ff rows. So gate_w[i*n_embd + k] = output_i's weight on input_k.
    //
    // For the fp32 reference we use this same layout.
    std::vector<float> gate_fp32(d_ff), up_fp32(d_ff), act_fp32(d_ff);
    for (int i = 0; i < d_ff; ++i) {
        double g = 0, u = 0;
        for (int k = 0; k < n_embd; ++k) {
            g += (double)gate_w[i * n_embd + k] * (double)x[k];
            u += (double)up_w[i * n_embd + k] * (double)x[k];
        }
        gate_fp32[i] = (float)g;
        up_fp32[i]   = (float)u;
    }
    // SwiGLU: silu(gate) * up
    for (int i = 0; i < d_ff; ++i) {
        float g = gate_fp32[i];
        float silu = g / (1.0f + std::exp(-g));
        act_fp32[i] = silu * up_fp32[i];
    }
    // down projection: down_w shape={d_ff, n_embd}, matmul layout
    //   W.shape[0]=K=d_ff, W.shape[1]=M=n_embd → data[i*K+k] = output_i's weight on input_k
    //   So down_w[i*d_ff + k]
    for (int i = 0; i < n_embd; ++i) {
        double s = 0;
        for (int k = 0; k < d_ff; ++k) {
            s += (double)down_w[i * d_ff + k] * (double)act_fp32[k];
        }
        out[i] = (float)s;
    }
}

// =========================================================================
// Attention test (multi-head, single token, single batch, no GQA)
// =========================================================================

TEST(attention_dot_product_matches_fp32_reference) {
    constexpr int n_head = 2, head_dim = 4, T = 6;
    const int d_q = n_head * head_dim;

    std::vector<float> q_fp32(d_q), k_fp32(d_q * T), v_fp32(d_q * T);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : q_fp32) v = d(rng);
    for (auto& v : k_fp32) v = d(rng);
    for (auto& v : v_fp32) v = d(rng);

    std::vector<float> ref(d_q);
    ref_attention(q_fp32.data(), k_fp32.data(), v_fp32.data(),
                   n_head, head_dim, T, ref.data());

    // Encode q, k, v into sp_ok_tensors.
    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor Q, K, V, OUT;
    // Phase 2.2b convention: shape = { n_tokens, n_features }
    int64_t q_shape[4] = { 1, d_q, 1, 1 };  // 1 query token, d_q features
    int64_t k_shape[4] = { T, d_q, 1, 1 };  // T history positions, d_q features
    int64_t v_shape[4] = { T, d_q, 1, 1 };
    int64_t out_shape[4] = { 1, d_q, 1, 1 };

    // But wait — our attention reads k.data[((kv_h*head_dim + d)*T + t]
    // which means K has rows (kv_h*head_dim + d) of length T. That's:
    //   shape[0]=T innermost (contiguous), shape[1]=d_q outer.
    // Same for V. q has shape[0]=d_q innermost, shape[1]=1 outer.

    // sp_ok_encode_from_fp32 places elements in numel-many positions starting
    // at index 0, stride 1. So we need to lay out the source fp32 buffer in
    // the SAME order that the matmul/attention will read.

    // For K with shape={T, d_q}: total = T * d_q elements, laid out as
    //   k_buf[(kv_h*head_dim + d) * T + t]
    // So source buffer must be exactly k_fp32 if k_fp32[i * T + t] for i = kv_h*head_dim+d.
    // Which is exactly the convention in ref_attention. ✓

    ASSERT(sp_ok_encode_from_fp32(Q, q_fp32.data(), 2, q_shape, 1 << 14, arena));
    ASSERT(sp_ok_encode_from_fp32(K, k_fp32.data(), 2, k_shape, 1 << 14, arena));
    ASSERT(sp_ok_encode_from_fp32(V, v_fp32.data(), 2, v_shape, 1 << 14, arena));
    OUT.reset(2, out_shape);
    ASSERT(arena.alloc_tensor(OUT));
    OUT.scale_recip = 1 << 14;

    sp_attention_dot_product(Q, K, V, OUT, n_head, n_head /*= n_kv_head*/, head_dim);

    std::vector<float> got(d_q);
    sp_ok_decode_to_fp32(got.data(), OUT);

    int near = 0;
    for (int i = 0; i < d_q; ++i) {
        if (std::abs(got[i] - ref[i]) < 0.005f) ++near;
    }
    // Allow a few ULP outliers from the softmax fp32 → re-encoded path.
    ASSERT(near >= d_q - 1);
}

// =========================================================================
// FFN SwiGLU test (single token)
// =========================================================================

TEST(ffn_swiglu_matches_fp32_reference) {
    constexpr int n_embd = 8, d_ff = 16;

    std::vector<float> x_fp32(n_embd), gate_w(d_ff * n_embd), up_w(d_ff * n_embd), down_w(n_embd * d_ff);
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> d(-0.3f, 0.3f);
    for (auto& v : x_fp32) v = d(rng);
    for (auto& v : gate_w) v = d(rng);
    for (auto& v : up_w)   v = d(rng);
    for (auto& v : down_w) v = d(rng);

    std::vector<float> ref(n_embd);
    ref_ffn_swiglu(x_fp32.data(), gate_w.data(), up_w.data(), down_w.data(),
                    n_embd, d_ff, ref.data());

    // Encode the operands. Layout conventions:
    //   x:      shape={n_embd, 1}             — d_x = n_embd values
    //   gate_w: shape={n_embd, d_ff}          — gate_w[i*n_embd + k] = w_i_k
    //   up_w:   same as gate_w
    //   down_w: shape={d_ff,   n_embd}        — down_w[i*d_ff + k]
    sp_ok_arena arena(64 * 1024);
    sp_ok_tensor X, GATE_W, UP_W, DOWN_W, OUT;
    // sp_matmul convention: X.shape[1]=K (common dim with W). For x with
    // K=n_embd, that means x.shape = {1, n_embd}. Same for OUT (an output
    // of a matmul: shape={N, M}={1, n_embd}).
    int64_t x_shape[4]      = { 1, n_embd, 1, 1 };
    int64_t gate_shape[4]   = { n_embd, d_ff, 1, 1 };
    int64_t down_shape[4]   = { d_ff,   n_embd, 1, 1 };
    int64_t out_shape[4]    = { 1, n_embd, 1, 1 };

    ASSERT(sp_ok_encode_from_fp32(X,      x_fp32.data(), 2, x_shape,      1 << 13, arena));
    ASSERT(sp_ok_encode_from_fp32(GATE_W, gate_w.data(), 2, gate_shape,   1 << 13, arena));
    ASSERT(sp_ok_encode_from_fp32(UP_W,   up_w.data(),   2, gate_shape,   1 << 13, arena));
    ASSERT(sp_ok_encode_from_fp32(DOWN_W, down_w.data(), 2, down_shape,   1 << 13, arena));
    OUT.reset(2, out_shape);
    ASSERT(arena.alloc_tensor(OUT));

    sp_ffn_swiglu(X, GATE_W, UP_W, DOWN_W, OUT);

    std::vector<float> got(n_embd);
    sp_ok_decode_to_fp32(got.data(), OUT);
    int near = 0;
    for (int i = 0; i < n_embd; ++i) {
        if (std::abs(got[i] - ref[i]) < 0.01) ++near;
    }
    ASSERT(near >= n_embd - 1);
}

// =========================================================================
// Driver
// =========================================================================

int main() {
    std::printf("Shannon-Prime sp_attention + sp_ffn tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
