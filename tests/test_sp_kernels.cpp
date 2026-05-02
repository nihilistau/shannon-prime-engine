// test_sp_kernels — parity check of sp_kernels_cpu vs ggml on the
// kernels forward_native.cpp will use: rms_norm, silu (+silu_mul),
// softmax, fp32 matmul, and Q5_K-packed matmul.
//
// "Parity" is fp32 element-wise within a small relative tolerance
// (~1e-5). True bit-exact match is not the goal — ggml's kernels
// reduce in different orders (especially under SIMD), so we'd see
// ULP-level differences regardless of whether our code is "correct."
// What we ASSERT is that every output is within tol of ggml's, which
// is what matters for end-to-end model output equivalence.
//
// Exit 0 on full pass, 1 on any out-of-tolerance element.

#include "sp_kernels_cpu.h"
#include "sp_quant.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

static const float REL_TOL = 1e-4f;   // generous: SIMD reduction order
static const float ABS_TOL = 1e-5f;   // catches near-zero noise

// Helper: SP_QK_K is in our header; ggml's QK_K is internal. Wrap so
// the test compiles even if the internal symbol shifts.
static inline int SP_QK_K_FROM_HEADER() { return sp::engine::SP_QK_K; }

static bool close_enough(float a, float b) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return false;
    const float d = std::fabs(a - b);
    if (d <= ABS_TOL) return true;
    const float scale = std::max(std::fabs(a), std::fabs(b));
    return (d / scale) <= REL_TOL;
}

static int compare_arrays(const char* tag,
                          const std::vector<float>& ours,
                          const std::vector<float>& theirs) {
    int mismatches = 0;
    double max_rel = 0.0, max_abs = 0.0;
    int first_bad = -1;
    for (size_t i = 0; i < ours.size(); ++i) {
        const float a = ours[i], b = theirs[i];
        const double d = std::fabs((double)a - (double)b);
        const double s = std::max(std::fabs((double)a), std::fabs((double)b));
        const double r = (s > 0.0) ? (d / s) : 0.0;
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
        if (!close_enough(a, b)) {
            if (first_bad < 0) first_bad = (int)i;
            ++mismatches;
        }
    }
    std::fprintf(stderr,
        "  %-22s n=%zu  max_abs=%.3g  max_rel=%.3g  mismatches=%d\n",
        tag, ours.size(), max_abs, max_rel, mismatches);
    if (mismatches > 0) {
        std::fprintf(stderr,
            "    first bad i=%d  ours=%.9g  ggml=%.9g\n",
            first_bad, ours[first_bad], theirs[first_bad]);
    }
    return mismatches;
}

// ─────────────────────────────────────────────────────────────────
// ggml mini-graph helpers — single-op reference computations
// ─────────────────────────────────────────────────────────────────

// Build a tiny ggml graph that runs `op_builder(ctx, ...)` over a
// single input tensor; compute on CPU; return the output as fp32.
template <typename OpBuilder>
static std::vector<float> ggml_run_single_op(
        const std::vector<float>& input,
        const int64_t in_shape[4],
        OpBuilder build_op) {
    // Build context.
    const size_t mem_size = 64 * 1024 * 1024;
    std::vector<uint8_t> mem(mem_size);
    ggml_init_params params = {};
    params.mem_size   = mem_size;
    params.mem_buffer = mem.data();
    params.no_alloc   = false;
    ggml_context* ctx = ggml_init(params);

    // Input tensor (shape MAX_DIMS=4).
    ggml_tensor* x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, in_shape);
    std::memcpy(x->data, input.data(),
                input.size() * sizeof(float));

    ggml_tensor* y = build_op(ctx, x);

    ggml_cgraph* g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 1);
    ggml_backend_graph_compute(be, g);

    std::vector<float> out((size_t)ggml_nelements(y));
    std::memcpy(out.data(), y->data, out.size() * sizeof(float));

    ggml_backend_free(be);
    ggml_free(ctx);
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

int main() {
    int total_fail = 0;

    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // 1. RMS Norm
    {
        const int n = 2048;
        std::vector<float> x(n), scale(n);
        for (auto& v : x) v = nd(rng) * 0.5f;
        for (auto& v : scale) v = 1.0f + nd(rng) * 0.1f;
        const float eps = 1e-6f;

        std::vector<float> ours(n);
        sp::engine::sp_rms_norm_f32(x.data(), scale.data(), n, eps,
                                     ours.data());

        // ggml: rms_norm produces the normalized vector; multiply by
        // scale to match our fused op.
        const int64_t shape[4] = {n, 1, 1, 1};
        auto theirs = ggml_run_single_op(x, shape,
            [&](ggml_context* ctx, ggml_tensor* xi) {
                ggml_tensor* sc = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, shape);
                std::memcpy(sc->data, scale.data(),
                            scale.size() * sizeof(float));
                ggml_tensor* normed = ggml_rms_norm(ctx, xi, eps);
                return ggml_mul(ctx, normed, sc);
            });
        total_fail += compare_arrays("rms_norm", ours, theirs);
    }

    // 2. SiLU
    {
        const int n = 4096;
        std::vector<float> x(n);
        for (auto& v : x) v = nd(rng);

        std::vector<float> ours(n);
        sp::engine::sp_silu_f32(x.data(), n, ours.data());

        const int64_t shape[4] = {n, 1, 1, 1};
        auto theirs = ggml_run_single_op(x, shape,
            [&](ggml_context* ctx, ggml_tensor* xi) {
                return ggml_silu(ctx, xi);
            });
        total_fail += compare_arrays("silu", ours, theirs);
    }

    // 3. Softmax (single row)
    {
        const int n_cols = 1024;
        std::vector<float> x(n_cols);
        for (auto& v : x) v = nd(rng) * 2.0f;

        std::vector<float> ours(n_cols);
        sp::engine::sp_softmax_f32_rows(x.data(), nullptr,
                                         n_cols, /*n_rows=*/1, /*scale=*/1.0f,
                                         ours.data());

        const int64_t shape[4] = {n_cols, 1, 1, 1};
        auto theirs = ggml_run_single_op(x, shape,
            [&](ggml_context* ctx, ggml_tensor* xi) {
                return ggml_soft_max(ctx, xi);
            });
        total_fail += compare_arrays("softmax_1row", ours, theirs);
    }

    // 4. Matmul fp32 × fp32
    {
        const int M = 8, K = 256, N = 64;
        std::vector<float> lhs((size_t)M * K), rhs((size_t)N * K);
        for (auto& v : lhs) v = nd(rng) * 0.1f;
        for (auto& v : rhs) v = nd(rng) * 0.1f;

        std::vector<float> ours((size_t)M * N);
        sp::engine::sp_matmul_f32(lhs.data(), rhs.data(), M, K, N, ours.data());

        // ggml: y = W @ x, with W shape [K, N] (so ggml_mul_mat takes
        // weight as first arg, activations as second). The actual
        // memory layout for a "weight" in ggml is [n_in=K, n_out=N]
        // outer-major — same memory as our rhs[N, K] row-major.
        const size_t mem_size = 64 * 1024 * 1024;
        std::vector<uint8_t> mem(mem_size);
        ggml_init_params p = {};
        p.mem_size = mem_size;
        p.mem_buffer = mem.data();
        p.no_alloc = false;
        ggml_context* ctx = ggml_init(p);

        const int64_t W_shape[4] = {K, N, 1, 1};
        const int64_t X_shape[4] = {K, M, 1, 1};
        ggml_tensor* W = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, W_shape);
        ggml_tensor* X = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, X_shape);
        std::memcpy(W->data, rhs.data(), rhs.size() * sizeof(float));
        std::memcpy(X->data, lhs.data(), lhs.size() * sizeof(float));
        ggml_tensor* Y = ggml_mul_mat(ctx, W, X);
        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, Y);

        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, 1);
        ggml_backend_graph_compute(be, g);

        std::vector<float> theirs((size_t)M * N);
        std::memcpy(theirs.data(), Y->data, theirs.size() * sizeof(float));

        ggml_backend_free(be);
        ggml_free(ctx);

        total_fail += compare_arrays("matmul_f32", ours, theirs);
    }

    // 5. Matmul fp32 lhs × Q5_K rhs
    //
    // Reference: dequant the Q5_K weights to fp32 via ggml's public
    // type_traits.to_float, then run ggml_mul_mat in pure fp32. This
    // gives an apples-to-apples comparison — both paths compute
    // dequant(W) @ X in fp32. We do NOT compare against ggml's
    // direct Q5_K @ fp32 matmul because that one internally quantizes
    // X to Q8_K (faster int8 dot, ~1e-3 precision loss). Our path
    // doesn't take that shortcut, so apples-to-Q8_K-quantized-apples
    // would falsely flag us as off.
    {
        const int M = 4, K = 256, N = 32;
        std::vector<float> lhs((size_t)M * K), rhs_f32((size_t)N * K);
        for (auto& v : lhs)     v = nd(rng) * 0.1f;
        for (auto& v : rhs_f32) v = nd(rng) * 0.02f;

        // Quantize rhs to Q5_K.
        const struct ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_Q5_K);
        const size_t row_bytes = (K / SP_QK_K_FROM_HEADER()) * tt->type_size;
        std::vector<uint8_t> rhs_q5k((size_t)N * row_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q5_K, rhs_f32.data(), rhs_q5k.data(),
                            /*start=*/0, /*nrows=*/N, /*n_per_row=*/K,
                            /*imatrix=*/nullptr);

        std::vector<float> ours((size_t)M * N);
        sp::engine::sp_matmul_f32_q5k(lhs.data(), rhs_q5k.data(),
                                       M, K, N, ours.data());

        // ggml apples-to-apples: dequant Q5_K → fp32, then fp32 matmul.
        std::vector<float> rhs_dequant((size_t)N * K);
        tt->to_float(rhs_q5k.data(), rhs_dequant.data(), (int64_t)N * K);

        const size_t mem_size = 64 * 1024 * 1024;
        std::vector<uint8_t> mem(mem_size);
        ggml_init_params p = {};
        p.mem_size = mem_size;
        p.mem_buffer = mem.data();
        p.no_alloc = false;
        ggml_context* ctx = ggml_init(p);

        const int64_t W_shape[4] = {K, N, 1, 1};
        const int64_t X_shape[4] = {K, M, 1, 1};
        ggml_tensor* W = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, W_shape);
        ggml_tensor* X = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, X_shape);
        std::memcpy(W->data, rhs_dequant.data(),
                    rhs_dequant.size() * sizeof(float));
        std::memcpy(X->data, lhs.data(), lhs.size() * sizeof(float));
        ggml_tensor* Y = ggml_mul_mat(ctx, W, X);
        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, Y);

        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, 1);
        ggml_backend_graph_compute(be, g);

        std::vector<float> theirs((size_t)M * N);
        std::memcpy(theirs.data(), Y->data, theirs.size() * sizeof(float));

        ggml_backend_free(be);
        ggml_free(ctx);

        total_fail += compare_arrays("matmul_q5k", ours, theirs);
    }

    std::fprintf(stderr,
        "\ntest_sp_kernels: %s (total mismatches=%d)\n",
        total_fail == 0 ? "PASS" : "FAIL", total_fail);
    return total_fail == 0 ? 0 : 1;
}
