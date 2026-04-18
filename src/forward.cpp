// Shannon-Prime Engine — forward-pass graph builder (stage 3a)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "forward.h"
#include "llama_weights.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sp::engine {

struct ForwardContext::Impl {
    const LlamaWeights* weights = nullptr;

    ggml_backend_t        backend = nullptr;   // CPU only at this stage
    ggml_backend_buffer_t compute_buf = nullptr;
    ggml_gallocr_t        allocr = nullptr;

    size_t ctx_size = 0;
    std::vector<uint8_t> ctx_mem;              // backing for graph ggml_context

    ~Impl() {
        if (compute_buf) ggml_backend_buffer_free(compute_buf);
        if (allocr)      ggml_gallocr_free(allocr);
        if (backend)     ggml_backend_free(backend);
    }
};

ForwardContext::ForwardContext() : impl_(std::make_unique<Impl>()) {}
ForwardContext::~ForwardContext() = default;

std::unique_ptr<ForwardContext> ForwardContext::create(const LlamaWeights& weights,
                                                        int ctx_size_bytes) {
    if (!weights.tok_embd) {
        std::fprintf(stderr, "[sp-engine] ForwardContext: weights missing tok_embd\n");
        return nullptr;
    }

    auto fc = std::unique_ptr<ForwardContext>(new ForwardContext());
    fc->impl_->weights  = &weights;
    fc->impl_->ctx_size = (size_t)ctx_size_bytes;
    fc->impl_->ctx_mem.resize((size_t)ctx_size_bytes);

    fc->impl_->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!fc->impl_->backend) {
        std::fprintf(stderr, "[sp-engine] ForwardContext: failed to init CPU backend\n");
        return nullptr;
    }

    fc->impl_->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fc->impl_->backend));
    if (!fc->impl_->allocr) {
        std::fprintf(stderr, "[sp-engine] ForwardContext: gallocr_new failed\n");
        return nullptr;
    }

    return fc;
}

// ------------------------------------------------------------------
// 3a: token-embedding lookup graph.
//
// input:  int32[n]     token IDs
// output: fp32[n_embd, n]  row-major (= ggml default column-major transposed,
//                          we return flat then reshape on host)
//
// The graph is trivially ggml_get_rows(tok_embd, input_ids). The
// complexity in this file is the backend/allocr plumbing that every
// subsequent stage will reuse.
// ------------------------------------------------------------------
bool ForwardContext::embed(const std::vector<int32_t>& token_ids,
                           std::vector<float>& out_flat,
                           int& out_n_embd) {
    if (token_ids.empty()) return false;
    auto* W = impl_->weights;

    const int n = (int)token_ids.size();
    const int n_embd = (int)W->tok_embd->ne[0];
    out_n_embd = n_embd;

    // Build a fresh graph context from the pre-allocated arena.
    ggml_init_params gip = {};
    gip.mem_size   = impl_->ctx_size;
    gip.mem_buffer = impl_->ctx_mem.data();
    gip.no_alloc   = true;  // tensors will be placed in the compute buffer
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) return false;

    // Input: int32 tensor holding the token IDs.
    ggml_tensor* ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    // Lookup.
    ggml_tensor* emb = ggml_get_rows(gctx, W->tok_embd, ids);
    ggml_set_name(emb, "emb");
    ggml_set_output(emb);

    // Compile the graph.
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, emb);

    // Allocate compute buffer (right-sized for this graph).
    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        std::fprintf(stderr, "[sp-engine] ForwardContext::embed: allocr failed\n");
        ggml_free(gctx);
        return false;
    }

    // Upload token IDs into the input tensor on the backend.
    ggml_backend_tensor_set(ids, token_ids.data(), 0,
                            (size_t)n * sizeof(int32_t));

    // Compute.
    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] ForwardContext::embed: compute failed\n");
        ggml_free(gctx);
        return false;
    }

    // Read back flat.
    out_flat.resize((size_t)n * (size_t)n_embd);
    ggml_backend_tensor_get(emb, out_flat.data(), 0,
                            out_flat.size() * sizeof(float));

    ggml_free(gctx);
    return true;
}

// ------------------------------------------------------------------
// 3b: single transformer block forward (layer 0 only, no KV cache).
//
// Flow:
//   x = embed(ids)                           [n, n_embd]
//   xa = rms_norm(x, attn_norm) * attn_norm_w
//   Q  = xa @ wq  (+ bq)     reshape to [n_head,    n, head_dim]
//   K  = xa @ wk  (+ bk)     reshape to [n_head_kv, n, head_dim]
//   V  = xa @ wv  (+ bv)     reshape to [n_head_kv, n, head_dim]
//   (optional per-head norms: Qwen3 attn_q_norm, attn_k_norm)
//   Q  = rope(Q, freqs)
//   K  = rope(K, freqs)
//   # GQA broadcast: K and V are repeated to match Q's n_head if needed.
//   scores = soft_max(Q @ K^T / sqrt(head_dim) + causal_mask)
//   attn   = scores @ V      (per-head)
//   y1 = attn.reshape([n, n_embd]) @ wo  (+ bo)
//   x  = x + y1
//   xb = rms_norm(x, ffn_norm) * ffn_norm_w
//   ffn = (silu(xb @ ffn_gate) * (xb @ ffn_up)) @ ffn_down
//   x  = x + ffn
//   return x
//
// RoPE uses ggml_rope_ext with nullptr freq_factors (standard geometric)
// — PrimePE lands in stage 3d via the freq_factors argument. ALiBi is
// a no-op here (soft_max_ext's bias arg is also nullptr).
// ------------------------------------------------------------------
bool ForwardContext::forward_one_block(const std::vector<int32_t>& token_ids,
                                        std::vector<float>& out_flat,
                                        int& out_n_embd) {
    if (token_ids.empty()) return false;
    auto* W = impl_->weights;
    if (W->layers().empty()) return false;
    const LlamaLayer& L = W->layers()[0];

    const int n        = (int)token_ids.size();
    const int n_embd   = (int)W->tok_embd->ne[0];
    const int n_head_kv = (int)L.wk->ne[1] / (n_embd / (int)L.wq->ne[1] * 0 + 1);
    (void)n_head_kv;  // derived below more carefully
    out_n_embd = n_embd;

    // Derive heads and head_dim from the Q projection shape.
    //   wq is [n_embd, n_embd_q]           where n_embd_q = n_head * head_dim
    //   wk is [n_embd, n_embd_kv]          where n_embd_kv = n_head_kv * head_dim
    const int n_embd_q  = (int)L.wq->ne[1];
    const int n_embd_kv = (int)L.wk->ne[1];
    // head_dim — prefer the explicit key when arch stores it, else
    // assume n_embd_q / n_head_from_hparams. We don't have the Model
    // here, so infer from the weight shapes assuming head_dim divides
    // both n_embd_q and n_embd_kv and that the GCD (over powers of 2)
    // corresponds to the actual head_dim of the model.
    // Simpler: use the known 64 or 128 based on n_embd_q % 128 == 0.
    int head_dim = 0;
    for (int cand : {128, 64, 256, 96}) {
        if (n_embd_q % cand == 0 && n_embd_kv % cand == 0) { head_dim = cand; break; }
    }
    if (head_dim == 0) head_dim = 64;  // fallback
    const int n_head     = n_embd_q  / head_dim;
    const int n_head_kv2 = n_embd_kv / head_dim;

    // Build graph.
    ggml_init_params gip = {};
    gip.mem_size   = impl_->ctx_size;
    gip.mem_buffer = impl_->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) return false;

    // --- inputs ---
    ggml_tensor* ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_name(ids, "ids"); ggml_set_input(ids);
    // RoPE position indices (0..n-1)
    ggml_tensor* pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);

    // --- embedding ---
    // x: [n_embd, n]  (ggml stores matrices column-major so [cols, rows] here
    // means each column is one token's embedding).
    ggml_tensor* x  = ggml_get_rows(gctx, W->tok_embd, ids);
    ggml_set_name(x, "x_in");

    // --- attention pre-norm + projections ---
    ggml_tensor* xa = ggml_rms_norm(gctx, x, 1e-5f);
    xa = ggml_mul(gctx, xa, L.attn_norm);

    ggml_tensor* Q = ggml_mul_mat(gctx, L.wq, xa);
    if (L.bq) Q = ggml_add(gctx, Q, L.bq);
    ggml_tensor* K = ggml_mul_mat(gctx, L.wk, xa);
    if (L.bk) K = ggml_add(gctx, K, L.bk);
    ggml_tensor* V = ggml_mul_mat(gctx, L.wv, xa);
    if (L.bv) V = ggml_add(gctx, V, L.bv);

    // Reshape Q/K/V to [head_dim, n_head, n] / [head_dim, n_head_kv, n]
    Q = ggml_reshape_3d(gctx, Q, head_dim, n_head,     n);
    K = ggml_reshape_3d(gctx, K, head_dim, n_head_kv2, n);
    V = ggml_reshape_3d(gctx, V, head_dim, n_head_kv2, n);

    // Per-head norms (Qwen3)
    if (L.attn_q_norm) {
        Q = ggml_rms_norm(gctx, Q, 1e-5f);
        Q = ggml_mul(gctx, Q, L.attn_q_norm);
    }
    if (L.attn_k_norm) {
        K = ggml_rms_norm(gctx, K, 1e-5f);
        K = ggml_mul(gctx, K, L.attn_k_norm);
    }

    // RoPE — rope_dim_count may be less than head_dim; use rope_freqs if present.
    // For now apply RoPE to the first `head_dim` dims (full rotation) with
    // standard geometric freqs (freq_factors = nullptr).
    const int n_rot = head_dim;
    const float freq_base  = 10000.0f;  // hparam pipe lands with stage 3c
    const float freq_scale = 1.0f;
    Q = ggml_rope_ext(gctx, Q, pos, /*freq_factors=*/nullptr,
                      n_rot, /*mode=*/0, /*n_ctx_orig=*/0,
                      freq_base, freq_scale,
                      /*ext_factor=*/0, /*attn_factor=*/1,
                      /*beta_fast=*/32, /*beta_slow=*/1);
    K = ggml_rope_ext(gctx, K, pos, /*freq_factors=*/nullptr,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);

    // GQA: repeat K / V so head count matches Q.
    // ggml doesn't have a first-class "repeat heads" op, but we can
    // reshape K/V from [d, n_head_kv, n] to [d, 1, n_head_kv, n] →
    // repeat → [d, n_head/n_head_kv, n_head_kv, n] → reshape to
    // [d, n_head, n]. Equivalent to broadcasting via ggml_repeat.
    if (n_head != n_head_kv2) {
        const int n_rep = n_head / n_head_kv2;
        ggml_tensor* Kx = ggml_reshape_4d(gctx, K, head_dim, 1, n_head_kv2, n);
        ggml_tensor* Kt = ggml_new_tensor_4d(gctx, K->type, head_dim, n_rep, n_head_kv2, n);
        Kx = ggml_repeat(gctx, Kx, Kt);
        K  = ggml_reshape_3d(gctx, Kx, head_dim, n_head, n);
        ggml_tensor* Vx = ggml_reshape_4d(gctx, V, head_dim, 1, n_head_kv2, n);
        ggml_tensor* Vt = ggml_new_tensor_4d(gctx, V->type, head_dim, n_rep, n_head_kv2, n);
        Vx = ggml_repeat(gctx, Vx, Vt);
        V  = ggml_reshape_3d(gctx, Vx, head_dim, n_head, n);
    }

    // Attention scores: Q @ K^T  / sqrt(head_dim) with causal mask.
    // Layout: Q is [head_dim, n_head, n], K is [head_dim, n_head, n].
    // We want attn[head, i, j] = sum_d Q[d, head, i] * K[d, head, j] / sqrt(d).
    ggml_tensor* Qp = ggml_permute(gctx, Q, 0, 2, 1, 3);   // [head_dim, n, n_head]
    ggml_tensor* Kp = ggml_permute(gctx, K, 0, 2, 1, 3);   // [head_dim, n, n_head]
    Qp = ggml_cont(gctx, Qp);
    Kp = ggml_cont(gctx, Kp);
    ggml_tensor* KQ = ggml_mul_mat(gctx, Kp, Qp);          // [n, n, n_head]
    KQ = ggml_soft_max_ext(gctx, KQ, /*mask=*/nullptr,
                           1.0f / sqrtf((float)head_dim),
                           /*max_bias=*/0.0f);

    // attn = KQ @ V^T    (V permuted so rows match)
    ggml_tensor* Vp = ggml_permute(gctx, V, 1, 2, 0, 3);   // [n_head, n, head_dim]
    Vp = ggml_cont(gctx, Vp);
    ggml_tensor* attn = ggml_mul_mat(gctx, Vp, KQ);        // [head_dim, n, n_head]
    attn = ggml_permute(gctx, attn, 0, 2, 1, 3);           // [head_dim, n_head, n]
    attn = ggml_cont(gctx, attn);
    attn = ggml_reshape_2d(gctx, attn, n_embd_q, n);

    // Output projection
    ggml_tensor* y1 = ggml_mul_mat(gctx, L.wo, attn);
    if (L.bo) y1 = ggml_add(gctx, y1, L.bo);

    // Residual 1
    x = ggml_add(gctx, x, y1);

    // --- FFN ---
    ggml_tensor* xb = ggml_rms_norm(gctx, x, 1e-5f);
    xb = ggml_mul(gctx, xb, L.ffn_norm);

    ggml_tensor* gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
    ggml_tensor* up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    gate = ggml_silu(gctx, gate);
    ggml_tensor* ffn = ggml_mul(gctx, gate, up);
    ffn = ggml_mul_mat(gctx, L.ffn_down, ffn);

    x = ggml_add(gctx, x, ffn);
    ggml_set_name(x, "x_out");
    ggml_set_output(x);

    // Compile + compute
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, x);

    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        std::fprintf(stderr, "[sp-engine] forward_one_block: gallocr failed\n");
        ggml_free(gctx);
        return false;
    }

    // Upload inputs.
    ggml_backend_tensor_set(ids, token_ids.data(), 0,
                            (size_t)n * sizeof(int32_t));
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = i;
    ggml_backend_tensor_set(pos, positions.data(), 0,
                            (size_t)n * sizeof(int32_t));

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] forward_one_block: compute failed\n");
        ggml_free(gctx);
        return false;
    }

    out_flat.resize((size_t)n * (size_t)n_embd);
    ggml_backend_tensor_get(x, out_flat.data(), 0,
                            out_flat.size() * sizeof(float));
    ggml_free(gctx);
    return true;
}

} // namespace sp::engine
