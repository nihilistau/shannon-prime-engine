// Shannon-Prime Engine — forward-pass graph builder (stage 3a)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "forward.h"
#include "gguf_loader.h"
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

    // Cached hparams (pipe-in from Model at create time)
    int   n_embd    = 0;
    int   n_vocab   = 0;
    int   n_layer   = 0;
    int   n_head    = 0;
    int   n_head_kv = 0;
    int   head_dim  = 0;
    int   n_rot     = 0;          // RoPE dim count (often = head_dim)
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

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

int ForwardContext::n_embd()  const { return impl_->n_embd;  }
int ForwardContext::n_vocab() const { return impl_->n_vocab; }
int ForwardContext::n_layer() const { return impl_->n_layer; }

std::unique_ptr<ForwardContext> ForwardContext::create(const Model& model,
                                                        const LlamaWeights& weights,
                                                        int ctx_size_bytes) {
    if (!weights.tok_embd) {
        std::fprintf(stderr, "[sp-engine] ForwardContext: weights missing tok_embd\n");
        return nullptr;
    }

    auto fc = std::unique_ptr<ForwardContext>(new ForwardContext());
    fc->impl_->weights  = &weights;
    fc->impl_->ctx_size = (size_t)ctx_size_bytes;
    fc->impl_->ctx_mem.resize((size_t)ctx_size_bytes);

    // Pipe hparams from the Model into the compute context.
    fc->impl_->n_embd    = (int)model.n_embd();
    fc->impl_->n_vocab   = (int)(model.vocab_size() ? model.vocab_size() :
                                 weights.output ? weights.output->ne[1] : 0);
    fc->impl_->n_layer   = (int)model.n_layer();
    fc->impl_->n_head    = (int)model.n_head();
    fc->impl_->n_head_kv = (int)model.n_head_kv();
    fc->impl_->head_dim  = (int)model.head_dim();
    if (fc->impl_->head_dim == 0 && fc->impl_->n_head > 0) {
        fc->impl_->head_dim = fc->impl_->n_embd / fc->impl_->n_head;
    }
    // rope_dim_count: use arch-specific key when non-zero, else head_dim.
    fc->impl_->n_rot = (int)model.rope_dim_count();
    if (fc->impl_->n_rot == 0) fc->impl_->n_rot = fc->impl_->head_dim;
    fc->impl_->rope_freq_base = model.rope_freq_base();

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
// ------------------------------------------------------------------
// Shared per-block graph builder. Given an already-embedded hidden
// state `x` and the current RoPE position tensor `pos`, builds the
// attention + FFN block from `L` and returns the updated hidden state.
// Called once by forward_one_block and `n_layer` times by forward_full.
// ------------------------------------------------------------------
static ggml_tensor* build_block(ggml_context* gctx,
                                 ggml_tensor* x,
                                 ggml_tensor* pos,
                                 const LlamaLayer& L,
                                 int n,
                                 int head_dim,
                                 int n_head,
                                 int n_head_kv,
                                 int n_rot,
                                 float freq_base,
                                 float freq_scale) {
    const int n_embd_q = n_head * head_dim;

    // Attention pre-norm + projections.
    ggml_tensor* xa = ggml_rms_norm(gctx, x, 1e-5f);
    xa = ggml_mul(gctx, xa, L.attn_norm);

    ggml_tensor* Q = ggml_mul_mat(gctx, L.wq, xa);
    if (L.bq) Q = ggml_add(gctx, Q, L.bq);
    ggml_tensor* K = ggml_mul_mat(gctx, L.wk, xa);
    if (L.bk) K = ggml_add(gctx, K, L.bk);
    ggml_tensor* V = ggml_mul_mat(gctx, L.wv, xa);
    if (L.bv) V = ggml_add(gctx, V, L.bv);

    Q = ggml_reshape_3d(gctx, Q, head_dim, n_head,    n);
    K = ggml_reshape_3d(gctx, K, head_dim, n_head_kv, n);
    V = ggml_reshape_3d(gctx, V, head_dim, n_head_kv, n);

    if (L.attn_q_norm) {
        Q = ggml_rms_norm(gctx, Q, 1e-5f);
        Q = ggml_mul(gctx, Q, L.attn_q_norm);
    }
    if (L.attn_k_norm) {
        K = ggml_rms_norm(gctx, K, 1e-5f);
        K = ggml_mul(gctx, K, L.attn_k_norm);
    }

    Q = ggml_rope_ext(gctx, Q, pos, /*freq_factors=*/nullptr,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_ext(gctx, K, pos, /*freq_factors=*/nullptr,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);

    // GQA broadcast.
    if (n_head != n_head_kv) {
        const int n_rep = n_head / n_head_kv;
        ggml_tensor* Kx = ggml_reshape_4d(gctx, K, head_dim, 1, n_head_kv, n);
        ggml_tensor* Kt = ggml_new_tensor_4d(gctx, K->type, head_dim, n_rep, n_head_kv, n);
        Kx = ggml_repeat(gctx, Kx, Kt);
        K  = ggml_reshape_3d(gctx, Kx, head_dim, n_head, n);
        ggml_tensor* Vx = ggml_reshape_4d(gctx, V, head_dim, 1, n_head_kv, n);
        ggml_tensor* Vt = ggml_new_tensor_4d(gctx, V->type, head_dim, n_rep, n_head_kv, n);
        Vx = ggml_repeat(gctx, Vx, Vt);
        V  = ggml_reshape_3d(gctx, Vx, head_dim, n_head, n);
    }

    // Attention: Q·K^T / sqrt(d) + causal softmax.
    ggml_tensor* Qp = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
    ggml_tensor* Kp = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
    ggml_tensor* KQ = ggml_mul_mat(gctx, Kp, Qp);
    KQ = ggml_soft_max_ext(gctx, KQ, nullptr,
                           1.0f / sqrtf((float)head_dim), 0.0f);

    ggml_tensor* Vp = ggml_cont(gctx, ggml_permute(gctx, V, 1, 2, 0, 3));
    ggml_tensor* attn = ggml_mul_mat(gctx, Vp, KQ);
    attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(gctx, attn, n_embd_q, n);

    ggml_tensor* y1 = ggml_mul_mat(gctx, L.wo, attn);
    if (L.bo) y1 = ggml_add(gctx, y1, L.bo);

    x = ggml_add(gctx, x, y1);

    // FFN.
    ggml_tensor* xb = ggml_rms_norm(gctx, x, 1e-5f);
    xb = ggml_mul(gctx, xb, L.ffn_norm);

    ggml_tensor* gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
    ggml_tensor* up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    gate = ggml_silu(gctx, gate);
    ggml_tensor* ffn  = ggml_mul(gctx, gate, up);
    ffn  = ggml_mul_mat(gctx, L.ffn_down, ffn);

    x = ggml_add(gctx, x, ffn);
    return x;
}

bool ForwardContext::forward_one_block(const std::vector<int32_t>& token_ids,
                                        std::vector<float>& out_flat,
                                        int& out_n_embd) {
    if (token_ids.empty()) return false;
    auto* W = impl_->weights;
    if (W->layers().empty()) return false;

    const int n = (int)token_ids.size();
    out_n_embd = impl_->n_embd;

    ggml_init_params gip = {};
    gip.mem_size   = impl_->ctx_size;
    gip.mem_buffer = impl_->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) return false;

    ggml_tensor* ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_input(ids);
    ggml_tensor* pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);
    x = build_block(gctx, x, pos, W->layers()[0], n,
                     impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                     impl_->n_rot, impl_->rope_freq_base, impl_->rope_freq_scale);
    ggml_set_output(x);

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, x);
    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        ggml_free(gctx); return false;
    }

    ggml_backend_tensor_set(ids, token_ids.data(), 0, (size_t)n * sizeof(int32_t));
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = i;
    ggml_backend_tensor_set(pos, positions.data(), 0, (size_t)n * sizeof(int32_t));

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx); return false;
    }
    out_flat.resize((size_t)n * (size_t)impl_->n_embd);
    ggml_backend_tensor_get(x, out_flat.data(), 0, out_flat.size() * sizeof(float));
    ggml_free(gctx);
    return true;
}

// ------------------------------------------------------------------
// 3c: full forward — loop over all layers, output_norm + output head.
// Returns (n, n_vocab) logits flat.
// ------------------------------------------------------------------
bool ForwardContext::forward_full(const std::vector<int32_t>& token_ids,
                                   std::vector<float>& logits_flat,
                                   int& out_n_vocab) {
    if (token_ids.empty()) return false;
    auto* W = impl_->weights;
    if (W->layers().empty()) return false;

    const int n = (int)token_ids.size();
    out_n_vocab = impl_->n_vocab;

    ggml_init_params gip = {};
    gip.mem_size   = impl_->ctx_size;
    gip.mem_buffer = impl_->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) return false;

    ggml_tensor* ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_input(ids);
    ggml_tensor* pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n);
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);

    for (int i = 0; i < impl_->n_layer; ++i) {
        x = build_block(gctx, x, pos, W->layers()[(size_t)i], n,
                         impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                         impl_->n_rot, impl_->rope_freq_base, impl_->rope_freq_scale);
    }

    // Output norm + projection.
    ggml_tensor* h = ggml_rms_norm(gctx, x, 1e-5f);
    h = ggml_mul(gctx, h, W->output_norm);
    ggml_tensor* logits = ggml_mul_mat(gctx, W->output, h);
    ggml_set_output(logits);

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, logits);
    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        std::fprintf(stderr, "[sp-engine] forward_full: gallocr failed\n");
        ggml_free(gctx); return false;
    }

    ggml_backend_tensor_set(ids, token_ids.data(), 0, (size_t)n * sizeof(int32_t));
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = i;
    ggml_backend_tensor_set(pos, positions.data(), 0, (size_t)n * sizeof(int32_t));

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] forward_full: compute failed\n");
        ggml_free(gctx); return false;
    }

    logits_flat.resize((size_t)n * (size_t)impl_->n_vocab);
    ggml_backend_tensor_get(logits, logits_flat.data(), 0,
                            logits_flat.size() * sizeof(float));
    ggml_free(gctx);
    return true;
}

} // namespace sp::engine
