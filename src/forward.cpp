// Shannon-Prime Engine — forward-pass graph builder (stage 3a)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "forward.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "prime_pe.h"

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

    // PrimePE-RoPE-ALiBi precomputed values. `freq_factors_vec` is
    // empty when PE is Standard/AlibiOnly (ggml_rope_ext gets nullptr);
    // `alibi_max_bias` is 0 unless an ALiBi mode is selected.
    PeSettings         pe;
    std::vector<float> freq_factors_vec;
    float              alibi_max_bias = 0.0f;

    ggml_backend_t        backend = nullptr;   // CPU only at this stage
    ggml_backend_buffer_t compute_buf = nullptr;
    ggml_gallocr_t        allocr = nullptr;

    size_t ctx_size = 0;
    std::vector<uint8_t> ctx_mem;              // backing for graph ggml_context

    // Stage 5b: stateful session over a bound (non-owning) KvCache.
    KvCache* cache  = nullptr;
    int      kv_pos = 0;

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

void ForwardContext::bind_cache(KvCache* cache) {
    impl_->cache  = cache;
    impl_->kv_pos = 0;
}

int ForwardContext::kv_pos() const { return impl_->kv_pos; }

std::unique_ptr<ForwardContext> ForwardContext::create(const Model& model,
                                                        const LlamaWeights& weights,
                                                        int ctx_size_bytes,
                                                        PeSettings pe) {
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

    // PrimePE precompute. One allocation per ForwardContext; re-uploaded
    // into each graph's freq_factors input tensor.
    fc->impl_->pe               = pe;
    fc->impl_->freq_factors_vec = prime_pe_freq_factors(pe.pe_mode, pe.pe_alpha,
                                                        pe.pe_tier, fc->impl_->n_rot,
                                                        fc->impl_->rope_freq_base);
    fc->impl_->alibi_max_bias   = prime_pe_alibi_max_bias(pe.pe_mode, pe.pe_alpha);

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
                                 ggml_tensor* kq_mask,       // [n_kv, n_q] fp32
                                 ggml_tensor* freq_factors,  // may be nullptr
                                 float        alibi_max_bias,
                                 const LlamaLayer& L,
                                 int n,
                                 int head_dim,
                                 int n_head,
                                 int n_head_kv,
                                 int n_rot,
                                 float freq_base,
                                 float freq_scale,
                                 // Optional captures: if non-null, the
                                 // post-RoPE pre-GQA-broadcast K and V
                                 // tensors are returned through these.
                                 // Caller marks them as outputs (so the
                                 // gallocr keeps them addressable) and
                                 // reads them after compute.
                                 ggml_tensor** k_capture = nullptr,
                                 ggml_tensor** v_capture = nullptr) {
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

    Q = ggml_rope_ext(gctx, Q, pos, freq_factors,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_ext(gctx, K, pos, freq_factors,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);

    // V is a ggml_reshape_3d view over the projection output. ggml_set_output
    // on a view does not preserve the underlying buffer through subsequent
    // ops (the gallocr can repurpose the source buffer once the view is
    // "consumed" by the next reshape). Materialise V via ggml_cont and
    // reuse the materialised tensor downstream so the gallocr sees a real
    // buffer-owning node on the compute path.
    V = ggml_cont(gctx, V);

    if (k_capture) *k_capture = K;
    if (v_capture) *v_capture = V;

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
    KQ = ggml_soft_max_ext(gctx, KQ, kq_mask,
                           1.0f / sqrtf((float)head_dim), alibi_max_bias);

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

    ggml_tensor* freq_factors = nullptr;
    if (!impl_->freq_factors_vec.empty()) {
        freq_factors = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                          (int64_t)impl_->freq_factors_vec.size());
        ggml_set_input(freq_factors);
    }

    // Causal mask [n_kv=n, n_q=n] fp32. Fixes self-attention's
    // bidirectional bug from stage 3b/3c AND carries ALiBi position
    // offsets when max_bias > 0 (slope * mask added to scores).
    ggml_tensor* kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
    ggml_set_input(kq_mask);

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);
    x = build_block(gctx, x, pos, kq_mask, freq_factors, impl_->alibi_max_bias,
                     W->layers()[0], n,
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
    if (freq_factors) {
        ggml_backend_tensor_set(freq_factors, impl_->freq_factors_vec.data(), 0,
                                impl_->freq_factors_vec.size() * sizeof(float));
    }
    // Causal mask: element [kv, q] (row-major, column = fastest) is
    // -INF if kv > q, else the ALiBi distance (-(q-kv)) when ALiBi is
    // on, else 0. With max_bias=0 ggml uses slope=1 on the mask, so
    // the 0/−INF convention degenerates to the usual causal mask.
    {
        std::vector<float> mask((size_t)n * (size_t)n);
        const bool alibi = (impl_->alibi_max_bias > 0.0f);
        for (int q = 0; q < n; ++q) {
            for (int kv = 0; kv < n; ++kv) {
                float v;
                if (kv > q)      v = -INFINITY;
                else if (alibi)  v = -(float)(q - kv);
                else             v = 0.0f;
                mask[(size_t)q * n + kv] = v;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

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
//
// Stage 5a: when `per_layer_K` / `per_layer_V` are non-null, the
// post-RoPE pre-GQA K/V for every layer are pulled out after compute
// and copied into the host vectors in [head_dim, n_head_kv, n] order.
// This is what KvCache::write expects.
// ------------------------------------------------------------------
bool ForwardContext::forward_full(const std::vector<int32_t>& token_ids,
                                   std::vector<float>& logits_flat,
                                   int& out_n_vocab,
                                   std::vector<std::vector<float>>* per_layer_K,
                                   std::vector<std::vector<float>>* per_layer_V,
                                   std::vector<float>* dbg_X_layer0) {
    if (token_ids.empty()) return false;
    auto* W = impl_->weights;
    if (W->layers().empty()) return false;

    const int n = (int)token_ids.size();
    out_n_vocab = impl_->n_vocab;
    const bool capture = (per_layer_K && per_layer_V);
    if (capture) {
        per_layer_K->assign((size_t)impl_->n_layer, std::vector<float>{});
        per_layer_V->assign((size_t)impl_->n_layer, std::vector<float>{});
    }

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

    ggml_tensor* freq_factors = nullptr;
    if (!impl_->freq_factors_vec.empty()) {
        freq_factors = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                          (int64_t)impl_->freq_factors_vec.size());
        ggml_set_input(freq_factors);
    }
    ggml_tensor* kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
    ggml_set_input(kq_mask);

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);

    std::vector<ggml_tensor*> cap_K, cap_V;
    if (capture) {
        cap_K.assign((size_t)impl_->n_layer, nullptr);
        cap_V.assign((size_t)impl_->n_layer, nullptr);
    }

    ggml_tensor* x_layer0 = nullptr;
    for (int i = 0; i < impl_->n_layer; ++i) {
        ggml_tensor* k_cap = nullptr;
        ggml_tensor* v_cap = nullptr;
        x = build_block(gctx, x, pos, kq_mask, freq_factors, impl_->alibi_max_bias,
                         W->layers()[(size_t)i], n,
                         impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                         impl_->n_rot, impl_->rope_freq_base, impl_->rope_freq_scale,
                         capture ? &k_cap : nullptr,
                         capture ? &v_cap : nullptr);
        if (capture) {
            ggml_set_output(k_cap);
            ggml_set_output(v_cap);
            cap_K[(size_t)i] = k_cap;
            cap_V[(size_t)i] = v_cap;
        }
        if (i == 0 && dbg_X_layer0) {
            x_layer0 = x;
            ggml_set_output(x_layer0);
        }
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
    if (freq_factors) {
        ggml_backend_tensor_set(freq_factors, impl_->freq_factors_vec.data(), 0,
                                impl_->freq_factors_vec.size() * sizeof(float));
    }
    {
        std::vector<float> mask((size_t)n * (size_t)n);
        const bool alibi = (impl_->alibi_max_bias > 0.0f);
        for (int q = 0; q < n; ++q) {
            for (int kv = 0; kv < n; ++kv) {
                float v;
                if (kv > q)      v = -INFINITY;
                else if (alibi)  v = -(float)(q - kv);
                else             v = 0.0f;
                mask[(size_t)q * n + kv] = v;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] forward_full: compute failed\n");
        ggml_free(gctx); return false;
    }

    logits_flat.resize((size_t)n * (size_t)impl_->n_vocab);
    ggml_backend_tensor_get(logits, logits_flat.data(), 0,
                            logits_flat.size() * sizeof(float));

    // Pull per-layer K/V back to host for the KvCache write (stage 5a).
    if (capture) {
        const size_t kv_elems = (size_t)n * impl_->n_head_kv * impl_->head_dim;
        for (int i = 0; i < impl_->n_layer; ++i) {
            (*per_layer_K)[(size_t)i].resize(kv_elems);
            (*per_layer_V)[(size_t)i].resize(kv_elems);
            ggml_backend_tensor_get(cap_K[(size_t)i],
                                    (*per_layer_K)[(size_t)i].data(),
                                    0, kv_elems * sizeof(float));
            ggml_backend_tensor_get(cap_V[(size_t)i],
                                    (*per_layer_V)[(size_t)i].data(),
                                    0, kv_elems * sizeof(float));
        }
    }

    if (x_layer0 && dbg_X_layer0) {
        const size_t nbytes = (size_t)n * impl_->n_embd * sizeof(float);
        dbg_X_layer0->resize((size_t)n * impl_->n_embd);
        ggml_backend_tensor_get(x_layer0, dbg_X_layer0->data(), 0, nbytes);
    }

    ggml_free(gctx);
    return true;
}

// ------------------------------------------------------------------
// Stage 5b — single-token decode reading past K/V from a bound cache.
//
// Differs from build_block in two places:
//  * K/V projections produce only n=1 vectors. The freshly-projected
//    post-RoPE K (and post-projection V) is captured for write-back.
//  * Past K/V come in as graph inputs of shape [head_dim, n_head_kv,
//    past_n], populated from KvCache::read each step. We concat past
//    + new along the position axis, then GQA-broadcast the union.
//
// Mask: for non-ALiBi attention with a single query, no causal mask
// is needed (the new token attends to itself + every past token).
// soft_max_ext accepts a nullptr mask when max_bias = 0. For ALiBi
// modes, we build a 1×kv_total mask of -(kv_pos - kv) values.
//
// Note on the V capture: V at the projection point is a
// `ggml_reshape_3d` view over the mul_mat output and does not own a
// buffer. ggml_set_output on a view does not pin the underlying
// source buffer through subsequent ops, so the gallocr would
// repurpose it during the GQA broadcast / concat — corrupting both
// the captured V and the V data the attention op actually consumes.
// We materialise V via ggml_cont once and reuse the materialised
// tensor for both capture and the downstream concat / GQA path; the
// gallocr then sees one real buffer-owning node on the compute path
// and everything stays put. K does not need this because it comes
// out of ggml_rope_ext, which is a real op output (own buffer).
// ------------------------------------------------------------------
static ggml_tensor* build_block_decode(ggml_context* gctx,
                                        ggml_tensor* x,
                                        ggml_tensor* pos,
                                        ggml_tensor* past_K,
                                        ggml_tensor* past_V,
                                        ggml_tensor* mask,        // may be nullptr
                                        ggml_tensor* freq_factors,
                                        float        alibi_max_bias,
                                        const LlamaLayer& L,
                                        int past_n,
                                        int head_dim,
                                        int n_head,
                                        int n_head_kv,
                                        int n_rot,
                                        float freq_base,
                                        float freq_scale,
                                        ggml_tensor** k_capture,
                                        ggml_tensor** v_capture) {
    const int n_embd_q = n_head * head_dim;
    const int n        = 1;

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

    Q = ggml_rope_ext(gctx, Q, pos, freq_factors,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_ext(gctx, K, pos, freq_factors,
                      n_rot, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);

    // V at this point is a ggml_reshape_3d view over the projection
    // output. Materialise it once via ggml_cont and reuse for both the
    // capture and the downstream concat — that way the gallocr sees a
    // single buffer-owning node on the compute path. K is the output of
    // ggml_rope_ext, already a real op result, so no cont needed.
    V = ggml_cont(gctx, V);

    // Capture for cache write-back BEFORE the concat; what the cache
    // stores is the new single-token K/V, not the union.
    if (k_capture) *k_capture = K;
    if (v_capture) *v_capture = V;

    // Concat past + new along the position axis.
    ggml_tensor* K_full = K;
    ggml_tensor* V_full = V;
    if (past_n > 0) {
        K_full = ggml_concat(gctx, past_K, K, /*dim=*/2);
        V_full = ggml_concat(gctx, past_V, V, /*dim=*/2);
    }
    const int kv_total = past_n + 1;

    if (n_head != n_head_kv) {
        const int n_rep = n_head / n_head_kv;
        ggml_tensor* Kx = ggml_reshape_4d(gctx, K_full, head_dim, 1, n_head_kv, kv_total);
        ggml_tensor* Kt = ggml_new_tensor_4d(gctx, K_full->type, head_dim, n_rep, n_head_kv, kv_total);
        Kx = ggml_repeat(gctx, Kx, Kt);
        K_full = ggml_reshape_3d(gctx, Kx, head_dim, n_head, kv_total);
        ggml_tensor* Vx = ggml_reshape_4d(gctx, V_full, head_dim, 1, n_head_kv, kv_total);
        ggml_tensor* Vt = ggml_new_tensor_4d(gctx, V_full->type, head_dim, n_rep, n_head_kv, kv_total);
        Vx = ggml_repeat(gctx, Vx, Vt);
        V_full = ggml_reshape_3d(gctx, Vx, head_dim, n_head, kv_total);
    }

    ggml_tensor* Qp = ggml_cont(gctx, ggml_permute(gctx, Q,      0, 2, 1, 3));
    ggml_tensor* Kp = ggml_cont(gctx, ggml_permute(gctx, K_full, 0, 2, 1, 3));
    ggml_tensor* KQ = ggml_mul_mat(gctx, Kp, Qp);
    KQ = ggml_soft_max_ext(gctx, KQ, mask,
                           1.0f / sqrtf((float)head_dim), alibi_max_bias);

    ggml_tensor* Vp = ggml_cont(gctx, ggml_permute(gctx, V_full, 1, 2, 0, 3));
    ggml_tensor* attn = ggml_mul_mat(gctx, Vp, KQ);
    attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(gctx, attn, n_embd_q, n);

    ggml_tensor* y1 = ggml_mul_mat(gctx, L.wo, attn);
    if (L.bo) y1 = ggml_add(gctx, y1, L.bo);

    x = ggml_add(gctx, x, y1);

    ggml_tensor* xb = ggml_rms_norm(gctx, x, 1e-5f);
    xb = ggml_mul(gctx, xb, L.ffn_norm);
    ggml_tensor* gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
    ggml_tensor* up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    gate = ggml_silu(gctx, gate);
    ggml_tensor* ffn  = ggml_mul(gctx, gate, up);
    ffn  = ggml_mul_mat(gctx, L.ffn_down, ffn);

    return ggml_add(gctx, x, ffn);
}

bool ForwardContext::prefill(const std::vector<int32_t>& token_ids,
                              std::vector<float>& last_logits,
                              int& out_n_vocab) {
    if (!impl_->cache) {
        std::fprintf(stderr, "[sp-engine] prefill: no cache bound\n");
        return false;
    }
    if (token_ids.empty()) return false;
    const int n = (int)token_ids.size();

    std::vector<float> all_logits;
    std::vector<std::vector<float>> Ks, Vs;
    if (!forward_full(token_ids, all_logits, out_n_vocab, &Ks, &Vs)) {
        return false;
    }

    // Push every layer to the bound cache at offset = current kv_pos.
    for (int L = 0; L < impl_->n_layer; ++L) {
        if (!impl_->cache->write(L, impl_->kv_pos, n,
                                 Ks[(size_t)L].data(), Vs[(size_t)L].data())) {
            std::fprintf(stderr, "[sp-engine] prefill: cache write layer %d failed\n", L);
            return false;
        }
    }
    impl_->kv_pos += n;

    // Slice last token's logits.
    last_logits.assign(all_logits.end() - out_n_vocab, all_logits.end());
    return true;
}

bool ForwardContext::decode(int32_t token_id,
                             std::vector<float>& logits,
                             int& out_n_vocab,
                             std::vector<float>* dbg_K_layer0,
                             std::vector<float>* dbg_X_layer0) {
    if (!impl_->cache) {
        std::fprintf(stderr, "[sp-engine] decode: no cache bound\n");
        return false;
    }
    auto* W      = impl_->weights;
    if (W->layers().empty()) return false;
    const int past_n  = impl_->kv_pos;
    const int kv_tot  = past_n + 1;
    const int hd      = impl_->head_dim;
    const int n_kv    = impl_->n_head_kv;
    out_n_vocab       = impl_->n_vocab;

    // Pull past K/V for every layer out of the cache up-front, into
    // host buffers we'll upload as input tensors. Index 0 = layer 0.
    std::vector<std::vector<float>> past_K_all(impl_->n_layer);
    std::vector<std::vector<float>> past_V_all(impl_->n_layer);
    if (past_n > 0) {
        for (int L = 0; L < impl_->n_layer; ++L) {
            if (!impl_->cache->read(L, past_n, past_K_all[(size_t)L], past_V_all[(size_t)L])) {
                std::fprintf(stderr, "[sp-engine] decode: cache read layer %d failed\n", L);
                return false;
            }
        }
    }

    ggml_init_params gip = {};
    gip.mem_size   = impl_->ctx_size;
    gip.mem_buffer = impl_->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) return false;

    ggml_tensor* ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(ids);
    ggml_tensor* pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(pos);

    ggml_tensor* freq_factors = nullptr;
    if (!impl_->freq_factors_vec.empty()) {
        freq_factors = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                          (int64_t)impl_->freq_factors_vec.size());
        ggml_set_input(freq_factors);
    }

    // Per-layer past K/V inputs (only when past_n > 0).
    std::vector<ggml_tensor*> past_K_tens(impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> past_V_tens(impl_->n_layer, nullptr);
    if (past_n > 0) {
        for (int L = 0; L < impl_->n_layer; ++L) {
            past_K_tens[(size_t)L] = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hd, n_kv, past_n);
            past_V_tens[(size_t)L] = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hd, n_kv, past_n);
            ggml_set_input(past_K_tens[(size_t)L]);
            ggml_set_input(past_V_tens[(size_t)L]);
        }
    }

    // Causal mask: 1 query × kv_total keys. Always present so the
    // graph topology is identical between Standard and ALiBi PE
    // (max_bias > 0 requires a mask tensor; max_bias = 0 still uses
    // the mask values as additive offsets with slope=1).
    ggml_tensor* mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kv_tot, 1);
    ggml_set_input(mask);

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);

    std::vector<ggml_tensor*> new_K(impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> new_V(impl_->n_layer, nullptr);

    ggml_tensor* x_layer0 = nullptr;
    for (int L = 0; L < impl_->n_layer; ++L) {
        ggml_tensor* k_cap = nullptr;
        ggml_tensor* v_cap = nullptr;
        x = build_block_decode(gctx, x, pos,
                               past_n > 0 ? past_K_tens[(size_t)L] : nullptr,
                               past_n > 0 ? past_V_tens[(size_t)L] : nullptr,
                               mask, freq_factors, impl_->alibi_max_bias,
                               W->layers()[(size_t)L],
                               past_n, hd, impl_->n_head, n_kv,
                               impl_->n_rot,
                               impl_->rope_freq_base, impl_->rope_freq_scale,
                               &k_cap, &v_cap);
        ggml_set_output(k_cap);
        ggml_set_output(v_cap);
        new_K[(size_t)L] = k_cap;
        new_V[(size_t)L] = v_cap;
        if (L == 0 && dbg_X_layer0) {
            x_layer0 = x;
            ggml_set_output(x_layer0);
        }
    }

    ggml_tensor* h = ggml_rms_norm(gctx, x, 1e-5f);
    h = ggml_mul(gctx, h, W->output_norm);
    ggml_tensor* logits_t = ggml_mul_mat(gctx, W->output, h);
    ggml_set_output(logits_t);

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, logits_t);
    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        std::fprintf(stderr, "[sp-engine] decode: gallocr failed (past_n=%d)\n", past_n);
        ggml_free(gctx); return false;
    }

    // Upload all inputs.
    int32_t id_buf[1] = { token_id };
    int32_t pos_buf[1] = { past_n };  // new token sits at position past_n
    ggml_backend_tensor_set(ids, id_buf, 0, sizeof(id_buf));
    ggml_backend_tensor_set(pos, pos_buf, 0, sizeof(pos_buf));
    if (freq_factors) {
        ggml_backend_tensor_set(freq_factors, impl_->freq_factors_vec.data(), 0,
                                impl_->freq_factors_vec.size() * sizeof(float));
    }
    if (past_n > 0) {
        const size_t nbytes = (size_t)hd * n_kv * past_n * sizeof(float);
        for (int L = 0; L < impl_->n_layer; ++L) {
            ggml_backend_tensor_set(past_K_tens[(size_t)L],
                                    past_K_all[(size_t)L].data(), 0, nbytes);
            ggml_backend_tensor_set(past_V_tens[(size_t)L],
                                    past_V_all[(size_t)L].data(), 0, nbytes);
        }
    }
    {
        std::vector<float> mvals((size_t)kv_tot, 0.0f);
        if (impl_->alibi_max_bias > 0.0f) {
            for (int kv = 0; kv < kv_tot; ++kv) mvals[(size_t)kv] = -(float)(past_n - kv);
        }
        ggml_backend_tensor_set(mask, mvals.data(), 0, mvals.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] decode: compute failed (past_n=%d)\n", past_n);
        ggml_free(gctx); return false;
    }

    // Pull new K/V from each layer and write to cache.
    const size_t kv_elems = (size_t)hd * n_kv;
    std::vector<float> K_one(kv_elems), V_one(kv_elems);
    for (int L = 0; L < impl_->n_layer; ++L) {
        ggml_backend_tensor_get(new_K[(size_t)L], K_one.data(), 0, kv_elems * sizeof(float));
        ggml_backend_tensor_get(new_V[(size_t)L], V_one.data(), 0, kv_elems * sizeof(float));
        if (L == 0 && dbg_K_layer0) *dbg_K_layer0 = K_one;
        if (!impl_->cache->write(L, past_n, 1, K_one.data(), V_one.data())) {
            std::fprintf(stderr, "[sp-engine] decode: cache write layer %d failed\n", L);
            ggml_free(gctx); return false;
        }
    }

    logits.resize((size_t)out_n_vocab);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));

    if (x_layer0 && dbg_X_layer0) {
        const size_t nbytes = (size_t)impl_->n_embd * sizeof(float);
        dbg_X_layer0->resize((size_t)impl_->n_embd);
        ggml_backend_tensor_get(x_layer0, dbg_X_layer0->data(), 0, nbytes);
    }

    ggml_free(gctx);
    impl_->kv_pos += 1;
    return true;
}

} // namespace sp::engine
