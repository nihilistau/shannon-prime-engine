// Shannon-Prime Engine — forward-pass graph builder (stage 3a)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "forward.h"
#include "gdn_state.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "prime_pe.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef SP_ENGINE_WITH_CUDA
#include <cuda_runtime.h>
#endif

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
    int   rope_mode = 0;          // 0 = LLaMA-style, 2 = NEOX (qwen, phi3, …)
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
    float rms_norm_eps    = 1e-5f;  // Llama-3 default; Qwen3 uses 1e-6
    // Token-embedding scale. Gemma3 multiplies tok_embd lookups by
    // sqrt(n_embd) before the first block; other archs leave embeddings
    // untouched (scale = 1.0).
    float embd_scale      = 1.0f;
    // Gemma3 sliding-window attention. When `swa_window > 0`, layers
    // where (layer_idx + 1) % 6 != 0 ("local" layers) attend only to
    // the last `swa_window` positions AND use `swa_rope_freq_base`
    // (10000) for RoPE, while the 1-in-6 "global" layers keep the
    // GGUF-provided `rope_freq_base` (typically 1e6) and the standard
    // causal mask. The 5-local : 1-global pattern is hardcoded —
    // gemma3 doesn't parameterise it per-layer in the GGUF.
    int   swa_window         = 0;
    float swa_rope_freq_base = 10000.0f;

    // FFN activation. Llama/qwen/mistral/phi/granite use SwiGLU
    // (silu(gate) * up); gemma (1/2/3) uses GeGLU with the tanh
    // approximation of GELU (gelu_pytorch_tanh). ggml_gelu is the
    // tanh variant, matching gemma's act_fn exactly.
    bool  ffn_gelu           = false;

    // PrimePE-RoPE-ALiBi precomputed values. `freq_factors_vec` is
    // empty when PE is Standard/AlibiOnly (ggml_rope_ext gets nullptr);
    // `alibi_max_bias` is 0 unless an ALiBi mode is selected.
    PeSettings         pe;
    std::vector<float> freq_factors_vec;
    float              alibi_max_bias = 0.0f;
    // Model-provided RoPE scaling factors (rope_freqs.weight in GGUF).
    // Llama-3.2 ships these to extend the 8K base context to 128K via
    // YaRN-style per-frequency scaling — without them, attention
    // degrades at long positions. When PrimePE is inactive we hand
    // this tensor straight to ggml_rope_ext as freq_factors.
    ggml_tensor*       model_rope_freqs = nullptr;

    ggml_backend_t        backend = nullptr;   // CPU by default; set via
                                                // external_backend param or
                                                // SP_ENGINE_BACKEND=gpu|cuda|vulkan
    bool                  owns_backend = false; // true if created here, false if
                                                // passed in from outside
    ggml_backend_buffer_t compute_buf = nullptr;
    ggml_gallocr_t        allocr = nullptr;

    size_t ctx_size = 0;
    std::vector<uint8_t> ctx_mem;              // backing for graph ggml_context

    // Stage 5b: stateful session over a bound (non-owning) KvCache.
    KvCache* cache  = nullptr;
    int      kv_pos = 0;

    // Hybrid-arch companion (qwen35moe): bound GdnStateCache for the
    // recurrent state of GDN layers. Non-owning; null for pure-attention
    // archs. Phase 3 reads/writes this alongside `cache` inside the
    // forward builder, dispatched on LlamaLayer::kind.
    class GdnStateCache* gdn_state = nullptr;

    ~Impl() {
        if (compute_buf) ggml_backend_buffer_free(compute_buf);
        if (allocr)      ggml_gallocr_free(allocr);
        if (backend && owns_backend) ggml_backend_free(backend);
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
    if (cache) {
        std::fprintf(stderr,
            "[sp-engine:diag] bind_cache: is_gpu=%d n_layer=%d n_head_kv=%d "
            "head_dim=%d max_seq=%d\n",
            (int)cache->is_gpu(), cache->n_layer(),
            cache->n_head_kv(), cache->head_dim(), cache->max_seq());
    }
}

void ForwardContext::bind_gdn_state(GdnStateCache* gdn) {
    impl_->gdn_state = gdn;
    if (gdn) {
        std::fprintf(stderr,
            "[sp-engine:diag] bind_gdn_state: n_layer=%d n_gdn=%d conv_kernel=%d "
            "conv_channels=%d head_v_dim=%d num_v_heads=%d\n",
            gdn->n_layer(), gdn->n_gdn_layers(), gdn->conv_kernel(),
            gdn->conv_channels(), gdn->head_v_dim(), gdn->num_v_heads());
    }
}

int ForwardContext::kv_pos() const { return impl_->kv_pos; }

std::unique_ptr<ForwardContext> ForwardContext::create(const Model& model,
                                                        const LlamaWeights& weights,
                                                        int ctx_size_bytes,
                                                        PeSettings pe,
                                                        ggml_backend_t external_backend) {
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
    // RMS norm epsilon: Llama-3 = 1e-5, Qwen3 = 1e-6, etc. The key is
    // namespaced by architecture (e.g., "qwen3.attention.layer_norm_rms_epsilon").
    {
        const std::string key = model.architecture() + ".attention.layer_norm_rms_epsilon";
        const double eps = model.get_f64(key, 1e-5);
        fc->impl_->rms_norm_eps = (float)eps;
    }
    // RoPE rotation layout (NORMAL = adjacent pairs / NEOX = offset by n_rot/2).
    // The two layouts differ in how the half-rotations are interleaved within
    // each head's vector; the wrong choice scrambles K/Q relative phase and
    // wrecks long-context attention. Mirrors llama.cpp's per-arch table.
    {
        const std::string& a = model.architecture();
        if (a == "llama" || a == "mistral" || a == "mistral3" ||
            a == "granite" || a == "minicpm" || a == "command-r") {
            fc->impl_->rope_mode = 0;          // GGML_ROPE_TYPE_NORMAL
        } else {
            fc->impl_->rope_mode = 2;          // GGML_ROPE_TYPE_NEOX (qwen*, phi*, gemma*, ...)
        }
    }

    // Token-embedding scale. Gemma family (gemma / gemma2 / gemma3)
    // multiplies tok_embd lookups by sqrt(hidden_dim) before the first
    // block; other archs leave embeddings untouched.
    {
        const std::string& a = model.architecture();
        if (a == "gemma" || a == "gemma2" || a == "gemma3") {
            fc->impl_->embd_scale = sqrtf((float)fc->impl_->n_embd);
        }
    }

    // Gemma3 sliding-window attention. Reads the `sliding_window`
    // hparam — 1024 on gemma3-12B, 512 on smaller variants. Non-
    // gemma3 archs leave swa_window=0 (SWA off).
    if (model.architecture() == "gemma3") {
        const int64_t w = model.get_i64("gemma3.attention.sliding_window", 0);
        if (w > 0) fc->impl_->swa_window = (int)w;
    }

    // FFN activation flavor. Gemma family uses GELU (tanh approx)
    // instead of SiLU in the gated MLP.
    {
        const std::string& a = model.architecture();
        fc->impl_->ffn_gelu = (a == "gemma" || a == "gemma2" || a == "gemma3");
    }

    // PrimePE precompute. One allocation per ForwardContext; re-uploaded
    // into each graph's freq_factors input tensor.
    fc->impl_->pe               = pe;
    fc->impl_->freq_factors_vec = prime_pe_freq_factors(pe.pe_mode, pe.pe_alpha,
                                                        pe.pe_tier, fc->impl_->n_rot,
                                                        fc->impl_->rope_freq_base);
    fc->impl_->alibi_max_bias   = prime_pe_alibi_max_bias(pe.pe_mode, pe.pe_alpha);

    // Capture the GGUF-provided RoPE scaling factors (Llama-3.2's 32-fp32
    // long-context scaling, etc.). Used as fallback freq_factors for
    // ggml_rope_ext when PrimePE is inactive.
    fc->impl_->model_rope_freqs = weights.rope_freqs;

    // Sidecar discovery: sp_inject_freqs.py writes an fp32 sidecar
    // `<model>.sp_freq_factors.bin` alongside the injected GGUF. It's
    // primarily debug/diagnostic now (the factors are also embedded in
    // the modified GGUF as rope_freqs.weight), but we still load it if
    // found — handy for A/B-ing different alphas against an unmodified
    // GGUF. PrimePE takes precedence when active; otherwise the sidecar
    // replaces model_rope_freqs.
    if (fc->impl_->freq_factors_vec.empty()) {
        const std::string& mp = model.path();
        const size_t dot = mp.find_last_of('.');
        const std::string base = (dot == std::string::npos) ? mp : mp.substr(0, dot);
        const std::string sidecar_path = base + ".sp_freq_factors.bin";
        if (std::FILE* sf = std::fopen(sidecar_path.c_str(), "rb")) {
            std::fseek(sf, 0, SEEK_END);
            const long bytes = std::ftell(sf);
            std::fseek(sf, 0, SEEK_SET);
            const size_t expect = (size_t)(fc->impl_->n_rot / 2) * sizeof(float);
            if (bytes > 0 && (size_t)bytes == expect) {
                fc->impl_->freq_factors_vec.resize((size_t)(fc->impl_->n_rot / 2));
                if (std::fread(fc->impl_->freq_factors_vec.data(), 1, expect, sf) == expect) {
                    std::fprintf(stderr,
                        "[sp-engine] loaded RoPE sidecar: %s (%d freqs)\n",
                        sidecar_path.c_str(), fc->impl_->n_rot / 2);
                } else {
                    fc->impl_->freq_factors_vec.clear();
                    std::fprintf(stderr,
                        "[sp-engine] sidecar read failed: %s\n", sidecar_path.c_str());
                }
            } else if (bytes > 0) {
                std::fprintf(stderr,
                    "[sp-engine] sidecar size mismatch (%ld B, expected %zu B) — ignoring %s\n",
                    bytes, expect, sidecar_path.c_str());
            }
            std::fclose(sf);
        }
    }

    // Backend selection. If the caller supplied one (usually picked
    // in main.cpp so it can be shared with LlamaWeights::load for
    // weights offload), use it non-owning. Otherwise pick CPU here.
    if (external_backend) {
        fc->impl_->backend = external_backend;
        fc->impl_->owns_backend = false;
    } else {
        fc->impl_->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        fc->impl_->owns_backend = (fc->impl_->backend != nullptr);
        if (!fc->impl_->backend) {
            std::fprintf(stderr, "[sp-engine] ForwardContext: failed to init CPU backend\n");
            return nullptr;
        }
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
    if (impl_->embd_scale != 1.0f) {
        emb = ggml_scale(gctx, emb, impl_->embd_scale);
    }
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
// Gemma3 SWA pattern: 5 sliding-window layers for every global layer,
// globals every 6th layer starting at index 5 (5, 11, 17, ...). A
// layer is SWA iff (layer_idx + 1) % 6 != 0. When swa_window == 0
// the caller has SWA disabled entirely (all non-gemma3 archs and any
// gemma3 variant that ships without the sliding_window hparam).
// ------------------------------------------------------------------
static inline bool sp_is_gemma3_swa_layer(int layer_idx, int swa_window) {
    return swa_window > 0 && ((layer_idx + 1) % 6 != 0);
}

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
                                 float rms_eps,
                                 int   rope_mode,
                                 bool  ffn_gelu,             // gemma* → GELU, else SiLU
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
    ggml_tensor* xa = ggml_rms_norm(gctx, x, rms_eps);
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
        Q = ggml_rms_norm(gctx, Q, rms_eps);
        Q = ggml_mul(gctx, Q, L.attn_q_norm);
    }
    if (L.attn_k_norm) {
        K = ggml_rms_norm(gctx, K, rms_eps);
        K = ggml_mul(gctx, K, L.attn_k_norm);
    }

    Q = ggml_rope_ext(gctx, Q, pos, freq_factors,
                      n_rot, rope_mode, 0, freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_ext(gctx, K, pos, freq_factors,
                      n_rot, rope_mode, 0, freq_base, freq_scale, 0, 1, 32, 1);

    // V is a ggml_reshape_3d view over the projection output. ggml_set_output
    // on a view does not preserve the underlying buffer through subsequent
    // ops (the gallocr can repurpose the source buffer once the view is
    // "consumed" by the next reshape). Materialise V via ggml_cont and
    // reuse the materialised tensor downstream so the gallocr sees a real
    // buffer-owning node on the compute path.
    V = ggml_cont(gctx, V);

    if (k_capture) *k_capture = K;
    if (v_capture) *v_capture = V;

    // Attention via ggml_flash_attn_ext — fused, numerically stable for
    // long sequences, handles GQA broadcast internally (no manual repeat
    // needed). Permute Q/K/V to [head_dim, n, n_head_or_kv]; cast K and
    // V to F16 (the op's expected operand type for the matmul kernels);
    // F16 mask is required when max_bias > 0 (and tolerated otherwise).
    // Output comes back shaped [head_dim, n_head, n] which reshape_2d
    // flattens to [n_embd_q, n] for the wo projection. Output precision
    // is set to F32 explicitly.
    ggml_tensor* qp = ggml_permute(gctx, Q, 0, 2, 1, 3);
    ggml_tensor* kp = ggml_permute(gctx, K, 0, 2, 1, 3);
    ggml_tensor* vp = ggml_permute(gctx, V, 0, 2, 1, 3);
    if (kp->type == GGML_TYPE_F32) kp = ggml_cast(gctx, kp, GGML_TYPE_F16);
    if (vp->type == GGML_TYPE_F32) vp = ggml_cast(gctx, vp, GGML_TYPE_F16);
    ggml_tensor* mask_f16 = ggml_cast(gctx, kq_mask, GGML_TYPE_F16);
    ggml_tensor* attn = ggml_flash_attn_ext(gctx, qp, kp, vp, mask_f16,
                                            1.0f / sqrtf((float)head_dim),
                                            alibi_max_bias, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(gctx, attn, n_embd_q, n);

    ggml_tensor* y1 = ggml_mul_mat(gctx, L.wo, attn);
    if (L.bo) y1 = ggml_add(gctx, y1, L.bo);

    // Gemma3 sandwich norm: an extra RMSNorm on the attention output
    // BEFORE the residual add. For non-gemma archs this tensor is
    // nullptr and we fall through to the classic pre-norm path.
    if (L.attn_post_norm) {
        y1 = ggml_rms_norm(gctx, y1, rms_eps);
        y1 = ggml_mul(gctx, y1, L.attn_post_norm);
    }

    x = ggml_add(gctx, x, y1);

    // FFN.
    ggml_tensor* xb = ggml_rms_norm(gctx, x, rms_eps);
    xb = ggml_mul(gctx, xb, L.ffn_norm);

    ggml_tensor* gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
    ggml_tensor* up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    gate = ffn_gelu ? ggml_gelu(gctx, gate) : ggml_silu(gctx, gate);
    ggml_tensor* ffn  = ggml_mul(gctx, gate, up);
    ffn  = ggml_mul_mat(gctx, L.ffn_down, ffn);

    // Gemma3 sandwich norm on the FFN output (mirrors attn_post_norm above).
    if (L.ffn_post_norm) {
        ffn = ggml_rms_norm(gctx, ffn, rms_eps);
        ffn = ggml_mul(gctx, ffn, L.ffn_post_norm);
    }

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
    } else if (impl_->model_rope_freqs) {
        // Fall back to the GGUF-provided RoPE scaling (Llama-3.2 ships
        // 32 fp32 factors that extend its context from 8K to 128K).
        freq_factors = impl_->model_rope_freqs;
    }

    // Causal mask [n_kv=n, n_q=n] fp32. Fixes self-attention's
    // bidirectional bug from stage 3b/3c AND carries ALiBi position
    // offsets when max_bias > 0 (slope * mask added to scores).
    ggml_tensor* kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
    ggml_set_input(kq_mask);

    // Gemma3 SWA: layer 0 is always a "local" layer (since (0+1)%6 != 0),
    // so when SWA is active we use the window-bound mask + base 10000
    // for this single-layer graph.
    const bool layer0_is_swa = sp_is_gemma3_swa_layer(0, impl_->swa_window);
    const float block0_freq_base =
        layer0_is_swa ? impl_->swa_rope_freq_base : impl_->rope_freq_base;

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);
    if (impl_->embd_scale != 1.0f) {
        x = ggml_scale(gctx, x, impl_->embd_scale);
    }
    x = build_block(gctx, x, pos, kq_mask, freq_factors, impl_->alibi_max_bias,
                     W->layers()[0], n,
                     impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                     impl_->n_rot, block0_freq_base, impl_->rope_freq_scale,
                     impl_->rms_norm_eps, impl_->rope_mode, impl_->ffn_gelu);
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
    // Gemma3 SWA: when layer 0 is a local layer, additionally -INF
    // any kv farther than (swa_window - 1) positions back.
    {
        std::vector<float> mask((size_t)n * (size_t)n);
        const bool alibi = (impl_->alibi_max_bias > 0.0f);
        const int w = layer0_is_swa ? impl_->swa_window : 0;
        for (int q = 0; q < n; ++q) {
            for (int kv = 0; kv < n; ++kv) {
                float v;
                if (kv > q)                 v = -INFINITY;
                else if (w > 0 && q - kv >= w) v = -INFINITY;
                else if (alibi)             v = -(float)(q - kv);
                else                        v = 0.0f;
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
    } else if (impl_->model_rope_freqs) {
        freq_factors = impl_->model_rope_freqs;
    }
    ggml_tensor* kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
    ggml_set_input(kq_mask);
    // Gemma3 SWA: second mask tensor only allocated when SWA is active.
    // Global layers (1-in-6) use kq_mask; local layers use kq_mask_swa.
    ggml_tensor* kq_mask_swa = nullptr;
    if (impl_->swa_window > 0) {
        kq_mask_swa = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
        ggml_set_input(kq_mask_swa);
    }

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);
    if (impl_->embd_scale != 1.0f) {
        x = ggml_scale(gctx, x, impl_->embd_scale);
    }

    std::vector<ggml_tensor*> cap_K, cap_V;
    if (capture) {
        cap_K.assign((size_t)impl_->n_layer, nullptr);
        cap_V.assign((size_t)impl_->n_layer, nullptr);
    }

    ggml_tensor* x_layer0 = nullptr;
    for (int i = 0; i < impl_->n_layer; ++i) {
        ggml_tensor* k_cap = nullptr;
        ggml_tensor* v_cap = nullptr;
        // Per-layer SWA dispatch: local gemma3 layers swap mask + rope base.
        const bool local = sp_is_gemma3_swa_layer(i, impl_->swa_window);
        ggml_tensor* layer_mask = local ? kq_mask_swa : kq_mask;
        const float layer_freq_base = local ? impl_->swa_rope_freq_base
                                             : impl_->rope_freq_base;
        x = build_block(gctx, x, pos, layer_mask, freq_factors, impl_->alibi_max_bias,
                         W->layers()[(size_t)i], n,
                         impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                         impl_->n_rot, layer_freq_base, impl_->rope_freq_scale,
                         impl_->rms_norm_eps, impl_->rope_mode, impl_->ffn_gelu,
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
    ggml_tensor* h = ggml_rms_norm(gctx, x, impl_->rms_norm_eps);
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

        // Gemma3 SWA mask — identical to the causal mask above but with
        // the additional constraint that (q - kv) < swa_window. Keys
        // older than the window are -INF. ggml_backend_tensor_set is
        // synchronous on the CUDA backend so reusing `mask` in place
        // after the first upload is safe.
        if (kq_mask_swa) {
            const int w = impl_->swa_window;
            for (int q = 0; q < n; ++q) {
                for (int kv = 0; kv < n; ++kv) {
                    if (kv <= q && q - kv >= w) {
                        mask[(size_t)q * n + kv] = -INFINITY;
                    }
                }
            }
            ggml_backend_tensor_set(kq_mask_swa, mask.data(), 0,
                                    mask.size() * sizeof(float));
        }
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
                                        float rms_eps,
                                        int   rope_mode,
                                        ggml_tensor** k_capture,
                                        ggml_tensor** v_capture) {
    const int n_embd_q = n_head * head_dim;
    const int n        = 1;

    ggml_tensor* xa = ggml_rms_norm(gctx, x, rms_eps);
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
        Q = ggml_rms_norm(gctx, Q, rms_eps);
        Q = ggml_mul(gctx, Q, L.attn_q_norm);
    }
    if (L.attn_k_norm) {
        K = ggml_rms_norm(gctx, K, rms_eps);
        K = ggml_mul(gctx, K, L.attn_k_norm);
    }

    Q = ggml_rope_ext(gctx, Q, pos, freq_factors,
                      n_rot, rope_mode, 0, freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_ext(gctx, K, pos, freq_factors,
                      n_rot, rope_mode, 0, freq_base, freq_scale, 0, 1, 32, 1);

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

    ggml_tensor* xb = ggml_rms_norm(gctx, x, rms_eps);
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

    // ── Adaptive calibration: if cache hasn't been calibrated yet,
    // use this prefill's K vectors as calibration data. Feed every
    // per-head K vector from every layer into the calibration
    // accumulator, then finalize.
    //
    // Hierarchical mode needs per-slot feeding (slot = layer*H + head)
    // so each predictor is trained on its own head's data. The shared
    // modes (sqfree Knight mask, ship variance-ranked permutation) use
    // the single-arg feed that accumulates globally across all heads.
    // Diagnostic escape hatch: SHANNON_PRIME_NO_CALIBRATE=1 skips
    // calibration entirely so we can A/B what calibration contributes
    // to PPL per path. Both CPU and GPU caches fall back to their
    // static reorder tables (Möbius for ship).
    const char* env_no_calib = std::getenv("SHANNON_PRIME_NO_CALIBRATE");
    const bool skip_calibration = env_no_calib && std::atoi(env_no_calib) != 0;
    if (!skip_calibration && !impl_->cache->is_calibrated()) {
        const bool hier = impl_->cache->is_hierarchical();
        // Hierarchical needs per-slot calibration; samples per slot = n
        // (tokens in this prefill batch). Underdetermined slots produce
        // garbage predictors. Warn when n is too small for hier.
        if (hier && n < 24) {
            std::fprintf(stderr,
                "[sp-engine] warning: hierarchical cache is calibrating on only "
                "%d samples per slot; predictor is underdetermined. Use a prompt "
                "of >= 24 tokens for coherent decode, or switch to --sqfree.\n", n);
        }
        if (impl_->cache->calibrate_begin()) {
            const int H  = impl_->n_head_kv;
            const int hd = impl_->head_dim;
            for (int L = 0; L < impl_->n_layer; ++L) {
                const float* K_data = Ks[(size_t)L].data();
                // Layout: K_data[(q * H + h) * hd + d]
                for (int q = 0; q < n; ++q) {
                    for (int h = 0; h < H; ++h) {
                        const float* vec = K_data + (size_t)(q * H + h) * hd;
                        if (hier) {
                            impl_->cache->calibrate_feed(L * H + h, vec);
                        } else {
                            impl_->cache->calibrate_feed(vec);
                        }
                    }
                }
            }
            impl_->cache->calibrate_end();
        }
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

    // GPU-resident cache path: skip the host readback entirely. Past
    // K/V come straight out of VRAM via read_gpu into the per-layer
    // views of past_K_big once it's allocated by the gallocr.
    const bool gpu_cache = impl_->cache->is_gpu();
    {
        static bool logged_once = false;
        if (!logged_once) {
            std::fprintf(stderr,
                "[sp-engine:diag] decode() first call: gpu_cache=%d past_n=%d\n",
                (int)gpu_cache, past_n);
            logged_once = true;
        }
    }

    // Pull past K/V for every layer out of the cache up-front, into
    // host buffers we'll upload as input tensors. Index 0 = layer 0.
    // Skipped when cache is GPU-resident.
    std::vector<std::vector<float>> past_K_all(impl_->n_layer);
    std::vector<std::vector<float>> past_V_all(impl_->n_layer);
    if (!gpu_cache && past_n > 0) {
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
    } else if (impl_->model_rope_freqs) {
        freq_factors = impl_->model_rope_freqs;
    }

    // Per-layer past K/V inputs (only when past_n > 0).
    //
    // Step-2 batching: allocate ONE contiguous [hd, n_kv, past_n, n_layer]
    // tensor for all layers' past K (same for V), expose per-layer
    // [hd, n_kv, past_n] views into it. The upload then uses 2
    // ggml_backend_tensor_set calls per decode step instead of
    // 2 * n_layer. On Qwen3-8B (32 layers) that's 64 -> 2 cudaMemcpys.
    ggml_tensor* past_K_big = nullptr;
    ggml_tensor* past_V_big = nullptr;
    std::vector<ggml_tensor*> past_K_tens(impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> past_V_tens(impl_->n_layer, nullptr);
    if (past_n > 0) {
        past_K_big = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, hd, n_kv, past_n, impl_->n_layer);
        past_V_big = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, hd, n_kv, past_n, impl_->n_layer);
        ggml_set_input(past_K_big);
        ggml_set_input(past_V_big);
        const size_t layer_stride = (size_t)hd * n_kv * past_n * sizeof(float);
        for (int L = 0; L < impl_->n_layer; ++L) {
            past_K_tens[(size_t)L] = ggml_view_3d(gctx, past_K_big, hd, n_kv, past_n,
                                                   past_K_big->nb[1], past_K_big->nb[2],
                                                   (size_t)L * layer_stride);
            past_V_tens[(size_t)L] = ggml_view_3d(gctx, past_V_big, hd, n_kv, past_n,
                                                   past_V_big->nb[1], past_V_big->nb[2],
                                                   (size_t)L * layer_stride);
        }
    }

    // Step-2 batching (output side): allocate ONE contiguous
    // [hd, n_kv, 1, n_layer] tensor for all layers' new K (same for V),
    // ggml_cpy each layer's k_cap/v_cap into its slice, mark only the
    // big tensors as outputs. Download is then 2 ggml_backend_tensor_get
    // calls instead of 2 * n_layer.
    ggml_tensor* new_K_big = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, hd, n_kv, 1, impl_->n_layer);
    ggml_tensor* new_V_big = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, hd, n_kv, 1, impl_->n_layer);
    const size_t new_layer_stride = (size_t)hd * n_kv * 1 * sizeof(float);

    // Causal mask: 1 query × kv_total keys. Always present so the
    // graph topology is identical between Standard and ALiBi PE
    // (max_bias > 0 requires a mask tensor; max_bias = 0 still uses
    // the mask values as additive offsets with slope=1).
    ggml_tensor* mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kv_tot, 1);
    ggml_set_input(mask);
    // Gemma3 SWA: parallel mask for local layers. Only allocated when
    // SWA is active; otherwise nullptr and every layer uses `mask`.
    ggml_tensor* mask_swa = nullptr;
    if (impl_->swa_window > 0) {
        mask_swa = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kv_tot, 1);
        ggml_set_input(mask_swa);
    }

    ggml_tensor* x = ggml_get_rows(gctx, W->tok_embd, ids);
    if (impl_->embd_scale != 1.0f) {
        x = ggml_scale(gctx, x, impl_->embd_scale);
    }

    // Per-layer ggml_cpy results (kept so build_forward_expand reaches
    // them — the graph needs the cpy ops as live nodes).
    std::vector<ggml_tensor*> cpy_K(impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> cpy_V(impl_->n_layer, nullptr);

    ggml_tensor* x_layer0 = nullptr;
    for (int L = 0; L < impl_->n_layer; ++L) {
        ggml_tensor* k_cap = nullptr;
        ggml_tensor* v_cap = nullptr;
        // Per-layer SWA dispatch: local gemma3 layers swap mask + rope base.
        const bool local = sp_is_gemma3_swa_layer(L, impl_->swa_window);
        ggml_tensor* layer_mask = local ? mask_swa : mask;
        const float layer_freq_base = local ? impl_->swa_rope_freq_base
                                             : impl_->rope_freq_base;
        x = build_block_decode(gctx, x, pos,
                               past_n > 0 ? past_K_tens[(size_t)L] : nullptr,
                               past_n > 0 ? past_V_tens[(size_t)L] : nullptr,
                               layer_mask, freq_factors, impl_->alibi_max_bias,
                               W->layers()[(size_t)L],
                               past_n, hd, impl_->n_head, n_kv,
                               impl_->n_rot,
                               layer_freq_base, impl_->rope_freq_scale,
                               impl_->rms_norm_eps, impl_->rope_mode,
                               &k_cap, &v_cap);
        // Copy this layer's capture into its slice of the batched
        // output tensors. ggml_cpy returns the destination view.
        ggml_tensor* dst_k = ggml_view_3d(gctx, new_K_big, hd, n_kv, 1,
                                           new_K_big->nb[1], new_K_big->nb[2],
                                           (size_t)L * new_layer_stride);
        ggml_tensor* dst_v = ggml_view_3d(gctx, new_V_big, hd, n_kv, 1,
                                           new_V_big->nb[1], new_V_big->nb[2],
                                           (size_t)L * new_layer_stride);
        cpy_K[(size_t)L] = ggml_cpy(gctx, k_cap, dst_k);
        cpy_V[(size_t)L] = ggml_cpy(gctx, v_cap, dst_v);
        if (L == 0 && dbg_X_layer0) {
            x_layer0 = x;
            ggml_set_output(x_layer0);
        }
    }
    ggml_set_output(new_K_big);
    ggml_set_output(new_V_big);

    ggml_tensor* h = ggml_rms_norm(gctx, x, impl_->rms_norm_eps);
    h = ggml_mul(gctx, h, W->output_norm);
    ggml_tensor* logits_t = ggml_mul_mat(gctx, W->output, h);
    ggml_set_output(logits_t);

    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, logits_t);
    // Wire every per-layer cpy into the graph so the backend actually
    // executes the K/V write-into-slice (otherwise build_forward_expand
    // on logits_t won't reach them — they're a side-effect path).
    for (int L = 0; L < impl_->n_layer; ++L) {
        ggml_build_forward_expand(graph, cpy_K[(size_t)L]);
        ggml_build_forward_expand(graph, cpy_V[(size_t)L]);
    }
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
    // Past K/V upload. Two paths:
    //   - gpu_cache: past_K_big / past_V_big were allocated by the
    //     gallocr on the CUDA backend. Their ->data fields are device
    //     pointers; we call KvCache::read_gpu per layer and have the
    //     decompress kernels write directly into the ggml tensors'
    //     VRAM slices — ZERO host↔device transfer. This is the step-3
    //     payoff.
    //   - host cache (step-2 fallback): pack n_layer host buffers into
    //     one contiguous region and do 2 ggml_backend_tensor_set calls.
    std::vector<float> packed_K, packed_V;
    if (past_n > 0) {
        if (gpu_cache) {
            const size_t layer_stride = (size_t)hd * n_kv * past_n;
            float* d_past_K = (float*)past_K_big->data;
            float* d_past_V = (float*)past_V_big->data;
            if (!d_past_K || !d_past_V) {
                std::fprintf(stderr, "[sp-engine] decode(gpu): past_K_big has no device ptr\n");
                ggml_free(gctx); return false;
            }
            for (int L = 0; L < impl_->n_layer; ++L) {
                if (!impl_->cache->read_gpu(L, past_n,
                                            d_past_K + (size_t)L * layer_stride,
                                            d_past_V + (size_t)L * layer_stride)) {
                    std::fprintf(stderr, "[sp-engine] decode(gpu): read_gpu layer %d failed\n", L);
                    ggml_free(gctx); return false;
                }
            }
#ifdef SP_ENGINE_WITH_CUDA
            // Cache kernels ran on their own stream (sp_cuda_cache's
            // stream). ggml's CUDA backend uses a different stream for
            // graph compute. A device-wide sync here is the simplest
            // way to guarantee the reads are visible to the graph.
            cudaDeviceSynchronize();
#endif
        } else {
            const size_t elems_per_layer = (size_t)hd * n_kv * past_n;
            packed_K.resize(elems_per_layer * impl_->n_layer);
            packed_V.resize(elems_per_layer * impl_->n_layer);
            for (int L = 0; L < impl_->n_layer; ++L) {
                std::memcpy(packed_K.data() + (size_t)L * elems_per_layer,
                            past_K_all[(size_t)L].data(),
                            elems_per_layer * sizeof(float));
                std::memcpy(packed_V.data() + (size_t)L * elems_per_layer,
                            past_V_all[(size_t)L].data(),
                            elems_per_layer * sizeof(float));
            }
            ggml_backend_tensor_set(past_K_big, packed_K.data(), 0,
                                    packed_K.size() * sizeof(float));
            ggml_backend_tensor_set(past_V_big, packed_V.data(), 0,
                                    packed_V.size() * sizeof(float));
        }
    }
    {
        std::vector<float> mvals((size_t)kv_tot, 0.0f);
        if (impl_->alibi_max_bias > 0.0f) {
            for (int kv = 0; kv < kv_tot; ++kv) mvals[(size_t)kv] = -(float)(past_n - kv);
        }
        ggml_backend_tensor_set(mask, mvals.data(), 0, mvals.size() * sizeof(float));

        // Gemma3 SWA mask for the 1-query decode row. Keys at kv=0..past_n-1
        // are past positions 0..past_n-1; kv=past_n is the new token at
        // position past_n. SWA windows out any key whose position is more
        // than (swa_window - 1) behind the new token. Position of kv index
        // k is simply k, and the query position is past_n, so the mask is
        // -INF when (past_n - k) >= swa_window.
        if (mask_swa) {
            const int w = impl_->swa_window;
            for (int kv = 0; kv < kv_tot; ++kv) {
                if (past_n - kv >= w) {
                    mvals[(size_t)kv] = -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask_swa, mvals.data(), 0,
                                    mvals.size() * sizeof(float));
        }
    }

    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-engine] decode: compute failed (past_n=%d)\n", past_n);
        ggml_free(gctx); return false;
    }

    // New K/V write-back. Two paths:
    //   - gpu_cache: skip the host download. new_K_big/new_V_big hold
    //     the fresh single-token K/V in VRAM at stride kv_elems per
    //     layer. Call KvCache::write_gpu for each layer directly off
    //     the ggml tensor's device pointer — compress kernels run on
    //     GPU without a host round-trip.
    //   - host cache: batched download into 2 contiguous host buffers,
    //     then per-layer host-side compress.
    const size_t kv_elems = (size_t)hd * n_kv;
    if (gpu_cache) {
        const float* d_new_K = (const float*)new_K_big->data;
        const float* d_new_V = (const float*)new_V_big->data;
        if (!d_new_K || !d_new_V) {
            std::fprintf(stderr, "[sp-engine] decode(gpu): new_K_big has no device ptr\n");
            ggml_free(gctx); return false;
        }
        for (int L = 0; L < impl_->n_layer; ++L) {
            if (!impl_->cache->write_gpu(L, past_n, 1,
                                          d_new_K + (size_t)L * kv_elems,
                                          d_new_V + (size_t)L * kv_elems)) {
                std::fprintf(stderr, "[sp-engine] decode(gpu): write_gpu L=%d failed\n", L);
                ggml_free(gctx); return false;
            }
        }
        if (dbg_K_layer0) {
            dbg_K_layer0->assign(kv_elems, 0.0f);
            ggml_backend_tensor_get(new_K_big, dbg_K_layer0->data(), 0,
                                    kv_elems * sizeof(float));
        }
    } else {
        std::vector<float> packed_new_K(kv_elems * impl_->n_layer);
        std::vector<float> packed_new_V(kv_elems * impl_->n_layer);
        ggml_backend_tensor_get(new_K_big, packed_new_K.data(), 0,
                                packed_new_K.size() * sizeof(float));
        ggml_backend_tensor_get(new_V_big, packed_new_V.data(), 0,
                                packed_new_V.size() * sizeof(float));
        for (int L = 0; L < impl_->n_layer; ++L) {
            const float* K_one = packed_new_K.data() + (size_t)L * kv_elems;
            const float* V_one = packed_new_V.data() + (size_t)L * kv_elems;
            if (L == 0 && dbg_K_layer0) {
                dbg_K_layer0->assign(K_one, K_one + kv_elems);
            }
            if (!impl_->cache->write(L, past_n, 1, K_one, V_one)) {
                std::fprintf(stderr, "[sp-engine] decode: cache write layer %d failed\n", L);
                ggml_free(gctx); return false;
            }
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
