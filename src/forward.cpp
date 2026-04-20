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
#include <utility>
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

    // ---- qwen35moe MoE + mRoPE hparams (zero/empty for other archs) --
    // When `is_moe` is true, forward_full dispatches per-layer based on
    // LlamaLayer::kind: standard-attention layers take the MOE_ATTN
    // path (mRoPE + build_moe_ffn), the rest take the GDN scaffold
    // (MOE_GDN). `rope_sections` is the 4-entry [t, h, w, extra] split
    // used by ggml_rope_multi; for a text-only model the caller writes
    // the same token position into all 4 per-token slots of pos_mrope,
    // reducing mRoPE to standard RoPE. `rope_mode_mrope` =
    // GGML_ROPE_TYPE_MROPE (8) for qwen35moe — mRoPE already implies
    // NeoX-style interleaving internally; ORing in NEOX yields mode=10
    // which the ggml CPU rope kernel rejects as an unknown type.
    bool  is_moe              = false;
    int   rope_sections[GGML_MROPE_SECTIONS] = {0, 0, 0, 0};
    int   rope_mode_mrope     = GGML_ROPE_TYPE_MROPE;
    int   n_expert            = 0;
    int   n_expert_used       = 0;
    bool  norm_topk_prob      = false;
    float expert_weights_scale = 1.0f;

    // ---- qwen35moe Gated DeltaNet hparams (MOE_GDN layers) ----------
    // Mirrors what cli/main.cpp's smoke test reads from the GGUF.
    //   conv_kernel    = qwen35moe.ssm.conv_kernel      (4)
    //   d_state        = qwen35moe.ssm.state_size       (128) — per-head Q/K dim
    //   n_group        = qwen35moe.ssm.group_count      (16)  — num Q/K heads
    //   num_v_heads    = qwen35moe.ssm.time_step_rank   (32)
    //   d_inner        = qwen35moe.ssm.inner_size       (4096) — V output width
    //   head_v_dim     = d_inner / num_v_heads          (128)
    //   conv_channels  = d_inner + 2 * n_group * d_state (8192)
    //   head_qk_dim    = d_state                         (128)
    // v_repeat = num_v_heads / n_group (= 2) encodes the K broadcast.
    int gdn_conv_kernel   = 0;
    int gdn_conv_channels = 0;
    int gdn_num_v_heads   = 0;
    int gdn_head_v_dim    = 0;
    int gdn_num_qk_heads  = 0;
    int gdn_head_qk_dim   = 0;
    int gdn_d_inner       = 0;

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

    // qwen35moe MoE + mRoPE hparams. These stay zero/default for any
    // arch that isn't qwen35moe, and `is_moe` gates the dispatch.
    if (model.architecture() == "qwen35moe") {
        fc->impl_->is_moe             = true;
        fc->impl_->n_expert           = (int)model.get_i64("qwen35moe.expert_count", 0);
        fc->impl_->n_expert_used      = (int)model.get_i64("qwen35moe.expert_used_count", 0);
        fc->impl_->norm_topk_prob     =      model.get_i64("qwen35moe.expert_gating_func.norm_topk_prob",
                                                           model.get_i64("qwen35moe.norm_topk_prob", 0)) != 0;
        // expert_weights_scale: default 1.0 if key missing. Qwen families
        // typically ship 1.0; DeepSeek-style archs scale by a global.
        fc->impl_->expert_weights_scale = (float)model.get_f64(
            "qwen35moe.expert_weights_scale", 1.0);

        // mRoPE rotation layout: GGML_ROPE_TYPE_MROPE (=8) is a standalone
        // rope-type value, not a bitmask — the CPU kernel switches on it
        // directly and interleaves NeoX-style inside ggml_mrope_cache_init.
        // Do NOT OR in GGML_ROPE_TYPE_NEOX (=2); the resulting mode=10
        // matches no case and the kernel aborts with "rope type not
        // supported".
        fc->impl_->rope_mode_mrope = GGML_ROPE_TYPE_MROPE;

        // Section split for rope_multi. For a text-only run every entry
        // in pos_mrope is written with the same value, so the split is
        // load-bearing only when/if image tokens enter. Still, ggml
        // requires sum(sections) == n_rot/2; fall back to a single
        // full-width section if the GGUF key is missing.
        std::vector<int32_t> secs = model.get_i32_array("qwen35moe.rope.dimension_sections");
        for (int i = 0; i < GGML_MROPE_SECTIONS; ++i) fc->impl_->rope_sections[i] = 0;
        if ((int)secs.size() == GGML_MROPE_SECTIONS) {
            for (int i = 0; i < GGML_MROPE_SECTIONS; ++i) {
                fc->impl_->rope_sections[i] = secs[i];
            }
        } else {
            // One big section covering the whole rotation; equivalent
            // to standard RoPE when all per-token pos slots are equal.
            fc->impl_->rope_sections[0] = fc->impl_->n_rot / 2;
        }

        // GDN (linear-attention) hparams for MOE_GDN layers. The same
        // numbers also feed cli/main.cpp's GdnStateCache sizing — we
        // re-read them here so forward_full doesn't need to reach into
        // the cache object to build the compute graph.
        const int conv_kernel   = (int)model.get_i64("qwen35moe.ssm.conv_kernel",    4);
        const int d_state       = (int)model.get_i64("qwen35moe.ssm.state_size",     128);
        const int n_group       = (int)model.get_i64("qwen35moe.ssm.group_count",    16);
        const int num_v_heads   = (int)model.get_i64("qwen35moe.ssm.time_step_rank", 32);
        const int d_inner       = (int)model.get_i64("qwen35moe.ssm.inner_size",     4096);
        fc->impl_->gdn_conv_kernel   = conv_kernel;
        fc->impl_->gdn_conv_channels = d_inner + 2 * n_group * d_state;
        fc->impl_->gdn_num_v_heads   = num_v_heads;
        fc->impl_->gdn_head_v_dim    = (num_v_heads > 0) ? (d_inner / num_v_heads) : 0;
        fc->impl_->gdn_num_qk_heads  = n_group;
        fc->impl_->gdn_head_qk_dim   = d_state;
        fc->impl_->gdn_d_inner       = d_inner;
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
// MoE FFN helper (qwen35moe).
//
// Input:
//   cur [n_embd, n_tokens] — the RMS-normed hidden state entering the
//                            FFN block. Already residual-streamed by
//                            the attention / GDN stage.
// Output:
//   [n_embd, n_tokens] — MoE + shared-expert contribution. The caller
//                        is responsible for the residual add.
//
// Flow (matches llama.cpp / reference transformers qwen3moe):
//   1. logits = ffn_gate_inp @ cur           [n_expert, n_tokens]
//   2. probs  = softmax(logits)              [n_expert, n_tokens]
//   3. ids    = top_k(probs, k)              [k, n_tokens]   I32
//   4. w      = gather(probs, ids)           [1, k, n_tokens]
//      (optional) renormalise w so sum_k == 1, or scale by a global.
//   5. For each of {gate, up}: ggml_mul_mat_id over the expert bank —
//      output shape [n_ff, k, n_tokens]. gate → silu; par = gate * up.
//      down = mul_mat_id(ffn_down_exps, par, ids) → [n_embd, k, n_tokens].
//   6. Weight each expert output by w and sum over k. Produces the
//      per-token MoE contribution [n_embd, n_tokens].
//   7. Shared expert (optional): silu(gate @ cur) * (up @ cur) @ down,
//      optionally gated by sigmoid(ffn_gate_inp_shexp @ cur). Added to
//      the MoE contribution.
// ------------------------------------------------------------------
static ggml_tensor* build_moe_ffn(ggml_context* gctx,
                                   ggml_tensor* cur,
                                   const LlamaLayer& L,
                                   int   n_expert,
                                   int   n_expert_used,
                                   bool  norm_topk_prob,
                                   float expert_weights_scale) {
    const int n_tokens = (int)cur->ne[1];
    const int n_embd   = (int)cur->ne[0];

    // 1-2. Router scores over the expert pool.
    ggml_tensor* logits = ggml_mul_mat(gctx, L.ffn_gate_inp, cur);   // [n_expert, n]
    ggml_tensor* probs  = ggml_soft_max(gctx, logits);               // [n_expert, n]

    // 3. Top-k expert indices per token.
    ggml_tensor* selected = ggml_top_k(gctx, probs, n_expert_used);  // [k, n]   I32

    // 4. Gather probs at the selected slots. ggml_get_rows treats dim 0
    //    as the data width and indexes along dim 1, carrying the higher
    //    dims through untouched. Reshape probs to [1, n_expert, n] so
    //    dim 1 is the row-axis and dim 2 is the token batch; the I32
    //    index tensor selected[k, n] then yields [1, k, n].
    ggml_tensor* probs_3d = ggml_reshape_3d(gctx, probs, 1, n_expert, n_tokens);
    ggml_tensor* weights  = ggml_get_rows(gctx, probs_3d, selected); // [1, k, n]

    if (norm_topk_prob) {
        // Sum over the k-dim. sum_rows sums dim 0, so permute k to
        // dim 0, sum, and let broadcasting handle the divide. The
        // summed tensor is [1, 1, n]; dividing weights [1, k, n] by
        // it broadcasts along dim 1 (ggml_div uses can_repeat).
        ggml_tensor* w_kfirst = ggml_cont(gctx,
            ggml_permute(gctx, weights, 1, 0, 2, 3));                // [k, 1, n]
        ggml_tensor* w_sum    = ggml_sum_rows(gctx, w_kfirst);       // [1, 1, n]
        weights = ggml_div(gctx, weights, w_sum);
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        weights = ggml_scale(gctx, weights, expert_weights_scale);
    }

    // 5. Expert bank application. cur → [n_embd, 1, n_tokens] for
    //    mul_mat_id; the k-dim is introduced by the indirect lookup.
    ggml_tensor* cur_3d = ggml_reshape_3d(gctx, cur, n_embd, 1, n_tokens);

    ggml_tensor* up_e   = ggml_mul_mat_id(gctx, L.ffn_up_exps,   cur_3d, selected);  // [n_ff, k, n]
    ggml_tensor* gate_e = ggml_mul_mat_id(gctx, L.ffn_gate_exps, cur_3d, selected);  // [n_ff, k, n]
    gate_e = ggml_silu(gctx, gate_e);
    ggml_tensor* par = ggml_mul(gctx, gate_e, up_e);                                  // [n_ff, k, n]
    ggml_tensor* down_e = ggml_mul_mat_id(gctx, L.ffn_down_exps, par, selected);     // [n_embd, k, n]

    // 6. Weight each expert output and sum across k.
    ggml_tensor* weighted = ggml_mul(gctx, down_e, weights);                          // [n_embd, k, n]

    // Reduce over k (dim 1). sum_rows sums dim 0, so permute k to the
    // front, sum, and reshape the leading 1 off.
    ggml_tensor* weighted_kfirst = ggml_cont(gctx,
        ggml_permute(gctx, weighted, 1, 0, 2, 3));                                    // [k, n_embd, n]
    ggml_tensor* summed = ggml_sum_rows(gctx, weighted_kfirst);                       // [1, n_embd, n]
    ggml_tensor* moe_out = ggml_reshape_2d(gctx, summed, n_embd, n_tokens);           // [n_embd, n]

    // 7. Shared expert (qwen35moe: sigmoid-gated, runs for every token).
    if (L.ffn_gate_shexp && L.ffn_up_shexp && L.ffn_down_shexp) {
        ggml_tensor* s_gate = ggml_mul_mat(gctx, L.ffn_gate_shexp, cur);   // [n_ff_shexp, n]
        ggml_tensor* s_up   = ggml_mul_mat(gctx, L.ffn_up_shexp,   cur);   // [n_ff_shexp, n]
        s_gate = ggml_silu(gctx, s_gate);
        ggml_tensor* s_par  = ggml_mul(gctx, s_gate, s_up);
        ggml_tensor* s_out  = ggml_mul_mat(gctx, L.ffn_down_shexp, s_par); // [n_embd, n]

        if (L.ffn_gate_inp_shexp) {
            // Per-token gating scalar — ffn_gate_inp_shexp is (n_embd, 1),
            // so mul_mat yields [1, n]. sigmoid keeps it in [0, 1]; the
            // broadcast-multiply with s_out [n_embd, n] repeats along
            // dim 0.
            ggml_tensor* g = ggml_mul_mat(gctx, L.ffn_gate_inp_shexp, cur); // [1, n]
            g = ggml_sigmoid(gctx, g);
            s_out = ggml_mul(gctx, s_out, g);
        }
        moe_out = ggml_add(gctx, moe_out, s_out);
    }

    return moe_out;
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

    // QKV projection. Two paths:
    //   * Classic (llama / qwen / mistral / gemma / granite): three separate
    //     matmuls against Wq / Wk / Wv plus optional biases.
    //   * Phi3 fused: one matmul against attn_qkv produces a packed
    //     [Q | K | V] row-slab which we view-split. nb[1] is the full
    //     fused-row stride; each subset view has nb[0]=elem_size and the
    //     same row stride, so the views are non-contiguous and must be
    //     materialised via ggml_cont before reshape_3d.
    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;
    if (L.attn_qkv) {
        ggml_tensor* qkv = ggml_mul_mat(gctx, L.attn_qkv, xa);   // [q+kv+kv, n]
        const size_t row_stride = qkv->nb[1];
        const size_t elem_size  = ggml_element_size(qkv);
        const int64_t n_embd_kv = (int64_t)n_head_kv * head_dim;
        const size_t q_bytes    = (size_t)n_embd_q  * elem_size;
        const size_t kv_bytes   = (size_t)n_embd_kv * elem_size;
        Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_q,  n, row_stride, 0));
        K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_kv, n, row_stride, q_bytes));
        V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_kv, n, row_stride, q_bytes + kv_bytes));
    } else {
        Q = ggml_mul_mat(gctx, L.wq, xa);
        if (L.bq) Q = ggml_add(gctx, Q, L.bq);
        K = ggml_mul_mat(gctx, L.wk, xa);
        if (L.bk) K = ggml_add(gctx, K, L.bk);
        V = ggml_mul_mat(gctx, L.wv, xa);
        if (L.bv) V = ggml_add(gctx, V, L.bv);
    }

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

    // FFN. Two paths:
    //   * Classic (llama / qwen / gemma / ...): separate ffn_gate and ffn_up
    //     matmuls, SiLU/GELU on gate, elementwise mul, then ffn_down.
    //   * Phi3 packed SwiGLU: the GGUF's ffn_up tensor is 2*n_ff wide and
    //     encodes [gate | up] along the output-row axis. One matmul plus
    //     a view-split (mirroring the fused-QKV pattern above) reconstructs
    //     gate and up. phi3 never has ffn_gate bound, so that signals the
    //     packed layout.
    ggml_tensor* xb = ggml_rms_norm(gctx, x, rms_eps);
    xb = ggml_mul(gctx, xb, L.ffn_norm);

    ggml_tensor* gate;
    ggml_tensor* up;
    if (L.ffn_gate) {
        gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
        up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    } else {
        ggml_tensor* gu = ggml_mul_mat(gctx, L.ffn_up, xb);   // [2*n_ff, n]
        const int64_t n_ff       = gu->ne[0] / 2;
        const size_t  row_stride = gu->nb[1];
        const size_t  gate_bytes = (size_t)n_ff * ggml_element_size(gu);
        gate = ggml_cont(gctx, ggml_view_2d(gctx, gu, n_ff, n, row_stride, 0));
        up   = ggml_cont(gctx, ggml_view_2d(gctx, gu, n_ff, n, row_stride, gate_bytes));
    }
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

// ------------------------------------------------------------------
// qwen35moe full-attention layer builder.
//
// Mirrors build_block's attention section but with three swaps:
//   1. ggml_rope_multi (multi-section mRoPE) instead of ggml_rope_ext.
//      The pos tensor carries 4 * n_tokens I32 values (one per section
//      per token; for text-only models the four per-token values are
//      identical to the token position).
//   2. attn_post_norm plays the role of FFN pre-norm here — the GGUF
//      binds `blk.N.post_attention_norm.weight` as `attn_post_norm`,
//      and for qwen35moe this sits between the attention residual
//      and the MoE block (not as a gemma3-style sandwich norm on the
//      attention output).
//   3. Dense FFN replaced by the MoE bank (build_moe_ffn).
// ------------------------------------------------------------------
static ggml_tensor* build_block_moe_attn(ggml_context* gctx,
                                          ggml_tensor* x,
                                          ggml_tensor* pos_mrope,   // I32[4*n]
                                          ggml_tensor* kq_mask,
                                          ggml_tensor* freq_factors,
                                          const LlamaLayer& L,
                                          int n,
                                          int head_dim,
                                          int n_head,
                                          int n_head_kv,
                                          int n_rot,
                                          int rope_sections[GGML_MROPE_SECTIONS],
                                          int rope_mode_mrope,   // NEOX | MROPE
                                          float freq_base,
                                          float freq_scale,
                                          float rms_eps,
                                          int   n_expert,
                                          int   n_expert_used,
                                          bool  norm_topk_prob,
                                          float expert_weights_scale,
                                          ggml_tensor** k_capture = nullptr,
                                          ggml_tensor** v_capture = nullptr) {
    // Qwen3-Next gated attention.
    //
    // The wq projection is 2× wider than a standard attention head
    // stack: wq->ne[1] == n_head * head_dim * 2. Each head's slab
    // splits into two head_dim halves along the innermost axis:
    //
    //     [head_dim (Q) | head_dim (gate)] × n_head
    //
    // The first half is the actual Q tensor that attends against K
    // (subject to q_norm and RoPE); the second half is a linear "gate"
    // that is passed through sigmoid and multiplied element-wise into
    // the flash-attention output before wo. This is the pattern that
    // makes wo's input dim (n_head * head_dim = 4096) match the scored
    // output rather than 2 * n_head * head_dim.
    //
    // K / V are ordinary GQA projections (n_head_kv heads of head_dim);
    // the residual stream width (n_embd) is decoupled from
    // n_head * head_dim, with wo bringing the attention output back to
    // the 2048-wide residual.
    const int n_embd_q = n_head * head_dim;                    // 4096
    const size_t ele_q = ggml_type_size(GGML_TYPE_F32);         // Q/K/V are f32 after mul_mat

    // --- Attention -------------------------------------------------
    ggml_tensor* xa = ggml_rms_norm(gctx, x, rms_eps);
    xa = ggml_mul(gctx, xa, L.attn_norm);

    ggml_tensor* Q_full = ggml_mul_mat(gctx, L.wq, xa);   // [2*head_dim*n_head, n]
    if (L.bq) Q_full = ggml_add(gctx, Q_full, L.bq);
    ggml_tensor* K = ggml_mul_mat(gctx, L.wk, xa);
    if (L.bk) K = ggml_add(gctx, K, L.bk);
    ggml_tensor* V = ggml_mul_mat(gctx, L.wv, xa);
    if (L.bv) V = ggml_add(gctx, V, L.bv);

    // Reshape Q_full to [2*head_dim, n_head, n], then split the
    // innermost axis into the Q half (offset 0) and the gate half
    // (offset head_dim). The views are non-contiguous along dim 0 so
    // we cont() them before further ops; rms_norm and rope_multi both
    // require contiguous input along the rotated axis.
    Q_full = ggml_reshape_3d(gctx, Q_full, 2 * head_dim, n_head, n);
    ggml_tensor* Q = ggml_view_3d(gctx, Q_full,
                                  head_dim, n_head, n,
                                  Q_full->nb[1], Q_full->nb[2],
                                  0);
    ggml_tensor* Gate = ggml_view_3d(gctx, Q_full,
                                     head_dim, n_head, n,
                                     Q_full->nb[1], Q_full->nb[2],
                                     (size_t)head_dim * ele_q);
    Q    = ggml_cont(gctx, Q);
    Gate = ggml_cont(gctx, Gate);

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

    // Multi-section RoPE. `sections[]` is the [t, h, w, extra] split of
    // n_rot/2; sum(sections) == n_rot/2. For text-only inputs (our
    // qwen35moe case) the same token position is written to all four
    // per-token slots in `pos_mrope` by the caller, so mRoPE collapses
    // to standard RoPE. Keeping the rope_multi op means a
    // future image-capable qwen variant could reuse this path unchanged.
    Q = ggml_rope_multi(gctx, Q, pos_mrope, freq_factors,
                        n_rot, rope_sections, rope_mode_mrope, 0,
                        freq_base, freq_scale, 0, 1, 32, 1);
    K = ggml_rope_multi(gctx, K, pos_mrope, freq_factors,
                        n_rot, rope_sections, rope_mode_mrope, 0,
                        freq_base, freq_scale, 0, 1, 32, 1);

    V = ggml_cont(gctx, V);
    if (k_capture) *k_capture = K;
    if (v_capture) *v_capture = V;

    ggml_tensor* qp = ggml_permute(gctx, Q, 0, 2, 1, 3);
    ggml_tensor* kp = ggml_permute(gctx, K, 0, 2, 1, 3);
    ggml_tensor* vp = ggml_permute(gctx, V, 0, 2, 1, 3);
    if (kp->type == GGML_TYPE_F32) kp = ggml_cast(gctx, kp, GGML_TYPE_F16);
    if (vp->type == GGML_TYPE_F32) vp = ggml_cast(gctx, vp, GGML_TYPE_F16);
    ggml_tensor* mask_f16 = ggml_cast(gctx, kq_mask, GGML_TYPE_F16);
    ggml_tensor* attn = ggml_flash_attn_ext(gctx, qp, kp, vp, mask_f16,
                                            1.0f / sqrtf((float)head_dim),
                                            0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    // attn comes back as [head_dim, n_head, n]. Apply the sigmoid gate
    // element-wise before the wo projection.
    ggml_tensor* gate_sig = ggml_sigmoid(gctx, Gate);
    attn = ggml_mul(gctx, attn, gate_sig);
    attn = ggml_reshape_2d(gctx, attn, n_embd_q, n);

    ggml_tensor* y1 = ggml_mul_mat(gctx, L.wo, attn);
    if (L.bo) y1 = ggml_add(gctx, y1, L.bo);

    x = ggml_add(gctx, x, y1);

    // --- MoE FFN ---------------------------------------------------
    // In qwen35moe, `attn_post_norm` is the FFN pre-norm (not a
    // sandwich norm on y1). If it's absent for some reason, skip the
    // extra norm and feed the residual straight into the MoE block.
    ggml_tensor* xb = x;
    if (L.attn_post_norm) {
        xb = ggml_rms_norm(gctx, x, rms_eps);
        xb = ggml_mul(gctx, xb, L.attn_post_norm);
    }
    ggml_tensor* moe = build_moe_ffn(gctx, xb, L,
                                      n_expert, n_expert_used,
                                      norm_topk_prob, expert_weights_scale);

    x = ggml_add(gctx, x, moe);
    return x;
}

// ------------------------------------------------------------------
// qwen35moe Gated-DeltaNet layer builder (Phase 3c-bis).
//
// Implements the real Qwen3-Next linear-attention block against the
// bound weights:
//
//   xa = RMSNorm(x) * attn_norm                           [n_embd, n]
//   qkv_raw = gdn_qkv @ xa                                [conv_channels, n]
//   z       = gdn_gate @ xa                               [d_inner,      n]
//   sx = [zero_state (d_conv-1) | qkv_rawᵀ]               [d_conv-1+n, conv_channels]
//   qkv_c = silu(ssm_conv(sx, ssm_conv1d))                [conv_channels, n]
//   split qkv_c into (Q, K, V) along channel axis:
//     Q: [head_qk_dim, num_qk_heads, n]  (16*128=2048 ch)
//     K: [head_qk_dim, num_qk_heads, n]  (16*128=2048 ch)
//     V: [head_v_dim,  num_v_heads,  n]  (32*128=4096 ch)
//   dt    = softplus(ssm_alpha @ xa + ssm_dt)             [num_v_heads, n]
//   g     = (-exp(ssm_a)) * dt                            [num_v_heads, n]
//   beta  = sigmoid(ssm_beta @ xa)                        [num_v_heads, n]
//   attn  = gated_delta_net(Q, K, V, g, beta, zero_state) [head_v_dim, num_v_heads, n]
//   attn  = silu(z) * (RMSNorm(attn) * ssm_norm)          [head_v_dim, num_v_heads, n]
//   y     = ssm_out @ flatten(attn)                       [n_embd, n]
//
// STAGE 2: per-layer state IO. The caller allocates one (conv_state_in,
// ssm_state_in) pair per MOE_GDN layer and fills them from a bound
// GdnStateCache before compute (or zero-fills them for a fresh prefill).
// We return the NEW state via out-params — the last (d_conv-1) rows of
// sx for the conv history, and the tail S_v*S_v*H columns of the
// gated_delta_net output for the SSM state — both materialised via
// ggml_cont so their bytes are contiguous for ggml_backend_tensor_get.
// forward_full marks them as graph outputs and writes them back into
// the cache after compute.
// ------------------------------------------------------------------
static ggml_tensor* build_block_gdn(ggml_context* gctx,
                                     ggml_tensor* x,                 // [n_embd, n]
                                     ggml_tensor* conv_state_in,     // [d_conv-1, conv_channels, 1]
                                     ggml_tensor* ssm_state_in,      // [head_v_dim*head_v_dim*num_v_heads, 1]
                                     ggml_tensor** out_conv_state,   // NEW conv history to persist
                                     ggml_tensor** out_ssm_state,    // NEW ssm state   to persist
                                     const LlamaLayer& L,
                                     int n,
                                     int conv_kernel,
                                     int conv_channels,
                                     int num_v_heads,
                                     int head_v_dim,
                                     int num_qk_heads,
                                     int head_qk_dim,
                                     float rms_eps,
                                     int   n_expert,
                                     int   n_expert_used,
                                     bool  norm_topk_prob,
                                     float expert_weights_scale) {
    const int qk_dim = num_qk_heads * head_qk_dim;  // 2048
    const int v_dim  = num_v_heads  * head_v_dim;   // 4096
    const size_t ele = ggml_type_size(GGML_TYPE_F32);

    // --- Pre-norm + input projections -----------------------------
    ggml_tensor* xa = ggml_rms_norm(gctx, x, rms_eps);
    xa = ggml_mul(gctx, xa, L.attn_norm);                           // [n_embd, n]

    ggml_tensor* qkv_raw = ggml_mul_mat(gctx, L.gdn_qkv, xa);       // [conv_channels, n]
    ggml_tensor* z       = ggml_mul_mat(gctx, L.gdn_gate, xa);      // [d_inner,       n]

    // --- Causal depthwise 1D conv over QKV stream -----------------
    // ssm_conv expects sx with shape [d_conv-1+n_t, d_inner, n_s]
    // (ne[0]=time axis, ne[1]=channel). qkv_raw is [channel, n] so
    // transpose to [n, channel] and prepend `d_conv-1` zero rows.
    ggml_tensor* qkv_t = ggml_cont(gctx, ggml_transpose(gctx, qkv_raw)); // [n, conv_channels]

    // conv_state_in is [d_conv-1, conv_channels, 1]; squeeze to 2D for concat.
    ggml_tensor* conv_pad = ggml_reshape_2d(gctx, conv_state_in,
                                            conv_kernel - 1, conv_channels); // [d_conv-1, conv_channels]

    // Concatenate along time axis (dim 0): [d_conv-1+n, conv_channels].
    ggml_tensor* sx = ggml_concat(gctx, conv_pad, qkv_t, /*dim=*/0);
    sx = ggml_reshape_3d(gctx, sx, conv_kernel - 1 + n, conv_channels, 1);

    ggml_tensor* qkv_conv = ggml_ssm_conv(gctx, sx, L.ssm_conv1d);  // [conv_channels, n, 1]
    qkv_conv = ggml_silu(gctx, qkv_conv);
    qkv_conv = ggml_reshape_2d(gctx, qkv_conv, conv_channels, n);   // [conv_channels, n]

    // --- Split into Q, K, V along channel axis -------------------
    // Channel layout (HF Qwen3Next in_proj convention): [Q | K | V].
    ggml_tensor* Qc = ggml_view_2d(gctx, qkv_conv, qk_dim, n,
                                    qkv_conv->nb[1], 0);
    ggml_tensor* Kc = ggml_view_2d(gctx, qkv_conv, qk_dim, n,
                                    qkv_conv->nb[1], (size_t)qk_dim * ele);
    ggml_tensor* Vc = ggml_view_2d(gctx, qkv_conv, v_dim,  n,
                                    qkv_conv->nb[1], (size_t)(2 * qk_dim) * ele);
    Qc = ggml_cont(gctx, Qc);
    Kc = ggml_cont(gctx, Kc);
    Vc = ggml_cont(gctx, Vc);
    Qc = ggml_reshape_4d(gctx, Qc, head_qk_dim, num_qk_heads, n, 1);
    Kc = ggml_reshape_4d(gctx, Kc, head_qk_dim, num_qk_heads, n, 1);
    Vc = ggml_reshape_4d(gctx, Vc, head_v_dim,  num_v_heads,  n, 1);

    // --- Input-dependent gates (g, beta) --------------------------
    // dt = softplus(ssm_alpha @ xa + ssm_dt_bias)     [num_v_heads, n]
    // g  = -exp(ssm_a) * dt                           [num_v_heads, n]
    // beta = sigmoid(ssm_beta @ xa)                   [num_v_heads, n]
    ggml_tensor* alpha_raw = ggml_mul_mat(gctx, L.ssm_alpha, xa);   // [num_v_heads, n]
    ggml_tensor* dt_raw    = ggml_add(gctx, alpha_raw, L.ssm_dt);   // broadcast [num_v_heads]
    ggml_tensor* dt_sp     = ggml_softplus(gctx, dt_raw);

    ggml_tensor* neg_exp_a = ggml_scale(gctx, ggml_exp(gctx, L.ssm_a), -1.0f); // [num_v_heads]
    ggml_tensor* g         = ggml_mul(gctx, dt_sp, neg_exp_a);      // broadcast over n
    // Reshape to 4D [1, num_v_heads, n, 1] — non-KDA mode (scalar g per head).
    g = ggml_reshape_4d(gctx, g, 1, num_v_heads, n, 1);

    ggml_tensor* beta_raw = ggml_mul_mat(gctx, L.ssm_beta, xa);     // [num_v_heads, n]
    ggml_tensor* beta     = ggml_sigmoid(gctx, beta_raw);
    beta = ggml_reshape_4d(gctx, beta, 1, num_v_heads, n, 1);

    // --- Gated delta-rule recurrence -----------------------------
    // state: [head_v_dim * head_v_dim * num_v_heads, n_seqs=1].
    // ggml_gated_delta_net output layout is [S_v*H, n_tokens + S_v]:
    // the first `n` columns hold the attention scores; the trailing
    // S_v columns hold the updated per-head S_v*S_v state matrices,
    // laid out so head h occupies columns [h*S_v/H, (h+1)*S_v/H).
    ggml_tensor* attn_out = ggml_gated_delta_net(gctx, Qc, Kc, Vc, g, beta, ssm_state_in);
    ggml_tensor* attn = ggml_view_3d(gctx, attn_out,
                                     head_v_dim, num_v_heads, n,
                                     (size_t)head_v_dim * ele,
                                     (size_t)head_v_dim * num_v_heads * ele,
                                     0);
    attn = ggml_cont(gctx, attn);

    // --- Extract new SSM state: tail S_v columns of attn_out -----
    // Shape [S_v*H, S_v] — S_v*S_v*H floats, same flat byte layout
    // the kernel expects as state input on the next forward call.
    if (out_ssm_state) {
        ggml_tensor* ssm_new_view = ggml_view_2d(gctx, attn_out,
                                                  (int64_t)head_v_dim * num_v_heads,
                                                  (int64_t)head_v_dim,
                                                  attn_out->nb[1],
                                                  (size_t)n * attn_out->nb[1]);
        *out_ssm_state = ggml_cont(gctx, ssm_new_view);
    }

    // --- Extract new conv history: last (d_conv-1) rows of sx ----
    // sx is [d_conv-1+n, conv_channels, 1]; we want rows [n, n+d_conv-2]
    // so the next forward_full can feed them back as the left-context
    // of its ssm_conv. For n=0 (reset case) this degenerates to a
    // zero tensor; for 0<n<d_conv-1 it still slides correctly because
    // the offset n*nb[0] keeps the window anchored on the tail.
    if (out_conv_state) {
        ggml_tensor* conv_new_view = ggml_view_3d(gctx, sx,
                                                   (int64_t)(conv_kernel - 1),
                                                   (int64_t)conv_channels,
                                                   (int64_t)1,
                                                   sx->nb[1], sx->nb[2],
                                                   (size_t)n * sx->nb[0]);
        *out_conv_state = ggml_cont(gctx, conv_new_view);
    }

    // --- Gated RMSNorm + output-gate multiply + output proj ------
    attn = ggml_rms_norm(gctx, attn, rms_eps);
    attn = ggml_mul(gctx, attn, L.ssm_norm);   // broadcast [head_v_dim]

    // Apply silu(z) gate reshaped to match [head_v_dim, num_v_heads, n].
    ggml_tensor* z_view = ggml_reshape_3d(gctx, z, head_v_dim, num_v_heads, n);
    z_view = ggml_silu(gctx, z_view);
    attn   = ggml_mul(gctx, attn, z_view);

    attn = ggml_reshape_2d(gctx, attn, v_dim, n);                   // [d_inner, n]
    ggml_tensor* y1 = ggml_mul_mat(gctx, L.ssm_out, attn);          // [n_embd, n]

    x = ggml_add(gctx, x, y1);

    // --- MoE FFN (identical to build_block_moe_attn) -------------
    ggml_tensor* xb = x;
    if (L.attn_post_norm) {
        xb = ggml_rms_norm(gctx, x, rms_eps);
        xb = ggml_mul(gctx, xb, L.attn_post_norm);
    }
    ggml_tensor* moe = build_moe_ffn(gctx, xb, L,
                                      n_expert, n_expert_used,
                                      norm_topk_prob, expert_weights_scale);

    x = ggml_add(gctx, x, moe);
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

    // qwen35moe mRoPE pos tensor. ggml_rope_multi expects I32[4 * n]
    // (one position per section per token). Only allocated for MoE
    // archs — non-MoE paths never consult it. For text-only inputs we
    // write the same token position into all four per-token slots, so
    // mRoPE collapses to standard RoPE.
    ggml_tensor* pos_mrope = nullptr;
    if (impl_->is_moe) {
        pos_mrope = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,
                                       (int64_t)(GGML_MROPE_SECTIONS * n));
        ggml_set_input(pos_mrope);
    }

    // qwen35moe GDN: per-layer recurrent state I/O. Each MOE_GDN layer
    // owns a (conv_state_in, ssm_state_in) pair the graph reads from
    // before its delta-rule step, and a (conv_state_out, ssm_state_out)
    // pair the graph writes its new state to. forward_full round-trips
    // both pairs through the bound GdnStateCache — or zero-fills them
    // on every call if no cache is bound (fresh-prefill semantics).
    // Non-MOE_GDN layer indices leave all four slots null; the dispatch
    // loop below skips them, and the read/write loops below check for
    // nullptrs before touching the cache.
    std::vector<ggml_tensor*> gdn_conv_state_in ((size_t)impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> gdn_ssm_state_in  ((size_t)impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> gdn_conv_state_out((size_t)impl_->n_layer, nullptr);
    std::vector<ggml_tensor*> gdn_ssm_state_out ((size_t)impl_->n_layer, nullptr);
    const bool have_gdn_shapes =
        impl_->is_moe && impl_->gdn_conv_channels > 0 &&
        impl_->gdn_conv_kernel > 1 && impl_->gdn_head_v_dim > 0 &&
        impl_->gdn_num_v_heads > 0;
    if (have_gdn_shapes) {
        for (int il = 0; il < impl_->n_layer; ++il) {
            if (W->layers()[(size_t)il].kind != LlamaLayerKind::MOE_GDN) continue;
            gdn_conv_state_in[(size_t)il] = ggml_new_tensor_3d(
                gctx, GGML_TYPE_F32,
                impl_->gdn_conv_kernel - 1,
                impl_->gdn_conv_channels,
                1);
            ggml_set_input(gdn_conv_state_in[(size_t)il]);
            gdn_ssm_state_in[(size_t)il] = ggml_new_tensor_2d(
                gctx, GGML_TYPE_F32,
                (int64_t)impl_->gdn_head_v_dim *
                    impl_->gdn_head_v_dim *
                    impl_->gdn_num_v_heads,
                1);
            ggml_set_input(gdn_ssm_state_in[(size_t)il]);
        }
    }

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

        const LlamaLayer& L = W->layers()[(size_t)i];

        // Dispatch on the layer's kind. STANDARD covers all non-MoE
        // archs (llama, qwen2/3, mistral, phi3, granite, gemma3); the
        // MOE_* variants are qwen35moe-specific.
        switch (L.kind) {
        case LlamaLayerKind::STANDARD: {
            // Per-layer SWA dispatch: local gemma3 layers swap mask + rope base.
            const bool local = sp_is_gemma3_swa_layer(i, impl_->swa_window);
            ggml_tensor* layer_mask = local ? kq_mask_swa : kq_mask;
            const float layer_freq_base = local ? impl_->swa_rope_freq_base
                                                 : impl_->rope_freq_base;
            x = build_block(gctx, x, pos, layer_mask, freq_factors, impl_->alibi_max_bias,
                             L, n,
                             impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                             impl_->n_rot, layer_freq_base, impl_->rope_freq_scale,
                             impl_->rms_norm_eps, impl_->rope_mode, impl_->ffn_gelu,
                             capture ? &k_cap : nullptr,
                             capture ? &v_cap : nullptr);
            break;
        }
        case LlamaLayerKind::MOE_ATTN: {
            x = build_block_moe_attn(gctx, x, pos_mrope, kq_mask, freq_factors,
                                      L, n,
                                      impl_->head_dim, impl_->n_head, impl_->n_head_kv,
                                      impl_->n_rot, impl_->rope_sections,
                                      impl_->rope_mode_mrope,
                                      impl_->rope_freq_base, impl_->rope_freq_scale,
                                      impl_->rms_norm_eps,
                                      impl_->n_expert, impl_->n_expert_used,
                                      impl_->norm_topk_prob, impl_->expert_weights_scale,
                                      capture ? &k_cap : nullptr,
                                      capture ? &v_cap : nullptr);
            break;
        }
        case LlamaLayerKind::MOE_GDN: {
            // GDN layers: no KV capture (no standard cache contribution).
            // Per-layer recurrent-state slots come from the bound
            // GdnStateCache; build_block_gdn writes the new state views
            // into our two out-vectors so forward_full can mark them as
            // graph outputs and persist them back to the cache.
            ggml_tensor* conv_out = nullptr;
            ggml_tensor* ssm_out  = nullptr;
            x = build_block_gdn(gctx, x,
                                 gdn_conv_state_in[(size_t)i],
                                 gdn_ssm_state_in[(size_t)i],
                                 &conv_out, &ssm_out,
                                 L, n,
                                 impl_->gdn_conv_kernel,
                                 impl_->gdn_conv_channels,
                                 impl_->gdn_num_v_heads,
                                 impl_->gdn_head_v_dim,
                                 impl_->gdn_num_qk_heads,
                                 impl_->gdn_head_qk_dim,
                                 impl_->rms_norm_eps,
                                 impl_->n_expert, impl_->n_expert_used,
                                 impl_->norm_topk_prob, impl_->expert_weights_scale);
            if (conv_out) { ggml_set_output(conv_out); gdn_conv_state_out[(size_t)i] = conv_out; }
            if (ssm_out)  { ggml_set_output(ssm_out);  gdn_ssm_state_out[(size_t)i]  = ssm_out;  }
            break;
        }
        }

        // K/V capture is only meaningful for attention layers. GDN
        // layers leave k_cap/v_cap null — downstream readback skips
        // those slots.
        if (capture && k_cap && v_cap) {
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

    // GGML_DEFAULT_GRAPH_SIZE (2048) is enough for a dense n_layer=80
    // attention model, but qwen35moe's MOE_GDN layers contribute ~35
    // ggml ops each (ssm_conv, delta_net, two proj MLPs, norms,
    // silu gates, reshapes…) on top of the MoE FFN. With n_layer=40
    // and 30 of those being GDN, the graph exceeds 2048 nodes. Bump
    // the capacity to 8192 which is still small (≈256 KiB overhead).
    const size_t graph_size = (impl_->is_moe) ? 8192u : (size_t)GGML_DEFAULT_GRAPH_SIZE;
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, graph_size, /*grads=*/false);
    ggml_build_forward_expand(graph, logits);
    // The GDN per-layer state outputs (conv history + ssm state) are
    // terminal ggml_cont nodes — nothing downstream consumes them. The
    // logits-rooted expand therefore skips them entirely, gallocr sees
    // no nodes for the ggml_cont, and the post-compute tensor_get fires
    // "tensor buffer not set". Expand from each state-output root so
    // gallocr allocates their backing buffers. ggml_set_output merely
    // hints at the role; the expand is what puts the node in the graph.
    if (have_gdn_shapes) {
        for (int il = 0; il < impl_->n_layer; ++il) {
            if (gdn_conv_state_out[(size_t)il]) {
                ggml_build_forward_expand(graph, gdn_conv_state_out[(size_t)il]);
            }
            if (gdn_ssm_state_out[(size_t)il]) {
                ggml_build_forward_expand(graph, gdn_ssm_state_out[(size_t)il]);
            }
        }
    }
    if (!ggml_gallocr_alloc_graph(impl_->allocr, graph)) {
        std::fprintf(stderr, "[sp-engine] forward_full: gallocr failed\n");
        ggml_free(gctx); return false;
    }

    ggml_backend_tensor_set(ids, token_ids.data(), 0, (size_t)n * sizeof(int32_t));
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = i;
    // The `pos` tensor is only consumed by STANDARD-kind layers via
    // ggml_rope_ext. Architectures like qwen35moe that route every
    // layer through MOE_ATTN (which uses pos_mrope) or MOE_GDN (which
    // uses no positional input) leave `pos` unreachable from the
    // graph, so gallocr skips it and it has no backend buffer.
    // Checking `pos->buffer` (via the view_src walk that
    // ggml_backend_tensor_set does internally) keeps this safe.
    if (pos->buffer || (pos->view_src && pos->view_src->buffer)) {
        ggml_backend_tensor_set(pos, positions.data(), 0, (size_t)n * sizeof(int32_t));
    }
    if (pos_mrope) {
        // Text-only mRoPE: replicate each token's position across all
        // four per-section slots. ggml_rope_multi reads four consecutive
        // I32 values per token — one per (t, h, w, extra) section — so
        // with identical values mRoPE degenerates to standard RoPE.
        std::vector<int32_t> positions_mrope((size_t)GGML_MROPE_SECTIONS * (size_t)n);
        for (int t = 0; t < n; ++t) {
            const int32_t p = (int32_t)t;
            for (int s = 0; s < GGML_MROPE_SECTIONS; ++s) {
                positions_mrope[(size_t)t * GGML_MROPE_SECTIONS + s] = p;
            }
        }
        ggml_backend_tensor_set(pos_mrope, positions_mrope.data(), 0,
                                positions_mrope.size() * sizeof(int32_t));
    }
    if (freq_factors) {
        ggml_backend_tensor_set(freq_factors, impl_->freq_factors_vec.data(), 0,
                                impl_->freq_factors_vec.size() * sizeof(float));
    }
    // GDN per-layer state inputs. If a GdnStateCache is bound, pull each
    // layer's last-persisted (conv_history, ssm_state) into the graph's
    // input slots; otherwise zero-fill for a fresh prefill. The cache's
    // read_conv / read_ssm are no-ops for non-GDN layer indices, but we
    // already skip those via the nullptr check.
    if (have_gdn_shapes) {
        const size_t conv_floats =
            (size_t)(impl_->gdn_conv_kernel - 1) *
            (size_t)impl_->gdn_conv_channels;
        const size_t ssm_floats =
            (size_t)impl_->gdn_head_v_dim *
            (size_t)impl_->gdn_head_v_dim *
            (size_t)impl_->gdn_num_v_heads;
        std::vector<float> conv_buf; conv_buf.reserve(conv_floats);
        std::vector<float> ssm_buf;  ssm_buf.reserve(ssm_floats);
        const std::vector<float> zero_conv(conv_floats, 0.0f);
        const std::vector<float> zero_ssm (ssm_floats,  0.0f);
        for (int il = 0; il < impl_->n_layer; ++il) {
            ggml_tensor* tc = gdn_conv_state_in[(size_t)il];
            ggml_tensor* ts = gdn_ssm_state_in [(size_t)il];
            if (!tc || !ts) continue;
            const float* conv_src = zero_conv.data();
            const float* ssm_src  = zero_ssm.data();
            if (impl_->gdn_state) {
                conv_buf.clear(); ssm_buf.clear();
                if (impl_->gdn_state->read_conv(il, conv_buf) &&
                    conv_buf.size() == conv_floats) {
                    conv_src = conv_buf.data();
                }
                if (impl_->gdn_state->read_ssm(il, ssm_buf) &&
                    ssm_buf.size() == ssm_floats) {
                    ssm_src = ssm_buf.data();
                }
            }
            ggml_backend_tensor_set(tc, conv_src, 0, conv_floats * sizeof(float));
            ggml_backend_tensor_set(ts, ssm_src,  0, ssm_floats  * sizeof(float));
        }
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
    // GDN layers in qwen35moe contribute no K/V (their cap_* slots are
    // null from the dispatch loop) — leave the corresponding per_layer
    // slots empty so the caller can skip them cleanly.
    if (capture) {
        const size_t kv_elems = (size_t)n * impl_->n_head_kv * impl_->head_dim;
        for (int i = 0; i < impl_->n_layer; ++i) {
            if (!cap_K[(size_t)i] || !cap_V[(size_t)i]) {
                (*per_layer_K)[(size_t)i].clear();
                (*per_layer_V)[(size_t)i].clear();
                continue;
            }
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

    // Persist the post-step GDN state back into the bound cache so the
    // next forward_full call resumes the delta-rule recurrence from
    // where we stopped instead of re-zeroing. With no cache bound we
    // still need the read-back-and-discard: the graph outputs are
    // allocated by gallocr and their buffers get reused on the next
    // call; leaving them dangling is harmless, so just skip the copy.
    if (have_gdn_shapes && impl_->gdn_state) {
        const size_t conv_floats =
            (size_t)(impl_->gdn_conv_kernel - 1) *
            (size_t)impl_->gdn_conv_channels;
        const size_t ssm_floats =
            (size_t)impl_->gdn_head_v_dim *
            (size_t)impl_->gdn_head_v_dim *
            (size_t)impl_->gdn_num_v_heads;
        std::vector<float> conv_buf(conv_floats);
        std::vector<float> ssm_buf (ssm_floats);
        for (int il = 0; il < impl_->n_layer; ++il) {
            ggml_tensor* tc = gdn_conv_state_out[(size_t)il];
            ggml_tensor* ts = gdn_ssm_state_out [(size_t)il];
            if (tc) {
                ggml_backend_tensor_get(tc, conv_buf.data(), 0,
                                        conv_floats * sizeof(float));
                impl_->gdn_state->write_conv(il, conv_buf.data());
            }
            if (ts) {
                ggml_backend_tensor_get(ts, ssm_buf.data(), 0,
                                        ssm_floats * sizeof(float));
                impl_->gdn_state->write_ssm(il, ssm_buf.data());
            }
        }
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

    // QKV projection. Classic (separate Wq/Wk/Wv) or phi3 fused (one
    // matmul + view-split). See build_block for detail on the fused
    // layout and why each subset view needs a ggml_cont before reshape.
    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;
    if (L.attn_qkv) {
        ggml_tensor* qkv = ggml_mul_mat(gctx, L.attn_qkv, xa);
        const size_t row_stride = qkv->nb[1];
        const size_t elem_size  = ggml_element_size(qkv);
        const int64_t n_embd_kv = (int64_t)n_head_kv * head_dim;
        const size_t q_bytes    = (size_t)n_embd_q  * elem_size;
        const size_t kv_bytes   = (size_t)n_embd_kv * elem_size;
        Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_q,  n, row_stride, 0));
        K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_kv, n, row_stride, q_bytes));
        V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, n_embd_kv, n, row_stride, q_bytes + kv_bytes));
    } else {
        Q = ggml_mul_mat(gctx, L.wq, xa);
        if (L.bq) Q = ggml_add(gctx, Q, L.bq);
        K = ggml_mul_mat(gctx, L.wk, xa);
        if (L.bk) K = ggml_add(gctx, K, L.bk);
        V = ggml_mul_mat(gctx, L.wv, xa);
        if (L.bv) V = ggml_add(gctx, V, L.bv);
    }

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
    // FFN dispatch — same logic as build_block:
    //   * Classic: separate L.ffn_gate and L.ffn_up matmuls.
    //   * Phi3 packed SwiGLU: L.ffn_gate is nullptr; L.ffn_up is 2*n_ff wide
    //     and the two halves are split into gate and up along output-rows.
    ggml_tensor* gate;
    ggml_tensor* up;
    if (L.ffn_gate) {
        gate = ggml_mul_mat(gctx, L.ffn_gate, xb);
        up   = ggml_mul_mat(gctx, L.ffn_up,   xb);
    } else {
        ggml_tensor* gu = ggml_mul_mat(gctx, L.ffn_up, xb);  // [2*n_ff, n]
        const int64_t n_ff       = gu->ne[0] / 2;
        const int64_t n_tok      = gu->ne[1];
        const size_t  row_stride = gu->nb[1];
        const size_t  gate_bytes = (size_t)n_ff * ggml_element_size(gu);
        gate = ggml_cont(gctx, ggml_view_2d(gctx, gu, n_ff, n_tok, row_stride, 0));
        up   = ggml_cont(gctx, ggml_view_2d(gctx, gu, n_ff, n_tok, row_stride, gate_bytes));
    }
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
                // Skip layers with no K (qwen35moe GDN layers) — they
                // don't contribute to the attention KV cache.
                if (Ks[(size_t)L].empty()) continue;
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
    // GDN layers leave Ks[L]/Vs[L] empty — those are fed through the
    // separate GdnStateCache bound alongside. Skip them here.
    for (int L = 0; L < impl_->n_layer; ++L) {
        if (Ks[(size_t)L].empty() || Vs[(size_t)L].empty()) continue;
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
