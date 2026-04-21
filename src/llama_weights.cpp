// Shannon-Prime Engine — Llama-family weight binding (implementation)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "llama_weights.h"
#include "gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef SP_ENGINE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sp::engine {

struct LlamaWeights::Impl {
    ggml_context* ctx = nullptr;
    // Paired metadata ggml_context from gguf_init_from_file. When set
    // (cpu-mmap and partial-offload paths), we own it jointly with `gguf`
    // below. gguf_free only releases the gguf_context itself, so the
    // paired ggml_context must be freed explicitly.
    ggml_context* meta_ctx = nullptr;
    gguf_context* gguf = nullptr;
    // When we offloaded to a non-CPU backend, this holds the allocated
    // buffer so the tensor data stays alive for the Weights' lifetime.
    ggml_backend_buffer* backend_buf = nullptr;
    ~Impl() {
        if (backend_buf) ggml_backend_buffer_free(backend_buf);
        if (ctx)         ggml_free(ctx);
        if (meta_ctx)    ggml_free(meta_ctx);
        if (gguf)        gguf_free(gguf);
    }
};

LlamaWeights::LlamaWeights() : impl_(std::make_unique<Impl>()) {}
LlamaWeights::~LlamaWeights() = default;

ggml_context* LlamaWeights::ctx() const { return impl_->ctx; }

// ------------------------------------------------------------------
// Arch registry — which archs we know have the standard layout.
// ------------------------------------------------------------------
static bool supported_arch(const std::string& a) {
    static const std::unordered_set<std::string> ok = {
        "llama", "qwen2", "qwen3", "mistral3", "phi3", "granite", "gemma3",
        "qwen35moe"
    };
    return ok.count(a) != 0;
}

// Layer-type classification for qwen35moe: every `full_attention_interval`-th
// layer is full attention (indices full_attn_interval-1, 2*full_attn_interval-1, ...)
// per the reference llama.cpp hparams check. The rest are Gated DeltaNet.
//
// Empirical confirmation on Qwen3.6-35B-A3B (full_attn_interval=4):
//   blk.0.attn_qkv.weight exists + ssm_* tensors       -> GDN
//   blk.3.attn_q.weight / attn_k.weight exist          -> ATTN
static bool is_qwen_moe_attn_layer(int il, int full_attn_interval) {
    if (full_attn_interval <= 1) return true;   // degenerate: all attn
    return ((il + 1) % full_attn_interval) == 0;
}

// ------------------------------------------------------------------
// Bind pass — shared between CPU-mmap and backend-offload loads.
// ------------------------------------------------------------------
bool LlamaWeights::bind_tensors_(LlamaWeights& w, ggml_context* tctx,
                                  const Model& model) {
    int& n_bound   = w.n_bound_tensors_;
    int& n_missing = w.n_missing_optional_;
    auto bind_opt = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(tctx, name.c_str());
        if (t) n_bound++;
        else   n_missing++;
        return t;
    };
    auto bind_req = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(tctx, name.c_str());
        if (!t) {
            std::fprintf(stderr,
                "[sp-engine] LlamaWeights: required tensor missing: %s\n",
                name.c_str());
        } else {
            n_bound++;
        }
        return t;
    };

    w.tok_embd    = bind_req("token_embd.weight");
    w.output_norm = bind_req("output_norm.weight");
    w.output      = bind_opt("output.weight");
    if (!w.output) w.output = w.tok_embd;     // tied embeddings
    w.rope_freqs  = bind_opt("rope_freqs.weight");
    if (!w.tok_embd || !w.output_norm) return false;

    const int n_layer = (int)model.n_layer();
    w.layers_.resize((size_t)n_layer);

    auto layer_name = [](int i, const char* suffix) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
        return std::string(buf);
    };

    // Arch-specific dispatch. qwen35moe has two layer types within the
    // same model, both of which replace the dense FFN with an MoE bank;
    // phi3 has STANDARD layers but with fused QKV + packed SwiGLU FFN;
    // other archs are uniform classic standard layers.
    const std::string& arch = model.architecture();
    const bool is_qwen_moe  = (arch == "qwen35moe");
    const bool is_phi3      = (arch == "phi3");
    int full_attn_interval = 1;
    if (is_qwen_moe) {
        full_attn_interval =
            (int)model.get_i64("qwen35moe.full_attention_interval", 4);
    }

    for (int i = 0; i < n_layer; ++i) {
        LlamaLayer& L = w.layers_[(size_t)i];

        // Pre-norm is always present.
        L.attn_norm = bind_req(layer_name(i, "attn_norm.weight"));

        if (is_qwen_moe) {
            const bool attn_layer = is_qwen_moe_attn_layer(i, full_attn_interval);
            L.kind = attn_layer ? LlamaLayerKind::MOE_ATTN : LlamaLayerKind::MOE_GDN;

            if (attn_layer) {
                // Full attention layer: standard Q/K/V/O with Q/K per-head norms.
                L.wq          = bind_req(layer_name(i, "attn_q.weight"));
                L.wk          = bind_req(layer_name(i, "attn_k.weight"));
                L.wv          = bind_req(layer_name(i, "attn_v.weight"));
                L.wo          = bind_req(layer_name(i, "attn_output.weight"));
                L.attn_q_norm = bind_opt(layer_name(i, "attn_q_norm.weight"));
                L.attn_k_norm = bind_opt(layer_name(i, "attn_k_norm.weight"));
            } else {
                // Gated DeltaNet layer: re-uses the "attn_qkv" / "attn_gate"
                // GGUF names for the fused input projection and the output
                // gate. The tensor sizes are completely different from a
                // standard attention layer, so we bind them to dedicated
                // slots (gdn_qkv / gdn_gate) to keep the forward dispatch
                // unambiguous.
                L.gdn_qkv     = bind_req(layer_name(i, "attn_qkv.weight"));
                L.gdn_gate    = bind_req(layer_name(i, "attn_gate.weight"));
                L.ssm_conv1d  = bind_req(layer_name(i, "ssm_conv1d.weight"));
                L.ssm_a       = bind_req(layer_name(i, "ssm_a"));
                L.ssm_alpha   = bind_req(layer_name(i, "ssm_alpha.weight"));
                L.ssm_beta    = bind_req(layer_name(i, "ssm_beta.weight"));
                L.ssm_dt      = bind_req(layer_name(i, "ssm_dt.bias"));
                L.ssm_norm    = bind_req(layer_name(i, "ssm_norm.weight"));
                L.ssm_out     = bind_req(layer_name(i, "ssm_out.weight"));
            }

            // MoE FFN (both layer types): router, expert bank, + shared expert.
            L.ffn_gate_inp       = bind_req(layer_name(i, "ffn_gate_inp.weight"));
            L.ffn_gate_exps      = bind_req(layer_name(i, "ffn_gate_exps.weight"));
            L.ffn_up_exps        = bind_req(layer_name(i, "ffn_up_exps.weight"));
            L.ffn_down_exps      = bind_req(layer_name(i, "ffn_down_exps.weight"));
            L.ffn_gate_inp_shexp = bind_opt(layer_name(i, "ffn_gate_inp_shexp.weight"));
            L.ffn_gate_shexp     = bind_opt(layer_name(i, "ffn_gate_shexp.weight"));
            L.ffn_up_shexp       = bind_opt(layer_name(i, "ffn_up_shexp.weight"));
            L.ffn_down_shexp     = bind_opt(layer_name(i, "ffn_down_shexp.weight"));

            // post_attention_norm is present in qwen35moe too (ties the
            // hidden state through a second RMSNorm before the MoE FFN).
            L.attn_post_norm = bind_opt(layer_name(i, "post_attention_norm.weight"));

            if (!L.attn_norm
                || (attn_layer && (!L.wq || !L.wk || !L.wv || !L.wo))
                || (!attn_layer && (!L.gdn_qkv || !L.gdn_gate || !L.ssm_conv1d
                                    || !L.ssm_a || !L.ssm_alpha || !L.ssm_beta
                                    || !L.ssm_dt || !L.ssm_norm || !L.ssm_out))
                || !L.ffn_gate_inp || !L.ffn_gate_exps || !L.ffn_up_exps
                || !L.ffn_down_exps) {
                std::fprintf(stderr,
                    "[sp-engine] LlamaWeights: qwen35moe layer %d (%s) missing required tensor(s)\n",
                    i, attn_layer ? "attn" : "gdn");
                return false;
            }
            continue;
        }

        // --- phi3 path: fused QKV + packed SwiGLU FFN ------------------
        // Tensor layout (verified on phi-4-Q4_K_M and Phi-3.1-mini):
        //   blk.N.attn_norm.weight
        //   blk.N.attn_qkv.weight     — fused [Q|K|V] along the output-row axis
        //   blk.N.attn_output.weight
        //   blk.N.ffn_norm.weight
        //   blk.N.ffn_up.weight       — packed [gate|up] along the output-row axis
        //   blk.N.ffn_down.weight
        // No separate attn_q / attn_k / attn_v / ffn_gate. Biases are
        // absent on phi3 (GQA-lite, no bias variants). The layer kind
        // stays STANDARD so the forward dispatch goes through build_block;
        // build_block checks `attn_qkv != nullptr` / `ffn_gate == nullptr`
        // to switch to the fused paths.
        if (is_phi3) {
            L.kind       = LlamaLayerKind::STANDARD;
            L.attn_qkv   = bind_req(layer_name(i, "attn_qkv.weight"));
            L.wo         = bind_req(layer_name(i, "attn_output.weight"));
            L.ffn_norm   = bind_req(layer_name(i, "ffn_norm.weight"));
            L.ffn_up     = bind_req(layer_name(i, "ffn_up.weight"));
            L.ffn_down   = bind_req(layer_name(i, "ffn_down.weight"));
            // Optional bias terms (absent on stock phi3 but allowed by GGUF spec).
            L.bo         = bind_opt(layer_name(i, "attn_output.bias"));
            if (!L.attn_norm || !L.attn_qkv || !L.wo
                || !L.ffn_norm || !L.ffn_up || !L.ffn_down) {
                std::fprintf(stderr,
                    "[sp-engine] LlamaWeights: phi3 layer %d missing required tensor(s)\n", i);
                return false;
            }
            continue;
        }

        // --- Standard llama-family path (all other archs) --------------
        L.kind = LlamaLayerKind::STANDARD;
        L.wq        = bind_req(layer_name(i, "attn_q.weight"));
        L.wk        = bind_req(layer_name(i, "attn_k.weight"));
        L.wv        = bind_req(layer_name(i, "attn_v.weight"));
        L.wo        = bind_req(layer_name(i, "attn_output.weight"));

        L.attn_q_norm = bind_opt(layer_name(i, "attn_q_norm.weight"));
        L.attn_k_norm = bind_opt(layer_name(i, "attn_k_norm.weight"));

        L.bq        = bind_opt(layer_name(i, "attn_q.bias"));
        L.bk        = bind_opt(layer_name(i, "attn_k.bias"));
        L.bv        = bind_opt(layer_name(i, "attn_v.bias"));
        L.bo        = bind_opt(layer_name(i, "attn_output.bias"));

        L.ffn_norm  = bind_req(layer_name(i, "ffn_norm.weight"));
        L.ffn_gate  = bind_req(layer_name(i, "ffn_gate.weight"));
        L.ffn_up    = bind_req(layer_name(i, "ffn_up.weight"));
        L.ffn_down  = bind_req(layer_name(i, "ffn_down.weight"));

        // Gemma3 sandwich norms. Optional — stay nullptr for llama/qwen/etc.
        L.attn_post_norm = bind_opt(layer_name(i, "post_attention_norm.weight"));
        L.ffn_post_norm  = bind_opt(layer_name(i, "post_ffw_norm.weight"));

        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo
            || !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            std::fprintf(stderr,
                "[sp-engine] LlamaWeights: layer %d missing required tensor(s)\n", i);
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------
// Load path A: CPU mmap (current behaviour).
// Tensor data stays in the GGUF file's memory map; zero-copy for
// unquantised tensors. Fast load, no extra VRAM.
// ------------------------------------------------------------------
std::unique_ptr<LlamaWeights> LlamaWeights::load_cpu_mmap_(const Model& model) {
    auto w = std::unique_ptr<LlamaWeights>(new LlamaWeights());
    w->arch_ = model.architecture();

    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx      = &w->impl_->meta_ctx;
    w->impl_->gguf = gguf_init_from_file(model.path().c_str(), params);
    if (!w->impl_->gguf) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights: failed to reopen %s (mmap)\n",
            model.path().c_str());
        return nullptr;
    }

    if (!LlamaWeights::bind_tensors_(*w, w->impl_->meta_ctx, model)) {
        return nullptr;
    }
    return w;
}

// ------------------------------------------------------------------
// Load path B: backend-resident weights.
//
// Step 1: open the GGUF twice.
//   Pass 1 with no_alloc=false gives us an mmap-backed ctx we can
//   read tensor data from.
//   Pass 2 with no_alloc=true gives us a parallel ctx with
//   "placeholder" tensors (shape + type + name, no buffer) that we
//   can allocate on the target backend.
//
// Step 2: ggml_backend_alloc_ctx_tensors(final_ctx, backend) allocates
//   a single backend buffer large enough for every weight.
//
// Step 3: for each tensor, ggml_backend_tensor_set copies the mmap
//   bytes into the backend buffer (cudaMemcpy host → device for CUDA).
//
// Step 4: free the mmap ctx + gguf (data has been copied out). Keep
//   final_ctx + the backend_buf for the life of the LlamaWeights.
//
// The bind pass then runs on final_ctx so tok_embd / layers[] point
// at backend-resident tensors.
// ------------------------------------------------------------------
std::unique_ptr<LlamaWeights> LlamaWeights::load_backend_offload_(
        const Model& model, ggml_backend_t backend, int n_gpu_layers) {
    auto w = std::unique_ptr<LlamaWeights>(new LlamaWeights());
    w->arch_ = model.architecture();
    const int n_layer_total = (int)model.n_layer();
    const bool full_offload = n_gpu_layers >= n_layer_total;

    // Partial offload: layers [0, n_gpu_layers) go to GPU, the rest
    // plus non-layer tensors (output, token_embd, etc.) stay CPU-mmap.
    // ForwardContext builds a ggml_backend_sched_t that handles cross-
    // backend copies transparently, so mixed residency works.
    const bool backend_is_gpu = [&]{
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        return dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
    }();
    if (!full_offload && backend_is_gpu) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights::load: partial offload — "
            "layers [0,%d) on GPU, [%d,%d) + non-layer on CPU. "
            "ForwardContext scheduler handles cross-backend copies.\n",
            n_gpu_layers, n_gpu_layers, n_layer_total);
    }

#ifdef SP_ENGINE_WITH_CUDA
    auto dump_vram = [](const char* tag) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            std::fprintf(stderr,
                "[sp-engine:diag] %-40s VRAM free=%.2f GiB used=%.2f GiB / total=%.2f GiB\n",
                tag,
                free_b / (1024.0 * 1024.0 * 1024.0),
                (total_b - free_b) / (1024.0 * 1024.0 * 1024.0),
                total_b / (1024.0 * 1024.0 * 1024.0));
        }
    };
    dump_vram("load_backend_offload_: enter");
#endif

    // Pass 1: mmap copy.
    ggml_context* mmap_ctx = nullptr;
    gguf_init_params p1 = {};
    p1.no_alloc = false;
    p1.ctx      = &mmap_ctx;
    gguf_context* mmap_gguf = gguf_init_from_file(model.path().c_str(), p1);
    if (!mmap_gguf) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights: failed to reopen %s (pass 1)\n",
            model.path().c_str());
        return nullptr;
    }

    // Count tensors for ctx overhead budget, collect their metadata.
    struct TensorInfo {
        std::string name;
        ggml_type   type;
        int64_t     ne[GGML_MAX_DIMS];
        int         n_dims;
        size_t      n_bytes;
        void*       src;   // host pointer to mmapped bytes
    };
    // IMPORTANT: gguf_init_from_file populates mmap_ctx with BOTH the
    // named weight tensors AND a single "GGUF tensor data binary blob"
    // I8 tensor of the full file payload size. Walking the ggml_context
    // with ggml_get_first/next_tensor picks up both, doubling our
    // allocation request. Iterate via the gguf API instead — it lists
    // only the named weights.
    std::vector<TensorInfo> tensors;
    const int64_t n_gguf_tensors = gguf_get_n_tensors(mmap_gguf);
    tensors.reserve((size_t)n_gguf_tensors);
    std::unordered_map<int, size_t> bytes_by_type;
    std::unordered_map<int, int>    count_by_type;
    size_t src_total = 0;
    for (int64_t i = 0; i < n_gguf_tensors; ++i) {
        const char* name = gguf_get_tensor_name(mmap_gguf, i);
        ggml_tensor* t = ggml_get_tensor(mmap_ctx, name);
        if (!t) {
            std::fprintf(stderr,
                "[sp-engine] LlamaWeights: gguf index %lld name '%s' not in ggml ctx\n",
                (long long)i, name ? name : "(null)");
            continue;
        }
        TensorInfo ti;
        ti.name   = name;
        ti.type   = t->type;
        ti.n_dims = ggml_n_dims(t);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) ti.ne[d] = t->ne[d];
        ti.n_bytes = ggml_nbytes(t);
        ti.src     = t->data;
        src_total += ti.n_bytes;
        bytes_by_type[(int)ti.type] += ti.n_bytes;
        count_by_type[(int)ti.type] += 1;
        tensors.push_back(std::move(ti));
    }
    std::fprintf(stderr,
        "[sp-engine:diag] gguf named tensors: %zu, sum(ggml_nbytes)=%.2f MiB\n",
        tensors.size(), src_total / (1024.0 * 1024.0));
    for (const auto& kv : bytes_by_type) {
        std::fprintf(stderr,
            "[sp-engine:diag]   type=%d count=%d bytes=%.2f MiB\n",
            kv.first, count_by_type[kv.first], kv.second / (1024.0 * 1024.0));
    }
#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("after mmap_ctx walk");
#endif

    // Pass 2: create a final ctx we own, with tensor placeholders.
    const size_t ctx_mem = ggml_tensor_overhead() * (tensors.size() + 16);
    ggml_init_params gip = {};
    gip.mem_size   = ctx_mem;
    gip.mem_buffer = nullptr;
    gip.no_alloc   = true;          // we'll allocate on `backend`
    w->impl_->ctx  = ggml_init(gip);
    if (!w->impl_->ctx) {
        std::fprintf(stderr, "[sp-engine] LlamaWeights: ggml_init failed (backend ctx)\n");
        ggml_free(mmap_ctx);
        gguf_free(mmap_gguf);
        return nullptr;
    }
    for (const auto& ti : tensors) {
        ggml_tensor* t;
        if      (ti.n_dims == 1) t = ggml_new_tensor_1d(w->impl_->ctx, ti.type, ti.ne[0]);
        else if (ti.n_dims == 2) t = ggml_new_tensor_2d(w->impl_->ctx, ti.type, ti.ne[0], ti.ne[1]);
        else if (ti.n_dims == 3) t = ggml_new_tensor_3d(w->impl_->ctx, ti.type, ti.ne[0], ti.ne[1], ti.ne[2]);
        else                     t = ggml_new_tensor_4d(w->impl_->ctx, ti.type, ti.ne[0], ti.ne[1], ti.ne[2], ti.ne[3]);
        ggml_set_name(t, ti.name.c_str());

        // Memory Gate: Check if we offload this tensor or keep it mapped on CPU.
        bool offload = true;
        int layer = -1;
        if (std::sscanf(ti.name.c_str(), "blk.%d.", &layer) == 1) {
            if (layer >= n_gpu_layers) offload = false;
        } else {
            // Non-layer tensors (head, token_embd, output_norm, rope_freqs)
            // only go to the backend when every layer is offloaded.
            if (!full_offload) offload = false;
        }

        if (!offload) {
            t->data = ti.src;
        }
    }

#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("before ggml_backend_alloc_ctx_tensors");
#endif
    // Allocate storage on the target backend.
    w->impl_->backend_buf = ggml_backend_alloc_ctx_tensors(w->impl_->ctx, backend);
    if (!w->impl_->backend_buf) {
        std::fprintf(stderr, "[sp-engine] LlamaWeights: alloc_ctx_tensors on backend failed\n");
        ggml_free(mmap_ctx);
        gguf_free(mmap_gguf);
        return nullptr;
    }
    const size_t backend_buf_size = ggml_backend_buffer_get_size(w->impl_->backend_buf);
    std::fprintf(stderr,
        "[sp-engine:diag] backend_buf_size=%.2f MiB (what ggml allocated on backend)\n",
        backend_buf_size / (1024.0 * 1024.0));
#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("after ggml_backend_alloc_ctx_tensors");
#endif

    // Copy each tensor's mmap bytes into the backend buffer.
    size_t total_bytes = 0;
    for (const auto& ti : tensors) {
        ggml_tensor* dst = ggml_get_tensor(w->impl_->ctx, ti.name.c_str());
        if (!dst || dst->data == ti.src) continue; // Un-offloaded tensors keep zero-copy mapped src!
        ggml_backend_tensor_set(dst, ti.src, 0, ti.n_bytes);
        total_bytes += ti.n_bytes;
    }
    std::fprintf(stderr,
        "[sp-engine] LlamaWeights: offloaded %zu tensors (%.2f MiB) to backend\n",
        tensors.size(), total_bytes / (1024.0 * 1024.0));
#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("after all tensor_set copies");
#endif

    // Pass-1 data has been copied into the backend buffer; release it,
    // OR if we kept CPU mmap gates open for un-offloaded layers, bind them to the impl so they survive!
    if (full_offload) {
        ggml_free(mmap_ctx);
        gguf_free(mmap_gguf);
    } else {
        w->impl_->meta_ctx = mmap_ctx;
        w->impl_->gguf = mmap_gguf;
    }
#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("after mmap ctx + gguf free");
#endif

    if (!LlamaWeights::bind_tensors_(*w, w->impl_->ctx, model)) {
        return nullptr;
    }
    return w;
}

// ------------------------------------------------------------------
// Public load: dispatches to CPU-mmap or backend-offload based on
// the backend argument. CPU-type backends use mmap for zero-copy
// load; GPU-type backends require the offload path.
// ------------------------------------------------------------------
std::unique_ptr<LlamaWeights> LlamaWeights::load(const Model& model,
                                                  ggml_backend_t backend,
                                                  int n_gpu_layers) {
    if (!supported_arch(model.architecture())) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights: unsupported arch '%s'\n",
            model.architecture().c_str());
        return nullptr;
    }

    // Route: if backend is nullptr OR it's a CPU backend, use mmap.
    // Otherwise offload to the backend's buffer type.
    bool use_mmap = (backend == nullptr);
    if (backend != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            use_mmap = true;
        }
    }
    return use_mmap ? load_cpu_mmap_(model) : load_backend_offload_(model, backend, n_gpu_layers);
}

void LlamaWeights::print_summary(std::FILE* f) const {
    std::fprintf(f, "Weights (arch=%s):\n", arch_.c_str());
    std::fprintf(f, "  bound tensors:      %d\n", n_bound_tensors_);
    std::fprintf(f, "  missing (optional): %d\n", n_missing_optional_);
    std::fprintf(f, "  layers:             %d\n", (int)layers_.size());
    // Layer-kind breakdown (relevant for hybrid archs like qwen35moe).
    int n_std = 0, n_moe_attn = 0, n_moe_gdn = 0;
    for (const auto& L : layers_) {
        switch (L.kind) {
            case LlamaLayerKind::STANDARD: ++n_std;      break;
            case LlamaLayerKind::MOE_ATTN: ++n_moe_attn; break;
            case LlamaLayerKind::MOE_GDN:  ++n_moe_gdn;  break;
        }
    }
    if (n_moe_attn || n_moe_gdn) {
        std::fprintf(f, "    standard:         %d\n", n_std);
        std::fprintf(f, "    MoE full-attn:    %d\n", n_moe_attn);
        std::fprintf(f, "    MoE gated-dnet:   %d\n", n_moe_gdn);
    }
    std::fprintf(f, "  tok_embd:           %s\n", tok_embd   ? "OK" : "MISSING");
    std::fprintf(f, "  output_norm:        %s\n", output_norm? "OK" : "MISSING");
    std::fprintf(f, "  output:             %s%s\n",
                 output ? "OK" : "MISSING",
                 (output == tok_embd && output) ? " (tied to tok_embd)" : "");
    std::fprintf(f, "  rope_freqs:         %s (optional)\n",
                 rope_freqs ? "OK" : "absent");
}

} // namespace sp::engine
