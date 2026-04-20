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
    // Paired metadata ggml_context from gguf_init_from_file — owned by
    // gguf, freed alongside it.
    ggml_context* meta_ctx = nullptr;
    gguf_context* gguf = nullptr;
    // When we offloaded to a non-CPU backend, this holds the allocated
    // buffer so the tensor data stays alive for the Weights' lifetime.
    ggml_backend_buffer* backend_buf = nullptr;
    ~Impl() {
        if (backend_buf) ggml_backend_buffer_free(backend_buf);
        if (ctx)         ggml_free(ctx);
        // meta_ctx is owned by the gguf_context we get from gguf_loader
        // (not by us) — we don't free it here; Model destructor handles it.
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
        "llama", "qwen2", "qwen3", "mistral3", "phi3", "granite", "gemma3"
    };
    return ok.count(a) != 0;
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

    for (int i = 0; i < n_layer; ++i) {
        LlamaLayer& L = w.layers_[(size_t)i];
        L.attn_norm = bind_req(layer_name(i, "attn_norm.weight"));
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
        const Model& model, ggml_backend_t backend) {
    auto w = std::unique_ptr<LlamaWeights>(new LlamaWeights());
    w->arch_ = model.architecture();

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
        if (!dst) continue;
        ggml_backend_tensor_set(dst, ti.src, 0, ti.n_bytes);
        total_bytes += ti.n_bytes;
    }
    std::fprintf(stderr,
        "[sp-engine] LlamaWeights: offloaded %zu tensors (%.2f MiB) to backend\n",
        tensors.size(), total_bytes / (1024.0 * 1024.0));
#ifdef SP_ENGINE_WITH_CUDA
    dump_vram("after all tensor_set copies");
#endif

    // Pass-1 data has been copied into the backend buffer; release it.
    ggml_free(mmap_ctx);
    gguf_free(mmap_gguf);
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
                                                  ggml_backend_t backend) {
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
    return use_mmap ? load_cpu_mmap_(model) : load_backend_offload_(model, backend);
}

void LlamaWeights::print_summary(std::FILE* f) const {
    std::fprintf(f, "Weights (arch=%s):\n", arch_.c_str());
    std::fprintf(f, "  bound tensors:      %d\n", n_bound_tensors_);
    std::fprintf(f, "  missing (optional): %d\n", n_missing_optional_);
    std::fprintf(f, "  layers:             %d\n", (int)layers_.size());
    std::fprintf(f, "  tok_embd:           %s\n", tok_embd   ? "OK" : "MISSING");
    std::fprintf(f, "  output_norm:        %s\n", output_norm? "OK" : "MISSING");
    std::fprintf(f, "  output:             %s%s\n",
                 output ? "OK" : "MISSING",
                 (output == tok_embd && output) ? " (tied to tok_embd)" : "");
    std::fprintf(f, "  rope_freqs:         %s (optional)\n",
                 rope_freqs ? "OK" : "absent");
}

} // namespace sp::engine
