// Shannon-Prime Engine — Llama-family weight binding (implementation)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "llama_weights.h"
#include "gguf_loader.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace sp::engine {

struct LlamaWeights::Impl {
    ggml_context* ctx = nullptr;
    // Paired metadata ggml_context from gguf_init_from_file — owned by
    // gguf, freed alongside it.
    ggml_context* meta_ctx = nullptr;
    gguf_context* gguf = nullptr;
    ~Impl() {
        if (ctx)      ggml_free(ctx);
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
        "llama", "qwen2", "qwen3", "mistral3", "phi3", "granite"
    };
    return ok.count(a) != 0;
}

// ------------------------------------------------------------------
// Load
//
// Approach: we re-open the GGUF with gguf_init_from_file's `ctx`
// output parameter set, which makes gguf create a ggml_context with
// tensor placeholders pointing at the mmapped file data. Then we
// walk the expected tensor names per layer and bind them into our
// LlamaLayer structs via ggml_get_tensor(). Missing required tensors
// fail the load; missing optional ones just leave the field null.
// ------------------------------------------------------------------
std::unique_ptr<LlamaWeights> LlamaWeights::load(const Model& model) {
    if (!supported_arch(model.architecture())) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights: unsupported arch '%s'\n",
            model.architecture().c_str());
        return nullptr;
    }

    auto w = std::unique_ptr<LlamaWeights>(new LlamaWeights());
    w->arch_ = model.architecture();

    // Reopen the GGUF with a ggml_context allocator. We don't reuse the
    // Model's gguf_context here because that one was opened with
    // no_alloc=true and doesn't own tensor storage; this call mmaps
    // the file and materialises tensor handles.
    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx      = &w->impl_->meta_ctx;

    w->impl_->gguf = gguf_init_from_file(model.path().c_str(), params);
    if (!w->impl_->gguf) {
        std::fprintf(stderr,
            "[sp-engine] LlamaWeights: failed to reopen %s for tensor materialisation\n",
            model.path().c_str());
        return nullptr;
    }

    ggml_context* tctx = w->impl_->meta_ctx;

    auto bind_opt = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(tctx, name.c_str());
        if (t) w->n_bound_tensors_++;
        else   w->n_missing_optional_++;
        return t;
    };
    auto bind_req = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(tctx, name.c_str());
        if (!t) {
            std::fprintf(stderr,
                "[sp-engine] LlamaWeights: required tensor missing: %s\n",
                name.c_str());
            return nullptr;
        }
        w->n_bound_tensors_++;
        return t;
    };

    // Top-level
    w->tok_embd    = bind_req("token_embd.weight");
    w->output_norm = bind_req("output_norm.weight");
    w->output      = bind_opt("output.weight");
    if (!w->output) {
        // Tied embeddings: llama-3, some Qwens share the weight matrix.
        w->output = w->tok_embd;
    }
    w->rope_freqs  = bind_opt("rope_freqs.weight");

    if (!w->tok_embd || !w->output_norm) return nullptr;

    // Per-layer
    const int n_layer = (int)model.n_layer();
    w->layers_.resize((size_t)n_layer);

    auto layer_name = [](int i, const char* suffix) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
        return std::string(buf);
    };

    for (int i = 0; i < n_layer; ++i) {
        LlamaLayer& L = w->layers_[(size_t)i];
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

        // If any required tensor missed we bail with a clear message.
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo
            || !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            std::fprintf(stderr,
                "[sp-engine] LlamaWeights: layer %d missing required tensor(s)\n", i);
            return nullptr;
        }
    }

    return w;
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
