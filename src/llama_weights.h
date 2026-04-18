// Shannon-Prime Engine — Llama-family weight binding
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Materialises the tensors of a llama-family GGUF (arch ∈ {llama, qwen2,
// qwen3, mistral3, phi3, granite}) into a ggml_context and exposes typed
// per-layer handles. The weight layout is identical across these archs
// modulo a handful of optional bias / norm tensors; a single struct
// covers them all.

#pragma once

#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace sp::engine {

class Model;

struct LlamaLayer {
    // Attention
    ggml_tensor* attn_norm   = nullptr;  // "blk.N.attn_norm.weight"
    ggml_tensor* wq          = nullptr;
    ggml_tensor* wk          = nullptr;
    ggml_tensor* wv          = nullptr;
    ggml_tensor* wo          = nullptr;
    // Optional per-head attention norms (qwen3)
    ggml_tensor* attn_q_norm = nullptr;
    ggml_tensor* attn_k_norm = nullptr;
    // Optional biases (qwen2, granite)
    ggml_tensor* bq          = nullptr;
    ggml_tensor* bk          = nullptr;
    ggml_tensor* bv          = nullptr;
    ggml_tensor* bo          = nullptr;

    // FFN
    ggml_tensor* ffn_norm    = nullptr;
    ggml_tensor* ffn_gate    = nullptr;
    ggml_tensor* ffn_up      = nullptr;
    ggml_tensor* ffn_down    = nullptr;
};

class LlamaWeights {
public:
    // Load all tensors from `model` into a ggml_context owned by the
    // returned object. Tensor DATA stays mmapped from the GGUF file;
    // the ggml_tensor* handles point into that mmap. Zero-copy for
    // unquantised tensors; quantised tensors also stay mmapped but the
    // engine dequantises into scratch at inference time.
    //
    // Returns nullptr on unsupported arch, missing required tensors,
    // or I/O error.
    static std::unique_ptr<LlamaWeights> load(const Model& model);

    ~LlamaWeights();
    LlamaWeights(const LlamaWeights&) = delete;
    LlamaWeights& operator=(const LlamaWeights&) = delete;

    // --- shared top-level tensors ---
    ggml_tensor* tok_embd   = nullptr;  // "token_embd.weight"
    ggml_tensor* output_norm = nullptr; // "output_norm.weight"
    ggml_tensor* output      = nullptr; // "output.weight" (falls back to tok_embd if tied)
    ggml_tensor* rope_freqs  = nullptr; // "rope_freqs.weight" (optional)

    const std::vector<LlamaLayer>& layers() const { return layers_; }
    int n_layer() const { return (int)layers_.size(); }

    // Accessors for downstream (attention / FFN) code that wants the raw
    // ggml_context for allocating intermediate tensors.
    ggml_context* ctx() const;

    // Summary — for the `info` verb to confirm binding worked.
    void print_summary(std::FILE* f) const;

private:
    LlamaWeights();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<LlamaLayer> layers_;
    std::string arch_;
    int n_bound_tensors_ = 0;
    int n_missing_optional_ = 0;
};

} // namespace sp::engine
