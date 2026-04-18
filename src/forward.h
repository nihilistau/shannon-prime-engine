// Shannon-Prime Engine — forward-pass graph builder
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Builds a ggml compute graph for a llama-family forward pass. This
// file gets extended stage by stage:
//   3a (current)  : token-embedding lookup only
//   3b (next)     : single transformer block (norm + attn + FFN + res)
//   3c           : full n_layer loop + output norm + logits
//   4            : swap standard fp16 KV cache for Shannon-Prime compressed
//
// The ctx-size and mem estimation live in the ForwardContext so the
// engine can allocate scratch once and re-run the graph per-decode.

#pragma once

#include "engine.h"   // PeMode lives on Config

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace sp::engine {

class LlamaWeights;
class Model;

// Narrow view of the positional-encoding knobs. Kept as a tiny struct
// so ForwardContext::create stays call-site-light (most callers pass
// the default = Standard RoPE, no ALiBi).
struct PeSettings {
    Config::PeMode pe_mode  = Config::PeMode::Standard;
    float          pe_alpha = 0.0f;
    int            pe_tier  = 0;
};

class ForwardContext {
public:
    // Construct a reusable compute context for a given model. ctx_size
    // defaults large enough for typical 1B–8B prefill up to ~512 tokens;
    // caller can override. `pe` selects the positional-encoding variant
    // (default Standard = geometric RoPE, no ALiBi — byte-for-byte
    // compatible with llama.cpp).
    static std::unique_ptr<ForwardContext> create(const Model&         model,
                                                   const LlamaWeights& weights,
                                                   int ctx_size_bytes = 512 * 1024 * 1024,
                                                   PeSettings pe = {});

    ~ForwardContext();
    ForwardContext(const ForwardContext&) = delete;
    ForwardContext& operator=(const ForwardContext&) = delete;

    // Run the 3a-stage graph: for `token_ids` of length n, produce an
    // (n, n_embd) fp32 tensor of token embeddings. Returns the flat
    // embedding values in row-major order (n * n_embd floats). Resets
    // the graph context each call — cheap enough for scaffolding.
    bool embed(const std::vector<int32_t>& token_ids,
               std::vector<float>& out_flat,
               int& out_n_embd);

    // Run the 3b-stage graph: prefill a sequence through ONE transformer
    // block (layer 0). Returns the post-block hidden states as an
    // (n, n_embd) fp32 flat array. No KV cache — self-attention is
    // computed among the prefill tokens in a single pass.
    //
    // This is a diagnostic entry point; stage 3c adds forward_full()
    // which loops over all layers and emits logits.
    bool forward_one_block(const std::vector<int32_t>& token_ids,
                           std::vector<float>& out_flat,
                           int& out_n_embd);

    // Run the 3c-stage graph: prefill through all n_layer transformer
    // blocks, then output_norm + output head → logits. Returns the
    // (n, n_vocab) fp32 logits flat.
    //
    // Stage 5a: optional per-layer K/V capture. When `per_layer_K` and
    // `per_layer_V` are non-null, the post-RoPE pre-GQA-broadcast K
    // and V tensors for every layer are pulled back into host vectors
    // in `[head_dim, n_head_kv, n]` order — exactly what
    // KvCache::write expects.
    bool forward_full(const std::vector<int32_t>& token_ids,
                      std::vector<float>& logits_flat,
                      int& out_n_vocab,
                      std::vector<std::vector<float>>* per_layer_K = nullptr,
                      std::vector<std::vector<float>>* per_layer_V = nullptr,
                      // Optional: post-block hidden state of layer 0,
                      // shape [n_embd, n_tokens]. Used to compare
                      // against decode's per-step layer-0 hidden.
                      std::vector<float>* dbg_X_layer0 = nullptr);

    // ---- Stage 5b: stateful prefill + decode over a compressed KV ----
    //
    // Bind a KvCache to this context. The cache is non-owning; the
    // caller must outlive the ForwardContext. Resets kv_pos to 0.
    void bind_cache(class KvCache* cache);

    // Run prefill: forward_full(token_ids) → write every layer's K/V
    // into the bound cache → advance kv_pos by token_ids.size().
    // Returns the logits of ONLY the last token (n_vocab floats), which
    // is all decode loops actually consume.
    bool prefill(const std::vector<int32_t>& token_ids,
                 std::vector<float>& last_logits,
                 int& out_n_vocab);

    // Run a single decode step: read past K/V from the bound cache,
    // project new Q/K/V for `token_id`, concat, attend, write the new
    // K/V back to the cache at slot kv_pos, advance kv_pos by 1.
    // Returns logits for the new position.
    //
    // `dbg_K_layer0` (optional out): receives the freshly-projected
    // post-RoPE K for layer 0 (size n_head_kv * head_dim) BEFORE it
    // is written to cache. Useful for comparing against forward_full's
    // K at the same position to localise decode-graph bugs.
    bool decode(int32_t token_id,
                std::vector<float>& logits,
                int& out_n_vocab,
                std::vector<float>* dbg_K_layer0 = nullptr,
                std::vector<float>* dbg_X_layer0 = nullptr);

    int kv_pos() const;

    // Hparams the caller set at create(); exposed for diagnostics.
    int n_embd()  const;
    int n_vocab() const;
    int n_layer() const;

private:
    ForwardContext();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
