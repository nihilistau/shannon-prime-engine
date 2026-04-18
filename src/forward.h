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

class ForwardContext {
public:
    // Construct a reusable compute context for a given model. ctx_size
    // defaults large enough for typical 1B–8B prefill up to ~512 tokens;
    // caller can override.
    static std::unique_ptr<ForwardContext> create(const Model&         model,
                                                   const LlamaWeights& weights,
                                                   int ctx_size_bytes = 512 * 1024 * 1024);

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
    bool forward_full(const std::vector<int32_t>& token_ids,
                      std::vector<float>& logits_flat,
                      int& out_n_vocab);

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
