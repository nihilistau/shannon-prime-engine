// ForwardNativeContext — parallel to ForwardContext, drives the
// forward_native layer-step over a loaded GGUF model. No ggml graph.
//
// Activated by Engine::generate when SP_ENGINE_NATIVE=1; coexists with
// the existing ggml-based ForwardContext during validation. Once the
// native path is proven equivalent, ForwardContext becomes optional.

#pragma once

#include "forward_native.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sp::engine {

class Model;
class LlamaWeights;

class ForwardNativeContext {
public:
    // Bind to the loaded model + weights. Lifetime: the model and
    // weights MUST outlive this context (we hold raw ptrs into their
    // mmapped weight tensors).
    //
    // Returns nullptr if the model arch isn't supported by the native
    // path yet (currently: llama / qwen2 / qwen3 with dense FFN; MoE
    // and GDN archs go through the ggml path).
    static std::unique_ptr<ForwardNativeContext> create(
        const Model& model,
        const LlamaWeights& weights);

    ~ForwardNativeContext();

    // Run prefill over `ids` (full prompt). Writes the LAST-position
    // logits into `last_logits` (n_vocab fp32). Returns false on bad
    // shape, alloc failure, etc. Resets internal KV state — call
    // before any decode().
    bool prefill(const std::vector<int32_t>& ids,
                 std::vector<float>& last_logits,
                 int& n_vocab_out);

    // Single-token decode against the current KV state. Appends one
    // position. Returns logits[n_vocab].
    bool decode(int32_t tok,
                std::vector<float>& step_logits,
                int& n_vocab_out);

    // Reset KV / position state. Lets callers re-run prefill with a
    // fresh prompt without rebuilding the context.
    void reset();

    // Hyperparameters / shape info for the bound model.
    int n_layer()  const;
    int n_embd()   const;
    int n_vocab()  const;
    int head_dim() const;
    int n_head_kv() const;
    int max_seq()  const;

    // Forward-declared so .cpp helpers (run_layers, finalize_logits)
    // can take an Impl& by reference. The struct's full definition
    // lives in forward_native_context.cpp; consumers don't need to
    // see it.
    struct Impl;

private:
    ForwardNativeContext();
    std::unique_ptr<Impl> impl_;
};

}  // namespace sp::engine
