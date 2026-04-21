// Shannon-Prime Engine — Llama-family weight binding
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Materialises the tensors of a llama-family GGUF (arch ∈ {llama, qwen2,
// qwen3, mistral3, phi3, granite, gemma3, qwen35moe}) into a ggml_context
// and exposes typed per-layer handles. The layout is nearly identical
// across these archs modulo a handful of optional bias / norm tensors —
// a single struct covers them all:
//
//   • gemma3 adds sandwich norms (post_attention_norm, post_ffw_norm).
//   • phi3 (covers Phi-3 / Phi-3.1 / Phi-4) ships a FUSED QKV projection
//     as "blk.N.attn_qkv.weight" and a SwiGLU-PACKED up projection as
//     "blk.N.ffn_up.weight" (shape 2*n_ff, split into [gate | up] along
//     the output dim). No separate Wq/Wk/Wv or ffn_gate tensors exist.
//     The fused weight is bound to `attn_qkv`; the packed ffn_up is
//     bound to `ffn_up` with `ffn_gate` left nullptr (the builder
//     detects the packed layout by that).
//   • qwen35moe is a HYBRID model — every `full_attention_interval` layer
//     is standard attention (with multi-section RoPE); the rest are Gated
//     DeltaNet (linear attention with conv + stateful recurrence, its input
//     projections live under "attn_qkv"/"attn_gate" names in the GGUF).
//     Every layer has MoE FFN (top-k + sigmoid-gated shared expert) instead
//     of dense FFN. The MoE and GDN tensors stay nullptr for other archs.
//
// `LlamaLayer::kind` labels which combination is live so downstream forward
// builders can dispatch cleanly.

#pragma once

#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;

typedef struct ggml_backend * ggml_backend_t;

namespace sp::engine {

// Sentinel for "offload every layer". Passed to LlamaWeights::load when
// the caller wants the full-offload path (all blk.* and non-layer tensors
// land on the backend). Any value >= model.n_layer() is equivalent.
inline constexpr int N_GPU_LAYERS_ALL = 9999;

class Model;

enum class LlamaLayerKind : int {
    STANDARD     = 0,   // attn (Wq/Wk/Wv/Wo) + dense FFN (gate/up/down). All non-qwen35moe archs.
    MOE_ATTN     = 1,   // qwen35moe full-attention layers: standard Q/K/V/O + multi-section RoPE,
                        //                                   MoE FFN instead of dense.
    MOE_GDN      = 2,   // qwen35moe gated-deltanet layers: ssm_* + attn_qkv/attn_gate input projs,
                        //                                   MoE FFN instead of dense.
};

struct LlamaLayer {
    LlamaLayerKind kind = LlamaLayerKind::STANDARD;

    // --- Attention (STANDARD / MOE_ATTN layers) -------------------------
    ggml_tensor* attn_norm   = nullptr;  // "blk.N.attn_norm.weight"
    ggml_tensor* wq          = nullptr;
    ggml_tensor* wk          = nullptr;
    ggml_tensor* wv          = nullptr;
    ggml_tensor* wo          = nullptr;
    // phi3 fused QKV projection (alternative to separate wq/wk/wv).
    // When non-null, wq/wk/wv are nullptr and the builder does a single
    // matmul + view-split into Q/K/V along the output-row axis.
    // Output-row layout: [Q rows (n_head*head_dim) | K rows (n_head_kv*head_dim) | V rows (n_head_kv*head_dim)]
    ggml_tensor* attn_qkv    = nullptr;  // "blk.N.attn_qkv.weight" (phi3 only)
    // Optional per-head attention norms (qwen3, qwen35moe full-attn)
    ggml_tensor* attn_q_norm = nullptr;
    ggml_tensor* attn_k_norm = nullptr;
    // Optional biases (qwen2, granite)
    ggml_tensor* bq          = nullptr;
    ggml_tensor* bk          = nullptr;
    ggml_tensor* bv          = nullptr;
    ggml_tensor* bo          = nullptr;

    // --- Dense FFN (STANDARD layers only) ------------------------------
    ggml_tensor* ffn_norm    = nullptr;
    ggml_tensor* ffn_gate    = nullptr;
    ggml_tensor* ffn_up      = nullptr;
    ggml_tensor* ffn_down    = nullptr;

    // --- Gemma3 sandwich norms (optional) ------------------------------
    // Gemma3 applies an extra RMSNorm to each block's attention output
    // and FFN output BEFORE the residual add. Tensor names in GGUF:
    // "blk.N.post_attention_norm.weight" and "blk.N.post_ffw_norm.weight".
    ggml_tensor* attn_post_norm = nullptr;
    ggml_tensor* ffn_post_norm  = nullptr;

    // --- MoE FFN (MOE_ATTN / MOE_GDN layers) --------------------------
    // Router gate: (n_embd, n_expert). Softmax over expert scores, top-k
    // selected, weights renormalised (or scaled by expert_weights_scale).
    ggml_tensor* ffn_gate_inp       = nullptr;  // "blk.N.ffn_gate_inp.weight"
    // Expert bank (stacked along the last dim): 3D tensors of shape
    // (n_ff_expert, n_embd, n_expert) for up; (n_embd, n_ff_expert, n_expert) for down.
    ggml_tensor* ffn_gate_exps      = nullptr;
    ggml_tensor* ffn_up_exps        = nullptr;
    ggml_tensor* ffn_down_exps      = nullptr;
    // Shared expert (runs for every token, gated by a sigmoid scalar).
    ggml_tensor* ffn_gate_inp_shexp = nullptr;  // "blk.N.ffn_gate_inp_shexp.weight"  (n_embd,)
    ggml_tensor* ffn_gate_shexp     = nullptr;
    ggml_tensor* ffn_up_shexp       = nullptr;
    ggml_tensor* ffn_down_shexp     = nullptr;

    // --- Gated DeltaNet (MOE_GDN layers only) --------------------------
    // Input projections (the GGUF re-uses attention tensor names):
    //   attn_qkv  = "blk.N.attn_qkv.weight"   -> (n_embd, qkv_dim)   fused Q+K+V block (pre-conv)
    //   attn_gate = "blk.N.attn_gate.weight"  -> (n_embd, d_inner)   z (output gate)
    ggml_tensor* gdn_qkv          = nullptr;
    ggml_tensor* gdn_gate         = nullptr;
    // Stateful recurrence parameters:
    //   ssm_conv1d = (conv_kernel, conv_channels) — causal 1D conv over qkv
    //   ssm_a      = (n_v_heads,)                — log-eigenvalue diagonal
    //   ssm_alpha  = (n_embd,  n_v_heads)         — input-dep alpha projection
    //   ssm_beta   = (n_embd,  n_v_heads)         — input-dep beta projection
    //   ssm_dt     = (n_v_heads,)                — delta-t bias
    //   ssm_norm   = (head_v_dim,)                — gated-norm weight
    //   ssm_out    = (d_inner, n_embd)            — output projection
    ggml_tensor* ssm_conv1d       = nullptr;
    ggml_tensor* ssm_a            = nullptr;
    ggml_tensor* ssm_alpha        = nullptr;
    ggml_tensor* ssm_beta         = nullptr;
    ggml_tensor* ssm_dt           = nullptr;
    ggml_tensor* ssm_norm         = nullptr;
    ggml_tensor* ssm_out          = nullptr;
};

class LlamaWeights {
public:
    // Load all tensors from `model` into a ggml_context owned by the
    // returned object.
    //
    // * `backend == nullptr` (default) — tensor DATA stays mmapped from
    //   the GGUF file; the ggml_tensor* handles point into that mmap.
    //   Zero-copy for unquantised tensors; quantised tensors also stay
    //   mmapped but the engine dequantises into scratch at inference
    //   time. Best for CPU compute.
    //
    // * `backend != nullptr` — allocates backing storage for every
    //   weight tensor on `backend`'s buffer type and copies the mmapped
    //   GGUF data into it via ggml_backend_tensor_set. Required when
    //   the forward pass runs on a GPU backend (CUDA / Vulkan) — the
    //   GPU kernels can't dereference CPU mmap pointers.
    //
    // Returns nullptr on unsupported arch, missing required tensors,
    // or I/O error.
    static std::unique_ptr<LlamaWeights> load(const Model& model,
                                               ggml_backend_t backend = nullptr,
                                               int n_gpu_layers = N_GPU_LAYERS_ALL);

    // Multi-GPU variant: distribute layers across multiple GPU backends.
    // Layer L → backends[L * n_gpus / n_layer]. Non-layer tensors → backends[0]
    // when fully offloaded, CPU-mapped otherwise.
    // Returns nullptr if any backend init or allocation fails.
    static std::unique_ptr<LlamaWeights> load_multi_gpu(
            const Model& model,
            const std::vector<ggml_backend_t>& gpu_backends,
            int n_gpu_layers = N_GPU_LAYERS_ALL);

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

    // Internal helpers (implementation-only; declared here so they can
    // reach private members without a friend declaration).
    static bool                             bind_tensors_(LlamaWeights& w, ggml_context* tctx, const Model& model);
    static std::unique_ptr<LlamaWeights>    load_cpu_mmap_(const Model& model);
    static std::unique_ptr<LlamaWeights>    load_backend_offload_(const Model& model, ggml_backend_t backend, int n_gpu_layers);
    static std::unique_ptr<LlamaWeights>    load_multi_gpu_(const Model& model,
                                                             const std::vector<ggml_backend_t>& gpu_backends,
                                                             int n_gpu_layers);
};

} // namespace sp::engine
