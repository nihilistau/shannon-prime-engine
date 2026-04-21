// Shannon-Prime Engine — public API
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace sp::engine {

struct Config {
    std::string model_path;   // GGUF on disk
    int         n_ctx   = 2048;
    int         n_batch = 512;

    // Shannon-Prime compression switches. One struct, no env-var scavenging —
    // the caller picks the composition explicitly.
    bool        sqfree      = false;   // Enable sqfree + Knight skeleton
    bool        spinor      = false;   // SU(2) sheet bit (requires sqfree)
    bool        mobius      = true;    // Ship-path Möbius reorder
    int         residual_bits = 3;     // Sqfree residual depth
    std::string k_bits_csv  = "5,5,4,3"; // Per-band K bit allocation
    std::string v_bits_csv  = "3";       // Per-band V (default flat)

    // Model-pack preset selection — arch-aware defaults.
    //   ""    / "off"  — use shipping defaults / explicit flags (default)
    //   "auto"         — resolve from model's GGUF arch_name at load time
    //   "<preset>"     — force a specific preset (e.g. "qwen3-moe")
    // Preset overlays apply only when k_bits_csv/v_bits_csv/residual_bits
    // are still at their shipping defaults — any explicit user flag wins.
    std::string model_preset = "";
    // Populated from GGUF general.architecture at model load; used by
    // KvCache::create_gpu when model_preset == "auto".
    std::string arch_name = "";

    // Hierarchical Vilenkin predictor — maximum compression path.
    // Uses Kronecker sub-projection as a small skeleton (~9% of pad_dim)
    // and a calibrated linear map to predict the remaining coefficients.
    // Requires calibration (first prefill). Mutually exclusive with sqfree.
    bool        hierarchical    = false;
    int         hier_level      = 0;       // 0 = auto (second-to-last prime grouping)
    int         hier_res_bits   = 2;       // 1-4 bits for target residuals
    std::string hier_skel_bits  = "5,5";   // Band bits for skeleton quantisation

    // Backend selection. `Vulkan` is planned (engine stage 7d); today it
    // falls through to whatever `ggml_backend_init_by_type(GPU)` picks.
    enum class Backend { CPU, CUDA, Vulkan };
    Backend     backend = Backend::CPU;
    // n_gpu_layers semantics (when backend != CPU):
    //   0 (default)       — use engine default (all layers; equivalent to
    //                       passing N_GPU_LAYERS_ALL to LlamaWeights::load)
    //   1..model.n_layer  — offload only the first N blocks; the rest plus
    //                       non-layer tensors (head, token_embd) stay CPU
    //   >= model.n_layer  — full offload (same as 0)
    // When backend == CPU, this field is ignored (mmap zero-copy load).
    int         n_gpu_layers = 0;

    // Positional-encoding mode. Default is Standard (geometric RoPE, no
    // ALiBi, byte-for-byte compatible with llama.cpp). Non-standard modes
    // implement the paper's "PrimePE-RoPE-ALiBi" family — lattice-drawn
    // frequencies + distance-based attention bias:
    //   PrimePe        geometric freqs replaced by lattice-drawn integer
    //                  freqs, blended with geometric at pe_alpha
    //   PrimePeAlibi   PrimePe + per-head ALiBi slopes added to attn scores
    //   AlibiOnly      standard RoPE + ALiBi (ablation)
    enum class PeMode { Standard, PrimePe, PrimePeAlibi, AlibiOnly };
    PeMode      pe_mode  = PeMode::Standard;
    float       pe_alpha = 0.17f;   // blend factor 0..1; paper's sweet spot 0.15–0.22
    int         pe_tier  = 0;       // 0 = composite lattice, 1 = prime generators

    // Cauchy reset system — decode-chain causal stability.
    //   mode 0 = off (default)
    //   mode 1 = fixed-N resets every `cauchy_fixed_n` tokens
    //   mode 2 = dynamic: Ricci sentinel (p=3 band energy drift) + Mertens
    //           oracle (zeta-zero-scheduled proactive reset points)
    // params_b is the model size in billions — used to tune Ricci threshold
    // (small models need tighter detection).
    int         cauchy_mode     = 0;
    int         cauchy_fixed_n  = 512;
    // Minimum positions between resets. Default 64 keeps reset rate
    // bounded (measured ~3-5 resets per chunk). Lower values allow
    // more frequent resets — each one pays a full forward_full pass,
    // so use with care until partial-reset lands.
    int         cauchy_cooldown = 64;
    // Initial delay before Cauchy is allowed to fire, counted from
    // the start of the decode loop within a chunk. The first N decode
    // positions post-prefill have low accumulated compression error —
    // resetting them is pure overhead. Measured sweet spot on Qwen3-8B
    // ctx=1024: warmup=64 (best PPL recovery), warmup=128 (same PPL,
    // fewer resets). warmup=256 starves the mechanism.
    int         cauchy_warmup   = 64;
    // Opt-in the Ricci drift sentinel. Default off — measured to
    // contribute 0 incremental PPL on Qwen3-8B-Q8 ctx=1024
    // (full system 11.92 ≡ Mertens-only 11.92; Ricci-only 12.02).
    // Enable for drift-diagnostic research.
    bool        cauchy_use_ricci = false;
    // Ablation: when true, Cauchy mode 2 skips the Mertens
    // (proactive zeta-schedule) layer and runs Ricci-only. Implies
    // cauchy_use_ricci. Useful for isolating the reactive layer.
    bool        cauchy_ricci_only = false;
    // Ablation flag kept for backward compat — this is now the
    // default behavior (no Ricci, Mertens only).
    bool        cauchy_mertens_only = false;
    float       params_b        = 0.0f;
};

// Seed Config fields from environment variables. Called by each CLI verb
// immediately after Config construction, so the precedence ordering stays:
//   Config default → env var → CLI flag.
// Only fills fields the env var owns; never clobbers a value already set
// by the caller (so tests that construct a Config programmatically aren't
// affected). Keep this header-inline so every call site picks it up
// without a link dependency.
inline void seed_config_from_env(Config& cfg) {
    if (cfg.model_preset.empty()) {
        if (const char* s = std::getenv("SHANNON_PRIME_MODEL_PRESET")) {
            cfg.model_preset = s;
        }
    }
}

class Engine {
public:
    Engine();
    ~Engine();

    // Load model + build compute graph. Returns 0 on success.
    int load(const Config& cfg);

    // Run perplexity over a tokenised input file. Returns PPL on success,
    // negative on error. Writes per-chunk values to stderr when verbose.
    float perplexity(const std::string& wikitext_path,
                     int n_chunks, bool verbose = false);

    // Greedy generate: tokenise prompt, prefill via ForwardContext with a
    // KvCache bound (compression mode controlled by cfg), then argmax-decode
    // n_predict tokens (or until EOS). Sampling temperature is zero; richer
    // sampling hooks would layer on top of ForwardContext::decode directly.
    int generate(const std::string& prompt, int n_predict,
                 std::string& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
