// Shannon-Prime Engine — public API
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#pragma once

#include <cstdint>
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

    // Hierarchical Vilenkin predictor — maximum compression path.
    // Uses Kronecker sub-projection as a small skeleton (~9% of pad_dim)
    // and a calibrated linear map to predict the remaining coefficients.
    // Requires calibration (first prefill). Mutually exclusive with sqfree.
    bool        hierarchical    = false;
    int         hier_level      = 0;       // 0 = auto (second-to-last prime grouping)
    int         hier_res_bits   = 2;       // 1-4 bits for target residuals
    std::string hier_skel_bits  = "5,5";   // Band bits for skeleton quantisation

    // Backend selection.
    enum class Backend { CPU, CUDA, Vulkan };
    Backend     backend = Backend::CPU;
    int         n_gpu_layers = 0;  // 0 = CPU-only, otherwise offload count

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
    int         cauchy_mode    = 0;
    int         cauchy_fixed_n = 512;
    float       params_b       = 0.0f;
};

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

    // Streaming generate. Placeholder — expand when sampling lands.
    int generate(const std::string& prompt, int n_predict,
                 std::string& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
