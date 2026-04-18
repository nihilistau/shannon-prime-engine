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

    // Backend selection.
    enum class Backend { CPU, CUDA, Vulkan };
    Backend     backend = Backend::CPU;
    int         n_gpu_layers = 0;  // 0 = CPU-only, otherwise offload count
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
