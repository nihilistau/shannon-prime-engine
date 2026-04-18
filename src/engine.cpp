// Shannon-Prime Engine — implementation skeleton
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"

#include "shannon_prime.h"
#include "ggml.h"

#include <cstdio>

namespace sp::engine {

struct Engine::Impl {
    Config     cfg;
    bool       loaded = false;

    // Future: ggml_context, model weights, sp_sqfree_cache_t / sp_shadow_cache_t,
    // tokenizer handle, compute graph, batch scheduler. Intentionally omitted
    // from the scaffolding — each lands in its own focused commit.
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

int Engine::load(const Config& cfg) {
    impl_->cfg = cfg;

    std::fprintf(stderr,
        "[sp-engine] scaffolding: load(%s) — skeleton only, no model loaded yet.\n",
        cfg.model_path.c_str());

    // Minimal proof-of-life: touch both dependencies so we know linkage works.
    (void)sp_vht2_forward_f32;        // from lib/shannon-prime
    // (void)ggml_init;               // from vendor/ggml — will wire on first real model-load path

    impl_->loaded = false;  // nothing actually loaded
    return 0;
}

float Engine::perplexity(const std::string& wikitext_path,
                         int n_chunks, bool verbose) {
    (void)wikitext_path; (void)n_chunks; (void)verbose;
    std::fprintf(stderr,
        "[sp-engine] perplexity() not implemented yet — stage 4 of the roadmap.\n");
    return -1.0f;
}

int Engine::generate(const std::string& prompt, int n_predict,
                     std::string& out) {
    (void)prompt; (void)n_predict;
    out = "[sp-engine] generate() not implemented yet.";
    return -1;
}

} // namespace sp::engine
