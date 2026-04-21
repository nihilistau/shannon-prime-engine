// Shannon-Prime Engine — library-level Engine binding.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"

#include "forward.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "tokenizer.h"
#include "vocab.h"

#include "shannon_prime.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sp::engine {

namespace {
ggml_backend_t select_backend_from_cfg(const Config& cfg) {
    if (cfg.backend == Config::Backend::CPU) return nullptr;
    ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!b) {
        std::fprintf(stderr,
            "[sp-engine] Engine: requested GPU backend not registered; using CPU\n");
    }
    return b;
}

int argmax(const float* v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; ++i) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}
} // anon

struct Engine::Impl {
    Config                             cfg;
    bool                               loaded        = false;
    ggml_backend_t                     backend       = nullptr;
    std::unique_ptr<Model>             model;
    std::unique_ptr<Vocab>             vocab;
    std::unique_ptr<Tokenizer>         tok;
    std::unique_ptr<LlamaWeights>      weights;
    std::unique_ptr<ForwardContext>    fc;

    ~Impl() {
        fc.reset();
        weights.reset();
        tok.reset();
        vocab.reset();
        model.reset();
        if (backend) ggml_backend_free(backend);
    }
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

int Engine::load(const Config& cfg) {
    impl_->cfg = cfg;

    // Scaffold fallback: the smoke test constructs a Config with an empty
    // model_path to verify linkage only. Don't treat that as an error.
    if (cfg.model_path.empty()) {
        std::fprintf(stderr,
            "[sp-engine] Engine::load: no model_path — scaffold mode, nothing loaded.\n");
        (void)sp_vht2_forward_f32;  // force-link shannon-prime core
        impl_->loaded = false;
        return 0;
    }

    impl_->backend = select_backend_from_cfg(cfg);

    impl_->model = Model::load(cfg.model_path);
    if (!impl_->model) return 2;

    impl_->vocab = Vocab::load(*impl_->model);
    if (!impl_->vocab) return 3;

    impl_->tok = Tokenizer::create(*impl_->vocab);
    if (!impl_->tok) return 4;

    const int n_gpu_layers = (cfg.n_gpu_layers > 0) ? cfg.n_gpu_layers : N_GPU_LAYERS_ALL;
    impl_->weights = LlamaWeights::load(*impl_->model, impl_->backend, n_gpu_layers);
    if (!impl_->weights) return 5;

    PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
    impl_->fc = ForwardContext::create(*impl_->model, *impl_->weights,
                                       /*ctx_size_bytes*/ 1024 * 1024 * 1024,
                                       pe, impl_->backend);
    if (!impl_->fc) return 6;

    impl_->loaded = true;
    return 0;
}

float Engine::perplexity(const std::string& wikitext_path,
                         int n_chunks, bool verbose) {
    if (!impl_->loaded) {
        std::fprintf(stderr, "[sp-engine] Engine::perplexity: not loaded\n");
        return -1.0f;
    }

    std::FILE* fp = std::fopen(wikitext_path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[sp-engine] Engine::perplexity: cannot open %s\n",
                     wikitext_path.c_str());
        return -1.0f;
    }
    std::fseek(fp, 0, SEEK_END);
    const size_t fsize = (size_t)std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string text((size_t)fsize, '\0');
    if (std::fread(text.data(), 1, fsize, fp) != fsize) {
        std::fclose(fp);
        return -1.0f;
    }
    std::fclose(fp);

    std::vector<int32_t> all_ids;
    impl_->tok->encode(text, /*add_bos=*/true, all_ids);

    const int n_ctx    = impl_->cfg.n_ctx > 0 ? impl_->cfg.n_ctx : 512;
    const int total    = (int)(all_ids.size() / (size_t)n_ctx);
    const int n_eval   = (n_chunks > 0 && n_chunks < total) ? n_chunks : total;
    if (n_eval <= 0) {
        std::fprintf(stderr,
            "[sp-engine] Engine::perplexity: text too short for ctx=%d\n", n_ctx);
        return -1.0f;
    }

    const int32_t bos = impl_->vocab->bos_id();
    std::vector<int32_t> chunk((size_t)n_ctx);
    std::vector<float>   logits;
    double total_nll = 0.0;
    long long total_evalled = 0;

    for (int c = 0; c < n_eval; ++c) {
        for (int t = 0; t < n_ctx; ++t) {
            chunk[(size_t)t] = all_ids[(size_t)(c * n_ctx + t)];
        }
        if (bos >= 0) chunk[0] = bos;

        int n_vocab_out = 0;
        if (!impl_->fc->forward_full(chunk, logits, n_vocab_out)) {
            std::fprintf(stderr,
                "[sp-engine] Engine::perplexity: forward_full failed at chunk %d\n", c);
            return -1.0f;
        }

        // NLL over positions [1..n_ctx-1] predicting the next token.
        double chunk_nll = 0.0;
        for (int t = 0; t < n_ctx - 1; ++t) {
            const float* row = logits.data() + (size_t)t * n_vocab_out;
            double mx = row[0];
            for (int i = 1; i < n_vocab_out; ++i) if (row[i] > mx) mx = row[i];
            double sumexp = 0.0;
            for (int i = 0; i < n_vocab_out; ++i) sumexp += std::exp(row[i] - mx);
            const double log_z = mx + std::log(sumexp);
            const int32_t next = chunk[(size_t)(t + 1)];
            chunk_nll += log_z - row[next];
        }
        total_nll     += chunk_nll;
        total_evalled += (n_ctx - 1);

        if (verbose) {
            const double ppl_so_far = std::exp(total_nll / (double)total_evalled);
            std::fprintf(stderr,
                "[sp-engine] chunk %d/%d  running PPL = %.4f\n",
                c + 1, n_eval, ppl_so_far);
        }
    }

    return (float)std::exp(total_nll / (double)total_evalled);
}

int Engine::generate(const std::string& prompt, int n_predict,
                     std::string& out) {
    out.clear();
    if (!impl_->loaded) {
        std::fprintf(stderr, "[sp-engine] Engine::generate: not loaded\n");
        return -1;
    }
    if (n_predict <= 0) return 0;

    std::vector<int32_t> ids;
    impl_->tok->encode(prompt, /*add_bos=*/true, ids);
    if (ids.empty()) {
        std::fprintf(stderr, "[sp-engine] Engine::generate: empty prompt tokenisation\n");
        return -1;
    }

    const int n_layer   = impl_->fc->n_layer();
    const int head_dim  = (int)impl_->model->head_dim();
    const int n_head_kv = (int)impl_->model->n_head_kv();
    const int max_seq   = ids.size() + (size_t)n_predict;

    // Prefer the GPU-resident ship/sqfree cache when the engine holds a
    // non-CPU backend and the config doesn't require a CPU-only mode
    // (hierarchical). Falls back to the CPU shadow cache on any failure
    // so generate() remains usable if CUDA is misconfigured. Matches the
    // CLI `chat` / `perplexity --cache` pattern in main.cpp.
    std::unique_ptr<KvCache> kv;
    bool backend_is_gpu = false;
    if (impl_->backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(impl_->backend);
        backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    const bool gpu_ok = !impl_->cfg.hierarchical;
    if (backend_is_gpu && gpu_ok) {
        kv = KvCache::create_gpu(n_layer, n_head_kv, head_dim, max_seq,
                                 impl_->cfg, /*stream=*/nullptr);
        if (!kv) {
            std::fprintf(stderr,
                "[sp-engine] Engine::generate: create_gpu failed; falling back to host cache\n");
        }
    }
    if (!kv) {
        kv = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, impl_->cfg);
    }
    if (!kv) {
        std::fprintf(stderr, "[sp-engine] Engine::generate: KvCache::create failed\n");
        return -1;
    }
    impl_->fc->bind_cache(kv.get());

    std::vector<float> last_logits;
    int n_vocab_out = 0;
    if (!impl_->fc->prefill(ids, last_logits, n_vocab_out)) {
        std::fprintf(stderr, "[sp-engine] Engine::generate: prefill failed\n");
        return -1;
    }

    const int32_t eos = impl_->vocab->eos_id();
    std::vector<int32_t> generated;
    generated.reserve((size_t)n_predict);

    int32_t tok_id = argmax(last_logits.data(), n_vocab_out);
    for (int i = 0; i < n_predict; ++i) {
        generated.push_back(tok_id);
        if (eos >= 0 && tok_id == eos) break;
        std::vector<float> step_logits;
        int step_nv = 0;
        if (!impl_->fc->decode(tok_id, step_logits, step_nv)) {
            std::fprintf(stderr,
                "[sp-engine] Engine::generate: decode failed at step %d\n", i);
            break;
        }
        tok_id = argmax(step_logits.data(), step_nv);
    }

    out = impl_->tok->decode(generated);
    return 0;
}

} // namespace sp::engine
