// Shannon-Prime Engine — library-level Engine binding.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"

#include "forward.h"
#include "forward_native_context.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "tokenizer.h"
#include "vocab.h"

#include "shannon_prime.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sp::engine {

namespace {

int argmax(const float* v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; ++i) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

// Compute Shannon entropy H = -Σ p_i log(p_i) from raw logits using
// the log-sum-exp trick for numerical stability. Result is in nats.
float logit_entropy(const float* logits, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_exp += std::exp((double)(logits[i] - mx));
    }
    const double log_Z = (double)mx + std::log(sum_exp);
    // H = log_Z - (1/Z) * Σ logits_i * exp(logits_i - mx)
    double weighted_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        weighted_sum += (double)logits[i] * std::exp((double)(logits[i] - mx));
    }
    const double H = log_Z - weighted_sum / sum_exp;
    return (float)H;
}
} // anon

struct Engine::Impl {
    Config                             cfg;
    bool                               loaded        = false;
    ggml_backend_t                     backend       = nullptr;  // primary GPU (or nullptr for CPU)
    std::vector<ggml_backend_t>        gpu_backends;             // all GPUs (multi-GPU path)
    std::unique_ptr<Model>             model;
    std::unique_ptr<Vocab>             vocab;
    std::unique_ptr<Tokenizer>         tok;
    std::unique_ptr<LlamaWeights>      weights;
    std::unique_ptr<ForwardContext>    fc;
    // Phase 4: native forward path (no ggml graph). Constructed in
    // load() when SP_ENGINE_NATIVE=1; used in generate() in place of
    // fc->prefill / fc->decode. Mutually exclusive with fc for now.
    std::unique_ptr<ForwardNativeContext> fnc;
    bool                                  native_active = false;

    ~Impl() {
        fnc.reset();
        fc.reset();
        weights.reset();
        tok.reset();
        vocab.reset();
        model.reset();
        // Free GPU backends. In multi-GPU mode, backend == gpu_backends[0].
        for (auto* b : gpu_backends) {
            if (b) ggml_backend_free(b);
        }
        gpu_backends.clear();
        backend = nullptr;  // already freed via gpu_backends
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

    // ── GPU enumeration ─────────────────────────────────────────────
    //
    // When cfg.backend is GPU/CUDA/Vulkan, enumerate available GPUs.
    // cfg.n_gpus controls how many we actually use:
    //   0 = auto (use all available)
    //   1 = single GPU (classic path)
    //   N = use N GPUs (clamped to available)
    //
    if (cfg.backend != Config::Backend::CPU) {
        const size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;

            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (!b) {
                std::fprintf(stderr,
                    "[sp-engine] Engine: failed to init GPU device %zu (%s)\n",
                    i, ggml_backend_dev_name(dev));
                continue;
            }
            size_t free_mem = 0, total_mem = 0;
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            std::fprintf(stderr,
                "[sp-engine] Engine: GPU %zu: %s — %.2f / %.2f GiB\n",
                impl_->gpu_backends.size(),
                ggml_backend_dev_description(dev),
                free_mem / (1024.0 * 1024.0 * 1024.0),
                total_mem / (1024.0 * 1024.0 * 1024.0));
            impl_->gpu_backends.push_back(b);
        }

        // Clamp to requested count.
        int want = cfg.n_gpus;
        if (want == 0) want = (int)impl_->gpu_backends.size();  // auto = all
        if (want <= 0) want = 1;
        while ((int)impl_->gpu_backends.size() > want) {
            ggml_backend_free(impl_->gpu_backends.back());
            impl_->gpu_backends.pop_back();
        }

        if (impl_->gpu_backends.empty()) {
            std::fprintf(stderr,
                "[sp-engine] Engine: no GPU devices found; falling back to CPU\n");
        } else {
            impl_->backend = impl_->gpu_backends[0];  // primary = GPU 0
        }
    }

    const bool multi_gpu = (impl_->gpu_backends.size() > 1);
    if (multi_gpu) {
        std::fprintf(stderr,
            "[sp-engine] Engine: multi-GPU mode — %zu GPUs\n",
            impl_->gpu_backends.size());
    }

    impl_->model = Model::load(cfg.model_path);
    if (!impl_->model) return 2;

    impl_->vocab = Vocab::load(*impl_->model);
    if (!impl_->vocab) return 3;

    impl_->tok = Tokenizer::create(*impl_->vocab);
    if (!impl_->tok) return 4;

    const int n_gpu_layers = (cfg.n_gpu_layers > 0) ? cfg.n_gpu_layers : N_GPU_LAYERS_ALL;

    // ── Weight loading ──────────────────────────────────────────────
    if (multi_gpu) {
        impl_->weights = LlamaWeights::load_multi_gpu(
            *impl_->model, impl_->gpu_backends, n_gpu_layers);
    } else {
        impl_->weights = LlamaWeights::load(
            *impl_->model, impl_->backend, n_gpu_layers);
    }
    if (!impl_->weights) return 5;

    // ── ForwardContext ───────────────────────────────────────────────
    PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
    if (multi_gpu) {
        impl_->fc = ForwardContext::create_multi_gpu(
            *impl_->model, *impl_->weights, impl_->gpu_backends,
            /*ctx_size_bytes*/ 1024 * 1024 * 1024, pe);
    } else {
        impl_->fc = ForwardContext::create(
            *impl_->model, *impl_->weights,
            /*ctx_size_bytes*/ 1024 * 1024 * 1024,
            pe, impl_->backend);
    }
    if (!impl_->fc) return 6;

    // ── CRT multi-GPU tensor splitting (Beast Canyon) ──────────────
    if (cfg.crt_split) {
        const int max_dim = impl_->model->n_embd();
        if (impl_->fc->enable_crt(max_dim)) {
            std::fprintf(stderr, "[sp-engine] CRT multi-GPU enabled (max_dim=%d)\n", max_dim);
        } else {
            std::fprintf(stderr, "[sp-engine] CRT init failed — falling back to standard matmul\n");
        }
    }

    // ── MoE expert curriculum (Beast Canyon homeostatic balancer) ────
    if (cfg.moe_curriculum) {
        if (impl_->fc->enable_moe_curriculum()) {
            std::fprintf(stderr, "[sp-engine] MoE curriculum active\n");
        } else {
            std::fprintf(stderr, "[sp-engine] MoE curriculum not available (non-MoE model?)\n");
        }
    }

    // ── Native forward (Phase 4) ────────────────────────────────────
    // When SHANNON_PRIME_NATIVE=1 in env (or SP_ENGINE_NATIVE=1, both
    // accepted), build a parallel ForwardNativeContext that drives
    // the layer step without ggml's graph machinery. Engine::generate
    // will use it in place of fc->prefill / fc->decode.
    {
        const char* a = std::getenv("SHANNON_PRIME_NATIVE");
        const char* b = std::getenv("SP_ENGINE_NATIVE");
        const bool want_native =
            (a && a[0] == '1') || (b && b[0] == '1');
        if (want_native) {
            impl_->fnc = ForwardNativeContext::create(
                *impl_->model, *impl_->weights);
            if (impl_->fnc) {
                impl_->native_active = true;
                std::fprintf(stderr,
                    "[sp-engine] native forward path active "
                    "(SHANNON_PRIME_NATIVE=1)\n");
            } else {
                std::fprintf(stderr,
                    "[sp-engine] SHANNON_PRIME_NATIVE=1 set but "
                    "ForwardNativeContext::create returned null — "
                    "falling back to ggml ForwardContext\n");
            }
        }
    }

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

    // ── Cache creation ───────────────────────────────────────────
    //
    // System 1↔2 mode: create a DualKvCache that holds both a ship-path
    // (System 1) and a hier/sqfree (System 2) cache. Entropy-gated
    // routing is performed per decode step in the loop below.
    //
    // Standard mode: prefer GPU-resident ship/sqfree cache when the
    // engine holds a non-CPU backend. Falls back to the CPU shadow cache
    // on any failure so generate() remains usable if CUDA is misconfigured.
    std::unique_ptr<KvCache> kv;
    std::unique_ptr<DualKvCache> dual;
    bool backend_is_gpu = false;
    if (impl_->backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(impl_->backend);
        backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
    }

    if (impl_->cfg.system12) {
        // System 1↔2 dual cache. Prefill always goes through System 1
        // (ship path — all prefill tokens are bulk-written, entropy is
        // computed per decode step only). During decode, each position is
        // routed by entropy.
        if (backend_is_gpu) {
            dual = DualKvCache::create_gpu(n_layer, n_head_kv, head_dim, max_seq,
                                            impl_->cfg, impl_->cfg.s12_sys2,
                                            impl_->cfg.s12_threshold,
                                            /*stream=*/nullptr);
        }
        if (!dual) {
            dual = DualKvCache::create(n_layer, n_head_kv, head_dim, max_seq,
                                        impl_->cfg, impl_->cfg.s12_sys2,
                                        impl_->cfg.s12_threshold);
        }
        if (!dual) {
            std::fprintf(stderr,
                "[sp-engine] Engine::generate: DualKvCache create failed\n");
            return -1;
        }
        // Bind System 1 for prefill (all prefill tokens use ship path).
        // Route all prefill positions to System 1.
        for (int p = 0; p < (int)ids.size(); ++p) {
            dual->route_position(p, 0.0f);  // 0 entropy → System 1
        }
        impl_->fc->bind_cache(dual->sys1());
    } else {
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
    }

    std::vector<float> last_logits;
    int n_vocab_out = 0;
    const auto t_prefill_start = std::chrono::steady_clock::now();
    const bool use_native = impl_->native_active && impl_->fnc;
    if (use_native) {
        if (!impl_->fnc->prefill(ids, last_logits, n_vocab_out)) {
            std::fprintf(stderr, "[sp-engine] native prefill failed\n");
            return -1;
        }
    } else {
        if (!impl_->fc->prefill(ids, last_logits, n_vocab_out)) {
            std::fprintf(stderr, "[sp-engine] Engine::generate: prefill failed\n");
            return -1;
        }
    }
    const auto t_prefill_end = std::chrono::steady_clock::now();
    const double t_prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();
    const double prefill_tps = (t_prefill_ms > 0)
        ? (double)ids.size() * 1000.0 / t_prefill_ms : 0.0;
    std::fprintf(stderr,
        "[sp-engine:perf] prefill = %d tokens in %.1f ms (%.2f t/s)\n",
        (int)ids.size(), t_prefill_ms, prefill_tps);

    // System 1↔2: if hier System 2 cache needs calibration, calibrate
    // it now using the same prefill data. The System 1 cache was already
    // calibrated by prefill(). We need to run calibration on System 2
    // separately — feed K vectors from the prefill into sys2's calibrator.
    // For simplicity, we skip this in generate() (the library API) and
    // note that the CLI chat verb handles it. The hier cache will operate
    // un-calibrated (using its default identity predictor) which is still
    // valid, just not optimal.

    const int32_t eos = impl_->vocab->eos_id();
    std::vector<int32_t> generated;
    generated.reserve((size_t)n_predict);

    int sys2_routed = 0;
    int32_t tok_id = argmax(last_logits.data(), n_vocab_out);
    const auto t_decode_start = std::chrono::steady_clock::now();
    int decoded_count = 0;
    for (int i = 0; i < n_predict; ++i) {
        generated.push_back(tok_id);
        if (eos >= 0 && tok_id == eos) break;

        // System 1↔2: compute entropy of current logits to decide which
        // cache stores the NEXT token's K/V. The intuition: if the model
        // is uncertain now, the next token's context is "hard" and
        // deserves maximum-fidelity compression.
        if (dual) {
            const float H = logit_entropy(last_logits.data(), n_vocab_out);
            const int pos = (int)ids.size() + i;  // sequence position of next token
            const int routed = dual->route_position(pos, H);
            sys2_routed += routed;
            // Bind the routed cache for this decode step. The decode()
            // function reads past K/V and writes new K/V to the bound
            // cache. We need the NEW K/V to go to the routed cache, but
            // past K/V must come from the merged view.
            //
            // For now, bind System 1 always (it holds prefill + most
            // positions). The write-back in the decode loop inside
            // forward.cpp writes to whatever cache is bound. We'll
            // intercept at write time via the DualKvCache.
            //
            // Actually, the right approach: we can't easily reroute the
            // internal write inside decode(). Instead, we always decode
            // through System 1 (bound), and then copy the new K/V to
            // System 2 when routed. This is simpler and correct.
            //
            // TODO: For the host-cache path, we could override the bound
            // cache per step. For now, System 1 handles all decode graph
            // computation; the routing only affects which cache provides
            // the ground-truth reconstruction on read.
        }

        std::vector<float> step_logits;
        int step_nv = 0;
        bool ok;
        if (use_native) {
            ok = impl_->fnc->decode(tok_id, step_logits, step_nv);
        } else {
            ok = impl_->fc->decode(tok_id, step_logits, step_nv);
        }
        if (!ok) {
            std::fprintf(stderr,
                "[sp-engine] Engine::generate: decode failed at step %d\n", i);
            break;
        }
        last_logits = std::move(step_logits);
        n_vocab_out = step_nv;
        tok_id = argmax(last_logits.data(), step_nv);
        ++decoded_count;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();
    const double t_decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();
    const double decode_tps = (t_decode_ms > 0 && decoded_count > 0)
        ? (double)decoded_count * 1000.0 / t_decode_ms : 0.0;
    std::fprintf(stderr,
        "[sp-engine:perf] decode  = %d tokens in %.1f ms (%.2f t/s)\n",
        decoded_count, t_decode_ms, decode_tps);

    if (dual && sys2_routed > 0) {
        std::fprintf(stderr,
            "[sp-engine] System 1↔2: %d/%d tokens routed to System 2 (%.1f%%)\n",
            sys2_routed, (int)generated.size(),
            100.0f * sys2_routed / (float)generated.size());
    }

    out = impl_->tok->decode(generated);
    return 0;
}

} // namespace sp::engine
