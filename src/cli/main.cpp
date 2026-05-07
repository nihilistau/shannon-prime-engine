// Shannon-Prime Engine — sp-engine CLI
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"
#include "forward.h"
#include "gdn_state.h"
#include "gguf_loader.h"
#include "http_server.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "prime_pe.h"
#include "tokenizer.h"
#include "vocab.h"

#if defined(SP_ENGINE_WITH_QNN)
#include "qnn_bin_driver.h"
#include "speculative_oracle.h"
#endif

#if defined(SP_ENGINE_WITH_BEAST)
extern "C" {
#include "sp_beast_canyon.h"
#include "sp_optane.h"
}
#endif

#if defined(SP_ENGINE_HEXAGON_FASTRPC)
extern "C" {
#include "shannon_prime_hexagon.h"
#include "rpcmem.h"
}
#endif

#include <httplib.h>

extern "C" {
#include "shannon_prime.h"
#include "sp_crt.h"
}

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// RAII guard: owns an optional ggml_backend_t, frees on scope exit.
// Implicitly convertible to ggml_backend_t so it drops into the
// LlamaWeights / ForwardContext calls without extra .get() noise.
struct SpBackendGuard {
    ggml_backend_t b = nullptr;
    SpBackendGuard() = default;
    explicit SpBackendGuard(ggml_backend_t b) : b(b) {}
    ~SpBackendGuard() { if (b) ggml_backend_free(b); }
    SpBackendGuard(const SpBackendGuard&) = delete;
    SpBackendGuard& operator=(const SpBackendGuard&) = delete;
    operator ggml_backend_t() const { return b; }
};

// Compute Shannon entropy H = -Σ p_i log(p_i) from raw logits (nats).
// Uses log-sum-exp for numerical stability.
static float sp_logit_entropy(const float* logits, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_exp += std::exp((double)(logits[i] - mx));
    }
    const double log_Z = (double)mx + std::log(sum_exp);
    double weighted_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        weighted_sum += (double)logits[i] * std::exp((double)(logits[i] - mx));
    }
    return (float)(log_Z - weighted_sum / sum_exp);
}

// Backend selection helper shared across verbs: reads SP_ENGINE_BACKEND
// and returns a ggml_backend_t (caller owns, must ggml_backend_free).
// Falls through to nullptr for CPU (LlamaWeights / ForwardContext will
// then use their CPU default paths with mmap zero-copy load).
static ggml_backend_t sp_select_backend() {
    const char* env = std::getenv("SP_ENGINE_BACKEND");
    if (!env) return nullptr;
    const bool want_gpu = (std::strcmp(env, "gpu") == 0 ||
                           std::strcmp(env, "cuda") == 0 ||
                           std::strcmp(env, "vulkan") == 0);
    if (!want_gpu) return nullptr;
    ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!b) {
        std::fprintf(stderr, "[sp-engine] SP_ENGINE_BACKEND=%s but no GPU backend "
                     "registered; falling back to CPU\n", env);
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    std::fprintf(stderr, "[sp-engine] backend: %s (GPU)\n",
                 dev ? ggml_backend_dev_name(dev) : "?");
    return b;
}

// Multi-GPU enumeration: returns all GPU backends when n_gpus != 1.
// Caller owns the returned backends (must free each).
// When n_gpus == 0, auto-detects all. When n_gpus == 1 or no GPUs found,
// returns empty vector (caller falls through to single-GPU path).
struct SpMultiGpuGuard {
    std::vector<ggml_backend_t> backends;
    ~SpMultiGpuGuard() {
        for (auto* b : backends) {
            if (b) ggml_backend_free(b);
        }
    }
    SpMultiGpuGuard() = default;
    SpMultiGpuGuard(const SpMultiGpuGuard&) = delete;
    SpMultiGpuGuard& operator=(const SpMultiGpuGuard&) = delete;
};

static std::vector<ggml_backend_t> sp_enumerate_gpus(int n_gpus_requested) {
    if (n_gpus_requested == 1) return {};  // explicitly single-GPU

    // Check if GPU backend is requested via env.
    const char* env = std::getenv("SP_ENGINE_BACKEND");
    if (!env) return {};
    const bool want_gpu = (std::strcmp(env, "gpu") == 0 ||
                           std::strcmp(env, "cuda") == 0 ||
                           std::strcmp(env, "vulkan") == 0);
    if (!want_gpu) return {};

    std::vector<ggml_backend_t> gpus;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;

        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) continue;

        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        std::fprintf(stderr,
            "[sp-engine] GPU %zu: %s — %.2f / %.2f GiB\n",
            gpus.size(), ggml_backend_dev_description(dev),
            free_mem / (1024.0 * 1024.0 * 1024.0),
            total_mem / (1024.0 * 1024.0 * 1024.0));
        gpus.push_back(b);
    }

    // Clamp to requested count.
    int want = (n_gpus_requested == 0) ? (int)gpus.size() : n_gpus_requested;
    while ((int)gpus.size() > want) {
        ggml_backend_free(gpus.back());
        gpus.pop_back();
    }

    // Single GPU → return empty to use standard path.
    if (gpus.size() <= 1) {
        for (auto* b : gpus) ggml_backend_free(b);
        return {};
    }

    std::fprintf(stderr, "[sp-engine] multi-GPU mode: %zu GPUs\n", gpus.size());
    return gpus;
}

// Number of layers to offload to the backend. N_GPU_LAYERS_ALL ("all") is
// the default and matches the full-offload path that the CPU-mmap loader
// has always used. Env var SP_ENGINE_N_GPU_LAYERS sets the baseline;
// verb-local --n-gpu-layers flags may override further.
static int sp_default_n_gpu_layers() {
    if (const char* env = std::getenv("SP_ENGINE_N_GPU_LAYERS")) {
        const int v = std::atoi(env);
        if (v > 0) return v;
    }
    return sp::engine::N_GPU_LAYERS_ALL;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "sp-engine — Shannon-Prime reference inference engine\n"
        "\n"
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  version              Print version.\n"
        "  banner               Print banner + loaded submodule SHAs (sanity).\n"
        "  info --model <gguf>  Load a GGUF and print hparams + tensor summary.\n"
        "  encode --model <gguf> <text>  Tokenise text to IDs.\n"
        "  decode --model <gguf> <id1> [id2 ...]  Decode IDs to text.\n"
        "  embed  --model <gguf> <text>  Encode + run token-embedding lookup.\n"
        "  block1 --model <gguf> <text>  Run layer-0 transformer block forward.\n"
        "  logits --model <gguf> <text>  Run full forward pass, print logit stats.\n"
        "  kv_smoke [--sqfree] [--head-dim N] [--n-tokens N]\n"
        "                       Push synthetic K/V through compressed cache, report\n"
        "                       compression ratio + per-head correlation.\n"
        "  prefill --model <gguf> [--sqfree] <text>\n"
        "                       Run prefill, push real RoPE'd K/V through KvCache,\n"
        "                       report per-layer correlation (cache vs uncompressed).\n"
        "  chat --model <gguf> [--n-predict N] [--sqfree] <prompt>\n"
        "                       Greedy generation: prefill prompt, decode N tokens\n"
        "                       reading past K/V from the compressed cache.\n"
        "  perplexity --model <gguf> [--ctx N] [--chunks N] <textfile>\n"
        "                       Compute baseline PPL over a UTF-8 text file in\n"
        "                       contiguous N-token chunks. Hook-free reference.\n"
        "  cache_ppl --model <gguf> [--sqfree|--hierarchical] [--ctx N] <textfile>\n"
        "                       Run perplexity + compressed-cache correlation.\n"
        "                       Reports baseline PPL, K/V round-trip correlation,\n"
        "                       and compression ratio per chunk.\n"
        "  run --model <gguf> [--n-predict N] [--ctx N] <prompt>\n"
        "                       Library-API demo: loads via Engine class and does\n"
        "                       a greedy generate. For rich options use `chat`.\n"
        "\n"
        "Options:\n"
        "  --model <path.gguf>\n"
        "  --ctx <n>            default 2048\n"
        "  --n-gpu-layers <n>   (-ngl) layers to offload; default = all. Non-layer\n"
        "                       tensors (head, token_embd) go to backend only\n"
        "                       when n >= model.n_layer. Requires SP_ENGINE_BACKEND\n"
        "                       = gpu/cuda/vulkan; CPU loader ignores. Env var\n"
        "                       SP_ENGINE_N_GPU_LAYERS sets the default.\n"
        "  --n-gpus <n>         multi-GPU: use N GPUs (0=auto-detect, 1=single GPU)\n"
        "                       Layers are sharded: layer L → GPU[L*N/n_layer].\n"
        "                       (env: SP_ENGINE_N_GPUS)\n"
        "  --sqfree             enable sqfree + Knight skeleton\n"
        "  --spinor             enable SU(2) sheet bit (requires --sqfree)\n"
        "  --no-mobius          disable ship-path Möbius reorder\n"
        "  --k-bits <csv>       K band bits, e.g. 5,5,4,3\n"
        "  --v-bits <csv>       V band bits, default 3\n"
        "  --residual-bits <n>  sqfree residual bits, default 3\n"
        "  --model-preset <n>   model-pack overlay: auto | off | <arch-name>\n"
        "                       (auto resolves from GGUF general.architecture;\n"
        "                        explicit --k-bits/--v-bits/--residual-bits win)\n"
        "  --no-calibrate       skip automatic calibration during prefill\n"
        "\n"
        "Hot/cold tiered storage (GPU ↔ CPU offload):\n"
        "  --cold                enable cold storage (GPU→CPU mirror, unlimited)\n"
        "  --cold-mb <n>         enable cold storage with ring buffer of N MB\n"
        "  --evict-keep <n>      keep N recent positions on GPU, evict older ones\n"
        "                        (env: SP_ENGINE_COLD_MB, SP_ENGINE_EVICT_KEEP)\n"
        "\n"
        "Disk serialisation (cache save/load):\n"
        "  --save-cache <prefix> save compressed KV state after generation\n"
        "  --load-cache <prefix> load compressed KV state before generation\n"
        "                        (env: SP_ENGINE_SAVE_CACHE, SP_ENGINE_LOAD_CACHE)\n"
        "\n"
        "Hierarchical Vilenkin predictor (maximum compression):\n"
        "  --hierarchical       enable hierarchical predictor (~9%% skeleton)\n"
        "  --hier-level <n>     0 = auto, 1..n_primes-1 = explicit\n"
        "  --hier-res-bits <n>  K target residual bits, 1-4 (default: 2)\n"
        "  --hier-res-bits-v <n>  V target residual bits; 0 = same as K\n"
        "  --hier-skel-bits <csv>  skeleton band bits (default: 5,5)\n"
        "  --hier-ternary-mask <hex>  ternary band mask, e.g. 0x8 for band 3\n"
        "\n"
        "Main K/V ternary + FP8:\n"
        "  --ternary-k <hex>    ternary band mask for main K quant\n"
        "  --ternary-v <hex>    ternary band mask for main V quant\n"
        "  --fp8                use FP8 (E4M3) for V cache\n"
        "\n"
        "PrimePE-RoPE-ALiBi:\n"
        "  --pe-mode <name>     standard|primepe|primepe_alibi|alibi (default: standard)\n"
        "  --pe-alpha <f>       blend factor 0..1 (default: 0.0 = identity)\n"
        "  --pe-tier  <n>       0 = composite lattice, 1 = prime generators\n"
        "\n"
        "Cauchy reset system (decode-chain causal stability, cache_ppl/perplexity --cache):\n"
        "  --cauchy-mode <n>     0=off, 1=fixed-N, 2=dynamic (Ricci+Mertens)\n"
        "  --cauchy-fixed-n <n>  reset every N tokens (mode 1, default 512)\n"
        "  --cauchy-cooldown <n> min positions between resets (default 64)\n"
        "  --cauchy-warmup <n>   suppress resets for first N decode positions (default 64)\n"
        "  --cauchy-use-ricci    also enable reactive Ricci drift sentinel\n"
        "                        (default off — empirically contributes 0 extra PPL)\n"
        "  --cauchy-ricci-only   mode 2 without Mertens schedule (Ricci drift only, ablation)\n"
        "  --cauchy-mertens-only mode 2 without Ricci sentinel (now the default)\n"
        "  --params-b <f>        model size in billions (tunes Ricci threshold)\n"
        "\n"
        "System 1↔2 switching (entropy-gated dynamic cache routing):\n"
        "  --system12            enable dual-cache mode (ship + hier/sqfree)\n"
        "  --crt-split           CRT multi-GPU parallelism (needs n_gpus >= 2)\n"
        "  --moe-curriculum      MoE expert curriculum (EWMA heatmap + prefetch)\n"
        "  --beast <gguf>        Boot Beast Canyon orchestrator on Optane-mapped model\n"
        "  --s12-threshold <f>   entropy threshold in nats (default: 2.0)\n"
        "  --s12-sys2 <type>     System 2 cache type: hier (default) | sqfree\n"
        "                        (env: SP_ENGINE_SYSTEM12, SP_ENGINE_S12_THRESHOLD)\n"
        "\n", prog);
}

// ---------------------------------------------------------------------------
// Shared Config flag parser — one canonical parser for ALL engine flags.
// Returns the number of argv slots consumed (0 = not recognized, 1 = flag
// only, 2 = flag + value). Verbs call this in a loop; any leftover args
// are verb-specific (positional prompt, etc.).
// ---------------------------------------------------------------------------
static int parse_config_flag(sp::engine::Config& cfg, const char* a, const char* next) {
    auto a_eq = [&](const char* flag) { return std::strcmp(a, flag) == 0; };
    auto has_next = (next != nullptr);

    // -- Model / context --
    if (a_eq("--model")     && has_next) { cfg.model_path = next; return 2; }
    if (a_eq("--ctx")       && has_next) { cfg.n_ctx = std::atoi(next); return 2; }
    if (a_eq("--model-preset") && has_next) { cfg.model_preset = next; return 2; }

    // -- Compression mode --
    if (a_eq("--sqfree"))       { cfg.sqfree = true; cfg.hierarchical = false; return 1; }
    if (a_eq("--spinor"))       { cfg.spinor = true; cfg.sqfree = true; cfg.hierarchical = false; return 1; }
    if (a_eq("--hierarchical")) { cfg.hierarchical = true; cfg.sqfree = false; return 1; }
    if (a_eq("--no-mobius"))    { cfg.mobius = false; return 1; }
    if (a_eq("--k-bits")        && has_next) { cfg.k_bits_csv = next; return 2; }
    if (a_eq("--v-bits")        && has_next) { cfg.v_bits_csv = next; return 2; }
    if (a_eq("--residual-bits") && has_next) { cfg.residual_bits = std::atoi(next); return 2; }

    // -- Hierarchical options --
    if (a_eq("--hier-level")     && has_next) { cfg.hier_level = std::atoi(next); return 2; }
    if (a_eq("--hier-res-bits")  && has_next) { cfg.hier_res_bits = std::atoi(next); return 2; }
    if (a_eq("--hier-res-bits-v") && has_next) { cfg.hier_res_bits_v = std::atoi(next); return 2; }
    if (a_eq("--hier-skel-bits") && has_next) { cfg.hier_skel_bits = next; return 2; }
    if (a_eq("--hier-ternary-mask") && has_next) { cfg.hier_skel_ternary = (uint32_t)std::strtoul(next, nullptr, 0); return 2; }

    // -- Main K/V ternary + FP8 --
    if (a_eq("--ternary-k") && has_next) { cfg.k_ternary_mask = (uint32_t)std::stoul(next, nullptr, 0); return 2; }
    if (a_eq("--ternary-v") && has_next) { cfg.v_ternary_mask = (uint32_t)std::stoul(next, nullptr, 0); return 2; }
    if (a_eq("--fp8"))                   { cfg.use_fp8 = true; return 1; }

    // -- PrimePE / RoPE --
    if (a_eq("--pe-mode") && has_next) {
        std::string m = next;
        if      (m == "standard")      cfg.pe_mode = sp::engine::Config::PeMode::Standard;
        else if (m == "primepe")       cfg.pe_mode = sp::engine::Config::PeMode::PrimePe;
        else if (m == "primepe_alibi") cfg.pe_mode = sp::engine::Config::PeMode::PrimePeAlibi;
        else if (m == "alibi")         cfg.pe_mode = sp::engine::Config::PeMode::AlibiOnly;
        return 2;
    }
    if (a_eq("--pe-alpha") && has_next) { cfg.pe_alpha = (float)std::atof(next); return 2; }
    if (a_eq("--pe-tier")  && has_next) { cfg.pe_tier = std::atoi(next); return 2; }

    // -- Cauchy reset --
    if (a_eq("--cauchy-mode")     && has_next) { cfg.cauchy_mode = std::atoi(next); return 2; }
    if (a_eq("--cauchy-fixed-n")  && has_next) { cfg.cauchy_fixed_n = std::atoi(next); return 2; }
    if (a_eq("--cauchy-cooldown") && has_next) { cfg.cauchy_cooldown = std::atoi(next); return 2; }
    if (a_eq("--cauchy-warmup")   && has_next) { cfg.cauchy_warmup = std::atoi(next); return 2; }
    if (a_eq("--cauchy-use-ricci"))    { cfg.cauchy_use_ricci = true; return 1; }
    if (a_eq("--cauchy-ricci-only"))   { cfg.cauchy_ricci_only = true; return 1; }
    if (a_eq("--cauchy-mertens-only")) { cfg.cauchy_mertens_only = true; return 1; }
    if (a_eq("--params-b") && has_next) { cfg.params_b = (float)std::atof(next); return 2; }

    // -- GPU / offload --
    if ((a_eq("--n-gpu-layers") || a_eq("-ngl")) && has_next) { cfg.n_gpu_layers = std::atoi(next); return 2; }
    if (a_eq("--n-gpus") && has_next) { cfg.n_gpus = std::atoi(next); return 2; }

    // -- Cache persistence --
    if (a_eq("--save-cache") && has_next) { cfg.save_cache_path = next; return 2; }
    if (a_eq("--load-cache") && has_next) { cfg.load_cache_path = next; return 2; }
    if (a_eq("--cold-mb")    && has_next) { cfg.cold_mb = std::atoi(next); cfg.enable_cold = true; return 2; }
    if (a_eq("--cold"))                   { cfg.enable_cold = true; return 1; }
    if (a_eq("--evict-keep") && has_next) { cfg.evict_keep = std::atoi(next); return 2; }

    // -- System 1/2 + CRT + MoE --
    if (a_eq("--system12"))                       { cfg.system12 = true; return 1; }
    if (a_eq("--crt-split"))                      { cfg.crt_split = true; return 1; }
    if (a_eq("--moe-curriculum"))                  { cfg.moe_curriculum = true; return 1; }
    if (a_eq("--beast") && has_next)              { cfg.beast_gguf_path = next; return 2; }
    if (a_eq("--s12-threshold") && has_next) { cfg.s12_threshold = (float)std::atof(next); return 2; }
    if (a_eq("--s12-sys2")      && has_next) { cfg.s12_sys2 = next; return 2; }

    return 0;  // not recognized
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string cmd = argv[1];

    if (cmd == "version") {
        std::printf("sp-engine 0.1.0\n");
        return 0;
    }

    // ──────────────────────────────────────────────────────────────────
    // cache_ppl — compressed-cache perplexity benchmark.
    //
    // Runs baseline PPL via forward_full (capturing per-layer K/V),
    // then pushes K/V through the compressed KvCache and measures
    // reconstruction correlation. Reports both baseline PPL and
    // predicted compressed-PPL via the scaling law.
    // ──────────────────────────────────────────────────────────────────
    if (cmd == "cache_ppl") {
        sp::engine::Config cc;
        sp::engine::seed_config_from_env(cc);
        cc.n_ctx = 512;  // cache_ppl default (smaller than engine default)
        int  n_chunks = 0;
        int  ngl      = sp_default_n_gpu_layers();
        bool ngl_explicit = false;
        std::string textfile;
        for (int i = 2; i < argc; ++i) {
            const char* a    = argv[i];
            const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;

            // Verb-specific flags.
            if (std::strcmp(a, "--chunks") == 0 && next) { n_chunks = std::atoi(next); ++i; continue; }

            // Track explicit -ngl (local ngl overrides sp_default).
            if ((std::strcmp(a, "--n-gpu-layers") == 0 || std::strcmp(a, "-ngl") == 0) && next)
                ngl_explicit = true;

            int ate = parse_config_flag(cc, a, next);
            if (ate > 0) { i += ate - 1; continue; }

            // Unknown flag or positional.
            if (a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "cache_ppl: unknown flag %s\n", a); return 2;
            }
            textfile = a;
        }
        if (ngl_explicit) ngl = cc.n_gpu_layers;
        int n_ctx = cc.n_ctx;
        if (cc.model_path.empty()) {
            std::fprintf(stderr, "cache_ppl requires --model <path.gguf>\n"); return 1;
        }
        if (textfile.empty()) {
            std::fprintf(stderr, "cache_ppl requires a UTF-8 text file path\n"); return 1;
        }

        // Multi-GPU: enumerate GPUs, fall through to single-GPU if <= 1.
        SpMultiGpuGuard mgpu;
        mgpu.backends = sp_enumerate_gpus(cc.n_gpus);
        SpBackendGuard bk(mgpu.backends.empty() ? sp_select_backend() : nullptr);

        auto m  = sp::engine::Model::load(cc.model_path);
        if (!m) return 2;
        cc.arch_name = m->architecture();
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;

        std::unique_ptr<sp::engine::LlamaWeights> W;
        if (!mgpu.backends.empty()) {
            W = sp::engine::LlamaWeights::load_multi_gpu(*m, mgpu.backends, ngl);
        } else {
            W = sp::engine::LlamaWeights::load(*m, bk, ngl);
        }
        if (!tk || !W) return 3;

        std::FILE* fp = std::fopen(textfile.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", textfile.c_str()); return 4; }
        std::fseek(fp, 0, SEEK_END);
        const size_t fsize = (size_t)std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        std::string text((size_t)fsize, '\0');
        if (std::fread(text.data(), 1, fsize, fp) != fsize) {
            std::fclose(fp); return 4;
        }
        std::fclose(fp);

        std::vector<int32_t> all_ids;
        tk->encode(text, /*add_bos=*/true, all_ids);
        std::fprintf(stderr, "[sp-engine] tokenised %zu bytes -> %zu tokens\n",
                     fsize, all_ids.size());

        sp::engine::PeSettings pe{cc.pe_mode, cc.pe_alpha, cc.pe_tier};
        std::unique_ptr<sp::engine::ForwardContext> fc;
        if (!mgpu.backends.empty()) {
            fc = sp::engine::ForwardContext::create_multi_gpu(*m, *W, mgpu.backends, 1024*1024*1024, pe);
        } else {
            fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe, bk);
        }
        if (!fc) return 5;

        // CRT multi-GPU tensor splitting (Beast Canyon)
        if (cc.crt_split) {
            const int max_dim = m->n_embd();
            if (fc->enable_crt(max_dim))
                std::fprintf(stderr, "[sp-engine] CRT multi-GPU enabled (max_dim=%d)\n", max_dim);
            else
                std::fprintf(stderr, "[sp-engine] CRT init failed — falling back to standard matmul\n");
        }

        // MoE expert curriculum (Beast Canyon homeostatic balancer)
        if (cc.moe_curriculum) {
            if (fc->enable_moe_curriculum())
                std::fprintf(stderr, "[sp-engine] MoE curriculum active\n");
            else
                std::fprintf(stderr, "[sp-engine] MoE curriculum not available (non-MoE model?)\n");
        }

        // Hybrid-arch (qwen35moe / qwen35): allocate and bind a GdnStateCache
        // so the per-layer delta-rule recurrent state persists across chunks.
        const std::string& cp_arch = m->architecture();
        const bool cp_hybrid = (cp_arch == "qwen35moe" || cp_arch == "qwen35");
        std::unique_ptr<sp::engine::GdnStateCache> gdn_cache_cp;
        if (cp_hybrid) {
            const int conv_kernel   = (int)m->get_i64(cp_arch + ".ssm.conv_kernel",    4);
            const int d_state       = (int)m->get_i64(cp_arch + ".ssm.state_size",     128);
            const int n_group       = (int)m->get_i64(cp_arch + ".ssm.group_count",    16);
            const int num_v_heads   = (int)m->get_i64(cp_arch + ".ssm.time_step_rank", 32);
            const int d_inner       = (int)m->get_i64(cp_arch + ".ssm.inner_size",     4096);
            const int conv_channels = d_inner + 2 * n_group * d_state;
            const int head_v_dim    = (num_v_heads > 0) ? (d_inner / num_v_heads) : 0;
            std::vector<bool> is_gdn; is_gdn.reserve(W->layers().size());
            for (const auto& L : W->layers()) {
                is_gdn.push_back(L.kind == sp::engine::LlamaLayerKind::MOE_GDN);
            }
            gdn_cache_cp = sp::engine::GdnStateCache::create(
                is_gdn, conv_kernel, conv_channels, head_v_dim, num_v_heads, /*n_seqs=*/1);
            if (gdn_cache_cp) {
                gdn_cache_cp->reset();
                fc->bind_gdn_state(gdn_cache_cp.get());
            } else {
                std::fprintf(stderr,
                    "[sp-engine] cache_ppl: GdnStateCache alloc failed — "
                    "GDN layers will use zero state (results may be degraded).\n");
            }
        }

        const int n_layer   = fc->n_layer();
        const int head_dim  = (int)m->head_dim();
        const int n_head_kv = (int)m->n_head_kv();
        const int n_vocab_local = fc->n_vocab();

        const int total_chunks = (int)(all_ids.size() / (size_t)n_ctx);
        const int eval_chunks  = (n_chunks > 0 && n_chunks < total_chunks)
                                   ? n_chunks : total_chunks;
        if (eval_chunks <= 0) {
            std::fprintf(stderr, "text too short for ctx=%d\n", n_ctx); return 6;
        }

        // Create KvCache sized for one chunk at a time. Prefer GPU-resident
        // when the backend is GPU. All cache modes (ship, sqfree, hierarchical)
        // have GPU-resident CUDA kernels. Env gate SHANNON_PRIME_GPU_CACHE=0
        // forces the host cache for A/B comparison, matching chat / perplexity.
        std::unique_ptr<sp::engine::KvCache> kv;
        {
            const char* env_gpu = std::getenv("SHANNON_PRIME_GPU_CACHE");
            const bool prefer_gpu_cache = (env_gpu == nullptr) || (std::atoi(env_gpu) != 0);
            bool backend_is_gpu = !mgpu.backends.empty();  // multi-GPU → always GPU
            if (!backend_is_gpu && (ggml_backend_t)bk) {
                ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t)bk);
                backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            }
            const bool gpu_ok = true;  // all modes have GPU kernels (ship, sqfree, hier)
            if (prefer_gpu_cache && backend_is_gpu && gpu_ok) {
                kv = sp::engine::KvCache::create_gpu(n_layer, n_head_kv, head_dim,
                                                      n_ctx, cc, /*stream=*/nullptr);
                if (!kv) {
                    std::fprintf(stderr, "[sp-engine] create_gpu failed; falling back to host cache\n");
                }
            }
            if (!kv) {
                kv = sp::engine::KvCache::create(n_layer, n_head_kv, head_dim, n_ctx, cc);
            }
        }
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 7; }
        std::fprintf(stderr, "[sp-engine] %s%s\n", kv->describe().c_str(),
                     kv->is_gpu() ? " [GPU-resident]" : "");

        // Model hash for save/load validation.
        const uint64_t model_hash_cp = sp_fnv1a_hash(cc.model_path.c_str(),
                                                      cc.model_path.size());

        // Load cached state if --load-cache given.
        if (!cc.load_cache_path.empty()) {
            const int lp = kv->load_from_disk(cc.load_cache_path, model_hash_cp);
            if (lp < 0) {
                std::fprintf(stderr, "[sp-engine] WARNING: --load-cache failed\n");
            }
        }

        // Init the Cauchy reset system BEFORE the calibration pass so the
        // Ricci sentinel can learn its p=3 baseline from the same vectors
        // used to calibrate the main cache. A no-op when cauchy_mode=0.
        if (cc.cauchy_mode > 0) {
            kv->init_cauchy(cc.cauchy_mode, cc.cauchy_fixed_n, cc.params_b,
                             cc.cauchy_use_ricci);
            kv->cauchy_set_cooldown(cc.cauchy_cooldown);
        }

        auto corr = [](const float* a, const float* b, int len) {
            double ma = 0, mb = 0;
            for (int i = 0; i < len; ++i) { ma += a[i]; mb += b[i]; }
            ma /= len; mb /= len;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < len; ++i) {
                double da = a[i] - ma, db = b[i] - mb;
                sxy += da * db; sxx += da * da; syy += db * db;
            }
            double d = std::sqrt(sxx * syy);
            return d > 0 ? (float)(sxy / d) : 0.0f;
        };

        std::fprintf(stderr,
            "[sp-engine] cache_ppl: n_ctx=%d  chunks=%d  mode=%s\n",
            n_ctx, eval_chunks, kv->is_hierarchical() ? "hierarchical" :
            cc.sqfree ? "sqfree" : "shadow");

        double total_nll = 0.0;
        long long total_evalled = 0;
        double overall_k_corr = 0.0, overall_v_corr = 0.0;
        int    corr_samples = 0;
        bool   calibrated_once = false;

        // Opt-in: hierarchical predictor recalibrates per-chunk. The
        // default single-calibration flow drifts — measured K_corr
        // decay 0.9847 -> 0.9805 across 8 chunks on Qwen3-8B, because
        // the ridge W trained on chunk 1 overfits its local statistics.
        // Env SHANNON_PRIME_HIER_RECAL_PER_CHUNK=1 rebuilds the
        // predictor at every chunk boundary. Extra cost: one
        // calibrate_begin/feed/end per chunk (Cholesky solve over
        // n_skeleton × n_skeleton, fast). Ship / sqfree masks aren't
        // chunk-sensitive so this flag only fires on hierarchical.
        //
        // SHANNON_PRIME_HIER_EMA_KEEP=<0..1> switches the per-chunk
        // recalibration from full replacement to a sticky-EMA blend:
        //   W = keep · W_prev + (1 - keep) · W_fresh
        // Useful on very long contexts where you want the predictor to
        // track chunk statistics slowly rather than snap to each new
        // chunk. keep=0 is identical to full replacement (the default).
        // Typical values: 0.3-0.7. Only affects hierarchical mode.
        const bool hier_recal_per_chunk = []{
            const char* env = std::getenv("SHANNON_PRIME_HIER_RECAL_PER_CHUNK");
            return env && std::atoi(env) != 0;
        }();
        const float hier_ema_keep = []{
            const char* env = std::getenv("SHANNON_PRIME_HIER_EMA_KEEP");
            if (!env) return 0.0f;
            float v = (float)std::atof(env);
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            return v;
        }();

        std::vector<float> logits;
        std::vector<int32_t> chunk((size_t)n_ctx);
        const int32_t bos = v->bos_id();

        for (int c = 0; c < eval_chunks; ++c) {
            for (int t = 0; t < n_ctx; ++t) {
                chunk[(size_t)t] = all_ids[(size_t)(c * n_ctx + t)];
            }
            if (bos >= 0) chunk[0] = bos;

            // Run forward with K/V capture.
            int n_vocab_out = 0;
            std::vector<std::vector<float>> Ks, Vs;
            if (!fc->forward_full(chunk, logits, n_vocab_out, &Ks, &Vs)) {
                std::fprintf(stderr, "forward_full failed at chunk %d\n", c); return 8;
            }

            // Calibrate on first chunk, and (optionally, for hier) on
            // every subsequent chunk to counter predictor drift.
            const bool hier = kv->is_hierarchical();
            const bool do_calibrate =
                (!calibrated_once && !kv->is_calibrated()) ||
                (hier && hier_recal_per_chunk && c > 0);
            if (do_calibrate) {
                if (kv->calibrate_begin()) {
                    for (int L = 0; L < n_layer; ++L) {
                        // Hybrid archs (qwen35/qwen35moe): GDN layers produce
                        // no K/V — Ks[L] is empty. Skip to avoid null deref.
                        if (Ks[(size_t)L].empty()) continue;
                        const float* K_data = Ks[(size_t)L].data();
                        for (int q = 0; q < n_ctx; ++q) {
                            for (int h = 0; h < n_head_kv; ++h) {
                                const float* vec = K_data + (size_t)(q * n_head_kv + h) * head_dim;
                                if (hier) {
                                    kv->calibrate_feed(L * n_head_kv + h, vec);
                                } else {
                                    kv->calibrate_feed(vec);
                                }
                            }
                        }
                    }
                    // On the first calibration there's no prior W to blend
                    // against, so use plain end(). From chunk 2 onward (hier
                    // recal mode), use the EMA variant if keep > 0.
                    const bool have_prior = calibrated_once;
                    if (hier && have_prior && hier_ema_keep > 0.0f) {
                        kv->calibrate_end_ema(hier_ema_keep);
                    } else {
                        kv->calibrate_end();
                    }
                }
                calibrated_once = true;
            }

            // Push K/V through cache and measure round-trip correlation.
            // Hybrid archs: only attention layers have K/V; skip GDN layers.
            for (int L = 0; L < n_layer; ++L) {
                if (Ks[(size_t)L].empty()) continue;
                kv->write(L, 0, n_ctx, Ks[(size_t)L].data(), Vs[(size_t)L].data());
            }
            double chunk_k = 0, chunk_v = 0;
            int chunk_n = 0;
            for (int L = 0; L < n_layer; ++L) {
                if (Ks[(size_t)L].empty()) continue;
                std::vector<float> Krec, Vrec;
                kv->read(L, n_ctx, Krec, Vrec);
                for (int q = 0; q < n_ctx; ++q) {
                    for (int h = 0; h < n_head_kv; ++h) {
                        const size_t off = (size_t)(q * n_head_kv + h) * head_dim;
                        chunk_k += corr(Ks[(size_t)L].data() + off,
                                        Krec.data() + off, head_dim);
                        chunk_v += corr(Vs[(size_t)L].data() + off,
                                        Vrec.data() + off, head_dim);
                        chunk_n++;
                    }
                }
            }
            overall_k_corr += chunk_k;
            overall_v_corr += chunk_v;
            corr_samples   += chunk_n;

            // Baseline PPL scoring (same as perplexity command).
            const int first_eval = n_ctx / 2;
            for (int i = first_eval; i < n_ctx - 1; ++i) {
                const float* row = logits.data() + (size_t)i * n_vocab_out;
                const int32_t target = chunk[(size_t)(i + 1)];
                float mx = row[0];
                for (int k = 1; k < n_vocab_out; ++k) if (row[k] > mx) mx = row[k];
                double s = 0.0;
                for (int k = 0; k < n_vocab_out; ++k) s += std::exp((double)(row[k] - mx));
                const double lse = (double)mx + std::log(s);
                total_nll += lse - (double)row[target];
                total_evalled += 1;
            }

            const double running_ppl = std::exp(total_nll / (double)total_evalled);
            const double running_k   = overall_k_corr / corr_samples;
            const double running_v   = overall_v_corr / corr_samples;
            std::fprintf(stderr,
                "  chunk %3d/%d  PPL=%.4f  K_corr=%.4f  V_corr=%.4f\n",
                c + 1, eval_chunks, running_ppl, running_k, running_v);
        }

        const double ppl     = std::exp(total_nll / (double)total_evalled);
        const double mean_kc = overall_k_corr / corr_samples;
        const double mean_vc = overall_v_corr / corr_samples;

        std::printf("Baseline PPL = %.4f  (over %lld tokens, %d chunks at ctx=%d)\n",
                    ppl, total_evalled, eval_chunks, n_ctx);
        std::printf("Cache K_corr = %.6f  V_corr = %.6f\n", mean_kc, mean_vc);
        if (cc.cauchy_mode > 0) {
            kv->cauchy_print_stats();
            std::printf("Ricci drift  = %.6f\n", kv->ricci_drift());
        }
        std::printf("Compression  = %.2fx\n", kv->compression_ratio());

        // Predicted PPL delta via the scaling law:
        // log(PPL/base) ≈ 4700 · (1-K_corr)² / (params^1.1 · bits^1.5)
        // We don't know params_b here, so just report the numerator.
        const double corr_gap = 1.0 - mean_kc;
        std::printf("Corr gap     = %.6f  (1 - K_corr)\n", corr_gap);
        std::printf("Scaling term = %.4f  (4700 · gap²)\n", 4700.0 * corr_gap * corr_gap);

        // Save cache state if --save-cache given.
        if (!cc.save_cache_path.empty() && kv) {
            if (kv->save_to_disk(cc.save_cache_path, n_ctx, model_hash_cp) != 0) {
                std::fprintf(stderr, "[sp-engine] WARNING: --save-cache failed\n");
            }
        }
        return 0;
    }

    // perplexity handles its own argv (positional textfile + --ctx/--chunks)
    // so the strict --flag pass doesn't reject the textfile path.
    if (cmd == "perplexity") {
        sp::engine::Config pc;
        sp::engine::seed_config_from_env(pc);
        pc.n_ctx = 512;  // perplexity default (smaller than engine default)
        int  n_chunks = 0;     // 0 = all
        int  ngl      = sp_default_n_gpu_layers();
        bool ngl_explicit = false;
        bool use_cache = false;
        std::string textfile;
        for (int i = 2; i < argc; ++i) {
            const char* a    = argv[i];
            const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;

            // Verb-specific flags.
            if (std::strcmp(a, "--chunks") == 0 && next) { n_chunks = std::atoi(next); ++i; continue; }
            if (std::strcmp(a, "--cache") == 0)           { use_cache = true; continue; }

            // Track explicit -ngl.
            if ((std::strcmp(a, "--n-gpu-layers") == 0 || std::strcmp(a, "-ngl") == 0) && next)
                ngl_explicit = true;

            int ate = parse_config_flag(pc, a, next);
            if (ate > 0) { i += ate - 1; continue; }

            // Unknown flag or positional.
            if (a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "perplexity: unknown flag %s\n", a); return 2;
            }
            textfile = a;
        }
        if (ngl_explicit) ngl = pc.n_gpu_layers;
        int n_ctx = pc.n_ctx;
        if (pc.model_path.empty()) {
            std::fprintf(stderr, "perplexity requires --model <path.gguf>\n"); return 1;
        }
        if (textfile.empty()) {
            std::fprintf(stderr, "perplexity requires a UTF-8 text file path\n"); return 1;
        }

        SpMultiGpuGuard mgpu_ppl;
        mgpu_ppl.backends = sp_enumerate_gpus(pc.n_gpus);
        SpBackendGuard bk(mgpu_ppl.backends.empty() ? sp_select_backend() : nullptr);

        auto m  = sp::engine::Model::load(pc.model_path);
        if (!m) return 2;
        pc.arch_name = m->architecture();
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;

        std::unique_ptr<sp::engine::LlamaWeights> W;
        if (!mgpu_ppl.backends.empty()) {
            W = sp::engine::LlamaWeights::load_multi_gpu(*m, mgpu_ppl.backends, ngl);
        } else {
            W = sp::engine::LlamaWeights::load(*m, bk, ngl);
        }
        if (!tk || !W) return 3;

        // Read whole file. Binary mode so byte-perfect with what llama.cpp's
        // perplexity sees from wiki.test.raw.
        std::FILE* fp = std::fopen(textfile.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "cannot open %s\n", textfile.c_str()); return 4;
        }
        std::fseek(fp, 0, SEEK_END);
        const size_t fsize = (size_t)std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        std::string text((size_t)fsize, '\0');
        if (std::fread(text.data(), 1, fsize, fp) != fsize) {
            std::fclose(fp);
            std::fprintf(stderr, "short read on %s\n", textfile.c_str()); return 4;
        }
        std::fclose(fp);

        std::vector<int32_t> all_ids;
        tk->encode(text, /*add_bos=*/true, all_ids);
        std::fprintf(stderr, "[sp-engine] tokenised %zu bytes -> %zu tokens\n",
                     fsize, all_ids.size());

        sp::engine::PeSettings pe_empty;
        std::unique_ptr<sp::engine::ForwardContext> fc;
        if (!mgpu_ppl.backends.empty()) {
            fc = sp::engine::ForwardContext::create_multi_gpu(*m, *W, mgpu_ppl.backends, 1024*1024*1024, pe_empty);
        } else {
            fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe_empty, bk);
        }
        if (!fc) return 5;

        // Hybrid-arch (qwen35moe / qwen35): allocate and bind a GdnStateCache
        // so the per-layer delta-rule recurrent state persists across chunks.
        const std::string& arch = m->architecture();
        const bool is_hybrid_gdn = (arch == "qwen35moe" || arch == "qwen35");
        std::unique_ptr<sp::engine::GdnStateCache> gdn_cache_ppl;
        if (is_hybrid_gdn) {
            const int conv_kernel   = (int)m->get_i64(arch + ".ssm.conv_kernel",    4);
            const int d_state       = (int)m->get_i64(arch + ".ssm.state_size",     128);
            const int n_group       = (int)m->get_i64(arch + ".ssm.group_count",    16);
            const int num_v_heads   = (int)m->get_i64(arch + ".ssm.time_step_rank", 32);
            const int d_inner       = (int)m->get_i64(arch + ".ssm.inner_size",     4096);
            const int conv_channels = d_inner + 2 * n_group * d_state;
            const int head_v_dim    = (num_v_heads > 0) ? (d_inner / num_v_heads) : 0;
            std::vector<bool> is_gdn; is_gdn.reserve(W->layers().size());
            for (const auto& L : W->layers()) {
                is_gdn.push_back(L.kind == sp::engine::LlamaLayerKind::MOE_GDN);
            }
            gdn_cache_ppl = sp::engine::GdnStateCache::create(
                is_gdn, conv_kernel, conv_channels, head_v_dim, num_v_heads, /*n_seqs=*/1);
            if (gdn_cache_ppl) {
                gdn_cache_ppl->reset();
                fc->bind_gdn_state(gdn_cache_ppl.get());
            } else {
                std::fprintf(stderr,
                    "[sp-engine] perplexity: GdnStateCache alloc failed — "
                    "GDN layers will use zero state (results may be degraded).\n");
            }
        }

        const int n_vocab_local = fc->n_vocab();
        const int total_chunks  = (int)(all_ids.size() / (size_t)n_ctx);
        const int eval_chunks   = (n_chunks > 0 && n_chunks < total_chunks)
                                    ? n_chunks : total_chunks;
        if (eval_chunks <= 0) {
            std::fprintf(stderr, "text too short for ctx=%d (have %zu tokens)\n",
                         n_ctx, all_ids.size()); return 6;
        }

        // Cache-mode: allocate one KvCache re-used across chunks
        // (calibration state persists; bind_cache zeroes kv_pos but
        // keeps the cache's calibrated masks intact).
        std::unique_ptr<sp::engine::KvCache> kv;
        if (use_cache) {
            // Auto-select GPU-resident cache when the engine backend is
            // non-CPU AND the config is the ship path (sqfree/hier still
            // run host-side in MVP). Env SHANNON_PRIME_GPU_CACHE=0 forces
            // the host cache even on GPU, for A/B comparisons.
            const char* env_gpu = std::getenv("SHANNON_PRIME_GPU_CACHE");
            bool prefer_gpu_cache = (env_gpu == nullptr) || (std::atoi(env_gpu) != 0);
            bool backend_is_gpu = !mgpu_ppl.backends.empty();
            if (!backend_is_gpu && (ggml_backend_t)bk) {
                ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t)bk);
                backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            }
            // All cache modes (ship, sqfree, hierarchical) have GPU-resident
            // CUDA kernels — the hier GPU cache landed in shannon_prime_hier.cu.
            const bool gpu_ok = true;
            if (prefer_gpu_cache && backend_is_gpu && gpu_ok) {
                kv = sp::engine::KvCache::create_gpu(fc->n_layer(),
                                                      (int)m->n_head_kv(),
                                                      (int)m->head_dim(),
                                                      n_ctx + 8, pc,
                                                      /*stream=*/nullptr);
                if (!kv) {
                    std::fprintf(stderr, "[sp-engine] create_gpu failed; falling back to host cache\n");
                }
            }
            if (!kv) {
                kv = sp::engine::KvCache::create(fc->n_layer(),
                                                  (int)m->n_head_kv(),
                                                  (int)m->head_dim(),
                                                  n_ctx + 8, pc);
            }
            if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 5; }
            std::fprintf(stderr, "[sp-engine] %s%s\n", kv->describe().c_str(),
                         kv->is_gpu() ? " [GPU-resident]" : "");

            // Cauchy reset system. Enables `cauchy_check(pos)` polling in
            // the decode loop below. Must be initialised BEFORE the first
            // prefill so the Ricci sentinel gets calibrated on the same
            // vectors that build the cache's masks.
            if (pc.cauchy_mode > 0) {
                // Ricci sentinel is opt-in now (measured 0 incremental
                // PPL contribution over Mertens-only). Enable when the
                // user explicitly asks for it or for the Ricci-only
                // ablation path.
                const bool use_ricci = pc.cauchy_use_ricci || pc.cauchy_ricci_only;
                kv->init_cauchy(pc.cauchy_mode, pc.cauchy_fixed_n,
                                 pc.params_b, use_ricci);
                kv->cauchy_set_cooldown(pc.cauchy_cooldown);
                if (pc.cauchy_ricci_only)   kv->cauchy_disable_mertens();
                // cauchy_mertens_only is now the default — still accept
                // the flag for script compatibility.
            }
        }

        std::fprintf(stderr,
            "[sp-engine] perplexity: mode=%s n_ctx=%d  total_chunks=%d  eval=%d  n_vocab=%d\n",
            use_cache ? "cache" : "baseline",
            n_ctx, total_chunks, eval_chunks, n_vocab_local);

        double total_nll        = 0.0;
        long long total_evalled = 0;

        std::vector<float> logits;
        std::vector<int32_t> chunk((size_t)n_ctx);
        const int32_t bos = v->bos_id();
        const int first_eval = n_ctx / 2;

        for (int c = 0; c < eval_chunks; ++c) {
            for (int t = 0; t < n_ctx; ++t) {
                chunk[(size_t)t] = all_ids[(size_t)(c * n_ctx + t)];
            }
            if (bos >= 0) chunk[0] = bos;
            int n_vocab_out = 0;

            // Cache-mode path: prefill [0..first_eval], then decode each
            // input token, reading compressed past K/V each step.
            if (use_cache) {
                fc->bind_cache(kv.get());    // resets kv_pos, keeps calibration
                std::vector<int32_t> warmup(chunk.begin(),
                                            chunk.begin() + first_eval + 1);
                std::vector<float> last_logits;
                if (!fc->prefill(warmup, last_logits, n_vocab_out)) {
                    std::fprintf(stderr, "cache-prefill failed at chunk %d\n", c);
                    return 7;
                }
                auto score = [&](const float* row, int32_t target) {
                    float mx = row[0];
                    for (int k = 1; k < n_vocab_out; ++k) if (row[k] > mx) mx = row[k];
                    double s = 0.0;
                    for (int k = 0; k < n_vocab_out; ++k) s += std::exp((double)(row[k] - mx));
                    const double lse = (double)mx + std::log(s);
                    total_nll += lse - (double)row[target];
                    total_evalled += 1;
                };
                score(last_logits.data(), chunk[(size_t)(first_eval + 1)]);
                int chunk_resets = 0;
                for (int i = first_eval + 1; i < n_ctx - 1; ++i) {
                    std::vector<float> dlogits;
                    if (!fc->decode(chunk[(size_t)i], dlogits, n_vocab_out)) {
                        std::fprintf(stderr, "cache-decode failed chunk %d pos %d\n", c, i);
                        return 7;
                    }
                    score(dlogits.data(), chunk[(size_t)(i + 1)]);

                    // Cauchy reset: at this position, does the controller
                    // recommend refreshing the cache? 1=full, 2=partial
                    // (partial is ship-path downgrades to full for now —
                    // hierarchical partial-reset needs the shadow drop).
                    //
                    // Full reset: rebind cache (zeroes kv_pos, keeps
                    // calibrated masks), re-prefill tokens [0..i+1] from
                    // the original token stream. This pays a full
                    // forward_full but gives us a clean ground-truth
                    // cache for all subsequent decode steps.
                    // Initial warmup: suppress resets for the first N decode
                    // positions post-prefill. These positions have low
                    // accumulated compression error — resetting them is pure
                    // overhead. The `first_eval + 1` is the decode start;
                    // gate fires until pos >= start + cauchy_warmup.
                    const int decode_start = first_eval + 1;
                    const bool cauchy_active = (pc.cauchy_mode > 0) &&
                                                (i >= decode_start + pc.cauchy_warmup);
                    if (cauchy_active) {
                        int r = kv->cauchy_check(i);
                        if (r > 0) {
                            // After decode(chunk[i]): kv_pos = i+1, cache
                            // holds positions [0..i]. Refill `i+1` tokens
                            // (chunk[0..i+1)) so prefill leaves kv_pos at
                            // i+1 exactly — matching what the next iteration
                            // expects. Off-by-one here corrupts the cache
                            // layout and inverts the intended benefit.
                            std::vector<int32_t> refill(chunk.begin(),
                                                         chunk.begin() + i + 1);
                            fc->bind_cache(kv.get());
                            std::vector<float> refill_logits;
                            int nv = 0;
                            if (!fc->prefill(refill, refill_logits, nv)) {
                                std::fprintf(stderr,
                                    "cauchy reset: re-prefill failed c=%d pos=%d\n",
                                    c, i);
                                return 7;
                            }
                            kv->cauchy_record_reset(i);
                            chunk_resets++;
                        }
                    }
                }
                const double running_ppl = std::exp(total_nll / (double)total_evalled);
                if (pc.cauchy_mode > 0) {
                    std::fprintf(stderr,
                        "  chunk %3d/%d  PPL_running=%.4f  (cache)  cauchy_resets=%d\n",
                        c + 1, eval_chunks, running_ppl, chunk_resets);
                } else {
                    std::fprintf(stderr, "  chunk %3d/%d  PPL_running=%.4f  (cache)\n",
                                 c + 1, eval_chunks, running_ppl);
                }
                continue;
            }

            // Baseline path (original behaviour): forward_full, score all eval positions.
            if (!fc->forward_full(chunk, logits, n_vocab_out)) {
                std::fprintf(stderr, "forward_full failed at chunk %d\n", c); return 7;
            }
            for (int i = first_eval; i < n_ctx - 1; ++i) {
                const float* row = logits.data() + (size_t)i * n_vocab_out;
                const int32_t target = chunk[(size_t)(i + 1)];
                // log-softmax via logsumexp for numerical stability.
                float mx = row[0];
                for (int k = 1; k < n_vocab_out; ++k) if (row[k] > mx) mx = row[k];
                double s = 0.0;
                for (int k = 0; k < n_vocab_out; ++k) s += std::exp((double)(row[k] - mx));
                const double lse = (double)mx + std::log(s);
                const double nll = lse - (double)row[target];
                total_nll += nll;
                total_evalled += 1;
            }
            const double running_ppl = std::exp(total_nll / (double)total_evalled);
            std::fprintf(stderr, "  chunk %3d/%d  PPL_running=%.4f\n",
                         c + 1, eval_chunks, running_ppl);
        }
        const double mean_nll = total_nll / (double)total_evalled;
        const double ppl      = std::exp(mean_nll);
        std::printf("PPL = %.4f  (over %lld tokens, %d chunks at ctx=%d)\n",
                    ppl, total_evalled, eval_chunks, n_ctx);
        if (use_cache && pc.cauchy_mode > 0 && kv) {
            kv->cauchy_print_stats();
            std::printf("Ricci drift (final) = %.6f\n", kv->ricci_drift());
        }
        return 0;
    }

    // chat handles its own argv so the strict --flag pass doesn't reject
    // --n-predict and so on. Kept above the global parser for the same
    // reason kv_smoke is.
    if (cmd == "chat") {
        sp::engine::Config cc;
        sp::engine::seed_config_from_env(cc);
        int  n_predict = 32;
        bool ngl_explicit = false;
        int  ngl       = sp_default_n_gpu_layers();
        bool naive     = false;
        bool debug_decode = false;
        std::string text;
        for (int i = 2; i < argc; ++i) {
            const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;
            int consumed = parse_config_flag(cc, argv[i], next);
            if (consumed > 0) {
                // Track if -ngl was explicitly set via shared parser
                std::string a = argv[i];
                if (a == "--n-gpu-layers" || a == "-ngl") { ngl = cc.n_gpu_layers; ngl_explicit = true; }
                i += consumed - 1;
                continue;
            }
            std::string a = argv[i];
            if      (a == "--naive")        naive      = true;
            else if (a == "--debug-decode") debug_decode = true;
            else if (a == "--n-predict" && i + 1 < argc) n_predict = std::atoi(argv[++i]);
            else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "chat: unknown flag %s\n", a.c_str()); return 2;
            }
            else {
                if (!text.empty()) text.push_back(' ');
                text += a;
            }
        }
        if (cc.model_path.empty()) {
            std::fprintf(stderr, "chat requires --model <path.gguf>\n"); return 1;
        }
        if (text.empty()) {
            std::fprintf(stderr, "chat requires a prompt\n"); return 1;
        }

        // Multi-GPU: enumerate GPUs, fall through to single-GPU if <= 1.
        SpMultiGpuGuard mgpu_chat;
        mgpu_chat.backends = sp_enumerate_gpus(cc.n_gpus);
        SpBackendGuard bk(mgpu_chat.backends.empty() ? sp_select_backend() : nullptr);

        auto m = sp::engine::Model::load(cc.model_path);
        if (!m) return 2;
        cc.arch_name = m->architecture();
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;

        std::unique_ptr<sp::engine::LlamaWeights> W;
        if (!mgpu_chat.backends.empty()) {
            W = sp::engine::LlamaWeights::load_multi_gpu(*m, mgpu_chat.backends, ngl);
        } else {
            W = sp::engine::LlamaWeights::load(*m, bk, ngl);
        }
        if (!tk || !W) return 3;

        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);
        const int n_prompt = (int)ids.size();
        const int max_seq  = n_prompt + n_predict + 4;

        sp::engine::PeSettings pe{cc.pe_mode, cc.pe_alpha, cc.pe_tier};
        std::unique_ptr<sp::engine::ForwardContext> fc;
        if (!mgpu_chat.backends.empty()) {
            fc = sp::engine::ForwardContext::create_multi_gpu(*m, *W, mgpu_chat.backends, 1024*1024*1024, pe);
        } else {
            fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe, bk);
        }
        if (!fc) return 4;

        // CRT multi-GPU tensor splitting (Beast Canyon)
        if (cc.crt_split) {
            const int max_dim = m->n_embd();
            if (fc->enable_crt(max_dim)) {
                std::fprintf(stderr, "[sp-engine] CRT multi-GPU enabled (max_dim=%d)\n", max_dim);
            } else {
                std::fprintf(stderr, "[sp-engine] CRT init failed — falling back to standard matmul\n");
            }
        }

        // MoE expert curriculum (Beast Canyon homeostatic balancer)
        if (cc.moe_curriculum) {
            if (fc->enable_moe_curriculum())
                std::fprintf(stderr, "[sp-engine] MoE curriculum active\n");
            else
                std::fprintf(stderr, "[sp-engine] MoE curriculum not available (non-MoE model?)\n");
        }

        // Prefer GPU-resident cache when backend is GPU + ship path.
        std::unique_ptr<sp::engine::KvCache> kv;
        std::unique_ptr<sp::engine::DualKvCache> dual;
        {
            const char* env_gpu = std::getenv("SHANNON_PRIME_GPU_CACHE");
            const bool prefer_gpu_cache = (env_gpu == nullptr) || (std::atoi(env_gpu) != 0);
            bool backend_is_gpu = !mgpu_chat.backends.empty();
            if (!backend_is_gpu && (ggml_backend_t)bk) {
                ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t)bk);
                backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            }

            if (cc.system12) {
                // System 1↔2: dual cache mode.
                if (backend_is_gpu) {
                    dual = sp::engine::DualKvCache::create_gpu(
                        fc->n_layer(), (int)m->n_head_kv(),
                        (int)m->head_dim(), max_seq, cc,
                        cc.s12_sys2, cc.s12_threshold,
                        /*stream=*/nullptr);
                }
                if (!dual) {
                    dual = sp::engine::DualKvCache::create(
                        fc->n_layer(), (int)m->n_head_kv(),
                        (int)m->head_dim(), max_seq, cc,
                        cc.s12_sys2, cc.s12_threshold);
                }
                if (!dual) {
                    std::fprintf(stderr, "DualKvCache::create failed\n"); return 5;
                }
                std::fprintf(stderr, "[sp-engine] %s\n", dual->describe().c_str());
            } else {
                // Standard single-cache path.
                // All modes (ship, sqfree, hierarchical) are GPU-resident.
                const bool gpu_ok = true;
                if (prefer_gpu_cache && backend_is_gpu && gpu_ok) {
                    kv = sp::engine::KvCache::create_gpu(fc->n_layer(),
                                                          (int)m->n_head_kv(),
                                                          (int)m->head_dim(),
                                                          max_seq, cc,
                                                          /*stream=*/nullptr);
                    if (!kv) {
                        std::fprintf(stderr, "[sp-engine] create_gpu failed; falling back to host cache\n");
                    }
                }
                if (!kv) {
                    kv = sp::engine::KvCache::create(fc->n_layer(),
                                                      (int)m->n_head_kv(),
                                                      (int)m->head_dim(),
                                                      max_seq, cc);
                }
                if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 5; }
                std::fprintf(stderr, "[sp-engine] %s%s\n", kv->describe().c_str(),
                             kv->is_gpu() ? " [GPU-resident]" : "");
            }
        }
        std::fprintf(stderr, "[sp-engine] PE: %s\n",
                     sp::engine::prime_pe_describe(cc.pe_mode, cc.pe_alpha, cc.pe_tier).c_str());

        // Enable cold storage (tiered GPU→CPU offload) when requested.
        if (!dual && cc.enable_cold && kv && kv->is_gpu()) {
            if (!kv->enable_cold_storage(cc.cold_mb, cc.evict_keep)) {
                std::fprintf(stderr, "[sp-engine] WARNING: cold storage init failed\n");
            }
        }

        // Bind the primary cache for prefill. System 1↔2 uses System 1
        // for prefill; all prefill positions are routed to ship path.
        if (dual) {
            for (int p = 0; p < n_prompt; ++p) {
                dual->route_position(p, 0.0f);  // System 1
            }
            fc->bind_cache(dual->sys1());
        } else {
            fc->bind_cache(kv.get());
        }

        // Model hash for save/load validation.
        const uint64_t model_hash = sp_fnv1a_hash(cc.model_path.c_str(),
                                                   cc.model_path.size());

        // Load cached KV state from disk if --load-cache was given.
        int loaded_pos = 0;
        if (!cc.load_cache_path.empty()) {
            loaded_pos = kv->load_from_disk(cc.load_cache_path, model_hash);
            if (loaded_pos < 0) {
                std::fprintf(stderr, "[sp-engine] WARNING: --load-cache failed "
                             "(continuing with empty cache)\n");
                loaded_pos = 0;
            } else if (loaded_pos > 0) {
                fc->set_kv_pos(loaded_pos);
                std::fprintf(stderr, "[sp-engine] cache loaded: resuming at kv_pos=%d\n",
                             loaded_pos);
            }
        }

        // Cauchy reset system: same semantics as perplexity --cache.
        // On a recommended reset during generation, we re-prefill the
        // full token sequence so far (prompt + generated) to refresh
        // the cache against ground truth. No-op when cauchy_mode=0.
        if (cc.cauchy_mode > 0) {
            kv->init_cauchy(cc.cauchy_mode, cc.cauchy_fixed_n, cc.params_b,
                             cc.cauchy_use_ricci);
            kv->cauchy_set_cooldown(cc.cauchy_cooldown);
        }

        std::vector<float> last_logits;
        int n_vocab = 0;
        std::vector<int32_t> running = ids;  // for --naive path and cauchy refill
        if (naive) {
            std::vector<float> all;
            if (!fc->forward_full(running, all, n_vocab)) {
                std::fprintf(stderr, "naive forward failed\n"); return 6;
            }
            last_logits.assign(all.end() - n_vocab, all.end());
        } else {
            if (!fc->prefill(ids, last_logits, n_vocab)) {
                std::fprintf(stderr, "prefill failed\n"); return 6;
            }
        }

        // Print the prompt as-is so the user sees what we tokenised.
        std::printf("%s", text.c_str());
        std::fflush(stdout);

        std::string out_so_far;
        int sys2_routed = 0;
        for (int step = 0; step < n_predict; ++step) {
            int arg = 0;
            float best = last_logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (last_logits[(size_t)i] > best) { best = last_logits[(size_t)i]; arg = i; }
            }
            std::vector<int32_t> one = { arg };
            std::string piece = tk->decode(one);
            std::printf("%s", piece.c_str());
            std::fflush(stdout);

            if (step + 1 == n_predict) break;

            // System 1↔2: entropy-gate the NEXT token's cache routing.
            // High entropy → System 2 (maximum fidelity); low → System 1.
            if (dual) {
                const float H = sp_logit_entropy(last_logits.data(), n_vocab);
                const int pos = n_prompt + step;  // position of the token about to be decoded
                sys2_routed += dual->route_position(pos, H);
            }

            if (naive) {
                running.push_back(arg);
                std::vector<float> all;
                if (!fc->forward_full(running, all, n_vocab)) {
                    std::fprintf(stderr, "\nnaive forward failed at step %d\n", step); return 7;
                }
                last_logits.assign(all.end() - n_vocab, all.end());
            } else {
                std::vector<float> dbg_K, dbg_X;
                std::vector<float>* dbg_K_ptr = debug_decode ? &dbg_K : nullptr;
                std::vector<float>* dbg_X_ptr = debug_decode ? &dbg_X : nullptr;
                if (!fc->decode(arg, last_logits, n_vocab, dbg_K_ptr, dbg_X_ptr)) {
                    std::fprintf(stderr, "\ndecode failed at step %d\n", step); return 7;
                }
                // Track running so Cauchy reset (and debug_decode below)
                // have the full token sequence. Cost is a single push_back.
                running.push_back(arg);

                // Cauchy reset: same pattern as perplexity --cache. On
                // recommended reset, rebind and re-prefill `running` from
                // ground-truth tokens. Warmup gate suppresses early-chunk
                // resets when the cache is still fresh.
                if (cc.cauchy_mode > 0) {
                    const int decode_start = n_prompt;
                    const int pos          = (int)running.size() - 1;
                    if (pos >= decode_start + cc.cauchy_warmup) {
                        int r = kv->cauchy_check(pos);
                        if (r > 0) {
                            fc->bind_cache(kv.get());
                            std::vector<float> refill_logits;
                            int nv = 0;
                            if (!fc->prefill(running, refill_logits, nv)) {
                                std::fprintf(stderr,
                                    "\ncauchy reset: re-prefill failed step %d pos %d\n",
                                    step, pos);
                                return 7;
                            }
                            last_logits = std::move(refill_logits);
                            n_vocab     = nv;
                            kv->cauchy_record_reset(pos);
                        }
                    }
                }
                if (debug_decode) {
                    std::vector<float> ref_logits, ref_X;
                    int rn = 0;
                    std::vector<std::vector<float>> ref_K, ref_V;
                    if (!fc->forward_full(running, ref_logits, rn, &ref_K, &ref_V, &ref_X)) {
                        std::fprintf(stderr, "\nref forward failed at step %d\n", step); return 8;
                    }
                    const int n_kv     = (int)m->n_head_kv();
                    const int hd       = (int)m->head_dim();
                    const int n_embd   = fc->n_embd();
                    const int last_pos = (int)running.size() - 1;
                    const float* ref_K_last = ref_K[0].data() + (size_t)last_pos * n_kv * hd;
                    const float* ref_X_last = ref_X.data()    + (size_t)last_pos * n_embd;
                    auto corr = [&](const float* a, const float* b, int len) {
                        double ma = 0, mb = 0;
                        for (int i = 0; i < len; ++i) { ma += a[i]; mb += b[i]; }
                        ma /= len; mb /= len;
                        double sxy = 0, sxx = 0, syy = 0;
                        for (int i = 0; i < len; ++i) {
                            double da = a[i] - ma, db = b[i] - mb;
                            sxy += da * db; sxx += da * da; syy += db * db;
                        }
                        double d = std::sqrt(sxx * syy);
                        return d > 0 ? (float)(sxy / d) : 0.0f;
                    };
                    auto magn = [](const float* a, int len) {
                        double s = 0; for (int i = 0; i < len; ++i) s += a[i]*a[i];
                        return std::sqrt(s / len);
                    };
                    float k_corr_total = corr(dbg_K.data(), ref_K_last, n_kv * hd);
                    float x_corr       = corr(dbg_X.data(), ref_X_last, n_embd);
                    float dec_x_mag    = (float)magn(dbg_X.data(), n_embd);
                    float ref_x_mag    = (float)magn(ref_X_last, n_embd);
                    int ref_arg = 0; float ref_best = ref_logits[ref_logits.size() - n_vocab];
                    for (int i = 1; i < n_vocab; ++i) {
                        float v = ref_logits[ref_logits.size() - n_vocab + i];
                        if (v > ref_best) { ref_best = v; ref_arg = i; }
                    }
                    int dec_arg = 0; float dec_best = last_logits[0];
                    for (int i = 1; i < n_vocab; ++i) {
                        if (last_logits[(size_t)i] > dec_best) { dec_best = last_logits[(size_t)i]; dec_arg = i; }
                    }
                    std::fprintf(stderr,
                        "\n[debug step %d past_n=%d] L0 K corr=%.4f  L0 X corr=%.4f  X rms dec=%.3f ref=%.3f  "
                        "dec argmax=%d (%+.3f)  ref argmax=%d (%+.3f)\n",
                        step, fc->kv_pos() - 1, k_corr_total, x_corr,
                        dec_x_mag, ref_x_mag,
                        dec_arg, dec_best, ref_arg, ref_best);
                }
            }
        }
        std::printf("\n");
        std::fprintf(stderr, "[sp-engine] kv_pos=%d  (prompt=%d, generated=%d)\n",
                     fc->kv_pos(), n_prompt, n_predict);
        if (cc.cauchy_mode > 0 && kv) {
            kv->cauchy_print_stats();
        }
        if (dual && sys2_routed > 0) {
            std::fprintf(stderr,
                "[sp-engine] System 1↔2: %d/%d decode tokens → System 2 (%.1f%%)\n",
                sys2_routed, n_predict,
                100.0f * sys2_routed / (float)std::max(1, n_predict));
        }

        // Save cache to disk if --save-cache was given.
        if (!cc.save_cache_path.empty() && kv) {
            const int save_pos = fc->kv_pos();
            if (kv->save_to_disk(cc.save_cache_path, save_pos, model_hash) != 0) {
                std::fprintf(stderr, "[sp-engine] WARNING: --save-cache failed\n");
            }
        }
        return 0;
    }

    // kv_smoke handles its own argv parsing so the strict --flag check below
    // doesn't reject its tweakable knobs (--head-dim, --n-tokens, etc.).
    if (cmd == "kv_smoke") {
        sp::engine::Config kvc;
        int hd = 128, n_tokens = 32, n_head_kv = 4, n_layer = 2;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--sqfree")    { kvc.sqfree = true; kvc.hierarchical = false; }
            else if (a == "--spinor")    { kvc.spinor = true; kvc.sqfree = true; kvc.hierarchical = false; }
            else if (a == "--no-mobius") kvc.mobius = false;
            else if (a == "--hierarchical") kvc.hierarchical = true;
            else if (a == "--head-dim"  && i + 1 < argc) hd        = std::atoi(argv[++i]);
            else if (a == "--n-tokens"  && i + 1 < argc) n_tokens  = std::atoi(argv[++i]);
            else if (a == "--n-head-kv" && i + 1 < argc) n_head_kv = std::atoi(argv[++i]);
            else if (a == "--n-layer"   && i + 1 < argc) n_layer   = std::atoi(argv[++i]);
            else if (a == "--k-bits"    && i + 1 < argc) kvc.k_bits_csv = argv[++i];
            else if (a == "--v-bits"    && i + 1 < argc) kvc.v_bits_csv = argv[++i];
            else if (a == "--residual-bits" && i + 1 < argc) kvc.residual_bits = std::atoi(argv[++i]);
            else if (a == "--hier-level"     && i + 1 < argc) kvc.hier_level     = std::atoi(argv[++i]);
            else if (a == "--hier-res-bits"  && i + 1 < argc) kvc.hier_res_bits  = std::atoi(argv[++i]);
            else if (a == "--hier-res-bits-v" && i + 1 < argc) kvc.hier_res_bits_v = std::atoi(argv[++i]);
            else if (a == "--hier-skel-bits" && i + 1 < argc) kvc.hier_skel_bits = argv[++i];
            else if (a == "--hier-ternary-mask" && i + 1 < argc) kvc.hier_skel_ternary = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
            else if (a == "--ternary-k" && i + 1 < argc) kvc.k_ternary_mask = (uint32_t)std::stoul(argv[++i], nullptr, 0);
            else if (a == "--ternary-v" && i + 1 < argc) kvc.v_ternary_mask = (uint32_t)std::stoul(argv[++i], nullptr, 0);
            else if (a == "--fp8") kvc.use_fp8 = true;
            else { std::fprintf(stderr, "kv_smoke: unknown arg %s\n", a.c_str()); return 2; }
        }

        // When SP_ENGINE_BACKEND=gpu is set, route kv_smoke through the
        // GPU-resident cache so we can directly compare K_corr/V_corr
        // between CPU cache and GPU cache paths on identical input.
        SpBackendGuard kv_bk(sp_select_backend());
        std::unique_ptr<sp::engine::KvCache> kv;
        if ((ggml_backend_t)kv_bk) {
            ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t)kv_bk);
            const bool backend_is_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            // All modes (ship, sqfree, hierarchical) are GPU-resident.
            const bool gpu_ok = true;
            if (backend_is_gpu && gpu_ok) {
                kv = sp::engine::KvCache::create_gpu(n_layer, n_head_kv, hd, n_tokens, kvc,
                                                      /*stream=*/nullptr);
                if (!kv) {
                    std::fprintf(stderr, "[sp-engine] create_gpu failed; falling back to host cache\n");
                }
            }
        }
        if (!kv) {
            kv = sp::engine::KvCache::create(n_layer, n_head_kv, hd, n_tokens, kvc);
        }
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 2; }
        std::fprintf(stderr, "[sp-engine] %s%s\n", kv->describe().c_str(),
                     kv->is_gpu() ? " [GPU-resident]" : "");

        const size_t n_elems = (size_t)n_tokens * n_head_kv * hd;
        std::vector<float> K(n_elems), V(n_elems);
        uint64_t s = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)hd << 32) ^ (uint64_t)n_tokens;
        auto next = [&]() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint32_t u1 = (uint32_t)(s & 0xFFFFFFFFULL);
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint32_t u2 = (uint32_t)(s & 0xFFFFFFFFULL);
            float r1 = (u1 + 1.0f) / 4294967297.0f;
            float r2 = (u2 + 0.0f) / 4294967296.0f;
            return std::sqrt(-2.0f * std::log(r1)) * std::cos(6.2831853f * r2);
        };
        for (size_t i = 0; i < n_elems; ++i) K[i] = next();
        for (size_t i = 0; i < n_elems; ++i) V[i] = next();

        // Calibrate before writing (hierarchical requires per-slot calibration;
        // sqfree/shadow benefit from variance-ranked calibration too).
        if (!kv->is_calibrated()) {
            if (kv->calibrate_begin()) {
                const bool hier = kv->is_hierarchical();
                for (int L = 0; L < n_layer; ++L) {
                    for (int q = 0; q < n_tokens; ++q) {
                        for (int h = 0; h < n_head_kv; ++h) {
                            const float* vec = K.data() + (size_t)(q * n_head_kv + h) * hd;
                            if (hier) {
                                kv->calibrate_feed(L * n_head_kv + h, vec);
                            } else {
                                kv->calibrate_feed(vec);
                            }
                        }
                    }
                }
                kv->calibrate_end();
            }
        }

        for (int L = 0; L < n_layer; ++L) {
            if (!kv->write(L, 0, n_tokens, K.data(), V.data())) {
                std::fprintf(stderr, "kv->write layer %d failed\n", L); return 3;
            }
        }
        std::vector<float> Krec, Vrec;
        if (!kv->read(0, n_tokens, Krec, Vrec)) {
            std::fprintf(stderr, "kv->read failed\n"); return 4;
        }

        auto corr = [&](const float* a, const float* b, int n) {
            double ma = 0, mb = 0;
            for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
            ma /= n; mb /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; ++i) {
                double da = a[i] - ma, db = b[i] - mb;
                sxy += da * db; sxx += da * da; syy += db * db;
            }
            const double denom = std::sqrt(sxx * syy);
            return (denom > 0) ? (float)(sxy / denom) : 0.0f;
        };

        double k_sum = 0, v_sum = 0;
        float  k_min = 1.0f, v_min = 1.0f;
        const int per = n_head_kv * n_tokens;
        for (int q = 0; q < n_tokens; ++q) {
            for (int h = 0; h < n_head_kv; ++h) {
                const float* k0 = K.data()    + (size_t)(q * n_head_kv + h) * hd;
                const float* k1 = Krec.data() + (size_t)(q * n_head_kv + h) * hd;
                const float* v0 = V.data()    + (size_t)(q * n_head_kv + h) * hd;
                const float* v1 = Vrec.data() + (size_t)(q * n_head_kv + h) * hd;
                float kc = corr(k0, k1, hd);
                float vc = corr(v0, v1, hd);
                k_sum += kc; v_sum += vc;
                if (kc < k_min) k_min = kc;
                if (vc < v_min) v_min = vc;
            }
        }
        std::printf("K corr: mean=%.4f  min=%.4f  (over %d vectors, hd=%d)\n",
                    k_sum / per, k_min, per, hd);
        std::printf("V corr: mean=%.4f  min=%.4f\n", v_sum / per, v_min);
        std::printf("compression ratio = %.2fx\n", kv->compression_ratio());
        return 0;
    }

    // ── CRT multi-GPU smoke test (CPU reference path) ─────────────────
    //
    //   sp-engine crt_smoke [--dim N]
    //
    // Validates the full CRT pipeline: quantize → split → modular matmul
    // → Garner reconstruct → dequantize. Runs on CPU (no GPU required).
    // Tests:
    //   1. Single-element round-trip (sp_crt_verify_roundtrip)
    //   2. Small matmul correctness (M×N×K) against naive fp32
    //   3. Error statistics (max error, mean error, PASS/FAIL)

    if (cmd == "crt_smoke") {
        int dim = 64;  // default matmul dimension (dim × dim × dim)
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
        }

        std::printf("=== CRT Smoke Test ===\n\n");

        // ── Test 1: Round-trip verification (scalar) ────────────────
        std::printf("--- Test 1: scalar round-trip ---\n");
        int rt_pass = 0, rt_fail = 0;
        float rt_max_err = 0.0f;
        // Sweep representative values including edge cases.
        float test_vals[] = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.99f, -3.99f,
            0.001f, -0.001f, 2.718f, -1.414f, 3.141f, -2.236f
        };
        const int n_vals = sizeof(test_vals) / sizeof(test_vals[0]);
        for (int i = 0; i < n_vals; ++i) {
            for (int j = 0; j < n_vals; ++j) {
                float err = 0.0f;
                int rc = sp_crt_verify_roundtrip(test_vals[i], test_vals[j], &err);
                if (rc == 0) rt_pass++; else rt_fail++;
                if (err > rt_max_err) {
                    rt_max_err = err;
                    std::printf("  worst so far: a=%.4f b=%.4f expected=%.6f err=%.6f\n",
                                test_vals[i], test_vals[j],
                                test_vals[i] * test_vals[j], err);
                }
            }
        }
        std::printf("  %d / %d round-trips passed, max error = %.6f\n",
                    rt_pass, rt_pass + rt_fail, rt_max_err);
        std::printf("  %s\n\n", rt_fail == 0 ? "PASS" : "FAIL");

        // ── Test 2: Matmul correctness (M × K) × (K × N) ───────────
        std::printf("--- Test 2: %d×%d matmul ---\n", dim, dim);

        const int M = dim, N = dim, K = dim;
        const size_t a_sz = (size_t)M * K;
        const size_t b_sz = (size_t)K * N;
        const size_t c_sz = (size_t)M * N;

        std::vector<float> A(a_sz), B(b_sz), C_crt(c_sz), C_ref(c_sz);

        // Deterministic PRNG fill in [-1, 1]
        uint64_t s = 0xDEADBEEF12345678ULL;
        auto rng = [&]() -> float {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            return ((float)(s & 0xFFFFFFFF) / 2147483648.0f) - 1.0f;
        };
        for (size_t i = 0; i < a_sz; ++i) A[i] = rng();
        for (size_t i = 0; i < b_sz; ++i) B[i] = rng();

        // Naive fp32 reference
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (double)A[i * K + k] * (double)B[k * N + j];
                }
                C_ref[i * N + j] = (float)acc;
            }
        }

        // CRT matmul (CPU reference path)
        sp_crt_context_t ctx;
        int rc = sp_crt_init(&ctx, M, N, K, nullptr, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "sp_crt_init failed: %d\n", rc);
            return 2;
        }

        // Calibrate from actual data range
        float a_min = A[0], a_max = A[0];
        float b_min = B[0], b_max = B[0];
        for (size_t i = 1; i < a_sz; ++i) {
            if (A[i] < a_min) a_min = A[i];
            if (A[i] > a_max) a_max = A[i];
        }
        for (size_t i = 1; i < b_sz; ++i) {
            if (B[i] < b_min) b_min = B[i];
            if (B[i] > b_max) b_max = B[i];
        }
        sp_crt_quant_calibrate_k(&ctx.act_quant, a_min, a_max, K);
        sp_crt_quant_calibrate_k(&ctx.weight_quant, b_min, b_max, K);

        rc = sp_crt_matmul(&ctx, A.data(), B.data(), C_crt.data(), M, N, K);
        sp_crt_free(&ctx);
        if (rc != 0) {
            std::fprintf(stderr, "sp_crt_matmul failed: %d\n", rc);
            return 2;
        }

        // Compare CRT vs reference
        double max_abs_err = 0.0, sum_abs_err = 0.0;
        double max_rel_err = 0.0, sum_rel_err = 0.0;
        for (size_t i = 0; i < c_sz; ++i) {
            double err = std::fabs((double)C_crt[i] - (double)C_ref[i]);
            double rel = (std::fabs(C_ref[i]) > 1e-6)
                       ? err / std::fabs((double)C_ref[i])
                       : err;
            if (err > max_abs_err) max_abs_err = err;
            if (rel > max_rel_err) max_rel_err = rel;
            sum_abs_err += err;
            sum_rel_err += rel;
        }

        std::printf("  ref range: [%.4f, %.4f]\n",
                    *std::min_element(C_ref.begin(), C_ref.end()),
                    *std::max_element(C_ref.begin(), C_ref.end()));
        std::printf("  abs error: max=%.6f  mean=%.6f\n",
                    max_abs_err, sum_abs_err / c_sz);
        std::printf("  rel error: max=%.6f  mean=%.6f\n",
                    max_rel_err, sum_rel_err / c_sz);

        // Pass criterion: mean relative error < 1%
        bool matmul_pass = (sum_rel_err / c_sz) < 0.01;
        std::printf("  %s\n\n", matmul_pass ? "PASS" : "FAIL");

        // ── Test 3: Identity matrix — Residue Integrity Test ────────
        //
        // Feed A × I through CRT. If Garner merges correctly on the host,
        // the output must be A (within quantization tolerance). A "ghost"
        // value here means the residue rings are aliasing.
        std::printf("--- Test 3: identity matrix (residue integrity) ---\n");

        const int idim = std::min(dim, 128);  // cap for speed
        const size_t id_sz = (size_t)idim * idim;

        std::vector<float> Id(id_sz, 0.0f);
        for (int i = 0; i < idim; ++i) Id[i * idim + i] = 1.0f;

        // Random A matrix in [-1, 1]
        std::vector<float> A_id(id_sz);
        s = 0xCAFEBABE42424242ULL;  // fresh seed
        for (size_t i = 0; i < id_sz; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            A_id[i] = ((float)(s & 0xFFFFFFFF) / 2147483648.0f) - 1.0f;
        }

        std::vector<float> C_id(id_sz);
        sp_crt_context_t id_ctx;
        rc = sp_crt_init(&id_ctx, idim, idim, idim, nullptr, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "sp_crt_init failed for identity test: %d\n", rc);
            return 2;
        }
        // Identity matrix is in [0,1], A is in [-1,1]
        sp_crt_quant_calibrate_k(&id_ctx.act_quant, -1.0f, 1.0f, idim);
        sp_crt_quant_calibrate_k(&id_ctx.weight_quant, 0.0f, 1.0f, idim);

        rc = sp_crt_matmul(&id_ctx, A_id.data(), Id.data(), C_id.data(),
                           idim, idim, idim);
        sp_crt_free(&id_ctx);
        if (rc != 0) {
            std::fprintf(stderr, "sp_crt_matmul identity failed: %d\n", rc);
            return 2;
        }

        double id_max_err = 0.0, id_ghost_count = 0;
        for (size_t i = 0; i < id_sz; ++i) {
            double err = std::fabs((double)C_id[i] - (double)A_id[i]);
            if (err > id_max_err) id_max_err = err;
            if (err > 0.1) id_ghost_count++;
        }
        std::printf("  dim: %d×%d, max error vs input: %.8f\n", idim, idim, id_max_err);
        std::printf("  ghost values (err > 0.1): %d / %zu\n",
                    (int)id_ghost_count, id_sz);
        bool id_pass = (id_ghost_count == 0) && (id_max_err < 0.01);
        std::printf("  Garner merger returns: %s\n",
                    id_pass ? "1.0000 (clean)" : "GHOST detected");
        std::printf("  %s\n\n", id_pass ? "PASS" : "FAIL");

        // ── Summary ─────────────────────────────────────────────────
        bool all_pass = (rt_fail == 0) && matmul_pass && id_pass;
        std::printf("=== CRT Smoke: %s ===\n", all_pass ? "ALL PASS" : "FAIL");
        return all_pass ? 0 : 1;
    }

    // ── CRT model-level test — real activations through CRT pipeline ───
    //
    //   sp-engine crt_model --model <path.gguf> [--ctx N] [--layer L]
    //
    // Loads a GGUF model, runs one forward chunk to capture per-layer K
    // tensors (real activations from real weights), then replays head 0's
    // K × K^T (the attention score inner product) through both CRT and
    // fp32 reference. Reports error statistics.
    //
    // This proves CRT works with real model weight distributions —
    // not just synthetic random data.

    if (cmd == "crt_model") {
        std::string model_path;
        int n_ctx   = 128;
        int layer   = 0;
        int ngl     = sp_default_n_gpu_layers();
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model" && i + 1 < argc) model_path = argv[++i];
            else if (a == "--ctx"   && i + 1 < argc) n_ctx  = std::atoi(argv[++i]);
            else if (a == "--layer" && i + 1 < argc) layer  = std::atoi(argv[++i]);
            else if ((a == "--n-gpu-layers" || a == "-ngl") && i + 1 < argc) ngl = std::atoi(argv[++i]);
        }
        if (model_path.empty()) {
            std::fprintf(stderr, "crt_model requires --model <path.gguf>\n");
            return 1;
        }

        std::printf("=== CRT Model-Level Test ===\n\n");

        // Load model
        auto m = sp::engine::Model::load(model_path);
        if (!m) { std::fprintf(stderr, "failed to load model\n"); return 2; }

        SpBackendGuard bk(sp_select_backend());
        auto W = sp::engine::LlamaWeights::load(*m, bk, ngl);
        if (!W) { std::fprintf(stderr, "failed to load weights\n"); return 3; }

        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        if (!tk) { std::fprintf(stderr, "failed to create tokenizer\n"); return 3; }

        const int n_layer   = (int)m->n_layer();
        const int head_dim  = (int)m->head_dim();
        const int n_head_kv = (int)m->n_head_kv();

        if (layer >= n_layer) {
            std::fprintf(stderr, "layer %d >= n_layer %d\n", layer, n_layer);
            return 2;
        }

        std::printf("  model:     %s\n", model_path.c_str());
        std::printf("  arch:      %s\n", m->architecture().c_str());
        std::printf("  n_layer:   %d\n", n_layer);
        std::printf("  head_dim:  %d\n", head_dim);
        std::printf("  n_head_kv: %d\n", n_head_kv);
        std::printf("  test ctx:  %d tokens\n", n_ctx);
        std::printf("  test layer: %d\n\n", layer);

        // Create forward context
        sp::engine::PeSettings pe{};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe, bk);
        if (!fc) { std::fprintf(stderr, "failed to create forward context\n"); return 4; }

        // Generate token IDs — use a simple repeated sequence.
        // We don't need meaningful text, just real model activations.
        std::vector<int32_t> toks(n_ctx);
        toks[0] = 1; // BOS
        for (int i = 1; i < n_ctx; ++i) toks[i] = 100 + (i % 500);

        // Forward pass — capture per-layer K and V
        std::printf("--- Forward pass (capturing K/V) ---\n");
        std::vector<float> logits;
        int n_vocab_local = 0;
        std::vector<std::vector<float>> per_K, per_V;
        if (!fc->forward_full(toks, logits, n_vocab_local, &per_K, &per_V)) {
            std::fprintf(stderr, "forward_full failed\n"); return 5;
        }
        std::printf("  logits shape: [%d, %d]\n", n_ctx, n_vocab_local);

        if (per_K[layer].empty()) {
            std::fprintf(stderr, "layer %d K not captured\n", layer);
            return 5;
        }

        // K shape: [head_dim, n_head_kv, n_tokens] stored flat.
        // Extract head 0: stride = head_dim, step = head_dim * n_head_kv.
        // Reshape as K_head0: [n_tokens × head_dim] row-major.
        const float* K_raw = per_K[layer].data();
        const int n = n_ctx;
        const int d = head_dim;
        std::vector<float> K_mat(n * d);
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < d; ++h) {
                // K is [head_dim, n_head_kv, n_tokens] → element [h, 0, t]
                // = K_raw[h + 0*head_dim + t*head_dim*n_head_kv]
                //   but ggml stores in [head_dim, n_head_kv, n] order:
                //   index = h + head*head_dim + t*(head_dim*n_head_kv)
                K_mat[t * d + h] = K_raw[h + t * d * n_head_kv];
            }
        }

        // Report K statistics
        float k_min = K_mat[0], k_max = K_mat[0];
        double k_abssum = 0;
        for (size_t i = 0; i < K_mat.size(); ++i) {
            if (K_mat[i] < k_min) k_min = K_mat[i];
            if (K_mat[i] > k_max) k_max = K_mat[i];
            k_abssum += std::fabs(K_mat[i]);
        }
        std::printf("  K head-0 range: [%.4f, %.4f], mean_abs: %.4f\n\n",
                    k_min, k_max, k_abssum / K_mat.size());

        // Compute K × K^T — fp32 reference
        // K_mat: [n × d], result: [n × n]
        std::printf("--- K×K^T: %d×%d × %d×%d → %d×%d ---\n", n, d, d, n, n, n);

        // K^T: [d × n]
        std::vector<float> Kt(d * n);
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < d; ++h)
                Kt[h * n + t] = K_mat[t * d + h];

        const size_t out_sz = (size_t)n * n;
        std::vector<float> C_ref(out_sz), C_crt(out_sz);

        // fp32 reference: K × K^T
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int k = 0; k < d; ++k)
                    acc += (double)K_mat[i * d + k] * (double)Kt[k * n + j];
                C_ref[i * n + j] = (float)acc;
            }
        }

        // CRT path: K × K^T — run both CPU and GPU dispatch
        sp_crt_context_t ctx;
        int rc = sp_crt_init(&ctx, n, n, d, nullptr, nullptr);
        if (rc != 0) { std::fprintf(stderr, "sp_crt_init failed: %d\n", rc); return 6; }

        // Calibrate from actual K value range
        sp_crt_quant_calibrate_k(&ctx.act_quant, k_min, k_max, d);
        sp_crt_quant_calibrate_k(&ctx.weight_quant, k_min, k_max, d);

        // CPU reference path
        rc = sp_crt_matmul(&ctx, K_mat.data(), Kt.data(), C_crt.data(), n, n, d);
        if (rc != 0) { sp_crt_free(&ctx); std::fprintf(stderr, "sp_crt_matmul CPU failed: %d\n", rc); return 6; }

        // GPU dispatch path
        std::vector<float> C_gpu(out_sz, 0.0f);
        int gpu_rc = sp_crt_init_gpu(&ctx);
        if (gpu_rc == 0) {
            std::printf("\n--- GPU dispatch test ---\n");
            // Warm up
            sp_crt_matmul_gpu(&ctx, K_mat.data(), Kt.data(), C_gpu.data(), n, n, d);

            // Timed run
            auto t0 = std::chrono::high_resolution_clock::now();
            const int n_iters = 100;
            for (int iter = 0; iter < n_iters; ++iter) {
                sp_crt_matmul_gpu(&ctx, K_mat.data(), Kt.data(), C_gpu.data(), n, n, d);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iters;

            // Verify GPU matches CPU CRT
            double gpu_max_err = 0;
            for (size_t i = 0; i < out_sz; ++i) {
                double e = std::fabs((double)C_gpu[i] - (double)C_crt[i]);
                if (e > gpu_max_err) gpu_max_err = e;
            }
            std::printf("  GPU vs CPU CRT max delta: %.8f %s\n",
                        gpu_max_err, (gpu_max_err < 0.001) ? "(match)" : "(MISMATCH)");
            std::printf("  GPU dispatch: %.3f ms per %d×%d×%d matmul (%d iters)\n",
                        gpu_ms, n, n, d, n_iters);
        } else {
            std::printf("  (GPU dispatch not available: rc=%d — using CPU path)\n", gpu_rc);
        }
        sp_crt_free(&ctx);

        // Compare
        double max_abs = 0, sum_abs = 0;
        double max_rel = 0, sum_rel = 0;
        float ref_min = C_ref[0], ref_max = C_ref[0];
        for (size_t i = 0; i < out_sz; ++i) {
            if (C_ref[i] < ref_min) ref_min = C_ref[i];
            if (C_ref[i] > ref_max) ref_max = C_ref[i];
            double err = std::fabs((double)C_crt[i] - (double)C_ref[i]);
            double rel = (std::fabs(C_ref[i]) > 1e-6)
                       ? err / std::fabs((double)C_ref[i]) : err;
            if (err > max_abs) max_abs = err;
            if (rel > max_rel) max_rel = rel;
            sum_abs += err;
            sum_rel += rel;
        }

        std::printf("  ref range: [%.4f, %.4f]\n", ref_min, ref_max);
        std::printf("  abs error: max=%.6f  mean=%.6f\n", max_abs, sum_abs / out_sz);
        std::printf("  rel error: max=%.6f  mean=%.6f\n", max_rel, sum_rel / out_sz);

        bool pass = (sum_rel / out_sz) < 0.01;
        std::printf("  %s\n\n", pass ? "PASS" : "FAIL");

        // Diagonal check — self-attention scores K[i]·K[i] should be positive
        int diag_ok = 0;
        for (int i = 0; i < n; ++i) {
            if (C_crt[i * n + i] > 0.0f) diag_ok++;
        }
        std::printf("  diagonal (self-dot) positive: %d / %d\n", diag_ok, n);

        std::printf("\n=== CRT Model: %s ===\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    // ── Beast Canyon standalone test ─────────────────────────────────
    //   sp-engine beast_test <path-to-gguf> [--audit-only]
    //
    // Boots the Beast Canyon orchestrator on the specified GGUF model:
    // maps Optane reservoir, runs Day Zero stride audit, shredder bench,
    // expert table dump, and full engine boot dry-run.
#ifdef SP_ENGINE_WITH_BEAST
    if (cmd == "beast_test") {
        if (argc < 3) {
            std::fprintf(stderr, "Usage: sp-engine beast_test <gguf-path> [--audit-only]\n");
            return 1;
        }
        const char* gguf_path = argv[2];
        bool audit_only = (argc >= 4 && std::string(argv[3]) == "--audit-only");

        // Boot the reservoir (mmap + GGUF parse)
        sp_optane_reservoir_t reservoir;
        int rc = sp_optane_init(&reservoir, gguf_path);
        if (rc != 0) {
            std::fprintf(stderr, "[beast_test] Reservoir mapping failed (rc=%d)\n", rc);
            return 1;
        }
        sp_optane_print_status(&reservoir);

        // Optane stride audit
        std::fprintf(stderr, "\n=== OPTANE STRIDE AUDIT ===\n");
        double lat_us = sp_optane_measure_stride_latency(&reservoir);
        std::fprintf(stderr, "4KB stride latency: %.2f us\n", lat_us);
        if (lat_us > 0 && lat_us < 15.0)
            std::fprintf(stderr, "VERDICT: PASS — Optane-class latency\n");
        else if (lat_us > 0)
            std::fprintf(stderr, "VERDICT: ACCEPTABLE — NVMe SSD (%.1f us > 15 us Optane target)\n", lat_us);
        else
            std::fprintf(stderr, "VERDICT: FAIL — measurement error\n");

        if (!audit_only) {
            // Full engine boot
            std::fprintf(stderr, "\n=== BEAST CANYON ENGINE BOOT ===\n");
            sp_beast_config_t bcfg;
            sp_beast_config_init(&bcfg);
            bcfg.gguf_path = gguf_path;
            // Auto-detect GPUs (CUDA + Vulkan) unless --cpu-only passed
            bcfg.force_cpu_only = false;

            sp_beast_engine_t engine;
            rc = sp_beast_init(&engine, &bcfg);
            if (rc == 0) {
                sp_beast_print_status(&engine);
                sp_beast_free(&engine);
                std::fprintf(stderr, "\n=== BEAST CANYON TEST: PASS ===\n");
            } else {
                std::fprintf(stderr, "\n=== BEAST CANYON TEST: FAIL (rc=%d) ===\n", rc);
            }
        }

        sp_optane_free(&reservoir);
        return 0;
    }
#endif

    if (cmd == "banner") {
        std::printf("Shannon-Prime Engine — reference inference with compressed KV cache\n");
        std::printf("  linked: shannon-prime core (AGPLv3)\n");
        std::printf("  linked: ggml (MIT)\n");
        std::printf("  status: pre-alpha, full forward+decode with ship/sqfree/hierarchical cache\n");
        return 0;
    }

#if defined(SP_ENGINE_WITH_QNN)
    // Phase 5.0: prefill bench against AI Hub-compiled V69 .bins,
    // mirroring lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/
    // test_sp_qnn_prefill_batch.c — but called from sp-engine so the
    // same load+exec lifecycle runs from inside the engine binary.
    //   sp-engine qnn_bin_bench [--n-chunks N] split1 split2 split3 split4
    if (cmd == "qnn_bin_bench") {
        int n_chunks = 3;
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--n-chunks" && i + 1 < argc) n_chunks = std::atoi(argv[++i]);
            else splits.emplace_back(std::move(a));
        }
        if (splits.size() != 4) {
            std::fprintf(stderr,
                "qnn_bin_bench: expects exactly 4 split paths, got %zu\n",
                splits.size());
            return 1;
        }
        return sp::engine::qnn_bin_prefill_bench(splits, n_chunks);
    }

    // Phase 5.1: schema dump for the .bins. Prints input/output
    // tensor name+dtype+rank+dims for each split — needed to identify
    // which input is tokens, which is position_ids, which is residual,
    // which is KV cache, before wiring real prompts in Phase 5.2.
    //   sp-engine qnn_bin_schema split1 [split2 ...]
    if (cmd == "qnn_bin_schema") {
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) splits.emplace_back(argv[i]);
        if (splits.empty()) {
            std::fprintf(stderr,
                "qnn_bin_schema: expects at least 1 split path\n");
            return 1;
        }
        return sp::engine::qnn_bin_schema_dump(splits);
    }

    // Phase 5.2: real prompt → next token through the .bin chain.
    //   sp-engine qnn_bin_run --tokenizer <gguf> --prompt "..." \
    //                         [--ar 128] [--cl 2048] [--head-dim 128] \
    //                         [--rope-base 1000000] \
    //                         split1 split2 split3 split4
    if (cmd == "qnn_bin_run") {
        std::string tok_path, prompt;
        int   ar = 128, cl = 2048, hd = 128;
        float rope_base = 1000000.0f;
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--tokenizer" && i + 1 < argc) tok_path = argv[++i];
            else if (a == "--prompt"    && i + 1 < argc) prompt   = argv[++i];
            else if (a == "--ar"        && i + 1 < argc) ar = std::atoi(argv[++i]);
            else if (a == "--cl"        && i + 1 < argc) cl = std::atoi(argv[++i]);
            else if (a == "--head-dim"  && i + 1 < argc) hd = std::atoi(argv[++i]);
            else if (a == "--rope-base" && i + 1 < argc) rope_base = (float)std::atof(argv[++i]);
            else splits.emplace_back(std::move(a));
        }
        if (tok_path.empty() || prompt.empty() || splits.size() != 4) {
            std::fprintf(stderr,
                "qnn_bin_run: needs --tokenizer <gguf> --prompt <text> "
                "and exactly 4 split paths\n");
            return 1;
        }

        // Load model just for vocab/tokenizer.
        auto m = sp::engine::Model::load(tok_path);
        if (!m) { std::fprintf(stderr, "Model::load failed\n"); return 2; }
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        if (!tk) { std::fprintf(stderr, "tokenizer init failed\n"); return 3; }

        std::vector<int32_t> ids;
        tk->encode(prompt, /*add_bos=*/true, ids);
        std::fprintf(stderr,
            "[qnn_bin_run] prompt encoded to %zu tokens (ar=%d, cl=%d)\n",
            ids.size(), ar, cl);

        int next_id = -1;
        const int rc = sp::engine::qnn_bin_generate_one(
            splits, ids, ar, cl, hd, rope_base, &next_id);
        if (rc != 0) {
            std::fprintf(stderr, "qnn_bin_generate_one failed rc=%d\n", rc);
            return 4;
        }

        // Decode the predicted token back to text.
        const std::string next_text = tk->decode({next_id});
        std::fprintf(stderr,
            "[qnn_bin_run] next token id=%d text='%s'\n",
            next_id, next_text.c_str());
        std::printf("PROMPT: %s\nNEXT  : %s (id=%d)\n",
                    prompt.c_str(), next_text.c_str(), next_id);
        return 0;
    }

    // Phase 8: speculative oracle benchmark.
    //   sp-engine qnn_oracle_bench
    //     --tokenizer  <main_gguf>
    //     --oracle     <draft_gguf>
    //     --prompt     <text>
    //     [--n-predict N]   (default 64)
    //     [--ar 128] [--cl 2048] [--head-dim 2560] [--rope-base 1000000]
    //     split1 split2 split3 split4
    //
    // Runs QnnBinSession::generate() with the draft oracle attached.
    // Prints oracle accuracy and token-per-second stats.
    if (cmd == "qnn_oracle_bench") {
        std::string tok_path, oracle_path, prompt;
        int   ar = 128, cl = 2048, hd = 2560, n_predict = 64;
        float rope_base = 1000000.0f;
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--tokenizer"  && i + 1 < argc) tok_path     = argv[++i];
            else if (a == "--oracle"     && i + 1 < argc) oracle_path  = argv[++i];
            else if (a == "--prompt"     && i + 1 < argc) prompt       = argv[++i];
            else if (a == "--n-predict"  && i + 1 < argc) n_predict    = std::atoi(argv[++i]);
            else if (a == "--ar"         && i + 1 < argc) ar           = std::atoi(argv[++i]);
            else if (a == "--cl"         && i + 1 < argc) cl           = std::atoi(argv[++i]);
            else if (a == "--head-dim"   && i + 1 < argc) hd           = std::atoi(argv[++i]);
            else if (a == "--rope-base"  && i + 1 < argc) rope_base    = (float)std::atof(argv[++i]);
            else splits.emplace_back(std::move(a));
        }
        if (tok_path.empty() || prompt.empty() || splits.size() != 4) {
            std::fprintf(stderr,
                "qnn_oracle_bench: needs --tokenizer <gguf> --prompt <text> "
                "and exactly 4 split paths\n"
                "  optional: --oracle <draft_gguf>  --n-predict N\n");
            return 1;
        }

        // Tokenize the prompt.
        auto m  = sp::engine::Model::load(tok_path);
        if (!m) { std::fprintf(stderr, "Model::load failed: %s\n", tok_path.c_str()); return 2; }
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        if (!tk) { std::fprintf(stderr, "tokenizer init failed\n"); return 3; }
        std::vector<int32_t> ids;
        tk->encode(prompt, /*add_bos=*/true, ids);
        std::fprintf(stderr, "[qnn_oracle_bench] prompt: %zu tokens, n_predict=%d\n",
                     ids.size(), n_predict);

        // Load HTP session.
        sp::engine::QnnBinSession session;
        if (session.load(splits, ar, cl, hd, rope_base) != 0) {
            std::fprintf(stderr, "[qnn_oracle_bench] QnnBinSession::load failed\n");
            return 4;
        }

        // Optionally load oracle.
        sp::engine::SpOracle oracle;
        if (!oracle_path.empty()) {
            int orc = oracle.load(oracle_path.c_str());
            if (orc != 0) {
                std::fprintf(stderr, "[qnn_oracle_bench] oracle load failed (rc=%d) "
                             "— continuing without oracle\n", orc);
            } else {
                session.set_oracle(&oracle);
                std::fprintf(stderr, "[qnn_oracle_bench] oracle attached: %s\n",
                             oracle_path.c_str());
            }
        } else {
            std::fprintf(stderr, "[qnn_oracle_bench] no --oracle given — "
                         "running baseline HTP-only decode\n");
        }

        // Run generation, timed.
        std::vector<int32_t> out_ids;
        auto t0 = std::chrono::steady_clock::now();
        int grc = session.generate(ids, n_predict, out_ids);
        auto t1 = std::chrono::steady_clock::now();

        if (grc != 0) {
            std::fprintf(stderr, "[qnn_oracle_bench] generate failed rc=%d\n", grc);
            return 5;
        }

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tps     = (double)out_ids.size() / std::max(elapsed, 1e-9);

        std::string gen_text = tk->decode(out_ids);
        std::printf("PROMPT : %s\nOUTPUT : %s\n", prompt.c_str(), gen_text.c_str());
        std::fprintf(stderr, "[qnn_oracle_bench] generated %zu tokens in %.3fs = %.1f tok/s\n",
                     out_ids.size(), elapsed, tps);

        if (!oracle_path.empty() && oracle.n_total() > 0) {
            std::fprintf(stderr,
                "[qnn_oracle_bench] oracle accuracy: %.1f%% (%d/%d) — "
                "effective spec speedup estimate: %.2fx\n",
                oracle.accuracy() * 100.f, oracle.n_hits(), oracle.n_total(),
                // Expected tokens per step with p=accuracy, K=SP_ORACLE_DRAFT_N:
                // sum_{k=0}^{K} p^k = (1 - p^{K+1}) / (1 - p)
                [&]() -> double {
                    float p = oracle.accuracy();
                    int K   = sp::engine::SP_ORACLE_DRAFT_N;
                    if (p >= 1.0f) return (double)(K + 1);
                    double sum = 0.0;
                    double pk  = 1.0;
                    for (int k = 0; k <= K; ++k) { sum += pk; pk *= (double)p; }
                    return sum;
                }());
        }
        return 0;
    }

    // Phase 8: full multi-token decode via QnnBinSession.
    //   sp-engine qnn_bin_generate
    //     --tokenizer  <gguf>
    //     --prompt     <text>
    //     [--n-predict N]  (default 128)
    //     [--ar 128] [--cl 2048] [--head-dim 2560] [--rope-base 1000000]
    //     split1 split2 split3 split4
    if (cmd == "qnn_bin_generate") {
        std::string tok_path, prompt;
        int   ar = 128, cl = 2048, hd = 2560, n_predict = 128;
        float rope_base = 1000000.0f;
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--tokenizer" && i + 1 < argc) tok_path  = argv[++i];
            else if (a == "--prompt"    && i + 1 < argc) prompt    = argv[++i];
            else if (a == "--n-predict" && i + 1 < argc) n_predict = std::atoi(argv[++i]);
            else if (a == "--ar"        && i + 1 < argc) ar        = std::atoi(argv[++i]);
            else if (a == "--cl"        && i + 1 < argc) cl        = std::atoi(argv[++i]);
            else if (a == "--head-dim"  && i + 1 < argc) hd        = std::atoi(argv[++i]);
            else if (a == "--rope-base" && i + 1 < argc) rope_base = (float)std::atof(argv[++i]);
            else splits.emplace_back(std::move(a));
        }
        if (tok_path.empty() || prompt.empty() || splits.size() != 4) {
            std::fprintf(stderr,
                "qnn_bin_generate: needs --tokenizer <gguf> --prompt <text> "
                "and exactly 4 split paths\n");
            return 1;
        }
        auto m  = sp::engine::Model::load(tok_path);
        if (!m) { std::fprintf(stderr, "Model::load failed\n"); return 2; }
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        if (!tk) { std::fprintf(stderr, "tokenizer init failed\n"); return 3; }
        std::vector<int32_t> ids;
        tk->encode(prompt, /*add_bos=*/true, ids);

        sp::engine::QnnBinSession session;
        if (session.load(splits, ar, cl, hd, rope_base) != 0) return 4;

        std::vector<int32_t> out_ids;
        auto t0 = std::chrono::steady_clock::now();
        int grc = session.generate(ids, n_predict, out_ids);
        auto t1 = std::chrono::steady_clock::now();
        if (grc != 0) {
            std::fprintf(stderr, "qnn_bin_generate failed rc=%d\n", grc);
            return 5;
        }
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::string gen = tk->decode(out_ids);
        std::printf("PROMPT : %s\nOUTPUT : %s\n", prompt.c_str(), gen.c_str());
        std::fprintf(stderr, "[qnn_bin_generate] %zu tokens in %.3fs = %.1f tok/s\n",
                     out_ids.size(), elapsed,
                     (double)out_ids.size() / std::max(elapsed, 1e-9));
        return 0;
    }
#endif  // SP_ENGINE_WITH_QNN

#if defined(SP_ENGINE_HEXAGON_FASTRPC)
    // Phase 7B: raw DMA probe via dmaWrapper.h.
    //   sp-engine qnn_probe_dma [--bytes N]
    // Allocates an rpcmem buffer of N bytes (default 1 MB), fills it with a
    // known pattern, and calls sp_hexagon_probe_dma_raw. Prints the result
    // code and transfer time. Exit 0 = DMA engine accepted; non-zero = blocked.
    if (cmd == "qnn_probe_dma") {
        int probe_bytes = 1024 * 1024;  // 1 MB default
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--bytes" && i + 1 < argc) probe_bytes = std::atoi(argv[++i]);
        }
        if (probe_bytes <= 0 || (probe_bytes & 1)) {
            std::fprintf(stderr, "qnn_probe_dma: --bytes must be a positive even integer\n");
            return 1;
        }

        rpcmem_init();
        void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, probe_bytes);
        if (!buf) {
            std::fprintf(stderr, "qnn_probe_dma: rpcmem_alloc(%d) failed\n", probe_bytes);
            return 1;
        }

        // Fill with known pattern so we can verify correctness on pass.
        uint8_t *b = (uint8_t *)buf;
        for (int i = 0; i < probe_bytes; ++i) b[i] = (uint8_t)(i & 0xFF);

        int result = 0, timing_us = -1;
        int rpc = sp_hexagon_probe_dma_raw(buf, probe_bytes, &result, &timing_us);

        rpcmem_free(buf);
        rpcmem_deinit();

        if (rpc == 0 && result == 0) {
            std::printf("[qnn_probe_dma] PASS: DMA engine accepted raw bytes. "
                        "bytes=%d timing_us=%d\n", probe_bytes, timing_us);
            return 0;
        } else if (result == 0x4E) {
            std::printf("[qnn_probe_dma] BLOCKED: DMA engine inaccessible in unsigned PD "
                        "(0x4E EPERM — same as Halide DMA result). "
                        "Weight streaming requires signed PD / testsig.\n");
            return 2;  // distinct from general failure (1) — caller can check
        } else {
            std::printf("[qnn_probe_dma] FAIL: rpc=%d result=%d timing_us=%d\n",
                        rpc, result, timing_us);
            return 1;
        }
    }
#endif  // SP_ENGINE_HEXAGON_FASTRPC

    // `run` is dispatched here (before the global flag parser) so its
    // verb-local flags — --n-predict in particular — aren't rejected as
    // unknown by the strict global parser below. Same pattern as cache_ppl,
    // perplexity, and chat.
    if (cmd == "run") {
        // Library-level API: load via Engine class, greedy generate.
        // Accepts ALL engine Config flags via the shared parser.
        sp::engine::Config rcfg;
        sp::engine::seed_config_from_env(rcfg);
        if (const char* env = std::getenv("SP_ENGINE_BACKEND")) {
            if (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "cuda") == 0) {
                rcfg.backend = sp::engine::Config::Backend::CUDA;
            } else if (std::strcmp(env, "vulkan") == 0) {
                rcfg.backend = sp::engine::Config::Backend::Vulkan;
            }
        }
        int n_predict = 32;
        std::string prompt;
        for (int i = 2; i < argc; ++i) {
            const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;
            int consumed = parse_config_flag(rcfg, argv[i], next);
            if (consumed > 0) { i += consumed - 1; continue; }
            std::string a = argv[i];
            if      (a == "--n-predict" && i + 1 < argc) n_predict = std::atoi(argv[++i]);
            else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "run: unknown flag %s\n", a.c_str()); return 2;
            }
            else { if (!prompt.empty()) prompt.push_back(' '); prompt += a; }
        }
        if (rcfg.model_path.empty()) {
            std::fprintf(stderr, "run requires --model <path.gguf>\n"); return 1;
        }
        if (prompt.empty()) {
            std::fprintf(stderr, "run requires a prompt\n"); return 1;
        }
        sp::engine::Engine engine;
        int lr = engine.load(rcfg);
        if (lr != 0) return lr;
        std::string out;
        int gr = engine.generate(prompt, n_predict, out);
        if (gr != 0) return gr;
        std::printf("%s\n", out.c_str());
        return 0;
    }

    // Phase 3.7: serve — bring up an OpenAI-compatible HTTP server backed
    // by Engine::generate. Drop-in replacement for the llama.cpp server
    // that the FastAPI proxy (lms_custom_proxy.py) currently routes to
    // on phone:8082.
    //
    //   sp-engine serve --model <gguf> [--host 0.0.0.0] [--port 8082]
    //                   [--name <model-id>]
    //
    // Listens on host:port, blocks until Ctrl-C. Endpoints:
    //   GET  /health                  health check
    //   GET  /v1/models               list loaded model
    //   POST /v1/chat/completions     chat (non-streaming, ChatML template)
    //   POST /v1/completions          raw prompt (non-streaming)
    if (cmd == "serve") {
        // OpenAI-compatible HTTP server — accepts ALL engine Config flags.
        sp::engine::Config rcfg;
        sp::engine::seed_config_from_env(rcfg);
        if (const char* env = std::getenv("SP_ENGINE_BACKEND")) {
            if (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "cuda") == 0) {
                rcfg.backend = sp::engine::Config::Backend::CUDA;
            } else if (std::strcmp(env, "vulkan") == 0) {
                rcfg.backend = sp::engine::Config::Backend::Vulkan;
            }
        }
        std::string host = "0.0.0.0";
        int port = 8082;
        std::string model_id;
        std::string www_root;
        for (int i = 2; i < argc; ++i) {
            const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;
            int consumed = parse_config_flag(rcfg, argv[i], next);
            if (consumed > 0) { i += consumed - 1; continue; }
            std::string a = argv[i];
            if      (a == "--host"  && i + 1 < argc) host     = argv[++i];
            else if (a == "--port"  && i + 1 < argc) port     = std::atoi(argv[++i]);
            else if (a == "--name"  && i + 1 < argc) model_id = argv[++i];
            else if (a == "--www"   && i + 1 < argc) www_root = argv[++i];
            else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "serve: unknown flag %s\n", a.c_str()); return 2;
            }
        }
        if (rcfg.model_path.empty()) {
            std::fprintf(stderr, "serve requires --model <path.gguf>\n"); return 1;
        }
        if (model_id.empty()) {
            // Use the basename of the GGUF as the public model id.
            std::string p = rcfg.model_path;
            size_t slash = p.find_last_of("/\\");
            std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
            size_t dot = base.rfind('.');
            model_id = (dot == std::string::npos) ? base : base.substr(0, dot);
        }
        std::fprintf(stderr, "[sp-engine:serve] loading %s ...\n", rcfg.model_path.c_str());
        sp::engine::Engine engine;
        int lr = engine.load(rcfg);
        if (lr != 0) {
            std::fprintf(stderr, "[sp-engine:serve] engine.load failed (%d)\n", lr);
            return lr;
        }
        std::fprintf(stderr, "[sp-engine:serve] model loaded; binding HTTP server\n");
        sp::engine::HttpServer server;
        server.bind(&engine, model_id, www_root);
        return server.listen_and_serve(host, port);
    }

#ifdef SP_ENGINE_WITH_QNN
    if (cmd == "qnn_bin_serve") {
        std::string tok_path, host = "0.0.0.0", model_id = "shannon-prime-htp", www_root;
        int port = 8080, ar = 128, cl = 2048, hd = 2560;
        float rope_base = 1000000.0f;
        std::vector<std::string> splits;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--tokenizer" && i + 1 < argc) tok_path = argv[++i];
            else if (a == "--host"      && i + 1 < argc) host     = argv[++i];
            else if (a == "--port"      && i + 1 < argc) port     = std::atoi(argv[++i]);
            else if (a == "--name"      && i + 1 < argc) model_id = argv[++i];
            else if (a == "--www"       && i + 1 < argc) www_root = argv[++i];
            else if (a == "--ar"        && i + 1 < argc) ar = std::atoi(argv[++i]);
            else if (a == "--cl"        && i + 1 < argc) cl = std::atoi(argv[++i]);
            else if (a == "--head-dim"  && i + 1 < argc) hd = std::atoi(argv[++i]);
            else if (a == "--rope-base" && i + 1 < argc) rope_base = (float)std::atof(argv[++i]);
            else splits.push_back(std::move(a));
        }
        if (tok_path.empty() || splits.size() != 4) {
            std::fprintf(stderr, "qnn_bin_serve: needs --tokenizer <gguf> and 4 split paths\n");
            return 1;
        }
        sp::engine::QnnBinSession session;
        if (session.load(splits, ar, cl, hd, rope_base) != 0) return 2;

        // Load tokenizer
        auto m = sp::engine::Model::load(tok_path);
        auto v = m ? sp::engine::Vocab::load(*m) : nullptr;
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        if (!tk) { std::fprintf(stderr, "tokenizer init failed\n"); return 3; }

        httplib::Server svr;
        // CORS
        svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            if (req.method == "OPTIONS") { res.status = 204; return httplib::Server::HandlerResponse::Handled; }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        if (!www_root.empty()) {
            std::fprintf(stderr, "[qnn_bin_serve] serving static files from %s\n", www_root.c_str());
            svr.set_mount_point("/", www_root);
        }

        svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"data\":[{\"id\":\"" + model_id + "\",\"object\":\"model\"}]}", "application/json");
        });

        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            // Minimal ChatML extraction for testing
            size_t p = req.body.find("\"content\":");
            if (p == std::string::npos) {
                res.status = 400; res.set_content("{\"error\":\"missing content\"}", "application/json");
                return;
            }
            size_t q = req.body.find("\"", p + 10);
            size_t r = req.body.find("\"", q + 1);
            std::string prompt = req.body.substr(q + 1, r - q - 1);

            std::vector<int32_t> ids, out_ids;
            tk->encode(prompt, true, ids);
            std::fprintf(stderr, "[qnn_bin_serve] generating for %zu tokens...\n", ids.size());
            
            // Note: generate() in QnnBinSession currently needs to be implemented
            // to use the persistent splits. For this test, we'll use a 1-token 
            // placeholder or ensure generate() is fully wired.
            session.generate(ids, 128, out_ids);
            
            std::string text = tk->decode(out_ids);
            std::string escaped; 
            for (char c : text) {
                if (c == '"') escaped += "\\\"";
                else if (c == '\n') escaped += "\\n";
                else escaped += c;
            }

            res.set_content("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" + escaped + "\"}}]}", "application/json");
        });

        std::fprintf(stderr, "[qnn_bin_serve] listening on %s:%d\n", host.c_str(), port);
        svr.listen(host, port);
        return 0;
    }
#endif // SP_ENGINE_WITH_QNN

    // Flag parser — extracts known flags and stashes positional args in `rest`.
    // Per-command handlers below can consume those positionals however they like.
    // Uses the shared parse_config_flag() so info/logits/prefill/etc. get the
    // full engine flag set for free.
    sp::engine::Config cfg;
    sp::engine::seed_config_from_env(cfg);
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) {
        const char* a    = argv[i];
        const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;

        int ate = parse_config_flag(cfg, a, next);
        if (ate > 0) { i += ate - 1; continue; }

        if (a[0] == '-' && a[1] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a);
            return 2;
        }
        rest.emplace_back(a);
    }

    if (cmd == "info") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "info requires --model <path.gguf>\n");
            return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        m->print_summary(stdout);

        // Also show the first few tensors so the user can spot-check the
        // layout without loading a full inspection tool.
        std::printf("\n  first %d tensors:\n",
                    (int)std::min<size_t>(m->tensor_count(), 8));
        for (size_t i = 0; i < m->tensor_count() && i < 8; ++i) {
            auto ti = m->tensor_info(i);
            std::printf("    [%3zu] %-48s type=%d size=%llu B\n",
                        i, ti.name.c_str(), ti.type,
                        (unsigned long long)ti.n_bytes);
        }
        if (m->tensor_count() > 8) {
            std::printf("    ... (%zu total)\n", m->tensor_count());
        }

        std::printf("\n");
        auto v = sp::engine::Vocab::load(*m);
        if (v) {
            v->print_summary(stdout);
            if (v->size() >= 3) {
                std::printf("  sample: [0]=%-12s [1]=%-12s [2]=%-12s\n",
                            v->token(0).c_str(), v->token(1).c_str(), v->token(2).c_str());
            }
        } else {
            std::printf("Tokenizer: (no vocab section in this GGUF)\n");
        }

        std::printf("\n");
        auto w = sp::engine::LlamaWeights::load(*m);
        if (w) {
            w->print_summary(stdout);
        } else {
            std::printf("Weights: (arch binding failed — unsupported arch or missing tensor)\n");
        }

        // Hybrid-arch smoke test: for qwen35moe/qwen35, allocate a
        // GdnStateCache sized from the GGUF ssm.* keys and print its
        // footprint. This double-checks the shape math and the layer-kind
        // classification end-to-end.
        const std::string& info_arch = m->architecture();
        const bool info_hybrid = (info_arch == "qwen35moe" || info_arch == "qwen35");
        if (w && info_hybrid) {
            const int conv_kernel   = (int)m->get_i64(info_arch + ".ssm.conv_kernel",    4);
            const int d_state       = (int)m->get_i64(info_arch + ".ssm.state_size",     128);
            const int n_group       = (int)m->get_i64(info_arch + ".ssm.group_count",    16);
            const int num_v_heads   = (int)m->get_i64(info_arch + ".ssm.time_step_rank", 32);
            const int d_inner       = (int)m->get_i64(info_arch + ".ssm.inner_size",     4096);
            const int conv_channels = d_inner + 2 * n_group * d_state;
            const int head_v_dim    = (num_v_heads > 0) ? (d_inner / num_v_heads) : 0;

            std::vector<bool> is_gdn; is_gdn.reserve(w->layers().size());
            for (const auto& L : w->layers()) {
                is_gdn.push_back(L.kind == sp::engine::LlamaLayerKind::MOE_GDN);
            }
            auto gdn = sp::engine::GdnStateCache::create(
                is_gdn, conv_kernel, conv_channels, head_v_dim, num_v_heads, /*n_seqs=*/1);
            std::printf("\n");
            if (gdn) {
                gdn->print_summary(stdout);
            } else {
                std::printf("GdnStateCache: (allocation failed)\n");
            }
        }
        return 0;
    }

    if (cmd == "encode" || cmd == "decode") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "%s requires --model <path.gguf>\n", cmd.c_str());
            return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        if (!v) { std::fprintf(stderr, "no vocab\n"); return 3; }
        auto tk = sp::engine::Tokenizer::create(*v);
        if (!tk) return 4;

        if (cmd == "encode") {
            std::string text;
            for (size_t i = 0; i < rest.size(); ++i) {
                if (i) text.push_back(' ');
                text += rest[i];
            }
            std::vector<int32_t> ids;
            tk->encode(text, /*add_bos=*/true, ids);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::printf("%s%d", i ? " " : "", ids[i]);
            }
            std::printf("\n");
            std::fprintf(stderr, "(%zu tokens)\n", ids.size());
            return 0;
        }

        // decode
        std::vector<int32_t> ids;
        for (const auto& s : rest) ids.push_back(std::atoi(s.c_str()));
        std::string out = tk->decode(ids);
        std::printf("%s\n", out.c_str());
        return 0;
    }

    if (cmd == "embed") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "embed requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 512*1024*1024, pe);
        if (!fc) return 4;

        std::vector<float> emb;
        int n_embd = 0;
        if (!fc->embed(ids, emb, n_embd)) {
            std::fprintf(stderr, "embed failed\n"); return 5;
        }

        // Print a summary: shape, a few values at the start / middle / end,
        // and the mean + std of the whole block for sanity.
        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        for (float f : emb) { sum += f; sumsq += (double)f * f; }
        double mean = sum / emb.size();
        double var  = (sumsq / emb.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_embd=%d  n_elems=%zu\n", n, n_embd, emb.size());
        std::printf("mean=%.6f  std=%.6f  min=%.6f  max=%.6f\n",
                    mean, stdv,
                    *std::min_element(emb.begin(), emb.end()),
                    *std::max_element(emb.begin(), emb.end()));
        std::printf("emb[0][:4]   = %+.6f %+.6f %+.6f %+.6f\n",
                    emb[0], emb[1], emb[2], emb[3]);
        if (n >= 2) {
            std::printf("emb[1][:4]   = %+.6f %+.6f %+.6f %+.6f\n",
                        emb[n_embd], emb[n_embd+1], emb[n_embd+2], emb[n_embd+3]);
        }
        return 0;
    }

    if (cmd == "block1") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "block1 requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W,
                      /*ctx_size_bytes=*/256 * 1024 * 1024, pe);
        if (!fc) return 4;

        std::vector<float> out;
        int n_embd = 0;
        if (!fc->forward_one_block(ids, out, n_embd)) {
            std::fprintf(stderr, "forward_one_block failed\n"); return 5;
        }

        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        int n_nan = 0;
        for (float f : out) {
            if (std::isnan(f) || std::isinf(f)) { n_nan++; continue; }
            sum += f; sumsq += (double)f * f;
        }
        double mean = sum / out.size();
        double var  = (sumsq / out.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_embd=%d  n_elems=%zu  n_nan=%d\n",
                    n, n_embd, out.size(), n_nan);
        std::printf("mean=%.6f  std=%.6f\n", mean, stdv);
        std::printf("out[0][:4]  = %+.6f %+.6f %+.6f %+.6f\n",
                    out[0], out[1], out[2], out[3]);
        if (n >= 2) {
            std::printf("out[-1][:4] = %+.6f %+.6f %+.6f %+.6f\n",
                        out[(size_t)(n-1) * n_embd + 0], out[(size_t)(n-1) * n_embd + 1],
                        out[(size_t)(n-1) * n_embd + 2], out[(size_t)(n-1) * n_embd + 3]);
        }
        return 0;
    }

    if (cmd == "logits") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "logits requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        std::fprintf(stderr, "[sp-engine] PE: %s\n",
                     sp::engine::prime_pe_describe(cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier).c_str());
        auto fc = sp::engine::ForwardContext::create(*m, *W,
                      /*ctx_size_bytes=*/1024 * 1024 * 1024, pe);
        if (!fc) return 4;

        // Hybrid-arch (qwen35moe / qwen35): allocate and bind a GdnStateCache
        // so the per-layer delta-rule recurrent state persists across this
        // forward call. For a single-shot prefill the cache starts
        // zeroed, so behaviourally this matches Stage 1 — but the wiring
        // is in place for multi-call decode where the state matters.
        const std::string& fwd_arch = m->architecture();
        const bool fwd_hybrid = (fwd_arch == "qwen35moe" || fwd_arch == "qwen35");
        std::unique_ptr<sp::engine::GdnStateCache> gdn_cache;
        if (fwd_hybrid) {
            const int conv_kernel   = (int)m->get_i64(fwd_arch + ".ssm.conv_kernel",    4);
            const int d_state       = (int)m->get_i64(fwd_arch + ".ssm.state_size",     128);
            const int n_group       = (int)m->get_i64(fwd_arch + ".ssm.group_count",    16);
            const int num_v_heads   = (int)m->get_i64(fwd_arch + ".ssm.time_step_rank", 32);
            const int d_inner       = (int)m->get_i64(fwd_arch + ".ssm.inner_size",     4096);
            const int conv_channels = d_inner + 2 * n_group * d_state;
            const int head_v_dim    = (num_v_heads > 0) ? (d_inner / num_v_heads) : 0;
            std::vector<bool> is_gdn; is_gdn.reserve(W->layers().size());
            for (const auto& L : W->layers()) {
                is_gdn.push_back(L.kind == sp::engine::LlamaLayerKind::MOE_GDN);
            }
            gdn_cache = sp::engine::GdnStateCache::create(
                is_gdn, conv_kernel, conv_channels, head_v_dim, num_v_heads, /*n_seqs=*/1);
            if (gdn_cache) {
                gdn_cache->reset();
                fc->bind_gdn_state(gdn_cache.get());
            } else {
                std::fprintf(stderr,
                    "[sp-engine] logits: GdnStateCache alloc failed — "
                    "falling back to per-call zero state.\n");
            }
        }

        std::vector<float> logits;
        int n_vocab = 0;
        if (!fc->forward_full(ids, logits, n_vocab)) {
            std::fprintf(stderr, "forward_full failed\n"); return 5;
        }

        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        int n_nan = 0;
        for (float f : logits) {
            if (std::isnan(f) || std::isinf(f)) { n_nan++; continue; }
            sum += f; sumsq += (double)f * f;
        }
        double mean = sum / logits.size();
        double var  = (sumsq / logits.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_vocab=%d  n_elems=%zu  n_nan=%d\n",
                    n, n_vocab, logits.size(), n_nan);
        std::printf("mean=%.6f  std=%.6f  min=%.6f  max=%.6f\n",
                    mean, stdv,
                    *std::min_element(logits.begin(), logits.end()),
                    *std::max_element(logits.begin(), logits.end()));

        // Last-row argmax: the next-token prediction for the prompt.
        const float* last = logits.data() + (size_t)(n - 1) * n_vocab;
        int arg = 0;
        float best = last[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (last[i] > best) { best = last[i]; arg = i; }
        }
        std::printf("argmax(last) = %d  logit=%+.4f  token=\"%s\"\n",
                    arg, best,
                    (v && arg >= 0 && (size_t)arg < v->size()) ? v->token(arg).c_str() : "?");

        // Top-5 of the last row for a sanity sample.
        std::vector<std::pair<float,int>> topv;
        topv.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) topv.emplace_back(last[i], i);
        std::partial_sort(topv.begin(), topv.begin() + 5, topv.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        std::printf("top5:");
        for (int i = 0; i < 5; ++i) {
            int id = topv[i].second;
            std::printf("  [%d %s %+.3f]",
                        id,
                        (v && (size_t)id < v->size()) ? v->token(id).c_str() : "?",
                        topv[i].first);
        }
        std::printf("\n");
        return 0;
    }

    if (cmd == "prefill") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "prefill requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        cfg.arch_name = m->architecture();
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);
        const int n = (int)ids.size();
        if (n == 0) { std::fprintf(stderr, "prefill: empty text\n"); return 4; }

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe);
        if (!fc) return 5;

        std::vector<float> logits;
        int n_vocab = 0;
        std::vector<std::vector<float>> Ks, Vs;
        if (!fc->forward_full(ids, logits, n_vocab, &Ks, &Vs)) {
            std::fprintf(stderr, "prefill: forward_full failed\n"); return 6;
        }

        const int n_layer   = fc->n_layer();
        const int head_dim  = (int)m->head_dim();
        const int n_head_kv = (int)m->n_head_kv();

        // prefill is a CPU-only diagnostic verb — forward runs on CPU-mmap
        // weights and records per-layer K/V correlation. GPU cache routing
        // only makes sense when forward is already on GPU, so this verb
        // keeps the host cache unconditionally.
        auto kv = sp::engine::KvCache::create(n_layer, n_head_kv, head_dim, n, cfg);
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 7; }
        std::fprintf(stderr, "[sp-engine] %s\n", kv->describe().c_str());

        // Calibrate before writing.
        if (!kv->is_calibrated()) {
            if (kv->calibrate_begin()) {
                const bool hier = kv->is_hierarchical();
                for (int L = 0; L < n_layer; ++L) {
                    const float* K_data = Ks[(size_t)L].data();
                    for (int q = 0; q < n; ++q) {
                        for (int h = 0; h < n_head_kv; ++h) {
                            const float* vec = K_data + (size_t)(q * n_head_kv + h) * head_dim;
                            if (hier) {
                                kv->calibrate_feed(L * n_head_kv + h, vec);
                            } else {
                                kv->calibrate_feed(vec);
                            }
                        }
                    }
                }
                kv->calibrate_end();
            }
        }

        // Push every layer's captured K/V through the compressed cache.
        for (int L = 0; L < n_layer; ++L) {
            if (!kv->write(L, /*pos_offset=*/0, n, Ks[(size_t)L].data(), Vs[(size_t)L].data())) {
                std::fprintf(stderr, "kv->write layer %d failed\n", L); return 8;
            }
        }

        auto corr = [](const float* a, const float* b, int len) {
            double ma = 0, mb = 0;
            for (int i = 0; i < len; ++i) { ma += a[i]; mb += b[i]; }
            ma /= len; mb /= len;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < len; ++i) {
                double da = a[i] - ma, db = b[i] - mb;
                sxy += da * db; sxx += da * da; syy += db * db;
            }
            double denom = std::sqrt(sxx * syy);
            return (denom > 0) ? (float)(sxy / denom) : 0.0f;
        };

        // Read every layer back, compute mean K corr / V corr per layer.
        std::printf("layer  K_corr   V_corr  K_min   V_min\n");
        double overall_k = 0, overall_v = 0;
        for (int L = 0; L < n_layer; ++L) {
            std::vector<float> Krec, Vrec;
            if (!kv->read(L, n, Krec, Vrec)) {
                std::fprintf(stderr, "kv->read layer %d failed\n", L); return 9;
            }
            double k_sum = 0, v_sum = 0;
            float  k_min = 1.0f, v_min = 1.0f;
            const int per = n_head_kv * n;
            for (int q = 0; q < n; ++q) {
                for (int h = 0; h < n_head_kv; ++h) {
                    const float* k0 = Ks[(size_t)L].data() + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* k1 = Krec.data()           + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* v0 = Vs[(size_t)L].data() + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* v1 = Vrec.data()           + (size_t)(q * n_head_kv + h) * head_dim;
                    float kc = corr(k0, k1, head_dim);
                    float vc = corr(v0, v1, head_dim);
                    k_sum += kc; v_sum += vc;
                    if (kc < k_min) k_min = kc;
                    if (vc < v_min) v_min = vc;
                }
            }
            float km = (float)(k_sum / per), vm = (float)(v_sum / per);
            overall_k += km; overall_v += vm;
            // Show a sample of layers to keep the table small.
            if (L < 4 || L == n_layer - 1 || L == n_layer / 2) {
                std::printf("%3d   %6.4f  %6.4f  %6.4f  %6.4f\n", L, km, vm, k_min, v_min);
            }
        }
        std::printf("---\nmean over %d layers: K_corr=%.4f  V_corr=%.4f  compression=%.2fx\n",
                    n_layer, overall_k / n_layer, overall_v / n_layer,
                    kv->compression_ratio());
        return 0;
    }

    usage(argv[0]);
    return 1;
}
