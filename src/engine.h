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
    uint32_t    k_ternary_mask = 0;  // Ternary band mask for main K quant (0x8 = band 3)
    uint32_t    v_ternary_mask = 0;  // Ternary band mask for main V quant
    bool        use_fp8       = false; // FP8 (E4M3) for V cache (smooth distributions)

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
    bool        hierarchical    = true;
    int         hier_level      = 0;       // 0 = auto (second-to-last prime grouping)
    int         hier_res_bits   = 2;       // 1-4 bits for K target residuals
    int         hier_res_bits_v = 0;       // V target residuals; 0 → same as K
    std::string hier_skel_bits  = "5,5";   // Band bits for skeleton quantisation
    uint32_t    hier_skel_ternary = 0;     // Ternary band mask for skeleton (0x8 = band 3)

    // Backend selection.
    enum class Backend { CPU, CUDA, Vulkan };
    Backend     backend = Backend::CPU;
    int         n_gpu_layers = 0;

    // Multi-GPU sharding — distribute transformer layers across GPUs.
    //
    //   n_gpus = 0 → auto-detect all available GPUs (default)
    //   n_gpus = 1 → single GPU (current behaviour, no sharding)
    //   n_gpus > 1 → shard layers across that many GPUs
    //
    // Layer L is assigned to GPU[ L * n_gpus / n_layer ]. Non-layer
    // tensors (tok_embd, output_norm, output) go to GPU 0 when fully
    // offloaded, or stay CPU-mapped under partial offload.
    //
    // The scheduler handles cross-GPU copies transparently — when a
    // tensor produced on GPU i is consumed by an op on GPU j, it gets
    // an automatic copy node inserted.
    int         n_gpus = 0;

    // CRT (Chinese Remainder Theorem) multi-GPU parallelism.
    // When true and n_gpus >= 2, uses CRT tensor splitting instead of
    // layer sharding: each GPU computes in a different residue ring,
    // then the host recombines via Garner's algorithm. Eliminates all
    // inter-GPU communication during matmul. Particularly effective
    // for heterogeneous GPU pairs (e.g., RTX 2060 + Intel UHD).
    bool        crt_split = false;

    // MoE expert curriculum — homeostatic expert balancer for Beast Canyon.
    // When true and the model is MoE (n_expert >= 2), enables EWMA-based
    // expert heatmap tracking + predictive prefetch for zero-bubble inference.
    // Hot experts → RTX 2060 (Tier 1), cool experts → Intel UHD (Tier 2).
    bool        moe_curriculum = false;

    // Beast Canyon heterogeneous MoE orchestrator.
    // When set to a GGUF path, the Beast Canyon engine boots:
    //   1. Optane reservoir (mmap of the GGUF file)
    //   2. AVX-512 Shredder (dequant → fp16 staging)
    //   3. Dual-GPU dispatch (CUDA + Vulkan)
    //   4. S22U sidecar (optional)
    // This replaces the normal forward path for MoE layers.
    std::string beast_gguf_path;

    // Positional-encoding mode.
    // Default is PrimePe (lattice-aligned RoPE frequencies) — proven −0.6%
    // to −0.8% PPL improvement across architectures at zero runtime cost.
    // Use Standard to fall back to pure geometric RoPE.
    enum class PeMode { Standard, PrimePe, PrimePeAlibi, AlibiOnly };
    PeMode      pe_mode  = PeMode::PrimePe;
    float       pe_alpha = 0.17f;
    int         pe_tier  = 0;

    // Cauchy reset system — decode-chain causal stability.
    int         cauchy_mode     = 2;
    int         cauchy_fixed_n  = 512;
    int         cauchy_cooldown = 64;
    int         cauchy_warmup   = 64;
    bool        cauchy_use_ricci = false;
    bool        cauchy_ricci_only = false;
    bool        cauchy_mertens_only = false;
    float       params_b        = 0.0f;

    // Hot/cold tiered storage — GPU VRAM → CPU pinned RAM → disk.
    int         cold_mb      = 0;
    int         evict_keep   = 0;
    bool        enable_cold  = false;

    // Disk serialisation — save/load compressed KV cache state.
    std::string save_cache_path;
    std::string load_cache_path;

    // System 1↔2 switching — entropy-gated dynamic cache routing.
    //
    // When enabled, the engine maintains two caches:
    //   System 1: ship path (fast, moderate compression)
    //   System 2: hier or sqfree path (slower, maximum fidelity)
    //
    // During decode, the output logit entropy after each token determines
    // which cache stores the NEXT token's K/V. High entropy (model is
    // uncertain, distributing probability mass widely) → System 2 for
    // maximum reconstruction fidelity on these "hard" tokens. Low entropy
    // (model is confident) → System 1 for speed.
    //
    // The threshold is in nats (natural log). Typical softmax entropy for
    // an 8B model ranges from ~0.3 (very confident) to ~8 (very uncertain).
    // Default threshold 2.0 routes ~15-25% of tokens to System 2.
    //
    // On read, the DualKvCache merges positions from both caches
    // transparently — the decode graph sees a single unified K/V history.
    bool        system12          = false;
    float       s12_threshold     = 2.0f;  // entropy threshold (nats)
    // System 2 cache type: "hier" (default) or "sqfree"
    std::string s12_sys2          = "hier";

    // -------------------------------------------------------------------
    // Phase 0 Theory-First: Frobenius / Sato-Tate quantization tiers.
    // Backed by lib/shannon-prime/core/sp_frobenius.{c,h}; verified by
    // tests/test_sp_frobenius (bit-exact vs Python oracle).
    //
    // --frobenius-quant: single split-prime Frobenius (Paper D Config B).
    //                    Default p=41 (smallest split in K=Q(sqrt(-163))), k=8.
    //
    // --sato-tate-mix p1,k1,p2,k2: asymmetric inert+split (Paper D Config E).
    //                    Default 2,2,41,8 — inert (zero drift) + split.
    // -------------------------------------------------------------------
    bool        frobenius_quant    = false;
    int64_t     frobenius_p        = 41;     // split prime
    int64_t     frobenius_k        = 8;      // Frobenius power
    bool        sato_tate_mix      = false;
    int64_t     st_p1              = 2;      // inert prime (zero-drift)
    int64_t     st_k1              = 2;      // phi_p1^k1 (k1 must be even)
    int64_t     st_p2              = 41;     // split prime (bounded-drift)
    int64_t     st_k2              = 8;      // phi_p2^k2

    // Phase 4b: Friedman sieve cache hook over the KV-write path.
    // friedman_mode: 0=off, 1=observer (counters only), 2=policy (decisions gate writes — Phase 4d).
    bool        friedman_sieve     = false;
    int         friedman_mode      = 1;       // 1=observer by default
    int         friedman_capacity  = 4096;
    float       kste_tau_A         = 0.0f;    // Phase-1 bootstrap default
    float       kste_alpha         = 0.7f;

    // Phase 12 Step B-2: --frobenius-q8.
    // After the Frobenius shim has run, round-trip every shim-list
    // sp_ok_tensor through the packed int8 lattice (sp_ok_q8_t pair + per-
    // tensor power-of-2 shift) and decode back into the same arena. The
    // forward path is unchanged; every weight now carries the int8
    // quantization error. Used to measure end-to-end PPL drift from Q8
    // before committing to resident packed storage in Step C.
    bool        frobenius_q8       = false;

    // Phase 14: --frobenius-q4.
    // Same semantics as frobenius_q8 but with the codebook halved:
    // packed 4-bit nybble pair per coordinate (1 byte per ring element).
    // 16x memory compression vs raw sp_ok_t, 2x vs Q8. Quantization noise
    // per coordinate is ~16x larger than Q8 (16-level codebook vs 256);
    // whether Theorem 2's projective cancellation absorbs that noise is
    // a forward-pass empirical question.
    //
    // q4 and q8 are mutually exclusive — q4 wins if both are set.
    bool        frobenius_q4       = false;

    // Phase 14b: lattice-norm pruning threshold for --frobenius-q4.
    // For every coordinate pair (a, b), compute N(a + b*omega) = a^2 + ab + 41 b^2
    // and zero the pair if N < threshold. Produces runs of 0x00 packed bytes
    // that compress aggressively under downstream entropy coding (zstd/Huffman).
    //
    // 0 disables pruning (encode every value).
    uint64_t    frobenius_q4_prune = 0;

    // Phase 15: --gguf-block-quant.
    // Read GGUF Q8_0 / Q4_0 tensors directly. Fuses each block's fp16
    // scale with the Frobenius element π^k into per-block (B_a, B_b)
    // integers at load time, leaves the int4/int8 codepoints untouched.
    // Solves the Phase 14 Q4 per-tensor-shift blowout by inheriting
    // GGUF's per-block scale resolution.
    //
    // Detected per-tensor: tensors stored as GGML_TYPE_Q8_0 or
    // GGML_TYPE_Q4_0 get fused; others fall back to the standard fp16
    // path. Mutually exclusive with --frobenius-q8 and --frobenius-q4
    // at the per-weights-load level (only one packed storage flag wins).
    bool        gguf_block_quant   = false;
};

// Seed Config fields from environment variables. Called by each CLI verb
// immediately after Config construction, so the precedence ordering stays:
//   Config default → env var → CLI flag.
inline void seed_config_from_env(Config& cfg) {
    if (cfg.model_preset.empty()) {
        if (const char* s = std::getenv("SHANNON_PRIME_MODEL_PRESET")) {
            cfg.model_preset = s;
        }
    }
    if (!cfg.enable_cold) {
        if (const char* s = std::getenv("SP_ENGINE_COLD_MB")) {
            cfg.cold_mb = std::atoi(s);
            cfg.enable_cold = (cfg.cold_mb > 0);
        }
    }
    if (cfg.evict_keep == 0) {
        if (const char* s = std::getenv("SP_ENGINE_EVICT_KEEP")) {
            cfg.evict_keep = std::atoi(s);
        }
    }
    if (cfg.save_cache_path.empty()) {
        if (const char* s = std::getenv("SP_ENGINE_SAVE_CACHE")) {
            cfg.save_cache_path = s;
        }
    }
    if (cfg.load_cache_path.empty()) {
        if (const char* s = std::getenv("SP_ENGINE_LOAD_CACHE")) {
            cfg.load_cache_path = s;
        }
    }
    if (!cfg.system12) {
        if (const char* s = std::getenv("SP_ENGINE_SYSTEM12")) {
            cfg.system12 = (std::atoi(s) != 0);
        }
    }
    if (cfg.s12_threshold == 2.0f) {
        if (const char* s = std::getenv("SP_ENGINE_S12_THRESHOLD")) {
            cfg.s12_threshold = (float)std::atof(s);
        }
    }
    if (cfg.s12_sys2.empty() || cfg.s12_sys2 == "hier") {
        if (const char* s = std::getenv("SP_ENGINE_S12_SYS2")) {
            cfg.s12_sys2 = s;
        }
    }
    if (cfg.n_gpus == 0) {
        if (const char* s = std::getenv("SP_ENGINE_N_GPUS")) {
            cfg.n_gpus = std::atoi(s);
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
