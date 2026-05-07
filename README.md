# Shannon-Prime Engine

**The reference inference engine with native compressed KV-cache storage.**

Shannon-Prime Engine is a standalone GGUF inference engine that owns the compressed KV data path end-to-end. Compression happens on the write path by construction — there is no "decompress to fp16, run attention, recompress" round-trip. The engine loads a GGUF model, runs native forward passes with Shannon-Prime VHT2 spectral compression active from the first token, and serves results via CLI or HTTP.

This is the primary deployment path for Shannon-Prime. Unlike the llama.cpp integration (which patches compression into someone else's inference stack), the engine was designed from the ground up around compressed KV storage.

---

## Why the Engine Exists

llama.cpp is an excellent inference framework, but it was not designed for compressed KV caches. Integrating Shannon-Prime into llama.cpp requires patching the KV write/read paths, working around assumptions about fp16 cache layout, and maintaining a patch against a moving upstream target. Every llama.cpp update risks breaking the integration.

The engine eliminates this friction. It owns the entire data path: GGUF loading, tokenization, forward pass, KV cache, attention, and output. Shannon-Prime compression is not bolted on — it's the native storage format.

---

## Features

**Native Compressed KV Cache.** The `KvCache` class wraps the C-level shadow cache (`sp_shadow_cache_t`), sqfree cache (`sp_sqfree_cache_t`), or hierarchical cache (`sp_hier_cache_t`) behind a single C++ interface. The engine writes raw fp32 K/V vectors in and reads reconstructed fp32 vectors out. The compression tier is selected at config time and runs transparently.

**GPU-Resident Cache.** `KvCache::create_gpu()` keeps compressed K/V blocks in VRAM. Compress and decompress run as CUDA kernels — no host round-trip. On Qwen3-8B Q8, the GPU cache runs 15.6× faster than host fallback (1m28s vs 23 min for full PPL evaluation).

**Multi-GPU Sharding.** Layer L → GPU[L × n_gpus / n_layer]. Non-layer tensors go to GPU 0 or stay CPU-mapped. Cross-GPU copies are inserted automatically.

**QNN HTP Dispatch.** Pre-compiled AI Hub `.bin` files execute on Qualcomm's Hexagon Tensor Processor via the QNN C API. 4-split execution at 85 ms/split steady-state, projecting to 376 tok/sec prefill on S22U V69.

**PrimePE.** Lattice-aligned RoPE frequency injection enabled by default. −0.6% to −0.8% PPL improvement at zero runtime cost.

**Adaptive Calibration.** Warmup-based variance ranking for all cache types. Ridge regression calibration for the hierarchical predictor. Sticky-EMA online adaptation.

**Model-Pack Presets.** `--model-preset auto` reads the GGUF architecture tag and applies known-good compression defaults.

**HTTP Server.** OpenAI-compatible `/v1/chat/completions` endpoint via `sp-engine serve`.

**Cauchy Reset.** Dynamic reset scheduling (Ricci sentinel + Mertens oracle) for long decode chains.

**MoE Expert Curriculum.** `--moe-curriculum` enables the homeostatic expert balancer for Mixture-of-Experts models. Per-layer EWMA heatmap tracks expert activation rates; a Curriculum Pulse every 128 tokens re-ranks experts and assigns them to GPU tiers (hot → discrete GPU, cool → iGPU). Validated on Qwen3.6-35B-A3B (256 experts, top-8, 40 layers).

**Top-2 Speculative Prefetch.** Dual-slot shadow buffers pre-shred the two hottest predicted experts for layer L+1 while GPUs grind layer L. Hit → pointer swap (~10ns), miss → standard shred (~2ms). Confidence gate (tau=0.75) skips speculation when the heatmap is too flat.

**CRT Multi-GPU Dispatch.** Chinese Remainder Theorem sharding across heterogeneous GPUs (e.g., RTX 2060 + Intel UHD). Mersenne-prime M1 ring on discrete GPU, general M2 ring on iGPU, Garner reconstruction on CPU.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    sp-engine CLI / HTTP                  │
├─────────────────────────────────────────────────────────┤
│  engine.cpp        ← orchestration, config, lifecycle   │
│  gguf_loader.cpp   ← GGUF v2/v3 model loading          │
│  tokenizer.cpp     ← BPE/SentencePiece tokenization    │
│  forward.cpp       ← transformer forward pass           │
│  kv_cache.cpp      ← compressed KV (ship/sqfree/hier)   │
│  sp_kernels_cpu.cpp← CPU compute kernels                │
│  sp_quant.cpp      ← weight dequantization              │
│  sp_tensor.cpp     ← tensor operations                  │
│  sp_threadpool.cpp ← work scheduling                    │
│  prime_pe.cpp      ← PrimePE frequency factors          │
│  gdn_state.cpp     ← GDN compression state              │
│  http_server.cpp   ← OpenAI-compatible API server       │
│  qnn_bin_driver.cpp← QNN HTP .bin execution             │
│  speculative_oracle.cpp ← NEON draft model oracle       │
├─────────────────────────────────────────────────────────┤
│  lib/shannon-prime/ ← core math (git submodule)         │
│    backends/crt/    ← CRT dispatch, MoE curriculum,     │
│                       prefetch engine, CUDA/Shor kernels│
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run inference
./build/sp-engine run \
  --model /path/to/model.gguf \
  --prompt "Explain the Vilenkin-Hartley transform"

# With compression (default: ship path)
./build/sp-engine run \
  --model /path/to/model.gguf \
  --sp-compression ship \
  --sp-k-bits 5,5,4,3 \
  --prompt "Hello, world"

# Sqfree compression
./build/bin/sp-engine run \
  --model /path/to/model.gguf \
  --sqfree --spinor \
  --n-predict 128 "Hello, world"

# Hierarchical (maximum compression, default mode)
./build/bin/sp-engine run \
  --model /path/to/model.gguf \
  --hier-res-bits 2 --hier-res-bits-v 1 \
  --n-predict 128 "Hello, world"

# Perplexity evaluation
./build/bin/sp-engine cache_ppl \
  --model /path/to/model.gguf \
  --chunks 3 wiki.test.raw

# HTTP server (all engine flags accepted)
./build/bin/sp-engine serve \
  --model /path/to/model.gguf \
  --port 8082 --moe-curriculum --system12

# MoE model with expert curriculum (heterogeneous dispatch)
./build/bin/sp-engine chat \
  --model /path/to/Qwen3.6-35B-A3B.gguf \
  --moe-curriculum -ngl 99 \
  "Explain mixture of experts"

# QNN .bin benchmark (phone, via adb)
./build/sp-engine qnn_bin_bench \
  --n-chunks 3 \
  split1.bin split2.bin split3.bin split4.bin
```

---

## Configuration

```cpp
sp::engine::Config cfg;
cfg.model_path = "/path/to/model.gguf";
cfg.n_ctx = 4096;
cfg.n_batch = 512;

// Compression
cfg.mobius = true;              // Möbius reorder (default)
cfg.k_bits_csv = "5,5,4,3";    // Ship path K bands
cfg.v_bits_csv = "3";           // Flat V quantization

// Aggressive paths (pick one)
cfg.sqfree = true;              // Sqfree + Knight skeleton
cfg.spinor = true;              // SU(2) sheet bit (requires sqfree)
cfg.hierarchical = true;        // Hierarchical Vilenkin predictor

// GPU
cfg.backend = Config::Backend::CUDA;
cfg.n_gpu_layers = 99;          // Offload all layers
cfg.n_gpus = 0;                 // Auto-detect GPU count

// Positional encoding
cfg.pe_mode = Config::PeMode::PrimePe;
cfg.pe_alpha = 0.17f;

// Model preset
cfg.model_preset = "auto";      // Resolve from GGUF arch

// Cauchy reset
cfg.cauchy_mode = 2;            // Dynamic (Ricci + Mertens)

// MoE expert curriculum (for MoE models like Qwen3.6-35B-A3B)
cfg.moe_curriculum = true;      // Enable heatmap + tier dispatch + prefetch
```

---

## Validated Results

| Model | Config | Result | Hardware |
|---|---|---|---|
| Qwen3-8B Q8 | Ship GPU cache | 1m28s PPL eval (15.6× vs host) | RTX 2060 |
| Qwen3.6-35B-A3B | Ship + PrimePE | 26.92 tok/sec | LM Studio (via llama bridge) |
| Qwen3-4B w4a16 | QNN .bin 4-split | 104 t/s prefill | S22U V69 HTP |
| Qwen2.5-Coder-3B + 0.5B | Spec-decode + FUSED_KQ | 43.72 t/s (3.58×) | S22U CPU |
| RTX 2060 + Intel UHD | Dual-GPU Vulkan | Cross-device correlation 1.0000 | Desktop |
| Qwen3.6-35B-A3B | MoE curriculum (256 experts, top-8) | 5364-node graph, heatmap active | Desktop CPU |

---

## Sibling Repositories

| Repository | Role |
|---|---|
| [shannon-prime](https://github.com/nihilistau/shannon-prime) | Core math library (VHT2, Möbius, bands, sqfree, hier). Vendored here as `lib/shannon-prime/`. |
| [shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui) | ComfyUI nodes for video/image/audio/TTS. Uses the torch backend. |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | llama.cpp patch for LM Studio. Bridge for the existing ecosystem. |

**Voxtral TTS forks** with integrated VHT2 KV compression:
[Python](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS),
[Rust](https://github.com/nihilistau/voxtral-mini-realtime-rs),
[C](https://github.com/nihilistau/voxtral-tts.c).

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).
Commercial license available — contact raydaniels@gmail.com.
