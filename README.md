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
├─────────────────────────────────────────────────────────┤
│  lib/shannon-prime/ ← core math (git submodule)         │
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

# Aggressive compression
./build/sp-engine run \
  --model /path/to/model.gguf \
  --sp-compression sqfree \
  --sp-spinor \
  --prompt "Hello, world"

# Hierarchical (maximum compression, requires calibration)
./build/sp-engine run \
  --model /path/to/model.gguf \
  --sp-compression hier \
  --sp-hier-level 0 \
  --sp-hier-res-bits 2 \
  --prompt "Hello, world"

# Perplexity evaluation
./build/sp-engine ppl \
  --model /path/to/model.gguf \
  --dataset wikitext-2 \
  --chunks 3

# HTTP server
./build/sp-engine serve \
  --model /path/to/model.gguf \
  --port 8082

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
