# Shannon-Prime Engine

**A standalone inference engine with native compressed KV-cache storage.**

Shannon-Prime Engine is the reference implementation of the Shannon-Prime
VHT2 + sqfree + spinor compression stack, packaged as a complete inference
binary rather than a patch applied to someone else's codebase. It owns the
KV layout end-to-end, so compression happens on the write path by
construction — no "decompress to fp16, run attention, recompress" round-trips.

This repo is a sibling of:

- [shannon-prime](https://github.com/nihilistau/shannon-prime) — the
  canonical core math library. Vendored here as a git submodule at
  `lib/shannon-prime/`.
- [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama)
  — the llama.cpp post-decode hook (the "patch in" integration option).

**Option relationship:** this engine is the "own the data path" option.
shannon-prime-llama is the "inherit llama.cpp's 30+ model architectures"
option. They share 100% of the core math via the submodule; they serve
different user stories.

## Status

Pre-alpha but live. The full forward pass (token-embedding →
RMSNorm → attention with PrimePE-RoPE + optional ALiBi → SwiGLU
FFN → output_norm → output projection → logits) runs end-to-end
on Llama-family GGUFs. Greedy generation through a stateful
decode loop reads past K/V from a compressed KvCache, attends,
writes new K/V back — and produces clean continuations across
ship + sqfree + sqfree+spinor configs.

| Stage | Verb | What it proves | Status |
|---|---|---|---|
| 3a | `embed` | ggml graph + backend init + token-embedding lookup. | ✓ |
| 3b | `block1` | One transformer block: norm + attn + FFN + residual. | ✓ |
| 3c | `logits` | All `n_layer` blocks + output head → real logits. | ✓ |
| 3d | `logits --pe-mode primepe` | PrimePE-RoPE-ALiBi (composite/prime-tiered lattice, alpha-blended). | ✓ |
| 4   | `kv_smoke` | KvCache wrapper around `sp_shadow_cache_t` / `sp_sqfree_cache_t`. | ✓ |
| 5a  | `prefill` | Real RoPE'd K from prefill compressed through KvCache; per-layer K/V correlation reported. | ✓ |
| 5b  | `chat` | Stateful prefill + optimised single-token decode reading from compressed cache. | ✓ |
| 6   | (planned) | Persistent backend KV tensors (avoid the per-step host round-trip); perplexity verb. | — |
| 7   | (planned) | CUDA / Vulkan backend selection; release packaging. | — |

### Measured K correlation on real RoPE'd K (engine `prefill` verb)

| Model | Path | K_corr (mean) | V_corr | Compression |
|---|---|---|---|---|
| Dolphin3.0-Llama3.2-1B-Q8 (hd=64, 16 layers) | ship 5,5,4,3 / 3 | **0.9941** | 0.9712 | 3.76× |
| Dolphin3.0-Llama3.2-1B-Q8 | sqfree (pad 66) | 0.9768 | 0.9484 | 3.76× |
| Dolphin3.0-Llama3.2-1B-Q8 | sqfree+spinor | 0.9869 | 0.9601 | 3.76× |
| Qwen3-8B-Q8 (hd=128, 36 layers) | ship 5,5,4,3 / 3 | **0.9934** | 0.9691 | 4.06× |

Spinor's +0.008–0.010 K-corr lift on the Knight skeleton matches
the value documented in `lib/shannon-prime/CLAUDE.md`. Ship hits the
0.992+ target on real model data, end-to-end, on both architectures.

### Greedy chat smoke tests (n_predict=20, prompt = "The quick brown fox")

| Config | Output |
|---|---|
| Dolphin-1B ship | *"...jumps over the lazy dog. The sentence \"The quick brown fox jumps over the lazy dog.\" is"* |
| Dolphin-1B sqfree+spinor | *"...What is the correct order of the sentence? To determine the correct order"* |
| Qwen3-8B-Q8 ship | *"...This sentence is a well-known pangram. It is used to test"* |

## Building

Requires:

- CMake 3.14+
- Ninja (or another CMake generator)
- A C++17 compiler (MSVC 14.29+ / GCC 13+ / Clang 15+)
- Optional: CUDA Toolkit 12+ for GPU builds, Vulkan SDK 1.3+ for cross-GPU
- Submodules checked out (see below)

```bash
git clone --recursive https://github.com/nihilistau/shannon-prime-engine
cd shannon-prime-engine

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/sp-engine --help
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Runtime note (Windows + MinGW builds)

The produced `sp-engine.exe` dynamically links the MinGW C/C++ runtime.
If your shell doesn't expose the MinGW bin directory on PATH you'll see
exit code 127 ("not recognized"). Fix by copying three DLLs next to the
binary (one-time):

```bash
cp /c/ProgramData/mingw64/mingw64/bin/libgcc_s_seh-1.dll   build/bin/
cp /c/ProgramData/mingw64/mingw64/bin/libstdc++-6.dll      build/bin/
cp /c/ProgramData/mingw64/mingw64/bin/libwinpthread-1.dll  build/bin/
```

MSVC-based release builds (planned) will ship the runtime statically.

## Repo layout

```
shannon-prime-engine/
├── lib/
│   └── shannon-prime/       ← git submodule → github.com/nihilistau/shannon-prime
├── vendor/
│   └── ggml/                ← git submodule → github.com/ggml-org/ggml (MIT)
├── src/                     ← engine code (this repo's original contribution)
│   ├── engine.{h,cpp}       Public API + Config (PeMode, sqfree, mobius)
│   ├── gguf_loader.{h,cpp}  Typed view over gguf_context
│   ├── vocab.{h,cpp}        tokenizer.ggml.* arrays
│   ├── tokenizer.{h,cpp}    GPT-2-style BPE encode / decode
│   ├── llama_weights.{h,cpp} Llama-family arch binding
│   ├── forward.{h,cpp}      ggml graph: embed, block, full, prefill, decode
│   ├── prime_pe.{h,cpp}     PrimePE-RoPE-ALiBi lattice math
│   ├── kv_cache.{h,cpp}     Wrapper around sp_shadow_cache_t / sp_sqfree_cache_t
│   └── cli/main.cpp         Verbs: info, encode, decode, embed, block1,
│                             logits, kv_smoke, prefill, chat
├── tests/
└── CMakeLists.txt
```

## License

**AGPLv3** for open-source, academic, and non-proprietary use.
Everyone can use it and benefit. Derivative works must share alike.

**Dual License** — the primary goal is that the work belongs to the
commons and is protected from closure. A commercial license is
available for proprietary integration.

Third-party components (ggml) retain their original **MIT** license;
see [LICENSE.third_party](LICENSE.third_party).

## Contact

Email: raydaniels@gmail.com
