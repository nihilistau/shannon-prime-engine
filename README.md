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
writes new K/V back. Cache modes: ship (default), sqfree, sqfree+
spinor, and hierarchical Vilenkin predictor (9% skeleton).

## Headline result

**Qwen3-8B-Q8 ship cache-mode PPL = 18.14 vs baseline 18.05 → +0.5% at 4.06× KV compression** (wiki.test.raw, ctx=512 chunks=2, decode-per-token through the compressed cache).

That's the number this engine was built to produce: a hook-free,
end-to-end measurement of compressed-cache PPL on the size of
model where KV cache actually matters. Below 1B the KV cache
isn't the bottleneck anyway — compression-on-small-models
benchmarks are noise, and the scaling law
(`log(PPL_ratio) ∝ (1−K_corr)² / params^1.1`) explains why: the
penalty scales nearly inversely with model size.

### Scaling-law projection at K_corr=0.993 (ship), Q8 weights

| Model size | Ship PPL hit @ 4× compression | Regime |
|---|---|---|
| 1B  | +10%   (Dolphin measured +10.5% ✓) | noise — don't compress |
| 8B  | **+0.5% (Qwen3 measured +0.5% ✓)** | sweet spot, ship-ready |
| 14B | +0.2% predicted | ship-ready |
| 70B | +0.01% predicted | compression essentially free |
| 405B | +0.002% predicted | the regime this tech was built for |

The `params^1.1` denominator in the scaling law roughly halves the
penalty per doubling of parameters. The two measured points
(Dolphin-1B and Qwen3-8B) bracket the curve and match the
prediction to within fit error (±20%).

## Cache options (ship is default)

| Flag | Path | Skeleton | Compression | When to use |
|---|---|---|---|---|
| *(none)* | **Ship** VHT2 + Möbius + banded quant | full | ~4× | default, validated, K_corr≥0.993 |
| `--sqfree` | Sqfree Knight skeleton + residual + predictor | 50% | ~4× | aggressive, Q8+ backbones |
| `--sqfree --spinor` | + SU(2) sheet bit | 50% | ~4× | slight K-corr lift over sqfree |
| `--hierarchical` | Kronecker sub-projection + calibrated linear predictor | **9%** | ~4× | maximum skeleton reduction (≥24-token calibration prompt required) |

Opt-in modes auto-calibrate on the first prefill; calibration
state persists across decode steps. Short prompts with
`--hierarchical` emit a stderr warning pointing at `--sqfree`
instead (the linear predictor is underdetermined below ~24
samples per slot).

## Engine stage table

| Stage | Verb | Status |
|---|---|---|
| 3a–3d | `embed` / `block1` / `logits` (+ PrimePE-RoPE-ALiBi) | ✓ |
| 4     | `kv_smoke` — KvCache wrapper | ✓ |
| 5a    | `prefill` — real RoPE'd K through KvCache, per-layer correlation | ✓ |
| 5b    | `chat` — stateful prefill + single-token decode | ✓ |
| 6a    | `perplexity` (baseline + `--cache` decode-chain) | ✓ |
| 6b    | `cache_ppl` — baseline PPL + K/V correlation + scaling-law term | ✓ |
| 6c    | Sidecar auto-load (`<model>.sp_freq_factors.bin`) | ✓ |
| 7     | CUDA / Vulkan backend selection; release packaging | planned |

## Measured numbers

### Cache-mode PPL (`perplexity --cache`, ctx=512 chunks=2, wiki.test.raw)

| Model | Mode | PPL | ΔPPL | Compression |
|---|---|---|---|---|
| Qwen3-8B-Q8 | baseline (no cache) | 18.05 | — | — |
| Qwen3-8B-Q8 | **ship** | **18.14** | **+0.5%** | **4.06×** |

(Dolphin-1B cache-mode numbers omitted — the 1B regime is not
representative of target deployments and the scaling law
predicts its behaviour from the K_corr alone.)

### cache_ppl roundtrip (baseline PPL + K/V correlation)

| Model | Mode | K_corr | V_corr | Skeleton | Scaling term |
|---|---|---|---|---|---|
| Qwen3-8B-Q8 | ship | 0.9930 | 0.9652 | full | 0.229 |
| Qwen3-8B-Q8 | hierarchical | 0.9833 | 0.9368 | 9% (14/154) | 1.314 |
| Dolphin-1B-Q8 | ship | 0.9947 | 0.9708 | full | 0.133 |
| Dolphin-1B-Q8 | hierarchical | 0.9752 | 0.9318 | 9% (6/66) | 2.895 |

Hierarchical at 9% skeleton is architecturally competitive with
ship's K-corr at 5.5× smaller skeleton. The scaling-law projection
at 70B Q8 puts hierarchical compression-PPL within 0.05% of ship.

### Sidecar injection (`<model>.sp_freq_factors.bin` auto-load)

| Model | Path | PPL |
|---|---|---|
| Dolphin-1B-Q8 | GGUF rope_freqs.weight (no sidecar) | 12.65 |
| Dolphin-1B-Q8 | α=0.17 sidecar auto-loaded | **12.38** (−2.1%) |

The engine now reads the `.sp_freq_factors.bin` file
`sp_inject_freqs.py` writes alongside its injected GGUF — useful
for A/B-ing different alphas against the same base model without
regenerating the GGUF each time.

### Greedy chat smoke tests (n_predict=20, prompt = "The quick brown fox")

| Config | Output |
|---|---|
| Qwen3-8B-Q8 ship | *"...This sentence is a well-known pangram. It is used to test"* |
| Dolphin-1B ship | *"...jumps over the lazy dog. The sentence \"The quick brown fox jumps over the lazy dog.\" is"* |
| Dolphin-1B sqfree+spinor | *"...What is the correct order of the sentence? To determine the correct order"* |

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
