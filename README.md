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

Pre-alpha. Scaffolding only. Core dependencies wired, build system
produces a trivial banner binary, no inference implemented yet.

Roadmap:

- [ ] GGUF loader wrapper (thin shim over ggml)
- [ ] Qwen/Llama model arch — one family to start
- [ ] Tokenizer (wrap ggml BPE)
- [ ] Attention kernel (Q·K^T, softmax, ·V, causal) reading compressed KV directly
- [ ] RoPE application (ggml_rope_ext + optional sidecar factors)
- [ ] Compressed-by-default KV management (VHT2 + sqfree + spinor)
- [ ] Sampling (temperature / top-k / top-p / min-p)
- [ ] CLI: `sp-engine perplexity`, `sp-engine run`
- [ ] Release packaging (GitHub Actions → Windows / Linux / Mac / Android)

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

## License

AGPLv3 for open-source / academic / non-proprietary use. Commercial
license available — contact raydaniels@gmail.com.

Third-party components (ggml) retain their original MIT license; see
[LICENSE.third_party](LICENSE.third_party).

## Repo layout

```
shannon-prime-engine/
├── lib/
│   └── shannon-prime/       ← git submodule → github.com/nihilistau/shannon-prime
├── vendor/
│   └── ggml/                ← git submodule → github.com/ggml-org/ggml
├── src/                     ← engine code (this repo's original contribution)
│   ├── engine.cpp
│   └── cli/
│       └── main.cpp
├── tests/
└── CMakeLists.txt
```
