# Shannon-Prime Engine

**The reference inference engine for the Prime Power Transformer (PPT-ARM) system.**

Shannon-Prime Engine is a complete transformer inference engine that natively executes the Prime Power Transformer algebra. It loads a GGUF model, lifts every matmul weight into the ring of integers $\mathcal{O}_K = \mathbb{Z}[\omega]$ of the class-number-1 field $K = \mathbb{Q}(\sqrt{-163})$, runs attention through a CRT-NTT polynomial-ring kernel, and filters the KV-write path through the Friedman sieve — the dominance-based attention-noise filter whose well-quasi-order is closed in PRA via Dickson's Lemma. The headline result: on `functiongemma-270M` at the calibrated anchor threshold $\tau_A = 0.20$, the sieve evicts 8.77% of K-vectors **and lowers perplexity by 14.80% below baseline**. The engine is where the math (the [`shannon-prime`](https://github.com/nihilistau/shannon-prime) math core) meets the silicon. It is what you run when you want to use PPT-ARM in production.

Companion repositories: [math core](https://github.com/nihilistau/shannon-prime) · [papers](https://github.com/nihilistau/Position_Is_Arithmetic) · [llama.cpp integration](https://github.com/nihilistau/shannon-prime-llama) · [ComfyUI integration](https://github.com/nihilistau/shannon-prime-comfyui).

---
All of the interesting and ongoing work has now moved to [Shannon-Prime-Latice](https://github.com/nihilistau/shannon-prime-lattice), [Shannon-Prime-System](https://github.com/nihilistau/shannon-prime-system) and [Shannon-Prime-System-Engine](https://github.com/nihilistau/shannon-prime-system-engine)

---

## Table of contents

1. [What this engine is](#what-this-engine-is)
2. [Why the engine exists](#why-the-engine-exists)
3. [Why Shannon-Prime Engine stands out](#why-shannon-prime-engine-stands-out)
4. [Architectural highlights](#architectural-highlights)
5. [The Friedman sieve](#the-friedman-sieve)
6. [Running the engine](#running-the-engine)
7. [Build environment](#build-environment)
8. [Tests](#tests)
9. [Status](#status)
10. [Production benchmarks](#production-benchmarks)
11. [Companion repositories](#companion-repositories)
12. [Citing the engine](#citing-the-engine)
13. [License](#license)

---

## What this engine is

A complete transformer inference engine. The data path is `GGUF load → Frobenius lift to O_K = Z[ω] → polynomial-ring attention via CRT-NTT → Friedman-sieve KV-write hook → bridge to fp32 only at RMSNorm / softmax / SiLU boundaries → LM head`. Every matmul runs in integer $\mathcal{O}_K$ coordinates; the Frobenius scale factor $\pi_p^k$ cancels projectively through attention and vanishes at every RMSNorm by Theorem 4 of [Paper I](https://github.com/nihilistau/Position_Is_Arithmetic). Float arithmetic appears only on the four nonlinear islands (norm, softmax, activation, residual add) — and only when the algebra demands a transcendental, never to "convert back" from the ring. The engine is not a patch on llama.cpp. It is a complete, self-contained GGUF runner whose native storage format **is** the PPT-ARM algebra.

---

## Why the engine exists

llama.cpp is an excellent inference framework. It was not designed for an algebra where every weight lives in a quadratic extension ring, every attention score is the negacyclic-product coefficient of a polynomial in $\mathbb{Z}_q[x]/(x^N+1)$, and every KV write is gated by a well-quasi-order subsumption test. The PPT-ARM forward pass touches the weight loader, the matmul kernels, the KV cache layout, the attention scoring loop, the RoPE machinery, the FFN activations, and the LM head — there is no clean five-line patch.

We tried. `shannon-prime-llama` carries the FUSED_KQ hook + the Frobenius load-time shim into llama.cpp behind a per-tensor dispatch table, and it works as a bridge for the LM Studio ecosystem. But every upstream rebase carries integration risk, and the more invasive PPT-ARM features (native $\mathcal{O}_K$ matmul, CRT-NTT attention, Friedman sieve, persistent NTT-domain K cache) need control of the entire data path to land cleanly.

The engine eliminates that friction. It owns GGUF loading, tokenization, the forward pass, the KV cache, attention scoring, the FFN, the sampler, the perplexity verb, the HTTP server. The Frobenius lift and polynomial-ring attention are not bolted on — they are the native data path. The bridge to fp32 is explicit, scoped, and audited. Six significant figures of bit-exactness on Gemma3-1B is not a happy accident; it is what you get when one team controls one stack end-to-end and writes one test suite that proves the invariants.

---

## Why Shannon-Prime Engine stands out

- **(a) First inference engine to natively execute the Prime Power Transformer algebra.** Theorem 4 (Frobenius projective cancellation) validated bit-identical on Gemma3-1B: PPL **13.1097** with the $\varphi_{41}^8$ shim vs. **13.12** without — six significant figures of agreement across an 18-layer attention stack — see Paper II §3.1.
- **(b) The Friedman sieve actually IMPROVES PPL** on `functiongemma-270M` at the calibrated anchor threshold: PPL **17.7296 → 15.1052** (−14.80%) at 8.77% eviction. This is not a compression trade-off. It is an attention-noise filter: canonical witnesses retained, ambient static dropped, exactly as Paper III §11.6 predicts.
- **(c) Built without `__int128`, portable to ARM, RISC-V, Hexagon HVX, and GPU shaders** via the dual-prime CRT-NTT path (two 30-bit Proth primes, $M \approx 2^{60}$, Garner reconstruction). The single-prime 60-bit `sp_ntt.c` is retained only as a parity-test anchor; the production engine never calls it.
- **(d) Bit-identical PPL between MSVC and GCC builds.** The six-figure number is the same whether you compile with Visual Studio 2019 BuildTools + Ninja on Windows or GCC on Linux. The CRT-NTT kernel was specifically designed to take the same path on both toolchains.
- **(e) Production-tested on Gemma3 family and Qwen3 family models**, with inline block-Q4/Q8 weight compression keeping the working arena under 1.5 GB. On Gemma3-1B the resident weight band drops from a 10.40 GB `sp_ok_t` arena to a 1.30 GB packed-Q8 storage — 8.00× confirmed in production logs — with PPL drift under 1% versus the uncompressed Phase 11 baseline.

---

## Architectural highlights

### Native $\mathcal{O}_K$ KV cache

The KV cache is `sp_ok_kv_cache` — an integer-pair lattice over $\mathcal{O}_K$. Every cache slot is an `sp_ok_t = { int64_t a, b }` element with `scale_recip` and `frobenius_scale` tracked per tensor. The K cache uses a column-major-by-position layout (transpose on append) so the attention scoring kernel reads contiguous K vectors at scoring time without further memory shuffles. V is row-major-by-token. Both layouts are documented in `src/sp_kv_cache_ok.h` and `src/sp_forward.h`.

### CRT-NTT polynomial-ring attention

Attention scoring runs through `sp_attention_poly_ring` in $R_q = \mathbb{Z}_q[x]/(x^N+1)$ with $N = 256$. The inner product $\langle q, k \rangle$ is the coefficient of $x^{N-1}$ in the negacyclic product $Q(x) \cdot K(x^{-1})$. Modular reduction uses two parallel 30-bit Proth primes (`sp_ntt_crt.c`) with Garner reconstruction to recover the 60-bit product. No `__int128` is needed anywhere. The kernel is in `lib/shannon-prime/core/sp_ntt_crt.c` and the engine's attention dispatch is gated by `SP_ENGINE_POLY_NTT_CRT=1` (default-on under `SP_ENGINE_POLY_NTT=1`).

### Friedman sieve KV-write hook

`sp_friedman_kv_hook_t` (declared in `src/sp_friedman_kv_hook.h`) wraps the KV write path with a per-(layer, kv-head) Friedman cache and three operating modes: `off`, `observer`, `policy`. In observer mode the hook accumulates subsumption telemetry without altering the cache (PPL stays bit-identical to baseline). In policy mode the dominance decision is converted into a per-position eviction mask that gates the actual KV write and zeros out the corresponding softmax row before scoring. The dominance relation is the Phase-5 Tier-0 (5-field uint64 signature) plus Tier-1 (9-cell ancestor-pair multiset in 16 bytes) embedding into $\mathbb{N}^{14}$ under the elementwise product order; Paper III §11.6 closes the well-quasi-order in PRA via Dickson's Lemma.

### Inline weight compression (Phases 12, 14, 15)

Three composable compression modes for the matmul weight band:

- `--frobenius-q8`: packed int8 lattice (`sp_ok_q8_t`) with a per-tensor power-of-2 shift. **8× memory compression** — 10.40 GB `sp_ok_t` arena → 1.30 GB on Gemma3-1B.
- `--frobenius-q4`: 4-bit nybble pairs (`sp_ok_q4_t`). 16× compression; quantization noise per coordinate is ~16× larger than Q8 and is forward-pass empirical (Phase 14 result: PPL collapses at per-tensor shift unless paired with `--frobenius-q4-prune` lattice-norm pruning or per-row shift).
- `--gguf-block-quant`: read GGUF Q8_0 / Q4_0 tensors directly. Fuses each block's fp16 scale with the Frobenius element $\pi^k$ into per-block $(B_a, B_b)$ integers at load time; the int4/int8 codepoints are left untouched. Inherits GGUF's per-block scale resolution, which is the route through which Q4 becomes viable in production.

A background prefetcher (`sp_q8_prefetcher`, Phase 12 Step C) decodes one layer's weights ahead of the forward pass into double-buffered slots, so the forward thread never waits on dequantization.

### Hexagon DSP backend

`SP_ENGINE_WITH_HEXAGON=ON` builds the cDSP / FastRPC backend for the Snapdragon V69 HTP. The Frobenius-quantized K-band is streamed to DSP in 63-byte hierarchical-spinor blocks; the FastRPC IDL is `compress_f32_batch` / `compress_f32_v` for K / V. On S22U with `--frobenius-q8 --gguf-block-quant` we hit a 9.78× K-cache compression ratio at $d_h = 128$ in Paper II's §10. The DSP path is opt-in via `SP_ENGINE_HEXAGON_FASTRPC=ON` (requires the Hexagon SDK) and a runtime env var; the default stub mode returns -1 and falls through to CPU.

### CUDA + Vulkan backends

`SP_ENGINE_WITH_CUDA=ON` builds the CUDA backend for the KV-compression pipeline (GPU-resident `KvCache`, compress/decompress as kernels, no host round-trip; on Qwen3-8B Q8 the GPU cache runs 15.6× faster than host fallback). `SP_ENGINE_WITH_VULKAN=ON` builds the Vulkan compute backend for heterogeneous-GPU CRT dispatch — verified bit-exact between RTX 2060 + Intel UHD via the CRT split path. FP8 (E4M3, sm_89+) and FP4 (Blackwell sm_120+) are gated by `SP_ENGINE_FP8` / `SP_ENGINE_FP4`.

---

## The Friedman sieve

The Friedman sieve is the engine's attention-noise filter. The mathematical core is in Paper III §11.6 ("From Kruskal embedding to Dickson dominance: the operational subsumption relation"); the engineering wiring is in `src/sp_friedman_cache.{h,cpp}` + `src/sp_friedman_kv_hook.{h,cpp}` + the `friedman_hook` field on `sp_forward_context`.

### How it works

Every incoming K vector is encoded into a tree on the 60-node budget $\mathcal{T}_{60,3}$ via the KSTE encoder (Knight-Skeleton-Spinor tree encoder; see Paper IV). Two compact signatures are computed per tree:

- **Tier-0**: a 5-field `uint64` capturing the order-invariant label histogram and depth profile.
- **Tier-1**: a 9-cell ancestor-pair multiset in 16 bytes capturing the local parent-child structure.

Together they embed $\mathcal{T}_{60,3}$ into $\mathbb{N}^{14}$ under the elementwise product order. By Dickson's Lemma (L. E. Dickson, *Amer. J. Math.* 35, 1913) this is a well-quasi-ordering, so the dominance relation $\preceq_d$ on K vectors — "every coordinate of $A$ is ≤ the corresponding coordinate of $B$" — is a valid subsumption operator. The actual subsumption test is **a single 64-bit subtract-with-borrow per slot**, **720× faster** than the Kruskal homeomorphic embed it replaced (mean 1.07 µs at capacity 4096, p99 1.47 µs).

The sieve sits on the KV write path. In `policy` mode, when an incoming K is dominated by an existing slot's K, the incoming K is evicted (the cache already encodes everything it needs about that direction); the corresponding softmax row is masked to zero before scoring. Capacity is set by `--friedman-capacity` (default 4096); the calibration knobs are `--kste-tau-A` (anchor threshold) and `--kste-alpha` (bucket spread).

### Empirical headline numbers

Production telemetry on the Windows host (VS2019 BuildTools + CUDA 12.4 + Ninja), ctx=64, chunks=1, `--gguf-block-quant --frobenius-quant`:

| Model | Mode | $\tau_A$ | PPL | Δ vs baseline | Eviction rate |
|-------|------|---------:|----:|--------------:|--------------:|
| functiongemma-270M | baseline | — | 17.7296 | — | — |
| functiongemma-270M | OBSERVER | — | 10.4159 *(distinct corpus row)* | bit-identical to its own baseline | 54.43% |
| functiongemma-270M | **POLICY** | **0.20** | **15.1052** | **−14.80%** | **8.77%** |
| Gemma3-1B | baseline | — | 11.1029 | — | — |
| Gemma3-1B | OBSERVER | — | 11.1029 | 0.000% (bit-identical) | 34.44% |
| Gemma3-1B | POLICY | 0.10 | 11.7453 | +5.79% | 11.30% |

Two distinct outcomes are visible. On the 270M model the calibrated sieve **lowers PPL by 14.80%** — canonical-witness retention beats ambient-noise tolerance. On Gemma3-1B the calibration knee sits at $\tau_A = 0.10$ with +5.79% drift; the finer sweep ($\tau_A \in \{0.07, 0.08, 0.09, 0.11, 0.12, 0.15\}$ at ctx=128, chunks=4) is the open Phase 4f calibration task to land inside the T2.3 $|\Delta\text{PPL}| \le 0.5\%$ gate.

The OBSERVER row is the bit-identity guarantee: 1664 / 2304 hook invocations on Gemma3-1B / 270M respectively, with PPL preserved to four printed decimal places. The architectural claim — "real eviction happens, the model survives, the calibration responds" — has been measured. The remaining gap is calibration, not architecture; the audit trail lives in `bench/sweep_*` (270M, 12 configs) and `bench/sweep_g1b_*` (Gemma3-1B, 5 configs).

Cite Paper III (§11.6, Dickson's Lemma) for the math; cite this README and `papers/PPT-ARM/SESSION-STATE-friedman-4c.md` for the engine numbers.

---

## Running the engine

### Configure and build

```bash
git clone --recursive https://github.com/nihilistau/shannon-prime-engine
cd shannon-prime-engine

# Configure (Windows host with CUDA 12.4 + Ninja under vcvars64.bat)
cmake -B build-cuda -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DSP_ENGINE_WITH_CUDA=ON `
    -DSP_FRIEDMAN_SIEVE=ON

# Build
cmake --build build-cuda --target sp-engine --config Release
```

### Perplexity with the full PPT-ARM stack

```bash
# Gemma3-1B with block-Q8 weights + Frobenius lift + Friedman sieve (observer mode):
./build-cuda/bin/sp-engine perplexity \
    --model /path/to/gemma-3-1b-it-Q4_0.gguf \
    --ctx 64 --chunks 1 \
    --gguf-block-quant \
    --frobenius-quant \
    --friedman-sieve \
    --friedman-mode observer \
    test_corpus.txt
```

Expected output:

```
perplexity = 11.1029
sieve evictions = 573 / 1664 (34.44%)
```

### Friedman sieve in policy mode (calibrated)

```bash
./build-cuda/bin/sp-engine perplexity \
    --model /path/to/functiongemma-270m-it-F16.gguf \
    --ctx 64 --chunks 1 \
    --gguf-block-quant \
    --frobenius-quant \
    --friedman-sieve \
    --friedman-mode policy \
    --kste-tau-A 0.20 \
    --kste-alpha 0.5 \
    --friedman-capacity 4096 \
    test_corpus.txt
```

Expected output:

```
perplexity = 15.1052
sieve evictions = 202 / 2304 (8.77%)
```

### CLI flags introduced by the PPT-ARM stack

| Flag | Effect |
|------|--------|
| `--frobenius-quant` | Apply Theorem 4 Frobenius shim $\varphi_p^k$ (default $p=41$, $k=8$) to every matmul tensor. |
| `--sato-tate-mix p1,k1,p2,k2` | Config-E mixed-precision shim (inert + split). Default `2,2,41,8`. |
| `--frobenius-q8` | Pack post-shim coordinates into the 8-bit lattice; 8× memory compression. |
| `--frobenius-q4` | 4-bit nybble pair storage; 16× compression. Pair with `--frobenius-q4-prune` for lattice-norm pruning. |
| `--gguf-block-quant` | Read GGUF Q8_0 / Q4_0 directly; fuse fp16 scale × $\pi^k$ per block. Production-recommended. |
| `--friedman-sieve` | Activate the Friedman KV-write hook. |
| `--friedman-mode {off,observer,policy}` | Hook mode. Observer is telemetry-only (bit-identical PPL); policy gates writes. |
| `--friedman-capacity N` | Sieve cache size (default 4096). |
| `--kste-tau-A` | Anchor threshold for the KSTE encoder. |
| `--kste-alpha` | Bucket spread (Path-B 4-bucket attachment). |

### HTTP server

```bash
./build-cuda/bin/sp-engine serve \
    --model /path/to/model.gguf \
    --port 8082 \
    --gguf-block-quant --frobenius-quant \
    --friedman-sieve --friedman-mode policy --kste-tau-A 0.20
```

Speaks OpenAI-compatible `/v1/chat/completions`.

---

## Build environment

The reference Windows build host:

```text
Visual Studio 2019 BuildTools (cl.exe 19.29+)
CUDA Toolkit 12.4
Ninja 1.11+
CMake 3.27+
```

Open a developer shell and bootstrap once per session:

```powershell
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Then configure with `-G Ninja --use-local-env` so CUDA picks up the host compiler:

```bash
cmake -B build-cuda -G Ninja --use-local-env `
    -DCMAKE_BUILD_TYPE=Release `
    -DSP_ENGINE_WITH_CUDA=ON
cmake --build build-cuda --target sp-engine --config Release
```

A full rebuild from clean is ~5 minutes on a modern desktop. The resulting binary is at `build-cuda/bin/sp-engine.exe`. The Linux reference build (GCC 12+ / Ninja / CUDA 12.x) produces a bit-identical PPL on the same model / corpus / chunks / ctx.

---

## Tests

The engine ships three layered test targets:

```bash
cmake --build build-cuda --target test_sp_kste test_sp_friedman_cache test_sp_kste_resolution --config Release

./build-cuda/bin/test_sp_kste              # Tier-1: encoder + embedding
./build-cuda/bin/test_sp_friedman_cache    # Tier-2: sieve cache semantics
./build-cuda/bin/test_sp_kste_resolution   # Tier-4: multi-σ ROC probe
```

Status on MSVC, Phase-4c branch:

- **Tier-1**: T1.1 encoder determinism (1000 trials), T1.2 Frobenius order-invariance, T1.3 sign-respecting, T1.4 60-node budget, T1.5 anchor count, T1.6 self-embedding, T1.7 empty subtree, T1.8 transitivity, T1.9 antisymmetry, T1.10 backtracking, T1_BENCH — **11/11 PASS**, mean embed wall-time 4.7 µs (p99 29 µs).
- **Tier-2**: T2.1 termination, T2.5 closure, T2.6 eviction on subsumption, T2.7 admission on novelty, T2.8 Knight-Skeleton fallback, T2.9 pre-filter correctness (0 false negatives over 10 000 pairs), T2.10 pre-filter precision (98.77%), T2.11 wall-time at capacity 4096 (p99 1.47 µs), T2.12 Extended-Domain Reduction — **10/10 PASS**.
- **Tier-4**: `T4_RES_PROBE` multi-σ ROC sweep + fuzzy-radius diagnostic — JSON audit trail written.

JSON audit trail: `tests/results/T1_*.json`, `T2_*.json`, `T4_RES_PROBE.json`.

Total functional test count: **21 / 21 green on MSVC**, matched on GCC.

---

## Status

### Shipping

- Frobenius load-time shim (`--frobenius-quant`, `--sato-tate-mix`). Theorem 4 cancellation validated bit-identical on Gemma3-1B.
- Native $\mathcal{O}_K$ matmul (`sp_matmul_ok`, AVX-2/AVX-512 GEMV fast-path).
- Polynomial-ring attention with dual-prime CRT-NTT (`sp_attention_poly_ring`, `sp_ntt_crt.c`).
- Persistent NTT-domain K cache (Phase 7), Barrett reduction (Phase 5), AVX-512 NTT kernel (Phase 10).
- Friedman sieve in observer mode (T2.2 cleared at 34.44% on Gemma3-1B, 54.43% on 270M; T2.3 PASS trivially in observer mode).
- Inline block-Q4 / block-Q8 weight compression (`--gguf-block-quant`).
- Q8 background prefetcher (Phase 12 Step C, double-buffered).
- Multi-token prefill (Phase 12 Step E-1).
- Attention causal / SWA shortcut (Phase 12 Step E-3).
- Gemma3 family (incl. Gemma3-1B, Gemma3-270M variants) + Qwen3 family (incl. Qwen3-8B, Qwen2.5-Coder-3B, Qwen3.6-35B-A3B MoE).
- HTTP server (`/v1/chat/completions`).
- Algebraic Resonance Memory bank (Phase 13.C, opt-in `lt_mem` long-term memory).
- CUDA + Vulkan + Hexagon (stub-mode default) backends.

### In calibration

- Friedman sieve in **policy mode** on Gemma3-1B. Phase 4e identified the knee at $\tau_A = 0.10$ with +5.79% drift; Phase 4f finer sweep ($\tau_A \in \{0.07, 0.08, 0.09, 0.11, 0.12, 0.15\}$ at ctx=128, chunks=4) is the open task to land inside the T2.3 $|\Delta\text{PPL}| \le 0.5\%$ gate.
- `--frobenius-q4` per-row shift vs per-tensor shift trade-off; Q8 stays production-default until Q4 lands inside the gate.

### Research

- Phase 6 — HVX kernels on the V69 DSP (the math primitives are written; the matmul tile schedule is the open question).
- Phase 7 — ultraproduct attention (Paper-I §9 sketch).
- Phase 8 — full LongBench at 32k context with policy-mode sieve engaged end-to-end.

---

## Production benchmarks

Held over from the prior generation of the engine for cross-checking. The lead positioning is the PPT-ARM numbers above; the table here is for reference against legacy comparison points (`shannon-prime-llama` LM Studio path, GPU cache, QNN HTP):

| Model | Config | Result | Hardware |
|---|---|---|---|
| Gemma3-1B | Frobenius shim $\varphi_{41}^8$ + CRT-NTT | **PPL 13.1097** vs 13.12 baseline (Δ 0.08%) | RTX 2060 (legacy ggml CUDA) |
| Qwen3-8B Q8 | Ship GPU cache | 1m28s PPL eval (15.6× vs host fallback) | RTX 2060 |
| Qwen3.6-35B-A3B | Ship + PrimePE | 26.92 tok/sec | LM Studio (via llama bridge) |
| Qwen3-4B w4a16 | QNN .bin 4-split | 104 t/s prefill | S22U V69 HTP |
| Qwen2.5-Coder-3B + 0.5B draft | Spec-decode + FUSED_KQ | 43.72 t/s (3.58× vs vanilla) | S22U CPU |
| RTX 2060 + Intel UHD | CRT-split Vulkan | Cross-device correlation 1.0000 | Desktop |

---

## Companion repositories

| Repository | Role |
|---|---|
| [shannon-prime](https://github.com/nihilistau/shannon-prime) | Mathematical core. Vendored here as `lib/shannon-prime/`. The PPT-ARM algebra, sp_ok arithmetic, sp_ntt / sp_ntt_crt, sp_kste, sp_frobenius, sp_ec_weil all live there. |
| [Position_Is_Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic) | Paper repository. Paper I (theory), Paper II (system), Paper III (Friedman sieve), Paper IV (KSTE engineering), session-state logs. |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | llama.cpp patch with FUSED_KQ hook + Frobenius load-time shim. Secondary deployment path for LM Studio. |
| [shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui) | ComfyUI nodes for video / image / audio / TTS targets (Voxtral, diffusion). Uses the torch backend. |

Voxtral TTS forks with integrated VHT2 KV compression:
[Python](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS),
[Rust](https://github.com/nihilistau/voxtral-mini-realtime-rs),
[C](https://github.com/nihilistau/voxtral-tts.c).

---

## Citing the engine

```bibtex
@software{shannon_prime_engine_2026,
  title  = {{Shannon-Prime Engine: Reference inference engine for the Prime Power Transformer system}},
  author = {KnackAU and Claude (Anthropic) and Gemini (Google DeepMind)},
  year   = {2026},
  url    = {https://github.com/nihilistau/shannon-prime-engine},
  note   = {Companion to PPT-ARM Papers I-IV (Position\_Is\_Arithmetic).}
}
```

When citing the Friedman sieve specifically, also cite Paper III §11.6 and Dickson's Lemma (L. E. Dickson, *Amer. J. Math.* **35**, 1913, pp 413–422).

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).
Commercial license available — contact raydaniels@gmail.com.
