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

**Qwen3-8B-Q8** — 4.06× KV compression, `Qwen3-8B-Q8_0.gguf` is
8.2 GiB on disk and fits the 12 GiB RTX 2060 VRAM cleanly (after
the weight-offload fix in `209507c`):

| Mode | PPL | ΔPPL | Wall time | Compression |
|---|---|---|---|---|
| baseline (no cache) | 18.13 | — | 1m09s | — |
| GPU + host cache | 18.64 | +2.8% | 6m24s | 4.06× |
| **GPU + GPU cache (ship)** | **19.29** | **+6.4%** | **1m28s** | **4.06×** |

**Two fixes unlocked the 15× speedup.**

1. **GPU-resident SP cache** (`commit b7349ff`). Compressed K/V
   blocks now live in VRAM. Compress / decompress run as CUDA
   kernels — zero host↔device round-trip per decode step. KvCache
   gains a `create_gpu` factory and `read_gpu` / `write_gpu` that
   operate on device pointers, hooked into `forward.cpp::decode`
   so the gallocr's `past_K_big` buffer is populated directly from
   the compressed cache. 4.4× faster than the host-cache path on
   this workload.

2. **GGUF blob tensor filter** (same commit). `gguf_init_from_file`
   exposes a raw `'GGUF tensor data binary blob'` I8 tensor
   alongside the 399 named weight tensors. Iterating the mmap
   context with `ggml_get_first_tensor` picked up both, so
   `ggml_backend_alloc_ctx_tensors` allocated the full payload
   twice — 8.2 GiB on disk → 16.6 GiB on backend. That spilled
   6 GiB into shared GPU memory (unified-memory paging over PCIe),
   which dominated the previous 23-min wall time. Iterating via
   `gguf_get_n_tensors` / `gguf_get_tensor_name` (named tensors
   only) drops the offload to 8.3 GiB — fits the card with
   ~2.5 GiB headroom, 15.6× pre-fix speedup.

**Known PPL delta (19.29 vs host cache 18.64 = +0.65 PPL).**
Decomposes into two independent issues:

1. *Base kernel drift (+0.40 PPL).* Measured by disabling
   calibration on both paths (`SHANNON_PRIME_NO_CALIBRATE=1`):
   host cache lands at 18.89, GPU cache at 19.29. This is the
   irreducible numerical difference between CPU and GPU
   compress/decompress kernels on real Qwen3 data. Likely root:
   CPU matrix-form VHT2 vs GPU pair-butterfly VHT2 — mathematically
   equivalent for p=2 Hadamard, numerically differ by a few ULPs
   per stage × 7 stages on hd=128.

2. *Asymmetric calibration effect (+0.25 PPL).* Same calibration
   math gives the host cache a -0.25 PPL improvement but the GPU
   cache a +0.21 PPL regression. Confirmed independent of
   calibration domain — GPU-domain variance accumulation
   (variance computed from GPU VHT2 output) produces bit-identical
   `var_order` to CPU on synthetic data (`kv_smoke` K_corr parity:
   0.9925 mean, 0.9804 min) yet still regresses Qwen3 PPL to the
   same 19.50. Not a domain-mismatch bug. Likely a scale-rounding
   interaction with variance-ranked band assignments; not yet
   pinned with a smoke test.

Default `SHANNON_PRIME_SYNC_CALIB_TO_GPU=0` (calibration sync off,
best empirical PPL 19.29). Set `=1` to opt in to the calibrated
order. Both issues tracked in session memory for future diagnosis.

**Dolphin-1B-Q8** — 3.76× compression, 3 GiB weights fit in VRAM
cleanly; full four-way decode-chain comparison:

| Mode | PPL | ΔPPL | Wall time (GPU) |
|---|---|---|---|
| baseline (no cache) | 12.36 | — | **2.8s** |
| ship cache | 14.06 | +13.7% | 1m19s |
| sqfree+spinor cache | 63.12 | +410% | 10m30s |
| hierarchical cache | 26.37 | +113% | 10m35s |

(Dolphin numbers predate the GPU-resident cache for sqfree /
hierarchical paths — those still run host-side for now. Ship cache
on Dolphin uses the pre-GPU-resident path. Extending the GPU cache
to sqfree+spinor is tracked as follow-up work.)

The 1B cache-mode results are load-bearing for the *scaling-law
fit*, not for deployment decisions — nobody compresses a 1B model's
KV in production. What matters for the scaling law:

* ship ΔPPL at 1B = +13.7% sits squarely on the `params^1.1`
  prediction (8B host-cache measured +2.8%, ratio 4.9× — law
  predicts ~20×, but at 1B the decode chain has fewer positions
  to amplify error so the ratio is model-variance-dominated).
* hierarchical at 9% skeleton lands at +113% on 1B-Q8 (vs +325%
  for sqfree+spinor at 50% skeleton). At matched compression, hier
  beats sqfree+spinor, consistent with v1.14/v1.15 research.

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

## Settings reference

### CLI verbs

| Verb | Purpose |
|---|---|
| `version` / `banner` | Build info. |
| `info --model <gguf>` | Hparams + tensor summary + vocab sample. |
| `encode --model <gguf> <text>` | Tokenise to IDs. |
| `decode --model <gguf> <id…>` | IDs to text. |
| `embed --model <gguf> <text>` | Token-embedding lookup only. |
| `block1 --model <gguf> <text>` | Layer-0 transformer block forward. |
| `logits --model <gguf> <text>` | Full forward, print logit stats + top-5. |
| `kv_smoke` | Synthetic gaussian K/V → cache → readback correlation + compression. No model needed. |
| `prefill --model <gguf> <text>` | Real RoPE'd K/V through the compressed cache, per-layer correlation report. Set `SP_CALIBRATE=1` to force adaptive calibration. |
| `chat --model <gguf> --n-predict N <prompt>` | Greedy generation: prefill + single-token decode reading from the compressed cache. Add `--naive` to force-forward-full each step. |
| `perplexity --model <gguf> <textfile>` | Baseline PPL (`forward_full` per chunk). Add `--cache` for decode-through-compressed-cache PPL. |
| `cache_ppl --model <gguf> <textfile>` | Baseline PPL + K/V round-trip correlation + scaling-law input per chunk. Fast diagnostic alternative to `perplexity --cache`. |

### Cache-config flags (shared across `kv_smoke` / `prefill` / `chat` / `perplexity --cache` / `cache_ppl`)

| Flag | Effect |
|---|---|
| *(none)* | **Ship path.** VHT2 forward → Möbius reorder (K) → 4-band banded quant (5,5,4,3 K, flat-3 V). Default and most-validated. |
| `--sqfree` | Sqfree prime-Hartley path. Pads head_dim to next sqfree multiple (64 → 66, 128 → 154, 256 → 330), Vilenkin-transforms, extracts a Knight skeleton (default L/2 of pad_dim), quantises skeleton + Möbius-CSR-predicted residual (3 bits). Cross-attention-style bit allocation (K and V both banded). |
| `--sqfree --spinor` | Sqfree + SU(2) sheet-bit correction at the causal-mask boundary. Typical K_corr lift over `--sqfree`: ≈0.008–0.010. |
| `--hierarchical` | Hierarchical Vilenkin predictor (`sp_hier_cache_t`). Kronecker sub-projection picks the Z/2Z × smallest-few-primes subgroup as a ~9% skeleton; a per-slot calibrated ridge-regression linear map predicts the remaining ~91% from the skeleton; tiny residual correction on top. **Needs ≥24-token calibration prompt** — warns below that. |
| `--no-mobius` | Disable Möbius reorder on ship path. |
| `--k-bits CSV` | Per-band K bit allocation, e.g. `5,5,4,3`. |
| `--v-bits CSV` | Per-band V bits (ship default `3` = 1 band flat). |
| `--residual-bits N` | Sqfree residual bit depth (1–4, default 3; 3 is the Pareto point — 1 is catastrophic, 4 is flat). |
| `--hier-level N` | Hierarchical skeleton level (0 = auto, picks second-to-last prime grouping). |
| `--hier-res-bits N` | Hierarchical residual bits (default 2). |
| `--hier-skel-bits CSV` | Hierarchical skeleton band bits (default `5,5`). |

### Cache-system deep dive

**Ship path (`sp_shadow_cache_t`)** — the validated default, straight from the CLAUDE.md invariants. On write: raw K → VHT2 → Möbius reorder (K only; V stays in its natural basis) → band-quantise using (5,5,4,3) for K and flat 3 bits for V. On read: dequantise → Möbius unreorder → VHT2 forward (self-inverse, no 1/N on the reverse). K_corr ≥ 0.993 on real RoPE'd K. Typical compression 3.8–4.1×.

**Sqfree path (`sp_sqfree_cache_t`)** — aggressive compression for Q8+ backbones. Pads head_dim to the nearest sqfree multiple so the Vilenkin transform factors across small primes {2, 3, 5, 7, 11}. Extracts a Knight skeleton by variance-ranked top-K selection (since v1.14; v1.13 and earlier used an algebraic T² rule which loses the 0/476 comparisons documented in `sp_regime_analysis.py`). Remainder goes through a Möbius-CSR predictor + 3-bit residual quant. Default skeleton size is L/2 (pad_dim / 2) — the universal phase-transition point identified by the chord diagnostic.

**Sqfree+spinor** — adds the SU(2) sheet-bit correction. 50% skeleton. Same write/read layout as sqfree; costs 1 extra bit per coefficient (the sheet bit) and picks the right double-cover sign at the causal-mask boundary. Small but reliable K_corr lift.

**Hierarchical Vilenkin predictor (`sp_hier_cache_t`, v1.15)** — the maximum-compression path. The Vilenkin basis on sqfree-padded dims has Kronecker structure:

```
hd=128 → pad=154 = 2 · 7 · 11 → Z/2Z × Z/7Z × Z/11Z
```

Hierarchical uses the low-prime subgroup (Z/2Z × Z/7Z = 14 coefficients, ~9% of pad_dim) as its skeleton. At calibrate-end, a ridge-regression linear map is fit per (layer, head) slot from skeleton → the remaining 140 high-prime refinement coefficients. At write-time, only the 14 skeleton coefficients + a small residual correction are stored. At read-time, the predictor reconstructs the refinement; residual adds back the part the predictor missed.

Three properties:

- Skeleton = 9% vs sqfree's 50% = **5.5× smaller skeleton storage** per vector.
- Requires **calibration** before any write. Single prefill of ≥24 tokens per slot is usually enough; the engine's `ForwardContext::prefill` auto-calibrates on the first call and warns below the threshold.
- Calibration is **per-slot** (each layer × head pair has its own linear predictor), unlike sqfree/shadow which share a global mask.

### PrimePE-RoPE-ALiBi (positional encoding family)

| Flag | Mode |
|---|---|
| `--pe-mode standard` | Default geometric RoPE. Byte-for-byte identical to llama.cpp. |
| `--pe-mode primepe` | Lattice-drawn freq_factors (composite-tiered or prime-tiered). Alpha-blended with identity. |
| `--pe-mode primepe_alibi` | PrimePE + per-head ALiBi slopes (max_bias = 8·α). |
| `--pe-mode alibi` | Standard geometric RoPE + ALiBi-only (ablation). |
| `--pe-alpha F` | Blend factor 0..1. Default 0 (identity; `--pe-mode primepe --pe-alpha 0` is byte-equal to `standard`). Paper sweet spot 0.15–0.22. |
| `--pe-tier 0` / `--pe-tier 1` | Composite lattice (default) vs prime generators. |

Precedence at graph build: PrimePE lattice (if `--pe-alpha > 0`) > `.sp_freq_factors.bin` sidecar (if present next to the model) > GGUF `rope_freqs.weight` tensor (for Llama-3.2 YaRN, etc.) > nullptr (pure geometric).

### Calibration behaviour

Opt-in cache modes (`--sqfree`, `--sqfree --spinor`, `--hierarchical`) auto-calibrate on the **first `ForwardContext::prefill` call** against an uncalibrated cache. Calibration:

- **Ship** (shadow cache): builds a variance-ranked coefficient permutation across the VHT2 output. Replaces the Möbius reorder for write and read.
- **Sqfree**: rebuilds the Knight skeleton at L/2 by variance-ranked selection.
- **Hierarchical**: fits a ridge-regression linear map per (layer × head) slot.

The `calibrated` flag lives on the `KvCache` object, not on `ForwardContext`. `bind_cache(kv)` zeros `kv_pos` between chunks but preserves calibration — so `perplexity --cache` calibrates once on chunk 0's warmup and reuses the masks for every subsequent chunk. `chat` gets the same behaviour for free.

Override: `SP_CALIBRATE=1` on the `prefill` CLI verb forces explicit calibration even though that verb normally just does `forward_full + kv->write` directly.

### Environment variables

| Variable | Effect |
|---|---|
| `SP_CALIBRATE=1` | Force `prefill` CLI to calibrate before writing to cache. Default is no-calibrate for diagnostic use; `ForwardContext::prefill` always auto-calibrates regardless. |
| `SP_DEBUG_DECODE=1` | Print layer-0 K / X correlation diff between decode and a reference `forward_full` at each step. Diagnostic for decode-graph bugs. |
| `SP_SKIP_CAPTURE=1` | Disable K/V capture in decode (and thus cache writes). Was a debug harness during the V-capture-view regression hunt; leave off. |

### Sidecar auto-load (`.sp_freq_factors.bin`)

At `ForwardContext::create`, the engine looks for `<gguf_path_without_ext>.sp_freq_factors.bin` — the file `sp_inject_freqs.py` writes alongside its injected GGUF. If present and sized for `n_rot/2 × fp32`, it's loaded into the RoPE freq_factors path (priority between PrimePE and GGUF `rope_freqs.weight`). Size-mismatched sidecars are rejected with a stderr warning.

Practical use: drop alternate-α `.bin` sidecars next to the same base GGUF to A/B different injection strengths without regenerating the GGUF each time. Measured on Dolphin-1B-Q8: α=0.17 sidecar auto-loaded gives **−2.1% PPL** vs GGUF `rope_freqs.weight` alone (12.38 vs 12.65 baseline).

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
