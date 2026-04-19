# Step 3 — GPU-resident sqfree+spinor cache

Design doc for extending `KvCache::create_gpu` to support the sqfree
(and sqfree+spinor) aggressive paths. Mirrors what ship-path got in
step 3 of the original plan (commit `b7349ff` + `209507c`).

## Why

Current state on Qwen3-8B-Q8:

* Ship GPU cache: 1m28s / PPL 19.29 (vs 23 min pre-fix; 15.6× faster)
* Sqfree on GPU backend (host cache fallback): **>33 min wall** —
  terminated early, never hit chunk 2. Per-token CPU compress/
  decompress is far more expensive than the ship shadow cache
  because it adds Knight-mask gather, Möbius predict, residual
  quantisation, and optional spinor sheet extraction on top of
  the VHT2 + band quant pipeline.

A proper GPU sqfree cache should land somewhere in the 2–5 min
range (same order of magnitude as ship GPU cache at 1m28s, with
a small multiplier for the extra per-vec work). That's a ≥10×
speedup over what exists today, plus the infrastructure to enable
the spinor variant.

## Session scoping

* **MVP (landed this session):** sqfree *without* spinor. Covers
  the main compress/decompress loop. Validates the design on real
  Qwen3 data, measures the actual speedup.
* **Deferred (next session):** spinor sheet bit. Adds the SU(2)
  double-cover correction that gives sqfree+spinor its K_corr
  0.988 at 2.8× compression on Q8+ backbones. Design is below
  under "Full scope"; implementation follows MVP.
* Both run on top of the existing kernels in
  `lib/shannon-prime/backends/cuda/shannon_prime_sqfree.cu`. Most
  of the compute primitives already exist; what's missing is the
  cache wrapper (state + orchestration) and a few gather/scatter
  helpers.

## Architecture

The CPU reference is `sp_sqfree_cache_t` in `shannon_prime.h:348`
with `sp_sqfree_write_k/v` / `sp_sqfree_read_k/v` in
`shannon_prime_sqfree.c:738+`. Mirror this on GPU:

```c
typedef struct {
    sp_config_t      config;
    int              pad_dim;           // head_dim padded to Vilenkin-friendly
    int              sk_k;              // skeleton size
    int              n_res;             // residual count (pad_dim - sk_k)
    int              residual_bits;     // 1..4 (3 default)
    int              use_spinor;        // 0 for MVP, 1 in full scope
    int              use_skel_mobius;   // 0 for MVP

    sp_band_config_t k_bands;           // operates on skeleton, not pad_dim
    sp_band_config_t v_bands;

    // Device copies of Knight mask (immutable after init/calibration).
    int             *d_skeleton_idx;    // (sk_k,)
    int             *d_residual_idx;    // (n_res,)
    int             *d_csr_offsets;     // (n_res + 1,)
    int             *d_csr_skel_slot;   // (n_terms,)
    int8_t          *d_csr_mu_sign;     // (n_terms,)
    int              n_terms;

    // Device copy of the Vilenkin basis factor list (for the staged
    // kernel in shannon_prime_sqfree.cu).
    int             *d_vilenkin_factors;
    int              n_factors;

    // Compressed GPU-resident storage.
    // Per-slot bytes_per_pos =
    //     k_bands.total_bytes   (skeleton bands)
    //   + 4                    (fp32 residual magnitude)
    //   + ceil(n_res * residual_bits / 8)    (packed residual levels)
    //   + (use_spinor ? ceil(n_res / 8) : 0) (sheet bits)
    uint8_t         *d_k_cache;         // [n_slots * max_seq * bytes_per_pos_K]
    uint8_t         *d_v_cache;         // [n_slots * max_seq * bytes_per_pos_V]
    int              bytes_per_pos_k;
    int              bytes_per_pos_v;
    int              max_seq_len;
    int              n_slots;

    // Device scratch (shared across calls, serialized via cache stream).
    float           *d_pad_scratch;     // (pad_dim,) — padded input
    float           *d_coeff_scratch;   // (pad_dim,) — post-Vilenkin
    float           *d_skel_scratch;    // (sk_k,) — gathered skeleton
    float           *d_pred_scratch;    // (n_res,) — predicted residual
    float           *d_dev_scratch;     // (n_res,) — deviation
    uint8_t         *d_levels_scratch;  // (n_res,) — quantised levels
    void            *stream;            // CUDA stream
} sp_cuda_sqfree_cache_t;
```

## Compressed format (bit-identical to CPU)

Per slot, per position:

```
+------------------------------------------+
| fp16 scale | packed skel bits (by band)  |  k_bands.total_bytes
+------------------------------------------+
| fp32 residual magnitude (mag)            |  4 bytes
+------------------------------------------+
| packed residual levels (LSB-first)       |  ceil(n_res * r_bits / 8)
+------------------------------------------+
| sheet bits (spinor only)                 |  ceil(n_res / 8)
+------------------------------------------+
```

Total = `k_bands.total_bytes + 4 + ceil(n_res * r_bits / 8) +
(use_spinor ? ceil(n_res / 8) : 0)`.

## Write pipeline (per vec)

```
d_k_vec (device, head_dim)
    │
    ▼ [kernel_sqfree_pad]      (exists: shannon_prime_sqfree.cu)
d_pad_scratch (pad_dim)
    │
    ▼ [sp_cuda_vilenkin_inplace]
d_coeff_scratch (pad_dim)
    │
    ▼ [kernel_gather_by_index] (NEW — trivial kernel)
d_skel_scratch (sk_k)
    │
    ▼ [sp_cuda_band_quantize]  (exists)
write slot:[0 .. k_bands.total_bytes)
    │
    ▼ [kernel_mobius_predict]  (exists)
d_pred_scratch (n_res)
    │
    ▼ [kernel_gather_residual_deviation] (NEW — fuses gather +
    │                                     actual-vs-pred + optional spinor)
    │   inputs: d_coeff_scratch, d_residual_idx, d_pred_scratch
    │   outputs: d_dev_scratch, (spinor) d_sheet_bits
    ▼
d_dev_scratch (n_res)
    │
    ▼ [kernel_compute_mean_abs]  (NEW — reduction)
    │   → mag (fp32)
    │   write slot:[k_bands.total_bytes .. +4)
    ▼
d_dev_scratch + mag
    │
    ▼ [kernel_quantize_residual] (NEW)
    │   levels[i] = clamp(round((dev[i] / mag) * (L-1) / 2 + (L-1)/2), 0, L-1)
    ▼
d_levels_scratch (n_res)
    │
    ▼ [kernel_pack_levels]       (NEW — bit packing, LSB-first)
write slot:[offset_levels .. +res_bytes)

(spinor) write sheet bits → slot:[offset_sheet .. +sheet_bytes)
```

## Read pipeline (per vec)

```
read slot:[0 .. k_bands.total_bytes)
    │
    ▼ [sp_cuda_band_dequantize]  (exists)
d_skel_scratch (sk_k)
    │
    ▼ [kernel_scatter_by_index]  (NEW — trivial)
d_coeff_scratch (pad_dim, zero except at skeleton_idx)
    │
    │  in parallel:
    │
    ▼ [kernel_mobius_predict]    (exists)
d_pred_scratch (n_res)

read slot:[k_bands.total_bytes .. +4) → mag

read slot:[offset_levels .. +res_bytes)
    │
    ▼ [kernel_unpack_levels]     (NEW)
d_levels_scratch (n_res)
    │
    ▼ [kernel_dequantize_residual] (NEW)
    │   dev[i] = ((levels[i] - (L-1)/2) * 2 / (L-1)) * mag
    ▼
d_dev_scratch

(spinor) read sheet bits → d_sheet_bits
    │
    ▼ pred[i] = sheet[i] ? -pred[i] : pred[i]

d_coeff_scratch[residual_idx[i]] = pred[i] + dev[i]
    │
    ▼ [sp_cuda_vilenkin_inplace] (exists — self-inverse)
d_coeff_scratch (full pad_dim, now reconstructed)
    │
    ▼ [kernel_sqfree_unpad]      (exists)
d_k_out (device, head_dim)
```

## Calibration flow

Sqfree calibration is more involved than ship:
1. `calibrate_begin` allocates `calib_sum`/`calib_sum2` of length
   `pad_dim` (doubles).
2. `calibrate_feed` does: pad → VHT2 (= Vilenkin on pad_dim) →
   accumulate variance per-coefficient.
3. `calibrate_end` rebuilds the Knight mask with variance ranking
   at L/2, re-initialises K and V band quantisers, and re-initialises
   the skel-mobius mask if that flag is on.

For the GPU cache:
* Mirror step 3 (ship): `calibrate_feed` runs the pad + Vilenkin on
  GPU (`d_pad_scratch` + `sp_cuda_vilenkin_inplace`), downloads to
  host, accumulates variance in the shadow cache's `calib_sum` /
  `calib_sum2` buffers.
* `calibrate_end` needs a new GPU-specific path: rebuild the Knight
  mask on host (`sp_knight_mask_init` takes a variance array), then
  **re-upload all the mask arrays to the GPU** (skeleton_idx,
  residual_idx, csr_offsets, csr_skel_slot, csr_mu_sign,
  potentially bytes_per_pos if the allocation changed).
* If the skeleton size changes on calibration, the compressed
  storage layout changes too. For simplicity, pre-allocate at
  `sk_k = pad_dim / 2` (the L/2 target); calibration only re-ranks,
  not re-sizes. Static sk_k keeps the layout stable.
* Known PPL caveat: ship cache's calibration on GPU empirically
  regresses PPL (see `memory/gpu_cache_ppl_drift.md`). Keep the
  calibration sync behind the same env (`SHANNON_PRIME_SYNC_CALIB_
  TO_GPU=1`) and default to the static variance-first Knight mask
  for the sqfree GPU cache until the asymmetry is root-caused.

## MVP scope (this session)

Landing in `b7349ff+` commits:

1. New file `lib/shannon-prime/backends/cuda/shannon_prime_sqfree_cache.cu`
   (or extend `shannon_prime_sqfree.cu`) with:
   * `sp_cuda_sqfree_cache_init` / `_free`
   * `sp_cuda_sqfree_write_k` / `_v` (use_spinor=false path)
   * `sp_cuda_sqfree_read_k` / `_v` (use_spinor=false path)
   * Missing launchers: gather-by-index, scatter-by-index,
     gather-residual-deviation, quantize-residual, dequantize-
     residual, pack/unpack level bits, mean-abs reduction.
2. Corresponding declarations in `shannon_prime_cuda.h` (or a new
   sibling header).
3. CMake: already pulls both `.cu` files into
   `shannon_prime_cuda_backend`, nothing new needed.
4. `src/kv_cache.h`: no API change (same read_gpu / write_gpu
   surface; the create_gpu sqfree branch just uses a different
   internal cache pointer).
5. `src/kv_cache.cpp`:
   * `Impl` gains an `sp_cuda_sqfree_cache_t cuda_sqfree_cache`
     field + `bool cuda_sqfree_inited`.
   * `create_gpu` dispatches: ship → sp_cuda_cache, sqfree →
     sp_cuda_sqfree_cache.
   * `read_gpu` / `write_gpu` route to the right one via if-chain.
   * `calibrate_feed` for GPU sqfree cache = upload → sp_cuda_
     vilenkin_inplace → download → accumulate in shadow.
6. `src/forward.cpp`: no change — decode already calls `read_gpu`/
   `write_gpu`; transparent to backend type.
7. Smoke test + Qwen3-8B bench.

**Explicit non-goals for this MVP:**
* Spinor (full scope).
* Re-allocatable compressed storage on calibration-driven sk_k
  change.
* skel-mobius reorder on GPU (CPU path supports this; GPU stays
  on the vanilla Knight ordering).

## Full scope (next session)

After MVP lands cleanly, add on top:

1. Spinor sheet bit:
   * `kernel_gather_residual_deviation` gains an output branch that
     picks `v_plus = actual - pred` or `v_minus = actual + pred`
     and emits a sheet bit (1 per residual position).
   * Sheet bits packed and stored at the tail of each slot.
   * Read path: unpack sheet bits; flip pred sign when bit set.
   * Expected gain: matches CPU +0.01 ΔPPL at 2.8× on Q8+ 8B.
2. `use_skel_mobius` on GPU: inject a gather-by-order pass between
   skeleton extract and band_quantize (and inverse on read). The
   CPU path already shows this helps; GPU parity is a follow-up.
3. Verification vs. scaling law: measure K_corr on kv_smoke with
   calibrated Knight mask; expect 0.975–0.988 depending on
   residual_bits and spinor.
4. Consider lifting the +0.25 calibration asymmetry with sqfree's
   pad_dim=154 (non-power-of-2) since the Vilenkin transform uses
   the staged kernel, not the p=2 pair butterfly. The fp32
   accumulation order difference between CPU and GPU might
   disappear — would close the asymmetry for the sqfree path.

## Missing kernels (to write)

Tiny, one-liner kernels:

```cuda
// Gather: out[i] = in[idx[i]]
__global__ void kernel_gather(const float* in, const int* idx,
                              float* out, int n);

// Scatter: for i in [0, n): out[idx[i]] = in[i]
__global__ void kernel_scatter(const float* in, const int* idx,
                               float* out, int n);

// Fused: dev[i] = coeff[residual_idx[i]] - pred[i]   (MVP, no spinor)
__global__ void kernel_residual_deviation(const float* coeff,
                                          const int* residual_idx,
                                          const float* pred,
                                          float* dev, int n_res);

// Reduction: mag = mean(|dev[i]|)
__global__ void kernel_mean_abs(const float* dev, int n_res, float* mag);

// Residual (de)quantize: uses magnitude and residual_bits
__global__ void kernel_quantize_residual(const float* dev, int n_res,
                                         int bits, float mag,
                                         uint8_t* levels);
__global__ void kernel_dequantize_residual(const uint8_t* levels, int n_res,
                                           int bits, float mag, float* dev);

// Level packing (LSB-first, matches sp_sqfree_compress_one layout)
__global__ void kernel_pack_levels(const uint8_t* levels, int n_res,
                                   int bits, uint8_t* packed);
__global__ void kernel_unpack_levels(const uint8_t* packed, int n_res,
                                     int bits, uint8_t* levels);
```

## Testing strategy

1. `kv_smoke --sqfree` with GPU backend: K_corr ≥ 0.97 mean for
   `hd=128`, `residual_bits=3`. Match CPU sqfree `kv_smoke` within
   0.005.
2. `perplexity --cache --sqfree` on Qwen3-8B-Q8, ctx=512 chunks=2:
   wall time under 5 min (vs >33 min host-cache baseline).
3. PPL delta vs host sqfree cache: expected ≤ 0.5 PPL (same
   asymmetric-calib caveat as ship).
4. Coherent chat output on Qwen3 sqfree GPU.
5. No crash / OOM on max_seq=520 Qwen3 config.

## MVP status (commits `c4a25db`, `fca9fb2`, `95ff818`)

Landed:
* Full sqfree cache infrastructure: `sp_cuda_sqfree_cache_t` struct,
  `_init`/`_free`, `_write_k/v`, `_read_k/v`, all kernels
  (gather, scatter, residual deviation, mean_abs, quantize,
  dequantize, pack, unpack).
* `KvCache::create_gpu` sqfree branch + all 3 verb routing
  (perplexity, chat, kv_smoke).
* `KvCache::read`/`write` host-pointer shim routes to GPU-staging
  path for both ship and sqfree.
* **Numerical parity fix** (`95ff818`): `kernel_quantize_residual`
  and `kernel_dequantize_residual` now byte-identical to
  `sp_quantize_residual` / `sp_dequantize_residual`. Two bugs:
    - `sat = mag * bits` (was `sat = mag` alone — 3× off for
      residual_bits=3).
    - Plain round-half-up `(int)(q + 0.5f)` (was a centered-sign
      variant via `copysignf(0.5f, f - half)`).

Measured on `kv_smoke --sqfree hd=128 n_tokens=32 n_head_kv=4
n_layer=2 residual_bits=3`:

| Path              | K_corr mean | K_corr min | V_corr mean |
|-------------------|-------------|------------|-------------|
| CPU sqfree        | 0.9447      | 0.8867     | 0.9151      |
| GPU sqfree (pre)  | 0.8340      | 0.6594     | 0.8157      |
| GPU sqfree (post) | 0.9347      | 0.8291     | 0.9144      |

K_corr delta collapsed from −0.11 to −0.01 mean; V_corr essentially
matches. Remaining 0.01 is attributable to the calibration-sync gap
(open issue 2 below).

## Open issues for next session

1. **Kernel-launch-overhead bound on the real bench.** Qwen3-8B-Q8
   sqfree bench ran >28 min before being killed (vs ship GPU cache
   at 1m28s on identical config). The per-vec sqfree pipeline does
   ~9 kernel launches per slot; at past_n=256 × 36 layers × 8 heads
   = 73K slots per decode step → ~660K launches per step. At ~5 µs
   launch latency that's ~3 s/step of overhead alone — dwarfs the
   actual compute.

   Fix direction: **batched kernels**. Mirror the ship path's
   `sp_cuda_read_k_batch` — one kernel call per (layer, head) that
   processes all kv_len positions at once. Prefill writes can
   similarly batch across n_tokens.

   Target: Qwen3-8B sqfree in the 2–5 min range.

   **Status (2026-04-20):** batched-read infrastructure landed in
   shannon-prime `22639fb` + engine `c34740a`. Five new batched
   kernels (`scatter_batch`, `reconstruct_residual_batch`,
   `dequantize_residual_batch`, `unpack_levels_batch`,
   `gather_mag_batch`), scratch sized to `max_seq_len × per-vec`,
   public `sp_cuda_sqfree_read_k_batch` / `_v_batch` launchers.
   Subtlety: `sp_cuda_band_dequantize` assumes contiguous
   `total_bytes` stride per vec, but sqfree slot layout is
   `[total_bytes][4 mag][res_bytes]` — stride `bytes_per_pos`. A
   `cudaMemcpy2DAsync` repack into `d_pad_scratch` (reused as a byte
   buffer) fixes this. Verified K_corr 0.943 on kv_smoke random data
   (matches CPU). Real-model PPL on Qwen3-8B still regresses (+43
   PPL at ctx=512/chunks=1 vs CPU sqfree 11.82) — pending root cause.
   Per-vec fallback via `SHANNON_PRIME_SQFREE_NO_BATCH=1`.

2. **Calibration sync for the GPU Knight mask.** CPU companion
   `sp_sqfree_cache_t` rebuilds its mask with variance ranking on
   `calibrate_end`, but the GPU cache's `d_skeleton_idx` /
   `d_residual_idx` / `d_csr_*` stay on the initial squarefree-first
   ordering. Re-upload after `calibrate_end`.

   **Status (2026-04-20):** landed in engine `c34740a`. Handles
   `n_terms` changes via realloc, converts `int8_t csr_mu_sign` to
   int32 for the GPU kernel. Gated behind
   `SHANNON_PRIME_SYNC_CALIB_TO_GPU=1` — unconditional sync added
   +19 PPL on the same Qwen3-8B ctx=512 chunks=1 bench (74 vs 55
   without sync). Same asymmetric calibration effect the ship path
   already documented; tracked alongside the shared PPL drift work.

3. **Spinor sheet bit (full scope).**
   * `sp_cuda_sqfree_cache_init(use_spinor=1)` — allocate sheet bit
     storage per slot.
   * `kernel_residual_deviation_spinor` variant: emit sheet bit,
     use `min(|v_plus|, |v_minus|)`.
   * Read path reads sheet bits and flips pred sign accordingly.
   * Expected gain: matches CPU +0.01 PPL at 2.8× on Q8+ 8B.

## Files

| File | Change |
|---|---|
| `lib/shannon-prime/backends/cuda/shannon_prime_sqfree.cu` | new kernels + `sp_cuda_sqfree_cache_*` funcs |
| `lib/shannon-prime/backends/cuda/shannon_prime_cuda.h` | forward-declare `sp_cuda_sqfree_cache_t` + the new API |
| `src/kv_cache.h` | no API change, new internal branching |
| `src/kv_cache.cpp` | `Impl` gets cuda_sqfree_cache; create_gpu / read_gpu / write_gpu / calibrate_* branch |
| `src/forward.cpp` | no change |
| `tests/...` | new sqfree GPU fidelity test (nice-to-have) |
