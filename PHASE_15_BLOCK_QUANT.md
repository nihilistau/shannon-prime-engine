# Phase 15 — Block-quant ingest (steal the industry's calibration work)

Status: **design doc, no code written**. Drafted 2026-05-18 after the
Phase 14 Q4 per-tensor-shift blowout proved we need block-level scaling,
and the discovery that GGUF block-quants already give us exactly that
for free.

This phase doesn't add a new compression scheme. It changes where the
quant scales come from. Instead of `fp16 → encode → Frobenius → pack`,
we read pre-quantized GGUF blocks directly and fuse the block scale with
the Frobenius element π^k at load time. The int4/int8 codepoints inside
the GGUF stay untouched.

---

## The math (the part that took the conversation to nail down)

A GGUF block-quant tensor stores, per block of 32 elements:
- 32 scalar codepoints `w[k]` in int4 or int8 (the actual weight bits)
- one fp16 `block_scale` (and for Q4_K, a fp16 `block_min` too)

So the true continuous weight is `W[k] = w[k] · block_scale + block_min`.

In our ring O_K = Z[ω], the Frobenius shim multiplies each weight by
π^k. **π^k is not a scalar** — it is an element of O_K with two integer
components `(π_a, π_b)`, both non-zero for k > 0. Applying φ to a real
weight `W` (modeled as `(W, 0)` in O_K) produces:

```
φ(W) = W · π^k = (W·π_a, W·π_b)
```

So the lifted O_K coordinate of weight `k` in block `B` is:

```
W_ring[k] = ( w[k]·block_scale·π_a, w[k]·block_scale·π_b )
         = w[k] · (B_a, B_b)
```

where the **per-block lifted scales** are integers:

```
B_a = round(block_scale · π_a · scale_recip)
B_b = round(block_scale · π_b · scale_recip)
```

`scale_recip` is our existing per-tensor encoding scale (default 16384).
B_a and B_b are int64 — same dynamic range as our current post-Frobenius
coordinates (~2^38 for k=8), well inside int64.

**At decode time the codepoint stays scalar.** The (a, b) lift is
applied by the multiplication w[k] · B_a / w[k] · B_b, where w[k] is a
tiny 4- or 8-bit int and B_a, B_b are int64 constants per block.

### The factored matmul inner loop

For `Y[j, i] = Σ_k W_ring[i, k] * X[j, k]` (ring multiplication on
O_K coordinates), substituting `W_ring[i, k] = w[i,k] · (B_a, B_b)`:

```
acc_a = Σ_k w[i,k] · ( B_a · x.a[j,k] − 41 · B_b · x.b[j,k] )
acc_b = Σ_k w[i,k] · ( B_a · x.b[j,k] + B_b · x.a[j,k] + B_b · x.b[j,k] )
```

The bracketed expressions depend only on (B_a, B_b) and x[j,k]. Define
the per-`k` factors:

```
F_a[k] = B_a · x.a[j,k] − 41 · B_b · x.b[j,k]
F_b[k] = B_a · x.b[j,k] + B_b · x.a[j,k] + B_b · x.b[j,k]
```

Then the inner loop is:

```c
for k in block:
    F_a = B_a * x_a[k] - 41 * B_b * x_b[k]
    F_b = B_a * x_b[k] + B_b * x_a[k] + B_b * x_b[k]
    acc_a += w_int4[k] * F_a
    acc_b += w_int4[k] * F_b
```

This is structurally cheaper than the current Q8 kernel:
- 4 mults per k for the F factors (independent across k, high ILP)
- 2 mults per k for the w·F accumulation, where w is a 4-bit constant —
  on x86 this lowers to an `imul r64, r64, imm` form when w is small,
  or just `lea + add` when w fits in an addressing-mode immediate.
- vs. current Q8: 4 full int64×int64 mults per k for the (a, b) ring
  product. Factored kernel ≈ 6 mults but the two heavy ones are
  int64 × int4. Net ~40% fewer mul issue slots.

### What the existing scalar/SIMD Q8 path was doing wrong

The current `sp_ok_q8_t` stores `{int8_t a, int8_t b}` per element —
a quantized (a, b) pair. Per-tensor shift forces every element across
the whole tensor onto the same exponent. When one weight has |W| ~ 8.0
and most have |W| ~ 0.05, the picker chooses shift=27, and the small
weights round to 0. That's the Phase 14 Q4 blowout.

The block-quant scheme dodges this by construction: each 32-element
block carries its own `block_scale`, and the GGUF quantizer already
made sure the block's int4/int8 codepoints fit cleanly into the
codebook with low saturation. We don't pick the shift; the GGUF file
already did it for us, per block.

---

## Storage layout — my preferred design

**Layout decision: AoS at the block granularity, one block per cache
line.** The cost of mixing block metadata with packed weights is one
const-offset addition at decode; the win is a single DRAM line miss
per 32 elements processed.

### Q8 block (32 int8 codepoints):

```c
struct alignas(64) sp_ok_q8_block_t {
    int64_t B_a;            // 8 B  — fused π_a · block_scale · scale_recip
    int64_t B_b;            // 8 B  — fused π_b · block_scale · scale_recip
    int64_t reserved;       // 8 B  — block_min or zero-point if needed
    int8_t  reserved2[8];   // 8 B  — alignment pad / future metadata
    int8_t  packed[32];     // 32 B — the GGUF int8 codepoints, untouched
};
// total 64 bytes = exactly one cache line, 32 elements
// storage per element: 2.0 B (vs raw sp_ok_t 16.0 B → 8× compression)
```

### Q4 block (32 int4 codepoints):

```c
struct alignas(32) sp_ok_q4_block_t {
    int64_t B_a;            // 8 B
    int64_t B_b;            // 8 B
    uint8_t packed[16];     // 16 B — two int4 codepoints per byte
};
// total 32 bytes = half a cache line, 32 elements
// storage per element: 1.0 B (vs raw sp_ok_t → 16× compression)
```

Two Q4 blocks pack into one cache line at adjacent addresses — the HW
prefetcher walks them naturally.

### Tensor descriptor

```c
typedef struct {
    sp_ok_q8_block_t* blocks;   // numel / 32 blocks
    size_t            numel;    // total element count (multiple of 32)
    size_t            n_blocks; // numel / 32
    // No q8_shift here — the block's B_a/B_b already encode the scale.
    // No scale_recip here — also folded into B_a/B_b.
    // The frobenius p/k are still tracked for downstream forensics.
    int16_t           frobenius_p;
    int16_t           frobenius_k;
} sp_ok_block_q8_tensor;
```

The same pattern for `sp_ok_block_q4_tensor`.

---

## Where the intercept lives

Today the load path is:

```
GGUF mmap → ggml_get_rows (dequant to fp16) → fp16_to_fp32 → encode →
sp_ok_t arena → Frobenius shim → q8/q4 pack
```

The block-quant intercept short-circuits the middle:

```
GGUF mmap → detect block-quant type → read block_scale + int8s/int4s →
fuse (block_scale, π^k, scale_recip) → write sp_ok_block_q{4,8}_block →
DONE (no ggml dequant, no fp16, no sp_ok_t arena, no Frobenius shim)
```

### Intercept point in code

`src/sp_weights_loader.cpp:sp_weights_load_from_llama` is where we
currently iterate tensors. Per tensor:

```c
const ggml_tensor* t = layer.wq;
switch (t->type) {
case GGML_TYPE_F16:
    /* existing path: fp16 → encode → arena → Frobenius → q8 pack */
    break;
case GGML_TYPE_Q8_0:
    /* NEW: read blocks directly, fuse scales, write block_q8_tensor */
    sp_ok_block_q8_from_gguf_q8_0(out.block_q8_wq[L], t,
                                   cfg.frobenius_p, cfg.frobenius_k,
                                   scale_recip);
    break;
case GGML_TYPE_Q4_0:
case GGML_TYPE_Q4_K:
    /* NEW: same but for 4-bit blocks */
    sp_ok_block_q4_from_gguf_q4_x(out.block_q4_wq[L], t,
                                   cfg.frobenius_p, cfg.frobenius_k,
                                   scale_recip);
    break;
}
```

`sp_ok_block_q{4,8}_from_gguf_*` is a new function in
`lib/shannon-prime/core/sp_ok_block_quant.{c,h}`. It reads the
ggml block layout (defined in `ggml-common.h`):

```c
#define QK8_0 32
typedef struct {
    ggml_fp16_t d;             // block scale
    int8_t      qs[QK8_0];     // 32 int8 quants
} block_q8_0;

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;             // block scale (single)
    uint8_t     qs[QK4_0/2];   // 16 bytes, two int4 per byte
} block_q4_0;
```

And produces our block format with B_a, B_b pre-computed.

### What happens if the model is fp16 / bf16 / Q5_K / Q6_K / etc.?

For Phase 15a we support Q8_0 and Q4_0 only (simplest formats). For
fp16/bf16 we keep the existing path (no regression). For Q5_K, Q6_K,
Q4_K we fall back to the existing dequant-via-ggml path until the
block-format support is extended. Q4_K is the priority follow-up
because most production GGUF files use Q4_K_M.

---

## Kernel changes

New kernels:

```c
bool sp_matmul_ok_block_q8(const sp_ok_tensor&         W_shape,
                            const sp_ok_block_q8_tensor& W_blk,
                            const sp_ok_tensor&         X,
                            sp_ok_tensor&               Y);

bool sp_matmul_ok_block_q8_to_fp32(const sp_ok_tensor&         W_shape,
                                    const sp_ok_block_q8_tensor& W_blk,
                                    const sp_ok_tensor&         X,
                                    float*                      Y_fp32,
                                    int out_rows, int n_cols);

/* Q4 variants */
bool sp_matmul_ok_block_q4(...);
bool sp_matmul_ok_block_q4_to_fp32(...);
```

Inner loop (Q8 version, AVX-512):

```c
for (int64_t b = 0; b < n_blocks; ++b) {
    const sp_ok_q8_block_t& blk = W_blk.blocks[b];
    __m512i B_a_v = _mm512_set1_epi64(blk.B_a);
    __m512i B_b_v = _mm512_set1_epi64(blk.B_b);
    /* Load 32 int8 codepoints = 32 bytes from blk.packed.
     * Split into 4 chunks of 8 elements each; for each chunk:
     *   - sign-extend 8 int8 -> 8 int64
     *   - load 8 x.a and 8 x.b from X
     *   - compute F_a, F_b (4 mullo_epi64 + adds, no per-element shifts)
     *   - mullo with extended packed and accumulate
     * Net: 8 mullo per block (~4 cycles each) + 4 cache-line load per
     *      block. */
}
```

The Q4 version unpacks two nybbles per byte (the existing AVX-512
arithmetic-shift idiom from sp_ok_q4_decode_array), otherwise same
loop shape.

---

## Validation plan

Once code is written:

1. **Unit test**: `test_sp_ok_block_q8_from_gguf` — read a known Q8_0
   tensor (synthesize one with the existing ggml encoder), produce the
   block_q8 representation, dequant + Frobenius-multiply on both sides,
   compare per-element to the existing fp16 → encode → Frobenius pipeline.
   Tolerance: bounded by GGUF Q8 quant error, not by our quant error.

2. **Matmul parity test**: `test_sp_matmul_block_q8` — same inputs to
   both `sp_matmul_ok_q8` and `sp_matmul_ok_block_q8`. Outputs differ
   by GGUF block quant error (block-wise scale is strictly more accurate
   than per-tensor shift), but should be substantially closer to the
   fp16-baseline than per-tensor Q8 is.

3. **Perplexity bench**:
   - Need a Gemma3-1B-Q8_0 GGUF (re-quantize from fp16 via
     `llama-quantize` if we don't have one).
   - Bench: `--frobenius-quant --gguf-block-quant` (new flag),
     ctx=128, chunks=4.
   - Target: PPL within 0.5% of Step E Q8 (11.8311), and probably
     *better* than 11.8311 because GGUF's calibrated Q8 is closer to
     the original fp16 than our Frobenius-then-truncate scheme.

4. **Q4 perplexity** (the original Phase 14 target):
   - Gemma3-1B-Q4_0 or Q4_K_M GGUF.
   - Bench same fixture with `--gguf-block-quant`.
   - Target: PPL within a few % of fp16 baseline (whatever Q4_0 PPL
     would be without our shim — that's the ceiling).
   - **This is the Q4 path that didn't work in Phase 14.** With block
     scales it should actually work.

---

## Storage budget for a Gemma3-1B Q4_0 model

182 shim-list tensors, total numel ≈ 670M:
- 670M / 32 = 21M blocks
- Per block: 32 bytes (16 packed + 16 metadata)
- Total: 21M × 32 = **672 MB packed Q4 weights**

vs Phase 14 Q4 (per-tensor shift): **650 MB**, but PPL = 10⁹.
vs Phase 12 Q8 (per-tensor shift): **1.30 GB**, PPL = 11.8311.

Block-Q4 gives essentially the same disk footprint as per-tensor Q4
(~672 MB) but **with PPL that should land near 12-13 instead of 10⁹**.
That's the win.

---

## File inventory (when implemented)

**New files (math submodule):**
- `lib/shannon-prime/core/sp_ok_block_quant.h` — block struct definitions,
  GGUF-import functions, inline accessors.
- `lib/shannon-prime/core/sp_ok_block_quant.c` — implementation of the
  GGUF-format-to-block-fused-format converter for Q8_0, Q4_0.

**New files (engine):**
- `tests/test_sp_ok_block_q8_from_gguf.cpp`
- `tests/test_sp_matmul_block_q8.cpp`
- `tests/test_sp_ok_block_q4_from_gguf.cpp`
- `tests/test_sp_matmul_block_q4.cpp`
- `bench/phase15_block_q8_verify.bat`
- `bench/phase15_block_q4_verify.bat`

**Modified files:**
- `src/sp_ok_tensor.{h,cpp}` — add `alloc_tensor_block_q8/q4` to arena.
- `src/sp_matmul.{h,cpp}` — add the four new kernels.
- `src/sp_ffn.{h,cpp}` — add `sp_ffn_swiglu_to_fp32_block_q8/q4`.
- `src/sp_forward.{h,cpp}` — add `use_block_q8/q4` flag on `sp_weights`,
  the block_* slot vectors, the dispatch in `sp_forward_step_prefill`.
- `src/sp_weights_loader.cpp` — the GGUF-block-type intercept switch.
- `src/engine.h` — `Config::gguf_block_quant` bool.
- `src/cli/main.cpp` — `--gguf-block-quant` flag.
- `lib/shannon-prime/CMakeLists.txt` — add sp_ok_block_quant.c.
- `CMakeLists.txt`, `tests/CMakeLists.txt` — wire the new test executables.

---

## Open questions for before-code

1. **Q4_K's super-block hierarchy.** Q4_K has 256-element super-blocks
   with 8 × 32-element sub-blocks, each sub-block carrying a 6-bit
   scale and 6-bit min. Our current design assumes flat 32-element
   blocks. To support Q4_K we either flatten (compute B_a/B_b per
   sub-block and ignore the super-block structure) or carry the
   super-block metadata. Flatten is simpler and adds < 0.01 bytes/element
   overhead. Recommend: flatten, ignore super-block grouping at our level.

2. **Block_min term.** Q4_K (and Q5_K, Q6_K) have a per-block additive
   `block_min` so `W[k] = w[k] * block_scale + block_min`. This adds a
   constant per-block to every weight. In matmul, that becomes
   `Σ_k (block_min · 1) · x[j,k] = block_min · Σ_k x[j,k]`. The sum
   Σ_k x[j,k] is computed once per j per block, then multiplied by
   block_min and accumulated. Cheap. The block_min needs to be stored
   as an O_K element too (B_min_a, B_min_b) — adds 16 bytes per block.
   For Q4_K the block size is 32, so 16/32 = 0.5 B/elem overhead. Total
   Q4_K stays under 1.6 B/elem. Acceptable.

3. **Memory alignment of `sp_ok_q4_block_t` (32 bytes).** Two blocks
   per cache line on x86. On Apple M-series with 128-byte cache lines,
   four blocks per line. Both are fine — the AoS-at-block design holds.

4. **Bypass-list interaction.** Embedding tables and the LM head are
   bypassed (kept in fp32) per the Phase 1.7 policy. They're large
   (Gemma3-1B has 262144 × 1152 = 302M embedding params = 1.2 GB
   fp32). Block-quant ingest doesn't change this — bypass tensors
   never touch the new path.

---

## Status checklist

- [ ] Confirm GGUF block-quant struct layouts vs `ggml-common.h`
- [ ] Write `sp_ok_block_quant.{c,h}` with Q8_0 + Q4_0 importers
- [ ] Add block_q8/q4 arena allocators + tensor descriptors
- [ ] Write 4 new matmul kernels (block_q8, block_q8_to_fp32, q4 variants)
- [ ] Write 4 new FFN-fused helpers (block_q8/q4)
- [ ] Wire block tensors through `sp_weights`, `sp_weights_loader`,
      `sp_forward_step_prefill`
- [ ] Add `--gguf-block-quant` CLI flag
- [ ] Generate / acquire Gemma3-1B Q8_0 and Q4_0 GGUFs
- [ ] Run unit tests
- [ ] Run perplexity benches vs Step E baseline
- [ ] Q4_K extension (Phase 15b)
- [ ] Document final PPL numbers + memory footprint
