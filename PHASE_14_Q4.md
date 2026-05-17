# Phase 14 — Q4 disk-shrink + lattice-norm pruning

Written 2026-05-18. **File-only state — code is on disk but NOT BUILT
in this session.** Workspace shells were unavailable, so the build /
test / bench triangle is still pending. The numbers in the rationale
sections are projections, not measurements.

---

## Goal

Cut the resident weight footprint by another 2× over Phase 12 Step D's
Q8 path — without re-quantizing the underlying data and without
breaking Theorem 2's projective cancellation.

Memory hierarchy (per ring element, post-Frobenius):

| Storage | Bytes/elem | Compression vs raw sp_ok_t | Phase |
|---|---|---|---|
| `sp_ok_t` (raw int64 pair) | 16 | 1× | baseline |
| `sp_ok_q8_t` (two int8) | 2 | 8× | Phase 12 Step D |
| `sp_ok_q4_t` (one byte, nybble pair) | 1 | 16× | Phase 14 |

Phase 14 also opens the door to entropy coding: when the load shim
zeros below-norm coordinate pairs (Phase 14b pruning), the packed
output develops long runs of `0x00`, which zstd/Huffman compress
aggressively for on-disk storage.

---

## Storage contract

```c
typedef struct { uint8_t packed; } sp_ok_q4_t;
//   packed & 0x0F (signed 4-bit) = a coordinate
//   packed >> 4   (signed 4-bit) = b coordinate
// per-tensor:
//   int8_t q4_shift;            // shared by every entry
//   int64_t scale_recip;        // fp16 -> int scale (from encoder)
//   int64_t frobenius_scale;    // signed Frobenius scale
//   int16_t frobenius_p, frobenius_k;
//
// Reconstruction:
//   int8_t a4 = (int8_t)(packed << 4) >> 4;  // sign-extend low nybble
//   int8_t b4 = (int8_t)packed >> 4;          // sign-extend high nybble
//   sp_ok_t r;
//   r.a = ((int64_t)a4) << q4_shift;
//   r.b = ((int64_t)b4) << q4_shift;
```

Sign extension uses the arithmetic-shift idiom in 32-bit lanes (cheap
on every modern ISA, no mask tables), then promotion to int64.

The shift picker mirrors the Q8 picker:

```c
int8_t sp_ok_q4_pick_shift(int64_t absmax) {
    if (absmax <= 7) return 0;          // SP_OK_Q4_MAX
    int8_t s = 0;
    int64_t v = absmax;
    while (v > 7) {
        v = (v + 1) >> 1;               // ceiling-divide by 2
        ++s;
    }
    return s;
}
```

Ceiling-divide matters because the encoder rounds half-up; a naive
`v >> 1` undercounts the shift and saturates the codebook on the
largest input.

Quantization error per coordinate is bounded by `2^(q4_shift - 1)` —
the same half-step bound as Q8, just at a different scale. The 4-bit
codebook has 16 levels (vs 256 for int8), so for the same numerical
absmax the q4_shift is roughly 4 bits larger than the q8_shift, giving
~16× larger absolute quantization noise per coordinate. Whether that
noise survives the downstream invariant is an empirical question —
Theorem 2's projective cancellation absorbs uniform scale changes, so
the noise that matters is the *relative* deviation from the post-
Frobenius coordinate, which the Q4 path keeps within the half-step
bound by construction.

---

## Files written

All paths relative to `D:\F\shannon-prime-repos\shannon-prime-engine\`.

### Math core (under `lib/shannon-prime/core/`)

- **`sp_ok_q4.h`** — Public API. Inline shift picker, quantize-one,
  pack-pair, decode-one, max-error, lattice-norm helper.
- **`sp_ok_q4.c`** — Implementation. `sp_ok_q4_absmax`,
  `sp_ok_q4_encode_array`, `sp_ok_q4_encode_array_pruned`, and an
  AVX-512 vectorized `sp_ok_q4_decode_array` (32-bit arithmetic-shift
  sign-extension, promotion to int64, two `unpacklo_epi64`/`unpackhi`
  to interleave (a, b) for the sp_ok_t output layout).

### Engine wiring (under `src/`)

- **`sp_ok_tensor.{h,cpp}`** — added `sp_ok_arena::alloc_tensor_q4`
  (parallel to `alloc_tensor_q8`).
- **`sp_ok_encode.{h,cpp}`** — added
  `sp_ok_encode_q4_from_fp16_with_frobenius` (fp16 → sp_ok_t → φ_p^k →
  pack-to-nybble-pair pipeline, with the same `frobenius_scale`
  tracking the Q8 helper uses).
- **`sp_matmul.{h,cpp}`** — added `sp_matmul_ok_q4` and
  `sp_matmul_ok_q4_to_fp32`. Fused scalar inner loop: each iteration
  decodes one packed byte into (w_a, w_b) with the shift inlined, then
  performs the sp_ok ring multiply via the same omega formula as the
  Q8 path:
  ```
  acc_a += w_a * x.a - 41 * w_b * x.b
  acc_b += w_a * x.b + x.a * w_b + w_b * x.b
  ```
- **`sp_ffn.{h,cpp}`** — added `sp_ffn_swiglu_to_fp32_q4` (drop-in
  replacement for the Q8 fused FFN with packed Q4 weights).
- **`sp_forward.{h,cpp}`** — extended `sp_weights` with `use_q4` flag,
  7 `q4_*` packed-tensor slots per layer, and a
  `q4_layer_arenas` parallel arena. Added `sp_weights_convert_to_q4`
  with optional pruning threshold. Forward dispatch (Q/K/V, Wo, FFN)
  now picks Q4 → Q8 → raw in priority order. The Q8 prefetcher is
  skipped under Q4 (fused decode means no scratch buffer to prefetch).
- **`sp_weights_loader.cpp`** — `cfg.frobenius_q4` gate (priority over
  `frobenius_q8` when both are set).

### CLI (under `src/cli/`)

- **`main.cpp`** — added `--frobenius-q4` and
  `--frobenius-q4-prune <threshold>` flags. The prune flag implies
  `--frobenius-q4`.

### Config (under `src/`)

- **`engine.h`** — added `Config::frobenius_q4` and
  `Config::frobenius_q4_prune` (uint64_t threshold; 0 disables).

### Tests (under `tests/`)

- **`test_sp_ok_q4.cpp`** — 6 parity tests:
  1. shift picker is minimal (boundary cases for −8..7 range)
  2. encode/decode round-trip is bounded by `2^(shift-1)`
  3. nybble pack/unpack preserves sign for all 256 (a, b) pairs
  4. lattice-norm pruning zeros below-threshold entries
  5. compression ratio is exactly 16× vs raw sp_ok_t
  6. zero-shift is identity on the [-8, 7] range

- **`test_sp_ok_q4_load_path.cpp`** — 5 tests verifying the combined
  fp16 → Frobenius → pack pipeline matches manual reproduction
  byte-for-byte, with the same shift / scale / frobenius_scale on
  both paths.

- **`test_sp_matmul_q4.cpp`** — 3 tests:
  - Q4 vs Q8 matmul on a shared (W, X) pair, characterizing the
    Q4 relative-error budget (target: mean < 0.5).
  - Q4 fp32 bridge vs Q8 fp32 bridge.
  - Shape-mismatch rejection.

Test executables and `add_test()` entries wired into
`tests/CMakeLists.txt`.

---

## Phase 14b: lattice-norm pruning

For each coordinate pair (a, b) in the ring O_K = Z[ω], compute the
algebraic norm

```
N(a + bω) = a² + a·b + 41·b²        (positive-definite by classical
                                     theory of Q(√-163))
```

and zero the pair if N < threshold. The result is long runs of `0x00`
packed bytes in the output, which downstream entropy coding (zstd,
Huffman) collapses for on-disk shipping.

Wiring:

- `sp_ok_q4_encode_array_pruned(dst, src, numel, threshold)` in
  `sp_ok_q4.c`. Two passes: (1) zero below-threshold src entries
  in place; (2) standard `sp_ok_q4_encode_array`. The absmax/shift
  picker on pass 2 sees a tighter range when pruning fires, often
  picking a smaller shift and giving the survivors more precision.
- `sp_ok_q4_last_pruned_count` global counter so the load shim can
  report the pruned fraction without re-computing.
- `sp_weights_convert_to_q4(weights, prune_threshold)` takes the
  threshold; threshold == 0 falls back to `sp_ok_q4_encode_array`.
- `--frobenius-q4-prune <threshold>` CLI flag.

---

## How to validate when shells come back

```bash
cd D:\F\shannon-prime-repos\shannon-prime-engine
cmake --build build-cuda --config Release --parallel
ctest --test-dir build-cuda --output-on-failure -R "sp_ok_q4|sp_matmul_q4"
```

Three tests must pass:
- `sp_ok_q4` (math-core parity)
- `sp_ok_q4_load_path` (engine encoder parity vs manual pipeline)
- `sp_matmul_q4` (fused matmul vs Q8 reference)

Then run perplexity against the Step E baseline:

```bash
build-cuda\bin\Release\sp-engine.exe perplexity \
    --model bench\gemma3-1b.gguf \
    --corpus bench\corpus.txt \
    --ctx 128 --chunks 4 \
    --frobenius-quant --frobenius-quant-k 8 \
    --frobenius-q4
```

Baseline numbers to beat / match:

| Config | PPL | Wall time | Memory |
|---|---|---|---|
| Phase 11 (NTT + Frobenius shim, raw weights) | 12.6551 | 331 s | ~10.4 GB ok_arena |
| Phase 12 Step D (Q8 fused) | 11.8313 | 252 s | ~1.30 GB packed |
| Step E-pre (layout flip) | 11.8311 | 223.9 s | ~1.30 GB packed |
| Step E-3 (mask shortcut) | 11.8311 | ~220 s | ~1.30 GB packed |
| **Phase 14 target** | < ~13 (16× quant noise) | similar | **~0.65 GB packed** |
| **Phase 14b target (prune=200)** | ≤ Phase 14 | similar | + zstd-friendly |

The wall-time is unlikely to improve over Step E-3 on this rig — at
1 B/elem the working set already fits in L2 for Gemma3-1B, so the
matmul is DRAM-bandwidth bound and the extra compression only buys
GPU/disk wins. The PPL drift is the empirical question.

---

## Open follow-ups (post-validation)

1. **PPL sweep over prune thresholds.** Try 50, 100, 200, 500, 1000;
   plot PPL vs (pruned %, gzipped weight bytes). The sweet spot is
   wherever both PPL stays in the Q8 ballpark *and* the entropy-coded
   weight file shrinks meaningfully.

2. **AVX-512 inner loop for `sp_matmul_ok_q4`.** The decoder for the
   array case is already vectorized; the matmul fused inner loop is
   scalar. The trade-off is the omega cross-term — `acc_b += w_a *
   x.b + x.a * w_b + w_b * x.b` doesn't map onto a clean `mul_epi32`
   reduction the way the Q8 hot path does. Worth investigating once
   wall-time on the production rig is measured.

3. **CUDA Q4 kernel.** Mirror `shannon_prime_fp8.cu`'s pattern with the
   nybble unpack baked into the load. Requires the CUDA build path
   working, which was broken in the previous session and needs the
   `--use-local-env` recipe verified.

4. **On-disk format.** Today the convert-on-load path runs every
   start-up; with pruning + zstd we could ship a `.spq4` file that
   loads in O(read) instead of O(re-compute Frobenius from fp16).

---

## Status — checklist

- [x] sp_ok_q4 module + parity test (Task #154)
- [x] Load-path encoder + Frobenius integration (Task #155)
- [x] Fused matmul kernel (Task #156)
- [x] Matmul parity test (Task #157)
- [x] Wire through load shim + forward (Task #158)
- [x] Lattice-norm pruning in load path (Task #159)
- [ ] Build + tests pass
- [ ] PPL bench vs Step E baseline
- [ ] Prune-threshold sweep
- [ ] Git commit + tag
