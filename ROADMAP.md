# ROADMAP.md — Shannon-Prime Engine Implementation Plan
# Total Silicon Dominance: HTP → HVX → Halide → NEON → MoE → ISP

## Proven Foundation (Phases 1–5: COMPLETE)

Everything below is built on top of what's already working on-device:

| Achievement | Evidence |
|-------------|----------|
| 4-split HTP concurrent residency | shareResources=true, QNN_PRIORITY_LOW |
| 423–500+ tok/s prefill | 302ms/128-token chunk, S22U |
| UFIXED_16 quantization bridge | dtype 1046, zero-point 30833, all scales recovered |
| Zero-copy ION persistent tensors | rpcmem + QnnMem_register |
| 4096-byte aligned buffers | memalign, no DMA stutter |
| OpenAI-compat HTTP server | /v1/chat/completions on 0.0.0.0:8080 |
| Glassmorphic web frontend | Served from --www, live status + telemetry |
| Runtime graph build | MatMul + KQ+Softmax at 238 µs on V69 |
| Clean teardown | Exit code 0, dlclose deferred |
| Numerical coherence | nan=0 across all splits |

---

## Phase 6: HVX Logit Path — Keep Data on the DSP
**Status:** NOT STARTED
**Goal:** Move Softmax, LayerNorm, and Argmax from ARM CPU to Hexagon HVX.
Return only a single int32 token ID to the host — 4 bytes instead of megabytes.

### Why This Matters
Right now, after Split 4 executes on HTP, the entire logit tensor (vocab_size ×
uint16 = ~300 KB for Qwen3-4B) gets copied back to host memory where the CPU
runs argmax. This is a round-trip penalty on every single token. Moving the
logit reduction onto HVX eliminates this transfer entirely.

### Entry Points
| File | What to Change |
|------|---------------|
| `lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/sp_qnn.c` | Add HVX dispatch after final graph execute |
| `lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/sp_qnn.h` | New API: `sp_qnn_execute_with_hvx_argmax()` |
| `src/qnn_bin_driver.cpp` | Replace host-side argmax with HVX call |
| NEW: `lib/shannon-prime/backends/hexagon/sp_hvx_logits.c` | HVX Softmax + Argmax kernel |

### Implementation Steps
1. **Write HVX Argmax kernel** — vectorized comparison across uint16 logit buffer.
   HVX has 1024-bit vectors = 64 × uint16 per instruction. Qwen3-4B vocab =
   151,936 tokens = ~2,374 HVX vector loads for a full argmax scan.
2. **Write HVX Softmax kernel** (optional for sampling) — log-sum-exp in HVX
   int16 arithmetic using the UFIXED_16 scales. Only needed if we want
   temperature/top-p sampling on-DSP; greedy argmax doesn't need softmax.
3. **Wire into sp_qnn.c** — after `QnnGraph_execute()` on Split 4, call the HVX
   kernel on the output tensor BEFORE copying anything to host. Return only the
   winning token index.
4. **Update qnn_bin_driver** — receive int32 token ID instead of full logit tensor.

### HVX Programming Model
- HVX kernels run on the cDSP alongside HTP but on the vector units
- Use `#include <hexagon_types.h>` and `HVX_Vector` intrinsics
- Compile with `-mhvx -mhvx-length=128B` for V69 (128-byte vectors = 1024 bits)
- The output tensor from Split 4 is already in DSP-accessible memory (ION-backed)
  so the HVX kernel reads it directly — zero copy

### Success Criteria
- [ ] Argmax produces identical token IDs to current host-side argmax
- [ ] No logit tensor transfer to host (verify via profiling: no memcpy after Split 4)
- [ ] Frontend produces coherent responses (same quality as current)
- [ ] Measurable latency reduction on decode (even if small — the real win is bandwidth)

### Risks
- HVX scheduling conflicts with HTP — mitigated by QNN_PRIORITY_LOW
- UFIXED_16 argmax needs correct interpretation (compare raw uint16, NOT decoded fp32)

---

## Phase 7: Halide DMA Prefetch — Async Layer Streaming
**Status:** NOT STARTED (DMA probe built at 2a17f37)
**Goal:** While HTP crunches layer N, Halide-orchestrated DMA pulls layer N+1
weights from UFS into TCM. Triple-buffered zero-stall pipeline.

### Why This Matters
Currently, the 4-split execution is sequential: load all, then execute 1→2→3→4.
With Halide DMA, we can overlap execution and I/O. This becomes critical for
larger models (27B MoE) where we can't fit everything in memory at once.

### Entry Points
| File | What to Change |
|------|---------------|
| `lib/shannon-prime/backends/halide/` | DMA probe already exists — extend to production |
| NEW: `lib/shannon-prime/backends/halide/sp_dma_prefetch.cpp` | Halide generator for async weight streaming |
| `src/qnn_bin_driver.cpp` | Interleave DMA prefetch with split execution |
| `lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/sp_qnn.c` | Hook for async buffer swap |

### Implementation Steps
1. **Extend DMA probe** — the existing `dma_raw_blur_rw_async` pattern proves
   async rpcmem→VTCM transfer works. Adapt for weight tensor streaming.
2. **Triple-buffer allocator** — three ION-backed buffers:
   - Buffer A: Currently executing on HTP
   - Buffer B: DMA filling from UFS
   - Buffer C: Being prepared by NEON (tokenization/routing)
3. **Halide schedule** — use `eDmaFmt_RawData` to stream SP-packed weight bands
   as 1D RAW frames. The DMA engine treats them as Bayer-equivalent data.
4. **Pipeline coordinator** — modify the split execution loop to overlap:
   ```
   Start DMA(Split N+1 weights → Buffer B)
   Execute HTP(Split N, Buffer A)
   Wait DMA complete
   Swap A ↔ B
   ```
5. **VHT2 on-the-fly** (stretch) — decompress banded weights in HVX as they
   arrive via DMA, before feeding to HTP.

### Available Infrastructure
- `HalideRuntimeHexagonDma.h` — async read/write, UBWC, ION/rpcmem support
- `eDmaFmt_RawData` — any byte stream as 1D RAW frame
- UBWC hardware compression — can stack with SP band compression at bus level
- DMA probe validated: rpcmem→VTCM async transfer working (commit 2a17f37)

### Success Criteria
- [ ] Measurable prefill speedup (overlap eliminates I/O stall between splits)
- [ ] No coherence regression (same token output quality)
- [ ] Triple-buffer pattern verified via Snapdragon Profiler timeline
- [ ] Frontend remains responsive during streaming

### Risks
- TCM pressure from triple-buffering — may need to reduce buffer sizes
- DMA bandwidth contention with HTP memory accesses

---

## Phase 8: NEON Oracle — Speculative Prefetch & MoE Routing
**Status:** IMPLEMENTED (2026-05-05) — pending on-device validation
**Goal:** Use ARM NEON cores for speculative token prediction, MoE expert
routing, and high-speed tokenization. The CPU becomes the "brain" while
the DSP is the "muscle."

### Why This Matters
For MoE models (Qwen 3.6 27B), the bottleneck shifts from compute to
**expert routing**. If we know which 2-of-16 experts are needed 2–3 layers
in advance, Halide DMA can pre-stream just those experts from UFS.

### Components
1. **Speculative Draft Model** — tiny Qwen-Lite (~100M params) running in
   NEON-optimized fp32 on one ARM core. Predicts next 3–4 tokens to prime
   the DMA prefetch pipeline.
2. **MoE Gating Logic** — NEON-vectorized router that computes top-2 expert
   selection from the gating scores. Feeds selection to Halide DMA.
3. **Tokenization** — BPE encode/decode stays on ARM (string manipulation
   is ARM's strength, not DSP's).
4. **Agentic Parsing** — real-time text parsing for tool-call triggers and
   safety guardrails while HTP generates tokens.

### Entry Points
| File | What to Change |
|------|---------------|
| `src/sp_kernels_cpu.cpp` | NEON-optimized softmax/gating already here — extend |
| NEW: `src/speculative_oracle.cpp` | Draft model runner + prefetch trigger |
| `src/qnn_bin_driver.cpp` | Accept prefetch hints from oracle |
| `lib/shannon-prime/backends/adreno/shannon_prime_adreno.c` | NEON tier functions |

### Success Criteria
- [ ] Draft model predicts next token with >50% accuracy (sufficient for prefetch)
- [ ] Expert routing selects correct top-2 experts (validated against full model)
- [ ] Measurable latency reduction from prefetch hits
- [ ] No CPU thermal throttling from draft model (monitor via Snapdragon Profiler)

---

## Phase 9: MoE Scaling — Qwen 3.6 27B on S22 Ultra
**Status:** FUTURE
**Goal:** Run a 27B Mixture-of-Experts model on the S22U by treating the
model as a dynamic stream rather than a resident blob.

### Architecture
```
ARM NEON: Gating → which 2/16 experts?
    ↓
Halide DMA: Stream ONLY those 2 experts from UFS (2.1 GB/s)
    ↓
HVX: VHT2 decompress bands as they arrive
    ↓
HTP: Execute active experts in persistent context
    ���
HVX: Argmax → int32 token ID → ARM
```

### Key Constraint
27B params × 4-bit = ~13.5 GB on disk. S22U has 12 GB RAM. Only ~2–3 GB
available for model weights after OS + HTP contexts. Solution: **JIT Expert
Streaming** — only the active experts (2/16 per layer = ~1.7 GB active)
need to be in memory at any time.

### Prerequisites
- Phase 7 (Halide DMA) must be working for async weight streaming
- Phase 8 (NEON Oracle) must be working for expert prediction
- Per-expert .bin compilation from AI Hub (or per-expert QNN context binaries)

### Success Criteria
- [ ] 27B model generates coherent text on S22U
- [ ] >5 tok/s sustained decode (expert streaming bottleneck)
- [ ] No OOM under sustained generation
- [ ] Frontend shows coherent multi-turn conversation

---

## Phase 10: Mode D — ISP Spectral Reconstruction (Research)
**Status:** FUTURE RESEARCH
**Goal:** Use the Spectra 680 ISP's fixed-function MFNR (Multi-Frame Noise
Reduction) as an 18-bit spectral reconstructor for VHT2 band fusion.

### The Insight
The ISP's temporal fusion logic performs aligned weighted accumulation at
18-bit fixed precision — exactly what VHT2 band reconstruction needs. By
packing weight bands as "RAW frames" (treating them as Bayer data), the
ISP performs the skeleton+residual fusion at wire speed, parallel with HTP
matmul computation.

### Why 18-bit Fixed > fp32
In fp32 band summation, small residual magnitudes (B₂, B₃) get swallowed
by mantissa alignment of larger skeleton bands (B₀, B₁). In 18-bit fixed,
every bit of every band is preserved in the accumulation. This eliminates
the ~3.8e-6 error floor measured in April.

### Prerequisites
- Mode C (Phase 7) must be shipping first
- ISP access may be HAL-gated on production-locked S22U
- DMA probe Stage 2 (MFNR Config) must pass

### Status
- Stage 1 (Buffer Injection): UNKNOWN
- Stage 2 (MFNR Config): UNKNOWN
- Halide DMA unreachable in unsigned PD on production-locked S22U (Mode D
  Stage 1 result, 2026-05-01) — stay on rpcmem path for now

---

## The "Total Utilization" Map (Target State)

| Silicon Block | Role | Data Path | Phase |
|---------------|------|-----------|-------|
| **Hexagon HTP** | Dense MatMuls, Attention | w4a16 quantized | Done (Phase 5) |
| **Hexagon HVX** | Softmax, Argmax, LayerNorm | fp16/int16 | **Phase 6** |
| **Halide DMA** | Async weight streaming, VHT2 | int8 streamed | **Phase 7** |
| **ARM NEON** | Tokenization, Oracle, Routing | fp32/int32 | **Phase 8** |
| **UFS 3.1** | Weight storage, 2.1 GB/s | Sequential read | Phase 7+ |
| **DDR** | Skeleton weights (~200 MB resident) | Random access | All phases |
| **Spectra 680 ISP** | 18-bit spectral reconstruction | Fixed-point | Phase 10 |
| **Adreno 730 GPU** | Future hybrid GPU+HTP | fp16 | Future |

## The Zero-Stall Triple-Buffered Pipeline (Target)

```
Time →
         ┌──────────┐┌──────────┐┌──────────┐
Buffer A │ EXECUTE  ││ PREPARE  ││ EXECUTE  │  (HTP crunching)
         └──────────┘└──────────┘└──────────┘
         ┌──────────┐┌──────────┐┌──────────┐
Buffer B │ TRANSFER ││ EXECUTE  ││ PREPARE  │  (DMA filling)
         └──────────┘└──────────┘└──────────┘
         ┌──────────┐┌──────────┐┌──────────┐
Buffer C │ PREPARE  ││ TRANSFER ││ TRANSFER │  (NEON routing)
         └──────────┘└──────────┘└──────────┘
```

When HTP finishes Buffer A, it flips instantly to Buffer B (already filled).
Stall time = 0. The bottleneck becomes the slowest of {HTP, DMA, NEON},
not the sum.

## Validation Protocol

**Every phase must pass this acceptance test:**
1. Build for Android aarch64 with QNN+Hexagon enabled
2. Push to device, run qnn_bin_run with test prompt
3. Verify coherent next token + nan=0 + clean exit
4. Start serve mode, open frontend in browser
5. Have a multi-turn conversation through the frontend
6. Verify response quality matches pre-change baseline

If the frontend doesn't produce coherent, responsive chat — the work isn't done.
