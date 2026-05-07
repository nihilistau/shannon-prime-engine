# Shannon-Prime Phone Engine — Architecture & Roadmap

**Target:** Samsung Galaxy S22 Ultra (SM8450 / Snapdragon 8 Gen 1)
**Silicon:** Hexagon V69 HTP + HVX + ARM Cortex-A78/A510 + Adreno 730 + Spectra 680 ISP + UFS 3.1
**Engine:** sp-engine at `D:\F\Projects\Shannon-Prime-Refactor\shannon-prime-engine\`

This document covers what's shipping, what's next, and what new mathematics applies to the phone's constrained environment. Concepts are ordered by impact and feasibility — useful stuff first, theoretical research at the end.

---

## Part I: Proven & Shipping

### The Pipeline

```
ARM NEON                          Hexagon V69
┌────────────────┐    FastRPC    ┌─────────────────────────┐
│ Tokenizer      │──────────────▶│ HTP: 4-split MatMul     │
│ HTTP Server    │               │ (QNN graphs, UFIXED_16) │
│ KV Management  │◀──────────────│                         │
│ SP Compression │    int32 tok  │ HVX: Softmax/Argmax     │
└────────────────┘               └─────────────────────────┘
         │                                │
         ▼                                ▼
  Frontend (glassmorphic)          ION shared memory
  0.0.0.0:8080                     (zero-copy, 4096-aligned)
```

### Validated Performance

| Metric | Value | Evidence |
|--------|-------|----------|
| Prefill throughput | 423-500+ tok/s | 302ms/128-token chunk, 4-split |
| Spec-decode eval | 43.72 t/s | 3.58× vs vanilla 12.20 t/s |
| HTP residency | 4-split concurrent | shareResources=true, QNN_PRIORITY_LOW |
| Numerical coherence | nan=0 | UFIXED_16 bridge verified |
| VHT2 compression | 3.37× (ship) | 76 bytes from 256 bytes fp16 |
| Fused decompress-matmul | 1.79× vs vanilla | ARM CPU bandwidth crossover |

### Hardware Constants (Empirically Discovered)

| Constant | Value | Notes |
|----------|-------|-------|
| UFIXED_POINT_16 dtype | 1046 | AI Hub bakes uint16, NOT fp16 (534) |
| Zero-point offset | 30833 | fp32 ≈ 7.2e-6 × uint16 − 0.222 |
| Buffer alignment | 4096 bytes | ION/SMMU; unaligned = DMA stutter |
| HTP context priority | QNN_PRIORITY_LOW | Reduces TCM pressure for co-residency |
| FastRPC dispatch ceiling | 577 calls/sec | Per-layer dispatch (28 calls/tok) is DOA |

### Compression Stack on Phone

The full Shannon-Prime pipeline runs on ARM NEON:
1. VHT2 forward (in-place, hd=128)
2. Möbius reorder (squarefree indices → front)
3. Banded quantization (5/5/4/3)
4. System 1/2 dual-cache switching

The HTP handles MatMul + Attention. The CPU handles everything else including SP compression/decompression. This split is correct — VHT2 is memory-bound butterfly operations that the CPU handles well, while MatMul is compute-bound and belongs on the tensor accelerator.

---

## Part II: Active Roadmap (Committed Phases)

### Phase 6: HVX Logit Path [NOT STARTED]

**Goal:** Move Softmax/Argmax from ARM to HVX. Return int32 token ID instead of megabyte logit tensor.

**Why:** After Split 4 on HTP, the entire logit tensor (~300 KB for Qwen3-4B) gets copied back to host. HVX argmax on the DSP-side eliminates this transfer.

**HVX programming model:** 1024-bit vectors = 64 × uint16 per instruction. Qwen3-4B vocab = 151,936 tokens = ~2,374 HVX vector loads for a full scan.

**Key risk:** HVX scheduling conflicts with HTP — mitigated by QNN_PRIORITY_LOW.

### Phase 7: Halide DMA Prefetch [NOT STARTED]

**Goal:** While HTP executes layer N, Halide DMA pulls layer N+1 weights from UFS into TCM. Triple-buffered zero-stall pipeline.

```
Buffer A: EXECUTE (HTP)  →  PREPARE  →  EXECUTE
Buffer B: TRANSFER (DMA)  →  EXECUTE  →  PREPARE
Buffer C: PREPARE (NEON)  →  TRANSFER →  TRANSFER
```

Stall time → 0. Bottleneck becomes max(HTP, DMA, NEON), not the sum.

### Phase 8: NEON Oracle [IMPLEMENTED — pending validation]

Speculative draft model (~100M params) running NEON-optimized fp32 on one ARM core. Predicts next 3-4 tokens to prime DMA prefetch. MoE gating logic for top-2 expert selection.

### Phase 9: MoE Scaling — 27B on S22U [FUTURE]

JIT Expert Streaming: only active experts (2/16 per layer = ~1.7 GB active) in memory at any time. Requires Phase 7 (DMA) + Phase 8 (Oracle) working.

### Phase 10: ISP Spectral Reconstruction [RESEARCH]

Use Spectra 680 ISP's MFNR silicon as 18-bit spectral reconstructor for VHT2 band fusion. Currently blocked: Halide DMA unreachable in unsigned PD on production-locked S22U.

---

## Part III: New Mathematics for Phone

These are new concepts that are feasible within the phone's constraints — integer-strong DSP, limited FLOPs, fixed-point arithmetic, bandwidth-bound.

### A. GF(p) Mersenne Arithmetic for VHT2 Indices [HIGH PRIORITY]

**What:** Replace modular index arithmetic in VHT2 with Mersenne field GF(2^31 − 1) operations.

**Why it fits phone:** The Hexagon DSP has strong 32-bit integer pipelines. Mersenne reduction is a single-instruction pattern:

```c
static inline uint32_t mersenne_mod(uint64_t x) {
    uint32_t lo = (uint32_t)(x & 0x7FFFFFFF);
    uint32_t hi = (uint32_t)(x >> 31);
    uint32_t r = lo + hi;
    return r >= 0x7FFFFFFF ? r - 0x7FFFFFFF : r;
}
```

No integer division. The VHT2 forward pass computes multi-dimensional index mappings (linear index → factored coordinates → butterfly address). Currently uses generic `%` operator. Mersenne reduction replaces every `%` with shift+add.

**Expected impact:** 15-30% speedup on VHT2 index computation. Doesn't change the math — same transform, same coefficients, faster indexing.

**Implementation:**
1. Replace `idx % p` with `mersenne_mod()` in `sp_vht2_forward_f32()`
2. For primes {2, 3, 5, 7, 11}: embed them in GF(2^31−1) — each small prime divides (2^31−1−1) so the subgroup structure is preserved
3. Benchmark on S22U ARM + Hexagon HVX paths
4. If V69 HVX int32 vector ops benefit, write an HVX-vectorized Mersenne VHT2 kernel

### B. SVD Entropy Calibration Signal [MEDIUM PRIORITY]

**What:** Upgrade variance-ranked calibration to use SVD spectral entropy of the coefficient covariance matrix.

**Why it fits phone:** The SVD computation happens once at calibration time (first prefill), not on the hot path. The covariance matrix is head_dim × head_dim (128×128 = 16K floats). The SVD of a 128×128 matrix takes microseconds on ARM.

**Impact:** Better coefficient ranking → better band allocation → better compression at identical bit budget. The improvement compounds: even a 1% quality gain at calibration propagates through every subsequent token.

**Implementation:**
1. In `sp_shadow_calibrate_feed()`: accumulate covariance matrix (outer product sum)
2. In `sp_shadow_calibrate_end()`: SVD → entropy-ranked permutation
3. Falls back to variance ranking if SVD fails (rank-deficient input)

### C. Ricci Curvature for System 1/2 Routing [LOW PRIORITY]

**What:** Use discrete Ollivier-Ricci curvature of the attention graph as System 1→2 switch criterion.

**Why it fits phone (barely):** The computation is O(n_tokens × k²) where k is the attention top-k. For decode with n_tokens=512 and k=8: ~32K operations — feasible on NEON but not free. Worth it only if the routing decision is significantly better than the current entropy threshold.

**Alternative for phone:** Approximate Ricci curvature via the p=3 band energy ratio that the Ricci sentinel already tracks. The sentinel *is* a 1D projection of the curvature — potentially sufficient for phone where compute budget is tight.

### D. Free Energy Objective for Band Selection [LOW PRIORITY]

**What:** Use variational free energy to automatically select optimal band-bit allocation per model, replacing manual grid search.

**Why it fits phone:** Computation is at calibration/model-pack time only, not inference hot path. Pre-compute optimal band allocation per architecture, store in model-pack registry.

**Implementation:** At calibration, evaluate F = E[log q(x|z)] − KL(q||p) for each candidate allocation. Select the allocation minimizing F. Store result in model-pack preset.

---

## Part IV: Theoretical Research (Phone-Adjacent)

These concepts are theoretically interesting but face significant obstacles on phone hardware. Documented for completeness and future reference.

### T1. Poincaré Disk Attention

Hyperbolic attention could improve System 2's handling of hierarchically-structured text. However, the Poincaré metric's (1−||x||²) denominator is numerically catastrophic on int16 HVX lanes. The Lorentz hyperboloid model is stabler but still requires fp32 minimum precision — available on NEON but not on HVX.

**Verdict:** Not viable on Hexagon. Might work as a NEON-only System 2 fallback for the ~1% of queries that trigger System 2, since those already accept higher latency.

### T2. Quaternion KV Projections

Hamilton product matmul costs 4× the FLOPs. On a phone fighting for every millisecond in the attention path, this is unacceptable. The theoretical benefit (tighter spectral concentration in VHT2 domain) would need to deliver >4× compression improvement to break even — unlikely.

**Verdict:** Non-starter on phone. Desktop only.

### T3. CRT Multi-Device Splitting

The phone has one SoC. There's no second device to split to. The DSP+GPU heterogeneous split is already handled by the QNN pipeline + Adreno backend, which uses task parallelism not CRT algebraic decomposition.

**Verdict:** Irrelevant for single-SoC phone target.

### T4. Zeta-Zero Predictor Initialization

The hierarchical predictor's W matrix is calibrated via ridge regression. Ridge regression is a closed-form solution — the initialization is overwritten by the data. Zeta-zero initialization would only help if the calibration data is extremely scarce (< n_skeleton samples), which happens only for very short prompts. On phone, the calibration warmup is typically 128+ tokens — enough to dominate any initialization.

**Verdict:** Theoretically sound, practically irrelevant for phone where calibration data is sufficient.

### T5. Shor-Inspired Phase Masking

Requires model fine-tuning to learn with the phase mask. We don't train models, and on-device fine-tuning is impractical at phone scale. Post-training application without evidence would be reckless.

**Verdict:** Requires training pipeline we don't have. Park indefinitely.

---

## Part V: The Total Utilization Map

| Silicon Block | Current Role | Future Role | Phase |
|---------------|-------------|-------------|-------|
| Hexagon HTP | 4-split MatMul, Attention | Same + larger models via streaming | Done → Phase 9 |
| Hexagon HVX | Bridge only | Softmax, Argmax, LayerNorm, VHT2 | Phase 6 |
| ARM NEON | Everything else | Tokenizer, Oracle, Routing, SP compression | Phase 8 |
| Halide DMA | Probe built | Async weight streaming, triple-buffer | Phase 7 |
| Adreno 730 | Vulkan fallback | Hybrid GPU+HTP matmul | Future |
| UFS 3.1 | Model storage | JIT expert streaming at 2.1 GB/s | Phase 9 |
| DDR | Skeleton weights | ~200 MB resident skeleton set | All |
| Spectra 680 ISP | Unused | 18-bit spectral reconstruction | Phase 10 |

**The principle:** Use ALL the silicon. The phone is not a CPU with accessories — it's a heterogeneous supercomputer where every block has a role.
