# Shannon-Prime Desktop Engine — Architecture & Roadmap

**Target:** Desktop/Server — NVIDIA CUDA, Vulkan, multi-GPU, Optane, NVMe
**Engine:** sp-engine with `-DSP_ENGINE_WITH_CUDA=ON` (same codebase, different build flags)
**Reference hardware:** RTX 2060+ (CUDA), Intel UHD 770 (Vulkan), Beast Canyon (Optane + dual-GPU)

This document covers the desktop engine's current state, committed improvements, and new mathematics that become viable with desktop's compute and memory headroom. Useful stuff first, theoretical research at the end.

---

## Part I: Proven & Shipping

### The Desktop Pipeline

```
┌──────────────────────────────────────────────────┐
│  sp-engine CLI                                    │
│    verbs: generate | serve | ppl                  │
├──────────────────────────────────────────────────┤
│  GPU-Resident KV Cache                            │
│  ┌────────────────────────────────────────┐       │
│  │ Compressed bands live in VRAM          │       │
│  │ VHT2 + band quant/dequant as CUDA     │       │
│  │ kernels — no host round-trip           │       │
│  │ FP8 (E4M3FN) for V cache              │       │
│  │ FP4 (MXFP4) for Blackwell             │       │
│  └────────────────────────────────────────┘       │
├──────────────────────────────────────────────────┤
│  Vulkan fallback (AMD, Intel, cross-platform)     │
│  Beast Canyon: Optane + AVX-512 + dual-GPU        │
├──────────────────────────────────────────────────┤
│  HTTP server (OpenAI-compat) + frontend           │
└──────────────────────────────────────────────────┘
```

### Validated Performance

| Metric | Value | Evidence |
|--------|-------|----------|
| GPU cache PPL eval | 1m28s (full) | vs 23 min host fallback = 15.6× faster |
| Ship compression | 3.37× | 76 bytes from 256 bytes fp16 |
| Hierarchical compression | 7.0–12.6× | Kronecker sub-projection + linear predictor |
| Cross-GPU correlation | 1.0000 | RTX 2060 + Intel UHD, Vulkan + CUDA |
| FP8 V cache | Shipped | E4M3FN higher dynamic range for smooth distributions |
| Multi-GPU layer sharding | Working | GPU[L × n_gpus / n_layer] |

### Desktop Advantages Over Phone

| Dimension | Phone | Desktop |
|-----------|-------|---------|
| FLOPs budget | Tight (every ms counts) | Generous (GPU compute headroom) |
| Memory | 12 GB shared (3 GB usable) | 8-24 GB VRAM + system RAM + Optane |
| Precision | int16 HVX, UFIXED_16 HTP | fp32, fp16, fp8, fp4 native |
| Multi-device | Single SoC | Multi-GPU, GPU+Optane, NVMe tiers |
| Power | Thermal throttle risk | Sustained workloads |
| Numerical stability | Fragile (fixed-point) | Robust (IEEE float) |

These advantages unlock several mathematical concepts that are impractical on phone.

### Hierarchical Compression: Tradeoff Analysis

The hierarchical predictor (Kronecker sub-projection + calibrated W linear map) supports
two independent compression knobs: **ternary noise-tail quantization** and **split K/V
residual bits**. Both are zero-PPL-degradation techniques on the models tested so far.

**Benchmark (Qwen3-0.6B Q4_K_M, ctx=256, 2 chunks):**

| Config | K B/pos | V B/pos | ×/head | K_corr | V_corr | PPL |
|---|---|---|---|---|---|---|
| Default (5,5 K2/V2) | 53 | 53 | 4.8× | 0.992 | 0.924 | 15.31 |
| Ternary band 1 (5,tern K2/V2) | 50 | 50 | 5.1× | 0.960 | 0.890 | 15.31 |
| Split K/V (5,5 K1/V2) | 36 | 53 | 5.8× | 0.974 | 0.924 | 15.31 |
| Combo (5,tern K1/V2) | 33 | 50 | 6.2× | 0.941 | 0.890 | 15.31 |

**K vs V correlation asymmetry.** K tolerates aggressive compression because attention
scores pass through softmax, which amplifies relative differences — a K vector with 0.974
correlation still produces nearly identical attention patterns. V directly multiplies the
attention weights, so V_corr maps linearly to output error. Split K/V exploits this: K at
1-bit residual saves 17 bytes/pos (32% of K budget) while V stays at 2-bit to preserve
output fidelity.

**Compute overhead.** The hierarchical predictor adds ~0.01% of inference FLOPs. The
Kronecker sub-projection selects 14 of 154 dimensions (~9% skeleton), and the W linear map
is a single small matmul per head position. On desktop GPU this is invisible in profiling.
On phone the overhead is similarly negligible — the DSP path spends >99% of time in
attention matmul, not compression.

**Drift and Cauchy reset implications.** The Mertens sentinel monitors accumulated
reconstruction error. The scaling law is `4700 × gap²` per position, where gap = 1 − corr.
For the default config (gap=0.008), scaling is 0.30 — negligible even at 128K context.
For the combo config (K gap=0.059), scaling is 16.4 — the Mertens sentinel may trigger
periodic Cauchy resets at long context (>4K positions). This is by design: the sentinel
forces a clean re-anchor before error accumulates enough to affect output quality.

**Practical recommendations:**

- **Phone (memory-constrained):** Split K/V (`--hier-res-bits 1 --hier-res-bits-v 2`) is
  the default recommendation. 5.8× compression with V_corr preserved at 0.924. The 0.974
  K_corr is well above the Mertens trigger threshold for typical context lengths (≤4096).

- **Desktop long-context (>8K):** Default K2/V2 (`--hier-res-bits 2`). The 4.8× compression
  is still substantial, and 0.992 K_corr means drift is negligible at any context length.

- **Desktop short-context / throughput-optimized:** Combo (`--hier-res-bits 1
  --hier-res-bits-v 2 --hier-ternary-mask 0x2`) for maximum 6.2× compression. Mertens
  resets at long context are the tradeoff, but for serving workloads with short conversations
  this is free compression.

---

## Part II: Active Improvements

### A. GPU-Resident Cache Enhancements

The CUDA backend already runs VHT2 + band quant/dequant entirely on-GPU. Improvements in progress:

1. **Warp-level VHT2:** Current CUDA kernel uses one thread per butterfly. Warp-cooperative butterflies (shuffle instructions) eliminate shared memory for the p=2 stages.
2. **Persistent cache across requests:** Keep the compressed KV cache in VRAM between HTTP requests for multi-turn conversations. Currently deallocated per-request.
3. **Async compression:** Overlap VHT2+quantize of position N with attention computation for position N−1 using CUDA streams.

### B. Beast Canyon Pipeline

The heterogeneous desktop stack: Optane (byte-addressable persistent storage) + AVX-512 (LLC-staged dequant) + dual-GPU.

```
Optane mmap → AVX-512 shredder → LLC → GPU MatMul
     ↑              ↑                       ↑
 sp_optane.h    sp_avx512_shredder.h    sp_beast_canyon.h
```

The shredder reads compressed SP bands directly from Optane mmap, dequantizes via AVX-512 into the LLC (Last Level Cache), and feeds to GPU MatMul — zero DRAM touches on the read path.

### C. FP4 for Blackwell

MXFP4 (4-bit float with shared exponent) is native on Blackwell tensor cores (sm_120+). Shannon-Prime's banded quantization sits on top: VHT2 → Möbius → band → FP4 per-band packing. The combination gives VHT2 spectral precision with hardware-native FP4 throughput.

Requires CUDA 12.8+ and sm_120. Build flag: `SP_FP4=1`.

---

## Part III: New Mathematics for Desktop

These concepts leverage desktop's compute headroom, fp32 stability, and multi-device capability. Ordered by impact and readiness.

### 1. CRT Multi-GPU Tensor Splitting [HIGH PRIORITY]

**What:** Replace layer-sharded multi-GPU with Chinese Remainder Theorem algebraic decomposition. Each GPU computes in its own modular residue ring; results combine via CRT with no AllReduce.

**Why it fits desktop:** Multi-GPU is a desktop problem. The current layer-sharding (`GPU[L × n_gpus / n_layer]`) requires cross-GPU copies at shard boundaries. CRT decomposition eliminates these copies entirely.

**How CRT works for MatMul:**

For coprime moduli m₁, m₂ (one per GPU):

```
GPU 0: C₀ = A · B  mod m₁
GPU 1: C₁ = A · B  mod m₂
Host:  C  = CRT(C₀, C₁)    // reconstruct full result
```

No intermediate communication between GPUs during the matmul. Only the final CRT combination requires data from both devices.

**Connection to Shannon-Prime:** VHT2's Kronecker product V = H_{p₁} ⊗ H_{p₂} ⊗ ... is already a CRT-like decomposition over prime factor groups. CRT tensor splitting extends this from the *transform domain* to the *compute domain*, using the same algebraic structure.

**Implementation path:**
1. Choose coprime moduli matching GPU count (2 GPUs: m₁=2^16+1, m₂=2^16−1)
2. Implement modular matmul CUDA kernel (integer arithmetic in GF(m_i))
3. CRT recombination kernel (runs on one GPU or host)
4. Wire into sp_beast_canyon.h as an alternative to layer sharding
5. Benchmark: latency vs layer-sharding on RTX 2060 + Intel UHD

**Expected win:** Elimination of cross-GPU copies. For attention-heavy workloads where shard boundaries cause pipeline stalls, 20-40% improvement is plausible.

### 2. Mersenne GF(p) Arithmetic for VHT2 [HIGH PRIORITY]

**What:** Same as phone (Section III.A of PHONE-ENGINE.md), but on CUDA int32 lanes.

**Why it's even better on desktop:** GPU integer pipelines are wide (thousands of concurrent int32 operations). The VHT2 index computation — mapping linear indices to multi-dimensional factored coordinates — becomes a massively parallel Mersenne reduction across all head positions simultaneously.

**CUDA kernel sketch:**

```cuda
__device__ uint32_t mersenne_mod(uint64_t x) {
    uint32_t lo = (uint32_t)(x & 0x7FFFFFFF);
    uint32_t hi = (uint32_t)(x >> 31);
    uint32_t r = lo + hi;
    return r >= 0x7FFFFFFF ? r - 0x7FFFFFFF : r;
}
```

One warp computes all index mappings for one head position. At 128 heads × 128 dim × 32 threads/warp: full VHT2 index computation in ~512 warp-cycles.

### 3. Quaternion KV Projections [MEDIUM PRIORITY]

**What:** Replace scalar Q/K projection with Hamilton product (quaternion-valued) matrices.

**Why it fits desktop:** The 4× FLOPs overhead is acceptable on desktop GPU where compute is not the bottleneck — memory bandwidth is. If quaternion projections produce K vectors with tighter VHT2 spectral concentration, the downstream compression reads fewer bytes, and the memory bandwidth savings exceed the FLOPs cost.

**The Hamilton product matmul:**

For quaternion q = a + bi + cj + dk and quaternion weight w = w₀ + w₁i + w₂j + w₃k:

```
(q · w)_real = a·w₀ − b·w₁ − c·w₂ − d·w₃
(q · w)_i    = a·w₁ + b·w₀ + c·w₃ − d·w₂
(q · w)_j    = a·w₂ − b·w₃ + c·w₀ + d·w₁
(q · w)_k    = a·w₃ + b·w₂ − c·w₁ + d·w₀
```

This is 4 scalar matmuls with specific sign patterns — maps directly to a CUDA kernel using the same tensor core throughput as 4 independent matmuls.

**Connection to Shannon-Prime:** RoPE applies 2D rotations to key pairs. Quaternions naturally represent rotations. If the Q/K projection is quaternion-valued, the RoPE rotation becomes a quaternion multiplication — and the resulting VHT2 spectrum may have more structured energy decay because the rotational symmetry is exactly represented.

**Experiment protocol:**
1. Implement Hamilton product matmul CUDA kernel
2. Modify engine forward pass: Q/K projection → quaternion
3. Run VHT2 on quaternion-projected K vectors
4. Compare spectral concentration (band energy ratios) vs scalar projection
5. If concentration improves → measure end-to-end PPL impact
6. Decision point: does compression quality improvement justify 4× projection FLOPs?

### 4. SVD Entropy Calibration [MEDIUM PRIORITY]

Same as phone (Section III.B of PHONE-ENGINE.md). On desktop, the SVD of a 128×128 matrix is instantaneous. Can afford to compute full covariance and do proper spectral analysis.

**Desktop bonus:** Can compute SVD per-(layer, head) and find layer-specific coefficient rankings. On phone, a single global ranking is used for simplicity. Per-slot SVD entropy could improve quality for models with heterogeneous layer behavior (like MoE models where different layers have very different attention patterns).

### 5. Poincaré Disk Attention for System 2 [MEDIUM-LOW PRIORITY]

**What:** Replace Euclidean dot-product attention in System 2 with hyperbolic attention.

**Why it works on desktop (but not phone):** fp32 precision handles the (1−||x||²) denominator with standard radius clamping (||x|| < 0.95). The Poincaré metric:

```
d(x, y) = arcosh(1 + 2||x − y||² / ((1 − ||x||²)(1 − ||y||²)))
```

is numerically stable at fp32 with clamping. CUDA's `__acoshf()` intrinsic handles the arcosh.

**Connection to Shannon-Prime System 2:** System 2 engages when attention entropy is high — many tokens contribute roughly equally. In Euclidean space, these tokens are "equidistant" from the query. In hyperbolic space, hierarchical structure *separates* them: structurally important tokens stay near the origin (high precision), peripheral tokens move toward the boundary (lower precision). This natural hierarchy aligns with variance-ranked band allocation.

**Alternative: Lorentz hyperboloid model.** If Poincaré is too numerically fragile even at fp32:

```
d(x, y) = arcosh(-⟨x, y⟩_L)     where ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ
```

The Lorentz model admits GEMM-friendly operations (the inner product is a standard dot product with one sign flip). Easier to fuse with existing attention CUDA kernels.

**Implementation path:**
1. Implement Poincaré distance as a CUDA kernel
2. Embed K/V vectors into the Poincaré disk via exponential map
3. Replace Q·K^T with Poincaré distance in System 2 path only
4. Measure: does hierarchical separation improve attention quality for high-entropy queries?
5. If not, try Lorentz model as a stabler alternative

### 6. Ricci Curvature Routing [LOW PRIORITY]

Same concept as phone (Section III.C of PHONE-ENGINE.md), but on desktop the compute budget for Ollivier-Ricci curvature is generous. Can compute full graph curvature per layer without latency concerns.

**Desktop advantage:** Can maintain a running curvature estimate across the decode sequence and use trend (increasing curvature = approaching Cauchy horizon) as a predictor, not just a threshold. This turns the Ricci diagnostic into a *forecasting* tool — predict when System 2 will be needed 10-20 tokens in advance and pre-stage the full-band data.

### 7. Free Energy Band Selection [LOW PRIORITY]

Same as phone (Section III.D of PHONE-ENGINE.md). On desktop, can afford to evaluate F for a larger search space of band configurations. Instead of the fixed presets (5/5/4/3, 4/4/4/3, etc.), free energy optimization can find non-standard allocations like 6/5/3/3 or asymmetric band widths.

---

## Part IV: Theoretical Research (Desktop-Adjacent)

These concepts are theoretically interesting and potentially viable on desktop, but lack empirical evidence or implementation path clarity.

### T1. Zeta-Zero Spectral Initialization

Initialize the hierarchical predictor's W matrix with eigenvalues matching the imaginary parts of the Riemann zeta zeros.

**Desktop context:** The hierarchical predictor runs on desktop too (System 2 / deep compression path). On desktop, calibration data may be more abundant (longer contexts, more diverse prompts), which means initialization matters *less* because the ridge regression solution dominates.

**However:** For zero-shot or few-shot inference (where calibration data is scarce), the initialization *does* matter. If a desktop user loads a model and immediately queries it without a warmup pass, the predictor W is initialized but uncalibrated. In that window, zeta-zero initialization could provide better default predictions than random initialization.

**Experiment:** Initialize W from zeta spectrum vs Kaiming vs zero. Run first-token perplexity (before calibration kicks in). If zeta wins the cold-start, it's worth shipping as the default initialization.

### T2. Shor-Inspired Phase Masking

The phase mask concept (modulating pre-softmax attention logits by VHT2 frequency structure) requires the model to have been trained with the mask. On desktop, we have more compute for post-training interventions, but without fine-tuning evidence, applying a phase mask is still reckless.

**Desktop-specific variant:** Instead of modifying the attention score directly, use the phase structure as a *reweighting* of the System 1/2 routing decision. Tokens whose VHT2 spectral fingerprints align with the query's spectrum get routed to System 1 (fast path); mismatched tokens get routed to System 2 (full reconstruction). This is a routing heuristic, not a model modification — it can be applied post-training safely.

**Status:** Theoretical. The routing-heuristic variant is testable without fine-tuning. Worth a spike once System 1/2 is fully operational on desktop.

### T3. Octonion/Sedenion Extensions

Octonions lose associativity; sedenions lose associativity AND have zero divisors. cuBLAS assumes associativity — every fused GEMM optimization becomes invalid. Even on desktop with ample compute, the software engineering cost of non-associative algebra (custom matmul, no cuBLAS, no tensor core fast path) is prohibitive for marginal theoretical benefit.

**Verdict:** Hard no. The quaternion extension (Section III.3) captures the rotation coupling benefit while preserving associativity. Going beyond quaternions into octonions sacrifices too much infrastructure.

### T4. Full GF(p) Attention Mechanism

Computing the full attention in a finite field (not just using Mersenne arithmetic for VHT2 indices) faces a fundamental problem: softmax requires a continuous probability distribution. GF(p) has no natural notion of "smooth" normalization — you can't take e^x mod p and get a well-behaved distribution.

**Possible resolution:** Replace softmax with a discrete "hard attention" mechanism where the top-k entries are selected purely by comparison (no exponentiation needed). GF(p) comparison is well-defined. But hard attention loses the gradient signal that makes transformers trainable — and inference quality depends on the smooth attention distribution the model learned during training.

**Verdict:** Interesting algebra, wrong deployment target. We serve models trained with softmax attention; replacing it post-training would be catastrophic.

---

## Part V: Desktop Architecture Summary

### Compute Hierarchy

```
Tier 0 (Fastest):  GPU VRAM — compressed KV cache, MatMul
Tier 1:            System RAM — model weights, scratch buffers
Tier 2:            NVMe — band-major KV spill, progressive load
Tier 3:            Optane — byte-addressable KV reservoir (Beast Canyon)
Tier 4:            Network — distributed KV for multi-node (future)
```

### GPU Pipeline

```
Token in → Embedding (GPU) → Attention (GPU, SP-compressed KV)
    → FFN (GPU) → Logits → Sample → Token out
                      ↕
              SP Shadow Cache (VRAM)
              VHT2 + Band Quant kernels
              System 1/2 switching
```

### Multi-GPU (Current: Layer Sharding)

```
GPU 0: Layers 0..N/2    GPU 1: Layers N/2..N
     └── cross-GPU copy at shard boundary ──┘
```

### Multi-GPU (Future: CRT Decomposition)

```
GPU 0: A·B mod m₁       GPU 1: A·B mod m₂
     └── CRT combine (no intermediate copies) ──┘
```

### Priority Ranking for New Work

| Priority | Concept | Expected Impact | Effort |
|----------|---------|----------------|--------|
| HIGH | CRT multi-GPU | 20-40% multi-GPU speedup | Medium |
| HIGH | Mersenne GF(p) VHT2 | 15-30% VHT2 speedup | Low |
| MEDIUM | Quaternion KV projections | Unknown (needs experiment) | Medium |
| MEDIUM | SVD entropy calibration | 1-5% quality improvement | Low |
| MEDIUM-LOW | Poincaré System 2 | Better high-entropy attention | High |
| LOW | Ricci curvature routing | Better System 1/2 switching | Medium |
| LOW | Free energy band selection | Automatic band optimization | Low |
| RESEARCH | Zeta-zero initialization | Cold-start predictor quality | Low |
| RESEARCH | Phase mask routing | Spectral attention routing | Medium |
| NO | Octonions/Sedenions | Breaks associativity | N/A |
| NO | Full GF(p) attention | Breaks softmax | N/A |
