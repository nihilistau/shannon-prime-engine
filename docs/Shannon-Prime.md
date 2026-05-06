# Shannon-Prime: Complete System Formalization

**Version 3.0 — May 2026**
**Author: Ray Daniels (KnackAU / nihilistau)**

This document is the master reference for the Shannon-Prime system — the proven core, the active engineering, and the frontier research. Every concept is classified:

- **PROVEN** — Implemented, tested, validated on real hardware. In the codebase.
- **SHIPPED** — Proven AND deployed in a shipping path (engine, llama integration, or comfyui).
- **CONCRETE** — Clear implementation path, math verified, waiting on engineering time.
- **SPECULATIVE** — Theoretically sound, no implementation yet, may or may not survive contact with hardware.

---

## Part I: The Foundation (PROVEN)

### 1. Design Philosophy

Shannon-Prime is built on three observations:

1. **RoPE creates spectral structure.** Rotary position embeddings encode position as phase rotation across geometric frequency pairs. This imprints a predictable spectral signature on every K vector — energy concentrates in specific frequency bands and decays toward the tail.

2. **The multiplicative lattice is the natural basis.** The integers' multiplicative structure — primes, composites, squarefree numbers, the Möbius function μ(n) — provides a spectral basis that matches how RoPE distributes positional information. Squarefree indices carry 61.4% of the signal energy at N=210.

3. **Compression is a view, not a transform.** The KV cache is a projection of low-dimensional spectral structure. Shannon-Prime doesn't discard data; it changes basis to one where the signal is naturally sparse, then stores only the significant coefficients.

These lead to a system where compression quality *improves* with model scale (proven by the K-Corr scaling law), the same mathematics applies to every RoPE-based model regardless of modality, and the transform is its own inverse.

### 2. VHT2 — The Vilenkin-Hartley Transform [SHIPPED]

The core of Shannon-Prime. A staged orthonormal transform with the critical property: **self-inverse**.

```
VHT2(VHT2(x)) = x     (within float32 ULP)
```

**Definition.** For dimension n factoring into small primes {2, 3, 5, 7, 11}, VHT2 is the Kronecker product of individual Hartley stages:

```
V = H_{p₁} ⊗ H_{p₂} ⊗ ... ⊗ H_{pₖ}
```

where each p-point Hartley matrix has entries:

```
H_p[i,j] = cas(2πij/p) / √p     (cas(x) = cos(x) + sin(x))
```

**Power-of-2 specialization (ship path).** At n = 2^k, all stages are p=2 butterflies:

```
a' = (a + b) / √2
b' = (a − b) / √2
```

O(n log n) operations, in-place, no additional memory. This is the Walsh-Hadamard transform with Hartley normalization.

**Multi-prime case (sqfree path).** For squarefree-padded dimensions (e.g., 154 = 2·7·11), the Kronecker product structure creates a hierarchical decomposition. Progressive prime expansion monotonically increases spectral correlation: Walsh (0.9490) → Z/2Z × Z/3Z (0.9493) → ... → Z/2Z × Z/3Z × Z/5Z × Z/7Z (0.9513).

**Why self-inverse matters.** One function serves as encoder and decoder. One set of SIMD kernels. No encode/decode mismatch across platforms. Forward and inverse take identical FLOPs.

**Implementation:** `sp_vht2_forward_f32()`, `sp_vht2_forward_f16()` in `core/shannon_prime.h`. Every backend — CPU, CUDA, Vulkan, Adreno, Hexagon — implements the same VHT2.

### 3. Möbius Squarefree-First Reorder [SHIPPED]

Permutes VHT2 coefficients so that squarefree indices (μ(n) ≠ 0) appear first. Squarefree indices carry the majority of signal energy; pushing them to the front ensures highest-bit bands contain the most important coefficients.

**Impact:** +0.14 PPL improvement at identical bit budget. Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.

**Implementation:** `sp_mobius_mask_t`, `sp_mobius_reorder()`, `sp_mobius_unreorder()`. Hot-path scratch variants: `sp_mobius_reorder_ex()`, `sp_mobius_unreorder_ex()`.

### 4. Banded Quantization [SHIPPED]

Splits the (optionally reordered) VHT2 coefficient vector into N equal bands, each with its own fp16 scale factor and integer bit depth.

**Ship allocation (hd=128, 4 bands):** 5/5/4/3 bits → 76 bytes per vector from 256 bytes fp16 = **3.37× compression**.

**Key findings:**
- 5/5/4/3 BEATS lossless fp16 by 0.04% (spectral regularization effect)
- 4/4/4/4 is off the Pareto frontier (4/4/4/3 is strictly better)
- 3-bit floor: 2-bit on any band is catastrophic
- Flat beats banded for V vectors (no exceptions)

### 5. Ternary Noise-Tail [SHIPPED]

Band mask option quantizes tail bands to ternary {-1, 0, +1} (≈1.58 bits/coefficient). The strange-attractor analysis predicts the noise tail is statistically indistinguishable from ternary. The 5/5/4/1.58 configuration matches 5/5/4/3 quality.

### 6. The Multiplicative Lattice [PROVEN]

#### 6.1 Squarefree Numbers and Möbius Inversion

An integer n is squarefree iff μ(n) ≠ 0. The Möbius inversion formula:

```
f(n) = Σ_{d|n} μ(d) · g(n/d)
```

connects VHT2 coefficients through the divisor lattice. Squarefree coefficients are the "fundamental frequencies"; non-squarefree coefficients are predictable from them.

#### 6.2 Knight Skeleton + Möbius CSR Predictor [PROVEN]

Extracts top-K squarefree indices by variance as a skeleton. The Möbius function's divisor relationships predict remaining residual coefficients from the skeleton:

```
pred[r] = Σ_{d|(r+1), μ(d)≠0} μ(d) · skel_vals[slot((r+1)/d)]
```

Stored as Compressed Sparse Row (CSR) for O(1) per-residual prediction. On sqfree-padded Vilenkin basis: genuine predictor r=0.40–0.58 (vs r≈0 on power-of-2 WHT basis).

#### 6.3 Sqfree Padding [PROVEN]

Pads head_dim to the next squarefree-factorable number: hd=64→66, hd=128→154, hd=256→330. Mean-fill padding preserves DC component.

### 7. SU(2) Spinor Sheet Bit [PROVEN]

1-bit correction per residual position recording which sheet of the SU(2) double cover the coefficient occupies. Resolves the Möbius predictor's sign ambiguity.

**Impact:** First architectural feature that *shifts* the Pareto frontier. On Qwen3-8B Q8: K+μ+3bit+spinor 3/3/3/3/3 at PPL 7.32 @ 3.3× — matching MOBIUS default 7.31 @ 2.6×, with +27% additional compression.

### 8. Hierarchical Vilenkin Predictor [PROVEN]

Uses the Kronecker product structure to build a small core skeleton (sub-projection over a subset of primes, ~9% of coefficients) and a calibrated linear map W to predict all remaining coefficients.

For hd=128 (pad=154 = 2·7·11): 14 skeleton coefficients via Z/2Z × Z/7Z sub-projection. W is a 140×14 fp16 matrix per (layer, head) slot, calibrated via ridge regression.

**Storage:** 14×5 bits + 140×2 bits = 350 bits = 43.75 bytes from 308 bytes fp16 (padded) → **7.0× compression**. Aggressive: 14×4 + 140×1 = 196 bits = 24.5 bytes → **12.6× compression**.

### 9. Variance-Ranked Calibration [SHIPPED]

Adaptive reordering of VHT2 coefficients based on empirical per-coefficient variance from warmup data. High-variance coefficients land in high-bit bands; low-variance in low-bit bands.

Lifecycle: `calibrate_begin()` → `calibrate_feed(vec)` → `calibrate_end()`. Online adaptation via sticky-EMA blending.

### 10. Progressive Band Reads [SHIPPED]

Partial reconstruction from first N bands. Energy concentration makes this effective:

| Bands | Correlation | Bytes (hd=128) |
|-------|-------------|-----------------|
| 0     | 0.00        | 0 (sentinel)    |
| 1     | 0.30        | 22              |
| 2     | 0.86        | 44              |
| 3     | 0.88        | 60              |
| 4     | 0.99        | 76 (full)       |

Primitive enabling System 1/2 switching and disk-tier architecture.

### 11. System 1/2 Dual-Cache [SHIPPED]

Inspired by Kahneman's dual-process theory:

- **System 1 (Fast):** Read band 0 only (22 bytes/position). Accept if top-scoring attention position has sufficient margin.
- **System 2 (Careful):** Full reconstruction (76 bytes/position) when attention entropy is high.

Switching criterion: attention entropy threshold (tunable per model). System 1 reads 71% less data per position. `DualKvCache` wrapper manages this transparently.

### 12. K-Corr Scaling Law [PROVEN]

```
log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

- Quadratic in K-error: K·Q·V bilinearity squares the error
- Sub-linear in params (1.1): bigger models are more robust
- Super-linear in bits (1.5): Q4 amplifies K error ~2.8× vs Q8

Fits 9 configurations spanning 4 orders of magnitude, ±20% accuracy. Used as pre-bench filter via `sp_predicted_ppl_ratio()`.

### 13. PrimePE — Lattice-Aligned Positional Encoding [PROVEN]

Blends multiplicative-lattice frequencies into standard geometric RoPE:

```
freq_factor[i] = (1 − α) · 1.0 + α · lattice_factor[i]
```

- alpha=0.17: −0.6% PPL (deployment-robust)
- alpha=0.22: −0.8% PPL (slightly less robust)
- Universal across architectures and quantization levels. Zero retraining.

### 14. Cauchy Reset (Ricci Sentinel + Mertens Oracle) [PROVEN]

Detects accumulated compression error via p=3 band energy drift:

**Ricci sentinel:** EMA of (current_p3 / calibrated_p3). Threshold: 0.05 for 1B, 0.15 for 8B+.

**Mertens oracle:** M(n) = Σ μ(k) oscillates with half-period 200–500 tokens driven by zeta-function spectral decomposition. Pre-computed risk schedule, O(1) lookup.

### 15. Disk-Tier Progressive Loading [PROVEN]

Band-major file format (v3) enabling tiered storage: band 0 in RAM, band 1 on NVMe, bands 2–3 on Optane or network. Single contiguous read per tier on NVMe; direct memory reference on Optane (byte-addressable via DAX).

### 16. FP8/FP4 Banded Quantization [SHIPPED — CUDA only]

FP8 (E4M3FN) for V cache's smooth distributions. FP4 (MXFP4) for Blackwell tensor cores (sm_120+). Per-band scaling identical to int path.

### 17. Model-Pack Registry [SHIPPED]

Per-architecture compression defaults resolved from GGUF metadata. phi3 CALIBRATED (+2.44%), qwen3 edge-fail (+5.14%), 7 architectures PROVISIONAL.

### 18. Speculative Decoding Integration [PROVEN]

Per-model shadow caches: draft model gets aggressive settings (errors are recoverable), target model gets ship quality. Validated: Qwen2.5-Coder-3B + 0.5B draft + --draft 8 = **43.72 t/s** on phone CPU (3.58× vs vanilla).

---

## Part II: The Write/Read Pipeline (SHIPPED)

### Ship Path (Tier 1)

```
raw K/V → VHT2 → Möbius reorder → Banded quant (5/5/4/3) → Store (76 B)
```

### Sqfree + Spinor Path (Tier 2)

```
raw K/V → Sqfree pad (128→154) → Vilenkin-Hartley → Knight skeleton (L/2)
→ Band-quant skeleton → Möbius CSR predict residuals → N-bit residual + spinor → Store
```

### Hierarchical Path (Tier 3)

```
raw K/V → Sqfree pad → Vilenkin-Hartley → Kronecker sub-projection (~9%)
→ Band-quant skeleton → W·skeleton predicted targets → Residual quant → Store
```

### Read Path

```
Load → Band dequant (full or partial) → Möbius unreorder → VHT2 inverse → Reconstructed K/V
```

All paths are stateless: each KV vector compressed independently. Enables arbitrary eviction, parallel compression, reordering, and zero-copy access.

---

## Part III: Hardware Backends (SHIPPED)

9 backend implementations, all implementing the same VHT2 + banded quantization pipeline:

| # | Backend | Target | Status |
|---|---------|--------|--------|
| 1 | CPU (C reference) | Any x86-64 / ARM64 | Production — all features |
| 2 | CUDA | NVIDIA GPU (CC 6.0+) | Production — ship + sqfree + FP8 |
| 3 | Vulkan | Any Vulkan 1.1+ GPU | Production |
| 4 | Adreno | Snapdragon 8 Gen 1+ | Production |
| 5 | Hexagon HVX | Hexagon V69+ | Production — FastRPC bridge |
| 6 | QNN HTP | AI Engine Direct | Production — 4-split pipeline |
| 7 | Halide DMA | DMA engine orchestration | Probe validated |
| 8 | Beast Canyon | Optane + AVX-512 + dual-GPU | Implemented |
| 9 | ComfyUI | DiT video generation | Production |

---

## Part IV: New Mathematics — Integrated Concepts (CONCRETE)

These concepts emerge from the intersection of Shannon-Prime's proven foundations with deeper mathematical structures. Each has a clear implementation path within the existing architecture.

### 19. GF(p) Mersenne Arithmetic for VHT2 [CONCRETE]

**What:** Replace modular arithmetic in VHT2 index computation with operations in the Galois field GF(2^31 − 1), where p = 2^31 − 1 is a Mersenne prime.

**Why:** Mersenne modular reduction collapses to a shift-and-add:

```
x mod (2^31 − 1) = (x >> 31) + (x & 0x7FFFFFFF)
```

This eliminates integer division from the VHT2 index computation hot path. On integer-strong hardware (Hexagon DSP, GPU integer pipelines), this is a significant win.

**Connection to Shannon-Prime:** VHT2's staged prime-factor indexing involves modular arithmetic over each factor's group Z/pZ. The outer index computation — mapping linear indices to multi-dimensional coordinates and back — currently uses generic modular arithmetic. Mersenne reduction replaces this with bit operations.

**Implementation path:**
1. Identify all modular index operations in `sp_vht2_forward_f32()`
2. Lift index arithmetic to GF(2^31 − 1) — all intermediate results stay in the Mersenne field
3. Benchmark VHT2 forward pass with Mersenne path vs current implementation
4. Expected win: 15-30% on integer pipelines (Hexagon, CUDA int32)

### 20. SVD Spectral Entropy as Calibration Signal [CONCRETE]

**What:** Replace raw per-coefficient variance in calibration with the entropy of the SVD spectrum of the coefficient covariance matrix.

**Why:** Variance ranking treats each coefficient independently. SVD entropy captures the *joint* structure — how coefficients correlate with each other. Two coefficients with identical variance but different covariance patterns should be prioritized differently: the one carrying independent information is more valuable.

**Connection to Shannon-Prime:** Direct upgrade to `sp_shadow_calibrate_feed()`. The calibration cycle already accumulates statistics over warmup vectors. Instead of per-coefficient variance, accumulate the covariance matrix, compute its SVD spectrum at `calibrate_end()`, and rank by spectral entropy contribution.

**Implementation path:**
1. In `calibrate_feed()`: accumulate outer products (covariance matrix, head_dim × head_dim)
2. In `calibrate_end()`: SVD → singular values σ_i → entropy H = −Σ p_i log(p_i) where p_i = σ_i / Σσ
3. Rank coefficients by marginal entropy contribution (how much H drops if coefficient removed)
4. Use entropy-ranked permutation instead of variance-ranked permutation
5. Cost: one head_dim × head_dim SVD at calibration time (not on the hot path)

### 21. Ricci Curvature Diagnostic for System 1/2 Routing [CONCRETE]

**What:** Use discrete Ricci curvature of the attention graph as the System 1→2 switching criterion, replacing or supplementing the current attention entropy threshold.

**Why:** Attention entropy measures how flat the softmax distribution is, but it doesn't capture the *geometric* structure of how attention patterns relate across heads and layers. Ollivier-Ricci curvature of the token-to-token attention graph measures whether the attention manifold is locally flat (System 1 sufficient) or curved (System 2 needed for accuracy).

**Connection to Shannon-Prime:** The Cauchy Reset system already tracks spectral drift via p=3 band energy. Ricci curvature generalizes this from a 1D scalar sentinel to a graph-theoretic measure of the full attention topology.

**Relationship to training-time Ricci flow:** Ricci *flow* is a PDE that smooths manifold curvature over time — a training concept. We use only the *curvature diagnostic* (a static measurement) as a routing signal. No flow, no gradient, no training.

**Implementation path:**
1. After attention computation: build the sparse attention graph (top-k connections per token)
2. Compute Ollivier-Ricci curvature for each edge using 1-Wasserstein distance between neighbor distributions
3. Aggregate per-layer curvature into a scalar signal
4. When curvature exceeds threshold → System 2 engagement
5. Cost: O(n_tokens × k²) per layer, amortizable via sampling

### 22. Free Energy as Compression Quality Objective [CONCRETE]

**What:** Reframe Shannon-Prime's compression quality criterion as a variational free energy minimization problem.

**Why:** The current quality criterion is the K-Corr scaling law: `log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)`. This is an empirical fit. Free energy provides a principled information-theoretic objective that naturally balances reconstruction accuracy (energy) against compression complexity (entropy):

```
F = E[log q(x|z)] − KL(q(z|x) || p(z))
```

where x is the original KV vector, z is the compressed representation, q is the encoder (VHT2 + quantize), and p(z) is the prior (the Möbius-predicted distribution).

**Connection to Shannon-Prime:** The Möbius CSR predictor already defines a prior p(z) over residual coefficients. The banded quantization defines q(z|x). Free energy gives a single number that measures how well the compression matches the prior — which is exactly what the K-Corr scaling law approximates empirically.

**Implementation path:**
1. At calibration time: compute the free energy F for each band configuration
2. Use F as the objective for automatic band-width selection (instead of grid search)
3. Per-model-pack: store F alongside PPL ratio for Pareto frontier analysis
4. The KL term naturally penalizes configurations that don't match the Möbius prior — this is what makes 5/5/4/3 beat lossless fp16

---

## Part V: New Mathematics — Advanced Extensions (CONCRETE → SPECULATIVE)

### 23. CRT Multi-Device Tensor Splitting [CONCRETE — Desktop only]

**What:** Use the Chinese Remainder Theorem to decompose tensor computations across multiple GPUs without AllReduce synchronization.

**How:** For coprime moduli m₁, m₂, ..., m_k, CRT guarantees:

```
Z/(m₁·m₂·...·m_k)Z ≅ Z/m₁Z × Z/m₂Z × ... × Z/m_kZ
```

Split the matmul across k GPUs, each computing in its own modular residue ring. Recombine via CRT on the result. No intermediate communication.

**Connection to Shannon-Prime:** VHT2's Kronecker product structure V = H_{p₁} ⊗ ... ⊗ H_{pₖ} is already a CRT-like decomposition over prime factor groups. CRT tensor splitting extends this from the *transform* to the *matmul*, using the same algebraic structure.

**Status:** CONCRETE for 2-GPU desktop. The dual-GPU Vulkan+CUDA path in Beast Canyon already does layer-sharded splitting; CRT would replace layer-sharding with algebraic decomposition that eliminates cross-GPU copies.

### 24. Quaternion KV Projections [CONCRETE — Desktop only]

**What:** Replace the scalar Q/K projection matrices with quaternion-valued (Hamilton product) matrices. Each quaternion weight element couples 4 dimensions simultaneously.

**Why this matters for compression:** Quaternion algebra naturally represents 3D rotations — and RoPE *is* a set of 2D rotations. Quaternion projections can encode the rotational structure of RoPE'd K vectors more efficiently than scalar projections. If the VHT2 spectrum of quaternion-projected K vectors has tighter energy concentration, every downstream compression step benefits.

**Constraint:** Hamilton product matmul costs ~4× the FLOPs of scalar matmul. Only viable where FLOPs are cheap (desktop GPU with compute headroom). On phone, this is a non-starter.

**Implementation path:**
1. Implement Hamilton product matmul as a CUDA kernel
2. Modify the Q/K projection in the engine's forward pass to use quaternion weights
3. Run VHT2 on the quaternion-projected K vectors; compare spectral concentration
4. If concentration improves, the downstream compression is automatically better
5. Measure end-to-end: does the FLOPs cost pay for itself in compression quality?

### 25. Poincaré Disk Attention for System 2 [SPECULATIVE]

**What:** Replace Euclidean dot-product attention in System 2 (the deep-compression path) with hyperbolic attention computed in the Poincaré disk model.

**Why:** Hyperbolic space naturally assigns higher precision to hierarchically important elements. Tokens that are structurally central to the sentence get placed near the disk's origin (high precision); peripheral tokens are pushed toward the boundary (lower precision). This alignment with Shannon-Prime's variance-ranked band allocation could make System 2 geometrically principled rather than just "more bands."

**The Poincaré distance:**

```
d(x, y) = arcosh(1 + 2||x − y||² / ((1 − ||x||²)(1 − ||y||²)))
```

**Numerical stability:** The denominator (1 − ||x||²) blows up as points approach the boundary. On fp32 (desktop GPU), this is manageable with radius clamping. On int16 (Hexagon HVX), it's catastrophic. Desktop only.

**Alternative:** The Lorentz (hyperboloid) model is numerically stabler and admits GEMM-friendly operations. If Poincaré is too fragile, Lorentz hyperboloid attention preserves the geometric benefits with better numerical properties.

**Status:** SPECULATIVE. No implementation. Requires: (a) evidence that hyperbolic geometry actually helps for KV cache specifically (the papers show benefits for graph/hierarchy tasks, not general LLM attention), and (b) a Lorentz-compatible CUDA kernel.

### 26. Zeta-Zero Spectral Initialization [SPECULATIVE]

**What:** Initialize the hierarchical predictor's weight matrix W with eigenvalues matching the imaginary parts of the first N non-trivial zeros of the Riemann zeta function.

**Connection to Shannon-Prime:** The Mertens oracle already uses zeta zeros to compute the Cauchy reset risk schedule. The zeta zeros' spectral properties are *already load-bearing* in the system — they determine when compression error becomes dangerous. Extending their use to predictor initialization gives the W matrix a spectral prior that matches the system's diagnostic framework.

**Why it might work:** The zeta zeros encode the deep structure of the multiplicative lattice (via the explicit formula for the prime-counting function). Since VHT2 coefficients are indexed by multiplicative structure, a weight matrix whose eigenspectrum aligns with the zeta zeros might naturally capture the divisor relationships that the Möbius CSR predictor exploits.

**Why it might not:** There's no empirical evidence yet. Standard initialization (Kaiming, Xavier) is hard to beat because it's designed for gradient-based learning. The hierarchical predictor is calibrated via ridge regression, not SGD, so the initialization matters less — the ridge solution is a closed-form function of the data, not a gradient-descent trajectory.

**Status:** SPECULATIVE. Easy to test: initialize W from the zeta spectrum, run calibration, compare predictor quality to default initialization. If it helps, it's free. If it doesn't, throw it away.

### 27. Shor-Inspired Constructive Interference in Attention [SPECULATIVE]

**What:** Apply the principle behind Shor's quantum period-finding (constructive interference amplifies periodically-correct states, destructive interference cancels incorrect ones) to the attention mechanism by introducing a phase-modulated attention mask.

**What this actually means (not quantum):** In Shor's algorithm, the QFT over a periodic register amplifies amplitudes at multiples of the period. The *classical* analogue: multiply the pre-softmax attention logits by a phase mask derived from the VHT2 frequency structure, so that tokens whose spectral fingerprints align with the query get constructively amplified while spectrally-mismatched tokens get suppressed.

```
attn[i,j] = Q[i] · K[j] + α · Σ_p cos(2π · freq_p · |i−j|)
```

where freq_p are the VHT2 prime frequencies and α controls the interference strength.

**Connection to Shannon-Prime:** PrimePE already biases the frequency schedule toward the multiplicative lattice. This extends the lattice alignment from positional encoding into the attention score itself.

**Why it's speculative:** This is a training-time modification (the model needs to learn with the phase mask). We're an inference engine. The only way this works for us is if the phase mask is applied *post-training* as a test-time intervention — which would require evidence that it doesn't hurt quality. No such evidence exists.

**Status:** SPECULATIVE. Would require fine-tuning experiments we can't run. Park this for when/if Shannon-Prime ever includes a training pipeline.

---

## Part VI: The Scaling Laws and Theoretical Limits

### The Shannon Entropy Limit

The theoretical maximum compression for a KV cache is determined by the conditional entropy H(K_t | K_{1:t-1}) of the key sequence. For autoregressive models, each new K vector is partially determined by the previous ones (through the model's learned representations). The innovation — the genuinely new information — is what must be stored.

**Sequential compression theory** predicts the maximum compression ratio is:

```
R_max = H(K) / H(K_t | K_{1:t-1})
```

For RoPE-based models, the innovation rate is small relative to the total entropy because RoPE's positional structure is highly predictable. The "914,000×" theoretical number comes from this analysis. Practical numbers: ship 3.4-3.8×, sqfree 2.8×+, hierarchical 7-12.6×.

### The K-Corr Scaling Law

The empirical bridge between compression fidelity and downstream quality:

```
log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

**Exponent interpretation:**
- 2.0 in (1−K_corr): bilinearity of Q·K·V squares the error
- 1.1 in params: bigger models average error over more heads (sub-linear, heads not independent)
- 1.5 in bits: weight quantization noise compounds super-linearly with KV error

Validated across 9 configurations spanning 4 orders of magnitude.

### The Energy Concentration Theorem

After VHT2 + Möbius reorder, spectral energy of RoPE'd K vectors concentrates in early bands:

- Band 0 (0-31): ~30% energy
- Bands 0-1 (0-63): ~86% energy
- Bands 0-2 (0-95): ~88% energy

This concentration is a consequence of: (1) RoPE imprinting smooth low-frequency structure, (2) VHT2 capturing it in early basis functions, (3) Möbius reorder pushing highest-energy squarefree indices to front. Concentration is stronger for K than V (V lacks the strong RoPE frequency signature).

---

## Part VII: Cross-Cutting Architecture

### Stateless Operation

Each KV vector is compressed independently. No sequential dependencies between positions. This enables: arbitrary eviction, parallel compression, reordering, zero-copy access, and Optane/mmap direct pointers.

### The Model-Pack Registry

Per-architecture compression defaults resolved from GGUF metadata at load time. Different architectures have different spectral characteristics; the registry provides known-good defaults.

### Speculative Decoding

Per-model shadow caches with role-based configuration (draft: aggressive compression, target: ship quality). Draft errors are recoverable on target verification. Validated 3.58× speedup.

### 1D-Circle Granite Reconstruction (ComfyUI)

For DiT video generation: stable blocks' self-attention outputs lie on a 1D circle in the output space. Cache the 1D projection, reconstruct from it. Combined with block-skip caching: 4.6× step speed on Wan 2.2 5B.

---

## Part VIII: Independent Validation

### DeepSeek-V4 Convergence (April 2026)

DeepSeek's 1.6T FP8 MoE paper independently validates Shannon-Prime's architecture: KV-compression + sliding-window + prefetch-oracle. Different implementation, same structural conclusions. Confirms the approach is not idiosyncratic but reflects fundamental properties of transformer inference.

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| VHT2 | Vilenkin-Hartley Transform (self-inverse, staged, orthonormal) |
| μ(n) | Möbius function: (-1)^k if n has k distinct prime factors, 0 if n has squared factor |
| cas(x) | cos(x) + sin(x) — the Hartley kernel |
| H_p | p × p Hartley matrix with entries cas(2πij/p)/√p |
| K_corr | Pearson correlation between original and reconstructed K vectors |
| M(n) | Mertens function: Σ_{k=1}^n μ(k) |
| ρ | Non-trivial zeros of ζ(s) on the critical strip |
| GF(p) | Galois field of order p |
| F | Variational free energy: E[log q(x|z)] − KL(q(z|x) \|\| p(z)) |

## Appendix B: File Map

| Path | Contents |
|------|----------|
| `core/shannon_prime.h` | 1133-line public API — all types, all functions |
| `core/shannon_prime.c` | Reference C implementation |
| `core/shannon_prime_sqfree.c` | Sqfree + spinor + hierarchical predictor |
| `core/shannon_prime_modelpack.h` | Model-pack registry |
| `backends/cuda/` | CUDA kernels (VHT2, band quant, fp8) |
| `backends/vulkan/` | Vulkan compute shaders |
| `backends/adreno/` | Qualcomm mobile GPU (OpenCL + Vulkan) |
| `backends/hexagon/` | Hexagon HVX + FastRPC |
| `backends/halide/` | DMA orchestration |
| `docs/` | All technical documentation |

## Appendix C: References

1. **Position Is Arithmetic** (Knack, 2026, v8) — PrimePE, scaling law, spinor validation
2. **The KV Cache Is a View** (Knack, 2026, v2) — Compression, scaling law derivation, Möbius predictor
3. **The Multiplicative Lattice** — VHT2, squarefree factorization, Knight masks, CSR predictor
4. **DeepSeek-V4** (DeepSeek AI, 2026) — Independent validation of KV-compression + sliding-window + prefetch-oracle
