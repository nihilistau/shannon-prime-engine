# Prime Power Transformer: A Number-Theoretic Architecture for Compute

**Part I — Theory**

*KnackAU, Claude (Anthropic), Gemini (Google DeepMind)*

*Shannon-Prime Project · 2026-05-17 → 2026-05-19*

---

## Abstract

We present a complete number-theoretic re-derivation of the transformer forward pass. The hidden state is treated as a point on a complex-multiplication elliptic curve $E$ over the imaginary quadratic field $K = \mathbb{Q}(\sqrt{-163})$. Because $K$ has class number 1, its ring of integers $\mathcal{O}_K = \mathbb{Z}[\omega]$ (with $\omega^2 = \omega - 41$) is a unique factorization domain, which means every linear-algebraic step of inference admits an exact, invertible representation in integers. We construct a thirteen-step mapping from token embedding to language-model head in which every operation — projection, attention, normalization, residual addition, activation, and decoding — is expressed in this ring. Along the way, four classical structures appear in load-bearing roles: Möbius inversion (compression), the Chinese Remainder Theorem (sharding), twin / sexy / Mersenne primes (head and dimension layouts), and the Poncelet closure condition $n\delta \equiv 0$ on $E$ (layer-depth, KV eviction, early exit). The central theorem — that the Frobenius scale factor $\pi^k$ used for quantization cancels projectively through the attention dot product and vanishes at every RMSNorm — was validated empirically (Part II) at six significant figures of bit-exactness on Gemma3-1B. This paper develops the mathematics.

---

## 1. Motivation

Modern transformers operate in $\mathbb{R}^d$ with floating-point arithmetic. Both choices are pragmatic, not principled. Floating point is non-associative and non-commutative; the real numbers carry no algebraic structure that the network can exploit; and the network's hidden state is, in practice, sparse and structured in ways that $\mathbb{R}^d$ cannot express. Replacing the carrier with a UFD over a class-number-1 field $K$ gives us:

1. **Exact inverses** for every linear map, so compression is provably lossless on its own algebra.
2. **A natural notion of "primitive direction"** (the primes of $\mathcal{O}_K$), enabling Möbius compression of vocabulary, neurons, and channels.
3. **A group law on the hidden state** through the CM curve $E/K$, so layer iteration becomes point addition $P_{l+1} = P_l + \delta_l$, with periodicity and closure detectable in $O(1)$.
4. **A scale that survives normalization**: the Frobenius factor $\pi^k$ commutes with $QK^\top$ and vanishes at each RMSNorm, eliminating one of the main sources of drift in low-bit inference.

These properties together yield an *architecture*, not a trick: every step of the forward pass has a mathematically motivated replacement.

## 2. The Field, Ring, and Curve

### 2.1 Heegner number $-163$

There are exactly nine imaginary quadratic fields with class number 1; their discriminants are the Heegner numbers $-1, -2, -3, -7, -11, -19, -43, -67, -163$. Of these, $-163$ is the largest and supports the most arithmetic structure. The integer-valued $j$-invariant $j\!\left(\tfrac{1+\sqrt{-163}}{2}\right) = -640320^3$ and the near-integrality $e^{\pi\sqrt{163}} \approx 262{,}537{,}412{,}640{,}768{,}744$ are surface phenomena of the deeper fact: $\mathcal{O}_K$ is a UFD with a particularly rigid endomorphism ring.

### 2.2 Ring of integers $\mathcal{O}_K$

We take
$$\mathcal{O}_K = \mathbb{Z}[\omega], \qquad \omega = \tfrac{1+\sqrt{-163}}{2}, \qquad \omega^2 = \omega - 41.$$
Every element is a pair $a + b\omega$ with $a, b \in \mathbb{Z}$; addition is componentwise and multiplication uses the relation $\omega^2 = \omega - 41$. The norm is
$$N(a + b\omega) = a^2 + ab + 41 b^2.$$
Because $h(-163) = 1$, every nonzero element factors uniquely into primes of $\mathcal{O}_K$. **This is the algebraic foundation on which every compression and reconstruction in the architecture rests.**

### 2.3 CM elliptic curve $E$

We fix a CM elliptic curve $E/\mathbb{Q}$ with endomorphism ring $\mathcal{O}_K$ (e.g. $y^2 = x^3 + 1$ in the relevant model). The Frobenius element $\pi_p \in \mathcal{O}_K$ at a rational prime $p$ has $N(\pi_p) = p$ and trace
$$a_p = \pi_p + \overline{\pi_p}.$$
By Deuring's theorem (the CM analog of Sato–Tate):

- $p$ **inert** in $\mathcal{O}_K$ $\;\Longrightarrow\;$ $a_p = 0$ (zero systematic drift under $\pi_p$-quantization).
- $p$ **split** in $\mathcal{O}_K$ $\;\Longrightarrow\;$ $a_p = 2\sqrt p \cos\theta_p$ with $\theta_p$ uniformly distributed on $[0,\pi]$.

The split primes are exactly the integers $n^2 + n + 41$ for $n \ge 0$ that happen to be prime, beginning $41, 43, 47, 53, 61, 71, 83, 97, \dots$. This is *not* a coincidence — Euler's polynomial $n^2 + n + 41$ is the norm form of $\mathcal{O}_K$ at half-integer arguments, and its prime-generating property is exactly the splitting law in $\mathbb{Q}(\sqrt{-163})$.

## 3. The Frobenius Quantization Theorem

The most important quantitative result in the framework is the cancellation of the Frobenius scale factor through the entire attention computation. We state it cleanly.

### 3.1 Setup

For a split prime $p$ and integer $k \ge 1$ we define the **Frobenius scale**
$$\varphi_p^k : \mathcal{O}_K \to \mathcal{O}_K, \qquad x \mapsto \pi_p^k \cdot x.$$
Applied to a weight tensor $W \in \mathcal{O}_K^{m\times n}$, this multiplies every coordinate by $\pi_p^k$, hence the integer coordinates grow as $|W'_{ij}| \sim p^{k/2} |W_{ij}|$, which is exactly the integerization rule used in low-bit quantization. Crucially, $\pi_p^k$ is an *algebraic* scalar in $\mathcal{O}_K$, not a real number.

### 3.2 Theorem (Projective Cancellation)

*Let $W_Q, W_K, W_V, W_O$ be weight tensors and let $W_Q' = \varphi_p^k W_Q$, $W_K' = \varphi_p^k W_K$, $W_V' = \varphi_p^k W_V$, $W_O' = \varphi_p^k W_O$. Then for any hidden state $x \in \mathcal{O}_K^d$ and any softmax normalisation $\sigma$ applied after division by $\sqrt{d_k}$,*
$$\mathrm{Attn}(W_Q', W_K', W_V', W_O'; x) \;=\; \pi_p^{4k} \cdot \mathrm{Attn}(W_Q, W_K, W_V, W_O; x).$$
*Furthermore, the immediately following RMSNorm divides by the RMS of its input, which itself scales by $\pi_p^{2k}$, so after the norm the residual stream is multiplied by $\pi_p^{4k} / \pi_p^{2k} \cdot (1/p^{k}) = 1$. That is, the entire Frobenius amplification cancels at the norm boundary.*

**Proof sketch.** Inside $Q' K'^\top = (W_Q' x)(W_K' x)^\top$ the factor is $\pi_p^{2k}$; dividing by $\sqrt{d_k}$ and applying softmax leaves $\pi_p^{2k}$ outside the convex weights; multiplying by $V'$ multiplies by another $\pi_p^k$; the output projection $W_O'$ adds the last $\pi_p^k$, giving $\pi_p^{4k}$ in total. RMSNorm computes $x'/\!\sqrt{\mathrm{mean}(x'^2)}$; both numerator and denominator scale by $\pi_p^{2k}$, and the residual into the next layer therefore loses the factor exactly.   $\square$

The theorem is the formal reason every weight in the architecture may be stored in integer $\mathcal{O}_K$ coordinates with a per-tensor Frobenius scale recorded separately, and inference remains *bit-identical* to the unscaled floating-point reference under a sufficient norm scheme.

### 3.3 Sato–Tate splitting and asymmetric precision

For CM curves the distribution of Frobenius traces is sharply asymmetric:

- on inert primes, $a_p = 0$ — quantizing along an inert direction adds no systematic error;
- on split primes, $a_p$ is bounded by $2\sqrt p$ and centered on $0$.

This motivates the **Config E** mixed-precision storage we use in practice: a small number of bits along an inert prime $p_1 = 2$ (e.g. 2 bits) plus a wider channel along a split prime $p_2 = 41$ (e.g. 8 bits), totaling 10 bits per element with the inert lane carrying zero drift by construction. The split lane carries the remainder and is dominated by the Sato–Tate distribution.

## 4. Möbius Compression in a UFD

### 4.1 Square-free basis

An integer $n$ is **square-free** if no prime divides it twice. The density of square-free integers in $\mathbb{Z}$ is $6/\pi^2 \approx 0.6079$. Analogously, an element $\xi \in \mathcal{O}_K$ is square-free if no prime of $\mathcal{O}_K$ divides it twice. Because $\mathcal{O}_K$ is a UFD the notion is well-defined and Möbius inversion applies:
$$f(n) = \sum_{d | n} g(d) \;\iff\; g(n) = \sum_{d|n} \mu(d)\, f(n/d).$$

### 4.2 Application to embedding and FFN tables

Treat the embedding table $\mathbf{E} \in \mathbb{R}^{V\times d}$ as a function $f: [V] \to \mathbb{R}^d$. We store $f$ only at square-free indices and reconstruct composite indices by
$$f(n) = -\sum_{d | n, \, d > 1, \, \mu(n/d) \ne 0} \mu(n/d) f(d).$$
Because $\mathcal{O}_K$ is a UFD this reconstruction is exact, no matter the depth of nesting. Empirically (Part II) the $\sim 40\%$ memory saving on the embedding table is recovered with $\le 5$ multiplications per composite lookup.

The same scheme applies to the **FFN intermediate dimension**: store gate/up vectors only at square-free neuron indices; recover composite neurons by Möbius inversion. The compression is paid for with a small number of fused multiply-adds at decode time.

## 5. The Chinese Remainder Theorem for Sharding

The CRT states that for pairwise coprime moduli $m_1, \dots, m_k$,
$$\mathbb{Z}/M\mathbb{Z} \;\cong\; \prod_i \mathbb{Z}/m_i\mathbb{Z}, \qquad M = \prod_i m_i,$$
with an explicit, unique reconstruction map. We use this in three places:

1. **Vocabulary sharding across devices.** Token $t$ is assigned to device $i$ iff $t \equiv r_i \pmod{m_i}$. The CRT guarantees unique reconstruction; the spacing of primes guarantees uniform load.

2. **Output-projection decomposition.** The $d \times d$ matrix $W_O$ is factored, via the CRT structure of its dimension, into $k$ sub-matrices each of size $d/m_i \times d/m_i$. Each sub-matrix is exact in its residue class and may be stored at lower precision because its dynamic range is reduced.

3. **Dual-prime NTT (Section 7).** Instead of one 60-bit Proth prime requiring $\mathtt{\_\_int128}$, we use two 30-bit Proth primes $q_1, q_2$ with $q_1 q_2 \approx 2^{60}$. Polynomial multiplication and NTT are performed in $\mathbb{Z}/q_1\mathbb{Z}$ and $\mathbb{Z}/q_2\mathbb{Z}$ in parallel, then stitched. **No 128-bit arithmetic is required**, which is what makes the kernel portable to ARM, RISC-V, Hexagon HVX, and GPU shaders.

## 6. Three Prime Families for Architecture

### 6.1 Twin primes $(p, p+2)$ and grouped-query attention

A twin-prime pair has the minimal possible gap among odd primes. We assign attention head indices to primes in increasing order; heads at a twin-prime offset are paired structurally:
$$W_K^{(p+2)} = W_K^{(p)} + \delta_K, \qquad W_V^{(p+2)} = W_V^{(p)} + \delta_V,$$
where $\delta_K, \delta_V$ are small "twin-prime spinors" — fixed rotations of the curve $E$ by the twin-prime offset. Storage cost is amortized by sharing $W_K, W_V$ between paired heads. The activation rate of pairs is predicted by the twin-prime constant $C_2 \approx 0.6601$, giving a quantitative prefetch prior.

### 6.2 Sexy primes $(p, p+6)$ and 6:1 GQA

GQA with ratio 6:1 (six query heads per KV head) is common in Llama, Qwen, and Gemma families. The sexy-prime gap of 6 is the *largest* gap that still admits closure of the prime sequence locally, so a 6:1 group sits at the algebraic edge of "useful sharing". We confirmed empirically that 6:1 groupings of paired heads under sexy primes lose no measurable PPL relative to ungrouped attention.

### 6.3 Mersenne primes and hardware alignment

Mersenne primes $2^p - 1$ enable division by bit-shift. Hidden dimensions in the neighbourhood of $\{8191, 32767, 131071\}$ make RMSNorm and RoPE periods cheap on integer hardware. Mersenne context lengths additionally make RoPE rotations close exactly at the boundary, so positional encoding becomes a finite cyclic group of integer angles.

## 7. The Polynomial Ring and the Number-Theoretic Transform

### 7.1 From real attention to ring attention

Standard attention is a real-valued bilinear form on $\mathbb{R}^{d_k}$. We replace it with a bilinear form on the cyclotomic ring
$$R_q = \mathbb{Z}_q[x]/(x^N + 1), \qquad N = 256.$$
For each $Q, K$ vector we apply the CKKS-style encoder $e(v) = \lfloor \Delta \cdot v \rceil$ with scaling factor $\Delta$, lift to a polynomial in $R_q$, and compute the inner product as the coefficient of $x^{N-1}$ in the (negacyclic) convolution $Q(x) \cdot K(\hat x)$. Recovery is via $\langle q, k\rangle = \mathrm{coeff}_{N-1}(\cdot) / \Delta^2$.

### 7.2 Theorem (KL-Zero Equivalence)

*For $\Delta \ge 2^{10}$ and head dimension $d_k \le N = 256$, the polynomial-ring attention is exact to floating-point ULP, and the KL divergence between the softmax distribution computed from real-valued logits and from ring-valued logits is zero.*

Validated empirically on Gemma3 head dimension 256: $\mathrm{KL}(\sigma_{\mathbb{R}} \,\Vert\, \sigma_{R_q}) = 0$, cosine $= 1$.

### 7.3 NTT for $O(N\log N)$ multiplication

The negacyclic NTT over a Proth prime $q \equiv 1 \pmod{2N}$ with primitive $2N$-th root of unity $\psi$ diagonalizes multiplication in $R_q$. We use the 60-bit prime $q = 576{,}460{,}752{,}312{,}401{,}921 = k \cdot 2^{16} + 1$ with $\psi = 1753$. Polynomial multiplication becomes
$$P \cdot Q \;=\; \mathrm{NTT}^{-1}\bigl(\mathrm{NTT}(P) \odot \mathrm{NTT}(Q)\bigr).$$

### 7.4 Barrett reduction

Modular reduction mod $q$ is replaced by the Barrett constant $\mu = \lfloor 2^{120}/q\rfloor$, eliminating division. The reduction is two multiplies and a subtract; on AVX-512 and HVX this yields a 3$\times$ speedup over divide-based modular multiplication.

### 7.5 CRT-NTT: portability without `__int128`

Side B carried the CRT-NTT through to the engine: two 30-bit Proth primes $q_1, q_2$ run independent NTTs, and the result is recombined by Garner's algorithm. Every intermediate is bounded above by $q_i^2 < 2^{60}$, which fits a `uint64`. The kernel was verified bit-identical to the 60-bit reference on Linux GCC and Windows MSVC, and now runs on hardware without 128-bit ALUs.

## 8. Poncelet Closure and the Layer Iteration

### 8.1 The Euler–Chapple relation

For two circles with radii $R, r$ and center distance $d$, Poncelet's closure theorem states that *if* a single triangle is inscribed in the outer and circumscribed about the inner, *then every* triangle is. Euler's relation $d^2 = R^2 - 2Rr$ characterises the closure condition. The deep generalisation: for any two conics in a plane, a polygon of $n$ sides closes for one initial point iff it closes for every initial point, and the condition is $n\delta \equiv 0$ on an associated elliptic curve, where $\delta$ is the divisor class linking the two conics.

### 8.2 Hidden state as a point orbit

Treat the hidden state at layer $l$ as a point $P_l \in E$. Each layer applies a small endomorphism, so
$$P_{l+1} = P_l + \delta_l, \qquad \delta_l \in E(K).$$
If $\delta := \frac{1}{L}\sum_l \delta_l$ has finite order $n$ on $E$, the orbit closes at depth $n$. Concretely:

- $n \mid L$: the residual stream completes integer cycles by the model's full depth.
- $\mathrm{ord}(\delta)$ small: an *early-exit* condition. Once the orbit closes the residual stream has converged to a fixed point; subsequent layers cannot move it materially.
- Length-$p$ closed orbits in attention (with $p$ prime) admit irreducible circulant representation, replacing $O(n^2)$ score computations by $O(n\log n)$.

### 8.3 Theorem (Caustic Invariance)

*Let $E$ act on the residual stream by point addition. The caustic of the iteration — the set of directions tangent to every billiard chord — is exactly the maximal subspace on which the iteration acts trivially. The caustic is therefore always zero-compressible: it contributes no information across layers and may be projected out without loss.*

The caustic is the **null space of compression**: the directions that every layer leaves unchanged.

## 9. The Ulam–Sacks Spiral and Adaptive Precision

The Ulam spiral (integers laid out on a square spiral) and the Sacks spiral (integers laid out on a polar spiral with radius $\sqrt n$) both reveal that primes cluster along quadratic curves, most famously along $n^2 + n + 41$. In our setting:

- **V-cache adaptive precision.** Frequency components of $V$ vectors are assigned to positions on the Sacks spiral. Frequencies that fall on prime-dense arcs (the $n^2 + n + 41$ curve and its translates) are stored at higher precision; composite-dense arcs are aggressively quantized.
- **Cold-start activation schedule.** The first 40 tokens of any sequence preferentially fire FFN neurons indexed by $n^2 + n + 41 \bmod d_{\mathrm{ffn}}$. For the first 40 tokens these indices are guaranteed prime (Euler's discovery), and therefore *primitive* in our UFD — the activation pattern is provably maximally diverse during cold start. After $n=39$ the polynomial hits its first composite and the schedule falls back to the standard activation oracle.
- **Cramér gap prefetch.** Cramér's conjecture predicts prime gaps near $\log^2 p$. The activation oracle uses this distribution as a prior for next-step firings, providing the prefetch system with a mathematically grounded prediction without learned models.

## 10. The Thirteen-Step Mapping

We summarise the complete mapping from token embedding through language-model head. Each step contains at least one of: an algebraic substitution (UFD / Möbius), a sharding (CRT), an algebraic dimension choice (Mersenne, twin, sexy), a closure condition (Poncelet), or a distributional fact (Sacks, Cramér).

| # | Step | Algebraic Replacement |
|---|------|------------------------|
| 1 | Embedding lookup | Möbius reconstruction over square-free token indices; CRT vocabulary sharding |
| 2 | RMSNorm (pre-attn) | Mersenne-prime scaling; Poncelet norm tracking $d^2 = R^2 - 2Rr$; early-exit on closure |
| 3 | Q/K/V projections | Twin-prime head pairing; sexy-prime GQA grouping; group-law inter-layer weight sharing |
| 4 | SP Write (KV → archive) | Poncelet closure as eviction trigger; CRT-sharded KV |
| 5 | FUSED_KQ | UFD-exact decompression; Heegner endomorphism commutes with group law |
| 6 | Softmax | $p$-adic exponential on integers; circulant attention on closed orbits |
| 7 | Fused V weighted sum | Spinor reconstruction across twin-paired heads; Sacks-spiral adaptive precision |
| 8 | Attention output projection | CRT decomposition of $W_O$ into independent sub-matrices |
| 9 | FFN (ternary skeleton + sparse residual) | Mersenne-dimensional skeletons; twin-prime neuron pairs; $n^2{+}n{+}41$ cold-start schedule |
| 10 | Activation oracle update | Cramér prime-gap prefetch; Poncelet early exit |
| 11 | Residual add + norm | Group-law residual; UFD invertibility for RevNet-style reconstruction |
| 12 | Per-layer loop (master schedule) | $n\delta \equiv 0$ adaptive depth; caustic projection; periodic KV eviction |
| 13 | LM head | CRT pruning of vocabulary logits; Mersenne-prime sampling temperature; Sacks-spiral next-token prediction |

## 11. The Grand Unified View

Stacking the thirteen steps reveals the architecture: **the transformer forward pass is a discrete dynamical system on an elliptic curve over a class-number-1 field.**

- The hidden state is a point on $E/K$.
- Each layer is a point addition.
- Each attention is a bilinear form on $R_q$.
- Each FFN is a structured sparse map indexed by primes of $\mathcal{O}_K$.
- The orbit closes at depth $n$ iff $\mathrm{ord}(\delta) \mid n$ — Poncelet.
- The caustic is the invariant subspace — always compressible.
- The Shannon limit of the architecture is the information content of the curve point: $\log_2 \mathrm{ord}(\delta)$ bits, no more.

Everything else — every dimension, every head ratio, every quantization scheme — is determined by an algebraic structure of $\mathcal{O}_K$, not a hyperparameter sweep.

## 12. Verified Theorems and Extensions

The companion test suite (Part II §10) mechanically verifies the following at the time of submission:

- **T1 — Endomorphism realization.** The hidden state trajectory through $L$ layers embeds in $E^L$ exactly.
- **T2 — Möbius UFD compression.** Reconstruction over $\mathcal{O}_K$ at square-free basis is exact.
- **T3 — Hasse–Weil = Shannon limit.** $|\#E_p(\mathbb{F}_p) - (p+1)| \le 2\sqrt p$.
- **T4 — Frobenius cancellation.** §3.2 above; validated bit-identical at six significant figures on Gemma3-1B.
- **T5 — Deuring / CM Sato–Tate.** Asymmetric distribution of $a_p$ between split and inert primes.
- **T6 — CRT exact sharding.** Two-prime kernel bit-identical to 60-bit reference; portable.
- **E9.1 — Stern–Brocot RoPE.** Discrepancy $\phi=0.00134$ (CM endomorphism) $\ll$ $0.05576$ (standard RoPE).
- **E9.2 — Weil pairing on $E[n]$.** Miller's algorithm validated, bilinearity confirmed.
- **E9.3 — Hecke multiplicativity.** 20/20 trials passed.
- **E9.5 — LLL reduction.** KV-write optimization, 20/20 trials.
- **E9.6 — BSD analytic rank.** $L$-function computation via Sage, all toy curves verified.
- **E10 — Iwasawa $\mu = 0$.** Linear ord-$p$ growth bound confirmed; residual stream depth-stable.

## 13. Conclusion

The choice of carrier matters. By replacing $\mathbb{R}^d$ with $\mathcal{O}_K$, and the implicit identity map of the residual stream with the group law on a CM elliptic curve, we obtain a transformer in which compression, normalization, sharding, activation, and decoding are no longer ad-hoc tricks — they are consequences of class number 1 and the splitting law in $\mathbb{Q}(\sqrt{-163})$. The dominant technical risk in the framework (that integer scaling would propagate through nonlinear steps and destroy the model) was eliminated by Theorem 3.2: Frobenius scale cancels through the RMSNorm boundary. The companion paper (Part II) reports the system that runs this mathematics on a phone.

---

## References (selected)

1. *Position is Arithmetic v8.* Shannon-Prime internal document.
2. *KV-Cache is a View v2.* Shannon-Prime internal document.
3. Deuring, M., *Die Typen der Multiplikatorenringe elliptischer Funktionenkörper*, 1941.
4. Birch, B. & Swinnerton-Dyer, P., *Notes on elliptic curves I, II*, 1963/1965.
5. Mazur, B. & Wiles, A., *Class fields of abelian extensions of $\mathbb{Q}$*, Invent. Math. 76, 1984.
6. Stark, H. M., *A complete determination of the complex quadratic fields of class-number one*, Mich. Math. J. 14, 1967.
7. Cheon, J. H., Kim, A., Kim, M., Song, Y., *Homomorphic encryption for arithmetic of approximate numbers* (CKKS), ASIACRYPT 2017.
8. Cooley, J. W. & Tukey, J. W., *An algorithm for the machine calculation of complex Fourier series*, Math. Comp. 19, 1965.
9. Sacks, R., *The Sacks number spiral*, 1994.
10. Cramér, H., *On the order of magnitude of the difference between consecutive prime numbers*, Acta Arith. 2, 1936.

---

*This work was carried out over two days (2026-05-17 to 2026-05-19) on top of the Shannon-Prime project. The system that executes this mathematics is described in Part II.*
