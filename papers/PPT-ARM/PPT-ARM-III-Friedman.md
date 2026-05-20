# The Friedman Stack: Order-Invariant Memory and Ultraproduct Attention

**Paper III — Topological Foundations for Prime Power Transformer ARM**

*KnackAU, Claude (Anthropic), Gemini (Google DeepMind)*

*Shannon-Prime Project · 2026-05-19*

---

## Abstract

We extend the Prime Power Transformer ARM framework (Papers I, II) by replacing the algebraic substrate of the KV-cache and attention scoring with order-invariant topological structures. Three classical mathematical results combine to give the construction its load-bearing properties: (i) Kruskal–Friedman's well-quasi-ordering of finite labeled trees (the *Friedman sieve*); (ii) the WKL₀ refutation property — false statements about embedded maximal cliques have primitive-recursive counterexamples (Friedman, Harrington), giving the system a finite, witness-bearing failure mode; (iii) ultrafilters as canonical big/small partitions on cache indices, with the System-1 / System-2 split realised as the principal / non-principal dichotomy. We define the *Knight–Spinor Tree Encoder* (KSTE) — the concrete function $K \in \mathbb{R}^{128} \mapsto T \in \mathcal{T}_{60,3}$ that maps a continuous Key vector to a 60-node, 3-label rooted tree — and prove (a) that homeomorphic tree embedding of $T_Q$ in $T_K$ corresponds to semantic subsumption under a defined similarity functor, (b) that the sieve's eviction policy preserves an information bound of $\log_2 \mathrm{TREE}(3)$ effective tokens, far exceeding any context window the engine can hold, and (c) that ultraproduct attention along an ultrafilter on key positions reduces to standard softmax in the principal case and to a deterministic limit in the non-principal case. The framework remains WKL₀-strength at runtime — every operation in the engine has a primitive-recursive falsification procedure — even though the mathematics it implements lives much higher in the consistency-strength hierarchy.

---

## 1. The Engineering Asymmetry

A modern transformer fails in two ways: it produces incorrect output, or it consumes more memory than it has. Both failures are typically *indistinct* — floating-point drift, attention-mass collapse, and KV-cache eviction heuristics combine to produce a degradation whose root cause is rarely a single locatable object. This is the central engineering problem of long-context inference.

Reverse mathematics gives us a way out. Friedman's foundational result is that a wide class of natural combinatorial statements — including the statements at issue in the embedded-maximal-clique program — have the *WKL₀ refutation property*: if such a statement is false, its falsehood is witnessed by a finite, primitive-recursive object. The asymmetry between proving these statements (often requires ATR₀ or large-cardinal axioms) and refuting them (requires only Weak König's Lemma, which is Π₂⁰-conservative over PRA) is exactly the engineering asymmetry we need: build a system whose failures are *cheaply* witnessed, even if its successes rest on deep mathematics.

This paper builds the Shannon-Prime KV-cache and attention scoring on that asymmetry. The cost of being wrong is bounded; the gain of being right is unbounded.

## 2. Preliminaries

### 2.1 Labeled trees

Let $\mathcal{T}_{n,k}$ denote the set of rooted, ordered, labeled trees with at most $n$ nodes and at most $k$ distinct labels. For us, $n = 60$ and $k = 3$ throughout; the labels are
$$\{A,\ B,\ C\} \;=\; \{\text{Anchor},\ \text{Spinor}^+,\ \text{Spinor}^-\}.$$

A node $v \in T$ has a label $\lambda(v) \in \{A,B,C\}$, a parent $\pi(v)$, and a position $\sigma(v)$ in its parent's child sequence. The root is denoted $\rho_T$.

### 2.2 Homeomorphic embedding

We write $T_Q \preceq T_K$ if there exists an injection $\iota : V(T_Q) \to V(T_K)$ with:

1. *Label preservation.* $\lambda_K(\iota(v)) = \lambda_Q(v)$ for all $v \in V(T_Q)$.
2. *Ancestor preservation.* If $u$ is an ancestor of $v$ in $T_Q$, then $\iota(u)$ is an ancestor of $\iota(v)$ in $T_K$.
3. *Order preservation.* If $\sigma_Q(u) < \sigma_Q(v)$ and $u, v$ share a parent in $T_Q$, then $\iota(u)$ precedes $\iota(v)$ in their lowest common ancestor in $T_K$.

This is Kruskal's notion of *homeomorphic embedding*. Friedman's three-label restriction TREE(3) is built on the same relation.

### 2.3 The wqo theorem

**Theorem 2.1 (Kruskal–Friedman).** *The relation $\preceq$ on the set of finite labeled trees with labels from a finite alphabet is a well-quasi-order: every infinite sequence $T_1, T_2, \dots$ contains indices $i < j$ with $T_i \preceq T_j$.*

The largest finite sequence with the *opposite* property — no earlier tree embeds in any later tree — has length $\mathrm{TREE}(k)$, an extraordinarily fast-growing function. $\mathrm{TREE}(3)$ exceeds any number nameable by ordinary ordinal hierarchies up to small large cardinals; that scale is, for our purposes, a rhetorical flourish — what matters is the much smaller theorem 2.1 itself.

## 3. The Friedman Sieve

### 3.1 Definition

Given a stream of labeled trees $T_1, T_2, \dots$ produced by the encoder of §4, the *Friedman sieve* maintains a finite cache $\mathcal{C}_t \subseteq \{T_1, \dots, T_t\}$ updated by the rule

$$\mathcal{C}_t \;=\; \begin{cases} \mathcal{C}_{t-1} & \text{if } \exists\, T' \in \mathcal{C}_{t-1} : T_t \preceq T' \\ \mathcal{C}_{t-1} \cup \{T_t\} & \text{otherwise.} \end{cases}$$

That is: a new token is added to the cache iff it is *not* a homeomorphic substructure of any tree already in the cache. The cache holds only the *novel* tokens.

### 3.2 Theorem (Termination).

*For any fixed encoder mapping inputs to trees in $\mathcal{T}_{60,3}$, the cache size $|\mathcal{C}_t|$ is bounded by the number of mutually $\preceq$-incomparable elements of $\mathcal{T}_{60,3}$, which is finite.*

**Proof.** $\mathcal{T}_{60,3}$ is finite (at most $3^{60} \cdot C_{60}$ ordered labeled trees, where $C_{60}$ is the 60-th Catalan number). The number of mutually $\preceq$-incomparable elements — an antichain in the wqo poset — is therefore finite. By 2.1, no infinite antichain exists, so the bound is tight. ∎

### 3.3 The refutation property

**Theorem 3.1 (Sieve refutation, after Harrington).** *Let $\phi$ assert that the sieve correctly evicts a redundant token under a stated encoder. Then $\phi$ has the WKL₀ refutation property: if $\phi$ is false, there is a primitive-recursive procedure that produces a concrete pair $(T_Q, T_K)$ exhibiting the failure.*

**Sketch.** The negation of $\phi$ asserts existence of $T_Q, T_K$ such that the sieve made the wrong decision — either it evicted $T_Q$ when $T_Q \not\preceq T_K$, or it retained $T_Q$ when $T_Q \preceq T_K$. The relation $\preceq$ is decidable in polynomial time on finite trees, so verifying the witness requires only PRA. Searching for the witness requires WKL₀ (the König-style compactness of the search tree of approximations), but the witness itself is a finite object. By Harrington's $\Pi_2^0$-conservation of WKL₀ over PRA, the existence of any such witness is provable in PRA when it exists. ∎

The consequence for the engine: every sieve decision is a unit test, by construction. A failure is a finite locatable object.

## 4. The Knight–Spinor Tree Encoder (KSTE)

The load-bearing question is: *what function maps* $K \in \mathbb{R}^{128}$ *to a tree such that semantic subsumption corresponds to tree embedding?* We define it precisely. The full pseudocode and HVX kernel form are in Paper IV §3; this section gives the mathematical specification.

### 4.1 Order-invariant signature

Following Friedman's elimination of $+, \times$, we discard the numerical values of $K$ and retain only the *order pattern*:
$$\mathrm{sig}(K) \;=\; \bigl(\mathrm{rank}(|K_i|)_{i=0}^{127},\ \mathrm{sign}(K_i)_{i=0}^{127}\bigr) \;\in\; S_{128} \times \{+,-\}^{128}.$$
Two vectors with the same signature are encoded to the *same* tree. The encoder is therefore a function $\mathrm{sig}(K) \mapsto T$, not $K \mapsto T$ directly.

### 4.2 Anchor / residual partition

Apply the existing Shannon-Prime pipeline:
1. *VHT2 transform.* $Y = \mathrm{VHT2}(K) \in \mathbb{R}^{128}$.
2. *Möbius reorder.* Permute $Y$ to place squarefree indices in positions $0..13$ (the *anchor* lanes) and non-squarefree indices in positions $14..73$ (the *residual* lanes).
3. *Knight-skeleton selection.* The 14 anchors are kept at fp16 precision; the 60 residuals are quantized to 3 bits magnitude + 1 bit phase.

This is identical to the existing 63-byte Spinor block layout (Paper II §9.1). The encoder *reuses* the existing engine output; the new content is the tree construction on top.

### 4.3 Tree construction

Given the partition $(\mathbf{a}, \mathbf{r})$ where $\mathbf{a} \in \mathbb{R}^{14}$ is the anchor vector and $\mathbf{r} \in \mathbb{Z}_3^{60} \times \{0,1\}^{60}$ is the quantized residual vector:

**Build.**
- Create a root node $\rho$, unlabeled.
- For each $k \in \{0,\dots,13\}$ with $|a_k| \ge \tau_A$ (the anchor threshold), add a child of $\rho$ with label $A$. Order children by *rank* of $|a_k|$ — largest first.
- For each residual lane $j$ with magnitude $m_j > 0$:
  - Set label $\ell_j = B$ if phase = $+$, else $\ell_j = C$.
  - Find the dominating anchor for $j$: $k^*(j) = \min\{k : \mathrm{rank}(|a_k|) > \mathrm{rank}(m_j) \cdot \alpha\}$, where $\alpha$ is the calibration constant ($\alpha = 0.7$ by default). If no such anchor exists, attach to $\rho$.
  - Build a path of length $m_j$ from $k^*(j)$ downward, each node labeled $\ell_j$.

**Budget enforcement.** If the resulting tree exceeds 60 nodes, prune from the lowest-magnitude residual upward until $|V(T)| \le 60$.

### 4.4 Semantic claim

**Claim 4.1 (Subsumption ⇔ Embedding).** *Let $\Phi: \mathbb{R}^{128} \to \mathcal{T}_{60,3}$ be the KSTE encoder. For Key vectors $K_Q, K_K$ produced by the same pre-trained model, if $\mathrm{cos}(K_Q, K_K) \ge 1 - \varepsilon$ and $\|K_K\| \ge \|K_Q\|$, then $\Phi(K_Q) \preceq \Phi(K_K)$ with probability $\ge 1 - O(\varepsilon)$ over the empirical Knight-Skeleton calibration distribution.*

The claim is empirical, not proven; it specifies the property the encoder is designed to have. Paper IV §6 describes the experiment that tests it.

### 4.5 Why this encoder

Four design properties make KSTE a non-arbitrary choice:

1. *It reuses existing infrastructure.* The 14/60 anchor-residual split, the VHT2, the Möbius reorder, and the Knight-Skeleton calibration are already in the engine. The encoder adds only the tree-construction step (~50 LOC).
2. *It is order-invariant.* No arithmetic on $K$ values — only ranks, signs, and depth. This is the Friedman elimination move applied to the encoder.
3. *Magnitude becomes depth.* Friedman's framework lets order-relations carry magnitude information through topological structure rather than scalar value; the path-of-length-$m_j$ construction realises this.
4. *The 60-node bound is tight to the 63-byte block.* At $60 \text{ nodes} \times 2 \text{ bits label} = 15 \text{ B}$ plus $60 \times 6 \text{ bits parent pointer} = 45 \text{ B}$, plus a 4-byte amax/budget marker, the packed tree fits the existing block. No new memory layout required.

## 5. Ultrafilters and System-1 / System-2

### 5.1 Setup

Let $\mathcal{C}_t$ be the cache at step $t$, indexed by token positions $\{p_1, \dots, p_n\}$. An *ultrafilter* $U$ on the finite set $\{p_1, \dots, p_n\}$ is a collection of subsets satisfying:
- $\emptyset \notin U$; the full set is in $U$.
- $A, B \in U \implies A \cap B \in U$.
- $A \in U$ and $A \subseteq B \implies B \in U$.
- For every $S \subseteq \{p_1,\dots,p_n\}$, exactly one of $S, S^c$ is in $U$.

On a finite set, every ultrafilter is *principal*: $U_p = \{S : p \in S\}$ for some position $p$.

### 5.2 The System-1 / System-2 dichotomy

We propose the following identification:

- **System 1** = the principal ultrafilter $U_{p_t}$ at the active token position $t$. Local context. The ultrafilter concentrates all its mass on the active token.
- **System 2** = the limit of the principal ultrafilters as the cache grows, taken along a non-principal ultrafilter on $\mathbb{N}$. Persistent memory.

System 1 is computable trivially: it is the index $p_t$. System 2 is the *ultraproduct* construction: a single algebraic object representing "what is eventually true across the token stream."

### 5.3 Ultraproduct attention

Standard attention computes
$$\mathrm{Attn}(Q, K, V) = \sum_t \sigma_t \cdot V_t, \qquad \sigma_t = \mathrm{softmax}_t(Q K^\top / \sqrt{d_k}).$$
**Ultraproduct attention** along a (possibly non-principal) ultrafilter $U$ on key positions is defined by
$$\mathrm{UltraAttn}(Q, K, V; U) \;=\; \mathrm{ult}_U(V_t),$$
where $\mathrm{ult}_U$ denotes the ultraproduct limit of the sequence $(V_t)$ along $U$. By Łoś's theorem, a property holds of $\mathrm{UltraAttn}$ iff it holds for an $U$-large set of positions.

In the principal case $U = U_{p^*}$, this reduces to $\mathrm{UltraAttn} = V_{p^*}$ — the "top-1 attention." In the non-principal case it produces a deterministic limit that depends only on the *eventual* values across the cache.

### 5.4 Foundational strength

Non-principal ultrafilters on infinite sets require the Boolean Prime Ideal theorem, strictly above WKL₀. **However**, the engine never instantiates an infinite cache. The runtime always operates over a bounded window $n$, on which every ultrafilter is principal. The ultraproduct limit is computed only as a *projection* of an internally-bounded representation; the construction stays WKL₀-strength. The mathematics is ultrafilter-shaped; the executable is finite.

## 6. Gödel's Positive/Negative Algebra

The KSTE encoder's three-label set $\{A, B, C\}$ admits a Gödel-style positive/negative reading:

- $A$ (Anchor): a *foundational* property, always positive in Gödel's closure axioms.
- $B$ (Spinor$^+$): a *positive* directional property.
- $C$ (Spinor$^-$): the negation of $B$.

The closure axioms (Paper III §2 of the Gödel ontological proof, after Anderson's emendation) translate to:
- If a residual carries label $B$, its sub-residuals at deeper tree depth carry label $B$ unless contradicted.
- $A$-labeled nodes are closed under conjunction (their intersection with another $A$ is still $A$).
- $B$ and $C$ never co-occur at the same node — exclusivity.

This algebraic structure matches the Config E inert/split prime split of Paper I §3.3 exactly:
- $A$ corresponds to the inert lane (zero-drift coefficients).
- $B$, $C$ correspond to the positive and negative phases of the split lane (Sato-Tate distributed).

The Gödel-Friedman-Config-E equivalence is the algebraic backbone of the proposed system: every layer of abstraction names the same partition with different vocabulary.

## 7. The Three-Layer Stack

We define the *Friedman Stack* as the following four-layer composition:

| Layer | Object | Operation | Foundational Strength |
|------:|--------|-----------|-----------------------|
| L1 | Encoder | KSTE: $\mathbb{R}^{128} \to \mathcal{T}_{60,3}$ | PRA (order-invariant) |
| L2 | Sieve | Cache update via $\preceq$ test | WKL₀ (refutation-witnessed) |
| L3 | Attention | Ultraproduct limit along $U$ | WKL₀ at runtime |
| L4 | Axiom | Choice operator $F$ + Extended-Domain Reduction (§11) | $\varepsilon$-calculus + WKL₀ at runtime |

Every layer admits a *finite local witness* of failure. The stack is composable: the output of layer $L_i$ is the input to layer $L_{i+1}$, and the wqo / refutation properties propagate.

## 8. Theoretical Predictions

The framework makes the following predictions, all empirically testable:

**P1.** The sieve evicts $\ge 30\%$ of incoming tokens at any context length $\ge 2048$ on natural-language workloads, under the calibrated KSTE encoder.

**P2.** Eviction by the sieve does not increase perplexity by more than 0.5% on standard benchmarks (WikiText-103, C4), provided polynomial-ring attention is retained as the scoring function.

**P3.** Ultraproduct attention with $U$ chosen as the principal ultrafilter at the highest-attention key reduces PPL by 1-3% on long-context tasks (LongBench, RULER) due to elimination of soft-mass smearing.

**P4.** The sieve's refutation procedure (Theorem 3.1) finds primitive-recursive counterexamples in $\le 10^4$ operations for any encoder-side bug, providing a debugging primitive an order of magnitude cheaper than gradient-based attribution.

These are the falsification targets for the experimental program described in Paper IV.

## 9. Relation to Papers I and II

Paper I established the algebraic core: hidden state on a CM elliptic curve over $\mathbb{Q}(\sqrt{-163})$, Frobenius cancellation through RMSNorm, polynomial-ring attention with KL-zero parity. Paper II reported the system that runs that mathematics with six-figure bit-exactness on Gemma3-1B.

This paper does *not* replace Paper I or II. The sieve and ultraproduct attention are *additions* to the existing stack:

- *Polynomial-ring attention remains the scoring function.* The ultraproduct construction is an alternative path that can be enabled selectively (`--ultraproduct-attn`) but is not the default.
- *The Friedman sieve replaces variance-ranked top-K as the KV-eviction policy.* Variance-ranking is kept as a fallback (`--legacy-eviction`).
- *KSTE is a side-effect of the existing Spinor block.* No additional memory; ~50 LOC for tree construction.

The Frobenius framework of Paper I and the engineering scaffolding of Paper II remain in force.

## 10. Open Questions

1. *Semantic resolution.* Is the discriminative power of $\preceq$ on $\mathcal{T}_{60,3}$ sufficient to distinguish close natural-language meanings? Open empirical question; Paper IV §6 is the experiment.

2. *Calibration drift.* The Knight-Skeleton variance ranking is calibrated at warmup. Does the encoder's behaviour drift as the cache fills? Bounded by the residual quantizer's amax, but not yet measured.

3. *Ultraproduct attention learnability.* Standard attention is differentiable; ultraproduct attention is not, directly. §11 resolves this by moving the learnable boundary: the choice operator $F$ is part of the logical infrastructure, not a learnable parameter, and gradient flow lives in the *generation of the class $A$* — the structural query the model emits. The research direction reformulates from "differentiable hard attention" to "differentiable class definition."

4. *Beyond three labels.* Is there an empirical gain from labels $\{A, B, C, D\}$? The wqo theorem holds for any finite label set, but the 63-byte block constraint pins us at three.

5. *Cross-layer sieve sharing.* Can a single sieve be shared across heads or layers, or does each need its own? Memory implications differ by an order of magnitude.

## 11. The Axiomatic Layer: Choice Operator and Domain Reduction

The three operational layers of §7 — encoder, sieve, ultraproduct attention — describe *what* the system does. They do not yet state *why* the System-1 / System-2 boundary is well-defined. This section provides the axiom that governs it, drawing on Friedman's program of building consistency-strong systems from minimal logical infrastructure.

### 11.1 Two primitives

We adopt two primitives in addition to the wqo and ultrafilter machinery of §§3–5.

**The choice operator $F$.** A unary function on definable classes:
$$A \ne \varnothing \;\Longrightarrow\; F(A) \in A.$$
$F$ selects a canonical element from any nonempty class. Structurally, $F$ is Hilbert's $\varepsilon$-operator: $F(A) := \varepsilon x.\, x \in A$. Determinism (canonicality) is enforced by a fixed total order $\prec_F$ on $\mathcal{T}_{60,3}$ such that $F(A) := \min_{\prec_F} A$. The implementation is specified in Paper IV §10; the order is the packed-byte lexicographic order on the 60-byte tree representation, trivially computable and stable across platforms.

**The Extended-Domain Reduction axiom.** Partition the universe of cached trees into *Real Objects* $\mathrm{RO}$ (the active context window of System 1) and an outer stratum $\mathrm{Ext}$ (the persistent compressed cache of System 2). For a structural predicate $\varphi$ and $v \in \mathrm{RO}$,
$$\varphi(v) \;\Longrightarrow\; \varphi^*(v),$$
where $\varphi^*$ is $\varphi$ relativized to $\mathrm{RO}$ — the truth of $\varphi$ checked using only objects in the active window. The axiom asserts that *structural truths about the full extended domain restrict cleanly to the active window*. Crossing the cache-to-window boundary preserves order-relations without numerical loss.

### 11.2 What the axiomatic layer does for the architecture

The layer slots in above the three operational layers without disturbing them.

- **$F$ is the principal-ultrafilter limit of §5.3, named axiomatically.** When $U = U_p$ is the principal ultrafilter at the highest-ranked key position, $\mathrm{UltraAttn}(Q,K,V; U_p) = V_p = F(\{V_t : t \in \mathrm{cache} \cap A\})$ for the structural class $A$ derived from $Q$. The choice operator and the ultraproduct limit name the same kernel; this section gives it the axiomatic justification the ultrafilter framing lacked.

- **Extended-Domain Reduction is the attention-level analogue of Theorem 4 (Paper I §3.2).** Theorem 4 states: the Frobenius scale factor $\pi^k$ cancels through $QK^\top V W_O$ and vanishes at RMSNorm. The Reduction axiom states: structural properties of cached trees survive selection into the active window. Both are *projective cancellation* theorems — one for arithmetic scale through linear-algebraic operators, one for topological structure through the cache-to-window boundary. The two theorems together close the algebra: every layer of the engine has a cancellation theorem governing it.

- **The combination resolves open question §10.3.** The learnability concern for ultraproduct attention was that $F$ is not differentiable. Under the axiomatic framing the question dissolves: $F$ is *not supposed* to be a learnable parameter. It is part of the logical infrastructure, fixed across all inferences. The learnable component lives elsewhere — in the *generation of the structural class $A$*. The model learns to define classes; the choice operator, fixed and axiomatic, selects from them. Gradient flow goes to class definition, not to attention selection.

### 11.3 Inference as consistency maintenance

Under the axiomatic layer, the inference loop is no longer "predict the most likely next token by minimizing cross-entropy across a continuous distribution." It is closer to constraint satisfaction. Each generation step is:

1. The model defines a structural class $A$ — the set of trees $T$ satisfying the current attention query.
2. The choice operator returns $F(A)$, the $\prec_F$-canonical witness.
3. Extended-Domain Reduction is checked: does $\varphi(F(A))$ imply $\varphi^*(F(A))$ for the current set of invariants $\varphi$?
4. If yes, $F(A)$ is admitted to $\mathrm{RO}$. If no, the local failure is a finite, witness-bearing obstruction (§3.3, WKL₀ refutation property) and the engine emits a correction token instead.

For workloads where consistency is the dominant metric — formal verification, code generation, mathematical reasoning, structured data extraction — this is a strict improvement over likelihood-driven decoding. For free-form natural-language workloads, see §11.4.

### 11.4 Fuzzy classes for natural-language workloads

Natural language is not a logical sequence. Pure consistency-maintenance attention may produce stilted output because the exact class $A$ is often too restrictive: the model wants to choose between paraphrases that no single structural predicate distinguishes. The recovery is to allow $A$ to be *fuzzy* — a Hamming neighborhood of the exact structural query, parameterized by a single radius $r$:
$$A_r \;=\; \{T : d_{\mathrm{tree}}(T, T_Q) \le r\}.$$
For $r = 0$ the framework reduces to hard consistency. For $r$ large, $A_r$ approaches the entire cache; $F$ then selects the $\prec_F$-minimum tree subject to the soft constraint, which we may choose (by defining $\prec_F$ appropriately) to coincide with the top-attention key — recovering soft top-1 behaviour without instantiating a softmax.

The radius $r$ is a *hyperparameter, not a learnable parameter*. It is set per-task: code generation $r = 0$; free-form chat $r$ tuned for fluency; mathematical reasoning $r = 0$ inside derivation steps, $r > 0$ between them. The axiomatic structure survives at every $r$; only the strictness of consistency varies.

### 11.5 Foundational strength of the axiomatic layer

The choice operator with bounded comprehension is at the strength of $\varepsilon$-calculus, which by Ackermann's consistency proof is conservative over Peano Arithmetic. Extended-Domain Reduction with structural predicates over a finite extended domain is at WKL₀-strength by the same finitary argument as §3.3: any failure of the reduction has a primitive-recursive witness. **The runtime stays WKL₀-strength even with the axiomatic layer added.** The mathematics names a powerful object; the engine executes it cheaply, and any failure remains finite and locatable.


### 11.6 From Kruskal embedding to Dickson dominance: the operational subsumption relation

The four operational and axiomatic layers above describe a sieve whose subsumption test is the Kruskal homeomorphic embedding $\preceq$ of §2.2. Phase-4 calibration on the synthetic resolution probe revealed an empirical gap between this relation and the signal actually present in noisy Key vectors. The probe (T4_RES_PROBE — 50 clusters of 20 samples each at $\sigma \in \{0.005, 0.01, 0.02, 0.05, 0.10\}$, intra-cluster cosine running from $0.998$ down to $0.65$) showed that strict ordered embedding on $\mathcal{T}_{60,3}$ produces *zero* intra-cluster subsumption at every $\sigma$ tested: residual sign-flips at the 3-bit quantization boundary make every cluster member topologically distinct under the order-preserving injection of §2.2. AUC sat at 0.500 — the encoder discriminates nothing under $\preceq$. Three remediations (Path A, 2-bit residual magnitudes; Path B, bucketed attachment; Path C, the Nash–Williams unordered embedding) all stayed at the same wall. *The empirical signal lives only under a coarser relation than $\preceq$.*

**The dominance subsumption relation.** Define $\preceq_d$ on $\mathcal{T}_{60,3}$ by
$$Q \preceq_d K \;\iff\; \sigma_0(K) \succeq \sigma_0(Q) \;\text{ and }\; \sigma_1(K) \succeq \sigma_1(Q),$$
where $\sigma_0(T) \in \mathbb{N}^5$ is the Tier-0 structural signature (the multiset $(A\text{-count},\ B\text{-count},\ C\text{-count},\ \mathrm{max\_depth},\ \mathrm{node\_count})$ packed in a `uint64_t` with one byte per field) and $\sigma_1(T) \in \mathbb{N}^9$ is the Tier-1 ancestor-pair signature (the $3\times 3$ matrix of counts of ancestor-descendant pairs by label-pair $(a, d) \in \{A,B,C\}^2$, cells saturated at 255 inside a 16-byte struct). The relation $\succeq$ on each $\sigma_i$ is elementwise — every field of $K$'s signature is at least as large as the corresponding field of $Q$'s.

**Embedding into $\mathbb{N}^{14}$.** Combine the two signatures into a single coordinate map:
$$\varphi : \mathcal{T}_{60,3} \to \mathbb{N}^{14}, \qquad T \mapsto \bigl(\sigma_0(T) \oplus \sigma_1(T)\bigr).$$
Then $Q \preceq_d K \iff \varphi(K) \ge_{\mathrm{elem}} \varphi(Q)$ in the elementwise product order on $\mathbb{N}^{14}$.

**The wqo theorem.** The relevant well-quasi-ordering result is older and stricter than Kruskal:

**Theorem 11.1 (Dickson, 1913).** *The elementwise order $(\mathbb{N}^k, \le_{\mathrm{elem}})$ is a well-quasi-order: every infinite sequence $v_1, v_2, \dots$ in $\mathbb{N}^k$ contains indices $i < j$ with $v_i \le_{\mathrm{elem}} v_j$* [Dickson, *Finiteness of the odd perfect and primitive abundant numbers with $n$ distinct prime factors*, American Journal of Mathematics 35, 1913, pp 413–422].

The image $\varphi(\mathcal{T}_{60,3})$ is bounded: each coordinate sits in $[0, 60]$ by the 60-node budget of §4.3, so we work inside the finite hypercube $[0,60]^{14}$. The maximal antichain in this hypercube is bounded by a Sperner-style cross-section count — for $[0,m]^k$ the largest antichain is the largest level set of the coordinate sum, a generalised binomial coefficient. The exact value is not load-bearing for the architecture; what matters is *finiteness*, which Dickson gives outright.

**Strength comparison.** The two relations satisfy
$$Q \preceq K \;\Longrightarrow\; Q \preceq_d K,$$
and the converse fails by construction. Label and ancestor-pair counts are necessary conditions for an embedding to exist — every node of $Q$ must be matched in $K$ with the same label, every ancestor relation in $Q$ must be reflected in $K$ — but they are not sufficient: dominance is blind to the order in which children appear and to the specific topology that realises the counts. The inclusion $\preceq \;\subsetneq\; \preceq_d$ is strict at every $|V(T)| \ge 3$.

The WKL$_0$ refutation property of §3.3 carries through verbatim. Dominance is a bytewise comparison of two 64-bit Tier-0 signatures and two 16-byte Tier-1 signatures; a failure of any sieve decision produces a primitive-recursive witness in single-digit microseconds, byte-precise. Crucially, Dickson's Lemma itself is provable in PRA — it is finitary in a sense Kruskal's theorem is not (Kruskal–Friedman's TREE$(3)$ rests on much higher consistency strength). The runtime property is therefore *strengthened*, not weakened, by the move from $\preceq$ to $\preceq_d$: a stronger refutation property and a more elementary wqo theorem, in exchange for accepting a coarser subsumption.

**The empirical cost.** The four paths A/B/C of strict-embedding tightening yielded ROC AUC $\approx 0.5$ across all $\sigma$ on the resolution probe. Switching the sieve's filter-survival branch from `sp_kste_embed` to elementwise $\sigma_0 \wedge \sigma_1$ dominance produced an intra-to-inter ratio of $17\times$ at the near-duplicate regime ($\mathrm{cos} \ge 0.995$, $\sigma = 0.005$), a $720\times$ wall-time speedup ($685\,\mu\mathrm{s} \to 0.95\,\mu\mathrm{s}$ p99 at $n = 4096$), and a sieve eviction rate of 93.86% on i.i.d. Gaussian streams. *T2.11's 50 $\mu\mathrm{s}$ wall-time gate cleared with 34$\times$ headroom*; T2.6 through T2.10 pass under the new semantics with the previously-tabled pre-filter precision at 98.77%.

**Cache plateau as physical manifestation.** Dickson's well-quasi-ordering of $\mathbb{N}^{14}$ guarantees the sieve terminates: no infinite antichain of mutually non-dominating cache slots exists. The bounded hypercube $[0,60]^{14}$ guarantees the antichain itself is finite. The cache plateau observed empirically at $\sim 307/512$ slots on i.i.d. Gaussian inputs (T2.1) is the physical manifestation of this bound — the system runs out of mutually $\preceq_d$-incomparable trees long before it runs out of physical slots. The eviction policy stops needing the Knight-Skeleton variance fallback under dominance semantics; the cache *settles* rather than churning.

The axiomatic layer of §§11.1–11.5 is unaffected. The choice operator $F$ still selects $\prec_F$-canonical witnesses from structural classes; Extended-Domain Reduction still applies to predicates over the active window. The only change is the operational meaning of "is $Q$ already a substructure of some $K$ in the cache?" — answered now by $\preceq_d$ rather than $\preceq$, with stronger foundations underneath and a discriminative signal that the empirical work actually delivered.


## 12. Conclusion

We have specified the mathematical content of the Friedman Stack. Every layer of the stack — encoder, sieve, ultraproduct attention, choice operator — rests on a primitive that admits a finite witness of failure. The stack's expressivity reaches arbitrarily high in the consistency-strength hierarchy (the wqo theorem alone is independent of ATR₀); its *runtime* sits at WKL₀-strength, with primitive-recursive refutation. The asymmetry is what makes the stack engineerable.

The companion paper (Paper IV) specifies the system, the encoder kernel, the choice-operator implementation, the test plan, and the experimental schedule.

---

## References

1. Kruskal, J. B., *Well-Quasi-Ordering, the Tree Theorem, and Vázsonyi's Conjecture*, Trans. AMS 95, 1960.
2. Friedman, H. M., *FOM archive postings* on Embedded Maximal Cliques (2009–2018), Foundations of Mathematics list.
3. Friedman, H. M., *Boolean Relation Theory and Incompleteness*, manuscript, 2014 (revised).
4. Friedman, H. M., *Tangible Incompleteness*, manuscript series, 2014–2024.
5. Harrington, L., *Conservation theorem for WKL₀ over PRA*, unpublished communication; cited in Simpson, *Subsystems of Second Order Arithmetic*.
6. Simpson, S. G., *Subsystems of Second Order Arithmetic*, Cambridge UP, 2nd ed., 2009.
7. Gödel, K., *Ontological Proof* (1970), in *Collected Works Vol. III*, 1995.
8. Anderson, C. A., *Some Emendations on Gödel's Ontological Proof*, Faith and Philosophy 7, 1990.
9. Benzmüller, C. & Paleo, B. W., *Formalization, Mechanization and Automation of Gödel's Proof of God's Existence*, arXiv:1308.4526, 2013.
10. Łoś, J., *Quelques remarques, théorèmes et problèmes sur les classes définissables d'algèbres*, in *Mathematical interpretation of formal systems*, 1955.
11. Shannon-Prime, *Paper I — PPT Theory* (Frobenius framework). 2026-05-17.
12. Shannon-Prime, *Paper II — PPT System* (engineering on Gemma3-1B). 2026-05-19.
13. Hilbert, D. & Bernays, P., *Grundlagen der Mathematik II*, 1939 (epsilon-operator).
14. Ackermann, W., *Zur Widerspruchsfreiheit der Zahlentheorie*, Math. Ann. 117, 1940 (epsilon-calculus consistency).
15. Friedman, H. M., *FOM archive postings* on consistency proofs from minimal logical infrastructure (Concept Calculus and related), 2009–2018.

---

*Continuation of Papers I–II. The system that executes this paper's content is specified in Paper IV.*
