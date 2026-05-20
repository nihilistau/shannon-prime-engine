# Test Suite — The Friedman Stack

**Test specifications for Papers III–IV implementation. Companion to `IMPLEMENTATION-ROADMAP.md`.**

*Shannon-Prime Project · 2026-05-19*

---

## Conventions

- Every test has a unique ID (T1.1, T2.3, etc.) referenced from the roadmap.
- Each test specifies: **Inputs**, **Procedure**, **Expected**, **Tolerance**, **Pass criterion**.
- All tests must produce a JSON report into `tests/results/T<id>.json`.
- Tests are written in C++ (gtest) for kernel-level, Python (pytest) for end-to-end PPL.
- A test passes iff its JSON `verdict` field equals `"PASS"`.

---

## Tier 1 — Encoder & Embedding (Phase 1–2)

### T1.1 — Encoder determinism

**Inputs.** A fixed Key vector $K_0 \in \mathbb{R}^{128}$ sampled from $\mathcal{N}(0, I)$ with seed 42.

**Procedure.**
1. Run `sp_kste_encode(K_0)` 1000 times.
2. Compare each output tree byte-for-byte.

**Expected.** All 1000 trees identical.

**Tolerance.** Zero. Any byte difference is a failure.

**Pass criterion.** `bit_identical_count == 1000`.

---

### T1.2 — Order-invariance under Frobenius shim

**Inputs.** $K_0$ as in T1.1. Frobenius parameters $(p=41, k=8)$.

**Procedure.**
1. Compute $T_0 = \Phi(K_0)$.
2. Compute the shimmed $K_0' = \pi_{41}^8 \cdot K_0$ via the engine's load-shim path.
3. Compute $T_0' = \Phi(K_0')$.
4. Compare $T_0$ and $T_0'$.

**Expected.** $T_0 = T_0'$ byte-for-byte.

**Tolerance.** Zero.

**Pass criterion.** `bit_identical == true`.

**Rationale.** The encoder is order-invariant; multiplying by $\pi^k$ preserves rank and sign patterns, so the tree must not change. This is the encoder-side analogue of Paper I Theorem 4 (projective cancellation).

---

### T1.3 — Sign-respecting

**Inputs.** $K_0$ and $-K_0$.

**Procedure.**
1. $T = \Phi(K_0)$, $T' = \Phi(-K_0)$.
2. Compare label distributions: count of $B$ in $T$ vs count of $C$ in $T'$, and vice versa.

**Expected.** $|B \text{ in } T| = |C \text{ in } T'|$ and $|C \text{ in } T| = |B \text{ in } T'|$. $A$ counts identical.

**Tolerance.** Zero.

**Pass criterion.** Label counts agree under the B↔C swap.

---

### T1.4 — 60-node budget

**Inputs.** 1000 samples $K \sim \mathcal{N}(0, I_{128})$, seed 0..999.

**Procedure.**
1. For each $K$, compute $T = \Phi(K)$.
2. Record $|V(T)|$.

**Expected.** $|V(T)| \le 60$ for all samples.

**Tolerance.** Zero overflow tolerated.

**Pass criterion.** $\max_t |V(T_t)| \le 60$.

---

### T1.5 — Anchor count

**Inputs.** Same as T1.4.

**Procedure.** Record number of $A$-labeled children of root for each tree.

**Expected.** Anchor count $\in [12, 16]$ (i.e., 14 ± 2).

**Tolerance.** ±2.

**Pass criterion.** All trees satisfy the bound.

---

### T1.6 — Self-embedding

**Inputs.** 1000 trees from T1.4.

**Procedure.** For each $T$, run `sp_kste_embed(T, T)`.

**Expected.** All return 1.

**Pass criterion.** `embed_count == 1000`.

---

### T1.7 — Empty-subtree embedding

**Inputs.** A trivial tree $T_\rho$ with only the root. 1000 trees from T1.4.

**Procedure.** Run `sp_kste_embed(T_\rho, T_i)` for each $T_i$.

**Expected.** All return 1.

**Pass criterion.** `embed_count == 1000`.

---

### T1.8 — Transitivity

**Inputs.** Random triples $(T_1, T_2, T_3)$ such that $T_1 \preceq T_2$ and $T_2 \preceq T_3$.

**Procedure.** Generate 100 such triples by construction (build $T_3$, take a substructure for $T_2$, take a substructure of $T_2$ for $T_1$). Test $T_1 \preceq T_3$.

**Expected.** All 100 succeed.

**Pass criterion.** `transitive_count == 100`.

---

### T1.9 — Antisymmetry

**Inputs.** Random pairs $(T_1, T_2)$ with $T_1 \preceq T_2 \preceq T_1$.

**Procedure.** Generate 100 such pairs. Test $T_1$ and $T_2$ have the same canonical form (sorted children, no relabeling).

**Expected.** All 100 pairs are isomorphic.

**Pass criterion.** `canonical_match == 100`.

---

### T1.10 — Backtracking necessity

**Inputs.** A hand-constructed adversarial pair $(T_Q, T_K)$ where greedy embedding fails on the first choice but a different choice succeeds.

**Procedure.** Run the embedding test; verify the implementation backtracks.

**Expected.** Test returns 1 (embedding exists).

**Pass criterion.** `result == 1` and (instrumented build) backtrack counter > 0.

---

## Tier 2 — Sieve Behaviour (Phase 3–5)

### T2.1 — Sieve termination

**Inputs.** 100,000 random tokens from $\mathcal{N}(0, I_{128})$, fixed seed.

**Procedure.**
1. Run sieve with $n_{\max} = 4096$.
2. Record cache size after each insertion.
3. Compare to the theoretical antichain bound.

**Expected.** Cache size monotone non-decreasing, plateaus before 100k inserts, bounded by antichain count.

**Pass criterion.** `final_cache_size < 4096` (i.e., the bound is not hit) **OR** the bound is hit and the eviction-on-full path engages.

---

### T2.2 — Eviction rate

**Inputs.** WikiText-103 validation split, Gemma3-1B tokenizer, ctx=2048.

**Procedure.**
1. Run inference with sieve enabled.
2. Record admit/evict ratio.

**Expected.** Steady-state eviction rate $\ge 20\%$.

**Tolerance.** Single run; report mean ± std over 5 chunks.

**Pass criterion.** $\text{eviction\_rate} \ge 0.20$.

---

### T2.3 — PPL gating ★

**Inputs.** WikiText-103 validation split, Gemma3-1B (`gemma3-1b.gguf`), ctx=2048, 4 chunks, threads=16.

**Procedure.**
1. Run `sp-engine.exe perplexity-sp` with `--frobenius-quant -p 41 -k 8 --poly-attn --ntt-crt` (baseline).
2. Run same command with `--friedman-sieve` added.
3. Compute $\Delta = (\text{PPL}_{\text{sieve}} - \text{PPL}_{\text{baseline}}) / \text{PPL}_{\text{baseline}}$.

**Expected.** $|\Delta| \le 0.005$ (0.5%).

**Tolerance.** Single number; report exact value.

**Pass criterion.** $|\Delta| \le 0.005$. **This is the ship/no-ship gate.** ★

---

### T2.4 — Refutation procedure

**Inputs.** A manually-injected adversarial $K$ designed to violate the sieve's correctness (e.g., a $K$ whose tree should embed in a cached tree but the sieve fails to detect).

**Procedure.**
1. Initialize sieve.
2. Insert the adversarial $K$.
3. Run the WKL₀-style refutation: exhaustively search the embedding-test decision tree for the failure point.

**Expected.** A specific $(T_Q, T_K, \text{decision-path})$ triple is produced in $\le 10{,}000$ operations.

**Pass criterion.** `refutation_found == true` and `operation_count <= 10000`.

---

### T2.5 — Closure axiom

**Inputs.** A populated cache with 1000 entries.

**Procedure.**
1. Define a *big-subset* as the set of indices whose anchor magnitude exceeds the median.
2. For each pair of cached trees $(T_i, T_j)$, compute the big-subset $A_i$, $A_j$ and the intersection $A_i \cap A_j$.
3. Verify the intersection is itself a big-subset under the sieve's invariants.

**Expected.** All 999 × 500 pairs satisfy closure.

**Pass criterion.** No closure violation.

---

### T2.6 — Eviction on subsumption

**Inputs.** Synthetic pair $(T_Q, T_K)$ with $T_Q \preceq T_K$ by construction; cache initialized with $T_K$.

**Procedure.** Insert $T_Q$, observe sieve decision.

**Expected.** $T_Q$ evicted.

**Pass criterion.** `evicted == true`.

---

### T2.7 — Admission on novelty

**Inputs.** Synthetic $T_Q$ with no embedding into any cached tree; cache populated with $N=1000$ random trees.

**Procedure.** Insert $T_Q$.

**Expected.** $T_Q$ admitted.

**Pass criterion.** `admitted == true`.

---

### T2.8 — Knight-skeleton fallback

**Inputs.** Cache filled to capacity (4096) with novel tokens (none embed into others); insert one more novel token.

**Procedure.** Insert, observe which existing token is evicted.

**Expected.** The token with lowest Knight-Skeleton variance is evicted.

**Pass criterion.** `evicted_idx == argmin(variance)`.

---

### T2.9 — Pre-filter correctness (Phase 5)

**Inputs.** 10,000 random tree pairs.

**Procedure.**
1. Run the label-multiset pre-filter.
2. Run the full embedding test.
3. For every pair where pre-filter returns *not embedding*, verify the full test agrees.

**Expected.** Zero false negatives (pre-filter saying "not embedding" when full test says "embedding").

**Pass criterion.** `false_negatives == 0`. False positives are allowed (and expected).

---

### T2.10 — Pre-filter precision

**Inputs.** Same 10,000 pairs.

**Procedure.** Measure the fraction of pairs the pre-filter correctly rules out (true negatives over all negatives).

**Expected.** $\ge 90\%$ precision.

**Pass criterion.** `precision >= 0.90`.

---

### T2.11 — Sieve wall-time at scale

**Inputs.** Cache of size 4096; 1000 incoming tokens.

**Procedure.** Run sieve, time per-token average and max.

**Expected.** Mean $\le 5$ µs, max $\le 50$ µs.

**Pass criterion.** Both bounds satisfied.

---

### T2.12 — Extended-Domain Reduction invariant ★

**Inputs.** A populated cache of 500 trees with the active-window subset $\mathrm{RO}$ identified (the most-recent 64 tokens by default). Three standard structural predicates from `sp_kste_predicates.h`:
- $\varphi_1$ = "anchor count $\ge 12$"
- $\varphi_2$ = "label-$B$ count $\ge 10$"
- $\varphi_3$ = "max depth $\le 5$"

100 randomly-selected canonical witnesses $v = F(A)$ from the cache.

**Procedure.**
1. For each $(v, \varphi)$ pair, evaluate $\varphi(v)$ on the full cached representation of $v$.
2. If true, evaluate $\varphi^*(v)$ — the relativization of $\varphi$ to $\mathrm{RO}$, computed by checking $\varphi$ across the active-window subset alone.
3. Assert $\varphi(v) \Rightarrow \varphi^*(v)$ for every pair.

**Expected.** All implications hold. No counterexample produced. If a counterexample exists, it is reported as a concrete $(v, \varphi)$ pair — the primitive-recursive witness predicted by the WKL₀ refutation property.

**Tolerance.** Zero. The reduction axiom is structural; any failure is a primitive-recursive witness of a code bug.

**Pass criterion.** `reduction_failures == 0`. **This is the axiomatic-layer unit test.** ★

---

## Tier 3 — Ultraproduct Attention (Phase 7–8)

### T3.1 — Principal ⇒ Top-1

**Inputs.** A toy 16-token cache with synthetic Q and V vectors such that the maximum attention weight is unambiguously at position 7.

**Procedure.** Run `UltraAttn` with $U = U_7$ (principal ultrafilter at position 7).

**Expected.** Output equals $V_7$ exactly.

**Pass criterion.** `output == V_7` (bit-exact).

---

### T3.2 — Łoś on toy

**Inputs.** A 100-element sequence of 16-dimensional vectors where 60 elements have property $\phi$ (some specific predicate) and 40 do not.

**Procedure.**
1. Choose an ultrafilter $U$ such that the 60-element subset is in $U$.
2. Compute $\mathrm{ult}_U(\text{sequence})$.
3. Verify $\mathrm{ult}_U$ satisfies $\phi$.

**Expected.** Pass.

**Pass criterion.** $\phi(\mathrm{ult}_U) = \text{true}$.

---

### T3.3 — PPL on LongBench

**Inputs.** LongBench QA subset, Gemma3-1B, contexts {2k, 8k, 32k}.

**Procedure.**
1. Run baseline (softmax attention).
2. Run with `--ultraproduct-attn=principal`.
3. Compute PPL deltas.

**Expected.** $|\Delta\text{PPL}| \le 3\%$ at every context length.

**Pass criterion.** Bound satisfied at all three context lengths.

---

### T3.4 — RULER long-context

**Inputs.** RULER probes at ctx=32k.

**Procedure.**
1. Run baseline.
2. Run with ultraproduct attention.

**Expected.** Score within 5 percentage points of baseline.

**Pass criterion.** $|\text{score}_\text{ultra} - \text{score}_\text{baseline}| \le 0.05$.

---

### T3.5 — Wall-time

**Inputs.** ctx=8192, single decode step.

**Procedure.** Time both attention kernels.

**Expected.** Ultraproduct kernel within 20% of softmax wall-time.

**Pass criterion.** $\text{time}_\text{ultra} / \text{time}_\text{softmax} \le 1.20$.

---

### T3.6 — Choice operator canonicality

**Inputs.** A class $A$ of 100 mutually $\preceq$-incomparable trees, constructed deterministically from seed 1337.

**Procedure.**
1. Invoke `sp_kste_select_canonical(A)` 1000 times in different orderings of the candidate list (random permutations under fresh seeds 0..999).
2. Compare the returned tree byte-for-byte across all 1000 invocations.

**Expected.** All 1000 invocations return the same byte-identical tree — the $\prec_F$-minimum under packed lexicographic order.

**Tolerance.** Zero. Hilbert's $\varepsilon$-operator requires deterministic selection.

**Pass criterion.** `unique_result_count == 1`. The single returned tree matches the manual-computed lex-minimum.

---

## Tier 6 — HVX Kernel (Phase 6)

### T6.1 — HVX/CPU parity

**Inputs.** 10,000 random tree pairs.

**Procedure.** Run both CPU and HVX `sp_kste_embed`; compare results.

**Expected.** Bit-identical decisions.

**Pass criterion.** `disagreement_count == 0`.

---

### T6.2 — Single-dispatch latency

**Inputs.** 1000 single embed-test dispatches on S22U.

**Procedure.** Time round-trip including FastRPC overhead.

**Expected.** Mean $\le 25$ µs.

**Pass criterion.** Met.

---

### T6.3 — Batched dispatch

**Inputs.** 16-test batches, 100 batches.

**Procedure.** Time each batch end-to-end.

**Expected.** Mean batch latency $\le 200$ µs.

**Pass criterion.** Met.

---

### T6.4 — First-light on S22U

**Inputs.** Qwen3-4B Q6_K on S22U with `--friedman-sieve --hexagon-backend`.

**Procedure.** Run a 64-token prefill.

**Expected.** No crash. PPL within 1% of CPU reference.

**Pass criterion.** Both conditions met.

---

## Tier 0 — Smoke (always-on)

These tests run on every commit; they are *not* phase-gated.

### T0.1 — Build clean

`cmake -B build && cmake --build build` produces zero warnings, zero errors on MSVC, GCC, and HVX cross-compile.

### T0.2 — Existing engine PPL unchanged

`sp-engine.exe perplexity-sp` on Gemma3-1B reproduces the pre-Friedman PPL to 6 significant figures when `--friedman-sieve` is **off**.

### T0.3 — Memory leak audit

`valgrind --leak-check=full` on `test_sp_friedman_cache` reports zero definite leaks.

### T0.4 — Determinism across threads

Sieve with `--threads=1` and `--threads=8` produce the same final cache contents on the same input.

---

## Reporting format

Each test writes `tests/results/T<id>.json`:

```json
{
  "test_id": "T2.3",
  "phase": 4,
  "timestamp": "2026-05-21T14:33:00Z",
  "config": {
    "model": "gemma3-1b",
    "ctx": 2048,
    "frobenius": "p=41,k=8",
    "sieve": "on",
    "chunks": 4
  },
  "metrics": {
    "ppl_baseline": 11.8311,
    "ppl_sieve": 11.8367,
    "delta_pct": 0.047,
    "eviction_rate": 0.342,
    "mean_tree_size": 41.2,
    "embed_test_mean_us": 18.4,
    "embed_test_max_us": 47.1
  },
  "verdict": "PASS",
  "notes": "Eviction rate exceeds T2.2 minimum (0.20). PPL delta well under T2.3 gate (0.5%)."
}
```

A central script `scripts/run_all_tests.bat` aggregates every JSON into `tests/results/SUMMARY.json` and prints a green/red dashboard.

---

## Test ordering for first execution

A new session running this suite for the first time should execute tests in the following order (matching phase order):

```
T0.1 → T0.2 → T0.3 → T0.4         (smoke)
T1.1 → T1.2 → T1.3 → T1.4 → T1.5  (encoder)
T1.6 → T1.7 → T1.8 → T1.9 → T1.10 (embedding)
T2.1 → T2.5 → T2.6 → T2.7 → T2.8  (cache structure)
T2.2 → T2.3 → T2.4                (eviction behaviour) ★ T2.3 is the gate
T2.9 → T2.10 → T2.11              (optimization)
T2.12                             (axiomatic layer: Extended-Domain Reduction) ★
T6.1 → T6.2 → T6.3 → T6.4         (HVX)
T3.1 → T3.2 → T3.6                (ultraproduct toys + choice-op canonicality)
T3.3 → T3.4 → T3.5                (ultraproduct benchmarks)
```

Halt at the first failure. Investigate. Resume from the same test once fixed.

---

## See also

- `IMPLEMENTATION-ROADMAP.md` — phase-by-phase plan referencing these test IDs.
- `BOOTSTRAP-PROMPT.md` — context loader for a new session.
- `PPT-ARM-IV-KSTE.md` §6 — original test plan that this document expands.
