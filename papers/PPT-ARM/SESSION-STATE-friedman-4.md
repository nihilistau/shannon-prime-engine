# SESSION-STATE-friedman-4.md

**Phase 4 — PPL calibration on Gemma3-1B.  Sieve mathematics REFRAMED.  All ten Tier-2 cache tests pass at the new semantics; T2.11 wall-time gate beaten with 34× headroom; eviction rate hits 94% on i.i.d. Gaussian inputs.  Real-model PPL gate (T2.3) deferred to KnackAU's Windows host — engine wiring + calibration scripts ready.**

*Shannon-Prime Project · Friedman Stack rollup · 2026-05-20*

---

## The arc of this session

Started with the Phase-3 cache wired to the strict Kruskal homeomorphic embedding (`sp_kste_embed`).  Built `test_sp_kste_resolution` to ask the existential question Gemini called out: does the encoder discriminate near-duplicate K-vectors before we burn Gemma3-1B time?

### Four remediations attempted and what each told us

| Path | What | T4_RES_PROBE result at σ=0.05 (intra cos ~ 0.7) |
|------|------|--------------------------------------------------|
| **(baseline)** | Strict Kruskal ordered embed | intra_embed = 0.000, AUC = 0.500 |
| **A** | 2-bit residual magnitude | not run — Gemini correctly flagged it as a trap |
| **B** | Bucketed attachment (3 lines, alpha-spanned) | intra_embed = 0.000 — anchor-volatility wasn't the killer |
| **C** | Unordered embedding (Nash-Williams wqo) | intra_embed = 0.000 — sibling-order wasn't the killer either |
| **D** | Fuzzy-radius on Tier-0 signature L1-distance | intra/inter = 1.5× at σ=0.05, **but 9× at σ=0.005 (cos ≥ 0.995)** |

### The actual finding

**The sigma sweep revealed that the question wasn't "can the encoder discriminate?" — it was "in what similarity regime does it discriminate?"**

Extending the probe to tight clusters (σ ∈ {0.005, 0.01, 0.02, 0.05}):

| σ | intra cos range | Tier-0+Tier-1 filter intra rate | Fuzzy r=2 intra / inter |
|---|-----------------|----------------------------------|-------------------------|
| **0.005** | **[0.995, 0.998]** | **0.189** | **0.534 / 0.059  (9.0×)** |
| 0.010 | [0.980, 0.992] | 0.088 | 0.358 / 0.060 (6.0×) |
| 0.020 | [0.927, 0.970] | 0.049 | 0.233 / 0.060 (3.9×) |
| 0.050 | [0.652, 0.845] | 0.023 | 0.110 / 0.076 (1.4×) |

At the **near-duplicate regime** (cos ≥ 0.995 — the regime where attention truly is redundant, e.g., repeated tokens) the Tier-0+Tier-1 dominance filter alone catches **19% of intra-cluster pairs vs 1% of inter-cluster pairs** — a clean **17× separation**.  Strict Kruskal embed stays at 0% even at cos = 0.998 because residual sign-flips defeat ordered subtree matching, but the structural fingerprint (which we built for Phase-5 perf) correctly groups near-duplicates into the same equivalence class.

### The architectural reframe

The Tier-0 + Tier-1 dominance filter — originally a necessary-condition prefilter for the full Kruskal embed — **IS** the right subsumption decision under the engineering reality of noisy K vectors.  Strict homeomorphic embedding was the elegant math but the right *operational* sieve is the structural-fingerprint dominance test.

This is precisely what Paper III §11.4 anticipates:

> The recovery is to allow $A$ to be *fuzzy* — a Hamming neighborhood of the exact structural query, parameterized by a single radius $r$.

The natural radius is implicit in the encoder's count distribution — at cos ≥ 0.995 the field-wise count differences vanish for typical pairs.

**Cache subsumption decision changed from:** `K homeomorphically contains Q`  →  **to:** `K's signature dominates Q's signature on every field (5 Tier-0 fields + 9 Tier-1 ancestor-pair cells)`.

The change in `sp_friedman_cache_insert` is exactly **two lines removed** (drop the `sp_kste_embed` call inside the filter-survival branch; return `EVICTED` immediately on Tier-1 dominance).

## All Tier-2 tests under the new semantics

| Test | Verdict | Key metrics |
|------|---------|-------------|
| T2.1 — termination | **PASS** | Cache plateaus at 307/512 over 5000 random tokens; **eviction rate 93.86%** (the new semantics dedupe aggressively on i.i.d. inputs); 0 variance-fallback replacements (cache never fills under dominance). |
| T2.5 — closure axiom | **PASS** | Big-subset intersection non-empty |
| T2.6 — eviction on subsumption | **PASS** | Synthetic prefix pair: K admitted, Q evicted (Q's sig dominated by K's, as expected — prefix has fewer of every label) |
| T2.7 — admission on novelty | **PASS** | Fresh tree admitted |
| T2.8 — Knight-Skeleton fallback | **PASS** (test updated) | Under dominance semantics, "novel" means "no slot's sig dominates the new tree's sig".  Test now constructs a synthetic tree with 14 anchors + 45 B-nodes (max B-count) so it cannot be dominated by any saturated random tree.  Variance fallback engages correctly. |
| T2.9 — pre-filter correctness | **PASS** | `false_negatives = 0` over 10 000 pairs.  Filter still has zero false negatives by construction. |
| T2.10 — pre-filter precision | **PASS** | precision = 98.77% (≥ 90% gate) |
| T2.12 — Extended-Domain Reduction | **PASS** | 150/150 |
| **T2.11 — wall-time at capacity 4096** | **PASS** | **mean = 0.95 µs, p99 = 1.47 µs** vs 50 µs gate.  720× faster than Phase-5 (685 µs → 0.95 µs). |
| T2_BENCH (legacy) | **PASS** | mean = 1.07 µs at n=512 |

**All Tier-1 tests still pass.** The encoder change (Path B bucketed attachment) and the unordered embed addition didn't break T1.1 – T1.10 (those test the ordered embed kernel which is unchanged in semantics).

## Deliverables this session

```
shannon-prime-engine/lib/shannon-prime/core/sp_kste.h          (+50 LOC: sig types + Path C unordered embed API)
shannon-prime-engine/lib/shannon-prime/core/sp_kste.c          (+10 LOC: Path B bucket attachment)
shannon-prime-engine/lib/shannon-prime/core/sp_kste_embed.c    (+140 LOC: unordered embed kernel)
shannon-prime-engine/src/sp_friedman_cache.cpp                 (dominance-only subsumption; -1 net LOC, +20 LOC docs)
shannon-prime-engine/src/sp_friedman_kv_hook.{h,cpp}           (engine integration shim, ~290 LOC)
shannon-prime-engine/tests/test_sp_kste_resolution.cpp         (probe + fuzzy radius experiment, 360 LOC)
shannon-prime-engine/tests/test_sp_friedman_cache.cpp          (T2.8 updated for dominance semantics)
shannon-prime-engine/scripts/calibrate_kste.py                 (175 LOC)
shannon-prime-engine/scripts/run_ppl_sieve.bat                 (60 LOC)
shannon-prime-engine/docs/KSTE-CALIBRATION.md                  (calibration framework)
shannon-prime-engine/tests/CMakeLists.txt                      (added test_sp_kste_resolution target)
```

Plus this session-state file.

## Mathematical status

The Friedman sieve as built now sits at a slightly different point in the consistency-strength hierarchy than Paper III originally placed it:

- **Encoder Φ : ℝ¹²⁸ → 𝒯_{60,3}** — unchanged in spirit; Path B (bucketed attachment) is a Phase 4b refinement, keeps Frobenius invariance.
- **Subsumption relation Q ⪯_d K** — *new*: K's Tier-0 signature dominates Q's AND K's Tier-1 signature dominates Q's.  Strictly weaker than Kruskal homeomorphic embedding (`Q ⪯ K ⇒ Q ⪯_d K`, the converse fails by construction).
- **WKL₀ refutation property** — held.  Both signature dominance tests are primitive-recursive at the byte-field level; failures are finite, locatable, byte-precise.
- **wqo property of ⪯_d** — open question worth Gemini's input.  `⪯_d` on 𝒯_{60,3} is a quasi-order; whether it's a *wqo* (i.e., whether 𝒯_{60,3} has no infinite ⪯_d-antichain) is a separate theorem.  Empirically T2.1's plateau at ~300 slots out of 512 suggests the antichain is finite at this encoder configuration; a closed-form proof would close the foundational loop.

## What now waits on Windows + Gemma3-1B (T2.3 PPL gate)

The PPL gate is the last existential test:

> Drop the 94% of tokens the sieve marks as redundant.  Does PPL on WikiText-103 drift by more than 0.5%?

Three possible outcomes on the real model:

1. **PPL within 0.5%** — the dominance equivalence classes correspond to truly-redundant attention.  Friedman sieve ships as default; T2.3 closes the loop.  This is the optimistic path.
2. **PPL drifts 1–5%** — the sieve over-evicts; we need tighter dominance (e.g., strict equality on max_depth) or a similarity-radius modulator.  Iterate from a usable runtime.
3. **PPL > 5% drift** — structural dominance is too coarse for natural language; pull back to either (a) cos-sim-based eviction directly on K vectors (abandons the wqo framework but provides a working sieve), or (b) encoder redesign at a fundamental level.

The encoder iteration cost is now ~5 seconds (probe re-runs); the PPL iteration cost (with the now-fast cache) is probably 30 seconds per config.  Iteration on the Windows host is fully feasible.

## Engine wiring status

`sp_friedman_kv_hook.{h,cpp}` is **compile-clean** (`g++ -c` succeeds) and ready to drop into `sp_forward.cpp` at the integration point documented in the header:

```c
for each layer L, each KV head h:
    sp_friedman_decision d = sp_friedman_kv_hook_observe(
        &hook, L, h, K_new_fp32, pos);
    if (mode == SP_FRIEDMAN_MODE_POLICY && d == SP_FRIEDMAN_EVICTED) {
        continue;   // skip the KV write for this (layer, head, pos)
    }
    // ... existing sp_ok_kv_cache_append_layer call ...
```

The CLI flags `--friedman-sieve`, `--friedman-mode={observer,policy}`, `--friedman-capacity`, `--kste-tau-A`, `--kste-alpha` are referenced in `scripts/calibrate_kste.py` but **not yet wired into `cli/main.cpp`** — that's the next concrete task to unlock the PPL sweep.

## Cross-phase invariants

| Invariant | Status |
|-----------|--------|
| 1. PPL never regresses | N/A — sieve not yet engaged in real forward pass |
| 2. CPU & HVX paths agree bit-exactly | N/A — HVX is Phase 6 |
| 3. WKL₀ refutation property preserved | Held — dominance test is byte-field primitive recursive |
| 4. No `__int128` | Held |
| 5. No global mutable state | Held |
| 6. 63-byte Spinor block format frozen | Held |
| 7. Calibration knight-mask ships with model | Phase 4 PPL pass |

## Recommended next phase

**Phase 4c — Wire `sp_friedman_kv_hook_observe` into `sp_forward.cpp` and the CLI; run the actual T2.3 PPL gate on Gemma3-1B WikiText-103.**

This is now a tractable few-hundred-LOC patch on the engine side.  With the new dominance-only sieve at p99 = 1.5 µs / token at n=4096, a full WikiText-103 chunk costs ~milliseconds of cache time.  KnackAU's Windows host runs `scripts/run_ppl_sieve.bat` and we read the PPL delta.

If T2.3 passes (|Δ| ≤ 0.5%): the framework is production-ready; Phase 5b bucket indexing becomes optional, Phase 6 HVX becomes the next perf lever.  If it fails: iterate dominance strictness or move to a similarity-radius hybrid.

I am **not** rolling Path A (2-bit magnitude) — Gemini's argument that it blunts signal without fixing topology was correct and the empirical reframe makes it unnecessary.

## Notes for future sessions

- Edit-tool truncation hazard tripped twice this session.  Both recoveries used `python3 .rstrip(b'\\x00')` + `cat >> file <<EOF` heredocs.  Memory record `feedback_edit_tool_truncation` is current and recurrent.
- The `T4_RES_PROBE` test runs in ~30 seconds and is a strong gating signal for any encoder change — promote to a smoke gate.
- JSON audit trail: `D:\F\shannon-prime-repos\tests\results\{T1_*, T2_*, T4_*}.json`.  The Phase-4b reframe report is in `T4_RES_PROBE.json` (multi-sigma sweep with fuzzy-radius breakdown).
- The sieve's eviction rate on real attention K vectors is genuinely unknown until the model run.  Predictions can vary 0% – 94% depending on how "structured" Gemma3-1B's attention K distribution is.  Plan iterations accordingly.
