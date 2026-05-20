# SESSION-STATE-friedman-5.md

**Phase 5 — sieve performance optimisation (Tier-0 + Tier-1 layered filter). CORRECTNESS + PRECISION COMPLETE; wall-time gate partially closed; the remaining gap is dominated by full-embed cost on ~1.3 % of filter-survivors.**

*Shannon-Prime Project · Friedman Stack rollup · 2026-05-20*

---

## Why we did Phase 5 before Phase 4

KnackAU's go-ahead from this session: the naive O(N) cache from Phase 3 makes Phase 4's WikiText-103 PPL calibration impractical to iterate on (hours per run). Pulling Phase 5 forward gives Phase 4 a usable runtime to land calibration against. The order remains 4 → 5 → 6 → … in the formal roadmap; this session inverted just 4 ↔ 5.

## Deliverables shipped

```
shannon-prime-engine/lib/shannon-prime/core/sp_kste.h       (+50 LOC: sig0/sig1 types, dominance API)
shannon-prime-engine/lib/shannon-prime/core/sp_kste.c       (+85 LOC: compute_signature, compute_anc_sig,
                                                              sig_dominates, anc_sig_dominates)
shannon-prime-engine/src/sp_friedman_cache.h                (slot struct extended +24 B,
                                                              cache counters +3 fields)
shannon-prime-engine/src/sp_friedman_cache.cpp              (insert path: Tier-0 -> Tier-1 -> full embed)
shannon-prime-engine/tests/test_sp_friedman_cache.cpp       (+310 LOC: T2.9 / T2.10 / T2.11)
```

Total: ~470 LOC new code + 310 LOC tests. Within roadmap §5 budget of 600 LOC.

## Mathematical correctness

Both filter tiers encode **necessary conditions** for Kruskal-Friedman homeomorphic embedding (no false negatives by construction; T2.9 verified):

- **Tier 0** (single uint64 compare). For any embedding ι : V(Q) → V(K), the injection preserves labels and ancestor relations, so:
  - count_A(Q) ≤ count_A(K), and same for B, C.
  - max_depth(Q) ≤ max_depth(K) (ancestor preservation).
  - node_count(Q) ≤ node_count(K).
  Pack each into a 7-bit field with high-bit guard; dominance via `(K | 0x80…) - (Q & 0x7F…) & 0x80…` → constant time, no branches.

- **Tier 1** (16-byte ancestor-pair multiset). For each (label_a, label_d) pair in {A,B,C}×{A,B,C}, count (u,v) where u is a proper ancestor of v with labels (a,d). The injection maps each such Q-pair to a K-pair with the same label types, so K's 9 cells must dominate Q's. Saturate at 255 per cell.

Computation cost per encode: Tier 0 ≈ 60 ops (single linear pass); Tier 1 ≈ 60 × max_depth ≈ 540 ops. Cached on the slot — no recompute on the hot scan path.

## Tests run

All seven Phase-3 functional tests **STILL PASS** with the layered filter in place (no regression).

| Test | Verdict | Key metrics |
|------|---------|-------------|
| T2.1 — termination | **PASS** | unchanged from Phase 3 |
| T2.5 — closure axiom | **PASS** | unchanged |
| T2.6 — eviction on subsumption | **PASS** | unchanged |
| T2.7 — admission on novelty | **PASS** | unchanged |
| T2.8 — Knight-Skeleton fallback | **PASS** | unchanged |
| T2.12 — Extended-Domain Reduction | **PASS** | unchanged |
| **T2.9 — pre-filter correctness** | **PASS** | 10000 pairs (5023 embed YES, 4977 NO); `false_negatives = 0`. **Tier-0 + Tier-1 filter NEVER rejects a pair that the full embed would accept.** |
| **T2.10 — pre-filter precision** | **PASS** | 9959 negatives / 10000 pairs; `filter_rejected = 9837`; **precision = 98.77%** (>> 90% gate). Tier-0 alone caught 9696 (97.4%); Tier-1 added 141 more. |
| **T2.11 — wall-time at capacity 4096** | **FAIL (gate)** | mean 685 µs, p50 326 µs, p99 4606 µs, max 9817 µs. **Tier-0 survival 2.75%, Tier-1 survival 46.5%** ⇒ ~1.28% of pairs reach the full embed; 51 085 full embeds across 1000 bench inserts = ~51 per insert. |
| T2_BENCH (n=512, legacy from Phase 3) | **FAIL (gate)** | mean 73 µs vs Phase-3 2420 µs — **33× speedup** at n=512. Retained for cross-phase comparison. |

## Empirical comparison (Phase 3 vs Phase 5)

At cache capacity 512 (the comparable bench from Phase 3):

| Metric | Phase 3 (naive) | Phase 5 (Tier-0 + Tier-1) | Speedup |
|--------|----------------:|--------------------------:|--------:|
| mean / insert | 2420 µs | 73 µs | **33×** |
| p50 / insert  | 2000 µs | 35 µs | **57×** |
| p99 / insert  | 7060 µs | 587 µs | **12×** |

At cache capacity 4096 (the T2.11 gate target):

| Metric | Value | Note |
|--------|------:|------|
| Slot tests per insert | ~4000 | linear scan, Tier-0 cost dominated |
| Tier-0 survival | 2.75% | the 64-bit dominance gate filters aggressively |
| Tier-1 survival of Tier-0 survivors | 46.5% | adds ~half-decade of further filtering |
| Full embeds per insert | ~51 | the residual cost driver (51 × ~10 µs = 510 µs) |
| **p99 / insert** | **4606 µs** | **above 50 µs gate by ~92×, down from naive ~140×** |

## Where the residual cost lives

The Tier-0 scan over 4096 slots × ~3 ns each = ~12 µs lower bound. We're at p50 = 326 µs, which is roughly:

```
326 µs ≈ (4096 × 3 ns scan) + (51 × ~6 µs full embed)
       ≈      12 µs           +         300 µs
```

So **~92% of the cost at p50 is the full embed running on the 51 filter-survivors per insert.** Two complementary levers close the remaining gap:

1. **Tier-2 structural fingerprint** — a 32-byte hash combining sub-tree label vectors at depths 1, 2, 3 would catch many of the 51 survivors before the full embed. Estimated additional rejection rate: 50-80%. Would put us in the 50-100 µs band at n=4096.

2. **Bucket indexing on (max_depth, node_count)** — the Tier-0 dominance already requires both fields to dominate, so we can pre-bucket slots and only iterate the eligible bucket(s). For random N(0,I) keys at HD=128 with mostly-saturated 60-node trees, this collapses 4096 candidates to ~50-100. Combined with Tier-0/1 above, would land us inside the 5 µs/token gate.

3. **HVX vectorisation** — uint64 dominance vectorises trivially to 4-32 lanes per HVX dispatch. Phase 6 territory.

I would recommend **bucket indexing in Phase 5b** as the next-best lever — it's a structural change in the cache (modest LOC) and gets us inside or near the 50 µs gate even without HVX. Tier-2 fingerprint could come alongside if needed.

## Deviations from spec

1. **`SP_KSTE_TAU_A_DEFAULT` still 0.0** — same Phase-1 bootstrap; Phase 4 calibration will set the production value.
2. **T2.11 wall-time still over gate** — see §"Where the residual cost lives" above. Layered filter delivered the bulk of the speedup; bucket indexing or HVX is the next-required lever.
3. **`sp_kste_signature_t` / `sp_kste_anc_sig_t` exposed in `sp_kste.h`** — beyond Paper IV §4 spec. Necessary so the cache can hoist signature computation out of the embed hot path. The types are pure-data POD with no extra dependencies.

## What the engine got in this session

- **A constant-time dominance test.** A single 64-bit subtract-with-borrow tells the cache whether an embedding is even mathematically possible. The 9-cell ancestor-pair multiset is a second-tier exact filter, no Bloom-style false positives.
- **An audit trail of how many slots survive each filter tier.** `cache.slot_tests`, `cache.tier1_tests`, `cache.full_embeds` — the engine can now self-report its filter health at any moment without rerunning a bench.
- **No-false-negative guarantee.** T2.9 over 10000 pairs (with synthetically forced positives) confirms that whenever the full embed says YES, the filter agrees. This is the WKL₀-strength guard: the filter cannot silently lose a valid subsumption.

## Cross-phase invariants

| Invariant | Status |
|-----------|--------|
| 1. PPL never regresses | N/A (no inference path touched) |
| 2. CPU & HVX paths agree bit-exactly | N/A (HVX is Phase 6) |
| 3. WKL₀ refutation property preserved | Held — every filter decision is primitive-recursive: a 64-bit ALU op and a 2-uint64 ALU op. Witnesses are the bit fields themselves. |
| 4. No `__int128` | Held |
| 5. No global mutable state | Held |
| 6. 63-byte Spinor block format frozen | Held — signatures live on the cache slot, not the block |
| 7. Calibration knight-mask ships with model | Phase 4 |

## Recommended next phase

Pick one of the following — explicit go-ahead from KnackAU:

- **Phase 5b: bucket indexing on (max_depth, node_count).** Closes (or nearly closes) the T2.11 gate in pure CPU. Estimated 200 LOC, half a day.
- **Phase 4: PPL calibration on Gemma3-1B.** Now feasible: at p50 = 35 µs per insert at n = 512, a chunk-128 WikiText-103 run is on the order of seconds, not hours. Drives the production `tau_A` / `alpha` values.
- **Phase 6: HVX kernel for Tier-0 dominance + full embed.** Vectorises the same operations across 32 slots/dispatch. Bigger lift (~800 LOC, Snapdragon V69 tooling).

My recommendation in priority order: **5b → 4 → 6**. Phase 5b is the cheapest closure of T2.11; Phase 4 is the actual ship gate; Phase 6 is the mobile-target win. Open to KnackAU/Gemini override.

## Notes for future sessions

- JSON audit trail at `D:\F\shannon-prime-repos\tests\results\T2_*.json`. Ten Tier-2 reports plus the legacy bench. Aggregator `T2_SUMMARY.json` lists all verdicts.
- Edit-tool truncation hazard hit this session on `sp_kste.h`, `sp_friedman_cache.h`, and `sp_friedman_cache.cpp`. Recovery via `python3 .rstrip(b'\\x00')` + `cat >> file <<EOF` heredocs. Memory record `feedback_edit_tool_truncation` is current and recurrent — for any insertion > ~50 lines, default to bash heredoc and re-verify the tail line.
- Phase-3's legacy `bench_insert_walltime` is still wired and useful for cross-phase comparison at small cache sizes; I recommend leaving it in place. Phase 5b should add T2_BENCH_BUCKETED on top.
- The cmake build was not exercised on the Windows host this session. Wiring is unchanged from Phase 3 (`SP_FRIEDMAN_SIEVE` option, self-contained test target). First MSVC build remains the external gate.
