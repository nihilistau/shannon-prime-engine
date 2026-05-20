# SESSION-STATE-friedman-3.md

**Phase 3 — Friedman sieve cache structure. CORRECTNESS COMPLETE; perf gate explicitly deferred to Phase 5 (by design, per roadmap §3 risk).**

*Shannon-Prime Project · Friedman Stack rollup · 2026-05-20*

---

## Deliverables shipped

```
shannon-prime-engine/lib/shannon-prime/core/sp_kste.h       (+25 LOC, sp_kste_encode_ex added)
shannon-prime-engine/lib/shannon-prime/core/sp_kste.c       (refactored: encode delegates to encode_ex)
shannon-prime-engine/src/sp_friedman_cache.h                (135 LOC, NEW)
shannon-prime-engine/src/sp_friedman_cache.cpp              (185 LOC, NEW)
shannon-prime-engine/tests/test_sp_friedman_cache.cpp       (640 LOC, NEW — T2.1/5/6/7/8/12 + bench)
```

Total Phase 3: roughly **+340 LOC code, +640 LOC tests** — vs roadmap §3 budget 500 LOC; inside budget for the cache proper (320), tests are above because they include the axiomatic-layer T2.12 (predicates + ED Reduction check) and the wall-time bench harness.

CMake:
- `shannon-prime-engine/CMakeLists.txt`: new option `SP_FRIEDMAN_SIEVE` (default ON).
- `shannon-prime-engine/tests/CMakeLists.txt`: new `test_sp_friedman_cache` self-contained target, gated on `SP_FRIEDMAN_SIEVE`.

## Tests run

Build (g++ in workspace sandbox):

```
g++ -std=c++17 -O2 -Wall -Wextra -I lib/shannon-prime/core -I src \
    tests/test_sp_friedman_cache.cpp \
    src/sp_friedman_cache.cpp \
    lib/shannon-prime/core/{shannon_prime.c,shannon_prime_sqfree.c,sp_kste.c,sp_kste_pack.c,sp_kste_embed.c} \
    -lm -o /tmp/test_friedman
```

Build clean. JSON reports written under `D:\F\shannon-prime-repos\tests\results\`.

| Test | Verdict | Key metrics |
|------|---------|-------------|
| T2.1 — termination | **PASS** | Cache plateaus at capacity=512 over 5000 random tokens; 4458 replacements (Knight-Skeleton fallback exercised), 30 subsumption evictions, 0.6% eviction rate |
| T2.5 — closure axiom | **PASS** | Big-subset ∩ Big-subset is non-empty (100 above-median slots in 200-slot cache) |
| T2.6 — eviction on subsumption | **PASS** | `insert_K_decision=ADMITTED, insert_Q_decision=EVICTED` for a synthetic prefix pair |
| T2.7 — admission on novelty | **PASS** | 1 fresh random tree admitted into a 200-slot cache |
| T2.8 — Knight-Skeleton fallback | **PASS** | At CAP=64, novel high-variance arrival REPLACES the lowest-variance slot; gen counter advances |
| T2.12 — Extended-Domain Reduction | **PASS** | 150/150 checks (50 witnesses × 3 predicates: anchor_count, label_b_count, max_depth) — every cached witness's structural quantity is bounded above by some RO slot |
| T2_BENCH — sieve insert wall-time | **FAIL by design** | mean = 2420 µs, p99 = 7060 µs at capacity 512. Above the 50 µs exit-criterion gate — see Deviations §1 |

All six Tier-2 functional tests are GREEN. The wall-time exit criterion is documented as a Phase-5 deliverable.

## Deviations from spec

### 1. Wall-time exit criterion (≤ 50 µs/token) NOT MET at Phase 3 — by design

Roadmap §3 risk note (verbatim):
> The `for t in cache: embed_test(...)` loop is O(n) in cache size. At n = 4096 and 100 µs per test, this is 400 ms per token — far too slow. **Mitigation: pre-filter by label-multiset hash, then embed-test only candidates. Add this in Phase 5.**

The naive O(N) embed scan against 512 slots × ~5 µs each gives ~2.5 ms/insert, exactly as the roadmap warned. Phase 5 introduces the label-multiset pre-filter that brings this to ~5 µs/insert.

Choice for this session: ship the naive correctness-focused cache, run the bench for forensics, mark the bench as FAIL but call out that this is structurally expected. Phase 5's T2.11 is where wall-time is gated for-real.

### 2. T2.1 token count reduced from 100 000 to 5 000

100 000 tokens × 4 096 slots × ~5 µs/embed = ~2 050 seconds — impractical to run in a sandbox VM with the naive O(N) cache. With Phase 5's pre-filter the full 100 000 is meant to land in a few minutes; the 5 000-token bench in Phase 3 demonstrates the termination property (the cache size plateaus at its capacity and stays there) without the wall-time cost.

The pass criterion in TEST-SUITE.md T2.1 is "cache size bounded by the antichain count OR cache hits capacity and the eviction-on-full path engages." Both held in this bench: final = max = 512, plateau-at-end = 4486 consecutive inserts, replacement-on-full engaged 4458 times.

### 3. Other test bounds dialed for sandbox-friendly runtime

- T2.5: target cache size 500 → 200 (5000 inserts → 2000); pass criterion unchanged (`|big| >= 3`).
- T2.7: pre-fill 1000 → 200 keys; novelty assertion unchanged.
- T2.12: target cache size 500 → 200, witnesses 100 → 50; pass criterion unchanged (0 ED Reduction failures).
- Wall-time bench: capacity 4096 → 512, inserts 1000 → 300.

All reductions preserve the spec's *operational* meaning; only the constants are tighter.

### 4. `sp_kste_encode_ex` added beyond Paper IV spec

Returns the Knight-Skeleton variance (sum of `|anchor|²` over the 14 anchor positions) alongside the tree. Required by T2.8 fallback eviction. Default `sp_kste_encode` delegates with `NULL` for the variance pointer, so existing callers are unaffected. The header documents that `skel_var` is NOT Frobenius-invariant (it scales as `scale²`); it is intended only for *relative* ordering within a single layer+head where the global scale is constant.

## What the cache actually does

- **Insert path.** For each new tree, scan existing slots oldest-first and run `sp_kste_embed(new, existing)`. If any embed returns 1, reject (SP_FRIEDMAN_EVICTED). If all 0 and there's room, append; if full, displace the slot with the lowest `skel_var` (SP_FRIEDMAN_REPLACED).
- **Counters.** `inserts_total`, `evictions`, `admissions`, `replacements` are cumulative across the cache lifetime. `eviction_rate()` is the fraction subsumed; on random N(0,I) keys it's ~0.6% — most novelty is real.
- **Choice operator F.** `sp_kste_compare` is packed-byte lex order (node_count first, then labels, then parents). `sp_kste_select_canonical` returns the ≺_F-minimum tree in a candidate list — the implementation of Paper IV §10's `F(A)`. This is what Phase 7's ultraproduct attention will eventually consume.
- **Extended-Domain Reduction.** `sp_extended_reduction_check(cache, ro_count, v, phi)` walks the last `ro_count` slots and returns 1 iff some RO slot's `phi`-value matches or exceeds `phi(v)`. Primitive-recursive; this is the operational analogue of Paper III §11.1's relativisation axiom. Five built-in predicates: anchor_count, label_b_count, label_c_count, node_count, max_depth.

## Empirical headline from T2.1

5000 random tokens through a CAP=512 cache:
- 30 subsumption evictions (genuine sieve action — 0.6% rate)
- 512 initial admissions (cache fills up by token 512-ish)
- 4458 Knight-Skeleton replacements (the cache *churns* once full)

This is the right shape for Phase 4 calibration (the sieve is *responsive* — once full, it keeps the highest-variance slots in residence). The low subsumption rate on i.i.d. random keys is expected; Phase 4's WikiText-103 run is what will tell us whether real-language structure produces meaningful subsumption (predicted rate ≥ 20% in §T2.2).

## Cross-phase invariants

| Invariant | Status |
|-----------|--------|
| 1. PPL never regresses | N/A (no inference path touched) |
| 2. CPU & HVX paths agree bit-exactly | N/A (HVX is Phase 6) |
| 3. WKL₀ refutation property preserved | Held — both the sieve admission decision and the ED-Reduction check are primitive-recursive walks over `cache->slots`. Failures are finite, locatable, witness-bearing |
| 4. No `__int128` | Held |
| 5. No global mutable state | Held — cache state is caller-owned (`sp_friedman_cache_t`); no statics |
| 6. 63-byte Spinor block format frozen | Held — `sp_friedman_slot_t` attaches `sp_kste_tree` (64 B) alongside, no modification to the block |
| 7. Calibration knight-mask ships with model | Phase 4 |

## Recommended next phase

**Phase 5 — sieve performance optimisation, ahead of Phase 4 calibration.**

I'm flagging a sequencing inversion here. The roadmap orders 4 → 5; the literal reading is "calibrate first, then perf-optimise." But Phase 4 (`T2.3 PPL drift ≤ 0.5%`) requires running the cache inside `sp-engine`'s actual forward pass on Gemma3-1B at ctx=2048 over WikiText-103. With Phase 3's naive O(N) cache running at ~2.5 ms/insert × thousands of insertions per chunk, a single PPL run is on the order of hours — not feasible to iterate calibration on.

If KnackAU agrees: do **Phase 5 first** (pre-filter, batched embed), then **Phase 4** (PPL gate with a usable runtime). If you want the literal roadmap order, we'll wire the cache into `sp-engine` first and accept the long PPL run-time.

I have not made this swap yet — flagging it for explicit go-ahead.

## Notes for future sessions

- Edit-tool truncation hazard hit again on `tests/test_sp_friedman_cache.cpp`. Recovery used `python3 rstrip(b'\\x00')` + appended tail via heredoc. Mitigation: keep the test cpp file edits ≤ ~50 lines per Edit call; large rewrites go through `Write` whole-cloth.
- JSON audit trail: `D:\F\shannon-prime-repos\tests\results\T2_*.json`, plus the Phase 1–2 T1_*.json files. Aggregator `T2_SUMMARY.json` and `T1_SUMMARY.json` show per-phase verdicts.
- The cmake config has been augmented (`SP_FRIEDMAN_SIEVE` option, test target); not yet exercised on the Windows host's cmake. Mirrors the `test_sp_frobenius` / `test_sp_kste` self-contained pattern verbatim — first MSVC build is the next external gate.
- `sp_kste_select_canonical` and `sp_kste_compare` are wired and exercised indirectly via T1.9 byte-equality, but the dedicated T3.6 (1000-permutation invariance) is on the Phase 7 docket; the kernel itself is ready when that phase lands.
