# Implementation Roadmap — The Friedman Stack

**Companion to Papers III–IV. A new-session-ready phase plan.**

*Shannon-Prime Project · 2026-05-19*

---

## How to read this document

This roadmap is a contract between sessions. Each phase has:

- **Goal** — one sentence stating what success looks like.
- **Inputs** — files / tags that must exist before the phase starts.
- **Deliverables** — files / commits the phase produces.
- **Tests** — references to `TEST-SUITE.md` test IDs that gate the phase.
- **Exit criteria** — explicit pass conditions for moving to the next phase.
- **Estimated LOC** — rough budget for sanity-checking scope.
- **Risk** — what could go wrong and what to watch for.

**Hard rule:** do not begin phase $N+1$ until all exit criteria for phase $N$ are met. If a phase fails, *stop and report*, do not silently move on.

---

## Phase 0 — Read existing state (estimated 30 min, no code)

**Goal.** A new session is bootstrapped with the current state of the engine, the math core, and Papers I–IV.

**Inputs.**

- `D:\F\shannon-prime-repos\` (workspace root, 4 sibling repos).
- `D:\F\shannon-prime-repos\papers\PPT-ARM\PPT-ARM-Theory.md` (Paper I).
- `D:\F\shannon-prime-repos\papers\PPT-ARM\PPT-ARM-System.md` (Paper II).
- `D:\F\shannon-prime-repos\papers\PPT-ARM\PPT-ARM-III-Friedman.md` (Paper III).
- `D:\F\shannon-prime-repos\papers\PPT-ARM\PPT-ARM-IV-KSTE.md` (Paper IV).
- `D:\F\shannon-prime-repos\prompt.txt` (canon philosophy doc).

**Deliverables.** A 1-page session-state summary written to `SESSION-STATE-friedman-N.md` with:
- Last commit SHA of each of the 4 repos.
- Which phases of this roadmap are complete / in-progress / blocked.
- Any deviations from the spec discovered while reading.

**Tests.** None — this is a read phase.

**Exit criteria.** Session can answer, without re-reading, the following:
- What is the existing Spinor block format and its byte layout?
- What is the FastRPC dispatch ceiling and how does Strike 16 relate?
- What is Theorem 4 and where in the engine does its cancellation live?
- What does Paper III claim about the WKL₀ refutation property?
- What is KSTE and why does it reuse the existing 14/60 split?

**Risk.** Skipping this phase causes every later phase to re-derive context. Don't skip.

---

## Phase 1 — KSTE encoder, CPU reference (estimated 200 LOC, ~1 day)

**Goal.** A deterministic, order-invariant encoder $\Phi: \mathbb{R}^{128} \to \mathcal{T}_{60,3}$ written in plain C, with no SIMD, that passes T1.1–T1.5.

**Inputs.**
- Phase 0 complete.
- `lib/shannon-prime/sp_vht2.h`, `sp_vht2_block_q8.h` (existing).

**Deliverables.**
- `lib/shannon-prime/sp_kste.h` — public API.
- `lib/shannon-prime/sp_kste.c` — encoder + embedding test (CPU reference).
- `lib/shannon-prime/sp_kste_pack.c` — bit-packing helpers (2-bit labels, 6-bit parents).
- `tests/test_sp_kste.cpp` — Tier-1 tests.

**Tests.**
- **T1.1** Determinism (1000 trials).
- **T1.2** Order-invariance under Frobenius shim.
- **T1.3** Sign-respecting.
- **T1.4** 60-node budget enforced.
- **T1.5** Anchor count = 14 ± 2.

**Exit criteria.**
- All five Tier-1 tests green on Windows MSVC and Linux GCC.
- Encoder runs in $\le 5$ µs on a single thread (CPU reference, no SIMD).
- No floating-point arithmetic in the inner loop after the initial fp16→rank conversion.

**Risk.**
- Bit-packing bugs. Watch for endian issues on big-endian targets (irrelevant for x86/ARM, but document the assumption).
- `tau_A` calibration: 0.05 is the default; the real number is data-dependent. Phase 1 ships with the default; Phase 4 calibrates.

---

## Phase 2 — Homeomorphic embedding kernel, CPU (estimated 400 LOC, ~1 day)

**Goal.** A correct, complete embedding-test kernel `sp_kste_embed(Q, K)` that handles all 60-node cases and backtracks when greedy fails.

**Inputs.** Phase 1 complete.

**Deliverables.**
- `sp_kste_embed_cpu.c` — backtracking embedding test.
- Augment `test_sp_kste.cpp` with embedding tests.

**Tests.**
- **T1.6** Embed of identity tree: $T \preceq T$ always.
- **T1.7** Embed of empty subtree: $\rho \preceq T$ for any non-empty $T$.
- **T1.8** Embed transitivity: $T_1 \preceq T_2 \preceq T_3 \Rightarrow T_1 \preceq T_3$.
- **T1.9** Embed antisymmetry on canonical forms: $T_1 \preceq T_2 \preceq T_1 \Rightarrow T_1 = T_2$.
- **T1.10** Backtracking necessity: a constructed case where greedy fails but backtracking succeeds.

**Exit criteria.**
- All Tier-1 embedding tests green.
- Worst-case embedding test runs in $\le 100$ µs on CPU (the budget for the HVX kernel of Phase 6).
- No undefined behaviour under MSVC `/W4 /WX` or GCC `-Wall -Wextra -Werror`.

**Risk.**
- Backtracking explosion on pathological trees. Add a 60-step depth limit and return *conservative-yes* (treat as embedding) when the limit is hit. Document this as a defined non-failure mode.

---

## Phase 3 — Friedman sieve cache structure (estimated 500 LOC, ~2 days)

**Goal.** A `sp_friedman_cache_t` data structure with admit/evict/sieve semantics, wired behind a feature flag.

**Inputs.** Phases 1, 2 complete.

**Deliverables.**
- `shannon-prime-engine/src/sp_friedman_cache.h`
- `shannon-prime-engine/src/sp_friedman_cache.cpp`
- `tests/test_sp_friedman_cache.cpp`
- CMake option `SP_FRIEDMAN_SIEVE`.

**Tests.**
- **T2.1** Termination: 100k random tokens → cache size bounded.
- **T2.5** Closure axiom: big ∩ big = big across all pairs in the cache.
- **T2.6** Eviction-on-subsumption: synthetic trees where Q ⪯ K → Q evicted.
- **T2.7** Admission-on-novelty: synthetic trees where Q ⊀ any K_i → Q admitted.
- **T2.8** Knight-Skeleton fallback: cache full + novel token → variance-based eviction.
- **T2.12** Extended-Domain Reduction invariant — the axiomatic layer's unit test. ★

**Exit criteria.**
- All Tier-2 cache-level tests green.
- Cache write path runs in $\le 50$ µs per token (CPU, single-threaded).
- Memory: 192 B per slot × 4096 slots = 786 KB per layer × 26 layers = **20 MB** total cache (acceptable on phone).

**Risk.**
- The `for t in cache: embed_test(...)` loop is $O(n)$ in cache size. At $n = 4096$ and 100 µs per test, this is 400 ms per token — far too slow. Mitigation: pre-filter by label-multiset hash, then embed-test only candidates. Add this in Phase 5.

---

## Phase 4 — Calibration and PPL gating (estimated 300 LOC + benchmarks, ~2 days)

**Goal.** Calibrate `tau_A`, `alpha`, and the cache capacity to pass T2.3 (PPL drift $\le 0.5\%$ on Gemma3-1B at ctx=2048).

**Inputs.** Phase 3 complete.

**Deliverables.**
- `scripts/calibrate_kste.py` — sweep over (tau_A, alpha) on a calibration corpus.
- `scripts/run_ppl_sieve.bat` — Windows runner producing PPL with sieve on/off.
- `docs/KSTE-CALIBRATION.md` — recorded calibration ledger.

**Tests.**
- **T2.2** Eviction rate $\ge 20\%$ at steady state on WikiText-103.
- **T2.3** PPL drift $\le 0.5\%$ — **THE GATE**.

**Exit criteria.**
- T2.3 passes at the calibrated `(tau_A*, alpha*)`.
- Calibration ledger committed; the chosen values become the default in `sp_kste.h`.

**Risk.**
- T2.3 may not pass. If the best-case calibration produces PPL drift $> 0.5\%$, the encoder is insufficiently discriminative. **STOP.** Report the failure. Tier 3 (ultraproduct attention) is *not* a recovery for an encoder failure — return to Paper IV §9 risk #1 and consider widening labels to 4-bit, which requires re-spec of the Spinor block.

---

## Phase 5 — Sieve performance optimization (estimated 600 LOC, ~3 days)

**Goal.** Sieve write path runs in $\le 5$ µs per token at cache size 4096.

**Inputs.** Phase 4 passing T2.3.

**Deliverables.**
- Label-multiset hash pre-filter in `sp_friedman_cache.cpp`.
- Bloom-filter or fingerprint pre-filter on tree shape.
- Batched embed-test path: test new $T_Q$ against a window of $\ge 16$ candidates per HVX dispatch.

**Tests.**
- **T2.9** Pre-filter correctness: no false negatives (cache stays sound).
- **T2.10** Pre-filter precision: $\ge 90\%$ of candidates correctly excluded.
- **T2.11** Wall-time at $n = 4096$: $\le 5$ µs / token average, $\le 50$ µs worst case.

**Exit criteria.**
- All three optimization tests green.
- PPL still passes T2.3 (no regression).

**Risk.**
- Pre-filter false negatives cause sieve to admit tokens that should be evicted → cache fills faster. Mitigation: log discrepancies between pre-filter and full test during dev builds; should be zero.

---

## Phase 6 — HVX kernel for the embedding test (estimated 800 LOC, ~5 days)

**Goal.** `sp_hex_kste_embed` IDL method shipped, FastRPC dispatch validated on S22U.

**Inputs.** Phase 5 complete.

**Deliverables.**
- `shannon-prime-engine/src/sp_hex_kste_embed.idl`
- `shannon-prime-engine/src/sp_hex_kste_embed_imp.c` — V69 implementation.
- ARM-side dispatcher in `sp_friedman_cache.cpp`.
- Build recipe in `docs/HEXAGON-BUILD.md` (Strike-style increment).

**Tests.**
- **T6.1** HVX kernel parity vs CPU: 10000 random pairs, bit-identical result.
- **T6.2** Single-dispatch latency: $\le 25$ µs end-to-end (FastRPC + HVX + return).
- **T6.3** Batched dispatch: 16-test batch in $\le 200$ µs.
- **T6.4** First-light on S22U: prefill on Qwen3-4B with sieve on, no crash.

**Exit criteria.**
- All HVX tests green.
- First-light passes; PPL with sieve+HVX matches the CPU reference exactly.

**Risk.**
- Dispatch density wall (as in Strike 15a). Mitigation: batched test is mandatory; do not ship single-dispatch path. Refer to the Strike 16 design notes for the queueing pattern.
- VTCM contention with existing compress/decode pipeline. Mitigation: stage embedding tests *between* attention forward steps, not during.

---

## Phase 7 — Ultraproduct attention prototype (estimated 600 LOC, ~3 days)

**Goal.** A working `--ultraproduct-attn=principal` path that produces sensible outputs on a 16-token toy.

**Inputs.** Phase 6 complete.

**Deliverables.**
- `shannon-prime-engine/src/sp_ultraproduct_attn.h`
- `shannon-prime-engine/src/sp_ultraproduct_attn.cpp`
- `tests/test_sp_ultraproduct_attn.cpp`

**Tests.**
- **T3.1** Principal ⇒ Top-1 attention.
- **T3.2** Łoś property on hand-crafted toy.
- **T3.6** Choice operator canonicality — `sp_kste_select_canonical` returns the same tree across 1000 invocations regardless of input order.

**Exit criteria.**
- Both T3.1, T3.2 green.
- Toy 16-token output makes semantic sense (greedy decode produces coherent text on a fixed prompt).

**Risk.**
- Ultraproduct attention is hard-attention; gradient flow is gone. This is *only* an inference-path experiment. Document this prominently; do not let a future session try to enable it during training.

---

## Phase 8 — Long-context benchmarks (estimated benchmarks only, ~3 days)

**Goal.** Decide whether ultraproduct attention belongs in the default path.

**Inputs.** Phase 7 complete.

**Deliverables.**
- `scripts/run_longbench.bat` — runs LongBench at ctx ∈ {2k, 8k, 32k}.
- `scripts/run_ruler.bat` — runs RULER long-context probes.
- Results in `docs/RESULTS-ULTRAPRODUCT.md`.

**Tests.**
- **T3.3** PPL on LongBench: $\Delta \le 3\%$ either direction.
- **T3.4** RULER at ctx=32k: match or beat softmax.
- **T3.5** Wall-time within 20% of softmax baseline.

**Exit criteria.**
- T3.3, T3.4, T3.5 results recorded.
- A *decision*: ship as default / opt-in / shelved. The decision is committed as a one-line entry in `docs/DECISIONS.md`.

**Risk.**
- Even if ultraproduct attention loses, the framework's value is the *primitive*, not the headline number. Don't shelve unless all three tests cleanly lose.

---

## Phase 9 — Documentation and paper revision (estimated docs only, ~2 days)

**Goal.** Papers III and IV updated to reflect measured results, not predicted ones. `docs/AUDIT-TRAIL.md` complete.

**Inputs.** Phase 8 complete.

**Deliverables.**
- Updated `PPT-ARM-III-Friedman.md` (replace Section 8 predictions with measured outcomes).
- Updated `PPT-ARM-IV-KSTE.md` (replace Section 6 schedule with results).
- `docs/AUDIT-TRAIL.md` — every test report from §6 of Paper IV, in JSON form.
- Updated LaTeX and PDF for both papers.

**Tests.** None — documentation phase.

**Exit criteria.**
- PDFs rebuild cleanly via `make papers`.
- Audit trail contains a JSON report for every test ID in `TEST-SUITE.md`.

**Risk.** None worth flagging at this point.

---

## Cross-phase invariants

The following must remain true through every phase:

1. **PPL never regresses below the baseline.** Each phase's exit gate explicitly re-runs the baseline PPL with the new code disabled. If it differs, you have introduced a latent bug.

2. **CPU and HVX paths agree bit-exactly.** Every kernel ships with both implementations and a parity test. CPU is the reference.

3. **WKL₀ refutation property is preserved.** Every sieve or attention decision must have a primitive-recursive procedure that, given the decision and the inputs, validates correctness. Add this procedure to the test suite as you add the kernel.

4. **No `__int128`.** Inherited invariant from Paper II Phase 9 (CRT-NTT). The new kernels follow this rule.

5. **No `localStorage` / no global mutable state in kernels.** Inherited from existing engine conventions; relevant for the threading model.

6. **The 63-byte Spinor block format is frozen.** All new structures attach *alongside* it, not within it. Block format changes are a major version event.

7. **Calibration knight-mask ships with the model.** Build artefacts include the mask; runtime loads from artefact, not from re-calibration. (Re-calibration is a tooling pass, not an inference-time operation.)

---

## Estimated total budget

| Phase | LOC | Days | Cumulative |
|------:|----:|-----:|-----------:|
| 0 | 0 | 0.5 | 0.5 |
| 1 | 200 | 1 | 1.5 |
| 2 | 400 | 1 | 2.5 |
| 3 | 500 | 2 | 4.5 |
| 4 | 300 | 2 | 6.5 |
| 5 | 600 | 3 | 9.5 |
| 6 | 800 | 5 | 14.5 |
| 7 | 600 | 3 | 17.5 |
| 8 | 0 (bench) | 3 | 20.5 |
| 9 | 0 (docs) | 2 | 22.5 |
| **Total** | **3400** | **23 days** | — |

23 days is the optimistic estimate for a single full-time human-equivalent operator (which means: Claude + KnackAU paired). Triple it for first-time issues, hardware setup time, and idle waiting on benchmarks.

---

## When to stop

Stop after any phase whose exit gate cleanly fails *twice*. A single fail is acceptable; iterate. Two fails on the same phase means the spec is wrong — return to Paper III/IV and revise, do not push code through.

The framework is engineered to fail loudly and finitely. Trust the failure mode.

---

## See also

- `TEST-SUITE.md` — every test ID referenced above, with inputs, expected outputs, and tolerances.
- `BOOTSTRAP-PROMPT.md` — the prompt to load into a new session to begin Phase 0.
- `PPT-ARM-III-Friedman.md` — the theory.
- `PPT-ARM-IV-KSTE.md` — the system spec.
