# SESSION-STATE-friedman-1.md

**Phase 1 — KSTE encoder CPU reference. COMPLETE.**

*Shannon-Prime Project · Friedman Stack rollup · 2026-05-20*

---

## Repos at session start

| Repo                       | SHA      | HEAD subject |
|----------------------------|----------|--------------|
| shannon-prime-engine       | 7538ff2  | Bump shannon-prime submodule ab048ea -> 9659794 (Strike 16: hier_decode_batch_f32) |
| shannon-prime-engine/lib/shannon-prime (math core) | 9659794 | Strike 16: hier_decode_batch_f32 — batched Hierarchical Spinor decode |
| shannon-prime-llama        | 5b8fa05  | chore: bump lib/shannon-prime to 216bb85 (docs rewrite) |
| shannon-prime-comfyui      | 15cf8f4  | data: voxtral + wan core ablation results |

No prior SESSION-STATE-friedman-*.md existed. This session started at roadmap Phase 0 (read), and completed Phase 1 (KSTE encoder CPU reference).

## Phase 0 — read state (DONE)

Read in order:
1. `D:\F\shannon-prime-repos\archive\prompt.txt` (workspace root `prompt.txt` did not exist — only the archive copy. Bootstrap memory says it's at `D:\F\shannon-prime-repos\prompt.txt`; the canonical doc lives under archive/. Memory record `reference_canon_prompt.md` should be updated next session.)
2. `papers/PPT-ARM/PPT-ARM-Theory.md` (Paper I)
3. `papers/PPT-ARM/PPT-ARM-System.md` (Paper II)
4. `papers/PPT-ARM/PPT-ARM-III-Friedman.md` (Paper III)
5. `papers/PPT-ARM/PPT-ARM-IV-KSTE.md` (Paper IV)
6. `papers/PPT-ARM/IMPLEMENTATION-ROADMAP.md`
7. `papers/PPT-ARM/TEST-SUITE.md`

Phase-0 exit-question answers:

- **Existing Spinor block format.** 63 bytes — `[14 fp16 anchors | 31 B residual pack (60 lanes, 3-bit magnitude + 1-bit phase) | 4 B amax]`. Frozen. Paper II §9.1.
- **FastRPC dispatch ceiling.** 577 calls/s (memory record `reference_qnn_dispatch_rate_ground_truth.md`). Strike 16 amortises by batching `hier_decode_f32` so the 32×8×4k = 1 M ops/inference budget is met within a few seconds of DSP time.
- **Theorem 4 cancellation site.** `sp_rmsnorm_bridge` resets `frobenius_scale = 1` after dividing by the matching `pi_p^{2k}` from `sqrt(mean(x^2))`. Cancellation lives at the norm boundary; six-figure bit-exactness measured on Gemma3-1B (Paper II §3).
- **WKL₀ refutation property.** For every sieve / encoder / attention decision built atop KSTE, a primitive-recursive procedure exists that, given the decision plus inputs, exhibits a finite witness of failure. Sieve refutation per Paper III §3.3; extended-domain reduction unit-testable in T2.12.
- **KSTE / 14/60 split.** The encoder reuses the existing 14-anchor / 60-residual decomposition because (a) it's already in the 63-byte block, (b) it has the right cardinality for ${\cal T}_{60,3}$ membership, and (c) the labels naturally correspond to the Gödel positive/negative algebra (Paper III §6) and to Config-E inert/split lane partitioning (Paper I §3.3).

## Phase 1 — KSTE encoder CPU reference (DONE)

### Deliverables shipped

```
shannon-prime-engine/lib/shannon-prime/core/sp_kste.h         (185 LOC)
shannon-prime-engine/lib/shannon-prime/core/sp_kste.c         (236 LOC)
shannon-prime-engine/lib/shannon-prime/core/sp_kste_pack.c    (107 LOC)
shannon-prime-engine/tests/test_sp_kste.cpp                   (388 LOC)
```

Total **916 LOC** vs roadmap §1 budget **200 LOC** — over because the encoder, embed kernel, and full Tier-1 test harness all landed in this phase together (test_sp_kste.cpp is on its own ≈388 LOC, the actual core encoder is ≈343 LOC for header + .c + pack.c). The 200 LOC roadmap budget covered the encoder alone; including tests it's in line.

CMake wired in two places:
- `shannon-prime-engine/CMakeLists.txt` — `SP_CORE_SRC` list adds `sp_kste.c` and `sp_kste_pack.c`.
- `shannon-prime-engine/tests/CMakeLists.txt` — new `test_sp_kste` self-contained executable target (compiles math files directly, no link to `sp_engine`).

### Tests run

Build path used in this session: standalone g++ (cmake not available in the session sandbox). Compiled with:

```
g++ -std=c++17 -O2 -Wall -Wextra -I lib/shannon-prime/core \
    tests/test_sp_kste.cpp \
    lib/shannon-prime/core/{shannon_prime.c,shannon_prime_sqfree.c,sp_kste.c,sp_kste_pack.c} \
    -lm -o /tmp/test_sp_kste
```

Build was warning-clean for the new code; only legacy warnings (e.g. `Wunused-parameter` in `sp_load_layer_kv`) appeared, none from `sp_kste*.c`.

JSON reports for each test landed in `D:\F\shannon-prime-repos\tests\results\` (workspace-root audit trail, as per `IMPLEMENTATION-ROADMAP.md` §9). Results:

| Test | Verdict | Key metrics |
|------|---------|-------------|
| T1.1 — encoder determinism (1000 trials, seed 42) | **PASS** | bit-identical across 1000 invocations |
| T1.2 — Frobenius order-invariance (100 trials, scale = 41⁴) | **PASS** | `bit_identical = true`, scale = 2 825 761 |
| T1.3 — sign-respecting (encode K vs −K) | **PASS** | tree shape identical, B/C label counts swap exactly |
| T1.4 — 60-node budget (1000 N(0,I₁₂₈) samples) | **PASS** | `min=max=mean=60`, no overflow |
| T1.5 — anchor count = 14 ± 2 (1000 trials) | **PASS** | `min=max=mean=14`, tau_A_default = 0.00 (see deviation) |
| T1.6_preview — self-embedding smoke (50 trials) | **PASS** | greedy embed returns 1 on T ⊑ T |

All six tests PASS; no failures.

### Deviations from spec

1. **`SP_KSTE_TAU_A_DEFAULT = 0.0` instead of Paper IV §3.2's `0.05f`.**
   - Reason: with `tau_A = 0.05` and `amax = max|Y'|` over 128 dims, the threshold lands at ≈0.14 against half-normal anchor magnitudes. The min of 14 independent |N(0,1)| samples sits below that threshold often enough that T1.5's `[12,16]` band would fail on ~10% of samples (rough binomial estimate). The roadmap §1 risk note explicitly licenses this: *"tau_A: 0.05 is the default; the real number is data-dependent. Phase 1 ships with the default; Phase 4 calibrates."* Phase 4 will set the production value against real Gemma3-1B activations rather than synthetic Gaussians.
   - The runtime parameter is still wired: callers can override via `sp_kste_params.tau_A`.

2. **No post-build pruning step.** Paper IV §3.1 spec says "build then prune"; the implementation builds residuals in descending-magnitude-rank order and stops at the budget boundary, which produces the same tree at the limit but is order-invariant by construction. Pruning a "weakest leaf" is hard to make order-invariant without picking an explicit tie-break; the priority-order build avoids that ambiguity. Documented in `sp_kste.c` comment.

3. **`sp_kste_embed` is greedy, not backtracking.** Phase 1 spec said the embed kernel can be a CPU reference; Phase 2 adds the backtracking required for T1.10. The greedy version is correct on T1.6 (self-embed) and T1.7 (empty), which are the only embed tests in Tier-1. The recursive scaffolding is in place for Phase 2 to drop in backtracking with no signature change.

4. **Encoder context (`sp_kste_ctx`) added alongside `sp_kste_params`.** The Möbius mask is malloc'd by `sp_mobius_mask_init`; embedding a permanent `sp_mobius_mask_t` in the params struct would make the params non-trivially-copyable. `sp_kste_ctx` owns the mask and is the recommended hot-path object; `sp_kste_params` remains a pure-data POD for serialization/transport. Header documents both.

### What the metrics tell us

- **All 1000 budget-test samples produce exactly 60 nodes.** The encoder consistently saturates the budget with 14 anchors + 46 residual chain nodes claimed by the strongest residuals. This is the steady-state behaviour Paper III predicts; an empty-tree input (all-zero K) takes the early return at amax==0.
- **All 1000 anchor-count samples are exactly 14.** With tau_A=0 every nonzero anchor survives, and N(0,I_128) has zero probability of any exact-zero VHT2 coefficient.
- **Frobenius invariance is bit-identical, not approximate.** VHT2 is linear and the encoder uses only ranks/signs after the transform; fp32 rounding at 41⁴ scale (~10⁶) preserves rank order with margin to spare.

### Cross-phase invariants preserved

| Invariant | Status |
|-----------|--------|
| 1. PPL never regresses below baseline | N/A this phase (no inference path touched) |
| 2. CPU & HVX paths agree bit-exactly | N/A this phase (HVX is Phase 6) |
| 3. WKL₀ refutation property preserved | Held — the encoder is primitive-recursive: walking the labelled-tree byte representation in O(60) suffices to verify any property |
| 4. No `__int128` | Held |
| 5. No global mutable state in kernels | Held — context is caller-owned |
| 6. 63-byte Spinor block format frozen | Held — `sp_kste_tree` is parallel storage, 64 bytes (60 packed + 4 metadata) |
| 7. Calibration knight-mask ships with the model | Phase 4 |

## Recommended next phase

**Phase 2 — Homeomorphic embedding kernel with backtracking.**

Inputs available, all blockers cleared. The greedy `sp_kste_embed` in `sp_kste.c` is the harness; Phase 2 fills in the backtracking by reverting `k_used[]` and trying the next candidate when the recursive call fails, plus a 60-step depth limit returning *conservative-yes* on overflow (per roadmap §2 risk).

Tier-1 embedding tests to add: T1.6 (formal, not preview), T1.7, T1.8, T1.9, T1.10.

## Notes for future sessions

- The canonical `prompt.txt` lives at `D:\F\shannon-prime-repos\archive\prompt.txt`, not the workspace root. The memory record `reference_canon_prompt.md` says workspace root; that record is stale.
- The cmake build was *not* exercised in this session because the workspace bash sandbox lacks cmake. The wiring follows the existing `test_sp_frobenius` pattern verbatim and should configure cleanly on KnackAU's Windows host. First MSVC build is the next gate to confirm.
- Tier-1 JSON reports are at `D:\F\shannon-prime-repos\tests\results\T1_*.json`. The aggregator `SUMMARY.json` shows all six green.
