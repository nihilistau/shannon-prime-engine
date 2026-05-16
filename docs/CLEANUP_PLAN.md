# Engine + Server Cleanup Plan

**Status:** Multi-stage. Stage 1 lands in this commit. Stages 2-4 staged.
**Goal:** Make Theory-First (Frobenius + Sato-Tate) the only shipping path. Remove competing flags, dead verbs, and test-only scaffolding.

## Canonical end-to-end path

```
sp-engine run --model phi3.gguf --frobenius-quant --prompt "Hello"
sp-engine run --model phi3.gguf --sato-tate-mix 2,2,41,8 --prompt "Hello"
sp-engine chat --model phi3.gguf --frobenius-quant
sp-engine perplexity --model phi3.gguf --frobenius-quant --corpus wikitext.txt
sp-engine server --model phi3.gguf --frobenius-quant --port 8080
```

Everything else either supports this path (`version`, `info`, `encode/decode` for debugging) or is being removed.

## Shipping verbs (KEEP)

| Verb | Purpose | Notes |
|--|--|--|
| `version` | Print version + commit | KEEP |
| `banner` | Show submodule SHAs | KEEP |
| `info` | Inspect a GGUF | KEEP |
| `run` | One-shot generate | KEEP (primary) |
| `chat` | Interactive | KEEP (primary) |
| `perplexity` | Bench PPL on a corpus | KEEP (primary; bench for Paper D) |
| `server` | HTTP/OpenAI-compat API | KEEP (Phase 2 target) |
| `encode` | Tokenize text → IDs | KEEP (tiny utility) |
| `decode` | IDs → text | KEEP (tiny utility) |
| `embed` | Token embedding lookup | KEEP (utility) |

## Verbs to REMOVE (test-only, no shipping value)

| Verb | Why removable | Stage |
|--|--|--|
| `kv_smoke` | Synthetic K/V correlation test. Doesn't load a real model. Replaced by `test_sp_frobenius` C unit test. | Stage 2 |
| `block1` | Runs just layer-0. Debug-only. Replaceable by `run --max-layers 1` if needed. | Stage 2 |
| `logits` | Prints logit stats. Replaceable by `run --print-logits` flag. | Stage 2 |
| `prefill` | Runs prefill + reports cache correlation. Diagnostic-only. Replaceable by `perplexity` with `--correlation` flag. | Stage 2 |
| `cache_ppl` | Perplexity + cache correlation. Consolidate into `perplexity --correlation`. | Stage 2 |

## Flags: Theory-First (LEAD)

| Flag | Meaning |
|--|--|
| `--frobenius-quant` | Config B: single-prime Frobenius, calibration-free fp8 (Paper D §3) |
| `--frobenius-quant-p P` | Override split prime (default 41) |
| `--frobenius-quant-k K` | Override Frobenius power (default 8) |
| `--sato-tate-mix p1,k1,p2,k2` | Config E: inert + split mixed precision (Paper D §3.5) |

## Flags: DEPRECATED (Stage 3 — slated for removal)

These flags belong to pre-Theory-First experimental paths. They are NOT removed in Stage 1 — too much load-bearing code consumes them. Stage 3 removes them after the Theory-First forward pass covers all use cases.

| Flag | Status | Reason for removal |
|--|--|--|
| `--sqfree` | DEPRECATED | Subsumed by Theory-First Möbius/sqfree (Paper A §4 + Theorem 2 of suite) |
| `--spinor` | DEPRECATED | Subsumed by Theory-First spinor in V compression |
| `--hierarchical` | DEPRECATED | Subsumed by Theory-First ternary skeleton (Paper B §3.3) |
| `--no-compression` | DEPRECATED | Set `--frobenius-quant-k 0` instead |
| `--no-mobius` | DEPRECATED | Always-on under Theory-First (Theorem 2) |
| `--k-bits` / `--v-bits` | DEPRECATED | Subsumed by `--frobenius-quant-k` |
| `--residual-bits` | DEPRECATED | Subsumed by Sato-Tate split-channel bit-width |
| `--hier-level` | DEPRECATED | Subsumed by Theory-First adaptive depth (Paper A §7) |
| `--hier-res-bits` | DEPRECATED | Subsumed by `--frobenius-quant-k` |
| `--hier-res-bits-v` | DEPRECATED | Subsumed |
| `--hier-skel-bits` | DEPRECATED | Subsumed |
| `--hier-ternary-mask` | DEPRECATED | Subsumed |
| `--ternary-k` / `--ternary-v` | DEPRECATED | Subsumed |
| `--fp8` | DEPRECATED | Subsumed by `--frobenius-quant` (Frobenius IS the calibration-free fp8 path) |
| `--cauchy-mode` etc. | DEPRECATED | Cauchy reset replaced by Poncelet closure (Paper A §7, Theorem 5) |
| `--pe-tier` / `--pe-alpha` | DEPRECATED | Phase 4: replaced by golden-ratio (Stern–Brocot) RoPE (Paper A §9.1) |
| `--system12` / `--s12-threshold` | DEPRECATED | Phase 4: System 1/2 routing folded into Iwasawa adaptive depth (Paper A §9.6) |
| `--crt-split` | DEPRECATED | Subsumed by CRT exact sharding (Paper A §8, Theorem 6) |
| `--moe-curriculum` | DEPRECATED | Phase 4: covered by L-function activation oracle (Paper A §9.4) |
| `--beast <gguf>` | DEPRECATED | Beast Canyon orchestrator becomes a backend choice, not a flag |

## Environment variables

### KEEP

| Var | Purpose |
|--|--|
| `SP_ENGINE_BACKEND` | `cpu` / `cuda` / `vulkan` — backend selection |

### NEW (Stage 1)

| Var | Default | Purpose |
|--|--|--|
| `SP_ENGINE_THEORY_FIRST` | `1` | When `1`, banner shows Theory-First as the active path; when `0`, banner reverts to legacy mode |

### DEPRECATED (Stage 3 — remove)

| Var | Reason |
|--|--|
| `SP_ENGINE_SYSTEM12` | Subsumed |
| `SP_ENGINE_S12_THRESHOLD` | Subsumed |
| `SP_ENGINE_NATIVE` | All paths native under Theory-First |

## CMakeLists cleanup

### Stage 2 — sources to remove from `sp_engine` target

Once Phase 1 (theory-first forward pass) lands, these can leave the build:

- `src/forward.cpp` — ggml-graph forward (replaced by `sp_forward`)
- `src/llama_weights.cpp` — ggml-dependent weight loader (replaced by `sp_weights`)
- `src/forward_native.cpp`, `src/forward_native_context.cpp` — hand-rolled native forward (folded into `sp_forward`)
- `src/speculative_oracle.cpp` — kept under Phase 4 oracle work; gated by new flag

### Stage 4 — vendor directories to slim

- `vendor/ggml/` — keep only the bare minimum needed for GGUF parsing (header-only; no compute kernels)
- `vendor/cpp-httplib/` — KEEP (server)

## Stages

### Stage 1 (this commit) — non-breaking organization

1. CLEANUP_PLAN.md (this doc)
2. Reorder `print_usage()` in `main.cpp` to lead with Theory-First flags
3. Add a clear `DEPRECATED` section header in help text
4. Add `SP_ENGINE_THEORY_FIRST` env var (default 1) — informational banner only
5. No functional removals yet

### Stage 2 — remove dead verbs

1. Delete `kv_smoke`, `block1`, `logits`, `prefill` verb implementations
2. Consolidate `cache_ppl` into `perplexity --correlation`
3. Remove their help text + dispatch

### Stage 3 — remove deprecated flags

1. Strip flag parsers from `parse_config_flag` for all DEPRECATED items
2. Remove the corresponding `Config` struct fields
3. Remove the underlying code paths (forward.cpp, llama_weights.cpp, etc.)
4. Update README + ROADMAP

### Stage 4 — slim vendor + final pass

1. Reduce `vendor/ggml/` to GGUF reader only
2. Remove any test scripts / scratchpads under `archive/`
3. Final README + ROADMAP rewrite around Theory-First

## Validation

After each stage:
- `cmake --build build --target sp-engine` succeeds
- `./build/bin/sp-engine version` runs
- `./build/bin/sp-engine --help` shows the clean help text
- `./build/bin/test_sp_frobenius` still passes 17/17

After Stage 4:
- Phi-3 perplexity smoke test runs end-to-end with `--frobenius-quant`
- A100 bench script (`test-suite/bench/runpod_phi3_5config.sh`) runs to completion

## File-by-file disposition (final, post-Stage 4)

| Path | Stage-4 disposition |
|--|--|
| `src/engine.{cpp,h}` | KEEP (slimmed) |
| `src/cli/main.cpp` | KEEP (Theory-First only) |
| `src/sp_tensor.{cpp,h}` | KEEP |
| `src/sp_quant.{cpp,h}` | EVOLVE → drops legacy K-quant paths |
| `src/sp_quant_frobenius.{cpp,h}` | KEEP (Theory-First dispatch) |
| `src/sp_kernels_cpu.{cpp,h}` | EVOLVE → OK-coord ops |
| `src/forward.{cpp,h}` | REMOVE (Stage 2/3) |
| `src/forward_native.{cpp,h}` | REMOVE (folded into sp_forward) |
| `src/forward_native_context.{cpp,h}` | RENAME → `sp_context.cpp` |
| `src/llama_weights.{cpp,h}` | REMOVE (Stage 3) |
| `src/gguf_loader.{cpp,h}` | REFACTOR → `sp_gguf` (no ggml dep on parse) |
| `src/kv_cache.{cpp,h}` | EVOLVE → sp_ok_t elements |
| `src/http_server.{cpp,h}` | EVOLVE → `sp_server` (Phase 2 design doc) |
| `src/prime_pe.{cpp,h}` | EVOLVE → Stern–Brocot RoPE option |
| `src/sp_forward.{cpp,h}` | NEW (Phase 1) |
| `src/sp_attention.{cpp,h}` | NEW (Phase 1) |
| `src/sp_ffn.{cpp,h}` | NEW (Phase 1) |
| `src/sp_sampler.{cpp,h}` | NEW (Phase 2) |
| `src/sp_chat_template.{cpp,h}` | NEW (Phase 2) |
| `src/sp_batch.{cpp,h}` | NEW (Phase 2) |
| `src/sp_server.{cpp,h}` | NEW (Phase 2) |
| `lib/shannon-prime/core/sp_ok_arith.{c,h}` | KEEP (Phase 0) |
| `lib/shannon-prime/core/sp_frobenius.{c,h}` | KEEP (Phase 0) |
| `lib/shannon-prime/backends/cuda/sp_frobenius_quant.{cu,h}` | KEEP (Phase 0) |
| `vendor/ggml/` | SLIM to GGUF reader (Stage 4) |
| `archive/` | REMOVE (Stage 4) |

## Open questions for review

1. **`encode` / `decode` / `embed` verbs:** keep as thin utility verbs, or fold into a `sp-engine tools <subcmd>` namespace? Stage 4 decides.
2. **`speculative_oracle`:** Paper A §9.4 promotes this to a Theory-First L-function oracle. Stage 4 decides whether the existing scaffold is the right starting point or whether `sp_oracle.cpp` is a clean rewrite.
3. **Hexagon backend:** stays in `lib/shannon-prime/backends/hexagon/` per Phase 4 plan. Phone-side path is separate enough not to interfere with the desktop cleanup.
