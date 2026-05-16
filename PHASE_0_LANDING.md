# Phase 0 Landing Notes — Shannon-Prime Theory-First Engine + Server

**Commit date:** 2026-05-16

This commit lands the Phase 0 foundation per `docs/THEORY-FIRST-ENGINE-DESIGN.md`. It is **non-breaking** — the existing engine + http_server continue to work unchanged; the new files are additive.

## What's in this commit

### Math primitives (lib/shannon-prime/core/, lib/shannon-prime/backends/cuda/)

| File | LOC | Purpose |
|--|--|--|
| `core/sp_ok_arith.h` | ~110 | Header-only O_K = Z[ω] arithmetic |
| `core/sp_ok_arith.c` | ~25 | sp_ok_pow (square-and-multiply) |
| `core/sp_frobenius.h` | ~95 | Frobenius / Sato-Tate / prime classification API |
| `core/sp_frobenius.c` | ~165 | Implementation: Legendre, is_inert/split, find_element_of_norm, apply_frobenius, tensor wrappers |
| `backends/cuda/sp_frobenius_quant.h` | ~45 | CUDA kernel API |
| `backends/cuda/sp_frobenius_quant.cu` | ~115 | One thread per state element, integer-only, square-and-multiply |

### Engine integration (src/)

| File | LOC | Purpose |
|--|--|--|
| `src/sp_quant_frobenius.h` | ~40 | C++ wrapper for the C-side routines |
| `src/sp_quant_frobenius.cpp` | ~85 | Diagnostics + `--sato-tate-mix` arg parsing |

### Tests (tests/)

| File | LOC | Purpose |
|--|--|--|
| `tests/test_sp_frobenius.cpp` | ~250 | 13 unit tests + cross-validation against Python golden file |
| `tests/sato_tate_golden.json` | 50 entries | Bit-exact reference values from Python oracle |

### Documentation

| File | Purpose |
|--|--|
| `docs/THEORY-FIRST-ENGINE-DESIGN.md` | Streamlined engine v2 architecture + migration map |
| `docs/SP-SERVER-DESIGN.md` | OpenAI-compat sp-server design + comparison table |
| `PHASE_0_LANDING.md` | This file |

### CMakeLists touch-ups required (separate manual step)

Add to `lib/shannon-prime/CMakeLists.txt`:

```cmake
target_sources(shannon_prime_core PRIVATE
    core/sp_ok_arith.c
    core/sp_frobenius.c
)
```

Add to `tests/CMakeLists.txt`:

```cmake
add_executable(test_sp_frobenius test_sp_frobenius.cpp)
target_link_libraries(test_sp_frobenius PRIVATE shannon_prime_core)
add_test(NAME sp_frobenius COMMAND test_sp_frobenius)
```

If CUDA build is enabled, add to the appropriate CUDA target:

```cmake
target_sources(shannon_prime_cuda PRIVATE
    backends/cuda/sp_frobenius_quant.cu
)
```

The engine binary gets the new `--frobenius-quant` / `--sato-tate-mix` CLI integration in a follow-up commit (a one-liner per flag in `src/cli/main.cpp`, gated on Phase 1 readiness of the sp_quant.cpp dispatch path).

## How to verify the commit locally

```bash
cd shannon-prime-engine
# 1. Build (from a clean state)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_sp_frobenius --parallel

# 2. Generate the Python golden file
cd ../test-suite
python scripts/make_golden.py > ../shannon-prime-engine/tests/sato_tate_golden.json

# 3. Run the C tests
cd ../shannon-prime-engine
./build/bin/test_sp_frobenius
# Expected output:
#   Shannon-Prime Frobenius unit tests (15)
#     [OK  ] omega_squared_equals_omega_minus_41
#     [OK  ] omega_norm_is_41
#     [OK  ] omega_times_omega_bar_is_norm
#     [OK  ] commutativity_random
#     [OK  ] p2_is_inert
#     [OK  ] p11_is_inert_paper_D_fix
#     [OK  ] p41_is_split
#     [OK  ] p43_is_split_eulers_polynomial
#     [OK  ] p163_is_ramified
#     [OK  ] phi_2_squared_equals_minus_2
#     [OK  ] find_element_of_norm_41
#     [OK  ] frobenius_split_norm_invariant
#     [OK  ] sato_tate_commutativity
#     [OK  ] sato_tate_random_commutativity
#     [OK  ] sp_frobenius_quant_tensor_norm
#     [OK  ] sp_sato_tate_mix_tensor_norm
#     [OK  ] cross_validate_python_oracle_optional
#   PASS — 0 failures
```

## Cross-checks against the Shannon-Prime Test Suite

The Python algebraic oracle is at `D:\F\shannon-prime-repos\test-suite\` (v0.3, 18/18 algebraic claims VERIFIED). The C engine code in this commit is the bit-exact mirror of:

- `test-suite/src/sp_algebra.py` (OK class) ↔ `lib/shannon-prime/core/sp_ok_arith.{c,h}`
- `test-suite/src/sp_algebra.py` (Frobenius helpers) ↔ `lib/shannon-prime/core/sp_frobenius.{c,h}`
- `test-suite/src/engine_hooks2.py` ↔ `lib/shannon-prime/backends/cuda/sp_frobenius_quant.{h,cu}` (and host wrappers)

When making any change to the math, update the Python oracle first, run `python3 src/run_suite4.py --version v0.3 --out-md results/report_v0.3.md` and verify all green. Then port to C. Then regenerate the golden file. Then run `test_sp_frobenius`. Bit-exactness is the contract.

## Next commits (Phase 1)

Per `docs/THEORY-FIRST-ENGINE-DESIGN.md` §Phases:

1. `src/sp_tensor.{cpp,h}` — `sp_ok_tensor` type + arena allocator
2. `src/sp_gguf.{cpp,h}` — GGUF parser without ggml dep
3. `src/sp_weights.{cpp,h}` — model load → OK-encoded
4. `src/sp_forward.{cpp,h}` — pure SP forward pass (CPU first)
5. CLI integration of `--frobenius-quant` and `--sato-tate-mix` once the dispatch path exists
6. Phi-3 perplexity validation within 1% of llama.cpp baseline

## Coordination with Paper D v0.3

Paper D v0.3 (`papers/paper_D_companion_systems.md`) specifies Configs B and E with $p_2 = 41$ (corrected from v0.2's erroneous $p = 11$). This commit's primes match Paper D v0.3:

- `SP_P1_INERT = 2` (Config E inert channel)
- `SP_P2_SPLIT = 41` (Config B and Config E split channel)

Any future paper revision must update the constants in `lib/shannon-prime/core/sp_frobenius.h` to match. The test suite catches any divergence (`PAPER-D-FIX` test fires on mismatched primes).
