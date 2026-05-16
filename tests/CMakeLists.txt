add_executable(test_engine_smoke test_engine_smoke.cpp)
target_link_libraries(test_engine_smoke PRIVATE sp_engine)
add_test(NAME engine_smoke COMMAND test_engine_smoke)

# GGUF loader: runs only if SP_ENGINE_TEST_MODEL points at a GGUF on disk.
# CI without model assets skips this — compile still validates the API.
add_executable(test_gguf_loader test_gguf_loader.cpp)
target_link_libraries(test_gguf_loader PRIVATE sp_engine)
add_test(NAME gguf_loader COMMAND test_gguf_loader)

# Disk-tier round-trip: write -> read -> save_to_disk -> destroy ->
# create -> load_from_disk -> read, asserting bit-exact second read.
# Also covers model-hash guard (wrong hash must be rejected). CPU-only;
# disk I/O is host-side regardless of backend.
add_executable(test_disk_cache_roundtrip test_disk_cache_roundtrip.cpp)
target_link_libraries(test_disk_cache_roundtrip PRIVATE sp_engine)
add_test(NAME disk_cache_roundtrip COMMAND test_disk_cache_roundtrip)

# Disk-partial-load IO bench. NOT registered with ctest (it's a bench,
# not a pass/fail test); built so it's available to run manually.
# Usage: ./build/bin/Release/bench_disk_partial [--runs 5] [--pos 2048]
# Output: CSV to stdout with bytes_read, wall_ms, bandwidth per config.
# See tests/bench_disk_partial.cpp for the cold-cache instructions
# (RAMMap on Windows, drop_caches on Linux).
add_executable(bench_disk_partial bench_disk_partial.cpp)
target_link_libraries(bench_disk_partial PRIVATE sp_engine)

# Q5_K dequant parity vs ggml. Self-contained — synthesizes a random
# fp32 buffer, quantizes via ggml's encoder, then dequants with BOTH
# sp_dequant_q5_K and ggml's dequantize_row_q5_K and asserts the outputs
# are bit-exact. Foundation for forward_native.cpp's confidence that
# weight reads match ggml semantics.
add_executable(test_sp_quant_q5k test_sp_quant_q5k.cpp)
target_link_libraries(test_sp_quant_q5k PRIVATE sp_engine ggml-base ggml-cpu)
add_test(NAME sp_quant_q5k COMMAND test_sp_quant_q5k)

# CPU kernels parity vs ggml (RMSNorm, SiLU, Softmax, fp32 matmul,
# Q5_K-packed matmul). Tolerances: relative 1e-4, absolute 1e-5 — SIMD
# reduction order varies between us and ggml so true bit-exact match
# isn't expected; we want "model output equivalence" which this catches.
add_executable(test_sp_kernels test_sp_kernels.cpp)
target_link_libraries(test_sp_kernels PRIVATE sp_engine ggml ggml-base ggml-cpu)
add_test(NAME sp_kernels COMMAND test_sp_kernels)

# ─── SP Frobenius / Sato-Tate unit tests (Phase 0 Theory-First) ───────
# Self-contained: compiles the math submodule's C files directly so this
# test target doesn't pull in the full sp_engine library. Validates
# bit-exact contract against test-suite/src/sp_algebra.py via
# tests/sato_tate_golden.json.
add_executable(test_sp_frobenius
    test_sp_frobenius.cpp
    ../lib/shannon-prime/core/sp_ok_arith.c
    ../lib/shannon-prime/core/sp_frobenius.c
)
target_include_directories(test_sp_frobenius PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_frobenius PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_frobenius COMMAND test_sp_frobenius
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ─── SP encode + sampler unit tests (Phase 1.2/1.3 + Phase 2 prep) ─────
add_executable(test_sp_encode test_sp_encode.cpp)
target_link_libraries(test_sp_encode PRIVATE sp_engine)
add_test(NAME sp_encode COMMAND test_sp_encode)

# ─── SP load-time Frobenius shim (Phase 1.7) ──────────────────────────
add_executable(test_sp_load_shim test_sp_load_shim.cpp)
target_link_libraries(test_sp_load_shim PRIVATE sp_engine)
add_test(NAME sp_load_shim COMMAND test_sp_load_shim)

# ─── SP native O_K matmul (Phase 2.0) ──────────────────────────────────
add_executable(test_sp_matmul test_sp_matmul.cpp)
target_link_libraries(test_sp_matmul PRIVATE sp_engine)
add_test(NAME sp_matmul COMMAND test_sp_matmul)

# ─── SP fp32 bridges (Phase 2.1) ──────────────────────────────────────
add_executable(test_sp_bridges test_sp_bridges.cpp)
target_link_libraries(test_sp_bridges PRIVATE sp_engine)
add_test(NAME sp_bridges COMMAND test_sp_bridges)

# ─── SP attention + FFN core kernels (Phase 2.2a) ─────────────────────
add_executable(test_sp_attention_ffn test_sp_attention_ffn.cpp)
target_link_libraries(test_sp_attention_ffn PRIVATE sp_engine)
add_test(NAME sp_attention_ffn COMMAND test_sp_attention_ffn)

# ─── SP Phase 2.2b: RoPE, KV cache, multi-token prefill, weights API ──
add_executable(test_sp_rope_kvcache test_sp_rope_kvcache.cpp)
target_link_libraries(test_sp_rope_kvcache PRIVATE sp_engine)
add_test(NAME sp_rope_kvcache COMMAND test_sp_rope_kvcache)

add_executable(test_sp_prefill test_sp_prefill.cpp)
target_link_libraries(test_sp_prefill PRIVATE sp_engine)
add_test(NAME sp_prefill COMMAND test_sp_prefill)

add_executable(test_sp_weights test_sp_weights.cpp)
target_link_libraries(test_sp_weights PRIVATE sp_engine)
add_test(NAME sp_weights COMMAND test_sp_weights)

# ─── SP Phase 2.2c: sp_weights_loader (fp16-source + LlamaWeights walker) ─
add_executable(test_sp_weights_loader test_sp_weights_loader.cpp)
target_link_libraries(test_sp_weights_loader PRIVATE sp_engine)
add_test(NAME sp_weights_loader COMMAND test_sp_weights_loader)

# ─── SP Phase 2.2d: sp_forward_step — native O_K end-to-end forward pass ─
add_executable(test_sp_forward_step test_sp_forward_step.cpp)
target_link_libraries(test_sp_forward_step PRIVATE sp_engine)
add_test(NAME sp_forward_step COMMAND test_sp_forward_step)

# ─── SP Phase 3 part 1: elliptic-curve + Weil pairing math core ────────
# Standalone — compiles the C source directly so the test can run
# even if the rest of sp_engine has a regression.
add_executable(test_sp_ec_weil
    test_sp_ec_weil.cpp
    ../lib/shannon-prime/core/sp_ec_weil.c
)
target_include_directories(test_sp_ec_weil PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ec_weil PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ec_weil COMMAND test_sp_ec_weil)

# ─── SP Phase 3 part 2+3: Weil-pairing attention prototype ─────────────
# Maps Q/K vectors to E[n] torsion via a fixed hash, computes the Weil
# pairing as attention score, compares against softmax(QK^T/sqrt(d))
# on the same random inputs.
add_executable(test_sp_weil_attention
    test_sp_weil_attention.cpp
    ../lib/shannon-prime/core/sp_ec_weil.c
)
target_include_directories(test_sp_weil_attention PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_weil_attention PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_weil_attention COMMAND test_sp_weil_attention)

# ─── SP Phase 3 pivot: polynomial ring kernel ──────────────────────────
# Naive Z[x]/(x^N+1) multiplier + CKKS-style fp32 round-trip.
# Validates the dot-product identity before plumbing into attention.
add_executable(test_sp_poly_ring
    test_sp_poly_ring.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
)
target_include_directories(test_sp_poly_ring PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_poly_ring PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_poly_ring COMMAND test_sp_poly_ring)

# ─── SP Phase 3 pivot: CKKS-polynomial attention vs softmax bench ──────
add_executable(test_sp_poly_attention
    test_sp_poly_attention.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
)
target_include_directories(test_sp_poly_attention PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_poly_attention PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_poly_attention COMMAND test_sp_poly_attention)

# ─── SP Phase 4: NTT parity test (sp_poly_mul_ntt vs O(N^2) baseline) ──
add_executable(test_sp_ntt
    test_sp_ntt.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
    ../lib/shannon-prime/core/sp_ntt.c
)
target_include_directories(test_sp_ntt PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ntt PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ntt COMMAND test_sp_ntt)

# ─── SP Phase 9: dual-prime CRT NTT parity test ─────────────────────────
# Builds the CRT NTT kernel + the existing 60-bit NTT side-by-side and
# verifies they produce bit-identical results on random poly pairs.
add_executable(test_sp_ntt_crt
    test_sp_ntt_crt.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
    ../lib/shannon-prime/core/sp_ntt.c
    ../lib/shannon-prime/core/sp_ntt_crt.c
)
target_include_directories(test_sp_ntt_crt PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ntt_crt PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ntt_crt COMMAND test_sp_ntt_crt)
