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

# ─── SP KSTE encoder Tier-1 tests (Friedman Stack Phase 1) ────────────
# Self-contained: compiles the math files directly. Tests T1.1–T1.5
# from papers/PPT-ARM/TEST-SUITE.md. Emits per-test JSON reports into
# tests/results/T1_<id>.json (working dir = tests/ source tree).
add_executable(test_sp_kste
    test_sp_kste.cpp
    ../lib/shannon-prime/core/shannon_prime.c
    ../lib/shannon-prime/core/shannon_prime_sqfree.c
    ../lib/shannon-prime/core/sp_kste.c
    ../lib/shannon-prime/core/sp_kste_pack.c
    ../lib/shannon-prime/core/sp_kste_embed.c
)
target_include_directories(test_sp_kste PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_kste PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_kste COMMAND test_sp_kste
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ─── SP Friedman sieve cache Tier-2 tests (Phase 3) ───────────────────
# Self-contained: compiles the math files + the cache cpp directly.
# Tests T2.1 / T2.5 / T2.6 / T2.7 / T2.8 / T2.12.
if(SP_FRIEDMAN_SIEVE)
    add_executable(test_sp_friedman_cache
        test_sp_friedman_cache.cpp
        ../src/sp_friedman_cache.cpp
        ../lib/shannon-prime/core/shannon_prime.c
        ../lib/shannon-prime/core/shannon_prime_sqfree.c
        ../lib/shannon-prime/core/sp_kste.c
        ../lib/shannon-prime/core/sp_kste_pack.c
        ../lib/shannon-prime/core/sp_kste_embed.c
    )
    target_include_directories(test_sp_friedman_cache PRIVATE
        ../src
        ../lib/shannon-prime/core
    )
    set_target_properties(test_sp_friedman_cache PROPERTIES
        C_STANDARD 11
        CXX_STANDARD 17
    )
    add_test(NAME sp_friedman_cache COMMAND test_sp_friedman_cache
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # ─── KSTE Phase-4 semantic resolution probe ───────────────────────
    add_executable(test_sp_kste_resolution
        test_sp_kste_resolution.cpp
        ../lib/shannon-prime/core/shannon_prime.c
        ../lib/shannon-prime/core/shannon_prime_sqfree.c
        ../lib/shannon-prime/core/sp_kste.c
        ../lib/shannon-prime/core/sp_kste_pack.c
        ../lib/shannon-prime/core/sp_kste_embed.c
    )
    target_include_directories(test_sp_kste_resolution PRIVATE
        ../lib/shannon-prime/core
    )
    set_target_properties(test_sp_kste_resolution PROPERTIES
        C_STANDARD 11
        CXX_STANDARD 17
    )
    add_test(NAME sp_kste_resolution COMMAND test_sp_kste_resolution
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # ─── Phase 7 — ultraproduct attention prototype tests ─────────────
    # T3.1 / T3.2 / T3.6 — principal -> Top-1, Łoś property, choice
    # operator canonicality.  Links sp_engine for the ultraproduct kernel
    # (which depends on sp_ok_tensor + sp_ok_t_ops); shannon_prime_core
    # comes in transitively via sp_engine for sp_kste_*.
    add_executable(test_sp_ultraproduct_attn test_sp_ultraproduct_attn.cpp)
    target_link_libraries(test_sp_ultraproduct_attn PRIVATE sp_engine)
    target_include_directories(test_sp_ultraproduct_attn PRIVATE
        ../src
        ../lib/shannon-prime/core
    )
    set_target_properties(test_sp_ultraproduct_attn PROPERTIES
        CXX_STANDARD 17
    )
    add_test(NAME sp_ultraproduct_attn COMMAND test_sp_ultraproduct_attn
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

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

# Phase HVX-1: HVX-Barrett-Proth NTT parity test. Validates that the
# HVX kernel produces bit-identical output to the AVX-512 reference for
# forward / inverse / pointwise_mul / dot product. On x86 / CI the HVX
# kernel runs through its scalar fallback — same algorithm, no Q6_*
# intrinsics — so this test fires on every host. The on-device run
# (cross-compiled with hexagon-clang, SP_HEXAGON_ENABLED=1) re-runs
# the same test against real intrinsics.
add_executable(test_sp_ntt_crt_hvx
    test_sp_ntt_crt_hvx.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
    ../lib/shannon-prime/core/sp_ntt.c
    ../lib/shannon-prime/core/sp_ntt_crt.c
    ../lib/shannon-prime/core/sp_ntt_crt_hvx.c
)
target_include_directories(test_sp_ntt_crt_hvx PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ntt_crt_hvx PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ntt_crt_hvx COMMAND test_sp_ntt_crt_hvx)

# Strike 2: Band-as-Block scale fusion parity test. Validates that
# VHT2 + Möbius + per-tile int8 + Frobenius (B_a, B_b) round-trips
# within int8 quantization error, and that the spectral-domain dot
# product matches the fp32 reference dot to within the same noise.
add_executable(test_sp_vht2_block_q8
    test_sp_vht2_block_q8.cpp
    ../lib/shannon-prime/core/shannon_prime.c
    ../lib/shannon-prime/core/shannon_prime_sqfree.c
    ../lib/shannon-prime/core/shannon_prime_cauchy.c
    ../lib/shannon-prime/core/shannon_prime_modelpack.c
    ../lib/shannon-prime/core/sp_ok_arith.c
    ../lib/shannon-prime/core/sp_frobenius.c
    ../lib/shannon-prime/core/sp_vht2_block_q8.c
)
target_include_directories(test_sp_vht2_block_q8 PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_vht2_block_q8 PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_vht2_block_q8 COMMAND test_sp_vht2_block_q8)

# Strike 4: NEON Oracle + UFS prefetch coordinator. Tests the A510-side
# prefetch worker — predictions enqueue, worker fills slots in the
# background, get() hits the cache. Latency-hiding is validated with a
# synthetic slow backend (30ms simulated UFS read). On device the
# backend swaps to a rpcmem-backed pread; the coordinator logic is
# unchanged.
add_executable(test_sp_oracle_prefetch
    test_sp_oracle_prefetch.cpp
    ../src/sp_oracle_prefetch.cpp
)
target_include_directories(test_sp_oracle_prefetch PRIVATE
    ../src
)
set_target_properties(test_sp_oracle_prefetch PROPERTIES
    CXX_STANDARD 17
)
# C++17 + threads — link pthreads on Linux, nothing extra on MSVC.
if(NOT WIN32)
    target_link_libraries(test_sp_oracle_prefetch PRIVATE pthread)
endif()
add_test(NAME sp_oracle_prefetch COMMAND test_sp_oracle_prefetch)

# Strike 5: HVX cyclotomic matmul kernel parity test. Validates that
# the two-phase (per-block D_a/D_b then fold) formulation produces
# bit-identical int64 (acc_a, acc_b) to the engine's direct-form
# reference (sp_matmul_ok_block_inner_scalar). On host this runs the
# scalar fallback in sp_hex_matmul_block_q8.c — same algorithm the
# DSP-side build will compile under __HVX__ with vector intrinsics.
# Bit-equality at the algebra level is the lock; on-device validation
# is the FastRPC test once the qaic IDL stubs are regenerated.
add_executable(test_sp_hex_matmul_block_q8
    test_sp_hex_matmul_block_q8.cpp
    ../lib/shannon-prime/backends/hexagon/sp_hex_matmul_block_q8.c
)
target_include_directories(test_sp_hex_matmul_block_q8 PRIVATE
    ../lib/shannon-prime/core
    ../lib/shannon-prime/backends/hexagon
)
set_target_properties(test_sp_hex_matmul_block_q8 PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_hex_matmul_block_q8 COMMAND test_sp_hex_matmul_block_q8)

# Strike 4 device telemetry runner. Cross-compile for arm64-android to
# measure UFS 3.1 read latency on the S22 Ultra with the dummy Oracle.
# Affinity-pins the worker to A510 silver and self to X2 prime.
add_executable(sp_oracle_telemetry
    sp_oracle_telemetry.cpp
    ../src/sp_oracle_prefetch.cpp
)
target_include_directories(sp_oracle_telemetry PRIVATE ../src)
set_target_properties(sp_oracle_telemetry PROPERTIES CXX_STANDARD 17)
# Android NDK bundles pthread in libc; only standalone Linux needs -lpthread.
if(NOT WIN32 AND NOT ANDROID)
    target_link_libraries(sp_oracle_telemetry PRIVATE pthread)
endif()

# Phase 13.A: Algebraic Resonance Memory — HRR-style associative memory
# native to the dual-prime CRT cyclotomic ring. Storage = pointwise
# accumulation of K*V products in NTT domain; recall = inverse NTT of
# M * Q_REV. Test characterizes recall SNR as a function of memories
# per slab (capacity sweep K=1..64).
add_executable(test_sp_arm
    test_sp_arm.cpp
    ../lib/shannon-prime/core/sp_poly_ring.c
    ../lib/shannon-prime/core/sp_ntt.c
    ../lib/shannon-prime/core/sp_ntt_crt.c
    ../lib/shannon-prime/core/sp_arm.c
)
target_include_directories(test_sp_arm PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_arm PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_arm COMMAND test_sp_arm)

# Phase 12 Step A: packed-int8 O_K storage. Standalone — depends only on
# sp_ok_arith, sp_frobenius, and the new sp_ok_q8 module. Verifies that
# Frobenius-transformed coordinates round-trip through 2-byte packing with
# bounded relative error (Theorem 2 / CM Sato-Tate variance budget).
add_executable(test_sp_ok_q8
    test_sp_ok_q8.cpp
    ../lib/shannon-prime/core/sp_ok_q8.c
    ../lib/shannon-prime/core/sp_frobenius.c
    ../lib/shannon-prime/core/sp_ok_arith.c
)
target_include_directories(test_sp_ok_q8 PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ok_q8 PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ok_q8 COMMAND test_sp_ok_q8)

# Phase 14: 4-bit packed O_K storage. Mirrors test_sp_ok_q8 with the
# codebook halved. Verifies shift picker boundary cases, encode/decode
# round-trip per-coord bound, nybble pack layout, lattice-norm pruning,
# 16x compression ratio, and zero-shift identity.
add_executable(test_sp_ok_q4
    test_sp_ok_q4.cpp
    ../lib/shannon-prime/core/sp_ok_q4.c
    ../lib/shannon-prime/core/sp_ok_arith.c
)
target_include_directories(test_sp_ok_q4 PRIVATE
    ../lib/shannon-prime/core
)
set_target_properties(test_sp_ok_q4 PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
add_test(NAME sp_ok_q4 COMMAND test_sp_ok_q4)

# Phase 12 Step B-1: combined load-path encoder. Verifies the
# sp_ok_encode_q8_from_fp16_with_frobenius() helper matches the manual
# (encode -> Frobenius -> pack) pipeline byte-for-byte, that the arena
# extension correctly sizes packed storage, and that decode recovers the
# post-Frobenius coordinates within the rounding budget.
add_executable(test_sp_ok_q8_load_path
    test_sp_ok_q8_load_path.cpp
)
target_link_libraries(test_sp_ok_q8_load_path PRIVATE sp_engine)
add_test(NAME sp_ok_q8_load_path COMMAND test_sp_ok_q8_load_path)

# Phase 14 Q4 load-path: mirrors the Q8 test with 4-bit packing.
add_executable(test_sp_ok_q4_load_path
    test_sp_ok_q4_load_path.cpp
)
target_link_libraries(test_sp_ok_q4_load_path PRIVATE sp_engine)
add_test(NAME sp_ok_q4_load_path COMMAND test_sp_ok_q4_load_path)

# Phase 14 Q4 fused matmul parity (Q4 vs Q8 vs raw).
add_executable(test_sp_matmul_q4
    test_sp_matmul_q4.cpp
)
target_link_libraries(test_sp_matmul_q4 PRIVATE sp_engine)
add_test(NAME sp_matmul_q4 COMMAND test_sp_matmul_q4)
