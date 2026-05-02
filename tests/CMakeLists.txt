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
