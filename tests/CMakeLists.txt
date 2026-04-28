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
