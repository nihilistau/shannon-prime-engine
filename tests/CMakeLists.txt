add_executable(test_engine_smoke test_engine_smoke.cpp)
target_link_libraries(test_engine_smoke PRIVATE sp_engine)
add_test(NAME engine_smoke COMMAND test_engine_smoke)

# GGUF loader: runs only if SP_ENGINE_TEST_MODEL points at a GGUF on disk.
# CI without model assets skips this — compile still validates the API.
add_executable(test_gguf_loader test_gguf_loader.cpp)
target_link_libraries(test_gguf_loader PRIVATE sp_engine)
add_test(NAME gguf_loader COMMAND test_gguf_loader)
