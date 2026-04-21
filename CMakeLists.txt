# Shannon-Prime Engine — build system
#
# Top-level targets:
#   sp-engine        (binary)     — CLI + inference loop
#   shannon_prime_core (static)   — from lib/shannon-prime
#   ggml             (static/shared) — from vendor/ggml

cmake_minimum_required(VERSION 3.14)
project(shannon-prime-engine
    VERSION 0.1.0
    DESCRIPTION "Shannon-Prime reference inference engine with native compressed KV"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Silence MSVC's "unsafe CRT" deprecation warnings for getenv/fopen/etc.
# We use the portable variants intentionally; the *_s wrappers are
# non-standard.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Default to Release; user can override with -DCMAKE_BUILD_TYPE=...
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Group output under bin/ and lib/ for all platforms.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --------------------------------------------------------------------
# Submodule sanity check — fail fast with a helpful message if the
# repo was cloned without `--recursive`.
# --------------------------------------------------------------------
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/lib/shannon-prime/core/shannon_prime.h)
    message(FATAL_ERROR
        "Missing submodule lib/shannon-prime.\n"
        "Run: git submodule update --init --recursive")
endif()
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/vendor/ggml/include/ggml.h)
    message(FATAL_ERROR
        "Missing submodule vendor/ggml.\n"
        "Run: git submodule update --init --recursive")
endif()

# --------------------------------------------------------------------
# Options
# --------------------------------------------------------------------
option(SP_ENGINE_BUILD_TESTS     "Build the engine's own unit tests"    ON)
option(SP_ENGINE_BUILD_CLI       "Build the sp-engine CLI binary"       ON)
option(SP_ENGINE_WITH_CUDA       "Enable CUDA backend for KV kernels"   OFF)
option(SP_ENGINE_WITH_VULKAN     "Enable Vulkan backend for KV kernels" OFF)

# --------------------------------------------------------------------
# Core shannon-prime math library (submodule)
# Built as a tiny static lib; the submodule ships its own core .c files.
# --------------------------------------------------------------------
set(SP_CORE_DIR ${CMAKE_SOURCE_DIR}/lib/shannon-prime)
set(SP_CORE_SRC
    ${SP_CORE_DIR}/core/shannon_prime.c
    ${SP_CORE_DIR}/core/shannon_prime_sqfree.c
    ${SP_CORE_DIR}/core/shannon_prime_cauchy.c
    ${SP_CORE_DIR}/core/shannon_prime_modelpack.c)
add_library(shannon_prime_core STATIC ${SP_CORE_SRC})
target_include_directories(shannon_prime_core PUBLIC ${SP_CORE_DIR}/core)
target_compile_features(shannon_prime_core PUBLIC c_std_11)
if(MSVC)
    target_compile_options(shannon_prime_core PRIVATE /W3)
else()
    target_compile_options(shannon_prime_core PRIVATE -Wall -Wextra -Wno-unused-function)
endif()

# --------------------------------------------------------------------
# CUDA backend: GPU-resident KV cache (step-3 goal)
# Built only when SP_ENGINE_WITH_CUDA is ON. Provides sp_cuda_cache_*
# and the VHT2 / Möbius / band-quant kernels that make SHIP-path
# compress/decompress run on GPU without a host round-trip.
# --------------------------------------------------------------------
if(SP_ENGINE_WITH_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    add_library(shannon_prime_cuda_backend STATIC
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_cuda.cu
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_sqfree.cu)
    target_include_directories(shannon_prime_cuda_backend PUBLIC
        ${SP_CORE_DIR}/backends/cuda
        ${SP_CORE_DIR}/core)
    target_link_libraries(shannon_prime_cuda_backend PUBLIC shannon_prime_core)
    set_target_properties(shannon_prime_cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
endif()

# --------------------------------------------------------------------
# ggml (submodule) — MIT-licensed compute graph + GGUF loader + matmul
# Disable the bits we don't need so the engine binary stays lean.
# --------------------------------------------------------------------
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
if(SP_ENGINE_WITH_CUDA)
    set(GGML_CUDA TRUE CACHE BOOL "" FORCE)
endif()
if(SP_ENGINE_WITH_VULKAN)
    set(GGML_VULKAN TRUE CACHE BOOL "" FORCE)
endif()
add_subdirectory(vendor/ggml EXCLUDE_FROM_ALL)

# --------------------------------------------------------------------
# Engine library (AGPLv3, this repo's original contribution)
# --------------------------------------------------------------------
add_library(sp_engine STATIC
    src/engine.cpp
    src/kv_cache.cpp
    src/gdn_state.cpp
    src/gguf_loader.cpp
    src/vocab.cpp
    src/tokenizer.cpp
    src/llama_weights.cpp
    src/forward.cpp
    src/prime_pe.cpp)
target_include_directories(sp_engine PUBLIC src)
target_link_libraries(sp_engine PUBLIC
    shannon_prime_core
    ggml)
if(SP_ENGINE_WITH_CUDA)
    target_link_libraries(sp_engine PUBLIC shannon_prime_cuda_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_CUDA=1)
    # Host-side C++ sources in sp_engine include cuda_runtime.h (for
    # cudaMalloc, cudaMemcpyAsync, etc.) so they need the CUDA Toolkit
    # headers. Nvcc-compiled .cu files pick these up automatically; the
    # .cpp files do not.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sp_engine PUBLIC CUDA::cudart)
endif()
target_compile_features(sp_engine PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sp_engine PRIVATE /W3)
else()
    target_compile_options(sp_engine PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------
# CLI binary
# --------------------------------------------------------------------
if(SP_ENGINE_BUILD_CLI)
    add_executable(sp-engine src/cli/main.cpp)
    target_link_libraries(sp-engine PRIVATE sp_engine)
endif()

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
if(SP_ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------
# Print a short config summary
# --------------------------------------------------------------------
message(STATUS "")
message(STATUS "Shannon-Prime Engine ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA       : ${SP_ENGINE_WITH_CUDA}")
message(STATUS "  Vulkan     : ${SP_ENGINE_WITH_VULKAN}")
message(STATUS "  CLI        : ${SP_ENGINE_BUILD_CLI}")
message(STATUS "  tests      : ${SP_ENGINE_BUILD_TESTS}")
message(STATUS "")
