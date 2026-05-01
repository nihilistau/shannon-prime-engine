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
option(SP_ENGINE_WITH_HEXAGON    "Enable Hexagon DSP backend (cDSP / FastRPC)" ON)
option(SP_ENGINE_HEXAGON_FASTRPC "Link real Qualcomm FastRPC libs (Android+SDK only)" OFF)
option(SP_ENGINE_FP8             "Enable FP8 (E4M3) KV cache quantization kernels" OFF)
option(SP_ENGINE_FP4             "Enable FP4 (Blackwell sm_120+) KV cache quantization" OFF)
option(SP_ENGINE_AUTO_PATCH      "Auto-apply vendor patches before build" ON)

# --------------------------------------------------------------------
# Auto-apply vendor patches (K-quant getrows, etc.)
# Applies all *.patch files from patches/ to their respective vendor
# submodules. Safe to re-run: git apply --check skips already-applied.
# --------------------------------------------------------------------
if(SP_ENGINE_AUTO_PATCH AND EXISTS ${CMAKE_SOURCE_DIR}/patches)
    file(GLOB _SP_PATCHES "${CMAKE_SOURCE_DIR}/patches/*.patch")
    foreach(_patch ${_SP_PATCHES})
        get_filename_component(_patch_name ${_patch} NAME)
        # Check if patch is already applied (dry-run)
        execute_process(
            COMMAND git apply --check --reverse ${_patch}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/vendor/ggml
            RESULT_VARIABLE _patch_applied
            OUTPUT_QUIET ERROR_QUIET)
        if(_patch_applied EQUAL 0)
            message(STATUS "Patch already applied: ${_patch_name}")
        else()
            # Try to apply
            execute_process(
                COMMAND git apply --check ${_patch}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/vendor/ggml
                RESULT_VARIABLE _patch_check
                OUTPUT_QUIET ERROR_QUIET)
            if(_patch_check EQUAL 0)
                execute_process(
                    COMMAND git apply ${_patch}
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/vendor/ggml
                    RESULT_VARIABLE _patch_result)
                if(_patch_result EQUAL 0)
                    message(STATUS "Applied patch: ${_patch_name}")
                else()
                    message(WARNING "Failed to apply patch: ${_patch_name}")
                endif()
            else()
                message(STATUS "Patch cannot be applied (conflicts?): ${_patch_name}")
            endif()
        endif()
    endforeach()
endif()

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
        # Default: Turing (sm_75). For Ada fp8 tensor cores add 89.
        # For Blackwell fp4 tensor cores add 120.
        set(_SP_CUDA_ARCHS "75")
        if(SP_ENGINE_FP8)
            # sm_89 = Ada Lovelace (fp8 tensor cores)
            list(APPEND _SP_CUDA_ARCHS "89")
        endif()
        if(SP_ENGINE_FP4)
            # sm_120 = Blackwell (fp4 tensor cores, requires CUDA >= 12.8)
            list(APPEND _SP_CUDA_ARCHS "120")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_SP_CUDA_ARCHS})
    endif()

    set(_SP_CUDA_SOURCES
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_cuda.cu
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_sqfree.cu
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_hier.cu)

    # FP8 quantization kernels (E4M3 for values, E5M2 for scales)
    if(SP_ENGINE_FP8)
        if(EXISTS ${SP_CORE_DIR}/backends/cuda/shannon_prime_fp8.cu)
            list(APPEND _SP_CUDA_SOURCES
                ${SP_CORE_DIR}/backends/cuda/shannon_prime_fp8.cu)
        endif()
    endif()

    # FP4 quantization kernels (Blackwell MXFP4 micro-scaling)
    if(SP_ENGINE_FP4)
        if(EXISTS ${SP_CORE_DIR}/backends/cuda/shannon_prime_fp4.cu)
            list(APPEND _SP_CUDA_SOURCES
                ${SP_CORE_DIR}/backends/cuda/shannon_prime_fp4.cu)
        endif()
    endif()

    add_library(shannon_prime_cuda_backend STATIC ${_SP_CUDA_SOURCES})
    target_include_directories(shannon_prime_cuda_backend PUBLIC
        ${SP_CORE_DIR}/backends/cuda
        ${SP_CORE_DIR}/core)
    target_link_libraries(shannon_prime_cuda_backend PUBLIC shannon_prime_core)
    set_target_properties(shannon_prime_cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

    if(SP_ENGINE_FP8)
        target_compile_definitions(shannon_prime_cuda_backend PUBLIC SP_FP8=1)
    endif()
    if(SP_ENGINE_FP4)
        target_compile_definitions(shannon_prime_cuda_backend PUBLIC SP_FP4=1)
    endif()
endif()

# --------------------------------------------------------------------
# Hexagon DSP backend: SP-banded KV cache compress/decompress on cDSP
#
# Two-mode build:
#   - Stub mode  (SP_ENGINE_HEXAGON_FASTRPC=OFF):
#       Compiles shannon_prime_hexagon.c WITHOUT defining SP_HEXAGON_FASTRPC.
#       Every entry point returns -1/NULL with a one-shot diagnostic.
#       Forward dispatch falls through to ggml CPU naturally. No SDK needed.
#   - Real FastRPC (SP_ENGINE_HEXAGON_FASTRPC=ON):
#       Defines SP_HEXAGON_FASTRPC + adds qaic-generated stubs + links
#       libcdsprpc.so. Android arm64 only. Requires Qualcomm Hexagon SDK 5.x.
#       Wired separately when SDK is available — not in this stub-mode
#       configuration.
# --------------------------------------------------------------------
if(SP_ENGINE_WITH_HEXAGON)
    set(_SP_HEXAGON_SOURCES
        ${SP_CORE_DIR}/backends/hexagon/shannon_prime_hexagon.c)
    add_library(shannon_prime_hexagon_backend STATIC ${_SP_HEXAGON_SOURCES})
    target_include_directories(shannon_prime_hexagon_backend PUBLIC
        ${SP_CORE_DIR}/backends/hexagon
        ${SP_CORE_DIR}/core)
    target_link_libraries(shannon_prime_hexagon_backend PUBLIC shannon_prime_core)
    target_compile_features(shannon_prime_hexagon_backend PUBLIC c_std_11)
    set_property(TARGET shannon_prime_hexagon_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
    if(SP_ENGINE_HEXAGON_FASTRPC)
        # Real FastRPC path — requires Hexagon SDK on the include/link paths.
        # Caller sets HEXAGON_SDK_ROOT to point at /c/Qualcomm/Hexagon_SDK/5.x.x.x
        if(NOT HEXAGON_SDK_ROOT)
            message(FATAL_ERROR
                "SP_ENGINE_HEXAGON_FASTRPC=ON requires HEXAGON_SDK_ROOT to be set "
                "(e.g. -DHEXAGON_SDK_ROOT=C:/Qualcomm/Hexagon_SDK/5.5.6.0)")
        endif()
        target_compile_definitions(shannon_prime_hexagon_backend PUBLIC SP_HEXAGON_FASTRPC=1)
        target_include_directories(shannon_prime_hexagon_backend PUBLIC
            ${HEXAGON_SDK_ROOT}/incs
            ${HEXAGON_SDK_ROOT}/incs/stddef
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc)
        # libcdsprpc is on-device only; we dlopen at runtime in stub fallback,
        # but explicit link when SDK target requires it. Caller passes -ldl.
        message(STATUS "Shannon-Prime Hexagon: real FastRPC mode (SDK=${HEXAGON_SDK_ROOT})")
    else()
        message(STATUS "Shannon-Prime Hexagon: stub mode (no SDK; backend returns -1 fallback)")
    endif()
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
    src/prime_pe.cpp
    src/http_server.cpp
    vendor/cpp-httplib/httplib.cpp)
target_include_directories(sp_engine PUBLIC src)
target_include_directories(sp_engine PUBLIC ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
# httplib needs winsock on Windows; pthread on POSIX (already implicit).
if(WIN32)
    target_link_libraries(sp_engine PUBLIC ws2_32 crypt32)
endif()
target_link_libraries(sp_engine PUBLIC
    shannon_prime_core
    ggml)
if(SP_ENGINE_WITH_HEXAGON)
    target_link_libraries(sp_engine PUBLIC shannon_prime_hexagon_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_HEXAGON=1)
    if(SP_ENGINE_HEXAGON_FASTRPC)
        target_compile_definitions(sp_engine PUBLIC SP_HEXAGON_FASTRPC=1)
    endif()
endif()
if(SP_ENGINE_WITH_CUDA)
    target_link_libraries(sp_engine PUBLIC shannon_prime_cuda_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_CUDA=1)
    if(SP_ENGINE_FP8)
        target_compile_definitions(sp_engine PUBLIC SP_FP8=1)
    endif()
    if(SP_ENGINE_FP4)
        target_compile_definitions(sp_engine PUBLIC SP_FP4=1)
    endif()
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
message(STATUS "  Hexagon    : ${SP_ENGINE_WITH_HEXAGON} (FastRPC=${SP_ENGINE_HEXAGON_FASTRPC})")
message(STATUS "  FP8 (E4M3) : ${SP_ENGINE_FP8}")
message(STATUS "  FP4 (BW)   : ${SP_ENGINE_FP4}")
message(STATUS "  auto-patch : ${SP_ENGINE_AUTO_PATCH}")
message(STATUS "  CLI        : ${SP_ENGINE_BUILD_CLI}")
message(STATUS "  tests      : ${SP_ENGINE_BUILD_TESTS}")
if(SP_ENGINE_WITH_CUDA)
    message(STATUS "  CUDA archs : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
