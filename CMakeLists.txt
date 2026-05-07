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
option(SP_ENGINE_WITH_QNN        "Enable Qualcomm QNN HTP backend (V69+ NPU dispatch)" OFF)
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
    ${SP_CORE_DIR}/core/shannon_prime_modelpack.c
    ${SP_CORE_DIR}/backends/crt/sp_crt.c
    ${SP_CORE_DIR}/backends/crt/sp_crt_dispatch.c)
# When CUDA is OFF, link stubs for the CRT CUDA/Vulkan symbols so the
# pure-C dispatch code resolves without the real .cu implementations.
if(NOT SP_ENGINE_WITH_CUDA)
    list(APPEND SP_CORE_SRC ${SP_CORE_DIR}/backends/crt/sp_crt_cuda_stub.c)
endif()
add_library(shannon_prime_core STATIC ${SP_CORE_SRC})
target_include_directories(shannon_prime_core PUBLIC
    ${SP_CORE_DIR}/core
    ${SP_CORE_DIR}/backends/crt
    ${CMAKE_SOURCE_DIR}/vendor/ggml/include)
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
        ${SP_CORE_DIR}/backends/cuda/shannon_prime_hier.cu
        ${SP_CORE_DIR}/backends/crt/sp_crt_cuda.cu
        ${SP_CORE_DIR}/backends/crt/sp_shor_kernel.cu)

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
        ${SP_CORE_DIR}/backends/crt
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
    if(SP_ENGINE_HEXAGON_FASTRPC)
        # qaic-generated stub. Generated by running build.cmd in
        # ${SP_CORE_DIR}/backends/hexagon/scaffold/ — produces sp_hex.h,
        # sp_hex_stub.c, sp_hex_skel.c. Same scaffold + same generated
        # outputs that shannon-prime-llama uses (verified byte-identical).
        set(_SP_HEX_SCAFFOLD ${SP_CORE_DIR}/backends/hexagon/scaffold/android_Debug_aarch64)
        if(NOT EXISTS "${_SP_HEX_SCAFFOLD}/sp_hex_stub.c")
            message(FATAL_ERROR
                "SP_ENGINE_HEXAGON_FASTRPC=ON but qaic-generated stubs not found at "
                "${_SP_HEX_SCAFFOLD}/sp_hex_stub.c — run "
                "${SP_CORE_DIR}/backends/hexagon/scaffold/build.cmd android first")
        endif()
        list(APPEND _SP_HEXAGON_SOURCES ${_SP_HEX_SCAFFOLD}/sp_hex_stub.c)
    endif()
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
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
            ${_SP_HEX_SCAFFOLD}
            # qaic-generated sp_hex.h — generated from sp_hex.idl by 'build.cmd arm'
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/shannon-prime/backends/hexagon/scaffold/src_generated)
        # FastRPC link deps. Mirrors shannon-prime-llama/android_build/
        # build_llama_cli_hexagon.sh — rpcmem.a is a static archive,
        # libcdsprpc.so is on-device dynamic. -llog satisfies
        # __android_log_print referenced from rpcmem_android.c.
        set(_RPCMEM_LIB   ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a)
        set(_CDSPRPC_LIB  ${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so)
        if(NOT EXISTS "${_RPCMEM_LIB}")
            message(FATAL_ERROR "rpcmem.a not found at ${_RPCMEM_LIB}")
        endif()
        if(NOT EXISTS "${_CDSPRPC_LIB}")
            message(FATAL_ERROR "libcdsprpc.so not found at ${_CDSPRPC_LIB}")
        endif()
        target_link_libraries(shannon_prime_hexagon_backend PUBLIC
            ${_RPCMEM_LIB} ${_CDSPRPC_LIB} log dl)
        message(STATUS "Shannon-Prime Hexagon: real FastRPC mode (SDK=${HEXAGON_SDK_ROOT})")
    else()
        message(STATUS "Shannon-Prime Hexagon: stub mode (no SDK; backend returns -1 fallback)")
    endif()
endif()

# --------------------------------------------------------------------
# QNN HTP backend: V69+ NPU dispatch via Qualcomm AI Engine Direct (QNN).
#
# Compiles the sp_qnn primitive (runtime graph build + ION-backed
# persistent K) from lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/.
# These are pure host-side libs that dlopen libQnnHtp.so + libQnnSystem.so
# at runtime — no link-time dependency on QNN libs, just the headers.
#
# Caller must point QAIRT_ROOT at a Qualcomm AI Engine Direct install
# (the SDK that ships the QNN headers). On Windows that's typically
# C:/Qualcomm/AIStack/QAIRT/2.45.40.260406. On Linux it's wherever the
# SDK is unpacked. The QNN runtime libs themselves (libQnnHtp.so etc.)
# only need to be present on the device at runtime, set in
# LD_LIBRARY_PATH.
#
# Activated at runtime via SHANNON_PRIME_QNN=1 env var (gated inside
# forward.cpp / Engine::generate). Turning the option ON without the
# runtime gate produces a binary that COULD dispatch but won't unless
# explicitly opted in.
# --------------------------------------------------------------------
if(SP_ENGINE_WITH_QNN)
    if(NOT QAIRT_ROOT)
        message(FATAL_ERROR
            "SP_ENGINE_WITH_QNN=ON requires QAIRT_ROOT to point at a "
            "Qualcomm AI Engine Direct install. Example: "
            "-DQAIRT_ROOT=C:/Qualcomm/AIStack/QAIRT/2.45.40.260406")
    endif()
    if(NOT EXISTS "${QAIRT_ROOT}/include/QNN/QnnInterface.h")
        message(FATAL_ERROR
            "QAIRT_ROOT='${QAIRT_ROOT}' does not contain "
            "include/QNN/QnnInterface.h — verify the SDK install path.")
    endif()

    set(_SP_QNN_RUNNER ${SP_CORE_DIR}/backends/qnn_aihub/sp_qnn_runner)
    set(_SP_QNN_SOURCES
        ${_SP_QNN_RUNNER}/sp_qnn.c
        ${_SP_QNN_RUNNER}/sp_llama_qnn.c)
    add_library(shannon_prime_qnn_backend STATIC ${_SP_QNN_SOURCES})
    target_include_directories(shannon_prime_qnn_backend PUBLIC
        ${_SP_QNN_RUNNER}
        ${QAIRT_ROOT}/include/QNN
        ${QAIRT_ROOT}/include
        ${SP_CORE_DIR}/core)
    target_link_libraries(shannon_prime_qnn_backend PUBLIC shannon_prime_core)
    target_compile_features(shannon_prime_qnn_backend PUBLIC c_std_11)
    set_property(TARGET shannon_prime_qnn_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
    # sp_qnn dlopen's libQnnHtp.so + libcdsprpc.so at runtime — needs libdl on
    # POSIX targets (Android/Linux). MSVC has dlopen via a shim or it's a no-op.
    if(NOT WIN32)
        target_link_libraries(shannon_prime_qnn_backend PUBLIC dl)
    endif()
    # QnnTypes.h has enum values that exceed signed int range; silence the
    # warning since it's a Qualcomm header issue, not our code.
    target_compile_options(shannon_prime_qnn_backend PRIVATE -Wno-pedantic)
    message(STATUS "Shannon-Prime QNN: enabled (QAIRT_ROOT=${QAIRT_ROOT})")
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
    src/sp_tensor.cpp
    src/sp_quant.cpp
    src/sp_kernels_cpu.cpp
    src/sp_threadpool.cpp
    src/forward_native.cpp
    src/forward_native_context.cpp
    src/speculative_oracle.cpp
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
        target_compile_definitions(sp_engine PUBLIC SP_ENGINE_HEXAGON_FASTRPC=1)
        # rpcmem + cdsprpc are the two Android-side FastRPC libraries.
        # rpcmem is a static archive shipped in the Hexagon SDK; cdsprpc is
        # the dynamic stub (.so) that wraps the kernel FastRPC driver. Both
        # live at ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/ship/android_aarch64
        # and the companion DSP stub .so lives in ${HEXAGON_SDK_ROOT}/libs.
        # rpcmem static archive for ARM (ion buffer allocator used by FastRPC).
        target_link_libraries(sp_engine PUBLIC
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a)
        # libcdsprpc.so is a device-only runtime library (FastRPC kernel shim).
        # Link the SDK's host stub so the linker records the soname in DT_NEEDED
        # and resolves the remote_* symbols. On device, /vendor/lib64/libcdsprpc.so
        # satisfies the dependency. The library only initializes the CDSP context
        # when remote_handle64_open() is called (lazily, after Split 4 executes),
        # so DT_NEEDED eager-load does NOT pre-empt QNN HTP memory allocation.
        set(_CDSPRPC_SO
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so)
        if(EXISTS ${_CDSPRPC_SO})
            add_library(cdsprpc_imported SHARED IMPORTED)
            set_target_properties(cdsprpc_imported PROPERTIES
                IMPORTED_LOCATION ${_CDSPRPC_SO}
                IMPORTED_SONAME   libcdsprpc.so)
            target_link_libraries(sp_engine PUBLIC cdsprpc_imported)
        else()
            message(WARNING "libcdsprpc.so not found at ${_CDSPRPC_SO}; "
                            "FastRPC will fail at runtime if not present on device")
        endif()
        # rpcmem_android.c calls __android_log_print which lives in liblog.
        target_link_libraries(sp_engine PUBLIC log)
        # qaic-generated sp_hex_stub.c must be compiled into the engine when
        # FastRPC is on. The stub file is generated by the scaffold build and
        # lives at scaffold/build/<variant>/sp_hex_stub.c — but for the engine
        # Android cross-build we pre-generate and commit it in src_generated/.
        # If FASTRPC is on and the stub source exists, compile it in.
        set(_SP_HEX_STUB_DIR
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/shannon-prime/backends/hexagon/scaffold/src_generated)
        if(EXISTS ${_SP_HEX_STUB_DIR}/sp_hex_stub.c)
            target_sources(sp_engine PRIVATE ${_SP_HEX_STUB_DIR}/sp_hex_stub.c)
            target_include_directories(sp_engine PRIVATE ${_SP_HEX_STUB_DIR})
        else()
            message(WARNING
                "SP_ENGINE_HEXAGON_FASTRPC=ON but qaic-generated stub not found at "
                "${_SP_HEX_STUB_DIR}/sp_hex_stub.c — run 'build.cmd arm' in the "
                "scaffold directory first, then copy sp_hex_stub.{c,h} here.")
        endif()
    endif()
endif()
if(SP_ENGINE_WITH_QNN)
    target_link_libraries(sp_engine PUBLIC shannon_prime_qnn_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_QNN=1)
    target_include_directories(sp_engine PUBLIC ${SP_CORE_DIR}/backends/qnn_aihub/sp_qnn_runner)
    # Phase 5: V69 QNN context-binary driver (load AI Hub-compiled .bins).
    target_sources(sp_engine PRIVATE src/qnn_bin_driver.cpp)
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
message(STATUS "  QNN HTP    : ${SP_ENGINE_WITH_QNN}")
message(STATUS "  FP8 (E4M3) : ${SP_ENGINE_FP8}")
message(STATUS "  FP4 (BW)   : ${SP_ENGINE_FP4}")
message(STATUS "  auto-patch : ${SP_ENGINE_AUTO_PATCH}")
message(STATUS "  CLI        : ${SP_ENGINE_BUILD_CLI}")
message(STATUS "  tests      : ${SP_ENGINE_BUILD_TESTS}")
if(SP_ENGINE_WITH_CUDA)
    message(STATUS "  CUDA archs : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
