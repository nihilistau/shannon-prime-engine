@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set VULKAN_SDK=C:\VulkanSDK\1.4.341.1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CudaToolkitDir=%CUDA_PATH%
set PATH=%VULKAN_SDK%\Bin;%CUDA_PATH%\bin;%PATH%
set SHANNON_PRIME_VERBOSE=1

set EXE=D:\F\shannon-prime-repos\shannon-prime-engine\build\bin\sp-engine.exe
set MODEL=D:\Files\Models\Qwen\Qwen2.5-Coder-3B-Instruct-GGUF\qwen2.5-coder-3b-instruct-q4_k_m.gguf
set CORPUS=D:\F\shannon-prime-repos\shannon-prime-engine\bench\test_corpus.txt
set OUTDIR=D:\F\shannon-prime-repos\shannon-prime-engine\bench\crt_bench

mkdir "%OUTDIR%" 2>nul

echo ============================================
echo CONFIG 1: CPU-only (no GPU layers)
echo ============================================
set SP_ENGINE_BACKEND=cpu
"%EXE%" cache_ppl --model "%MODEL%" --hierarchical --ctx 512 --chunks 4 "%CORPUS%" > "%OUTDIR%\1_cpu.txt" 2>&1
echo CPU done.

echo ============================================
echo CONFIG 2: dGPU only (RTX 2060 CUDA)
echo ============================================
set SP_ENGINE_BACKEND=cuda
"%EXE%" cache_ppl --model "%MODEL%" --hierarchical --ctx 512 --chunks 4 --n-gpus 1 "%CORPUS%" > "%OUTDIR%\2_dgpu.txt" 2>&1
echo dGPU done.

echo ============================================
echo CONFIG 3: iGPU only (Intel UHD Vulkan)
echo ============================================
set SP_ENGINE_BACKEND=vulkan
"%EXE%" cache_ppl --model "%MODEL%" --hierarchical --ctx 512 --chunks 4 --n-gpus 1 "%CORPUS%" > "%OUTDIR%\3_igpu.txt" 2>&1
echo iGPU done.

echo ============================================
echo CONFIG 4: CRT split (RTX 2060 + Intel UHD)
echo ============================================
set SP_ENGINE_BACKEND=cuda
"%EXE%" cache_ppl --model "%MODEL%" --hierarchical --ctx 512 --chunks 4 --crt-split "%CORPUS%" > "%OUTDIR%\4_crt.txt" 2>&1
echo CRT done.

echo ALL DONE
