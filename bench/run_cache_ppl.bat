@echo off
REM Shannon-Prime Engine — cache_ppl benchmark suite
REM Requires: CUDA 13.2 + VS18 build in ..\build\bin\sp-engine.exe
REM Model:    Qwen3-0.6B-Q4_K_M.gguf (symlinked from D:\Files\Models)
REM Text:     test_corpus.txt (~200KB concatenated prompts)

call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CudaToolkitDir=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%
set SHANNON_PRIME_VERBOSE=1

set ENGINE=..\build\bin\sp-engine.exe
set MODEL=Qwen3-0.6B-Q4_K_M.gguf
set TEXT=test_corpus.txt
set CTX=256
set CHUNKS=2

echo ============================================================
echo  1. DEFAULT: skel 5,5  K2/V2  no ternary
echo ============================================================
%ENGINE% cache_ppl --model %MODEL% --hierarchical --hier-res-bits 2 --ctx %CTX% --chunks %CHUNKS% %TEXT%

echo.
echo ============================================================
echo  2. TERNARY BAND 1: skel 5,tern  K2/V2
echo ============================================================
%ENGINE% cache_ppl --model %MODEL% --hierarchical --hier-res-bits 2 --hier-ternary-mask 0x2 --ctx %CTX% --chunks %CHUNKS% %TEXT%

echo.
echo ============================================================
echo  3. SPLIT K/V: skel 5,5  K1/V2
echo ============================================================
%ENGINE% cache_ppl --model %MODEL% --hierarchical --hier-res-bits 1 --hier-res-bits-v 2 --ctx %CTX% --chunks %CHUNKS% %TEXT%

echo.
echo ============================================================
echo  4. COMBO: skel 5,tern  K1/V2  (maximum compression)
echo ============================================================
%ENGINE% cache_ppl --model %MODEL% --hierarchical --hier-res-bits 1 --hier-res-bits-v 2 --hier-ternary-mask 0x2 --ctx %CTX% --chunks %CHUNKS% %TEXT%

echo.
echo ============================================================
echo  5. VANILLA (no compression, reference PPL)
echo ============================================================
%ENGINE% cache_ppl --model %MODEL% --ctx %CTX% --chunks %CHUNKS% %TEXT%

echo.
echo Done.
pause
