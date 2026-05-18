@echo off
REM Phase 14c verification: AVX-512 mullo_epi64 SIMD path ACTIVE.
REM PPL invariant target: 11.8311 (Step E baseline).
REM Wall-clock target: meaningfully less than scalar 221.5 s.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf" ^
    --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase14c_q8_simd_active.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase14c_q8_simd_active.log
