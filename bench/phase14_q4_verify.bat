@echo off
REM Phase 14 verification: Q4 packed weights vs Step E Q8 baseline (11.8311).
REM Same fixture (Gemma3-1B-f16, ctx=128, chunks=4) as bench\stepE_prefill_verify.bat
REM so PPLs are apples-to-apples.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf" ^
    --frobenius-quant --frobenius-q4 ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase14_q4_unpruned.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase14_q4_unpruned.log
