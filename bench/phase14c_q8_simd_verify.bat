@echo off
REM Step E prefill verification: batched n_tokens=n_ctx prefill.
REM Expect PPL == 11.8311 (Step E-pre N=1 baseline) by mathematical equivalence.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf" ^
    --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase14c_q8_simd.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase14c_q8_simd.log
