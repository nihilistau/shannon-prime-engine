@echo off
REM Phase 13.B verification: prefill + ARM write hook at chunk boundary.
REM Expect PPL == 11.8311 (writes are side-effect; forward unchanged).
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
set SP_ENGINE_MEMORY=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf" ^
    --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase13b_write_only.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase13b_write_only.log
