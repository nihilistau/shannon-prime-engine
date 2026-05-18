@echo off
REM Phase 15a verification: GGUF Q4_0 ingest with per-block scale fusion.
REM Target: PPL in a usable range (not the Phase 14 per-tensor Q4 blowout
REM of 1.16e9). GGUF per-block calibration is what makes this possible.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf" ^
    --frobenius-quant --gguf-block-quant ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase15_block_q4.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase15_block_q4.log
