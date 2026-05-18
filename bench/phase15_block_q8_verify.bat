@echo off
REM Phase 15a verification: GGUF Q8_0 ingest with per-block scale fusion.
REM Target PPL: ideally <= 11.8311 (Step E baseline). The fp16 source for
REM the baseline is the same Gemma3-1B model; this run reads from a
REM pre-quantised Q8_0 GGUF where GGUF's per-block calibration kicks in.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q8_0\gemma-3-1b-it-Q8_0.gguf" ^
    --frobenius-quant --gguf-block-quant ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase15_block_q8.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase15_block_q8.log
