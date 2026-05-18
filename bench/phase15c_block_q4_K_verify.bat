@echo off
REM Phase 15c: GGUF Q4_K_M ingest. Most common production format.
REM 256-element super-blocks with 8x32 sub-blocks, 6-bit (sc, m) per
REM sub-block, fp16 (d, dmin) per super-block. Fans out to 8 q4_1
REM sub-blocks in our storage, reuses the Q4_1 kernel.
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_K_M\gemma-3-1b-it-Q4_K_M.gguf" ^
    --frobenius-quant --gguf-block-quant ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase15c_block_q4_K.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase15c_block_q4_K.log
