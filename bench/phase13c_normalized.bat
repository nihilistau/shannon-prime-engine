@echo off
REM Phase 13.C — recall + inject with HRR ||q||^2 normalization
REM and write_stride=8 (≈16 memories per slab per chunk eviction).
REM Two scenarios:
REM   1. alpha=0.01 → modest bias, expect PPL near baseline
REM   2. alpha=0.10 → stronger bias, expect PPL deviation

set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
set SP_ENGINE_MEMORY=1
set SP_ENGINE_MEMORY_WRITE_STRIDE=8
set MODEL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf
cd /d D:\F\shannon-prime-repos\shannon-prime-engine

echo === alpha=0.01 stride=8 ===
set SP_ENGINE_MEMORY_ALPHA=0.01
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "%MODEL%" --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 bench\test_corpus.txt > bench\phase13c_n_a010.log 2>&1
echo DONE_a010 %ERRORLEVEL%

echo === alpha=0.10 stride=8 ===
set SP_ENGINE_MEMORY_ALPHA=0.1
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "%MODEL%" --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 bench\test_corpus.txt > bench\phase13c_n_a100.log 2>&1
echo DONE_a100 %ERRORLEVEL%
