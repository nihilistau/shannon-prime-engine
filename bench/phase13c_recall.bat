@echo off
REM Phase 13.C verification harness — runs three scenarios sequentially:
REM   1. alpha=0  → hook gated off, PPL must equal 11.8311 (strict bypass)
REM   2. alpha=0.001 → hook fires, tiny bias, PPL should stay close
REM   3. alpha=0.01  → hook fires, modest bias, see if PPL moves
REM
REM In all three: SP_ENGINE_MEMORY=1 so the WRITE hook accumulates the
REM bank at chunk boundaries. Difference is just the injection strength.

set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
set SP_ENGINE_MEMORY=1
set MODEL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf
cd /d D:\F\shannon-prime-repos\shannon-prime-engine

echo === alpha=0.000 (strict bypass)  ===
set SP_ENGINE_MEMORY_ALPHA=0.0
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "%MODEL%" --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 bench\test_corpus.txt > bench\phase13c_alpha000.log 2>&1
echo DONE %ERRORLEVEL%

echo === alpha=0.001 ===
set SP_ENGINE_MEMORY_ALPHA=0.001
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "%MODEL%" --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 bench\test_corpus.txt > bench\phase13c_alpha001.log 2>&1
echo DONE %ERRORLEVEL%

echo === alpha=0.01 ===
set SP_ENGINE_MEMORY_ALPHA=0.01
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "%MODEL%" --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 bench\test_corpus.txt > bench\phase13c_alpha010.log 2>&1
echo DONE %ERRORLEVEL%
