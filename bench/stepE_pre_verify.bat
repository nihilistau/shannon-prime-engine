@echo off
REM Step E-pre layout-flip verification: rerun Phase 12 Step D bench, expect PPL = 11.8313
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-f16.gguf" ^
    --frobenius-quant --frobenius-q8 ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\stepE_pre_layout_flip.log 2>&1
echo DONE %ERRORLEVEL% >> bench\stepE_pre_layout_flip.log
