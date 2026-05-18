@echo off
REM Phase 15a Q4_0 (QAT variant): quantization-aware-trained model, often
REM cleaner than vanilla llama-quantize output (less likely to mix Q4_1).
set SHANNON_PRIME_VERBOSE=1
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1
cd /d D:\F\shannon-prime-repos\shannon-prime-engine
.\build-cuda\bin\sp-engine.exe perplexity ^
    --model "D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-QAT-Q4\gemma-3-1b-it-q4_0.gguf" ^
    --frobenius-quant --gguf-block-quant ^
    --ctx 128 --chunks 4 ^
    bench\test_corpus.txt > bench\phase15_block_q4_qat.log 2>&1
echo DONE %ERRORLEVEL% >> bench\phase15_block_q4_qat.log
