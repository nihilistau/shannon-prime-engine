@echo off
REM Phase 8 wall-time scaling ladder.  Phase 7 probe at ctx=2048 confirmed
REM the engine survives prefill + block-Q4 + ultraproduct dispatch (no
REM silent exit), but did not produce a chunk-complete number in
REM ~18 min wall time.  This ladder establishes the aN + bN^2 scaling
REM coefficients at smaller ctx so we can project ctx=8k / ctx=32k
REM honestly before committing to a multi-hour bench.
REM
REM Six cells: ctx in {512, 1024, 2048} x mode in {softmax baseline,
REM ultraproduct principal}.  Single chunk each.  SP_ENGINE_PREFILL=1.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1

set PROG=ladder_phase8_progress.txt
echo Phase 8 ladder starting %DATE% %TIME% > %PROG%
echo Settings: prefill=1, gguf-block-quant + frobenius-quant, chunks=1 >> %PROG%
echo. >> %PROG%

for %%C in (512 1024 2048) do (
  echo === ctx=%%C softmax === >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx %%C --chunks 1 ^
    --gguf-block-quant --frobenius-quant ^
    %CORP% > "ladder_ctx%%C_softmax.out" 2> "ladder_ctx%%C_softmax.err"
  type "ladder_ctx%%C_softmax.out" >> %PROG%
  echo. >> %PROG%

  echo === ctx=%%C ultraproduct === >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx %%C --chunks 1 ^
    --gguf-block-quant --frobenius-quant ^
    --ultraproduct-attn principal ^
    %CORP% > "ladder_ctx%%C_ultra.out" 2> "ladder_ctx%%C_ultra.err"
  type "ladder_ctx%%C_ultra.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
