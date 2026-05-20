@echo off
REM Phase 9 — Ramanujan-Fourier lambda A/B on multi-needle RULER.
REM
REM Reuses the Phase 8e corpora (same byte content, same layout).  Only
REM the ultraproduct cells need re-running because lambda is structurally
REM a no-op on the softmax path (it gates entry to the F-over-top-m
REM bracket inside sp_ultraproduct_attn).
REM
REM Cells per lambda:
REM   ultra-b4 + lambda × planted
REM   ultra-b4 + lambda × control
REM
REM Phase 8e softmax baseline (no re-run needed):
REM   planted=13.9257  control=15.8494  ratio=0.879
REM Phase 8e ultra-b4 baseline (lambda=0):
REM   planted=20726.01  control=50565.40  ratio=0.410
REM
REM Wall: 2 cells × 2 lambdas × ~9 min ≈ 36 min total.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench

REM Use the same corpora the Phase 8e v2 run built.
set PLANTED=tmp_ruler_multi\multi_planted.txt
set CONTROL=tmp_ruler_multi\multi_control.txt
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set COMMON=--ctx 512 --chunks 2 --gguf-block-quant --frobenius-quant --ultraproduct-attn principal --ultraproduct-bracket 4
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1

set PROG=phase9_lambda_sweep_progress.txt
echo Phase 9 lambda sweep starting %DATE% %TIME% > %PROG%
echo Baseline Phase 8e: softmax 13.9257/15.8494 (0.879), ultra-b4 20726.01/50565.40 (0.410) >> %PROG%
echo. >> %PROG%

for %%L in (0.05 0.20) do (
  echo === lambda=%%L === >> %PROG%
  echo --- ultra-b4-L%%L planted --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" %COMMON% --kste-ramanujan-lambda %%L %PLANTED% > "phase9_L%%L_planted.out" 2> "phase9_L%%L_planted.err"
  type "phase9_L%%L_planted.out" >> %PROG%
  echo. >> %PROG%

  echo --- ultra-b4-L%%L control --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" %COMMON% --kste-ramanujan-lambda %%L %CONTROL% > "phase9_L%%L_control.out" 2> "phase9_L%%L_control.err"
  type "phase9_L%%L_control.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
