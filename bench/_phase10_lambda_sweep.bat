@echo off
REM Phase 10 — Ramanujan-Fourier lambda A/B with splitmix64-shuffled q-bank
REM index.  Reuses the Phase 8e / Phase 9 corpora byte-for-byte.
REM
REM Diagnostic test of the period-5 carrier-wave hypothesis: if the
REM splitmix64 shuffle restores Phase 8e's 0.410 ratio at lambda=0.05,
REM pre-VHT2 modulation is salvageable.  Same flag (--kste-ramanujan-lambda),
REM same harness layout, same kernel — only the bank-index lookup inside
REM sp_kste_ramanujan_modulate changed.
REM
REM Cells per lambda:
REM   ultra-b4 + lambda x planted
REM   ultra-b4 + lambda x control
REM
REM Phase 8e baselines (no re-run needed):
REM   softmax           planted=13.9257    control=15.8494    ratio=0.879
REM   ultra-b4 lambda=0 planted=20726.01   control=50565.40   ratio=0.410
REM
REM Phase 9 lambda values for direct comparison:
REM   lambda=0.05       planted=39077.94   control=84106.82   ratio=0.465
REM   lambda=0.20       planted=66162.47   control=90412.96   ratio=0.732
REM
REM Phase 10 success criterion at lambda=0.05:
REM   ratio <= 0.43           : pre-transform theory CONFIRMED
REM   ratio in (0.43, 0.46)   : ambiguous — shuffle helped partially
REM   ratio >= 0.46           : pre-transform theory FAILS → Option 2 forced
REM
REM Wall: 2 cells x 2 lambdas x ~9 min ~= 36 min total.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench

set PLANTED=tmp_ruler_multi\multi_planted.txt
set CONTROL=tmp_ruler_multi\multi_control.txt
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set COMMON=--ctx 512 --chunks 2 --gguf-block-quant --frobenius-quant --ultraproduct-attn principal --ultraproduct-bracket 4
set SP_ENGINE_NATIVE=1
set SP_ENGINE_PREFILL=1

set PROG=phase10_lambda_sweep_progress.txt
echo Phase 10 lambda sweep starting %DATE% %TIME% > %PROG%
echo Phase 8e baseline: softmax 13.9257/15.8494 (0.879), ultra-b4 20726.01/50565.40 (0.410) >> %PROG%
echo Phase 9 baseline:  L0.05 39077.94/84106.82 (0.465), L0.20 66162.47/90412.96 (0.732) >> %PROG%
echo Phase 10 patch: splitmix64-shuffled q-bank index (seed 0x9E3779B97F4A7C15) >> %PROG%
echo. >> %PROG%

for %%L in (0.05 0.20) do (
  echo === lambda=%%L === >> %PROG%
  echo --- ultra-b4-L%%L planted --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" %COMMON% --kste-ramanujan-lambda %%L %PLANTED% > "phase10_L%%L_planted.out" 2> "phase10_L%%L_planted.err"
  type "phase10_L%%L_planted.out" >> %PROG%
  echo. >> %PROG%

  echo --- ultra-b4-L%%L control --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" %COMMON% --kste-ramanujan-lambda %%L %CONTROL% > "phase10_L%%L_control.out" 2> "phase10_L%%L_control.err"
  type "phase10_L%%L_control.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
