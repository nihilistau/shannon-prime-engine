@echo off
REM Phase 4f — refined Friedman sieve sweep on Gemma3-1B.
REM ctx=128, chunks=4 (~512 tokens) for variance reduction vs Phase 4e ctx=64/1.
REM tau_A grid zooms around the Gemma3-1B knee (4e wide sweep showed tau=0.10
REM hit PPL 11.7453 vs baseline 11.1029, eviction 11.30%).
REM Capacity grid {1024,2048,4096} tests sieve memory sensitivity.
REM Alpha is FIXED at 0.5 — Path B 4-bucket attachment is degenerate at our
REM anchor counts so alpha effect was zero in 4e.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4f_progress.txt
echo Phase 4f sweep starting %DATE% %TIME% > %PROG%
echo Engine: %ENG% >> %PROG%
echo Model: %MDL% >> %PROG%
echo Corpus: %CORP% >> %PROG%
echo Settings: ctx=128 chunks=4 frobenius-quant + gguf-block-quant >> %PROG%
echo. >> %PROG%

echo === BASELINE === >> %PROG%
"%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 --gguf-block-quant --frobenius-quant %CORP% > "sweep_4f_baseline.out" 2> "sweep_4f_baseline.err"
type "sweep_4f_baseline.out" >> %PROG%
echo. >> %PROG%

for %%C in (1024 2048 4096) do (
  for %%T in (0.0700 0.0800 0.0900 0.1000 0.1100 0.1200 0.1500) do (
    echo --- cap=%%C tau=%%T --- >> %PROG%
    set OUTF=sweep_4f_c%%C_t%%T
    "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
      --gguf-block-quant --frobenius-quant ^
      --friedman-sieve --friedman-mode policy ^
      --friedman-capacity %%C ^
      --kste-tau-A %%T --kste-alpha 0.5000 ^
      %CORP% > "sweep_4f_c%%C_t%%T.out" 2> "sweep_4f_c%%C_t%%T.err"
    type "sweep_4f_c%%C_t%%T.out" >> %PROG%
    echo. >> %PROG%
  )
)

echo DONE %DATE% %TIME% >> %PROG%
