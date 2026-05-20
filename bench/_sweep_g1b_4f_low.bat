@echo off
REM Phase 4f LOW — bottom of the τ_A range.
REM
REM Findings from v1 and v2: at ctx=128 chunks=4 on Gemma3-1B, eviction
REM RATE goes UP with τ_A — counter-intuitive but explained: at higher
REM τ_A only the top |anchor| components survive, so trees collapse to
REM coarse generic shapes that dominate each other.  Lower τ_A → more
REM detailed trees → fewer dominance hits.
REM
REM Walk τ_A from 0.00 (default, accept-all anchors) up to 0.05.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4f_low_progress.txt
echo Phase 4f LOW sweep starting %DATE% %TIME% > %PROG%
echo Baseline (from v1): PPL_native = 10.4658 >> %PROG%
echo. >> %PROG%

for %%T in (0.0000 0.0050 0.0100 0.0200 0.0300 0.0500) do (
  echo --- tau=%%T --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
    --gguf-block-quant --frobenius-quant ^
    --friedman-sieve --friedman-mode policy ^
    --friedman-capacity 4096 ^
    --kste-tau-A %%T --kste-alpha 0.5000 ^
    %CORP% > "sweep_4f_low_t%%T.out" 2> "sweep_4f_low_t%%T.err"
  type "sweep_4f_low_t%%T.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
