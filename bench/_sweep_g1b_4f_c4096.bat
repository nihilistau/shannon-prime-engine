@echo off
REM Phase 4f — focused sweep at cap=4096 only.
REM cap=1024 cells in the wider sweep showed catastrophic chaotic PPL
REM (40-200) at 40% eviction rate; that capacity is too small for the
REM Gemma3-1B layer count × kv-head × ctx schedule.  Phase 4e found the
REM operating knee at cap=4096 (engine default), so this run zooms there.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4f_c4096_progress.txt
echo Phase 4f c4096 sweep starting %DATE% %TIME% > %PROG%
echo Engine: %ENG% >> %PROG%
echo Model: %MDL% >> %PROG%
echo Corpus: %CORP% >> %PROG%
echo Settings: ctx=128 chunks=4 frobenius-quant + gguf-block-quant cap=4096 >> %PROG%
echo. >> %PROG%

REM Baseline already done in sweep_4f_baseline.out (PPL_native = 10.4658).
REM Reuse it to save 4 minutes.
echo Re-using baseline from prior run: PPL_native = 10.4658 >> %PROG%
echo. >> %PROG%

for %%T in (0.0700 0.0800 0.0900 0.1000 0.1100 0.1200 0.1500) do (
  echo --- cap=4096 tau=%%T --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
    --gguf-block-quant --frobenius-quant ^
    --friedman-sieve --friedman-mode policy ^
    --friedman-capacity 4096 ^
    --kste-tau-A %%T --kste-alpha 0.5000 ^
    %CORP% > "sweep_4f_c4096_t%%T.out" 2> "sweep_4f_c4096_t%%T.err"
  type "sweep_4f_c4096_t%%T.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
