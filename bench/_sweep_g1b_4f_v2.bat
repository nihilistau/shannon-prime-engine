@echo off
REM Phase 4f v2 — wide τ_A sweep with cap fixed at engine default.
REM
REM Finding from v1: at ctx=128 chunks=4, the chunk-to-chunk eviction
REM cascade amplifies sieve PPL impact dramatically vs the single-chunk
REM Phase 4e regime.  τ_A=0.07/0.08/0.09/0.10 all produce PPL 40-200
REM range — eviction rate ≈ 42% even at cap=4096 (caches never fill;
REM dominance subsumption does ALL the eviction work).
REM
REM Capacity confirmed irrelevant (cap=1024 c0700 == cap=4096 c0700 to
REM the printed decimal).  Sweep here drops cap variation and walks τ_A
REM HIGHER, looking for the regime where eviction rate drops below 5%
REM (which Phase 4e suggests is needed for the T2.3 |Δ|≤0.5% gate).

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4f_v2_progress.txt
echo Phase 4f v2 sweep starting %DATE% %TIME% > %PROG%
echo Engine: %ENG% >> %PROG%
echo Model: %MDL% >> %PROG%
echo Corpus: %CORP% >> %PROG%
echo Settings: ctx=128 chunks=4 frobenius-quant + gguf-block-quant cap=4096 >> %PROG%
echo Baseline (carried over from v1): PPL_native = 10.4658 >> %PROG%
echo. >> %PROG%

REM τ_A walk from 0.20 (start of Phase 4e knee) to 5.0 (effectively sieve-off)
for %%T in (0.2000 0.3000 0.5000 0.7000 1.0000 1.5000 2.0000 3.0000 5.0000) do (
  echo --- tau=%%T --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
    --gguf-block-quant --frobenius-quant ^
    --friedman-sieve --friedman-mode policy ^
    --friedman-capacity 4096 ^
    --kste-tau-A %%T --kste-alpha 0.5000 ^
    %CORP% > "sweep_4f_v2_t%%T.out" 2> "sweep_4f_v2_t%%T.err"
  type "sweep_4f_v2_t%%T.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
