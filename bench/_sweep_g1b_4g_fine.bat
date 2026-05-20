@echo off
REM Phase 4g fine sweep — zoom on γ ∈ [0.3, 0.8] at tau_A = 0.30.
REM γ = 0.5 -> PPL 10.3777 (Δ -0.84%, just outside T2.3 gate)
REM γ = 1.0 -> PPL 10.5919 (Δ +1.21%, also outside gate)
REM Target: find γ ∈ (0.5, 1.0) where PPL ∈ [10.4135, 10.5181] (T2.3 |Δ|≤0.5%).
REM
REM The PPL curve is monotonic in γ across this band (verified by the wide
REM sweep), so a single γ value should land inside the gate.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4g_fine_progress.txt
echo Phase 4g fine sweep starting %DATE% %TIME% > %PROG%
echo. >> %PROG%

for %%G in (0.3000 0.4000 0.6000 0.7000 0.8000 0.9000) do (
  echo --- gamma=%%G --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
    --gguf-block-quant --frobenius-quant ^
    --friedman-sieve --friedman-mode policy ^
    --friedman-capacity 4096 ^
    --kste-tau-A 0.3000 --kste-alpha 0.5000 ^
    --friedman-gamma %%G ^
    %CORP% > "sweep_4g_fine_g%%G.out" 2> "sweep_4g_fine_g%%G.err"
  type "sweep_4g_fine_g%%G.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
