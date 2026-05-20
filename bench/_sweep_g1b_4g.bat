@echo off
REM Phase 4g — soft-attenuation mask sweep on Gemma3-1B at ctx=128/chunks=4.
REM
REM Anchored at the Phase 4f U-floor τ_A = 0.30 (eviction 34.35%, hard-mask
REM PPL 34.7317).  Walk γ from 0 (= hard NEG_INF, baseline for this point)
REM up through the regime where evicted positions retain progressively more
REM softmax mass.  Mathematically: after the (score -= γ) edit, an evicted
REM position keeps exp(-γ) × its un-evicted softmax weight before renorm.
REM   γ = 0     → exp(0)  = 1.000  (no effect; identical to baseline)
REM   γ = 0.5   → 0.607
REM   γ = 1.0   → 0.368
REM   γ = 2.0   → 0.135
REM   γ = 4.0   → 0.0183
REM   γ = 8.0   → 3.35e-4
REM   γ = 16.0  → 1.13e-7
REM   γ = 32.0  → 1.27e-14 (≈ hard mask)
REM
REM IMPORTANT: γ = 0 in the new code path means "no attenuation, no eviction
REM effect" — that should reproduce the baseline PPL 10.4658 (sanity check
REM that the mask code is correctly bypassed at γ = 0).  γ ≥ ~30 should
REM reproduce the Phase 4f hard-mask number (34.7317).  The interesting
REM regime is in between.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1

set PROG=sweep_4g_progress.txt
echo Phase 4g sweep starting %DATE% %TIME% > %PROG%
echo Anchored at tau_A=0.30 (Phase 4f U-floor) >> %PROG%
echo Baseline PPL (sieve off, from Phase 4f): 10.4658 >> %PROG%
echo Phase 4f hard-mask PPL at tau=0.30: 34.7317 >> %PROG%
echo Gate band [10.4135, 10.5181] >> %PROG%
echo. >> %PROG%

for %%G in (0.0000 0.5000 1.0000 2.0000 4.0000 8.0000 16.0000 32.0000) do (
  echo --- gamma=%%G --- >> %PROG%
  "%ENG%" perplexity --model "%MDL%" --ctx 128 --chunks 4 ^
    --gguf-block-quant --frobenius-quant ^
    --friedman-sieve --friedman-mode policy ^
    --friedman-capacity 4096 ^
    --kste-tau-A 0.3000 --kste-alpha 0.5000 ^
    --friedman-gamma %%G ^
    %CORP% > "sweep_4g_g%%G.out" 2> "sweep_4g_g%%G.err"
  type "sweep_4g_g%%G.out" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
