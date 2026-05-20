@echo off
setlocal enabledelayedexpansion
cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\lmstudio-community\functiongemma-270m-it-GGUF\functiongemma-270m-it-F16.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1
echo starting %DATE% %TIME% > sweep_progress.txt
for %%T in (0.0500 0.1000 0.2000 0.4000) do (
  for %%A in (0.3000 0.5000 0.7000) do (
    echo --- tau=%%T alpha=%%A ---
    "%ENG%" perplexity --model "%MDL%" --ctx 64 --chunks 1 --frobenius-quant --friedman-sieve --friedman-mode policy --kste-tau-A %%T --kste-alpha %%A %CORP% > "sweep_t%%T_a%%A.out" 2> "sweep_t%%T_a%%A.err"
    type "sweep_t%%T_a%%A.out" | findstr /R "^perplexity ^sieve" >> sweep_progress.txt
    echo tau=%%T alpha=%%A done >> sweep_progress.txt
  )
)
echo DONE >> sweep_progress.txt
