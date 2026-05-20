@echo off
cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench
set ENG=..\build-cuda\bin\sp-engine.exe
set MDL=D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf
set CORP=test_corpus.txt
set SP_ENGINE_NATIVE=1
echo starting g1b %DATE% %TIME% > sweep_g1b_progress.txt
echo BASELINE: >> sweep_g1b_progress.txt
"%ENG%" perplexity --model "%MDL%" --ctx 64 --chunks 1 --gguf-block-quant --frobenius-quant %CORP% > "sweep_g1b_baseline.out" 2>nul
type "sweep_g1b_baseline.out" | findstr /R "^perplexity" >> sweep_g1b_progress.txt
for %%T in (0.0500 0.1000 0.2000 0.4000) do (
  echo --- tau=%%T --- >> sweep_g1b_progress.txt
  "%ENG%" perplexity --model "%MDL%" --ctx 64 --chunks 1 --gguf-block-quant --frobenius-quant --friedman-sieve --friedman-mode policy --kste-tau-A %%T --kste-alpha 0.5000 %CORP% > "sweep_g1b_t%%T.out" 2>nul
  type "sweep_g1b_t%%T.out" | findstr /R "^perplexity ^sieve" >> sweep_g1b_progress.txt
)
echo DONE >> sweep_g1b_progress.txt
