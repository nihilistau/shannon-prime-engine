@echo off
REM scripts/run_ppl_sieve.bat — Phase-4 PPL gate runner for Windows.
REM
REM Invokes sp-engine.exe twice (baseline, with sieve) and prints the
REM delta percentage.  Exit code 0 iff |delta| <= 0.005 (T2.3 gate).
REM
REM Usage:
REM   scripts\run_ppl_sieve.bat <engine.exe> <model.gguf> <corpus.txt>
REM       [ctx=2048] [chunks=4] [threads=16]
REM
REM Example:
REM   scripts\run_ppl_sieve.bat ^
REM     build-cuda\bin\sp-engine.exe ^
REM     models\gemma3-1b.gguf ^
REM     data\wikitext-103-valid.txt ^
REM     2048 4 16

setlocal enabledelayedexpansion

set ENGINE=%~1
set MODEL=%~2
set CORPUS=%~3
set CTX=%~4
set CHUNKS=%~5
set THREADS=%~6

if "%ENGINE%"=="" goto :usage
if "%MODEL%"==""  goto :usage
if "%CORPUS%"=="" goto :usage
if "%CTX%"==""    set CTX=2048
if "%CHUNKS%"=="" set CHUNKS=4
if "%THREADS%"="" set THREADS=16

echo === Baseline (sieve OFF) ===
"%ENGINE%" perplexity-sp ^
    --model "%MODEL%" --corpus "%CORPUS%" ^
    --ctx %CTX% --chunks %CHUNKS% --threads %THREADS% ^
    --frobenius-quant -p 41 -k 8 ^
    --poly-attn --ntt-crt ^
    > baseline_ppl.txt 2>&1
type baseline_ppl.txt | findstr /R "perplexity[ ]*="

echo.
echo === Sieve ON (policy mode) ===
"%ENGINE%" perplexity-sp ^
    --model "%MODEL%" --corpus "%CORPUS%" ^
    --ctx %CTX% --chunks %CHUNKS% --threads %THREADS% ^
    --frobenius-quant -p 41 -k 8 ^
    --poly-attn --ntt-crt ^
    --friedman-sieve --friedman-mode policy ^
    > sieve_ppl.txt 2>&1
type sieve_ppl.txt | findstr /R "perplexity[ ]*="

echo.
echo === Delta ===
for /f "tokens=3 delims= " %%P in ('findstr /R "perplexity[ ]*=" baseline_ppl.txt') do set BASE_PPL=%%P
for /f "tokens=3 delims= " %%P in ('findstr /R "perplexity[ ]*=" sieve_ppl.txt')    do set SV_PPL=%%P

echo Baseline PPL: %BASE_PPL%
echo Sieve PPL:    %SV_PPL%

REM Compute delta via python (cmd's float arithmetic is awkward).
python -c "import sys; b=float('%BASE_PPL%'); s=float('%SV_PPL%'); d=(s-b)/b; print(f'Delta = {d*100:+.3f}%%  ({s-b:+.4f})'); sys.exit(0 if abs(d)<=0.005 else 1)"
exit /b %ERRORLEVEL%

:usage
echo Usage: %~nx0 ^<engine.exe^> ^<model.gguf^> ^<corpus.txt^> [ctx=2048] [chunks=4] [threads=16]
exit /b 2
