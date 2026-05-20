@echo off
REM Phase 8f — n=3 replication of the Phase 8e multi-needle RULER.
REM
REM Each trial uses a different cluster_pct value, which the harness
REM uses only as an rng seed (the actual planted-block insertion point
REM is now layout-fixed inside build_corpora).  Same 5 needles, same
REM byte layout, different surrounding haystack text per trial.  This
REM isolates the variance to "different background context" and lets
REM us put a CI on the 4.87× ratio observed at Phase 8e.
REM
REM Total wall: 3 trials × 4 cells × ~9 min = ~108 min.

cd /d D:\F\shannon-prime-repos\shannon-prime-engine\bench

set PROG=replicate_multi_n3_progress.txt
echo Phase 8f replication starting %DATE% %TIME% > %PROG%
echo. >> %PROG%

for %%P in (0.48 0.50 0.52) do (
  echo === trial cluster_pct=%%P === >> %PROG%
  python -u _ruler_multi.py ^
    --ctx 512 --chunks 2 --bracket 4 ^
    --cluster-pct %%P ^
    --out-json "ruler_multi_n3_p%%P.json" ^
    --tmp "tmp_ruler_multi_n3_p%%P" ^
    > "ruler_multi_n3_p%%P.stdout" 2> "ruler_multi_n3_p%%P.stderr"
  type "ruler_multi_n3_p%%P.stdout" >> %PROG%
  echo. >> %PROG%
)

echo DONE %DATE% %TIME% >> %PROG%
