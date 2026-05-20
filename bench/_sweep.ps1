Set-Location D:\F\shannon-prime-repos\shannon-prime-engine\bench
$model_270 = 'D:\Files\Models\lmstudio-community\functiongemma-270m-it-GGUF\functiongemma-270m-it-F16.gguf'
$baseline_270 = 10.4159
"" | Set-Content .\sweep_progress.txt
$ledger = @()
foreach ($tau in @(0.05, 0.10, 0.20, 0.40)) {
    foreach ($alpha in @(0.3, 0.5, 0.7)) {
        $nm = "270_t${tau}_a${alpha}"
        $args = @('perplexity','--model',$model_270,'--ctx','128','--chunks','1','--frobenius-quant','--friedman-sieve','--friedman-mode','policy','--kste-tau-A',("{0:F4}" -f $tau),'--kste-alpha',("{0:F4}" -f $alpha),'test_corpus.txt')
        $p = Start-Process -FilePath ..\build-cuda\bin\sp-engine.exe -ArgumentList $args -RedirectStandardOutput ".\sweep_${nm}.out" -RedirectStandardError ".\sweep_${nm}.err" -NoNewWindow -Environment @{ SP_ENGINE_NATIVE='1'; SHANNON_PRIME_VERBOSE='0' } -PassThru
        Wait-Process -Id $p.Id -Timeout 120 -ErrorAction SilentlyContinue
        $out = Get-Content ".\sweep_${nm}.out" -ErrorAction SilentlyContinue
        $ppl = 0.0; $evict = 0.0
        foreach ($line in $out) {
            if ($line -match 'perplexity\s*=\s*([0-9.]+)') { $ppl = [double]$matches[1] }
            if ($line -match 'sieve evictions\s*=\s*\d+\s*/\s*\d+\s*\(([0-9.]+)') { $evict = [double]$matches[1] }
        }
        $delta = if ($ppl -gt 0) { ($ppl - $baseline_270) / $baseline_270 * 100 } else { 0 }
        $line = "tau=$tau alpha=$alpha PPL=$ppl delta=$($delta.ToString('F2'))% evict=$($evict.ToString('F2'))%"
        Add-Content .\sweep_progress.txt $line
        $ledger += [pscustomobject]@{ tau=$tau; alpha=$alpha; ppl=$ppl; delta_pct=$delta; eviction_pct=$evict }
    }
}
$ledger | ConvertTo-Json | Set-Content .\sweep_270m_ledger.json
Add-Content .\sweep_progress.txt "DONE"
