Set-Location D:\F\shannon-prime-repos\shannon-prime-engine\bench
$model = 'D:\Files\Models\lmstudio-community\functiongemma-270m-it-GGUF\functiongemma-270m-it-F16.gguf'
$baseline = 10.4159
Set-Content -Path .\sweep_progress.txt -Value ("starting at " + (Get-Date).ToString())
foreach ($tau in @(0.05, 0.10, 0.20, 0.40)) {
    foreach ($alpha in @(0.3, 0.5, 0.7)) {
        Get-Process sp-engine -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        $nm = ("270_t{0}_a{1}" -f $tau, $alpha)
        $cmdArgs = @('perplexity','--model',$model,'--ctx','64','--chunks','1','--frobenius-quant','--friedman-sieve','--friedman-mode','policy','--kste-tau-A',("{0:F4}" -f $tau),'--kste-alpha',("{0:F4}" -f $alpha),'test_corpus.txt')
        $p = Start-Process -FilePath ..\build-cuda\bin\sp-engine.exe -ArgumentList $cmdArgs -RedirectStandardOutput ".\sweep_${nm}.out" -RedirectStandardError ".\sweep_${nm}.err" -NoNewWindow -Environment @{ SP_ENGINE_NATIVE='1'; SHANNON_PRIME_VERBOSE='0' } -PassThru
        Wait-Process -Id $p.Id -Timeout 240 -ErrorAction SilentlyContinue
        if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Add-Content .\sweep_progress.txt ("$nm TIMEOUT")
            continue
        }
        $out = Get-Content ".\sweep_${nm}.out" -ErrorAction SilentlyContinue
        $ppl = 0.0; $evict = 0.0
        foreach ($line in $out) {
            if ($line -match 'perplexity\s*=\s*([0-9.]+)') { $ppl = [double]$matches[1] }
            if ($line -match 'sieve evictions\s*=\s*\d+\s*/\s*\d+\s*\(([0-9.]+)') { $evict = [double]$matches[1] }
        }
        $delta = if ($ppl -gt 0) { ($ppl - $baseline) / $baseline * 100 } else { 0 }
        Add-Content .\sweep_progress.txt (("tau={0:F3} alpha={1:F2} PPL={2:F4} delta={3:+0.00;-0.00}% evict={4:F2}%") -f $tau, $alpha, $ppl, $delta, $evict)
    }
}
Get-Process sp-engine -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Add-Content .\sweep_progress.txt "DONE"
