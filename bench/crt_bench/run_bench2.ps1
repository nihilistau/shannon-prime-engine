$exe = "D:\F\shannon-prime-repos\shannon-prime-engine\build\bin\sp-engine.exe"
$outdir = "D:\F\shannon-prime-repos\shannon-prime-engine\bench\crt_bench"
$corpus = "D:\F\shannon-prime-repos\shannon-prime-engine\bench\test_corpus.txt"
$env:SHANNON_PRIME_VERBOSE = "1"
$env:SP_ENGINE_BACKEND = "cuda"
$prompt = "You are a helpful coding assistant. Write a Python function that computes the first N prime numbers using a sieve of Eratosthenes. Include type hints and a docstring explaining the algorithm step by step."

# ============ 9B MODEL ============
$model9b = "D:\Files\Models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"

# 9B: CPU
"[$(Get-Date)] 9B CPU starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cpu"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model9b --sqfree --ctx 512 --n-predict 128 $prompt 2>&1 | Out-File "$outdir\9b_cpu.txt" -Encoding UTF8
$sw.Stop()
"9B CPU: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\times2.txt" -Append
"[$(Get-Date)] 9B CPU done: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\progress2.txt" -Append

# 9B: dGPU
"[$(Get-Date)] 9B dGPU starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model9b --sqfree --ctx 512 --n-predict 128 --n-gpus 1 $prompt 2>&1 | Out-File "$outdir\9b_dgpu.txt" -Encoding UTF8
$sw.Stop()
"9B dGPU: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\times2.txt" -Append
"[$(Get-Date)] 9B dGPU done: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\progress2.txt" -Append

# 9B: CRT
"[$(Get-Date)] 9B CRT starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model9b --sqfree --ctx 512 --n-predict 128 --crt-split $prompt 2>&1 | Out-File "$outdir\9b_crt.txt" -Encoding UTF8
$sw.Stop()
"9B CRT: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\times2.txt" -Append
"[$(Get-Date)] 9B CRT done: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\progress2.txt" -Append

# 9B: PPL
"[$(Get-Date)] 9B PPL runs starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cpu"
& $exe cache_ppl --model $model9b --hierarchical --ctx 256 --chunks 2 $corpus 2>&1 | Out-File "$outdir\9b_ppl_cpu.txt" -Encoding UTF8
$env:SP_ENGINE_BACKEND = "cuda"
& $exe cache_ppl --model $model9b --hierarchical --ctx 256 --chunks 2 --crt-split $corpus 2>&1 | Out-File "$outdir\9b_ppl_crt.txt" -Encoding UTF8
"[$(Get-Date)] 9B PPL done" | Out-File "$outdir\progress2.txt" -Append

# ============ 35B MoE MODEL ============
$model35b = "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf"

# 35B: CRT (this is the money shot — 19.7 GB model, doesn't fit in 12 GB RTX alone)
"[$(Get-Date)] 35B CRT starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model35b --sqfree --ctx 256 --n-predict 64 --crt-split $prompt 2>&1 | Out-File "$outdir\35b_crt.txt" -Encoding UTF8
$sw.Stop()
"35B CRT: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\times2.txt" -Append
"[$(Get-Date)] 35B CRT done: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\progress2.txt" -Append

# 35B: CPU baseline
"[$(Get-Date)] 35B CPU starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cpu"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model35b --sqfree --ctx 256 --n-predict 64 $prompt 2>&1 | Out-File "$outdir\35b_cpu.txt" -Encoding UTF8
$sw.Stop()
"35B CPU: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\times2.txt" -Append
"[$(Get-Date)] 35B CPU done: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\progress2.txt" -Append

# 35B: PPL
"[$(Get-Date)] 35B PPL runs starting..." | Out-File "$outdir\progress2.txt" -Append
$env:SP_ENGINE_BACKEND = "cpu"
& $exe cache_ppl --model $model35b --hierarchical --ctx 256 --chunks 2 $corpus 2>&1 | Out-File "$outdir\35b_ppl_cpu.txt" -Encoding UTF8
$env:SP_ENGINE_BACKEND = "cuda"
& $exe cache_ppl --model $model35b --hierarchical --ctx 256 --chunks 2 --crt-split $corpus 2>&1 | Out-File "$outdir\35b_ppl_crt.txt" -Encoding UTF8
"[$(Get-Date)] 35B PPL done" | Out-File "$outdir\progress2.txt" -Append

"[$(Get-Date)] ALL DONE" | Out-File "$outdir\progress2.txt" -Append
