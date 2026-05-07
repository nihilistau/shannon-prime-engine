$exe = "D:\F\shannon-prime-repos\shannon-prime-engine\build\bin\sp-engine.exe"
$model = "D:\Files\Models\Qwen\Qwen2.5-Coder-3B-Instruct-GGUF\qwen2.5-coder-3b-instruct-q4_k_m.gguf"
$corpus = "D:\F\shannon-prime-repos\shannon-prime-engine\bench\test_corpus.txt"
$outdir = "D:\F\shannon-prime-repos\shannon-prime-engine\bench\crt_bench"
$env:SHANNON_PRIME_VERBOSE = "1"

New-Item -ItemType Directory -Path $outdir -Force | Out-Null

# Config 1: CPU
"[$(Get-Date)] Starting CPU..." | Out-File "$outdir\progress.txt" -Append
$env:SP_ENGINE_BACKEND = "cpu"
& $exe cache_ppl --model $model --hierarchical --ctx 256 --chunks 2 $corpus 2>&1 | Out-File "$outdir\1_cpu.txt" -Encoding UTF8
"[$(Get-Date)] CPU done" | Out-File "$outdir\progress.txt" -Append

# Config 2: dGPU (CUDA)
"[$(Get-Date)] Starting dGPU..." | Out-File "$outdir\progress.txt" -Append
$env:SP_ENGINE_BACKEND = "cuda"
& $exe cache_ppl --model $model --hierarchical --ctx 256 --chunks 2 --n-gpus 1 $corpus 2>&1 | Out-File "$outdir\2_dgpu.txt" -Encoding UTF8
"[$(Get-Date)] dGPU done" | Out-File "$outdir\progress.txt" -Append

# Config 3: iGPU (Vulkan)
"[$(Get-Date)] Starting iGPU..." | Out-File "$outdir\progress.txt" -Append
$env:SP_ENGINE_BACKEND = "vulkan"
& $exe cache_ppl --model $model --hierarchical --ctx 256 --chunks 2 --n-gpus 1 $corpus 2>&1 | Out-File "$outdir\3_igpu.txt" -Encoding UTF8
"[$(Get-Date)] iGPU done" | Out-File "$outdir\progress.txt" -Append

# Config 4: CRT (both GPUs)
"[$(Get-Date)] Starting CRT..." | Out-File "$outdir\progress.txt" -Append
$env:SP_ENGINE_BACKEND = "cuda"
& $exe cache_ppl --model $model --hierarchical --ctx 256 --chunks 2 --crt-split $corpus 2>&1 | Out-File "$outdir\4_crt.txt" -Encoding UTF8
"[$(Get-Date)] CRT done" | Out-File "$outdir\progress.txt" -Append

"[$(Get-Date)] ALL DONE" | Out-File "$outdir\progress.txt" -Append
