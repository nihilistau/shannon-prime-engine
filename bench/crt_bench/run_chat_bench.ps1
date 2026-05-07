$exe = "D:\F\shannon-prime-repos\shannon-prime-engine\build\bin\sp-engine.exe"
$model = "D:\Files\Models\Qwen\Qwen2.5-Coder-3B-Instruct-GGUF\qwen2.5-coder-3b-instruct-q4_k_m.gguf"
$outdir = "D:\F\shannon-prime-repos\shannon-prime-engine\bench\crt_bench"
$prompt = "Write a Python function that computes the Fibonacci sequence iteratively and explain each step"
$env:SHANNON_PRIME_VERBOSE = "0"

# Config 1: CPU
$env:SP_ENGINE_BACKEND = "cpu"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model --hierarchical --ctx 512 --n-predict 64 $prompt 2>&1 | Out-File "$outdir\chat_1_cpu.txt" -Encoding UTF8
$sw.Stop()
"CPU: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\chat_times.txt" -Append

# Config 2: dGPU (CUDA)
$env:SP_ENGINE_BACKEND = "cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model --hierarchical --ctx 512 --n-predict 64 --n-gpus 1 $prompt 2>&1 | Out-File "$outdir\chat_2_dgpu.txt" -Encoding UTF8
$sw.Stop()
"dGPU: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\chat_times.txt" -Append

# Config 3: CRT
$env:SP_ENGINE_BACKEND = "cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model --hierarchical --ctx 512 --n-predict 64 --crt-split $prompt 2>&1 | Out-File "$outdir\chat_4_crt.txt" -Encoding UTF8
$sw.Stop()
"CRT: $($sw.ElapsedMilliseconds) ms" | Out-File "$outdir\chat_times.txt" -Append

"DONE" | Out-File "$outdir\chat_times.txt" -Append
