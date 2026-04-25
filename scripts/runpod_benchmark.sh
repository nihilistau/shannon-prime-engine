#!/usr/bin/env bash
# =============================================================================
# Shannon-Prime RunPod A100 Benchmark Script
# =============================================================================
# Self-contained script for running PPL calibration benchmarks on RunPod
# A100 80GB instances. Detects environment, builds from source, downloads
# models + corpus, runs the full benchmark matrix, and outputs structured
# JSON results.
#
# Usage:
#   ./runpod_benchmark.sh [options] [model_spec ...]
#
# Options:
#   --dry-run         Show what would run without executing
#   --skip-build      Skip clone + build (reuse existing)
#   --skip-download   Skip model + corpus downloads
#   --ctx N           Context length (default: 2048)
#   --chunks N        Number of chunks (default: 8)
#   --results-dir D   Output directory (default: auto)
#   --help            Show this help
#
# Model spec format:
#   "repo_id:filename:short_name"
#   e.g. "Qwen/Qwen3.6-35B-A3B-GGUF:qwen3.6-35b-a3b-q6_k.gguf:qwen3moe"
#
# If no model specs given, uses the default A100 calibration set.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SCRIPT_NAME="$(basename "$0")"
SCRIPT_START=$(date +%s)
DRY_RUN=0
SKIP_BUILD=0
SKIP_DOWNLOAD=0
CTX=2048
CHUNKS=8
RESULTS_DIR=""
MODELS=()

# A100-80GB pricing (RunPod community cloud, approximate $/hr)
A100_COST_PER_HOUR=1.64

# Calibration budgets (fraction of baseline PPL)
BUDGET_SHIP=0.05
BUDGET_SQFREE=0.10
BUDGET_SQFREE_SPINOR=0.15
BUDGET_HIERARCHICAL=0.15
BUDGET_SHIP_CAUCHY2=0.05

# Default models for A100 80GB calibration runs
DEFAULT_MODELS=(
    "Qwen/Qwen3.6-35B-A3B-GGUF:qwen3.6-35b-a3b-q6_k.gguf:qwen3moe"
    "google/gemma-4-31b-it-GGUF:gemma-4-31b-it-Q8_0.gguf:gemma4"
    "meta-llama/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q8_0.gguf:llama3"
)

# Benchmark paths (names must match calibration ledger conventions)
declare -A BENCH_PATHS
BENCH_PATHS=(
    ["baseline"]=""
    ["ship"]="--cache"
    ["ship+cauchy2"]="--cache --cauchy-mode 2"
    ["sqfree"]="--cache --sqfree"
    ["sqfree+spinor"]="--cache --sqfree --spinor"
    ["hierarchical"]="--cache --hierarchical"
)

# Ordered list of path names (bash associative arrays don't preserve order)
BENCH_PATH_ORDER=(baseline ship ship+cauchy2 sqfree sqfree+spinor hierarchical)

# Budget for each path
declare -A PATH_BUDGETS
PATH_BUDGETS=(
    ["baseline"]=0
    ["ship"]=$BUDGET_SHIP
    ["ship+cauchy2"]=$BUDGET_SHIP_CAUCHY2
    ["sqfree"]=$BUDGET_SQFREE
    ["sqfree+spinor"]=$BUDGET_SQFREE_SPINOR
    ["hierarchical"]=$BUDGET_HIERARCHICAL
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*"
}

log_section() {
    echo ""
    echo "================================================================"
    log "$@"
    echo "================================================================"
}

die() {
    log "FATAL: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
detect_environment() {
    log_section "Detecting environment"

    if [[ -d /workspace ]] && [[ -f /etc/runpod/runpod.conf ]] 2>/dev/null || [[ "${RUNPOD_POD_ID:-}" != "" ]]; then
        ENV_TYPE="runpod"
        WORK_DIR="/workspace/shannon-prime-bench"
        MODELS_DIR="/workspace/models"
        NETWORK_DRIVE="/workspace"
        log "Environment: RunPod (pod=${RUNPOD_POD_ID:-unknown})"
    elif [[ -d /workspace ]]; then
        ENV_TYPE="cloud"
        WORK_DIR="/workspace/shannon-prime-bench"
        MODELS_DIR="/workspace/models"
        NETWORK_DRIVE="/workspace"
        log "Environment: Cloud (generic, /workspace available)"
    else
        ENV_TYPE="local"
        WORK_DIR="${HOME}/shannon-prime-bench"
        MODELS_DIR="${HOME}/models"
        NETWORK_DRIVE=""
        log "Environment: Local"
    fi

    if [[ -n "$RESULTS_DIR" ]]; then
        RESULTS_DIR="$RESULTS_DIR"
    else
        RESULTS_DIR="${WORK_DIR}/results/$(date '+%Y%m%d_%H%M%S')"
    fi

    ENGINE_DIR="${WORK_DIR}/shannon-prime-engine"
    ENGINE_BIN="${ENGINE_DIR}/build/bin/sp-engine"
    CORPUS_DIR="${WORK_DIR}/corpus"

    log "Work dir:    $WORK_DIR"
    log "Models dir:  $MODELS_DIR"
    log "Results dir: $RESULTS_DIR"
    log "Engine dir:  $ENGINE_DIR"
}

detect_gpu() {
    log_section "Detecting GPU"

    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found -- no CUDA GPU available"
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)

    # Auto-detect compute capability for cmake
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | xargs)
    if [[ -z "$GPU_ARCH" ]]; then
        # Fallback: parse from gpu name
        case "$GPU_NAME" in
            *A100*) GPU_ARCH="80" ;;
            *A10*)  GPU_ARCH="86" ;;
            *H100*) GPU_ARCH="90" ;;
            *L40*)  GPU_ARCH="89" ;;
            *)      GPU_ARCH="80" ; log "WARN: Unknown GPU '$GPU_NAME', defaulting to sm_80" ;;
        esac
    fi

    log "GPU:        $GPU_NAME ($GPU_COUNT device(s))"
    log "VRAM:       ${GPU_MEM_MB} MB"
    log "SM arch:    sm_${GPU_ARCH}"
    log "Driver:     $DRIVER_VER"
}

gpu_mem_snapshot() {
    local label="${1:-checkpoint}"
    if command -v nvidia-smi &>/dev/null; then
        local used free
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | xargs)
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | xargs)
        log "GPU mem [$label]: used=${used}MB free=${free}MB"
    fi
}

# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------
install_deps() {
    log_section "Installing dependencies"

    local need_install=0
    for cmd in cmake ninja git curl; do
        if ! command -v "$cmd" &>/dev/null; then
            log "Missing: $cmd"
            need_install=1
        fi
    done

    if [[ $need_install -eq 0 ]]; then
        log "All dependencies present"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] Would install: cmake ninja-build git curl"
        return 0
    fi

    if command -v apt-get &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq cmake ninja-build git curl build-essential
    elif command -v yum &>/dev/null; then
        yum install -y cmake ninja-build git curl gcc gcc-c++ make
    else
        die "No supported package manager (apt-get, yum)"
    fi

    # Verify cmake version >= 3.14
    local cmake_ver
    cmake_ver=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
    log "cmake version: $cmake_ver"
}

# ---------------------------------------------------------------------------
# Clone and build
# ---------------------------------------------------------------------------
clone_engine() {
    log_section "Cloning shannon-prime-engine"

    if [[ -d "$ENGINE_DIR/.git" ]]; then
        log "Engine repo exists, pulling latest"
        if [[ $DRY_RUN -eq 1 ]]; then
            log "[DRY-RUN] Would: cd $ENGINE_DIR && git pull && git submodule update --init --recursive"
        else
            cd "$ENGINE_DIR"
            git pull --ff-only || log "WARN: pull failed, using existing checkout"
            git submodule update --init --recursive
        fi
    else
        if [[ $DRY_RUN -eq 1 ]]; then
            log "[DRY-RUN] Would: git clone --recursive https://github.com/nihilistau/shannon-prime-engine.git $ENGINE_DIR"
        else
            mkdir -p "$(dirname "$ENGINE_DIR")"
            git clone --recursive https://github.com/nihilistau/shannon-prime-engine.git "$ENGINE_DIR"
        fi
    fi

    # Record SHAs for the calibration ledger
    if [[ -d "$ENGINE_DIR/.git" ]]; then
        cd "$ENGINE_DIR"
        ENGINE_SHA=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
        SP_CORE_SHA=""
        if [[ -d "lib/shannon-prime/.git" ]]; then
            SP_CORE_SHA=$(cd lib/shannon-prime && git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
        fi
        log "Engine SHA: $ENGINE_SHA"
        log "SP core SHA: $SP_CORE_SHA"
    else
        ENGINE_SHA="unknown"
        SP_CORE_SHA="unknown"
    fi
}

build_engine() {
    log_section "Building engine (CUDA sm_${GPU_ARCH})"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] Would: cmake + build with CUDA_ARCHITECTURES=${GPU_ARCH}"
        return 0
    fi

    cd "$ENGINE_DIR"
    mkdir -p build
    cd build

    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSP_ENGINE_WITH_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
        -DSP_ENGINE_BUILD_TESTS=OFF \
        2>&1 | tail -20

    ninja -j"$(nproc)" sp-engine 2>&1 | tail -10

    if [[ ! -x "${ENGINE_BIN}" ]]; then
        die "Build failed: ${ENGINE_BIN} not found"
    fi

    log "Build OK: ${ENGINE_BIN}"
    "${ENGINE_BIN}" version 2>/dev/null || "${ENGINE_BIN}" banner 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Corpus download
# ---------------------------------------------------------------------------
download_corpus() {
    log_section "Downloading wikitext-2-raw corpus"

    local wiki_file="${CORPUS_DIR}/wiki.test.raw"

    if [[ -f "$wiki_file" ]]; then
        log "Corpus already present: $wiki_file ($(wc -c < "$wiki_file") bytes)"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] Would download wikitext-2-raw-v1 to $wiki_file"
        return 0
    fi

    mkdir -p "$CORPUS_DIR"

    # Download from HuggingFace datasets
    local url="https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1/resolve/main/wiki.test.raw"
    local url_alt="https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"

    # Try the direct raw file first; fall back to constructing it
    if curl -fsSL -o "$wiki_file" "$url" 2>/dev/null; then
        log "Downloaded wikitext-103 test split: $(wc -c < "$wiki_file") bytes"
    else
        log "Direct download failed, trying wikitext-2 parquet conversion..."
        # Download and extract with python
        python3 -c "
import urllib.request, json, os, sys
# Simple fallback: download the parquet and extract text
try:
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet',
        '/tmp/wiki_test.parquet'
    )
    import pyarrow.parquet as pq
    table = pq.read_table('/tmp/wiki_test.parquet')
    texts = table.column('text').to_pylist()
    with open('${wiki_file}', 'w') as f:
        f.write('\n'.join(texts))
    print(f'Extracted {len(texts)} lines')
except Exception as e:
    # Last resort: generate a minimal test corpus from repeated text
    print(f'Parquet extraction failed ({e}), generating synthetic corpus', file=sys.stderr)
    with open('${wiki_file}', 'w') as f:
        text = 'The quick brown fox jumps over the lazy dog. ' * 5000
        f.write(text)
    print('Generated synthetic corpus (results will NOT match wikitext baselines)')
" 2>&1
    fi

    log "Corpus: $wiki_file ($(wc -c < "$wiki_file") bytes)"
}

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
download_model() {
    local repo_id="$1"
    local filename="$2"
    local short_name="$3"
    local target_dir="${MODELS_DIR}/${short_name}"
    local target_file="${target_dir}/${filename}"

    if [[ -f "$target_file" ]]; then
        log "Model present: $target_file ($(du -h "$target_file" | cut -f1))"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] Would download: ${repo_id}/${filename} -> ${target_file}"
        return 0
    fi

    mkdir -p "$target_dir"
    log "Downloading: ${repo_id}/${filename}"

    local hf_url="https://huggingface.co/${repo_id}/resolve/main/${filename}"

    # Use huggingface-cli if available, else curl
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo_id" "$filename" --local-dir "$target_dir"
    else
        curl -fSL \
            --retry 3 \
            --retry-delay 10 \
            -o "$target_file" \
            "$hf_url"
    fi

    if [[ ! -f "$target_file" ]]; then
        die "Download failed: $target_file"
    fi

    log "Downloaded: $target_file ($(du -h "$target_file" | cut -f1))"
}

download_all_models() {
    log_section "Downloading models"

    for spec in "${MODELS[@]}"; do
        IFS=':' read -r repo_id filename short_name <<< "$spec"
        download_model "$repo_id" "$filename" "$short_name"
    done
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
run_single_benchmark() {
    local model_file="$1"
    local short_name="$2"
    local path_name="$3"
    local path_flags="$4"
    local corpus_file="$5"
    local json_out="$6"

    local run_start
    run_start=$(date +%s)

    log "  Running: ${short_name} / ${path_name}"
    gpu_mem_snapshot "pre-${path_name}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [DRY-RUN] SP_ENGINE_BACKEND=gpu ${ENGINE_BIN} perplexity ${path_flags} --model ${model_file} --ctx ${CTX} --chunks ${CHUNKS} --model-preset auto ${corpus_file}"
        # Write a placeholder result
        cat > "$json_out" <<-ENDJSON
{
    "model": "${short_name}",
    "path": "${path_name}",
    "flags": "${path_flags}",
    "dry_run": true,
    "ppl": null
}
ENDJSON
        return 0
    fi

    local log_file="${RESULTS_DIR}/logs/${short_name}_${path_name}.log"
    mkdir -p "$(dirname "$log_file")"

    local preset_flag="--model-preset auto"
    if [[ "$path_name" == "baseline" ]]; then
        preset_flag="--model-preset off"
    fi

    # Build the command
    local cmd="SP_ENGINE_BACKEND=gpu ${ENGINE_BIN} perplexity ${path_flags} ${preset_flag} --model ${model_file} --ctx ${CTX} --chunks ${CHUNKS} ${corpus_file}"

    log "  CMD: $cmd"

    local ppl_value=""
    local exit_code=0

    # Run and capture output
    set +e
    eval "$cmd" > "$log_file" 2>&1
    exit_code=$?
    set -e

    local run_end
    run_end=$(date +%s)
    local run_secs=$((run_end - run_start))

    # Parse PPL from output (sp-engine prints "Final perplexity: X.XXXX" or similar)
    if [[ $exit_code -eq 0 ]]; then
        ppl_value=$(grep -oP 'perplexity[:\s]+\K[\d.]+' "$log_file" | tail -1 || true)
        if [[ -z "$ppl_value" ]]; then
            # Try alternate patterns
            ppl_value=$(grep -oP 'PPL[:\s=]+\K[\d.]+' "$log_file" | tail -1 || true)
        fi
        if [[ -z "$ppl_value" ]]; then
            ppl_value=$(grep -oP 'ppl\s*=\s*\K[\d.]+' "$log_file" | tail -1 || true)
        fi
    fi

    # Compute cost for this run
    local cost
    cost=$(echo "scale=4; $run_secs * $A100_COST_PER_HOUR / 3600" | bc 2>/dev/null || echo "0")

    # Write JSON result
    cat > "$json_out" <<-ENDJSON
{
    "model": "${short_name}",
    "model_file": "$(basename "$model_file")",
    "path": "${path_name}",
    "flags": "${path_flags}",
    "preset_flag": "${preset_flag}",
    "ctx": ${CTX},
    "chunks": ${CHUNKS},
    "ppl": ${ppl_value:-null},
    "exit_code": ${exit_code},
    "runtime_secs": ${run_secs},
    "estimated_cost_usd": ${cost},
    "gpu": "${GPU_NAME}",
    "gpu_arch": "sm_${GPU_ARCH}",
    "engine_sha": "${ENGINE_SHA}",
    "sp_core_sha": "${SP_CORE_SHA}",
    "timestamp": "$(date -Iseconds)",
    "log_file": "${log_file}"
}
ENDJSON

    if [[ $exit_code -ne 0 ]]; then
        log "  FAILED (exit=$exit_code) after ${run_secs}s -- see $log_file"
    elif [[ -z "$ppl_value" ]]; then
        log "  WARN: completed but PPL not parsed -- see $log_file"
    else
        log "  PPL=${ppl_value} (${run_secs}s, ~\$${cost})"
    fi

    gpu_mem_snapshot "post-${path_name}"
}

run_model_benchmarks() {
    local model_spec="$1"
    IFS=':' read -r repo_id filename short_name <<< "$model_spec"
    local model_file="${MODELS_DIR}/${short_name}/${filename}"
    local corpus_file="${CORPUS_DIR}/wiki.test.raw"
    local model_results_dir="${RESULTS_DIR}/${short_name}"

    log_section "Benchmarking: ${short_name} (${filename})"

    if [[ ! -f "$model_file" ]] && [[ $DRY_RUN -eq 0 ]]; then
        log "SKIP: model file not found: $model_file"
        return 0
    fi

    mkdir -p "$model_results_dir"

    local result_files=()

    for path_name in "${BENCH_PATH_ORDER[@]}"; do
        local path_flags="${BENCH_PATHS[$path_name]}"
        local json_out="${model_results_dir}/${path_name}.json"

        run_single_benchmark "$model_file" "$short_name" "$path_name" "$path_flags" "$corpus_file" "$json_out"
        result_files+=("$json_out")
    done

    # Combine into per-model summary
    combine_model_results "$short_name" "$model_results_dir" "${result_files[@]}"
}

combine_model_results() {
    local short_name="$1"
    local model_dir="$2"
    shift 2
    local result_files=("$@")
    local combined="${model_dir}/combined.json"

    log "  Combining results -> ${combined}"

    # Build combined JSON with python (handles null values cleanly)
    python3 -c "
import json, sys, os

results = []
for f in sys.argv[1:]:
    if os.path.exists(f):
        with open(f) as fh:
            results.append(json.load(fh))

# Find baseline PPL
baseline_ppl = None
for r in results:
    if r['path'] == 'baseline' and r.get('ppl') is not None:
        baseline_ppl = r['ppl']
        break

# Add calibration verdicts
budgets = {
    'baseline': 0,
    'ship': ${BUDGET_SHIP},
    'ship+cauchy2': ${BUDGET_SHIP_CAUCHY2},
    'sqfree': ${BUDGET_SQFREE},
    'sqfree+spinor': ${BUDGET_SQFREE_SPINOR},
    'hierarchical': ${BUDGET_HIERARCHICAL},
}

for r in results:
    path = r['path']
    ppl = r.get('ppl')
    r['baseline_ppl'] = baseline_ppl

    if path == 'baseline' or ppl is None or baseline_ppl is None:
        r['drift'] = None
        r['drift_pct'] = None
        r['budget_pct'] = None
        r['verdict'] = 'N/A' if path == 'baseline' else 'ERROR'
    else:
        drift = ppl - baseline_ppl
        drift_pct = drift / baseline_ppl
        budget = budgets.get(path, 0.05)
        r['drift'] = round(drift, 4)
        r['drift_pct'] = round(drift_pct * 100, 2)
        r['budget_pct'] = round(budget * 100, 1)
        r['verdict'] = 'PASS' if drift_pct <= budget else 'FAIL'

combined = {
    'model': '${short_name}',
    'date': '$(date +%Y-%m-%d)',
    'gpu': '${GPU_NAME}',
    'engine_sha': '${ENGINE_SHA}',
    'sp_core_sha': '${SP_CORE_SHA}',
    'ctx': ${CTX},
    'chunks': ${CHUNKS},
    'baseline_ppl': baseline_ppl,
    'paths': results,
    'total_runtime_secs': sum(r.get('runtime_secs', 0) for r in results),
    'total_cost_usd': round(sum(float(r.get('estimated_cost_usd', 0)) for r in results), 4),
}

with open('${combined}', 'w') as f:
    json.dump(combined, f, indent=2)
print(json.dumps(combined, indent=2))
" "${result_files[@]}" 2>&1
}

# ---------------------------------------------------------------------------
# Summary and verdicts
# ---------------------------------------------------------------------------
print_verdicts() {
    log_section "Calibration verdicts"

    python3 -c "
import json, glob, os, sys

results_dir = '${RESULTS_DIR}'
combined_files = sorted(glob.glob(os.path.join(results_dir, '*/combined.json')))

if not combined_files:
    print('No results found.')
    sys.exit(0)

all_results = []
total_cost = 0
total_time = 0

print()
print('=' * 100)
print(f'{\"Model\":<20} {\"Path\":<20} {\"Baseline\":>10} {\"Candidate\":>10} {\"Drift\":>8} {\"Drift%\":>8} {\"Budget\":>8} {\"Verdict\":>10}')
print('-' * 100)

for cf in combined_files:
    with open(cf) as f:
        data = json.load(f)
    all_results.append(data)
    total_cost += data.get('total_cost_usd', 0)
    total_time += data.get('total_runtime_secs', 0)

    for p in data.get('paths', []):
        model = data['model']
        path = p['path']
        baseline = p.get('baseline_ppl')
        ppl = p.get('ppl')
        drift = p.get('drift')
        drift_pct = p.get('drift_pct')
        budget = p.get('budget_pct')
        verdict = p.get('verdict', '?')

        baseline_s = f'{baseline:.4f}' if baseline else '-'
        ppl_s = f'{ppl:.4f}' if ppl else '-'
        drift_s = f'{drift:+.4f}' if drift is not None else '-'
        dpct_s = f'{drift_pct:.2f}%' if drift_pct is not None else '-'
        budget_s = f'<={budget:.1f}%' if budget else '-'
        verdict_s = f'**{verdict}**' if verdict in ('PASS', 'FAIL') else verdict

        print(f'{model:<20} {path:<20} {baseline_s:>10} {ppl_s:>10} {drift_s:>8} {dpct_s:>8} {budget_s:>8} {verdict_s:>10}')

    print('-' * 100)

print()
print(f'Total runtime: {total_time//60}m {total_time%60}s')
print(f'Estimated cost: \${total_cost:.4f}')
print()

# Write combined summary
summary = {
    'date': '$(date -Iseconds)',
    'env': '${ENV_TYPE}',
    'gpu': '${GPU_NAME}',
    'models': all_results,
    'total_runtime_secs': total_time,
    'total_cost_usd': round(total_cost, 4),
}
summary_file = os.path.join(results_dir, 'summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary written: {summary_file}')
" 2>&1
}

# ---------------------------------------------------------------------------
# Copy to network drive
# ---------------------------------------------------------------------------
copy_to_network() {
    if [[ -z "$NETWORK_DRIVE" ]] || [[ ! -d "$NETWORK_DRIVE" ]]; then
        log "No network drive available, skipping copy"
        return 0
    fi

    local net_results="${NETWORK_DRIVE}/sp-benchmark-results/$(date '+%Y%m%d_%H%M%S')"
    log_section "Copying results to network drive: ${net_results}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] Would copy $RESULTS_DIR -> $net_results"
        return 0
    fi

    mkdir -p "$net_results"
    cp -r "$RESULTS_DIR"/* "$net_results"/
    log "Copied to: $net_results"
}

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
print_cost_summary() {
    local elapsed=$(($(date +%s) - SCRIPT_START))
    local cost
    cost=$(echo "scale=4; $elapsed * $A100_COST_PER_HOUR / 3600" | bc 2>/dev/null || echo "?")

    log_section "Run complete"
    log "Total wall time: $((elapsed / 60))m $((elapsed % 60))s"
    log "Estimated total cost: \$${cost} (at \$${A100_COST_PER_HOUR}/hr)"
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    head -30 "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)    DRY_RUN=1 ;;
            --skip-build) SKIP_BUILD=1 ;;
            --skip-download) SKIP_DOWNLOAD=1 ;;
            --ctx)        CTX="$2"; shift ;;
            --chunks)     CHUNKS="$2"; shift ;;
            --results-dir) RESULTS_DIR="$2"; shift ;;
            --help|-h)    usage ;;
            *)
                # Assume model spec: "repo:file:name"
                if [[ "$1" == *:*:* ]]; then
                    MODELS+=("$1")
                else
                    die "Unknown argument: $1 (model specs must be repo_id:filename:short_name)"
                fi
                ;;
        esac
        shift
    done

    # Use defaults if no models specified
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        MODELS=("${DEFAULT_MODELS[@]}")
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"

    log_section "Shannon-Prime RunPod Benchmark"
    log "Date:     $(date)"
    log "Dry-run:  $DRY_RUN"
    log "Models:   ${#MODELS[@]}"
    log "ctx=$CTX chunks=$CHUNKS"
    echo ""

    if [[ $DRY_RUN -eq 1 ]]; then
        log "*** DRY-RUN MODE -- no destructive operations ***"
    fi

    detect_environment
    detect_gpu
    install_deps

    if [[ $SKIP_BUILD -eq 0 ]]; then
        clone_engine
        build_engine
    else
        log "Skipping build (--skip-build)"
        # Still try to pick up SHAs
        if [[ -d "$ENGINE_DIR/.git" ]]; then
            cd "$ENGINE_DIR"
            ENGINE_SHA=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
            SP_CORE_SHA=$(cd lib/shannon-prime 2>/dev/null && git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
        fi
    fi

    if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
        download_corpus
        download_all_models
    else
        log "Skipping downloads (--skip-download)"
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR/logs"

    # Run benchmarks for each model
    for model_spec in "${MODELS[@]}"; do
        run_model_benchmarks "$model_spec"
    done

    # Print verdicts and summary
    print_verdicts

    # Copy to network drive
    copy_to_network

    # Final cost
    print_cost_summary

    log "Results: $RESULTS_DIR"
    log "Parse with: python3 scripts/runpod_parse_results.py $RESULTS_DIR"
}

main "$@"
