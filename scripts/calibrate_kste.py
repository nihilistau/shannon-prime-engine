#!/usr/bin/env python3
"""calibrate_kste.py — Phase 4 KSTE calibration sweep.

Iterates over (tau_A, alpha, capacity) under sp-engine.exe perplexity-sp,
records baseline-vs-sieve PPL deltas, and writes a JSON ledger.

Per Paper IV §6.1 / TEST-SUITE.md §T2.3 the gate is |delta| <= 0.005
(0.5%) on Gemma3-1B at ctx=2048 over WikiText-103 validation.

Usage (Windows host):
    python scripts/calibrate_kste.py ^
        --engine D:\F\shannon-prime-repos\shannon-prime-engine\build-cuda\bin\sp-engine.exe ^
        --model  path/to/gemma3-1b.gguf ^
        --corpus path/to/wikitext103-valid.txt ^
        --ctx 2048 --chunks 4 --threads 16 ^
        --ledger docs/KSTE-CALIBRATION.md

The script expects sp-engine to support --friedman-sieve, --kste-tau-A,
--kste-alpha, --friedman-capacity, --friedman-mode={observer,policy}.
Wire those flags in the CLI before running.

Reads stdout looking for a line like
    "perplexity = 11.83 (n=4, ctx=2048)"
— adjust the regex if the engine's output format changes.

Phase 4 caveat: as of SESSION-STATE-friedman-4 the resolution probe
(test_sp_kste_resolution) found embed rate ≈ 0 on synthetic clusters,
which means *eviction rate will be near-zero on real text too* unless
the encoder is coarsened (see SESSION-STATE-friedman-4.md §Remediation).
Running this script before the encoder remediation will report PPL
delta ≈ 0 (sieve does no work) and eviction rate failing T2.2 (< 20%).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PPL_RE = re.compile(r"perplexity\s*=\s*([0-9.]+)")
EVICT_RE = re.compile(r"sieve evictions\s*=\s*\d+\s*/\s*\d+\s*\(([0-9.]+)\s*%\)",
                       re.IGNORECASE)


def run_engine(engine, model, corpus, ctx, chunks, threads, extra_args):
    cmd = [
        str(engine), "perplexity-sp",
        "--model", str(model),
        "--corpus", str(corpus),
        "--ctx", str(ctx),
        "--chunks", str(chunks),
        "--threads", str(threads),
        "--frobenius-quant", "-p", "41", "-k", "8",
        "--poly-attn", "--ntt-crt",
    ] + list(extra_args)
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    stdout = (out.stdout or "") + (out.stderr or "")
    ppl = None
    m = PPL_RE.search(stdout)
    if m:
        ppl = float(m.group(1))
    evict = None
    m2 = EVICT_RE.search(stdout)
    if m2:
        evict = float(m2.group(1)) / 100.0
    return {
        "cmd": cmd,
        "stdout_tail": stdout[-2048:],
        "returncode": out.returncode,
        "ppl": ppl,
        "eviction_rate": evict,
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser(description="KSTE Phase-4 calibration sweep")
    ap.add_argument("--engine",  required=True, help="path to sp-engine.exe")
    ap.add_argument("--model",   required=True, help="path to gemma3-1b.gguf")
    ap.add_argument("--corpus",  required=True, help="path to WikiText-103 valid txt")
    ap.add_argument("--ctx",     type=int, default=2048)
    ap.add_argument("--chunks",  type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ledger",  default="docs/KSTE-CALIBRATION.md")
    ap.add_argument("--out-json", default="docs/kste_calibration.json")
    ap.add_argument(
        "--tau-grid",   default="0.0,0.005,0.01,0.02,0.05",
        help="comma-separated tau_A values to sweep",
    )
    ap.add_argument(
        "--alpha-grid", default="0.3,0.5,0.7,0.9",
        help="comma-separated alpha values to sweep",
    )
    ap.add_argument(
        "--cap-grid",   default="1024,2048,4096",
        help="comma-separated capacity values to sweep",
    )
    args = ap.parse_args()

    taus   = [float(x) for x in args.tau_grid.split(",")]
    alphas = [float(x) for x in args.alpha_grid.split(",")]
    caps   = [int(x)   for x in args.cap_grid.split(",")]

    Path(os.path.dirname(args.out_json) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.ledger)   or ".").mkdir(parents=True, exist_ok=True)

    # 1. Baseline (sieve off).
    print("[1/2] Baseline run (sieve OFF)...", flush=True)
    base = run_engine(args.engine, args.model, args.corpus,
                      args.ctx, args.chunks, args.threads, [])
    print(f"    baseline PPL = {base['ppl']!r}  wall={base['wall_s']:.1f}s")
    if base["ppl"] is None:
        print("    FATAL: could not parse PPL from engine output:")
        print(base["stdout_tail"])
        sys.exit(1)

    # 2. Sweep with sieve enabled (POLICY mode).
    sweep = []
    n = len(taus) * len(alphas) * len(caps)
    idx = 0
    for cap in caps:
        for tau in taus:
            for alpha in alphas:
                idx += 1
                extra = [
                    "--friedman-sieve",
                    "--friedman-mode", "policy",
                    "--friedman-capacity", str(cap),
                    "--kste-tau-A",  f"{tau:.4f}",
                    "--kste-alpha",  f"{alpha:.4f}",
                ]
                print(f"[2/{n+1}] ({idx}/{n}) cap={cap} tau={tau:.4f} "
                      f"alpha={alpha:.4f} ...", flush=True)
                r = run_engine(args.engine, args.model, args.corpus,
                               args.ctx, args.chunks, args.threads, extra)
                d_abs = (r["ppl"] - base["ppl"]) if r["ppl"] is not None else None
                d_pct = (d_abs / base["ppl"]) if d_abs is not None else None
                sweep.append({
                    "cap": cap, "tau_A": tau, "alpha": alpha,
                    "ppl_sieve": r["ppl"], "ppl_baseline": base["ppl"],
                    "delta_abs": d_abs, "delta_pct": d_pct,
                    "eviction_rate": r["eviction_rate"],
                    "wall_s": r["wall_s"],
                    "passes_T2_3": (d_pct is not None and abs(d_pct) <= 0.005),
                    "passes_T2_2": (r["eviction_rate"] is not None
                                    and r["eviction_rate"] >= 0.20),
                })
                pct_str = f"{d_pct*100:+.3f}%" if d_pct is not None else "?"
                evict_str = f"{r['eviction_rate']*100:.1f}%" if r["eviction_rate"] is not None else "?"
                print(f"    PPL={r['ppl']!r}  delta={pct_str}  "
                      f"evict={evict_str}  wall={r['wall_s']:.1f}s")

    out = {
        "config": {
            "engine":  str(args.engine),
            "model":   str(args.model),
            "corpus":  str(args.corpus),
            "ctx":     args.ctx,
            "chunks":  args.chunks,
            "threads": args.threads,
            "tau_grid":   taus,
            "alpha_grid": alphas,
            "cap_grid":   caps,
        },
        "baseline": base,
        "sweep":    sweep,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_json}")

    # 3. Append a row to the ledger.
    best = None
    for row in sweep:
        if row["delta_pct"] is None: continue
        if not row["passes_T2_3"]:   continue
        if best is None or (row["eviction_rate"] or 0) > (best["eviction_rate"] or 0):
            best = row
    md = []
    md.append(f"\n## Sweep {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}\n")
    md.append(f"- Baseline PPL: **{base['ppl']:.4f}**\n")
    md.append(f"- Grid: tau {taus} x alpha {alphas} x cap {caps} = {n} runs\n")
    if best is not None:
        md.append(f"- Best T2.3-passing: cap={best['cap']}, "
                  f"tau_A={best['tau_A']:.4f}, alpha={best['alpha']:.4f} -> "
                  f"PPL {best['ppl_sieve']:.4f} (delta {best['delta_pct']*100:+.3f}%), "
                  f"eviction {(best['eviction_rate'] or 0)*100:.1f}%\n")
    else:
        md.append("- **No T2.3-passing configuration found.** Inspect "
                  "the per-config results in the JSON ledger.\n")
    with open(args.ledger, "a") as f:
        f.write("".join(md))
    print(f"Appended to {args.ledger}")


if __name__ == "__main__":
    main()
