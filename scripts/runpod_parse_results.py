#!/usr/bin/env python3
"""
Shannon-Prime RunPod Benchmark Results Parser
==============================================

Reads JSON output from runpod_benchmark.sh and generates:
  1. Markdown summary table
  2. Calibration verdicts per path per model
  3. MODEL-PACK-CALIBRATION.md ledger-format rows (append-ready)

Usage:
    python3 runpod_parse_results.py <results_dir> [--output <file>] [--ledger] [--json]

Arguments:
    results_dir     Path to the results directory from runpod_benchmark.sh
    --output FILE   Write markdown to FILE (default: stdout)
    --ledger        Also emit ledger rows in MODEL-PACK-CALIBRATION.md format
    --json          Emit combined JSON to stdout instead of markdown
    --reviewer NAME Reviewer handle for ledger rows (default: KnackAU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Calibration budgets — must match runpod_benchmark.sh and MODEL-PACK.md
# ---------------------------------------------------------------------------
BUDGETS = {
    "baseline": 0.0,
    "ship": 0.05,
    "ship+cauchy2": 0.05,
    "sqfree": 0.10,
    "sqfree+spinor": 0.15,
    "hierarchical": 0.15,
}

# Map path names to the ledger's "Path" column values
LEDGER_PATH_NAMES = {
    "baseline": "baseline",
    "ship": "ship",
    "ship+cauchy2": "ship+cauchy2",
    "sqfree": "sqfree",
    "sqfree+spinor": "sqfree+spinor",
    "hierarchical": "hierarchical",
}


@dataclass
class BenchResult:
    """Single benchmark path result."""
    model: str
    model_file: str
    path: str
    flags: str
    ppl: Optional[float]
    baseline_ppl: Optional[float]
    drift: Optional[float]
    drift_pct: Optional[float]
    budget_pct: Optional[float]
    verdict: str
    runtime_secs: int
    estimated_cost_usd: float
    exit_code: int
    engine_sha: str
    sp_core_sha: str
    ctx: int
    chunks: int
    gpu: str
    timestamp: str


@dataclass
class ModelResult:
    """All results for one model."""
    model: str
    date: str
    gpu: str
    engine_sha: str
    sp_core_sha: str
    ctx: int
    chunks: int
    baseline_ppl: Optional[float]
    paths: list[BenchResult]
    total_runtime_secs: int
    total_cost_usd: float


def load_results(results_dir: str) -> list[ModelResult]:
    """Load all combined.json files from the results directory."""
    combined_files = sorted(glob(os.path.join(results_dir, "*/combined.json")))
    if not combined_files:
        # Try the directory itself
        if os.path.exists(os.path.join(results_dir, "summary.json")):
            with open(os.path.join(results_dir, "summary.json")) as f:
                summary = json.load(f)
            return [parse_model_data(m) for m in summary.get("models", [])]
        print(f"No results found in {results_dir}", file=sys.stderr)
        return []

    models = []
    for cf in combined_files:
        with open(cf) as f:
            data = json.load(f)
        models.append(parse_model_data(data))
    return models


def parse_model_data(data: dict) -> ModelResult:
    """Parse a combined.json dict into a ModelResult."""
    paths = []
    for p in data.get("paths", []):
        paths.append(BenchResult(
            model=data.get("model", "?"),
            model_file=p.get("model_file", "?"),
            path=p.get("path", "?"),
            flags=p.get("flags", ""),
            ppl=p.get("ppl"),
            baseline_ppl=p.get("baseline_ppl"),
            drift=p.get("drift"),
            drift_pct=p.get("drift_pct"),
            budget_pct=p.get("budget_pct"),
            verdict=p.get("verdict", "?"),
            runtime_secs=p.get("runtime_secs", 0),
            estimated_cost_usd=float(p.get("estimated_cost_usd", 0)),
            exit_code=p.get("exit_code", -1),
            engine_sha=data.get("engine_sha", "?"),
            sp_core_sha=data.get("sp_core_sha", "?"),
            ctx=data.get("ctx", 0),
            chunks=data.get("chunks", 0),
            gpu=data.get("gpu", "?"),
            timestamp=p.get("timestamp", ""),
        ))

    return ModelResult(
        model=data.get("model", "?"),
        date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
        gpu=data.get("gpu", "?"),
        engine_sha=data.get("engine_sha", "?"),
        sp_core_sha=data.get("sp_core_sha", "?"),
        ctx=data.get("ctx", 0),
        chunks=data.get("chunks", 0),
        baseline_ppl=data.get("baseline_ppl"),
        paths=paths,
        total_runtime_secs=data.get("total_runtime_secs", 0),
        total_cost_usd=data.get("total_cost_usd", 0),
    )


def compute_verdict(ppl: Optional[float], baseline: Optional[float],
                    path: str) -> tuple[Optional[float], Optional[float], str]:
    """Compute drift, drift%, and verdict for a single path."""
    if path == "baseline":
        return None, None, "N/A"
    if ppl is None or baseline is None or baseline == 0:
        return None, None, "ERROR"

    drift = ppl - baseline
    drift_pct = drift / baseline
    budget = BUDGETS.get(path, 0.05)

    if drift_pct <= budget:
        verdict = "PASS"
    elif drift_pct <= budget * 1.1:  # within 10% of budget boundary
        verdict = f"FAIL (edge, +{(drift_pct - budget)*100:.2f} pp over budget)"
    else:
        verdict = "FAIL"

    return drift, drift_pct, verdict


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------
def fmt_float(v: Optional[float], decimals: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.2f} %"


def fmt_drift(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:+.4f}"


def generate_markdown(models: list[ModelResult], include_ledger: bool = False,
                      reviewer: str = "KnackAU") -> str:
    """Generate full markdown report."""
    lines: list[str] = []

    lines.append("# Shannon-Prime A100 Benchmark Results")
    lines.append("")
    if models:
        m0 = models[0]
        lines.append(f"- **Date:** {m0.date}")
        lines.append(f"- **GPU:** {m0.gpu}")
        lines.append(f"- **Engine SHA:** `{m0.engine_sha}`")
        lines.append(f"- **SP Core SHA:** `{m0.sp_core_sha}`")
        lines.append(f"- **Context:** {m0.ctx} / Chunks: {m0.chunks}")
    lines.append("")

    # Summary table
    lines.append("## Results")
    lines.append("")
    lines.append("| Model | Path | Baseline PPL | Candidate PPL | Drift | Drift % | Budget | Verdict |")
    lines.append("|-------|------|-------------:|--------------:|------:|--------:|-------:|---------|")

    for model in models:
        for p in model.paths:
            # Recompute verdict to be sure
            drift, drift_pct, verdict = compute_verdict(p.ppl, model.baseline_ppl, p.path)

            budget_str = "-"
            if p.path != "baseline" and p.path in BUDGETS:
                budget_str = f"<={BUDGETS[p.path]*100:.0f} %"

            v_str = f"**{verdict}**" if verdict in ("PASS",) else verdict
            if verdict.startswith("FAIL"):
                v_str = f"**{verdict}**"

            lines.append(
                f"| {model.model} | {p.path} | {fmt_float(model.baseline_ppl)} "
                f"| {fmt_float(p.ppl)} | {fmt_drift(drift)} "
                f"| {fmt_pct(drift_pct * 100 if drift_pct is not None else None)} "
                f"| {budget_str} | {v_str} |"
            )

    lines.append("")

    # Cost summary
    total_time = sum(m.total_runtime_secs for m in models)
    total_cost = sum(m.total_cost_usd for m in models)
    lines.append("## Cost Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total runtime | {total_time // 60}m {total_time % 60}s |")
    lines.append(f"| Estimated cost | ${total_cost:.4f} |")
    lines.append(f"| Models tested | {len(models)} |")
    lines.append(f"| Benchmark paths | {sum(len(m.paths) for m in models)} |")
    lines.append("")

    # Per-model detail
    for model in models:
        lines.append(f"### {model.model}")
        lines.append("")
        lines.append(f"- Baseline PPL: {fmt_float(model.baseline_ppl)}")
        lines.append(f"- Runtime: {model.total_runtime_secs // 60}m {model.total_runtime_secs % 60}s")
        lines.append(f"- Cost: ${model.total_cost_usd:.4f}")
        lines.append("")

        for p in model.paths:
            drift, drift_pct, verdict = compute_verdict(p.ppl, model.baseline_ppl, p.path)
            status = "BASELINE" if p.path == "baseline" else verdict
            emoji_free = status  # no emojis per project conventions
            lines.append(
                f"  - **{p.path}**: PPL={fmt_float(p.ppl)} "
                f"drift={fmt_drift(drift)} "
                f"({fmt_pct(drift_pct * 100 if drift_pct is not None else None)}) "
                f"-> {emoji_free}"
            )

        lines.append("")

    # Ledger rows
    if include_ledger:
        lines.append("## Calibration Ledger Rows")
        lines.append("")
        lines.append("Append these to `docs/MODEL-PACK-CALIBRATION.md`:")
        lines.append("")
        lines.append(
            "| Date       | Preset    | Path        | Model                              "
            "| GGUF SHA256 (first 16) | SP core SHA | Engine SHA "
            "| ctx / chunks | Baseline PPL | Candidate PPL | Drift  "
            "| Drift / baseline | Budget  | Result   | Reviewer |"
        )
        lines.append(
            "|------------|-----------|-------------|------------------------------------|"
            "------------------------|-------------|------------|"
            "--------------|-------------:|--------------:|-------:"
            "|-------------:|---------|----------|----------|"
        )

        for model in models:
            for p in model.paths:
                if p.path == "baseline":
                    continue

                drift, drift_pct, verdict = compute_verdict(
                    p.ppl, model.baseline_ppl, p.path
                )

                drift_s = fmt_drift(drift)
                dpct_s = fmt_pct(drift_pct * 100 if drift_pct is not None else None)
                budget = BUDGETS.get(p.path, 0.05)
                budget_s = f"<={budget*100:.0f} %"
                ppl_s = fmt_float(p.ppl)
                base_s = fmt_float(model.baseline_ppl)
                ctx_s = f"{model.ctx} / {model.chunks}"

                lines.append(
                    f"| {model.date[:10]} | {model.model:<9} | {p.path:<11} "
                    f"| {p.model_file:<34} | (not recorded)         "
                    f"| `{model.sp_core_sha}` | `{model.engine_sha}` "
                    f"| {ctx_s:<12} | {base_s:>12} | {ppl_s:>13} | {drift_s:>6} "
                    f"| {dpct_s:>12} | {budget_s:<7} | **{verdict}** | {reviewer} |"
                )

        lines.append("")

    return "\n".join(lines)


def generate_json_summary(models: list[ModelResult]) -> str:
    """Generate combined JSON summary."""
    output = {
        "generated": datetime.now().isoformat(),
        "models": [],
    }

    for model in models:
        m = {
            "model": model.model,
            "date": model.date,
            "gpu": model.gpu,
            "engine_sha": model.engine_sha,
            "sp_core_sha": model.sp_core_sha,
            "ctx": model.ctx,
            "chunks": model.chunks,
            "baseline_ppl": model.baseline_ppl,
            "total_runtime_secs": model.total_runtime_secs,
            "total_cost_usd": model.total_cost_usd,
            "paths": [],
        }

        for p in model.paths:
            drift, drift_pct, verdict = compute_verdict(
                p.ppl, model.baseline_ppl, p.path
            )
            m["paths"].append({
                "path": p.path,
                "flags": p.flags,
                "ppl": p.ppl,
                "drift": drift,
                "drift_pct": round(drift_pct * 100, 2) if drift_pct is not None else None,
                "budget_pct": BUDGETS.get(p.path, 0.05) * 100,
                "verdict": verdict,
                "runtime_secs": p.runtime_secs,
                "cost_usd": p.estimated_cost_usd,
            })

        output["models"].append(m)

    total_time = sum(m.total_runtime_secs for m in models)
    total_cost = sum(m.total_cost_usd for m in models)
    output["total_runtime_secs"] = total_time
    output["total_cost_usd"] = round(total_cost, 4)

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Parse Shannon-Prime RunPod benchmark results"
    )
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--output", "-o", help="Write markdown to file (default: stdout)")
    parser.add_argument("--ledger", action="store_true",
                        help="Include MODEL-PACK-CALIBRATION.md ledger rows")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of markdown")
    parser.add_argument("--reviewer", default="KnackAU",
                        help="Reviewer handle for ledger rows (default: KnackAU)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    models = load_results(args.results_dir)
    if not models:
        print("No results to parse.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = generate_json_summary(models)
    else:
        output = generate_markdown(models, include_ledger=args.ledger,
                                   reviewer=args.reviewer)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
