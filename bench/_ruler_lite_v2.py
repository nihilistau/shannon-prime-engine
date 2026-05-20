#!/usr/bin/env python3
"""ruler_lite_v2.py — Phase 8 T3.4 via the perplexity-native path.

v1 (_ruler_lite.py) tried to drive sp-engine's `run` verb, which uses
forward_native.cpp / Engine::generate — a DIFFERENT code path from the
perplexity-native verb where the Phase 7 ultraproduct dispatch lives.
The run verb also chokes on Q4_0 GGUFs (`unsupported W dtype=-1`).

v2 stays inside perplexity-native.  For each (depth, mode) cell we:

  1. Build a small corpus file:
        [haystack of test_corpus.txt slice at given depth]
        The magic word for {topic} is {value}.
        [more filler]
        Q: What is the magic word for {topic}? A: {value}
  2. Run `sp-engine perplexity` with --ctx and the matching mode flag.
     The aggregate PPL number includes the contribution of the answer
     tokens.  Lower PPL on the same corpus means the model retrieved
     the needle better.
  3. Also run a "control" corpus with the needle REMOVED but the
     question still present.  The model has no way to know the answer
     in that case; its PPL on the answer tokens reflects pure guessing.
  4. The retrieval signal is:
        delta_ppl = control_ppl - planted_ppl
     for the same mode.  Larger delta = better retrieval.  The
     comparison ultraproduct-Δppl vs softmax-Δppl is the T3.4 result.

This bypasses the run-verb brokenness AND keeps the Phase 7 dispatch
engaged (perplexity-native is the path we built it into).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"D:\F\shannon-prime-repos\shannon-prime-engine")
ENG = REPO / "build-cuda" / "bin" / "sp-engine.exe"
DEFAULT_MODEL = Path(
    r"D:\Files\Models\Mine\gemma-3-1b-it\gemma-3-1b-it-Q4_0\gemma-3-1b-it-Q4_0.gguf"
)
DEFAULT_CORPUS = REPO / "bench" / "test_corpus.txt"


NEEDLE_VOCAB = [
    ("gardens",      "saffron"),
    ("violins",      "tungsten"),
    ("hurricanes",   "petunia"),
    ("librarians",   "obsidian"),
    ("monasteries",  "marmalade"),
    ("typewriters",  "kingfisher"),
    ("astronauts",   "labyrinth"),
    ("merchants",    "vermilion"),
    ("apothecary",   "thunderhead"),
    ("cartographers","pomegranate"),
]


def build_corpora(corpus_text: str, ctx_target: int, depth_pct: float,
                  topic: str, value: str, tmp_dir: Path,
                  tag: str) -> tuple[Path, Path]:
    """Return (planted_path, control_path) — same haystack, with vs
    without the needle insertion."""
    needle = f" The magic word for {topic} is {value}. "
    question = f" Q: What is the magic word for {topic}? A: {value}"

    reserve_chars = len(question) + 40
    haystack_target = max(64, ctx_target * 4 - reserve_chars - len(needle))
    # ctx_target * 4: roughly 4 chars per token; we want at least ctx_target
    # tokens of source to slice from.

    if len(corpus_text) < haystack_target + 200:
        corpus_text = corpus_text * (haystack_target // len(corpus_text) + 2)
    start_max = len(corpus_text) - haystack_target - 50
    seed_rng = random.Random(hash((topic, value, depth_pct, tag)))
    start = seed_rng.randint(0, max(0, start_max))
    haystack = corpus_text[start:start + haystack_target]

    insert_pos = int(len(haystack) * depth_pct)
    while insert_pos < len(haystack) and not haystack[insert_pos].isspace():
        insert_pos += 1

    planted   = haystack[:insert_pos] + needle + haystack[insert_pos:] + question
    control   = haystack + question  # same length-ish, no needle

    pdir = tmp_dir / f"ruler_{tag}"
    pdir.mkdir(parents=True, exist_ok=True)
    p_path = pdir / "planted.txt"
    c_path = pdir / "control.txt"
    p_path.write_text(planted, encoding="utf-8")
    c_path.write_text(control, encoding="utf-8")
    return p_path, c_path


def run_perplexity(model: Path, corpus_path: Path, *,
                   ctx: int, ultraproduct: bool,
                   timeout_s: float = 900.0) -> tuple[float | None, float]:
    """Invoke sp-engine.exe perplexity on the corpus.  Returns
    (ppl_value or None on parse failure, wall_seconds)."""
    cmd = [
        str(ENG), "perplexity",
        "--model", str(model),
        "--ctx", str(ctx),
        "--chunks", "1",
        "--gguf-block-quant",
        "--frobenius-quant",
    ]
    if ultraproduct:
        cmd += ["--ultraproduct-attn", "principal"]
    cmd.append(str(corpus_path))

    env = dict(os.environ)
    env["SP_ENGINE_NATIVE"] = "1"
    env["SP_ENGINE_PREFILL"] = "1"

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True,
                              timeout=timeout_s, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return None, -1.0
    elapsed = time.time() - t0
    full = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"perplexity\s*=\s*([0-9.]+)", full)
    if not m:
        sys.stderr.write(f"[ruler-v2] could not parse PPL (rc={proc.returncode})\n")
        sys.stderr.write(full[-2000:])
        return None, elapsed
    return float(m.group(1)), elapsed


DEPTH_MAP = {"shallow": 0.10, "middle": 0.50, "deep": 0.90}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=str(DEFAULT_MODEL))
    ap.add_argument("--corpus",  default=str(DEFAULT_CORPUS))
    ap.add_argument("--ctx",     type=int, default=512)
    ap.add_argument("--depths",  default="middle",
                    help="comma-separated subset of {shallow,middle,deep}")
    ap.add_argument("--trials",  type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out-json", default=str(REPO / "bench" / "ruler_lite_v2_results.json"))
    ap.add_argument("--tmp", default=str(REPO / "bench" / "tmp_ruler_v2"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    tmp_dir = Path(args.tmp); tmp_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    depths = [d.strip() for d in args.depths.split(",") if d.strip()]
    for d in depths:
        if d not in DEPTH_MAP:
            sys.stderr.write(f"unknown depth {d}\n"); return 2

    results = []
    print(f"Phase 8 RULER-lite v2 — ctx={args.ctx} depths={depths} trials={args.trials}")
    print(f"Engine: {ENG}\nModel:  {args.model}\n")

    for depth in depths:
        depth_pct = DEPTH_MAP[depth]
        for trial in range(args.trials):
            topic, value = rng.choice(NEEDLE_VOCAB)
            tag = f"d{depth}_t{trial}"
            print(f"--- depth={depth} trial={trial} needle=({topic!r},{value!r}) ---")
            planted_path, control_path = build_corpora(
                corpus, args.ctx, depth_pct, topic, value, tmp_dir, tag)

            cell_results = {"depth": depth, "trial": trial,
                            "topic": topic, "value": value,
                            "ctx": args.ctx, "modes": {}}
            for ultra in [False, True]:
                mode = "ultra" if ultra else "softmax"
                p_ppl, p_wall = run_perplexity(Path(args.model), planted_path,
                                                ctx=args.ctx, ultraproduct=ultra,
                                                timeout_s=args.timeout)
                c_ppl, c_wall = run_perplexity(Path(args.model), control_path,
                                                ctx=args.ctx, ultraproduct=ultra,
                                                timeout_s=args.timeout)
                # Retrieval signal: lower PPL with the needle present = the
                # model leveraged it.  delta = control - planted.  Positive
                # delta = retrieval helps.
                delta = (c_ppl - p_ppl) if (p_ppl is not None and c_ppl is not None) else None
                print(f"  [{mode:8s}] planted={p_ppl}  control={c_ppl}  "
                      f"delta={delta}  wall=({p_wall:.0f}+{c_wall:.0f})s")
                cell_results["modes"][mode] = {
                    "planted_ppl": p_ppl, "control_ppl": c_ppl,
                    "delta": delta,
                    "planted_wall_s": p_wall, "control_wall_s": c_wall,
                }
            results.append(cell_results)

    Path(args.out_json).write_text(
        json.dumps({"ctx": args.ctx, "results": results}, indent=2),
        encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
