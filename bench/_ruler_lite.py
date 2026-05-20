#!/usr/bin/env python3
"""ruler_lite.py — Phase 8 T3.4 needle-in-haystack probe.

Compares softmax-attention vs ultraproduct-attn=principal on the cleanest
retrieval test the framework can be evaluated on: insert a unique
key/value pair at a known depth inside a haystack of filler text, then
ask the model to recall the value.

Per Paper III §8 prediction P3, ultraproduct attention should
*over-perform* softmax at long context because hard Top-1 attention does
not smear probability mass across the irrelevant filler.  At short
context softmax wins (Phase 7 smoke result: PPL 8 vs 2491 at ctx=128).
The interesting question is where the crossover sits.

Pipeline:
  1. Tokenize a haystack of `ctx_target - reserve` tokens from
     test_corpus.txt.
  2. Insert a planted needle "The magic word for {topic} is {value}."
     at a random depth (configurable: shallow / middle / deep).
  3. Append the question "Q: What is the magic word for {topic}? A:"
     and run the engine with --n-predict 16 in greedy mode.
  4. Score exact-match recall: did the engine's first generated tokens
     contain `value`?

The harness runs the same prompt twice — once with default softmax,
once with --ultraproduct-attn principal — and reports pass/fail for
each.  No PPL involved; this is a binary retrieval signal.

Usage:
    python _ruler_lite.py --ctx 512 --depths shallow,middle,deep \
        --trials 5

Defaults assume Gemma3-1B Q4_0 at D:\\Files\\Models\\Mine\\... .  Override
with --model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
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


# ---------- Needle generator ------------------------------------------------

# Memorable noun-adjective combos.  The "value" is a strong content word
# the model rarely produces by chance, the "topic" is a benign filler tag.
NEEDLE_VOCAB = [
    ("gardens",     "saffron"),
    ("violins",     "tungsten"),
    ("hurricanes",  "petunia"),
    ("librarians",  "obsidian"),
    ("monasteries", "marmalade"),
    ("typewriters", "kingfisher"),
    ("astronauts",  "labyrinth"),
    ("merchants",   "vermilion"),
    ("apothecary",  "thunderhead"),
    ("cartographers","pomegranate"),
]


def make_needle(rng: random.Random) -> tuple[str, str]:
    topic, value = rng.choice(NEEDLE_VOCAB)
    return topic, value


def build_prompt(corpus_text: str, ctx_target: int, depth_pct: float,
                 topic: str, value: str) -> str:
    """Build a prompt of approximately ctx_target characters (rough proxy
    for tokens; we don't have a Python tokenizer here, so we lean on
    character count + later trust the engine's tokenizer to truncate
    safely).  The planted needle sits at depth_pct of the haystack."""
    needle = f" The magic word for {topic} is {value}. "
    question = f" Q: What is the magic word for {topic}? A:"

    # Reserve space at the end for the question.
    reserve = len(question) + 40
    haystack_target = max(64, ctx_target - reserve - len(needle))

    # Sample a contiguous slice of the corpus.
    if len(corpus_text) < haystack_target + 100:
        # Repeat corpus until long enough.
        corpus_text = corpus_text * (haystack_target // len(corpus_text) + 2)
    start_max = len(corpus_text) - haystack_target - 10
    start = random.Random(hash((topic, value, depth_pct))).randint(0, max(0, start_max))
    haystack = corpus_text[start:start + haystack_target]

    insert_pos = int(len(haystack) * depth_pct)
    # Snap to whitespace to avoid splitting words.
    while insert_pos < len(haystack) and not haystack[insert_pos].isspace():
        insert_pos += 1
    haystack_with_needle = haystack[:insert_pos] + needle + haystack[insert_pos:]
    return haystack_with_needle + question


# ---------- Engine invocation ----------------------------------------------

def run_engine(model: Path, prompt: str, *, ctx: int, n_predict: int = 16,
               ultraproduct: bool, timeout_s: float = 600.0) -> tuple[str, float]:
    """Invoke sp-engine.exe run with the given prompt and return
    (generated_text, wall_seconds).  On error returns ("", -1.0)."""
    cmd = [
        str(ENG), "run",
        "--model", str(model),
        "--ctx", str(ctx),
        "--n-predict", str(n_predict),
        "--gguf-block-quant",
        "--frobenius-quant",
    ]
    if ultraproduct:
        cmd += ["--ultraproduct-attn", "principal"]
    cmd.append(prompt)

    env = dict(os.environ)
    env["SP_ENGINE_NATIVE"] = "1"
    # NB: SP_ENGINE_PREFILL is not respected on the `run` verb (which
    # uses Engine::generate, not perplexity-native).  We accept the
    # per-token decode cost here — the haystack is fed to the cache as
    # a one-shot prompt then we generate.

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True,
                              timeout=timeout_s, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return "", -1.0
    elapsed = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(f"[ruler-lite] sp-engine exited {proc.returncode}\n")
        sys.stderr.write(proc.stderr[-2000:])
        return "", elapsed
    return proc.stdout, elapsed


# ---------- Trial driver ---------------------------------------------------

DEPTH_MAP = {"shallow": 0.10, "middle": 0.50, "deep": 0.90}


def score(generated: str, value: str) -> bool:
    """Exact-match (case-insensitive, ignoring punctuation) for `value`
    anywhere in the first ~24 generated tokens."""
    # We don't have token boundaries here; take the first 200 chars.
    head = generated[:200].lower()
    head = re.sub(r"[^a-z0-9]+", " ", head)
    return value.lower() in head.split()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8 T3.4 RULER-lite probe")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--depths", default="shallow,middle,deep",
                    help="comma-separated subset of {shallow,middle,deep}")
    ap.add_argument("--trials", type=int, default=3,
                    help="trials per depth per attention mode")
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out-json", default=str(REPO / "bench" / "ruler_lite_results.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    rng = random.Random(args.seed)

    depths = [d.strip() for d in args.depths.split(",") if d.strip()]
    for d in depths:
        if d not in DEPTH_MAP:
            sys.stderr.write(f"unknown depth '{d}'; use {list(DEPTH_MAP)}\n")
            return 2

    results = []
    print(f"Phase 8 RULER-lite — ctx={args.ctx}  depths={depths}  trials={args.trials}")
    print(f"Engine: {ENG}")
    print(f"Model:  {args.model}")
    print()

    for depth in depths:
        depth_pct = DEPTH_MAP[depth]
        for trial in range(args.trials):
            topic, value = make_needle(rng)
            prompt = build_prompt(corpus, args.ctx, depth_pct, topic, value)
            tag = f"d={depth}({depth_pct:.2f}) t={trial}  needle=({topic!r},{value!r})"
            print(f"--- {tag} ---")

            for ultraproduct, mode in [(False, "softmax"), (True, "ultra")]:
                gen, wall = run_engine(Path(args.model), prompt,
                                       ctx=args.ctx,
                                       n_predict=args.n_predict,
                                       ultraproduct=ultraproduct,
                                       timeout_s=args.timeout)
                ok = score(gen, value) if wall > 0 else False
                head = gen.strip().replace("\n", " ")[:120]
                print(f"  [{mode:8s}] hit={int(ok)}  wall={wall:6.1f}s  out={head!r}")
                results.append({
                    "depth": depth, "depth_pct": depth_pct,
                    "trial": trial, "topic": topic, "value": value,
                    "mode": mode, "ctx": args.ctx,
                    "hit": ok, "wall_s": wall,
                    "out_head": head,
                })

    # Summary by (mode, depth).
    print()
    print("=== summary ===")
    summary = {}
    for r in results:
        key = (r["mode"], r["depth"])
        summary.setdefault(key, {"hit": 0, "n": 0, "wall_sum": 0.0})
        summary[key]["hit"] += int(r["hit"])
        summary[key]["n"]   += 1
        if r["wall_s"] > 0:
            summary[key]["wall_sum"] += r["wall_s"]
    for (mode, depth), s in sorted(summary.items()):
        rate = s["hit"] / max(1, s["n"])
        avg_wall = s["wall_sum"] / max(1, s["n"])
        print(f"  mode={mode:8s}  depth={depth:7s}  recall={rate*100:5.1f}%  "
              f"({s['hit']}/{s['n']})  avg_wall={avg_wall:.1f}s")

    Path(args.out_json).write_text(json.dumps({
        "ctx": args.ctx, "trials": args.trials, "depths": depths,
        "model": args.model, "seed": args.seed,
        "results": results,
        "summary": {f"{m}|{d}": v for (m, d), v in summary.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
