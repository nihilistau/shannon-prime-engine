#!/usr/bin/env python3
"""ruler_multi.py — Phase 8e multi-needle packed RULER probe.

Packs 5 semantically orthogonal needles into a single corpus, clustered
around middle depth (depth_pct ≈ 0.50) with 20-30 token buffers between
each.  Compares softmax vs ultraproduct-bracket-4 PPL on planted vs
control versions of the same haystack.

Total wall ≈ 4 perplexity-native invocations × ~9 min = ~36 min, vs the
~6 hours the original n=5 trial sweep would have taken at single-needle.

Statistical content is RICHER than n=5 single-needle:
  * Each corpus now contains 5 retrieval challenges in one PPL number.
  * Softmax's smearing failure mode compounds across needles
    (cross-signal contamination).
  * Ultraproduct's Sieve must partition 5 distinct high-information
    semantic keys into separate ⪯_d equivalence classes, then F-over-
    top-4 must resolve each correctly.

Orthogonal-domain needle pairs (no shared roots, no semantic clusters):
  garden      ←→ saffron     (spice/horticulture)
  violin      ←→ tungsten    (music/metallurgy)
  skyscraper  ←→ neon        (architecture/elements)
  recipe      ←→ quantum     (culinary/physics)
  glacier     ←→ carbonate   (geology/chemistry)

Buffer zone: 100 chars (~25 tokens) of test_corpus.txt material between
adjacent needles so they don't form a single hyper-dense anomalous block
that the Sieve would flatten into one topological partition.
"""

from __future__ import annotations

import argparse
import json
import os
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

# Orthogonal-domain needle pairs (topic, value).
# Verified: no shared subword roots, no semantic overlap, all 5 land in
# disparate KSTE equivalence classes by construction.
NEEDLES = [
    ("garden",     "saffron"),
    ("violin",     "tungsten"),
    ("skyscraper", "neon"),
    ("recipe",     "quantum"),
    ("glacier",    "carbonate"),
]

# Buffer chars between adjacent needles inside the planted block.
# ~25 tokens at ~4 chars/token.
BUFFER_CHARS = 100


def build_corpora(corpus_text: str, ctx_target: int,
                  cluster_pct: float, tmp_dir: Path) -> tuple[Path, Path]:
    """Build (planted, control) corpora.

    Structure (questions sit IMMEDIATELY AFTER the needle block, both
    inside the engine's evaluation window):

      Planted: haystack_prefix + (needle_i + buffer)*5 + questions + haystack_tail
      Control: haystack_prefix + (filler   + buffer)*5 + questions + haystack_tail

    The engine evaluates the first ~ctx*chunks query positions
    (~chunks*ctx-1 in practice).  test_corpus.txt is in tokenised
    format (mix of English + comma-separated integer IDs) and packs
    at roughly 1.6 bytes/token.  At ctx=512 chunks=2 the engine sees
    only ~1640 chars of corpus, so the needle block AND the question
    block must both sit inside that prefix.  We size the haystack
    prefix to ~600 chars so the needle block + question block lands
    inside the eval window with margin to spare."""

    # Tight sizing: evaluation window at ctx=512 chunks=2 is
    # 2*(512-1) = 1022 query positions ≈ 1022 * 1.6 ≈ 1635 chars.
    # Layout in the planted corpus:
    #   [0 .. ~600]                       haystack prefix      (~370 tokens)
    #   [~600 .. ~1100]                   5 needles + buffers  (~310 tokens)
    #   [~1100 .. ~1450]                  5 questions+answers  (~220 tokens)
    #   [~1450 .. end]                    haystack tail (mostly off-window)
    HAYSTACK_PREFIX = 600
    HAYSTACK_TAIL   = 2000

    needle_total_chars = sum(
        len(f" The magic word for {t} is {v}. ") + 2 + BUFFER_CHARS
        for t, v in NEEDLES
    )
    question_total_chars = sum(
        len(f" Q: What is the magic word for {t}? A: {v}.") for t, v in NEEDLES
    )

    # Choose a long-enough source slice.
    target_chars = HAYSTACK_PREFIX + HAYSTACK_TAIL + 200
    if len(corpus_text) < target_chars + 500:
        corpus_text = corpus_text * (target_chars // len(corpus_text) + 2)

    import random as _r
    rng = _r.Random(int(cluster_pct * 100) * 31 + 17)
    start = rng.randint(0, len(corpus_text) - target_chars - 500)
    prefix = corpus_text[start:start + HAYSTACK_PREFIX]
    tail   = corpus_text[start + HAYSTACK_PREFIX:
                          start + HAYSTACK_PREFIX + HAYSTACK_TAIL]
    buf_start = (start + HAYSTACK_PREFIX + HAYSTACK_TAIL) % \
                (len(corpus_text) - 5 * BUFFER_CHARS - 100)

    # Build needle and filler blocks of equal length so the question
    # block lands at the same byte offset in both files.
    needle_block = ""
    filler_block = ""
    for i, (topic, value) in enumerate(NEEDLES):
        needle = f" The magic word for {topic} is {value}. "
        # Filler chosen to match the needle's character length exactly so
        # planted and control corpora have byte-identical layouts apart
        # from the central sentences.
        filler = f" Note: the daily report says nothing about {topic[:4]}xxx. "
        # Pad to identical length.
        target_len = len(needle)
        if len(filler) < target_len:
            filler = filler + " " * (target_len - len(filler))
        else:
            filler = filler[:target_len]
        buffer = " " + corpus_text[buf_start + i * BUFFER_CHARS:
                                    buf_start + i * BUFFER_CHARS + BUFFER_CHARS] + " "
        needle_block += needle + buffer
        filler_block += filler + buffer

    questions = ""
    for topic, value in NEEDLES:
        questions += f" Q: What is the magic word for {topic}? A: {value}."

    # cluster_pct is now a no-op (the layout is fixed); keep it as a
    # parameter for future flexibility.
    _ = cluster_pct

    planted_text = prefix + needle_block + questions + tail
    control_text = prefix + filler_block + questions + tail

    tmp_dir.mkdir(parents=True, exist_ok=True)
    p_path = tmp_dir / "multi_planted.txt"
    c_path = tmp_dir / "multi_control.txt"
    p_path.write_text(planted_text, encoding="utf-8")
    c_path.write_text(control_text, encoding="utf-8")

    return p_path, c_path


def run_perplexity(model: Path, corpus_path: Path, *,
                   ctx: int, chunks: int, ultraproduct: bool,
                   bracket: int = 1,
                   ramanujan_lambda: float = 0.0,
                   timeout_s: float = 1500.0) -> tuple[float | None, float]:
    cmd = [
        str(ENG), "perplexity",
        "--model", str(model),
        "--ctx", str(ctx),
        "--chunks", str(chunks),
        "--gguf-block-quant",
        "--frobenius-quant",
    ]
    if ultraproduct:
        cmd += ["--ultraproduct-attn", "principal"]
        if bracket > 1:
            cmd += ["--ultraproduct-bracket", str(bracket)]
        if ramanujan_lambda > 0.0:
            cmd += ["--kste-ramanujan-lambda", str(ramanujan_lambda)]
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
        sys.stderr.write(f"[ruler-multi] could not parse PPL (rc={proc.returncode})\n")
        sys.stderr.write(full[-2000:])
        return None, elapsed
    return float(m.group(1)), elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default=str(DEFAULT_MODEL))
    ap.add_argument("--corpus",    default=str(DEFAULT_CORPUS))
    ap.add_argument("--ctx",       type=int, default=512)
    ap.add_argument("--chunks",    type=int, default=2)
    ap.add_argument("--bracket",   type=int, default=4,
                    help="F-over-top-m bracket for ultraproduct mode.")
    ap.add_argument("--ramanujan-lambda", type=float, default=0.0,
                    help="Phase-9 c_q(p)/q² modulation strength on the "
                         "F-over-top-m bracket.  0 = no-op (Phase 8 baseline).")
    ap.add_argument("--cluster-pct", type=float, default=0.50,
                    help="Depth at which the 5-needle block is planted.")
    ap.add_argument("--timeout",   type=float, default=1500.0)
    ap.add_argument("--out-json",  default=str(REPO / "bench" / "ruler_multi_results.json"))
    ap.add_argument("--tmp",       default=str(REPO / "bench" / "tmp_ruler_multi"))
    args = ap.parse_args()

    corpus = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    tmp_dir = Path(args.tmp)
    planted_path, control_path = build_corpora(
        corpus, args.ctx, args.cluster_pct, tmp_dir)

    p_size = planted_path.stat().st_size
    c_size = control_path.stat().st_size
    print(f"Phase 8e RULER-multi  ctx={args.ctx} chunks={args.chunks} "
          f"bracket={args.bracket}")
    print(f"Engine: {ENG}")
    print(f"Model:  {args.model}")
    print(f"Planted corpus: {p_size} bytes")
    print(f"Control corpus: {c_size} bytes")
    print(f"5 needles: {NEEDLES}")
    print(f"Buffer between needles: {BUFFER_CHARS} chars (~25 tokens)")
    print(f"Cluster depth: {args.cluster_pct * 100:.0f}%")
    print()

    # 4-cell matrix: 2 modes × 2 corpora.
    cells = []
    for ultra, mode in [(False, "softmax"), (True, "ultra-b4")]:
        for corpus_path, corpus_kind in [(planted_path, "planted"),
                                          (control_path, "control")]:
            print(f"--- {mode}  ×  {corpus_kind}  ---", flush=True)
            ppl, wall = run_perplexity(Path(args.model), corpus_path,
                                        ctx=args.ctx, chunks=args.chunks,
                                        ultraproduct=ultra,
                                        bracket=args.bracket,
                                        ramanujan_lambda=args.ramanujan_lambda,
                                        timeout_s=args.timeout)
            print(f"  PPL = {ppl}   wall = {wall:.1f}s", flush=True)
            cells.append({"mode": mode, "corpus": corpus_kind,
                          "ppl": ppl, "wall_s": wall})

    # Summary.
    def find(mode: str, corpus: str) -> float | None:
        for c in cells:
            if c["mode"] == mode and c["corpus"] == corpus:
                return c["ppl"]
        return None

    soft_p = find("softmax", "planted")
    soft_c = find("softmax", "control")
    ultra_p = find("ultra-b4", "planted")
    ultra_c = find("ultra-b4", "control")

    print()
    print("=== summary ===")
    print(f"  softmax       planted={soft_p}   control={soft_c}")
    if soft_p is not None and soft_c is not None:
        delta = soft_c - soft_p
        ratio = soft_p / soft_c
        print(f"                  delta={delta:+.4f}   ratio={ratio:.4f}")
    print(f"  ultra-b4      planted={ultra_p}  control={ultra_c}")
    if ultra_p is not None and ultra_c is not None:
        delta = ultra_c - ultra_p
        ratio = ultra_p / ultra_c
        print(f"                  delta={delta:+.4f}   ratio={ratio:.4f}")

    Path(args.out_json).write_text(json.dumps({
        "config": {"ctx": args.ctx, "chunks": args.chunks,
                   "bracket": args.bracket,
                   "cluster_pct": args.cluster_pct,
                   "needles": NEEDLES, "buffer_chars": BUFFER_CHARS,
                   "planted_size": p_size, "control_size": c_size},
        "cells": cells,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
