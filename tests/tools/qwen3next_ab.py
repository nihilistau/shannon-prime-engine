# Shannon-Prime Engine — Qwen3-Next A/B oracle (HF transformers reference)
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Runs a Qwen3-Next checkpoint through HuggingFace transformers on a fixed
# prompt and dumps top-k logits in the same shape as sp-engine.exe's
# `logits --prompt ...` verb. Lets us cross-check build_block_gdn's math
# against the canonical reference.
#
# Usage:
#   python qwen3next_ab.py --model <hf-repo-or-local-dir> --prompt "Hello"
#
# Requirements:
#   pip install "transformers>=4.46" accelerate torch --upgrade
#
# Matching notes vs sp-engine:
#   * sp-engine's tokenizer `encode(prompt, add_bos=true)` prepends a single
#     BOS token. HF's AutoTokenizer default (for this family) usually does
#     not — we explicitly prepend tokenizer.bos_token_id to keep parity.
#     If the GGUF's vocab has no bos_token, pass --no-bos to match.
#   * Precision: --dtype {fp32,fp16,bf16}; default fp16 matches our GGUF.
#   * We evaluate the whole prompt in a single forward pass and take the
#     last-row logits, same as sp-engine's forward_full + argmax(last).
#
# Output example (mirrors sp-engine's print format):
#   [qwen3next-ab] model=Qwen/Qwen3-Next-80B-A3B  prompt="Hello"
#   n_tokens=2  n_vocab=151936
#   mean=-2.8101  std=2.4012  min=-15.21  max=+16.55
#   argmax(last) = 11  logit=+16.5503  token=","
#   top5:  [11 , +16.550]  [0 ! +15.812]  [220 Ġ +14.903]  [13 . +14.120]  [25 : +13.870]

import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple


def _fail(msg: str, code: int = 2) -> None:
    print(f"[qwen3next-ab] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


@dataclass
class TopK:
    idx: int
    logit: float
    text: str


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HF-side A/B oracle for sp-engine qwen35moe forward.")
    ap.add_argument("--model", required=True,
                    help="HF repo id or local path to a Qwen3-Next checkpoint "
                         "(e.g. Qwen/Qwen3-Next-80B-A3B).")
    ap.add_argument("--prompt", default="Hello",
                    help="Prompt string to forward (default: 'Hello').")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
                    help="Torch dtype (default: fp16 — matches typical GGUF).")
    ap.add_argument("--device", default="auto",
                    help="Torch device. 'auto' picks cuda if available (default).")
    ap.add_argument("--top-k", type=int, default=5, dest="topk",
                    help="Top-K entries to print (default: 5).")
    ap.add_argument("--no-bos", action="store_true",
                    help="Do NOT prepend BOS (match sp-engine encode w/ add_bos=false).")
    ap.add_argument("--trust-remote-code", action="store_true", default=True,
                    help="Required for some Qwen3-Next HF configs.")
    args = ap.parse_args()

    # Import heavyweight deps lazily so `--help` is fast.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        _fail(f"transformers/torch import failed — install deps: {e}", code=3)

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[qwen3next-ab] loading tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)

    print(f"[qwen3next-ab] loading model: {args.model} (dtype={args.dtype}, device={device})",
          file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Print a quick arch sanity line so the user can confirm this matches their GGUF.
    cfg = model.config
    arch_line = (f"[qwen3next-ab] arch: {cfg.__class__.__name__}  "
                 f"n_layer={getattr(cfg, 'num_hidden_layers', '?')}  "
                 f"n_embd={getattr(cfg, 'hidden_size', '?')}  "
                 f"n_vocab={getattr(cfg, 'vocab_size', '?')}  "
                 f"n_experts={getattr(cfg, 'num_experts', '?')}  "
                 f"top_k={getattr(cfg, 'num_experts_per_tok', '?')}")
    print(arch_line, file=sys.stderr)

    # Tokenize. add_bos is controlled explicitly to match sp-engine.
    ids: List[int] = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not args.no_bos:
        bos = tokenizer.bos_token_id
        if bos is None:
            # Some Qwen vocabs have no BOS — fall back to match no-bos mode.
            print("[qwen3next-ab] note: tokenizer has no bos_token; skipping BOS "
                  "(equivalent to --no-bos)", file=sys.stderr)
        else:
            ids = [bos] + ids

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    torch.manual_seed(0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
    # logits: [1, n_tokens, n_vocab]; cast to fp32 for stable argmax/top-k.
    logits_t = out.logits.float()[0]  # [n_tokens, n_vocab]
    n_tokens, n_vocab = logits_t.shape
    last = logits_t[-1]

    # Stats across the whole block of logits — match sp-engine's print.
    flat = logits_t.flatten()
    mean = flat.mean().item()
    std = flat.std(unbiased=False).item()
    mn = flat.min().item()
    mx = flat.max().item()

    # Top-k of the last row.
    topk_vals, topk_idx = last.topk(args.topk)
    picks: List[TopK] = []
    for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
        tok = tokenizer.decode([i], skip_special_tokens=False)
        # Mirror sp-engine's print: single word with leading-space made visible
        # as 'Ġ' (GPT-style), as the CLI does.
        tok_disp = tok.replace(" ", "Ġ") if tok.startswith(" ") else tok
        picks.append(TopK(idx=i, logit=v, text=tok_disp))

    print(f"[qwen3next-ab] model={args.model}  prompt={args.prompt!r}")
    print(f"n_tokens={n_tokens}  n_vocab={n_vocab}")
    print(f"mean={mean:+.6f}  std={std:.6f}  min={mn:+.6f}  max={mx:+.6f}")
    best = picks[0]
    print(f'argmax(last) = {best.idx}  logit={best.logit:+.4f}  token="{best.text}"')

    parts = []
    for p in picks:
        parts.append(f"[{p.idx} {p.text} {p.logit:+.3f}]")
    print("top" + str(args.topk) + ": " + "  ".join(parts))

    return 0


if __name__ == "__main__":
    sys.exit(main())
