# A/B oracles for sp-engine forward passes

Scripts in this folder run a reference model through HuggingFace transformers
and print results in the same format as sp-engine's CLI verbs, so the two can
be eyeballed side-by-side.

## qwen3next_ab.py — Qwen3-Next / qwen35moe GDN correctness

Cross-checks `build_block_gdn` in src/forward.cpp against the canonical
`Qwen3NextForCausalLM` reference.

### Setup

```
pip install "transformers>=4.46" accelerate torch
```

Requires a local or hub Qwen3-Next checkpoint. The public one is
`Qwen/Qwen3-Next-80B-A3B` (both `-Instruct` and `-Thinking` variants);
match whatever converts to your qwen35moe GGUF.

### Running

Reference side (HF):

```
python tests/tools/qwen3next_ab.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --prompt "Hello" \
    --dtype fp16
```

Engine side (our build):

```
./build/bin/sp-engine.exe logits \
    --model <path-to-qwen3-next.gguf> \
    Hello
```

Both print an `argmax(last) = ... token=...` line plus `top5`. If the top-1
token ids match and the top-5 sets overlap heavily, Stage 1 of Phase 3c-bis
is validated. Systematic mismatches point at a specific bug:

* **Token ids diverge entirely** — channel split order [Q|K|V] likely wrong,
  or ssm_conv history dim mis-set, or the gate sign (`-exp(ssm_a)`) flipped.
* **Same top-5 set but reordered** — likely a scale factor or numerical
  precision difference; investigate with a wider `--top-k 20`.
* **Magnitudes wildly different (>2x)** — missing/extra `ssm_norm`, or the
  gated RMSNorm multiplier applied in the wrong order.

### Tokenization parity

sp-engine's tokenizer defaults to `add_bos=true`; the Python harness mirrors
this by prepending `tokenizer.bos_token_id`. If your GGUF was built without
a BOS token, add `--no-bos` to the Python side and omit `--add-bos` on the
engine side.
