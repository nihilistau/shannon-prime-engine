# Phase 5 handoff — what's working, what's the bug, what to do next

## What's actually working (validated empirically)

`shannon-prime-engine@7814b24` on `feat/native-forward`:

- `sp-engine qnn_bin_bench --n-chunks 3 split1..4` — runs all 4 Qwen3-4B
  w4a16 V69 .bins in the load-1+2/exec/destroy/load-3+4/exec/destroy
  swap pattern. **104 t/s steady-state**, prefill bench, on real S22U.
  This is the speed-of-light number for these specific .bins.

- `sp-engine qnn_bin_schema split[.bin]+` — dumps full I/O tensor map
  for each split: input_ids[1,128] int32 → embedding fp16; layers
  0-11/12-23/24-35 each take residual+12 KV pairs + attention_mask +
  position_ids_cos/sin; final split outputs logits[1,128,151936] fp16.

Empirical result that contradicts the architecture doc: **PHASE_2_4_ARCHITECTURE.md
section "The corrected architecture" says the 12-layer-per-split
shape can't compose past split 3+4 on V69 due to a 1.5 GB working-
memory ceiling. In our clean qnn_bin_driver run from sp-engine,
the chain composes and runs stable at 104 t/s.** The doc's failure
was likely measured with llama.cpp / ggml / thread-pool state
polluting HTP working memory; the standalone driver doesn't have
that overhead, so the chain fits.

## Update 2026-05-02 — root cause identified: the .bins are broken

After landing per-split residual instrumentation in `qnn_bin_driver.cpp`
and rerunning all three diagnostic env-gates plus a residual-override
sweep, the actual bug is upstream of every wiring choice we considered:

**Split 1 (just `input_ids → embedding` Gather) produces saturated +
NaN output in BOTH the bench and the run paths**, with identical
statistics regardless of token id, load pattern, or any other input:

```
[qnn_bin] split 1 residual: min=-0.0158 max=6.54e+04 abs_mean=1.35e+04
                            inf=0 nan=38528 n=327680
```

- max ≈ 65504 = fp16 saturation
- abs_mean ≈ 13500 (real embeddings have abs_mean ~0.05)
- ~12% of values are NaN
- Confirmed identical in `qnn_bin_bench` (load 1+2 swap pattern, the
  validated 104 t/s path) and `qnn_bin_run` (one-at-a-time pattern)
- Confirmed identical for tokens 0, 100, 1000, 10000, 151643
- 0.10% of output bytes never overwritten by exec (small, irrelevant)

**Split 2's output is invariant to its residual_in.** A residual-override
sweep filled residual_in with uniform fp16 (0.0, 0.5, 1.0, 100.0):

```
RES=0.0   → split2 abs_mean=0.596
RES=0.5   → split2 abs_mean=0.597
RES=1.0   → split2 abs_mean=0.597
RES=100.0 → split2 abs_mean=0.597
```

Split 3, run identically, DOES respond to its input (output scales
linearly with residual). So the wiring for residual handoff IS
correct end-to-end (sp_qnn binds clientBuf to host buffer per-call,
verified at sp_qnn.c:698-712); split 2's .bin just appears to be
swallowing its residual_in tensor and producing some saturation-
denoised constant via its trailing RMSNorm.

The 104 t/s benchmark was measuring throughput only; it never
validated output correctness. The .bins were never validated end-to-end
because Genie can't load all 4 on V69 (QNN_CONTEXT_ERROR_RESOURCE_LIMIT
5005, the 1.5 GB HTP working memory ceiling), and AI Hub's profile job
only times execution.

**The fix is re-export.** The current 4-bin export from
`qwen3_4b_v69_export.py` produces .bins with broken Gather/embedding
or broken AIMET dequant scales. Likely causes:
- AIMET calibration data was wrong (the `submit_w4a16.py` pattern uses
  random standard-normal calibration, which is not representative for
  Gather op range estimation)
- The Genie wrapper hijack pattern bypassed some required AI Hub export
  step that normally pre-bakes the embedding scale

The 36-layer-per-bin re-export (NUM_LAYERS_PER_SPLIT=1) the user
suggested would also pick up a fresh embedding split. With proper
calibration data (real Qwen3 token sequences instead of random noise)
the embedding scale should be correct.

## What's broken (Phase 5.2 — `qnn_bin_run`)

Same .bins, real prompt input, real tokenizer (Qwen2.5-Coder GGUF
which shares Qwen3 tiktoken vocab): pipeline runs to completion,
produces logits, but the LM head saturates to fp16 max (`inf` →
`65504` after the mask + padding fixes from this session). Argmax
picks random vocab tokens (`SOURCE`, `►`, `Pointer`).

The hardware path is fine — same .bins, same driver, same exec
pattern. The bug is in **how we populate the input tensors**, NOT
in the .bins or the driver.

## Three diagnostic env-gated bisects to identify the bug

All three are single-variable changes inside `qnn_bin_generate_one`
(`src/qnn_bin_driver.cpp`). Add them as env-gated diagnostics, run,
look at the logits magnitude:

### 1. `SP_QNN_BIN_MASK_VALUE` — finite vs infinite mask

Current "blocked" mask value is `sp_fp32_to_fp16(-1.0e30f)` which
overflows fp16 to `-inf`. AIMET-quantized attention may interpret
`-inf` as NaN inside the int16 activation path. Try:

```cpp
const uint16_t neg_inf_fp16 = (...) ?
    0xFBFF :  // -65504, most-negative finite fp16
    sp_fp32_to_fp16(-1.0e30f);  // current, saturates to -inf
```

Most Genie configs use the finite-large-negative convention. If
this fixes it, the .bin can't tolerate -inf in its mask path.

### 2. `SP_QNN_BIN_ROPE_INTERLEAVED` — cos/sin layout

`build_position_ids` currently writes cos/sin as `[seq, freq_pair]`
flat — the standard HuggingFace convention. AI Hub's Qwen3
PositionProcessor MIGHT emit them in interleaved-pair order (pair
i = (cos[i], sin[i]) at adjacent slots) instead of split (all cos
then all sin, or [seq, freq] for cos and separately for sin). RoPE
math is identical either way; only memory layout differs.

Diagnostic: write cos/sin in both layouts under the env gate, see
which one produces sane logits. Reference: pull the
`PositionProcessor.forward()` source from the qai_hub_models
install (path noted in `PHASE_2_4_ARCHITECTURE.md`'s file inventory).

### 3. `SP_QNN_BIN_PAD_TOKEN` — input_ids padding strategy

Current: pad `input_ids[n_real_clamped..127]` with the last real
token id. Genie configs use `pad-token = 151643` for Qwen3.

Diagnostic options:
- Pad with 151643 (Qwen3 endoftext/pad).
- Pad with 151645 (Qwen3 eos).
- Pad with 0.
- ALSO restrict the attention mask to only allow q < n_real_clamped
  rows (block padded positions entirely).

Whichever combination produces non-saturated logits is the right
convention.

## How to run the bisects

```bash
# Build engine with current env gates (need to add the gates first):
cd D:\F\shannon-prime-repos\shannon-prime-engine\build-android
cmake --build . -j 4

# Push:
adb push bin\sp-engine /data/local/tmp/sp-engine/sp-engine

# Run with a specific gate flipped:
adb shell 'cd /data/local/tmp/sp22u/qnn && \
    export LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn && \
    SP_QNN_BIN_MASK_VALUE=finite \
    /data/local/tmp/sp-engine/sp-engine qnn_bin_run \
        --tokenizer /data/local/tmp/sp/qwen-target.gguf \
        --prompt "The capital of France is" \
        qwen3_4b_1_of_4.bin qwen3_4b_2_of_4.bin \
        qwen3_4b_3_of_4.bin qwen3_4b_4_of_4.bin'
```

Look for: `argmax id=N value=V` where V is in normal range
(say -50 to +50) and the decoded text is sensible (`Paris` or
similar continuation).

## Once Phase 5.2 produces coherent output

The architecture for the rest is mechanical:
- **Phase 5.3** — multi-chunk prefill (prompts > 128 tokens). Same
  pipeline, KV history rolls forward via the past_key/past_value
  out → in chain. The KV ring buffer is per-layer
  `[8, 1, 128, 1920]` for K and `[8, 1, 1920, 128]` for V; new
  positions get appended at offset `(1920 - past_len_new)`.
- **Phase 5.3 cont.** — decode (AR=1). Past-Claude's notes say a
  separate AI Hub re-export with `ar=1` is needed; or use the
  existing AR=128 .bins with a single real token + 127 padded
  (3-4× slower than dedicated AR=1 .bins but works without
  re-export).
- **Phase 5.4** — wire `qnn_bin_generate_one` into `Engine::generate`
  behind `SHANNON_PRIME_QNN_BIN_DIR=...`.
- **Phase 5.5** — SP-banded KV as the disk-persistence layer
  outside the .bins (the actual product angle: dump compressed KV
  state at session end via `KvCache::write`, reload at session
  start to preserve context across runs without paying re-prefill).

## Misc artifacts produced this session

- `shannon-prime/backends/qnn_aihub/artifacts/qwen3_4b/`:
  - `qwen3-4b-htp.json` — Genie dialog config (validated to parse;
    fails on V69 working-memory wall during context creation —
    Genie can't do the swap pattern qnn_bin_driver does)
  - `tokenizer.json`, `tokenizer_config.json`, `config.json` —
    HF Qwen3-4B copies, staged for either future Genie path with
    weight-shared/per-layer .bins, OR for in-engine tokenizer
    parsing in the qnn_bin_driver path
- `shannon-prime@8770d4b` on `fix/qnn-runtime-dim-aliasing` — fixes
  static `dim_storage` aliasing across multi-shape sessions. Real
  bug, would have hit anyone running KQ + a second matmul shape
  on the runtime-graph path.
