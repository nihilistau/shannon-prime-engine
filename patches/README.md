# Vendor patches

Patches applied on top of pinned `vendor/` submodules. Each patch is a
`git diff`-format file that applies with `git apply` (strict path match) or
`patch -p1` (tolerant) from the root of the vendored tree.

These patches exist because the changes aren't upstream-ready yet — either
because they encode a Shannon-Prime-specific design decision, or because the
upstream review cycle is longer than our calibration schedule. When a patch
lands upstream, it should be removed from this directory.

## Applying patches

The engine build doesn't apply patches automatically — the working tree is
expected to already contain the patched files. The usual flow is:

1. `git submodule update --init --recursive` bootstraps the pinned submodule.
2. Run `scripts/apply-patches.ps1` (Windows) or `scripts/apply-patches.sh`
   (Linux/WSL) — each script walks this directory and applies every `*.patch`
   file against the relevant submodule, failing fast on conflicts.
3. Build normally. The submodule working tree stays dirty; commit to a
   fork or carry the patch out-of-tree.

If the scripts don't exist on your checkout yet, apply by hand:

```bash
cd vendor/ggml
git apply --check ../../patches/ggml-cuda-getrows-kquant.patch   # dry-run
git apply          ../../patches/ggml-cuda-getrows-kquant.patch
```

## Current patches

### `ggml-cuda-getrows-kquant.patch`

Adds K-quant / IQ* source-type dispatch to
`vendor/ggml/src/ggml-cuda/getrows.cu`. Upstream's CUDA `get_rows` only
handles F16/F32/I32/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — a K-quant
`token_embd.weight` (e.g. Q6_K on `gemma-3-12b-it-Q3_K_L.gguf`, Q4_K on
`phi-4-Q4_K_M.gguf`) aborts at the first token lookup.

The patch reuses the per-row K-quant dequantizers that already exist in
`vendor/ggml/src/ggml-cuda/convert.cu` (`dequantize_row_q*_K_cuda` and
friends, surfaced via `ggml_get_to_fp32_cuda` / `ggml_get_to_fp16_cuda`).
It stages the `src1` index array to host, then issues one full-row dequant
launch per requested output row.

Cost: one `cudaMemcpyAsync` device→host of the index array per getrows
call, plus `ne10 * ne11 * ne12` kernel launches. For decode that is a
single launch; for prefill with `ctx=2048` it is 2k tiny launches — still
negligible compared to the matmul cost of the decode itself.

Blocker this unblocks: **OI2b** (gemma-3-12b calibration), plus any phi-4
or phi-3.1-mini-128k run that uses a K-quant embed table.
