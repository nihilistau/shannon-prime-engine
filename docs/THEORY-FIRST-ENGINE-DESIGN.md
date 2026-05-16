# Theory-First Engine Design — Shannon-Prime v2

**Status:** Phase 0 (architecture lock). Phase 1 implementation kickoff.
**Author:** A. Knack (KnackAU)
**Date:** 2026-05-16

## Goal

Rewrite `shannon-prime-engine` as a streamlined, theory-pure inference engine grounded in the CM-elliptic-curve framework over $\mathbb{Q}(\sqrt{-163})$ (Paper A, B, C, D v0.3). The new engine is feature-equivalent to llama.cpp / vLLM / ollama, but every operation is derived from the SP framework rather than bolted onto ggml.

## Why rewrite

The current engine (v0.1, May 2026) was built incrementally as SP grew out of a KV-cache trick. It pulls in:

- ggml (vendored at `vendor/ggml/`) — for tensor format, dequant, backend abstraction
- llama_weights.cpp — for GGUF loading via ggml
- forward.cpp + forward_native.cpp — split forward pass (ggml graph vs hand-written)
- Optional QNN, Beast Canyon, Hexagon FastRPC backends

This works but has two problems:
1. **The math is split** between SP's own primitives (`lib/shannon-prime/core/`) and ggml's tensor ops. The "every op is an endomorphism in $\mathcal{O}_K$" claim of Paper A §3 cannot be enforced when half the ops route through ggml.
2. **Performance** has been bottlenecked by float reductions in the engine's ship build (see Paper B §6.3: 11.6% on A100 vs predicted larger gain). The engine artifact is a consequence of ggml-shaped tensors with float intermediates where O_K integer arithmetic should live.

The streamlined engine removes ggml from the hot path. ggml stays only as the GGUF format reader (we keep `gguf_loader.cpp` as a parser, but its output is converted to SP-native tensors immediately).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ src/                                                         │
│ ├── sp_tensor.{cpp,h}        ← OK-coordinate tensor type    │
│ ├── sp_arena.{cpp,h}         ← workspace allocator           │
│ ├── sp_gguf.{cpp,h}          ← GGUF parser (read-only)       │
│ ├── sp_weights.{cpp,h}       ← model load → OK-encoded       │
│ ├── sp_forward.{cpp,h}       ← pure SP forward pass          │
│ ├── sp_attention.{cpp,h}     ← Weil pairing OR softmax       │
│ ├── sp_ffn.{cpp,h}           ← SwiGLU on OK coordinates      │
│ ├── sp_sampler.{cpp,h}       ← top-k/top-p/mirostat          │
│ ├── sp_chat_template.{cpp,h} ← ChatML, Llama, Qwen3, Phi-3   │
│ ├── sp_server.{cpp,h}        ← OpenAI-compat HTTP/SSE        │
│ ├── sp_batch.{cpp,h}         ← continuous batching scheduler │
│ ├── sp_cuda/                 ← CUDA kernels (one per op)     │
│ ├── sp_hexagon/              ← Hexagon HVX/HTP backend       │
│ └── cli/main.cpp             ← sp-engine, sp-server, sp-tools│
│                                                              │
│ lib/shannon-prime/                                           │
│ ├── core/                                                    │
│ │   ├── sp_ok_arith.{c,h}    ← O_K integer arithmetic ✓     │
│ │   ├── sp_frobenius.{c,h}   ← Frobenius / Sato-Tate ✓      │
│ │   ├── sp_weil.{c,h}        ← Miller's algorithm            │
│ │   ├── sp_hecke.{c,h}       ← Hecke eigenform basis         │
│ │   ├── sp_mobius.{c,h}      ← Möbius UFD compression        │
│ │   └── sp_crt.{c,h}         ← CRT sharding                  │
│ └── backends/                                                │
│     ├── cuda/                                                │
│     │   ├── sp_frobenius_quant.cu ✓                          │
│     │   ├── sp_weil_attention.cu                             │
│     │   ├── sp_matmul_ok.cu  ← OK-coord matmul               │
│     │   └── sp_rmsnorm.cu                                    │
│     └── hexagon/  (existing)                                  │
└──────────────────────────────────────────────────────────────┘
```

Files marked ✓ are landed in Phase 0.

## Layer datatype

Every tensor in the hot path is one of two types:

```cpp
namespace sp::engine {

// CM-encoded state: a tensor of O_K elements.
struct sp_ok_tensor {
    sp_ok_t* data;       // n_rows * n_cols elements
    int      n_rows;
    int      n_cols;
    sp_device_t device;  // CPU, CUDA:0, HEX, ...
};

// Weights live in O_K too, after the load-time conversion.
// At precision tier q, each element is reduced mod p^q via Frobenius
// (Theorem 4 of Paper A).
struct sp_ok_weights {
    sp_ok_t* data;
    int      n_rows;
    int      n_cols;
    int64_t  prime;       // p for the precision tier
    int64_t  k;           // Frobenius power: q = 16 - k
    sp_device_t device;
};

} // namespace sp::engine
```

No fp16 in tensor data. No ggml_tensor. Pure 16-byte O_K elements.

## Forward pass

```cpp
namespace sp::engine {

void sp_forward(sp_context& ctx, const sp_tokens& input, sp_logits& out) {
    // 1. Embedding lookup (Möbius square-free reconstruction)
    sp_embed(ctx, input, ctx.x);

    for (int l = 0; l < ctx.n_layers; l++) {
        // 2. RMSNorm (in O_K, scalar action)
        sp_rmsnorm_ok(ctx.x, ctx.norm_w[l], ctx.x_norm);

        // 3. Q/K/V projection (multiply by principal ideals)
        sp_proj_qkv(ctx, l, ctx.x_norm, ctx.q, ctx.k, ctx.v);

        // 4. Attention via Weil pairing OR softmax-dot-product
        sp_attention(ctx, l, ctx.q, ctx.k, ctx.v, ctx.attn_out);

        // 5. Residual (point addition on E)
        sp_add_ok(ctx.x, ctx.attn_out, ctx.x);

        // 6. FFN (SwiGLU)
        sp_rmsnorm_ok(ctx.x, ctx.ffn_norm_w[l], ctx.x_norm);
        sp_ffn(ctx, l, ctx.x_norm, ctx.ffn_out);

        // 7. Residual
        sp_add_ok(ctx.x, ctx.ffn_out, ctx.x);

        // 8. Poncelet closure check (Paper A §7)
        if (sp_poncelet_closed(ctx)) {
            // Adaptive early exit
            break;
        }
    }

    // 9. Final norm + LM head (CRT-decomposed)
    sp_rmsnorm_ok(ctx.x, ctx.final_norm_w, ctx.x_norm);
    sp_lm_head_crt(ctx, ctx.x_norm, out);
}

} // namespace sp::engine
```

Every line of the loop is an O_K endomorphism. No float intermediates.

## Multi-head attention as Siegel polarization (optional, post-Phase-1)

By Paper A §3's multi-head generalization, with $g$ attention heads the state lives on a $g$-dim CM abelian variety $A$ with polarization $\lambda: A \to \widehat{A}$. The attention map is then a single bilinear application of $\lambda$ rather than $g$ independent dot products. This is the long-term plan; Phase 1 ships with the simpler $E^n$ realization (independent heads, each an endomorphism of $E$).

## Quantization tiers

Driven by `--frobenius-quant` (Config B) and `--sato-tate-mix` (Config E):

| Tier | Flag | Prime(s) | Bits/coord | Drift |
|--|--|--|--|--|
| fp16 baseline | (none, ggml path) | — | 16 | reference |
| SP-fp8 calibration-free | `--frobenius-quant -p 41 -k 8` | 41 (split) | ~5.78 | $|a_{41}| = 1$ |
| SP-fp10 Sato–Tate | `--sato-tate-mix 2,2,41,8` | 2 inert + 41 split | ~10 | inert: 0, split: 1 |
| SP-fp4 (research) | `--frobenius-quant -p 7 -k 4` | 7 (inert) | ~2 | 0 (supersingular) |

All quantizations are *Frobenius reductions* — structure-preserving by Theorem 4. No calibration data required.

## Phases

### Phase 0 — landed in this commit
- `sp_ok_arith.{c,h}` integer arithmetic
- `sp_frobenius.{c,h}` Frobenius + prime classification
- `sp_frobenius_quant.cu` CUDA kernel
- `tests/test_sp_frobenius.cpp` C unit tests + Python golden cross-validation
- This document + `SP-SERVER-DESIGN.md`
- Test suite v0.3 (18/18 algebraic claims VERIFIED)
- Paper D v0.3 corrected to use $p_2 = 41$ split prime

### Phase 1 — minimal viable theory-first stack (~1 week)
- `sp_tensor` type + arena allocator
- `sp_gguf` parser (no ggml dependency — read GGUF as bytes, hand parse)
- `sp_weights` load → OK-encoded
- `sp_forward` pure SP forward pass (CPU path)
- `sp_attention` softmax-dot-product (Weil pairing in Phase 2)
- `sp_ffn` SwiGLU on OK
- `sp_sampler` top-k / top-p / mirostat
- Wire `--frobenius-quant` and `--sato-tate-mix` into the existing CLI as bench-time switches
- **Validation:** Phi-3 perplexity on WikiText-103 within 1% of llama.cpp baseline

### Phase 2 — server (~1 week)
- `sp_server` with full OpenAI v1 API (see `SP-SERVER-DESIGN.md`)
- SSE streaming chat completions
- `sp_chat_template` for Llama-3, Qwen3, Phi-3, ChatML, Gemma
- `sp_batch` continuous batching scheduler (paged KV cache style)
- Tokenizer wrapper (BPE + tiktoken-compatible)
- **Validation:** drop-in replacement for ollama on a Phi-3 fp8 deployment

### Phase 3 — CUDA hot path (~1 week)
- `sp_matmul_ok.cu` OK-coordinate matmul (one thread per element pair, integer arithmetic)
- `sp_rmsnorm.cu`
- `sp_softmax.cu` (or its Weil-pairing replacement)
- `sp_kv_cache_cuda` (Paper D Config B/E hooks live here)
- **Validation:** Paper D §6.3 fix — 11.6% engine artifact resolved, predicted 20%+ on A100

### Phase 4 — adjacent extensions (~2 weeks each)
- `sp_weil` attention (replace softmax-dot-product)
- `sp_hecke` Hecke-eigenform embedding (replace learned embedding table)
- `sp_rope_phi` golden-ratio RoPE base
- `sp_iwasawa_train` Mordell-Weil rank training (replaces SGD; research)

## File-by-file migration map

| Current file | Disposition |
|--|--|
| `forward.cpp` | DELETE (replaced by `sp_forward.cpp`) |
| `forward_native.cpp` | DELETE (folded into `sp_forward.cpp`) |
| `forward_native_context.cpp` | KEEP as `sp_context.cpp` (rename) |
| `llama_weights.cpp` | REPLACE with `sp_weights.cpp` (no ggml) |
| `gguf_loader.cpp` | KEEP — refactor to `sp_gguf.cpp` (no ggml dep on parse) |
| `kv_cache.cpp` | KEEP, refactor: replace fp16 with `sp_ok_t` |
| `sp_quant.cpp` | KEEP, extend with `--frobenius-quant` + `--sato-tate-mix` |
| `sp_kernels_cpu.cpp` | EVOLVE: ops become OK-coord native |
| `sp_tensor.cpp` | EVOLVE: shape stays, element type is `sp_ok_t` |
| `prime_pe.cpp` | KEEP, integrate with `sp_rope_phi` for golden-ratio variant |
| `engine.cpp` | REFACTOR: drop ggml deps, become a thin sp_forward driver |
| `http_server.cpp` | DEPRECATE in favor of `sp_server.cpp` (richer feature set) |
| `qnn_bin_driver.cpp` | KEEP (Hexagon path) |

vendor/ggml stays for the .gguf reader only (binary format). All compute moves to SP.

## Comparison: SP-engine v2 vs llama.cpp / vLLM / ollama

| Feature | llama.cpp | vLLM | ollama | **SP-engine v2** |
|--|--|--|--|--|
| GGUF support | ✓ | partial | ✓ | ✓ |
| Continuous batching | partial | ✓ | partial | ✓ (Phase 2) |
| Paged KV cache | — | ✓ | — | ✓ (SP-Frobenius, Phase 3) |
| OpenAI API | ✓ | ✓ | ✓ | ✓ (Phase 2) |
| Streaming SSE | ✓ | ✓ | ✓ | ✓ (Phase 2) |
| Tool/function calling | partial | partial | ✓ | ✓ (Phase 2) |
| Chat templates | many | many | many | many (Phase 2) |
| LoRA | ✓ | ✓ | ✓ | Phase 4 |
| Flash attention | ✓ | ✓ | ✓ | ✓ + Weil (Phase 4) |
| CUDA / Metal / Vulkan | ✓ | CUDA-only | CUDA / Metal | CUDA, Hexagon (Metal Phase 5) |
| **Calibration-free fp8** | — | — | — | ✓ (SP-Frobenius, this commit) |
| **fp4 viable** | bench-only | — | — | ✓ (Theorem 4 corollary) |
| **Provably exact KV compression** | — | — | — | ✓ (Paper B §3) |
| **Adaptive layer depth** | — | — | — | ✓ (Paper A §7, Phase 3) |

The unique SP differentiation is the bottom four rows: provably exact compression with no calibration, fp4 viability, and adaptive depth — all flowing from the framework, not bolted on.

## Open architectural questions for review

1. **Where does fp16 enter the system?** Two options: (a) input/output only — all interior is O_K; (b) optional fp16 fallback for ops that don't have an O_K implementation yet (in Phase 1, softmax). Lean toward (a) once Phase 4 Weil attention lands.

2. **Conductor of the production curve.** The CM curve $E$ over $\mathbb{Q}$ with $j = -640320^3$ has a specific Weierstrass form and conductor. We need this nailed down before BSD training (Phase 4). Reference: LMFDB curve label 26244.a1 or similar — to be confirmed via Sage cross-check.

3. **GPU memory layout for `sp_ok_t` tensors.** AoS (interleaved a, b) vs SoA (parallel a-array + b-array). AoS is simpler and matches CPU layout; SoA enables coalesced loads for stride-1 ops. Phase 3 benchmarks decide.

## What's NOT in this rewrite

- Image/audio modalities. The framework extends to Siegel modular varieties (Paper A §3) but that's a separate project.
- Multi-node distributed serving. CRT sharding (Paper A §8) is the algebraic substrate but inter-node networking is out of scope until Phase 5.
- Training. The BSD/Mordell-Weil training framing (Paper A §10) is a research direction, not a production target.

---

## Status of Phase 0 deliverables (this commit)

| Artifact | Path | Status |
|--|--|--|
| `sp_ok_arith.h` | `lib/shannon-prime/core/sp_ok_arith.h` | LANDED |
| `sp_ok_arith.c` | `lib/shannon-prime/core/sp_ok_arith.c` | LANDED |
| `sp_frobenius.h` | `lib/shannon-prime/core/sp_frobenius.h` | LANDED |
| `sp_frobenius.c` | `lib/shannon-prime/core/sp_frobenius.c` | LANDED |
| `sp_frobenius_quant.h` | `lib/shannon-prime/backends/cuda/sp_frobenius_quant.h` | LANDED |
| `sp_frobenius_quant.cu` | `lib/shannon-prime/backends/cuda/sp_frobenius_quant.cu` | LANDED |
| `test_sp_frobenius.cpp` | `tests/test_sp_frobenius.cpp` | LANDED |
| `sato_tate_golden.json` | `tests/sato_tate_golden.json` | LANDED (50 states) |
| `THEORY-FIRST-ENGINE-DESIGN.md` | `docs/THEORY-FIRST-ENGINE-DESIGN.md` | LANDED (this file) |
| `SP-SERVER-DESIGN.md` | `docs/SP-SERVER-DESIGN.md` | LANDED |

Next commit (Phase 1 kickoff): `sp_forward.cpp`, `sp_attention.cpp`, `sp_ffn.cpp`, `sp_sampler.cpp`, integration in `engine.cpp`.
