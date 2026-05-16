// Shannon-Prime Engine — sp_weights loader.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.2c. Populates sp_weights from a fp16/fp32 source — typically a
// loaded GGUF via LlamaWeights, but the source is abstracted as a struct
// of typed pointers so the loader is testable in isolation.
//
// Pipeline:
//   1. sp_weights_alloc()                              — pre-sized slots
//   2. fp16/fp32 source pointers → per-slot fp32 dequant → setters
//   3. sp_weights_apply_frobenius_shim() per cfg flags
//
// The source layout follows the ggml/GGUF convention exactly:
//   tensor "blk.N.attn_q.weight" has ne = (n_embd, d_q), row-major,
//   meaning element (output_i, input_k) is at buf[i * n_embd + k].
// This matches the sp_weights slot data layout (sp_matmul: shape[0]=
// inner=n_embd, shape[1]=outer=d_q, data[i * n_embd + k]).

#pragma once

#include "sp_forward.h"

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace sp::engine {

class LlamaWeights;

// Per-layer source pointers. Each pointer addresses a row-major buffer
// of the appropriate fp16 (uint16_t) or fp32 (float) shape.
//
// Any pointer may be null — the corresponding slot is left at the
// default-allocated zero state.
struct sp_weights_layer_fp16_source {
    const uint16_t* wq         = nullptr;   // [d_q   * n_embd]
    const uint16_t* wk         = nullptr;   // [d_kv  * n_embd]
    const uint16_t* wv         = nullptr;   // [d_kv  * n_embd]
    const uint16_t* wo         = nullptr;   // [n_embd * d_q]
    const uint16_t* ffn_gate   = nullptr;   // [d_ff  * n_embd]
    const uint16_t* ffn_up     = nullptr;   // [d_ff  * n_embd]
    const uint16_t* ffn_down   = nullptr;   // [n_embd * d_ff]
    const float*    attn_norm  = nullptr;   // [n_embd]   (fp32 bypass)
    const float*    ffn_norm   = nullptr;   // [n_embd]   (fp32 bypass)
    // Phase 2.3b: optional Gemma3 / Qwen3 norms. Null = not applied.
    const float*    attn_q_norm    = nullptr;  // [head_dim]
    const float*    attn_k_norm    = nullptr;  // [head_dim]
    const float*    attn_post_norm = nullptr;  // [n_embd]
    const float*    ffn_post_norm  = nullptr;  // [n_embd]
};

struct sp_weights_fp16_source {
    int n_layers   = 0;
    int n_embd     = 0;
    int n_head     = 0;
    int n_kv_head  = 0;
    int d_ff       = 0;
    int vocab      = 0;
    // If 0: use n_embd / n_head (standard). For Gemma3 etc. pass the
    // model's actual head_dim — d_q = n_head * head_dim may be larger
    // than n_embd.
    int head_dim_override = 0;

    const uint16_t* tok_embd     = nullptr;   // [vocab * n_embd]
    const uint16_t* lm_head      = nullptr;   // [vocab * n_embd] (may = tok_embd if tied)
    const float*    final_norm   = nullptr;   // [n_embd]

    const sp_weights_layer_fp16_source* layers = nullptr;  // [n_layers]
};

// Populate `out` from `src`. Allocates the arena, encodes each slot,
// then runs sp_weights_apply_frobenius_shim per cfg.frobenius_quant /
// cfg.sato_tate_mix.
//
// `scale_recip`: encoding scale to use for every sp_ok_tensor. Pick the
// same recommended scale used by the load shim (typically 1<<14).
//
// Returns true on success.
bool sp_weights_load_from_fp16_source(sp_weights& out,
                                        const sp_weights_fp16_source& src,
                                        const Config& cfg,
                                        int64_t scale_recip);

// Convenience: build the fp16_source from a loaded LlamaWeights instance.
// Returns false on missing required tensors or unsupported arch.
//
// Dims must be passed explicitly (LlamaWeights doesn't carry them; the
// caller already has them from the Model).
//
// The returned weights remain bound to `weights` for the lifetime of the
// call only — sp_weights_load_from_fp16_source copies all data into the
// sp_weights arena, so `out` is self-contained afterwards.
bool sp_weights_load_from_llama(sp_weights& out,
                                  const LlamaWeights& weights,
                                  const Config& cfg,
                                  int n_head, int n_kv_head, int head_dim,
                                  int64_t scale_recip);

}  // namespace sp::engine
