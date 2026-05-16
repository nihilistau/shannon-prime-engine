// Shannon-Prime Engine — Theory-First forward pass (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// SKELETON: structure + signatures + stubs. Phase 1.6 work fills in the
// SP-native ops. Until then, calls to sp_forward_step delegate to the
// existing forward.cpp via the engine.cpp dispatch path.

#include "sp_forward.h"
#include "sp_attention.h"
#include "sp_ffn.h"
#include "sp_ok_encode.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_frobenius.h"
}

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sp::engine {

// =========================================================================
// Context init
// =========================================================================

void sp_forward_context_init(sp_forward_context& ctx,
                              const Config&       cfg,
                              int                 n_embd,
                              int                 n_layers,
                              int                 n_head,
                              int                 n_kv_head) {
    ctx.n_layers   = n_layers;
    ctx.n_embd     = n_embd;
    ctx.n_head     = n_head;
    ctx.n_kv_head  = n_kv_head;
    ctx.head_dim   = n_head > 0 ? (n_embd / n_head) : 0;

    // Reserve a scratch arena sized for the largest layer intermediate.
    // Per-layer working set:
    //   x       : n_embd elements
    //   x_norm  : n_embd
    //   q       : n_embd (Q is shape n_head * head_dim = n_embd)
    //   k, v    : n_kv_head * head_dim each
    //   attn_out: n_embd
    //   ffn_out : 4 * n_embd (gated FFN intermediate)
    size_t max_elements = (size_t)(8 * n_embd);
    ctx.arena.reserve(max_elements * sizeof(sp_ok_t) + 4096);

    ctx.poncelet_delta = sp_ok_t{ 0, 0 };
    (void)cfg;
}

// =========================================================================
// Weight init (skeleton)
//
// Phase 1.6 stub: returns false. The real implementation walks the
// loaded model's tensors, calls sp_ok_encode_from_fp16 for each,
// applies the Frobenius shim per cfg.frobenius_quant / cfg.sato_tate_mix.
// =========================================================================

bool sp_weights_init_from_fp16(sp_weights& out,
                                const void* loaded_model,
                                const Config& cfg) {
    (void)out;
    (void)loaded_model;
    (void)cfg;
    // Phase 1.6 placeholder. Phase 2 implementation:
    //   1. Iterate loaded_model.tensors
    //   2. For each: read fp16 buffer, encode to sp_ok_tensor, store in
    //      the appropriate slot of `out`.
    //   3. If cfg.frobenius_quant or cfg.sato_tate_mix, apply the shim.
    //   4. Return true.
    return false;
}

// =========================================================================
// Forward step (skeleton)
//
// Phase 1.6 stub. The full implementation runs the pure-SP pipeline:
//   1. embed(token_id) → x
//   2. for each layer:
//      a. RMSNorm → x_norm
//      b. Q/K/V projection (matmul in O_K)
//      c. RoPE on Q, K
//      d. attention via sp_attention
//      e. residual: x += attn_out
//      f. RMSNorm
//      g. SwiGLU FFN via sp_ffn
//      h. residual: x += ffn_out
//      i. Poncelet closure check → early exit if Δ_L == 0 mod p
//   3. final RMSNorm
//   4. LM head (CRT-decomposed) → logits_out
// =========================================================================

void sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out) {
    (void)ctx; (void)weights; (void)token_id; (void)position; (void)logits_out;
    // Phase 1.6 placeholder. The pure-SP forward pass is the deliverable
    // of the next major commit; until then, callers go through
    // engine.cpp → forward.cpp with shim-encoded weights.
    std::fprintf(stderr,
        "[sp_forward] Phase 1.6 SKELETON — direct sp_forward_step not yet "
        "implemented. Use Engine::generate which routes through "
        "forward.cpp with weights pre-encoded via sp_ok_encode.\n");
}

}  // namespace sp::engine
