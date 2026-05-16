// Shannon-Prime Engine — Theory-First forward pass (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The pure-SP forward pass. Every step is an endomorphism of E^n (or
// the Siegel-variety multi-head generalization, Paper A §3) realized
// as a sequence of sp_ok_tensor operations. ggml is NOT used.
//
// In Phase 1 SKELETON (this file), the implementation delegates to
// existing forward.cpp / forward_native.cpp with weights pre-encoded
// through sp_ok_encode (the Frobenius shim). Phase 1.6 work fills in
// the actual SP-native ops.
//
// Reference: docs/THEORY-FIRST-ENGINE-DESIGN.md §Forward pass.

#pragma once

#include "sp_ok_tensor.h"
#include "engine.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_forward_context — per-request inference state. Holds the residual
// stream, KV cache, current layer index, working arena.
// -----------------------------------------------------------------------
struct sp_forward_context {
    sp_ok_tensor x;          // residual stream  [n_embd, batch]
    sp_ok_tensor x_norm;     // post-RMSNorm scratch
    sp_ok_tensor q;          // Q-projection scratch
    sp_ok_tensor k;          // K-projection scratch
    sp_ok_tensor v;          // V-projection scratch
    sp_ok_tensor attn_out;   // attention output scratch
    sp_ok_tensor ffn_out;    // FFN output scratch

    sp_ok_arena  arena;      // per-step scratch arena (reset between layers)

    int          n_layers;
    int          n_embd;
    int          n_head;
    int          n_kv_head;
    int          head_dim;

    // Poncelet adaptive depth tracking (Paper A §7, Theorem 5).
    // Partial sum of layer-endomorphisms in O_K. When the partial sum
    // vanishes modulo the working prime, sp_forward_step exits early.
    sp_ok_t      poncelet_delta;
};

// -----------------------------------------------------------------------
// sp_weights — per-model weight tensors in O_K coordinates.
// -----------------------------------------------------------------------
struct sp_weights {
    sp_ok_tensor              tok_embed;       // [vocab, n_embd]
    std::vector<sp_ok_tensor> wq;              // per-layer Q projection
    std::vector<sp_ok_tensor> wk;
    std::vector<sp_ok_tensor> wv;
    std::vector<sp_ok_tensor> wo;              // attention output proj
    std::vector<sp_ok_tensor> ffn_gate;
    std::vector<sp_ok_tensor> ffn_up;
    std::vector<sp_ok_tensor> ffn_down;
    std::vector<sp_ok_tensor> norm_w;          // per-layer RMSNorm scale
    std::vector<sp_ok_tensor> ffn_norm_w;      // per-layer post-attn norm scale
    sp_ok_tensor              final_norm_w;
    sp_ok_tensor              lm_head;         // [n_embd, vocab]

    sp_ok_arena               storage;         // owns all the data
};

// -----------------------------------------------------------------------
// Top-level forward functions.
// -----------------------------------------------------------------------

// Run a single forward step: given a token id, produce logits[vocab].
//
// Phase 1.6 SKELETON: this delegates to forward.cpp with weights that
// have been encoded → Frobenius-applied → decoded back to fp16. The
// "pure" sp_forward (no decode step) is Phase 2 work.
void sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out);

// Initialize a forward context for a given model.
void sp_forward_context_init(sp_forward_context& ctx,
                              const Config&       cfg,
                              int                 n_embd,
                              int                 n_layers,
                              int                 n_head,
                              int                 n_kv_head);

// Initialize sp_weights by encoding fp16 weights from a loaded model.
// Returns true on success. Weights remain valid for the lifetime of
// `out` (the arena owns the backing storage).
bool sp_weights_init_from_fp16(sp_weights& out,
                                /* loaded model fp16 weight handle */ const void* loaded_model,
                                const Config& cfg);

}  // namespace sp::engine
