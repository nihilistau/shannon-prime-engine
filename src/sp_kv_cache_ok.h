// Shannon-Prime Engine — O_K-coordinate KV cache.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.2b. Stores K and V across the sequence axis as sp_ok_tensor
// elements. Distinct from the existing KvCache (which holds VHT2/sqfree/
// hierarchical compressed bytes) — that one targets the long-context
// compression path. This one is the working KV history that the native
// O_K forward pass reads inside sp_attention_dot_product on each step.
//
// Layout per layer:
//   K_layer.shape = { max_len, n_kv_head * head_dim }
//   V_layer.shape = { max_len, n_kv_head * head_dim }
//   K_layer.data[feature * max_len + position]
//
// Scale invariants (Theorem 4):
//   K_layer.scale_recip      = wk.scale_recip * x.scale_recip
//   K_layer.frobenius_scale  = 1   (set post-RoPE)
//   V_layer.scale_recip      = wv.scale_recip * x.scale_recip
//   V_layer.frobenius_scale  = wv.frobenius_scale  (V doesn't get RoPE'd)
//
// All slots are pre-allocated from the arena at init; append is just an
// indexed copy + cur_len advance. No reallocation, no fragmentation.

#pragma once

#include "sp_ok_tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp::engine {

// Per-layer K/V cache slot.
struct sp_ok_kv_layer {
    sp_ok_tensor K;       // shape = { max_len, n_kv_head * head_dim }
    sp_ok_tensor V;       // shape = { max_len, n_kv_head * head_dim }
};

// Full cache: per-layer slots + current write head.
struct sp_ok_kv_cache {
    std::vector<sp_ok_kv_layer> layers;
    int  cur_len  = 0;       // number of valid tokens currently in cache
    int  max_len  = 0;
    int  n_layers = 0;
    int  n_kv_head = 0;
    int  head_dim  = 0;

    // Arena that owns the storage. Caller holds a reference; the cache
    // does NOT own the arena (the forward context does).
    sp_ok_arena* arena = nullptr;
};

// Allocate K and V slots for every layer.
//   max_len: maximum sequence length the cache can hold.
//   n_kv_head, head_dim: model dims.
//   k_scale_recip, v_scale_recip: per-element scale for fp16 decode round-trip.
//   v_frobenius_scale: V keeps its Frobenius factor (no RoPE on V).
//
// Returns false if the arena can't hold the requested allocation.
bool sp_ok_kv_cache_init(sp_ok_kv_cache& cache,
                          int             n_layers,
                          int             max_len,
                          int             n_kv_head,
                          int             head_dim,
                          int64_t         k_scale_recip,
                          int64_t         v_scale_recip,
                          int64_t         v_frobenius_scale,
                          sp_ok_arena&    arena);

// Append `n_new_tokens` columns of K_new and V_new to the layer-`layer_idx`
// slot at offset cache.cur_len. Does NOT advance cur_len — the caller is
// responsible for advancing AFTER all layers have appended for this step.
//
// K_new.shape  = { n_new_tokens, n_kv_head * head_dim }
// V_new.shape  = { n_new_tokens, n_kv_head * head_dim }
//
// Returns false on shape/scale mismatch or out-of-bounds append.
bool sp_ok_kv_cache_append_layer(sp_ok_kv_cache&     cache,
                                  int                 layer_idx,
                                  const sp_ok_tensor& K_new,
                                  const sp_ok_tensor& V_new,
                                  int                 n_new_tokens);

// Advance cur_len after all layers have appended for this step.
inline void sp_ok_kv_cache_advance(sp_ok_kv_cache& cache, int n) {
    cache.cur_len += n;
}

// Construct a *view* tensor over the valid prefix [0, cur_len) of a layer's
// K (or V) slot. The view shares storage with the underlying cache slot,
// but exposes shape = { cur_len, n_kv_head * head_dim } so sp_attention
// reads only the populated history.
//
// Returns the view (data pointer = layer slot's data; same scale_recip and
// frobenius_scale).
sp_ok_tensor sp_ok_kv_cache_view_k(const sp_ok_kv_cache& cache, int layer_idx);
sp_ok_tensor sp_ok_kv_cache_view_v(const sp_ok_kv_cache& cache, int layer_idx);

// Reset cur_len to 0 (does not clear memory; subsequent appends will
// overwrite).
inline void sp_ok_kv_cache_clear(sp_ok_kv_cache& cache) { cache.cur_len = 0; }

}  // namespace sp::engine
