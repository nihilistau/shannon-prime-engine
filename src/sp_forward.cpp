// Shannon-Prime Engine — Theory-First forward pass.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 2.2b: sp_weights allocation + per-slot fp32 setter API, ready for
// the Phase 2.2c GGUF tensor walker to wire on top.
//
// Phase 2.2d will wire sp_forward_step end-to-end on top of sp_weights +
// sp_ok_kv_cache + sp_attention + sp_ffn + sp_rmsnorm_native + sp_rope.

#include "sp_forward.h"
#include "sp_attention.h"
#include "sp_ffn.h"
#include "sp_ok_encode.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_frobenius.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

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
    size_t max_elements = (size_t)(8 * n_embd);
    ctx.arena.reserve(max_elements * sizeof(sp_ok_t) + 4096);

    ctx.poncelet_delta = sp_ok_t{ 0, 0 };
    (void)cfg;
}

// =========================================================================
// sp_weights allocation
// =========================================================================

// Helper: required bytes for one sp_ok_tensor of `numel` elements, plus
// 64-byte alignment padding.
static inline size_t tensor_bytes(int64_t numel) {
    return (size_t)numel * sizeof(sp_ok_t) + 64;
}

size_t sp_weights_required_arena_bytes(int n_layers, int n_embd,
                                         int n_head, int n_kv_head,
                                         int d_ff, int vocab) {
    (void)n_head;
    const int64_t d_q  = (int64_t)n_embd;          // n_head * head_dim
    const int64_t d_kv = (int64_t)n_kv_head * (n_embd / std::max(n_head, 1));
    // Per layer: wq (n_embd*d_q) + wk (n_embd*d_kv) + wv + wo (d_q*n_embd)
    //          + gate (n_embd*d_ff) + up + down (d_ff*n_embd)
    const size_t per_layer =
          tensor_bytes((int64_t)n_embd * d_q)
        + tensor_bytes((int64_t)n_embd * d_kv)
        + tensor_bytes((int64_t)n_embd * d_kv)
        + tensor_bytes(d_q * (int64_t)n_embd)
        + tensor_bytes((int64_t)n_embd * d_ff)
        + tensor_bytes((int64_t)n_embd * d_ff)
        + tensor_bytes((int64_t)d_ff   * n_embd);
    // Top-level: tok_embed + lm_head
    const size_t top =
          tensor_bytes((int64_t)n_embd * vocab)
        + tensor_bytes((int64_t)n_embd * vocab);
    return per_layer * (size_t)n_layers + top + 4096;
}

bool sp_weights_alloc(sp_weights& out, int n_layers, int n_embd,
                       int n_head, int n_kv_head, int d_ff, int vocab,
                       int64_t scale_recip) {
    if (n_layers <= 0 || n_embd <= 0 || n_head <= 0 || n_kv_head <= 0 ||
        d_ff <= 0 || vocab <= 0 || scale_recip <= 0) return false;
    if (n_embd % n_head != 0) return false;
    const int head_dim = n_embd / n_head;
    const int64_t d_q  = (int64_t)n_head    * head_dim;
    const int64_t d_kv = (int64_t)n_kv_head * head_dim;

    out.n_layers   = n_layers;
    out.n_embd     = n_embd;
    out.n_head     = n_head;
    out.n_kv_head  = n_kv_head;
    out.head_dim   = head_dim;
    out.d_ff       = d_ff;
    out.vocab      = vocab;
    out.scale_recip = scale_recip;

    out.wq.resize(n_layers);
    out.wk.resize(n_layers);
    out.wv.resize(n_layers);
    out.wo.resize(n_layers);
    out.ffn_gate.resize(n_layers);
    out.ffn_up.resize(n_layers);
    out.ffn_down.resize(n_layers);
    out.attn_norm_w.resize(n_layers);
    out.ffn_norm_w.resize(n_layers);

    const size_t arena_bytes = sp_weights_required_arena_bytes(
        n_layers, n_embd, n_head, n_kv_head, d_ff, vocab);
    out.storage.reserve(arena_bytes);

    auto alloc_with_shape = [&](sp_ok_tensor& t, int64_t s0, int64_t s1) -> bool {
        int64_t shp[4] = { s0, s1, 1, 1 };
        t.reset(2, shp);
        if (!out.storage.alloc_tensor(t)) return false;
        t.scale_recip = scale_recip;
        t.frobenius_scale = 1;
        return true;
    };

    // Top-level.
    if (!alloc_with_shape(out.tok_embed, n_embd, vocab)) return false;
    if (!alloc_with_shape(out.lm_head,   n_embd, vocab)) return false;

    for (int L = 0; L < n_layers; ++L) {
        if (!alloc_with_shape(out.wq[L],       n_embd, d_q))   return false;
        if (!alloc_with_shape(out.wk[L],       n_embd, d_kv))  return false;
        if (!alloc_with_shape(out.wv[L],       n_embd, d_kv))  return false;
        if (!alloc_with_shape(out.wo[L],       d_q,    n_embd)) return false;
        if (!alloc_with_shape(out.ffn_gate[L], n_embd, d_ff))  return false;
        if (!alloc_with_shape(out.ffn_up[L],   n_embd, d_ff))  return false;
        if (!alloc_with_shape(out.ffn_down[L], d_ff,   n_embd)) return false;
        out.attn_norm_w[L].resize(n_embd, 1.0f);
        out.ffn_norm_w[L].resize(n_embd,  1.0f);
    }
    out.final_norm_w.assign(n_embd, 1.0f);
    return true;
}

// =========================================================================
// Per-slot setters
//
// Source convention: row-major [out_units, in_dims] with src[i*in_dims + k].
// Slot convention (sp_matmul):  shape[0]=in_dims (inner), shape[1]=out_units.
//                                slot.data[i * in_dims + k] for output i,
//                                input k.
// So the two conventions match directly — just a memcpy after fp32→int64
// round + scale.
// =========================================================================

static bool fill_slot(sp_ok_tensor& slot, const float* src) {
    if (slot.data == nullptr || src == nullptr) return false;
    const int64_t n = slot.numel();
    const double S = (double)slot.scale_recip;
    for (int64_t i = 0; i < n; ++i) {
        double v = (double)src[i] * S;
        slot.data[i] = sp_ok_t{ (int64_t)std::llrint(v), 0 };
    }
    slot.frobenius_scale = 1;
    return true;
}

bool sp_weights_set_tok_embed(sp_weights& out, const float* src) {
    return fill_slot(out.tok_embed, src);
}
bool sp_weights_set_lm_head(sp_weights& out, const float* src) {
    return fill_slot(out.lm_head, src);
}
bool sp_weights_set_wq(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.wq[L], src);
}
bool sp_weights_set_wk(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.wk[L], src);
}
bool sp_weights_set_wv(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.wv[L], src);
}
bool sp_weights_set_wo(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.wo[L], src);
}
bool sp_weights_set_ffn_gate(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.ffn_gate[L], src);
}
bool sp_weights_set_ffn_up(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.ffn_up[L], src);
}
bool sp_weights_set_ffn_down(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers) return false;
    return fill_slot(out.ffn_down[L], src);
}
bool sp_weights_set_attn_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    std::memcpy(out.attn_norm_w[L].data(), src, sizeof(float) * out.n_embd);
    return true;
}
bool sp_weights_set_ffn_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    std::memcpy(out.ffn_norm_w[L].data(), src, sizeof(float) * out.n_embd);
    return true;
}
bool sp_weights_set_final_norm(sp_weights& out, const float* src) {
    if (src == nullptr) return false;
    std::memcpy(out.final_norm_w.data(), src, sizeof(float) * out.n_embd);
    return true;
}

// =========================================================================
// Apply Frobenius shim
// =========================================================================

int sp_weights_apply_frobenius_shim(sp_weights& out,
                                      bool frobenius_quant,
                                      bool sato_tate_mix,
                                      int64_t p,  int64_t k,
                                      int64_t p1, int64_t k1,
                                      int64_t p2, int64_t k2) {
    int n_transformed = 0;
    auto apply_one = [&](sp_ok_tensor& t) {
        if (t.data == nullptr) return;
        if (sato_tate_mix) {
            sp_ok_encode_apply_sato_tate_mix(t, p1, k1, p2, k2);
        } else if (frobenius_quant) {
            sp_ok_encode_apply_frobenius_quant(t, p, k);
        } else {
            return;  // no-op tier
        }
        ++n_transformed;
    };

    // Top-level shim-list. Note: lm_head is BYPASS per Phase 1.7 policy
    // (it's the readout, not an interior matmul). tok_embed is bypass too.
    // For now we still shim everything to match the Phase 1.8 PPL baseline.
    // The Phase 2.2c GGUF walker will respect the bypass policy.
    apply_one(out.tok_embed);
    apply_one(out.lm_head);

    for (int L = 0; L < out.n_layers; ++L) {
        apply_one(out.wq[L]);
        apply_one(out.wk[L]);
        apply_one(out.wv[L]);
        apply_one(out.wo[L]);
        apply_one(out.ffn_gate[L]);
        apply_one(out.ffn_up[L]);
        apply_one(out.ffn_down[L]);
    }
    return n_transformed;
}

// =========================================================================
// Weight init (skeleton — Phase 2.2c will fill this in via LlamaWeights)
// =========================================================================

bool sp_weights_init_from_fp16(sp_weights& out,
                                const void* loaded_model,
                                const Config& cfg) {
    (void)out; (void)loaded_model; (void)cfg;
    // Phase 2.2c work. The unit-test path uses sp_weights_alloc +
    // sp_weights_set_* setters directly.
    return false;
}

// =========================================================================
// Forward step (skeleton — Phase 2.2d will wire end-to-end)
// =========================================================================

void sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out) {
    (void)ctx; (void)weights; (void)token_id; (void)position; (void)logits_out;
    std::fprintf(stderr,
        "[sp_forward] Phase 2.2d will wire end-to-end. Currently building "
        "the kernel + weight + cache primitives in Phase 2.2b.\n");
}

}  // namespace sp::engine
