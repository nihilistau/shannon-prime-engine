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
#include "sp_bridges.h"
#include "sp_matmul.h"
#include "sp_ok_encode.h"
#include "sp_rope.h"

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

bool sp_forward_context_init(sp_forward_context& ctx,
                              const sp_weights&   weights,
                              int                 n_ctx,
                              float               rope_base,
                              float               rms_eps) {
    if (weights.n_layers <= 0 || weights.n_embd <= 0 || n_ctx <= 0) {
        return false;
    }
    ctx.n_layers      = weights.n_layers;
    ctx.n_embd        = weights.n_embd;
    ctx.n_head        = weights.n_head;
    ctx.n_kv_head     = weights.n_kv_head;
    ctx.head_dim      = weights.head_dim;
    ctx.n_ctx         = n_ctx;
    ctx.residual_scale = weights.scale_recip;
    ctx.rope_base     = rope_base;
    ctx.rms_eps       = rms_eps;
    ctx.poncelet_delta = sp_ok_t{ 0, 0 };

    ctx.x_fp32.assign(ctx.n_embd, 0.0f);
    ctx.proj_out_fp32.assign(ctx.n_embd, 0.0f);
    ctx.logits_fp32.assign(weights.vocab, 0.0f);

    // Layer scratch arena — enough for x_ok + x_norm_ok + q + k + v + attn_out
    // plus an FFN scratch arena's worth (d_ff buffer + matmul scratches).
    const int64_t d_q  = (int64_t)ctx.n_head    * ctx.head_dim;
    const int64_t d_kv = (int64_t)ctx.n_kv_head * ctx.head_dim;
    const size_t per_layer_bytes =
          (size_t)(  /*x_ok*/      ctx.n_embd
                   + /*x_norm_ok*/ ctx.n_embd
                   + /*q_ok*/      d_q
                   + /*k_ok*/      d_kv
                   + /*v_ok*/      d_kv
                   + /*attn_out*/  d_q
                   + /*ffn act*/   weights.d_ff
                  ) * sizeof(sp_ok_t)
        + 4096;
    ctx.layer_arena.reserve(per_layer_bytes);

    // KV cache: K post-RoPE has frobenius_scale=1, scale_recip = matmul out
    //          = wk.scale_recip * residual_scale = weights.scale_recip^2
    //          V keeps frobenius_scale = wv[0].frobenius_scale (pi^k or 1)
    //          and scale_recip = weights.scale_recip^2 (matmul output).
    const int64_t matmul_scale = (int64_t)weights.scale_recip * weights.scale_recip;
    if (weights.n_layers == 0) return false;
    const int64_t v_frob = weights.wv[0].frobenius_scale;

    const int64_t d_kv_total = d_kv;
    const size_t  kv_bytes_per_layer =
        (size_t)(2 * n_ctx * d_kv_total) * sizeof(sp_ok_t) + 4096;
    ctx.kv_arena.reserve(kv_bytes_per_layer * (size_t)weights.n_layers + 4096);
    if (!sp_ok_kv_cache_init(ctx.kv_cache,
                               weights.n_layers,
                               n_ctx,
                               weights.n_kv_head,
                               weights.head_dim,
                               /*k_scale*/  matmul_scale,
                               /*v_scale*/  matmul_scale,
                               /*v_frob*/   v_frob,
                               ctx.kv_arena)) {
        std::fprintf(stderr, "[sp_forward] kv_cache_init failed\n");
        return false;
    }
    return true;
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
    const int head_dim_local = (n_head > 0) ? (n_embd / n_head) : 0;
    const int64_t d_q  = (int64_t)n_head    * head_dim_local;
    const int64_t d_kv = (int64_t)n_kv_head * head_dim_local;
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
                       int64_t scale_recip,
                       int head_dim_arg) {
    if (n_layers <= 0 || n_embd <= 0 || n_head <= 0 || n_kv_head <= 0 ||
        d_ff <= 0 || vocab <= 0 || scale_recip <= 0) return false;
    int head_dim;
    if (head_dim_arg > 0) {
        head_dim = head_dim_arg;
    } else {
        if (n_embd % n_head != 0) return false;
        head_dim = n_embd / n_head;
    }
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
    // Phase 2.3b — optional per-layer norms left empty by default.
    out.attn_q_norm_w.assign(n_layers, std::vector<float>{});
    out.attn_k_norm_w.assign(n_layers, std::vector<float>{});
    out.attn_post_norm_w.assign(n_layers, std::vector<float>{});
    out.ffn_post_norm_w.assign(n_layers, std::vector<float>{});

    // Compute exact arena requirement using the *actual* head_dim
    // (NOT n_embd/n_head — those can differ, e.g. Gemma3).
    const size_t per_layer =
          tensor_bytes((int64_t)n_embd * d_q)
        + tensor_bytes((int64_t)n_embd * d_kv)
        + tensor_bytes((int64_t)n_embd * d_kv)
        + tensor_bytes(d_q * (int64_t)n_embd)
        + tensor_bytes((int64_t)n_embd * d_ff)
        + tensor_bytes((int64_t)n_embd * d_ff)
        + tensor_bytes((int64_t)d_ff   * n_embd);
    const size_t top =
          tensor_bytes((int64_t)n_embd * vocab)
        + tensor_bytes((int64_t)n_embd * vocab);
    const size_t arena_bytes = per_layer * (size_t)n_layers + top + 4096;
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

// Phase 2.3b — Gemma3 / Qwen3 optional norms.
bool sp_weights_set_attn_q_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    out.attn_q_norm_w[L].assign(src, src + out.head_dim);
    return true;
}
bool sp_weights_set_attn_k_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    out.attn_k_norm_w[L].assign(src, src + out.head_dim);
    return true;
}
bool sp_weights_set_attn_post_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    out.attn_post_norm_w[L].assign(src, src + out.n_embd);
    return true;
}
bool sp_weights_set_ffn_post_norm(sp_weights& out, int L, const float* src) {
    if (L < 0 || L >= out.n_layers || src == nullptr) return false;
    out.ffn_post_norm_w[L].assign(src, src + out.n_embd);
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

    // Phase 1.7 BYPASS POLICY: tok_embed and lm_head stay un-shimmed.
    //   tok_embed: first residual stream, "scale-reset" valve
    //   lm_head:   logit readout, scale ≠ softmax temperature
    // Both keep frobenius_scale=1 so downstream uses see the original
    // fp32 values (with no Theorem-4 cancellation needed since they
    // never compose with each other in a single matmul).

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
// Forward step — Phase 2.2d, native O_K inference (single-token decode).
// =========================================================================

// Helper: encode an fp32 vector [n_tokens, n_embd] into an sp_ok_tensor
// allocated from `arena`. Sets scale_recip=S, frobenius_scale=1.
static bool encode_residual_to_ok(sp_ok_tensor& dst,
                                    const float* src,
                                    int n_tokens, int n_embd,
                                    int64_t S, sp_ok_arena& arena) {
    int64_t shape[4] = { n_tokens, n_embd, 1, 1 };
    return sp_ok_encode_from_fp32(dst, src, 2, shape, S, arena);
}

// Helper: embedding lookup. weights.tok_embed is shape {n_embd, vocab},
// so the token's row starts at data[token_id * n_embd]. Decode to fp32.
static void embed_lookup_fp32(float* x_fp32,
                                const sp_ok_tensor& tok_embed,
                                int token_id, int n_embd) {
    const double div = (double)tok_embed.scale_recip *
                       (double)tok_embed.frobenius_scale;
    const sp_ok_t* row = tok_embed.data + (int64_t)token_id * n_embd;
    for (int i = 0; i < n_embd; ++i) {
        x_fp32[i] = (float)((double)row[i].a / div);
    }
}

bool sp_forward_step(sp_forward_context& ctx,
                     const sp_weights&   weights,
                     int                 token_id,
                     int                 position,
                     std::vector<float>& logits_out) {
    if (token_id < 0 || token_id >= weights.vocab) {
        std::fprintf(stderr, "[sp_forward] bad token_id=%d\n", token_id);
        return false;
    }
    if (position < 0 || position >= ctx.n_ctx) {
        std::fprintf(stderr, "[sp_forward] position %d out of n_ctx=%d\n",
                     position, ctx.n_ctx);
        return false;
    }
    if (ctx.kv_cache.cur_len != position) {
        std::fprintf(stderr,
            "[sp_forward] kv_cache.cur_len=%d but position=%d — caller "
            "must call sp_forward_step in sequential order, or reset the "
            "cache before non-sequential reads.\n",
            ctx.kv_cache.cur_len, position);
        return false;
    }

    const int n_embd    = ctx.n_embd;
    const int n_head    = ctx.n_head;
    const int n_kv_head = ctx.n_kv_head;
    const int head_dim  = ctx.head_dim;
    const int d_q       = n_head    * head_dim;
    const int d_kv      = n_kv_head * head_dim;
    const int64_t S     = ctx.residual_scale;
    const int64_t matmul_out_scale = S * S;
    const int n_tokens  = 1;  // single-token decode in Phase 2.2d

    // 1) Embedding lookup → x_fp32.
    embed_lookup_fp32(ctx.x_fp32.data(), weights.tok_embed, token_id, n_embd);

    int32_t rope_pos[1] = { position };

    for (int L = 0; L < ctx.n_layers; ++L) {
        ctx.layer_arena.reset();

        // 2a) Encode x_fp32 → x_ok.
        if (!encode_residual_to_ok(ctx.x_ok, ctx.x_fp32.data(),
                                     n_tokens, n_embd, S, ctx.layer_arena)) {
            std::fprintf(stderr, "[sp_forward] L%d encode x failed\n", L);
            return false;
        }

        // 2b) x_norm_ok = RMSNorm(x_ok, attn_norm_w[L]).
        int64_t xnorm_shape[4] = { n_tokens, n_embd, 1, 1 };
        ctx.x_norm_ok.reset(2, xnorm_shape);
        if (!ctx.layer_arena.alloc_tensor(ctx.x_norm_ok)) return false;
        ctx.x_norm_ok.scale_recip = S;
        if (!sp_rmsnorm_native(ctx.x_ok,
                                 weights.attn_norm_w[L].data(),
                                 ctx.x_norm_ok,
                                 ctx.rms_eps, n_embd, n_tokens)) {
            std::fprintf(stderr, "[sp_forward] L%d rmsnorm-attn failed\n", L);
            return false;
        }

        // 2c) Q/K/V = W[qkv] @ x_norm  (sp_matmul_ok output scale_recip=S*S,
        //     frobenius_scale = W.frobenius_scale * 1 = wq[L].frobenius_scale).
        int64_t q_shape[4]  = { n_tokens, d_q,  1, 1 };
        int64_t kv_shape[4] = { n_tokens, d_kv, 1, 1 };
        ctx.q_ok.reset(2, q_shape);
        ctx.k_ok.reset(2, kv_shape);
        ctx.v_ok.reset(2, kv_shape);
        if (!ctx.layer_arena.alloc_tensor(ctx.q_ok)) return false;
        if (!ctx.layer_arena.alloc_tensor(ctx.k_ok)) return false;
        if (!ctx.layer_arena.alloc_tensor(ctx.v_ok)) return false;
        if (!sp_matmul_ok(weights.wq[L], ctx.x_norm_ok, ctx.q_ok)) return false;
        if (!sp_matmul_ok(weights.wk[L], ctx.x_norm_ok, ctx.k_ok)) return false;
        if (!sp_matmul_ok(weights.wv[L], ctx.x_norm_ok, ctx.v_ok)) return false;

        // 2c.5) Phase 2.3b: optional per-head Q/K norms (Gemma3 / Qwen3).
        // These reset frobenius_scale → 1, so subsequent RoPE re-encodes
        // at qk.scale_recip (unchanged) with frob=1.
        if (!weights.attn_q_norm_w[L].empty()) {
            if (!sp_per_head_rmsnorm_native(
                    ctx.q_ok,
                    weights.attn_q_norm_w[L].data(),
                    ctx.rms_eps, n_head, head_dim, n_tokens)) {
                std::fprintf(stderr, "[sp_forward] L%d q-norm failed\n", L);
                return false;
            }
        }
        if (!weights.attn_k_norm_w[L].empty()) {
            if (!sp_per_head_rmsnorm_native(
                    ctx.k_ok,
                    weights.attn_k_norm_w[L].data(),
                    ctx.rms_eps, n_kv_head, head_dim, n_tokens)) {
                std::fprintf(stderr, "[sp_forward] L%d k-norm failed\n", L);
                return false;
            }
        }

        // 2d) RoPE on Q and K (frobenius_scale → 1 after each call).
        if (!sp_rope_apply_ok(ctx.q_ok, n_head,    head_dim, n_tokens,
                                rope_pos, ctx.rope_base, 1.0f)) return false;
        if (!sp_rope_apply_ok(ctx.k_ok, n_kv_head, head_dim, n_tokens,
                                rope_pos, ctx.rope_base, 1.0f)) return false;
        // V keeps its post-matmul frobenius_scale (matches v cache slot).

        // Sanity: K/V must match the cache's stored scales for the strict
        // append guard.
        if (ctx.k_ok.scale_recip != ctx.kv_cache.layers[L].K.scale_recip ||
            ctx.k_ok.frobenius_scale != ctx.kv_cache.layers[L].K.frobenius_scale) {
            std::fprintf(stderr,
                "[sp_forward] L%d K cache scale mismatch: k=(%lld,%lld) cache=(%lld,%lld)\n",
                L, (long long)ctx.k_ok.scale_recip, (long long)ctx.k_ok.frobenius_scale,
                (long long)ctx.kv_cache.layers[L].K.scale_recip,
                (long long)ctx.kv_cache.layers[L].K.frobenius_scale);
            return false;
        }
        if (ctx.v_ok.scale_recip != ctx.kv_cache.layers[L].V.scale_recip ||
            ctx.v_ok.frobenius_scale != ctx.kv_cache.layers[L].V.frobenius_scale) {
            std::fprintf(stderr,
                "[sp_forward] L%d V cache scale mismatch: v=(%lld,%lld) cache=(%lld,%lld)\n",
                L, (long long)ctx.v_ok.scale_recip, (long long)ctx.v_ok.frobenius_scale,
                (long long)ctx.kv_cache.layers[L].V.scale_recip,
                (long long)ctx.kv_cache.layers[L].V.frobenius_scale);
            return false;
        }

        // 2e) KV cache append (cur_len doesn't advance until the loop ends).
        if (!sp_ok_kv_cache_append_layer(ctx.kv_cache, L,
                                           ctx.k_ok, ctx.v_ok, n_tokens)) {
            std::fprintf(stderr, "[sp_forward] L%d kv append failed\n", L);
            return false;
        }

        // 2f) Attention over the cache view.
        sp_ok_tensor K_view = sp_ok_kv_cache_view_k(ctx.kv_cache, L);
        sp_ok_tensor V_view = sp_ok_kv_cache_view_v(ctx.kv_cache, L);
        const int t_valid  = ctx.kv_cache.cur_len + n_tokens;
        const int t_stride = ctx.n_ctx;

        ctx.attn_out_ok.reset(2, q_shape);
        if (!ctx.layer_arena.alloc_tensor(ctx.attn_out_ok)) return false;
        // attn_out re-encodes the post-softmax V sum (which is in original
        // un-scaled units after Frobenius cancellation in sp_attention). We
        // pick scale_recip = S (NOT S^2) so the downstream Wo matmul's
        // combined fp64 divisor stays at S * S * pi^k ≈ 2^49, well under
        // the 2^53 fp64 mantissa limit. Higher scale would compound with
        // Wo's pi^k Frobenius factor and lose precision.
        ctx.attn_out_ok.scale_recip = S;
        sp_attention_dot_product(ctx.q_ok, K_view, V_view, ctx.attn_out_ok,
                                    n_head, n_kv_head, head_dim,
                                    t_valid, t_stride, position);
        // attn_out_ok now has frobenius_scale=1.

        // 2g) Wo projection → fp32 (absorbs pi^k via Theorem 4).
        if (!sp_matmul_ok_to_fp32(weights.wo[L], ctx.attn_out_ok,
                                    ctx.proj_out_fp32.data(),
                                    n_embd, n_tokens)) {
            std::fprintf(stderr, "[sp_forward] L%d Wo matmul failed\n", L);
            return false;
        }

        // 2g.5) Phase 2.3b: Gemma3 attn sandwich norm on Wo output.
        if (!weights.attn_post_norm_w[L].empty()) {
            sp_rmsnorm_fp32(
                ctx.proj_out_fp32.data(),
                weights.attn_post_norm_w[L].data(),
                ctx.proj_out_fp32.data(),
                n_embd, n_tokens, ctx.rms_eps);
        }

        // 2h) Residual: x_fp32 += wo_out_fp32 (both fp32).
        for (int i = 0; i < n_embd; ++i) {
            ctx.x_fp32[i] += ctx.proj_out_fp32[i];
        }

        // ------- FFN block -------
        ctx.layer_arena.reset();

        // 2i) re-encode x_fp32 → x_ok
        if (!encode_residual_to_ok(ctx.x_ok, ctx.x_fp32.data(),
                                     n_tokens, n_embd, S, ctx.layer_arena)) {
            return false;
        }
        // 2j) x_norm2_ok = RMSNorm(x_ok, ffn_norm_w[L])
        ctx.x_norm_ok.reset(2, xnorm_shape);
        if (!ctx.layer_arena.alloc_tensor(ctx.x_norm_ok)) return false;
        ctx.x_norm_ok.scale_recip = S;
        if (!sp_rmsnorm_native(ctx.x_ok,
                                 weights.ffn_norm_w[L].data(),
                                 ctx.x_norm_ok,
                                 ctx.rms_eps, n_embd, n_tokens)) {
            return false;
        }

        // 2k) FFN → fp32 (absorbs ffn_down's frobenius_scale)
        if (!sp_ffn_swiglu_to_fp32(ctx.x_norm_ok,
                                     weights.ffn_gate[L],
                                     weights.ffn_up[L],
                                     weights.ffn_down[L],
                                     ctx.proj_out_fp32.data(),
                                     n_tokens,
                                     ctx.layer_arena)) {
            std::fprintf(stderr, "[sp_forward] L%d FFN failed\n", L);
            return false;
        }

        // 2k.5) Phase 2.3b: Gemma3 FFN sandwich norm on down output.
        if (!weights.ffn_post_norm_w[L].empty()) {
            sp_rmsnorm_fp32(
                ctx.proj_out_fp32.data(),
                weights.ffn_post_norm_w[L].data(),
                ctx.proj_out_fp32.data(),
                n_embd, n_tokens, ctx.rms_eps);
        }

        // 2l) Residual: x_fp32 += ffn_out_fp32
        for (int i = 0; i < n_embd; ++i) {
            ctx.x_fp32[i] += ctx.proj_out_fp32[i];
        }
    }

    // Advance the KV cache write head AFTER all layers' appends.
    sp_ok_kv_cache_advance(ctx.kv_cache, n_tokens);

    // 3) Final RMSNorm.
    ctx.layer_arena.reset();
    if (!encode_residual_to_ok(ctx.x_ok, ctx.x_fp32.data(),
                                 n_tokens, n_embd, S, ctx.layer_arena)) {
        return false;
    }
    int64_t xnorm_shape[4] = { n_tokens, n_embd, 1, 1 };
    ctx.x_norm_ok.reset(2, xnorm_shape);
    if (!ctx.layer_arena.alloc_tensor(ctx.x_norm_ok)) return false;
    ctx.x_norm_ok.scale_recip = S;
    if (!sp_rmsnorm_native(ctx.x_ok,
                             weights.final_norm_w.data(),
                             ctx.x_norm_ok,
                             ctx.rms_eps, n_embd, n_tokens)) {
        return false;
    }

    // 4) LM head (bypass — frobenius_scale = 1).
    logits_out.assign(weights.vocab, 0.0f);
    if (!sp_matmul_ok_to_fp32(weights.lm_head, ctx.x_norm_ok,
                                logits_out.data(),
                                weights.vocab, n_tokens)) {
        return false;
    }
    return true;
}

}  // namespace sp::engine
