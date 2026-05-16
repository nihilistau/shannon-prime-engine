// Shannon-Prime Engine — sp_weights loader (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_weights_loader.h"
#include "llama_weights.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace sp::engine {

// =========================================================================
// fp16 → fp32 dequantization. We avoid a bulk converter to keep this TU
// self-contained — the loader runs once at startup, so per-element cost
// is negligible.
// =========================================================================

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

static void dequant_fp16_to_fp32(std::vector<float>& dst,
                                  const uint16_t* src, size_t n) {
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
}

// =========================================================================
// sp_weights_load_from_fp16_source
// =========================================================================

bool sp_weights_load_from_fp16_source(sp_weights& out,
                                        const sp_weights_fp16_source& src,
                                        const Config& cfg,
                                        int64_t scale_recip) {
    if (src.n_layers <= 0 || src.n_embd <= 0 || src.n_head <= 0 ||
        src.n_kv_head <= 0 || src.d_ff <= 0 || src.vocab <= 0) {
        std::fprintf(stderr, "[sp-weights-loader] invalid source dims\n");
        return false;
    }
    if (src.layers == nullptr) {
        std::fprintf(stderr, "[sp-weights-loader] null layer array\n");
        return false;
    }
    if (scale_recip <= 0) scale_recip = 1 << 14;

    if (!sp_weights_alloc(out, src.n_layers, src.n_embd, src.n_head,
                            src.n_kv_head, src.d_ff, src.vocab,
                            scale_recip)) {
        std::fprintf(stderr, "[sp-weights-loader] alloc failed\n");
        return false;
    }

    const int    head_dim = out.head_dim;
    const size_t d_q      = (size_t)src.n_head    * head_dim;
    const size_t d_kv     = (size_t)src.n_kv_head * head_dim;

    std::vector<float> scratch;
    int n_set_shim = 0, n_set_bypass = 0, n_missing = 0;

    // --- Top-level shim-list tensors ---
    if (src.tok_embd) {
        dequant_fp16_to_fp32(scratch, src.tok_embd, (size_t)src.vocab * src.n_embd);
        if (!sp_weights_set_tok_embed(out, scratch.data())) return false;
        ++n_set_bypass;  // tok_embd is BYPASS per Phase 1.7 policy
    } else { ++n_missing; }
    if (src.lm_head) {
        dequant_fp16_to_fp32(scratch, src.lm_head, (size_t)src.vocab * src.n_embd);
        if (!sp_weights_set_lm_head(out, scratch.data())) return false;
        ++n_set_bypass;
    } else if (src.tok_embd) {
        // Tied LM head — reuse the tok_embd buffer.
        dequant_fp16_to_fp32(scratch, src.tok_embd, (size_t)src.vocab * src.n_embd);
        if (!sp_weights_set_lm_head(out, scratch.data())) return false;
        ++n_set_bypass;
    }

    if (src.final_norm) {
        if (!sp_weights_set_final_norm(out, src.final_norm)) return false;
    }

    // --- Per-layer ---
    for (int L = 0; L < src.n_layers; ++L) {
        const auto& lyr = src.layers[L];

        if (lyr.wq) {
            dequant_fp16_to_fp32(scratch, lyr.wq, d_q * src.n_embd);
            if (!sp_weights_set_wq(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.wk) {
            dequant_fp16_to_fp32(scratch, lyr.wk, d_kv * src.n_embd);
            if (!sp_weights_set_wk(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.wv) {
            dequant_fp16_to_fp32(scratch, lyr.wv, d_kv * src.n_embd);
            if (!sp_weights_set_wv(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.wo) {
            dequant_fp16_to_fp32(scratch, lyr.wo, (size_t)src.n_embd * d_q);
            if (!sp_weights_set_wo(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.ffn_gate) {
            dequant_fp16_to_fp32(scratch, lyr.ffn_gate, (size_t)src.d_ff * src.n_embd);
            if (!sp_weights_set_ffn_gate(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.ffn_up) {
            dequant_fp16_to_fp32(scratch, lyr.ffn_up, (size_t)src.d_ff * src.n_embd);
            if (!sp_weights_set_ffn_up(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.ffn_down) {
            dequant_fp16_to_fp32(scratch, lyr.ffn_down, (size_t)src.n_embd * src.d_ff);
            if (!sp_weights_set_ffn_down(out, L, scratch.data())) return false;
            ++n_set_shim;
        } else { ++n_missing; }

        if (lyr.attn_norm) {
            sp_weights_set_attn_norm(out, L, lyr.attn_norm);
        } else { ++n_missing; }

        if (lyr.ffn_norm) {
            sp_weights_set_ffn_norm(out, L, lyr.ffn_norm);
        } else { ++n_missing; }
    }

    // --- Apply Frobenius shim per cfg ---
    int n_shimmed = 0;
    if (cfg.frobenius_quant || cfg.sato_tate_mix) {
        n_shimmed = sp_weights_apply_frobenius_shim(
            out,
            cfg.frobenius_quant, cfg.sato_tate_mix,
            cfg.frobenius_p,     cfg.frobenius_k,
            cfg.st_p1,           cfg.st_k1,
            cfg.st_p2,           cfg.st_k2);
    }

    const char* mode = cfg.sato_tate_mix    ? "sato-tate-mix"
                       : cfg.frobenius_quant ? "frobenius-quant"
                                              : "no-shim";
    std::fprintf(stderr,
        "[sp-weights-loader] mode=%s  shim-set=%d  bypass-set=%d  "
        "missing=%d  shimmed=%d\n",
        mode, n_set_shim, n_set_bypass, n_missing, n_shimmed);
    return true;
}

// =========================================================================
// LlamaWeights → fp16_source extractor.
// =========================================================================

// Extract fp16 data + dims from a ggml_tensor. Returns nullptr (and logs)
// if the tensor isn't fp16.
static const uint16_t* tensor_fp16(const ggml_tensor* t, const char* slot) {
    if (t == nullptr) return nullptr;
    if (t->type != GGML_TYPE_F16) {
        std::fprintf(stderr,
            "[sp-weights-loader] %s: SKIP non-fp16 (type=%d)\n",
            slot, (int)t->type);
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(t->data);
}

// Norms in GGUF are typically fp32 already. If they're fp16, dequant.
static const float* tensor_fp32_or_dequant(const ggml_tensor* t,
                                            std::vector<float>& scratch_pool,
                                            size_t expected_numel,
                                            const char* slot) {
    if (t == nullptr) return nullptr;
    if (t->type == GGML_TYPE_F32) {
        if ((size_t)ggml_nelements(t) != expected_numel) {
            std::fprintf(stderr, "[sp-weights-loader] %s: shape mismatch\n", slot);
            return nullptr;
        }
        return reinterpret_cast<const float*>(t->data);
    }
    if (t->type == GGML_TYPE_F16) {
        scratch_pool.resize(expected_numel);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(t->data);
        for (size_t i = 0; i < expected_numel; ++i) {
            scratch_pool[i] = fp16_to_fp32(src[i]);
        }
        return scratch_pool.data();
    }
    std::fprintf(stderr, "[sp-weights-loader] %s: norm has unsupported type %d\n",
                 slot, (int)t->type);
    return nullptr;
}

bool sp_weights_load_from_llama(sp_weights& out,
                                  const LlamaWeights& weights,
                                  const Config& cfg,
                                  int64_t scale_recip) {
    const auto& layers = weights.layers();
    const int n_layers = (int)layers.size();
    if (n_layers <= 0) {
        std::fprintf(stderr, "[sp-weights-loader] LlamaWeights has 0 layers\n");
        return false;
    }
    if (weights.tok_embd == nullptr) {
        std::fprintf(stderr, "[sp-weights-loader] missing tok_embd\n");
        return false;
    }

    // Recover dims from tok_embd: ne = [n_embd, vocab]
    const ggml_tensor* te = weights.tok_embd;
    const int n_embd = (int)te->ne[0];
    const int vocab  = (int)te->ne[1];

    // Recover n_head / n_kv_head / d_ff from layer 0.
    const auto& l0 = layers[0];
    if (l0.wq == nullptr || l0.wk == nullptr || l0.wv == nullptr ||
        l0.wo == nullptr) {
        std::fprintf(stderr,
            "[sp-weights-loader] layer 0 missing attn projections "
            "(this loader only supports STANDARD layers in Phase 2.2c)\n");
        return false;
    }
    if (l0.ffn_gate == nullptr || l0.ffn_up == nullptr || l0.ffn_down == nullptr) {
        std::fprintf(stderr,
            "[sp-weights-loader] layer 0 missing dense FFN tensors "
            "(MoE / packed-FFN models will land in Phase 2.2c2)\n");
        return false;
    }
    const int d_q     = (int)l0.wq->ne[1];
    const int d_kv    = (int)l0.wk->ne[1];
    if (d_q <= 0 || d_kv <= 0 || d_q % d_kv != 0) {
        std::fprintf(stderr,
            "[sp-weights-loader] bad d_q=%d / d_kv=%d\n", d_q, d_kv);
        return false;
    }
    const int n_head    = d_q  / (d_q  / (d_kv / 1));  // need head_dim
    // Simpler: assume head_dim derived from n_embd / n_head, and n_head
    // chosen such that d_q = n_head * head_dim. Without GGUF metadata
    // we can't recover n_head directly. Phase 2.2c assumes head_dim
    // divides n_embd and equals d_q's row stride / n_head. Use the
    // conservative fallback head_dim = 64 unless caller overrides via
    // Config (TODO: expose n_head + head_dim in Config).
    int head_dim_guess = 64;
    while (head_dim_guess > 1 &&
           (n_embd % head_dim_guess != 0 ||
            d_q    % head_dim_guess != 0 ||
            d_kv   % head_dim_guess != 0)) {
        head_dim_guess /= 2;
    }
    const int head_dim = head_dim_guess;
    const int n_head_use   = d_q  / head_dim;
    const int n_kv_head_use = d_kv / head_dim;
    const int d_ff         = (int)l0.ffn_gate->ne[1];

    // Cross-check: n_embd % n_head_use must be 0 and == head_dim.
    if (n_head_use <= 0 || n_kv_head_use <= 0 ||
        n_embd != n_head_use * head_dim) {
        std::fprintf(stderr,
            "[sp-weights-loader] dim derivation failed: "
            "n_embd=%d n_head=%d head_dim=%d d_q=%d d_kv=%d\n",
            n_embd, n_head_use, head_dim, d_q, d_kv);
        return false;
    }
    (void)n_head;  // silence unused

    // Build the layer sources.
    std::vector<sp_weights_layer_fp16_source> layer_srcs(n_layers);
    // Bypass-list norms are fp32 — dequant into per-tensor scratch pools
    // owned by this function so the pointers stay valid through
    // sp_weights_load_from_fp16_source.
    std::vector<std::vector<float>> norm_scratch(n_layers * 2);
    std::vector<float> final_norm_scratch;

    for (int L = 0; L < n_layers; ++L) {
        const auto& lyr = layers[L];
        if (lyr.kind != LlamaLayerKind::STANDARD) {
            std::fprintf(stderr,
                "[sp-weights-loader] layer %d is non-STANDARD "
                "(kind=%d) — Phase 2.2c2 will add MoE/GDN support\n",
                L, (int)lyr.kind);
            return false;
        }
        auto& s = layer_srcs[L];
        s.wq       = tensor_fp16(lyr.wq,       "wq");
        s.wk       = tensor_fp16(lyr.wk,       "wk");
        s.wv       = tensor_fp16(lyr.wv,       "wv");
        s.wo       = tensor_fp16(lyr.wo,       "wo");
        s.ffn_gate = tensor_fp16(lyr.ffn_gate, "ffn_gate");
        s.ffn_up   = tensor_fp16(lyr.ffn_up,   "ffn_up");
        s.ffn_down = tensor_fp16(lyr.ffn_down, "ffn_down");
        s.attn_norm = tensor_fp32_or_dequant(
            lyr.attn_norm, norm_scratch[L * 2 + 0], n_embd, "attn_norm");
        s.ffn_norm  = tensor_fp32_or_dequant(
            lyr.ffn_norm,  norm_scratch[L * 2 + 1], n_embd, "ffn_norm");
    }

    sp_weights_fp16_source src;
    src.n_layers  = n_layers;
    src.n_embd    = n_embd;
    src.n_head    = n_head_use;
    src.n_kv_head = n_kv_head_use;
    src.d_ff      = d_ff;
    src.vocab     = vocab;
    src.tok_embd  = tensor_fp16(weights.tok_embd, "tok_embd");
    src.lm_head   = tensor_fp16(weights.output,   "lm_head");
    src.final_norm = tensor_fp32_or_dequant(
        weights.output_norm, final_norm_scratch, n_embd, "final_norm");
    src.layers    = layer_srcs.data();

    return sp_weights_load_from_fp16_source(out, src, cfg, scale_recip);
}

}  // namespace sp::engine
