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

    // For Gemma3 et al, head_dim is NOT n_embd / n_head — we pass it
    // through the source. Default 0 = use n_embd / n_head.
    const int head_dim_from_src = src.head_dim_override;

    // Diagnostic — estimate the arena size before requesting it so a
    // huge model doesn't OOM silently.
    {
        const int hd = (head_dim_from_src > 0)
            ? head_dim_from_src : (src.n_embd / std::max(src.n_head, 1));
        const int64_t d_q  = (int64_t)src.n_head    * hd;
        const int64_t d_kv = (int64_t)src.n_kv_head * hd;
        const int64_t per_layer_elems =
              (int64_t)src.n_embd * d_q       // wq
            + (int64_t)src.n_embd * d_kv      // wk
            + (int64_t)src.n_embd * d_kv      // wv
            + d_q  * (int64_t)src.n_embd      // wo
            + (int64_t)src.n_embd * src.d_ff  // ffn_gate
            + (int64_t)src.n_embd * src.d_ff  // ffn_up
            + (int64_t)src.d_ff  * src.n_embd;// ffn_down
        const int64_t total_elems = per_layer_elems * src.n_layers;
        const double ok_bytes_gb =
            (double)total_elems * 16.0 / (1024.0 * 1024.0 * 1024.0);
        // Phase 2.3b iter 5: tok_embd + lm_head live as fp32 vectors
        // outside the O_K arena. Compute their footprint separately.
        const int64_t bypass_fp32_bytes =
            (int64_t)src.n_embd * src.vocab * 4 * 2;
        const double bypass_gb =
            (double)bypass_fp32_bytes / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "[sp-weights-loader] dims: n_layers=%d n_embd=%d n_head=%d "
            "n_kv_head=%d head_dim=%d d_ff=%d vocab=%d  ok_arena=%.2f GB "
            "(%lld elems @ 16 B)  bypass_fp32=%.2f GB  total=%.2f GB\n",
            src.n_layers, src.n_embd, src.n_head, src.n_kv_head,
            hd, src.d_ff, src.vocab,
            ok_bytes_gb, (long long)total_elems,
            bypass_gb, ok_bytes_gb + bypass_gb);
        const double bytes_gb = ok_bytes_gb + bypass_gb;
        if (bytes_gb > 24.0) {
            std::fprintf(stderr,
                "[sp-weights-loader] WARN: arena_estimate > 24 GB. "
                "The O_K AoS layout (16 B per element) makes large models "
                "expensive; consider running on a smaller variant or "
                "implementing lazy on-the-fly O_K encoding.\n");
        }
    }

    try {
        if (!sp_weights_alloc(out, src.n_layers, src.n_embd, src.n_head,
                                src.n_kv_head, src.d_ff, src.vocab,
                                scale_recip, head_dim_from_src)) {
            std::fprintf(stderr, "[sp-weights-loader] alloc failed\n");
            return false;
        }
    } catch (const std::bad_alloc& e) {
        std::fprintf(stderr,
            "[sp-weights-loader] arena malloc FAILED — out of memory. "
            "%s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[sp-weights-loader] alloc threw exception: %s\n", e.what());
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

        // Phase 2.3b: optional norms.
        if (lyr.attn_q_norm)    sp_weights_set_attn_q_norm(out, L, lyr.attn_q_norm);
        if (lyr.attn_k_norm)    sp_weights_set_attn_k_norm(out, L, lyr.attn_k_norm);
        if (lyr.attn_post_norm) sp_weights_set_attn_post_norm(out, L, lyr.attn_post_norm);
        if (lyr.ffn_post_norm)  sp_weights_set_ffn_post_norm(out, L, lyr.ffn_post_norm);
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

    // --- Phase 12 Step B-2: resident packed-int8 storage per --frobenius-q8.
    // Runs AFTER the Frobenius shim so we pack the post-phi^k coordinates.
    // Releases the original layer arenas back to the OS (memory win
    // appears in the next process metric, not just inside this function).
    if (cfg.frobenius_q4) {
        if (!(cfg.frobenius_quant || cfg.sato_tate_mix)) {
            std::fprintf(stderr,
                "[sp-weights-loader] WARNING: --frobenius-q4 set without "
                "--frobenius-quant or --sato-tate-mix; packing raw "
                "(a, 0) weights -- compression still works but the "
                "Theorem 4 cancellation hasn't been applied.\n");
        }
        int n_packed = sp_weights_convert_to_q4(out, cfg.frobenius_q4_prune);
        std::fprintf(stderr,
            "[sp-weights-loader] mode=%s + Q4  packed=%d tensors  "
            "(use_q4=%d, prune=%llu)\n",
            mode, n_packed, out.use_q4 ? 1 : 0,
            (unsigned long long)cfg.frobenius_q4_prune);
    } else if (cfg.frobenius_q8) {
        if (!(cfg.frobenius_quant || cfg.sato_tate_mix)) {
            std::fprintf(stderr,
                "[sp-weights-loader] WARNING: --frobenius-q8 set without "
                "--frobenius-quant or --sato-tate-mix; packing raw "
                "(a, 0) weights -- compression still works but the "
                "Theorem 4 cancellation hasn't been applied.\n");
        }
        int n_packed = sp_weights_convert_to_q8(out);
        std::fprintf(stderr,
            "[sp-weights-loader] mode=%s + Q8  packed=%d tensors  "
            "(use_q8=%d)\n",
            mode, n_packed, out.use_q8 ? 1 : 0);
    }

    return true;
}

// =========================================================================
// LlamaWeights → fp16_source extractor.
// =========================================================================

/* IEEE 754 fp32 -> fp16 round-to-zero. Matches the helper in
 * sp_ok_encode.cpp. */
static inline uint16_t spwl_fp32_to_fp16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}

/* fp16 -> fp32 helper, used by the Q8_0/Q4_0 dequant paths. */
static inline float spwl_fp16_to_fp32(uint16_t h) {
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

// Phase 15: fp16 if available, else Q8_0/Q4_0 dequant into scratch.
// For embeddings and lm_head from Q8_0 / Q4_0 GGUFs. Returns a pointer
// into scratch_pool with the fp16 view; pool is sized to numel uint16_t.
static const uint16_t* tensor_fp16_or_dequant_to_fp16(
    const ggml_tensor*       t,
    std::vector<uint16_t>&   scratch_pool,
    const char*              slot)
{
    if (t == nullptr) return nullptr;
    if (t->type == GGML_TYPE_F16) {
        return reinterpret_cast<const uint16_t*>(t->data);
    }
    const size_t numel = (size_t)ggml_nelements(t);
    scratch_pool.resize(numel);

    if (t->type == GGML_TYPE_Q8_0) {
        if ((numel % SP_OK_BLOCK_SIZE) != 0) {
            std::fprintf(stderr,
                "[sp-weights-loader] %s: Q8_0 numel %zu not multiple of 32\n",
                slot, numel);
            return nullptr;
        }
        const size_t n_blocks = numel / SP_OK_BLOCK_SIZE;
        const sp_gguf_block_q8_0* src =
            reinterpret_cast<const sp_gguf_block_q8_0*>(t->data);
        for (size_t b = 0; b < n_blocks; ++b) {
            const float s = spwl_fp16_to_fp32(src[b].d);
            for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
                const float v = s * (float)src[b].qs[k];
                scratch_pool[b * SP_OK_BLOCK_SIZE + k] = spwl_fp32_to_fp16(v);
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q8_0 -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    if (t->type == GGML_TYPE_Q4_0) {
        if ((numel % SP_OK_BLOCK_SIZE) != 0) {
            std::fprintf(stderr,
                "[sp-weights-loader] %s: Q4_0 numel %zu not multiple of 32\n",
                slot, numel);
            return nullptr;
        }
        const size_t n_blocks = numel / SP_OK_BLOCK_SIZE;
        const sp_gguf_block_q4_0* src =
            reinterpret_cast<const sp_gguf_block_q4_0*>(t->data);
        for (size_t b = 0; b < n_blocks; ++b) {
            const float s = spwl_fp16_to_fp32(src[b].d);
            for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
                const int8_t code = sp_ok_block_q4_decode_codepoint(
                    src[b].qs, k);
                const float v = s * (float)code;
                scratch_pool[b * SP_OK_BLOCK_SIZE + k] = spwl_fp32_to_fp16(v);
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q4_0 -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    /* Phase 15c: Q4_K dequant for bypass-list (tok_embd / lm_head). */
    if (t->type == GGML_TYPE_Q4_K) {
        if ((numel % SP_OK_Q4_K_SUPER) != 0) {
            std::fprintf(stderr,
                "[sp-weights-loader] %s: Q4_K numel %zu not multiple of 256\n",
                slot, numel);
            return nullptr;
        }
        const size_t n_super = numel / SP_OK_Q4_K_SUPER;
        const sp_gguf_block_q4_K* src =
            reinterpret_cast<const sp_gguf_block_q4_K*>(t->data);
        /* Inline get_scale_min_k4 to avoid a math-submodule cross-link. */
        auto get_sm = [](int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
            if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
            else {
                sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                m  = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
            }
        };
        for (size_t sb = 0; sb < n_super; ++sb) {
            const float d    = spwl_fp16_to_fp32(src[sb].d);
            const float dmin = spwl_fp16_to_fp32(src[sb].dmin);
            for (int s = 0; s < SP_OK_Q4_K_SUBBLOCKS; ++s) {
                const int group = s / 2;
                const int is_hi = (s & 1);
                const uint8_t* bs = src[sb].qs + group * 32;
                uint8_t sc6, m6;
                get_sm(s, src[sb].scales, sc6, m6);
                const float d_sub = d    * (float)sc6;
                const float m_sub = dmin * (float)m6;
                const size_t dst_off = sb * SP_OK_Q4_K_SUPER
                                     + (size_t)s * SP_OK_BLOCK_SIZE;
                for (int i = 0; i < SP_OK_BLOCK_SIZE; ++i) {
                    /* sub_A uses low nybbles of bs[0..31]; sub_B uses high nybbles. */
                    const uint8_t nyb = is_hi
                        ? (uint8_t)(bs[i] >> 4)
                        : (uint8_t)(bs[i] & 0x0F);
                    const float v = d_sub * (float)nyb - m_sub;
                    scratch_pool[dst_off + i] = spwl_fp32_to_fp16(v);
                }
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q4_K -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    /* Phase 15d: Q5_0 / Q5_1 / Q6_K dequant for both bypass-list AND
     * weight tensors. Q4_K_M models use these heavily; without these
     * paths most of a Q4_K_M's weights are unreachable. */
    if (t->type == 6 /* GGML_TYPE_Q5_0 */) {
        if ((numel % 32) != 0) return nullptr;
        const size_t n_blocks = numel / 32;
        struct gguf_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
        const gguf_q5_0* src = reinterpret_cast<const gguf_q5_0*>(t->data);
        for (size_t b = 0; b < n_blocks; ++b) {
            const float d = spwl_fp16_to_fp32(src[b].d);
            uint32_t qh; std::memcpy(&qh, src[b].qh, 4);
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
                const int32_t x0 = ((src[b].qs[j] & 0x0F) | xh_0) - 16;
                const int32_t x1 = ((src[b].qs[j] >>   4) | xh_1) - 16;
                scratch_pool[b * 32 + j     ] = spwl_fp32_to_fp16((float)x0 * d);
                scratch_pool[b * 32 + j + 16] = spwl_fp32_to_fp16((float)x1 * d);
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q5_0 -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    if (t->type == 7 /* GGML_TYPE_Q5_1 */) {
        if ((numel % 32) != 0) return nullptr;
        const size_t n_blocks = numel / 32;
        struct gguf_q5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
        const gguf_q5_1* src = reinterpret_cast<const gguf_q5_1*>(t->data);
        for (size_t b = 0; b < n_blocks; ++b) {
            const float d = spwl_fp16_to_fp32(src[b].d);
            const float m = spwl_fp16_to_fp32(src[b].m);
            uint32_t qh; std::memcpy(&qh, src[b].qh, 4);
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
                const int x0 = (src[b].qs[j] & 0x0F) | xh_0;
                const int x1 = (src[b].qs[j] >>   4) | xh_1;
                scratch_pool[b * 32 + j     ] = spwl_fp32_to_fp16((float)x0 * d + m);
                scratch_pool[b * 32 + j + 16] = spwl_fp32_to_fp16((float)x1 * d + m);
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q5_1 -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    if (t->type == 14 /* GGML_TYPE_Q6_K */) {
        /* QK_K = 256. Each super-block has ql[128] + qh[64] + scales[16] + d */
        if ((numel % 256) != 0) return nullptr;
        const size_t n_super = numel / 256;
        struct gguf_q6_K {
            uint8_t ql[128]; uint8_t qh[64];
            int8_t scales[16]; uint16_t d;
        };
        const gguf_q6_K* src = reinterpret_cast<const gguf_q6_K*>(t->data);
        for (size_t sb = 0; sb < n_super; ++sb) {
            const float d = spwl_fp16_to_fp32(src[sb].d);
            const uint8_t* ql_base = src[sb].ql;
            const uint8_t* qh_base = src[sb].qh;
            const int8_t*  sc_base = src[sb].scales;
            uint16_t* dst = scratch_pool.data() + sb * 256;
            for (int n = 0; n < 256; n += 128) {
                const uint8_t* ql = ql_base + (n / 128) * 64;
                const uint8_t* qh = qh_base + (n / 128) * 32;
                const int8_t*  sc = sc_base + (n / 128) * 8;
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    dst[n + l +  0] = spwl_fp32_to_fp16(d * (float)sc[is + 0] * (float)q1);
                    dst[n + l + 32] = spwl_fp32_to_fp16(d * (float)sc[is + 2] * (float)q2);
                    dst[n + l + 64] = spwl_fp32_to_fp16(d * (float)sc[is + 4] * (float)q3);
                    dst[n + l + 96] = spwl_fp32_to_fp16(d * (float)sc[is + 6] * (float)q4);
                }
            }
        }
        std::fprintf(stderr,
            "[sp-weights-loader] %s: dequanted Q6_K -> fp16 (%zu elems)\n",
            slot, numel);
        return scratch_pool.data();
    }

    std::fprintf(stderr,
        "[sp-weights-loader] %s: unsupported type %d for fp16-or-dequant\n",
        slot, (int)t->type);
    return nullptr;
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
                                  int n_head, int n_kv_head, int head_dim,
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

    // Dims from caller; cross-check against tok_embd / layer 0 tensors.
    const ggml_tensor* te = weights.tok_embd;
    const int n_embd = (int)te->ne[0];
    const int vocab  = (int)te->ne[1];

    if (n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) {
        std::fprintf(stderr,
            "[sp-weights-loader] bad caller dims: n_head=%d n_kv_head=%d head_dim=%d\n",
            n_head, n_kv_head, head_dim);
        return false;
    }
    // NOTE: head_dim is INDEPENDENT of n_embd / n_head for some archs
    // (e.g. Gemma3 has n_embd=640, n_head=4, head_dim=256 → d_q=1024 > n_embd).
    // The tensor-shape checks below catch any actual mismatches.

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
    const int d_q_check  = (int)l0.wq->ne[1];
    const int d_kv_check = (int)l0.wk->ne[1];
    if (d_q_check != n_head * head_dim || d_kv_check != n_kv_head * head_dim) {
        std::fprintf(stderr,
            "[sp-weights-loader] tensor-shape / dim mismatch: "
            "wq.ne[1]=%d (expected %d), wk.ne[1]=%d (expected %d)\n",
            d_q_check, n_head * head_dim, d_kv_check, n_kv_head * head_dim);
        return false;
    }
    const int n_head_use    = n_head;
    const int n_kv_head_use = n_kv_head;
    const int d_ff          = (int)l0.ffn_gate->ne[1];

    // Build the layer sources.
    std::vector<sp_weights_layer_fp16_source> layer_srcs(n_layers);
    /* Phase 15d: weight tensors can be Q5_0 / Q5_1 / Q6_K — types the
     * standard fp16 walker doesn't understand. Pre-allocate per-layer
     * per-slot scratch pools that outlive the load_from_fp16_source
     * call below. 7 slots per layer (wq, wk, wv, wo, ffn_gate, ffn_up,
     * ffn_down). For fp16 tensors the scratch stays empty (pass-through). */
    std::vector<std::vector<uint16_t>> weight_scratch(n_layers * 7);
    // Bypass-list norms are fp32 — dequant into per-tensor scratch pools
    // owned by this function so the pointers stay valid through
    // sp_weights_load_from_fp16_source. 6 slots per layer:
    //   [0]=attn_norm [1]=ffn_norm [2]=q_norm [3]=k_norm
    //   [4]=attn_post_norm [5]=ffn_post_norm
    std::vector<std::vector<float>> norm_scratch(n_layers * 6);
    std::vector<float> final_norm_scratch;
    int n_q_norm_seen = 0, n_k_norm_seen = 0;
    int n_attn_post_seen = 0, n_ffn_post_seen = 0;

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
        s.wq       = tensor_fp16_or_dequant_to_fp16(lyr.wq,       weight_scratch[L*7+0], "wq");
        s.wk       = tensor_fp16_or_dequant_to_fp16(lyr.wk,       weight_scratch[L*7+1], "wk");
        s.wv       = tensor_fp16_or_dequant_to_fp16(lyr.wv,       weight_scratch[L*7+2], "wv");
        s.wo       = tensor_fp16_or_dequant_to_fp16(lyr.wo,       weight_scratch[L*7+3], "wo");
        s.ffn_gate = tensor_fp16_or_dequant_to_fp16(lyr.ffn_gate, weight_scratch[L*7+4], "ffn_gate");
        s.ffn_up   = tensor_fp16_or_dequant_to_fp16(lyr.ffn_up,   weight_scratch[L*7+5], "ffn_up");
        s.ffn_down = tensor_fp16_or_dequant_to_fp16(lyr.ffn_down, weight_scratch[L*7+6], "ffn_down");
        s.attn_norm = tensor_fp32_or_dequant(
            lyr.attn_norm, norm_scratch[L * 6 + 0], n_embd, "attn_norm");
        s.ffn_norm  = tensor_fp32_or_dequant(
            lyr.ffn_norm,  norm_scratch[L * 6 + 1], n_embd, "ffn_norm");
        // Phase 2.3b: optional norms. attn_q_norm / attn_k_norm are
        // [head_dim]-sized; sandwich norms are [n_embd]-sized.
        if (lyr.attn_q_norm) {
            s.attn_q_norm = tensor_fp32_or_dequant(
                lyr.attn_q_norm, norm_scratch[L * 6 + 2], head_dim, "attn_q_norm");
            if (s.attn_q_norm) ++n_q_norm_seen;
        }
        if (lyr.attn_k_norm) {
            s.attn_k_norm = tensor_fp32_or_dequant(
                lyr.attn_k_norm, norm_scratch[L * 6 + 3], head_dim, "attn_k_norm");
            if (s.attn_k_norm) ++n_k_norm_seen;
        }
        if (lyr.attn_post_norm) {
            s.attn_post_norm = tensor_fp32_or_dequant(
                lyr.attn_post_norm, norm_scratch[L * 6 + 4], n_embd, "attn_post_norm");
            if (s.attn_post_norm) ++n_attn_post_seen;
        }
        if (lyr.ffn_post_norm) {
            s.ffn_post_norm = tensor_fp32_or_dequant(
                lyr.ffn_post_norm, norm_scratch[L * 6 + 5], n_embd, "ffn_post_norm");
            if (s.ffn_post_norm) ++n_ffn_post_seen;
        }
    }
    if (n_q_norm_seen + n_k_norm_seen + n_attn_post_seen + n_ffn_post_seen > 0) {
        std::fprintf(stderr,
            "[sp-weights-loader] Phase 2.3b norms: q_norm=%d k_norm=%d "
            "attn_post=%d ffn_post=%d (per-layer counts)\n",
            n_q_norm_seen, n_k_norm_seen, n_attn_post_seen, n_ffn_post_seen);
    }

    sp_weights_fp16_source src;
    src.n_layers  = n_layers;
    src.n_embd    = n_embd;
    src.n_head    = n_head_use;
    src.n_kv_head = n_kv_head_use;
    src.d_ff      = d_ff;
    src.vocab     = vocab;
    src.head_dim_override = head_dim;
    /* Phase 15: tok_embd / lm_head are Q8_0 in pre-quantised GGUFs.
     * The dequant helper dispatches on type: fp16 -> pass-through;
     * Q8_0 / Q4_0 -> dequant into a per-call scratch pool. The scratch
     * pools must outlive sp_weights_load_from_fp16_source below, so
     * they're stack-locals scoped here. */
    std::vector<uint16_t> tok_embd_scratch;
    std::vector<uint16_t> lm_head_scratch;
    src.tok_embd  = tensor_fp16_or_dequant_to_fp16(
        weights.tok_embd, tok_embd_scratch, "tok_embd");
    src.lm_head   = tensor_fp16_or_dequant_to_fp16(
        weights.output,   lm_head_scratch,  "lm_head");
    src.final_norm = tensor_fp32_or_dequant(
        weights.output_norm, final_norm_scratch, n_embd, "final_norm");
    src.layers    = layer_srcs.data();

    if (!sp_weights_load_from_fp16_source(out, src, cfg, scale_recip)) {
        return false;
    }

    /* Phase 15: --gguf-block-quant overlay. The fp16 path above already
     * loaded norms / embeddings and (when applicable) ran the Frobenius
     * shim on the per-tensor sp_ok_t weights. The ingest now walks the
     * LlamaWeights again, picks tensors with type Q8_0 / Q4_0, and
     * overlays the block-fused storage on top — nulling the original
     * sp_ok_tensor data ptrs so the forward dispatch routes through the
     * new block_q{8,4} kernels. */
    if (cfg.gguf_block_quant) {
        const int64_t p = cfg.frobenius_quant ? cfg.frobenius_p : (int64_t)41;
        const int64_t k = cfg.frobenius_quant ? cfg.frobenius_k : (int64_t)8;
        int n_fused = sp_weights_ingest_gguf_block_quant(
            out, weights, p, k, scale_recip);
        if (n_fused < 0) return false;
    }
    return true;
}

}  // namespace sp::engine
