// forward_native — Qwen2 layer step driven by sp_kernels_cpu.
// See forward_native.h for the contract.

#include "forward_native.h"

#include "kv_cache.h"          // KvCache — SP-banded compress/decompress
#include "sp_kernels_cpu.h"
#include "sp_quant.h"
#include "sp_tensor.h"

#ifdef SP_HEXAGON_FASTRPC
#include "shannon_prime_hexagon.h"   // sp_hexagon_cache_kq_matmul_fused
#endif

#include "ggml.h"   // type_traits.to_float for quant types we don't yet
                    // have native fused-matmul kernels for. Used at one
                    // call site (matmul_fp32_lhs fallback). NO ggml
                    // graph / scheduler / backend involvement.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sp::engine {

// ─────────────────────────────────────────────────────────────────
// Helper: allocate an [inner, outer] fp32 buffer from the arena.
// Returns pointer + 0 on success, nullptr on overflow.
// ─────────────────────────────────────────────────────────────────
static float* arena_f32(sp_arena& a, size_t n_elements) {
    return (float*)a.alloc(n_elements * sizeof(float), 64);
}

// ─────────────────────────────────────────────────────────────────
// Helper: dispatch a "fp32 lhs × W rhs" matmul where W's dtype is
// known via sp_dtype enum. Centralizes the per-dtype routing so the
// callers stay tidy.
//
//   lhs[m, k]  fp32
//   W[n, k]    in W_dtype layout (row-major over k inside each row)
//   out[m, n]  fp32
//
// On unsupported dtype, returns -1 (caller should bail).
// ─────────────────────────────────────────────────────────────────
// Map sp_dtype back to ggml_type for the type_traits fallback. This
// is the one place we accept a "soft" dependency on ggml's quant
// tables — purely for dequant math, no compute graph involvement.
static ggml_type sp_dtype_to_ggml(sp_dtype d) {
    switch (d) {
        case sp_dtype::F32:  return GGML_TYPE_F32;
        case sp_dtype::F16:  return GGML_TYPE_F16;
        case sp_dtype::Q4_K: return GGML_TYPE_Q4_K;
        case sp_dtype::Q5_K: return GGML_TYPE_Q5_K;
        case sp_dtype::Q6_K: return GGML_TYPE_Q6_K;
        case sp_dtype::Q8_0: return GGML_TYPE_Q8_0;
        default:             return GGML_TYPE_COUNT;  // sentinel
    }
}

static int matmul_fp32_lhs(const float* lhs, const void* W, sp_dtype W_dtype,
                            int m, int k, int n, float* out) {
    switch (W_dtype) {
        case sp_dtype::F32:
            sp_matmul_f32(lhs, (const float*)W, m, k, n, out);
            return 0;
        case sp_dtype::Q5_K:
            sp_matmul_f32_q5k(lhs, W, m, k, n, out);
            return 0;
        // F16, Q4_K, Q6_K, Q8_0 — fall through to ggml-dequant
        // fallback. Native fused kernels can be added per-quant later;
        // until then we trade some perf for correctness on any
        // GGUF-supported quant.
        default: {
            const ggml_type gt = sp_dtype_to_ggml(W_dtype);
            const struct ggml_type_traits* tt = ggml_get_type_traits(gt);
            if (!tt || !tt->to_float || gt == GGML_TYPE_COUNT) {
                std::fprintf(stderr,
                    "[sp_native] matmul: unsupported W dtype=%d (no fallback)\n",
                    (int)W_dtype);
                return -1;
            }
            // Dequant the entire weight matrix once, then fp32 matmul.
            // Slower than per-row L1 reuse (sp_matmul_f32_q5k pattern)
            // but trivially correct and fine until per-quant native
            // kernels land.
            std::vector<float> Wf((size_t)n * k);
            tt->to_float(W, Wf.data(), (int64_t)n * k);
            sp_matmul_f32(lhs, Wf.data(), m, k, n, out);
            return 0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Attention block
// ─────────────────────────────────────────────────────────────────
//
// Input  x: [n_embd, n_seq] fp32 (we treat n_seq as outer dim)
// Output:    [n_embd, n_seq] fp32 — the residual contribution
//
// Steps (no fused flash-attn — explicit per-head loop):
//   xn  = rms_norm(x, attn_norm)
//   Q   = wq @ xn  → [n_head*head_dim, n_seq]
//   K   = wk @ xn  → [n_head_kv*head_dim, n_seq]
//   V   = wv @ xn  → [n_head_kv*head_dim, n_seq]
//   apply optional q_norm/k_norm (Qwen3); RoPE on Q and K.
//   append K, V to KV cache at positions pos[0..n_seq-1].
//   for each head h:
//     read K_full = cache[h_kv][0..n_total_pos-1]
//     read V_full = cache[h_kv][0..n_total_pos-1]
//     scores = Q[h] @ K_full^T  → [n_seq, n_total_pos]
//     scores = softmax(scores * 1/sqrt(d_k) + causal_mask)
//     attn[h] = scores @ V_full  → [n_seq, head_dim]
//   out = wo @ concat_heads(attn)  → [n_embd, n_seq]
//
// All matmuls use sp_matmul_f32_q5k when the weight is Q5_K. Inner
// kernels run NEON; this driver is single-threaded.
int forward_native_attention(const ForwardNativeLayer&   layer,
                             const ForwardNativeHparams& hp,
                             const float*    x,
                             const int32_t*  pos,
                             int             n_seq,
                             ForwardNativeKv& kv,
                             sp_arena&       arena,
                             float*          out_residual) {
    const int n_embd     = hp.n_embd;
    const int n_head     = hp.n_head;
    const int n_head_kv  = hp.n_head_kv;
    const int head_dim   = hp.head_dim;
    const int n_q_dim    = n_head * head_dim;        // Q feature width
    const int n_kv_dim   = n_head_kv * head_dim;     // K/V feature width
    const int n_pos_past = kv.n_pos_past;
    const int n_pos_full = n_pos_past + n_seq;

    if (!kv.kv) {
        std::fprintf(stderr, "[sp_native] attn: kv.kv is null\n");
        return -1;
    }
    if (n_pos_full > kv.kv->max_seq()) {
        std::fprintf(stderr,
            "[sp_native] attn: kv overflow (have %d, need %d)\n",
            kv.kv->max_seq(), n_pos_full);
        return -1;
    }

    // ── attn_norm ─────────────────────────────────────────────────
    float* xn = arena_f32(arena, (size_t)n_embd * n_seq);
    if (!xn) return -2;
    sp_rms_norm_f32_rows(x, layer.attn_norm,
                         n_embd, n_seq, hp.rms_norm_eps, xn);

    // ── Q/K/V projections ────────────────────────────────────────
    // Layout convention for matmul:
    //   lhs[m=n_seq, k=n_embd]
    //   W[n=output_dim, k=n_embd]
    //   out[m=n_seq, n=output_dim]
    // sp_matmul takes lhs row-major, W row-major (each W row is one
    // output feature × n_embd inputs). Output is m row-major.
    float* Q = arena_f32(arena, (size_t)n_q_dim  * n_seq);
    float* K = arena_f32(arena, (size_t)n_kv_dim * n_seq);
    float* V = arena_f32(arena, (size_t)n_kv_dim * n_seq);
    if (!Q || !K || !V) return -2;

    if (matmul_fp32_lhs(xn, layer.wq, layer.wq_dtype,
                         n_seq, n_embd, n_q_dim,  Q) != 0) return -3;
    if (matmul_fp32_lhs(xn, layer.wk, layer.wk_dtype,
                         n_seq, n_embd, n_kv_dim, K) != 0) return -3;
    if (matmul_fp32_lhs(xn, layer.wv, layer.wv_dtype,
                         n_seq, n_embd, n_kv_dim, V) != 0) return -3;

    // Apply Q/K/V biases when present (Qwen2 has them; Qwen3 omits).
    sp_bias_add_f32_rows(Q, layer.bq, n_q_dim,  n_seq);
    sp_bias_add_f32_rows(K, layer.bk, n_kv_dim, n_seq);
    sp_bias_add_f32_rows(V, layer.bv, n_kv_dim, n_seq);

    // Q layout right now: [n_q_dim, n_seq] = [n_head*head_dim, n_seq]
    // For RoPE we need [head_dim, n_head, n_seq]. The memory is the
    // same — strides walk n_head*head_dim per seq position with
    // head_dim chunks per head — so we reuse Q in place, treating it
    // as a 3-d view.

    // ── Optional Q-norm / K-norm (Qwen3) ─────────────────────────
    // Per-head RMSNorm: each head's head_dim slice, scaled by the
    // attn_q_norm[head_dim] vector. Iterate (seq, head) pairs.
    if (layer.attn_q_norm) {
        for (int s = 0; s < n_seq; ++s) {
            for (int h = 0; h < n_head; ++h) {
                float* qh = Q + (size_t)s * n_q_dim + (size_t)h * head_dim;
                sp_rms_norm_f32(qh, layer.attn_q_norm,
                                head_dim, hp.rms_norm_eps, qh);
            }
        }
    }
    if (layer.attn_k_norm) {
        for (int s = 0; s < n_seq; ++s) {
            for (int h = 0; h < n_head_kv; ++h) {
                float* kh = K + (size_t)s * n_kv_dim + (size_t)h * head_dim;
                sp_rms_norm_f32(kh, layer.attn_k_norm,
                                head_dim, hp.rms_norm_eps, kh);
            }
        }
    }

    // ── RoPE on Q and K ──────────────────────────────────────────
    sp_rope_f32(Q, head_dim, n_head,    n_seq, pos,
                hp.n_rot, hp.rope_freq_base, hp.rope_freq_scale);
    sp_rope_f32(K, head_dim, n_head_kv, n_seq, pos,
                hp.n_rot, hp.rope_freq_base, hp.rope_freq_scale);

    // ── Append K/V to cache (SP-banded compression) ──────────────
    // K/V are laid out [n_seq, n_head_kv, head_dim] row-major — i.e.
    // K[(s * n_head_kv + h) * head_dim + d]. This matches KvCache::write's
    // K_flat layout exactly, so we hand the buffer through.
    //
    // Compression happens INSIDE write(): VHT2 → Möbius reorder → band
    // quantize. ~10× memory savings vs fp16 slabs. The per-layer state
    // (compressed bytes) lives in kv.kv (the engine's KvCache).
    //
    // Calibration pass: feed each K vector to calibrate_feed instead of
    // writing. Attention below will run on the local K/V buffer so the
    // cache never holds uncalibrated data. Caller does calibrate_end()
    // after the calibration pass, then re-runs prefill with the flag off.
    if (kv.calibrate_pass) {
        for (int s = 0; s < n_seq; ++s) {
            for (int h = 0; h < n_head_kv; ++h) {
                const float* k_src = K + (size_t)s * n_kv_dim
                                       + (size_t)h * head_dim;
                kv.kv->calibrate_feed(k_src);
            }
        }
    } else {
        if (!kv.kv->write(kv.layer_idx, n_pos_past, n_seq, K, V)) {
            std::fprintf(stderr,
                "[sp_native] attn: KvCache::write failed (layer=%d pos=%d n=%d)\n",
                kv.layer_idx, n_pos_past, n_seq);
            return -1;
        }
#ifdef SP_HEXAGON_FASTRPC
        // Mirror K into the hexagon cache so HVX kq_matmul_fused has
        // bytes to read against. The hexagon cache stores per-(layer,
        // head) so we gather strided per head. Buffer layout from
        // forward_native: K[(s * n_head_kv + h) * head_dim + d].
        if (kv.hex_cache) {
            sp_hexagon_cache_t* hex = (sp_hexagon_cache_t*)kv.hex_cache;
            std::vector<float> k_head((size_t)n_seq * head_dim);
            for (int h = 0; h < n_head_kv; ++h) {
                for (int s = 0; s < n_seq; ++s) {
                    const float* k_src = K + (size_t)s * n_kv_dim
                                           + (size_t)h * head_dim;
                    std::memcpy(k_head.data() + (size_t)s * head_dim,
                                k_src, (size_t)head_dim * sizeof(float));
                }
                sp_hexagon_cache_write_k_batch(hex, kv.layer_idx, h,
                                                n_pos_past, n_seq,
                                                k_head.data());
            }
        }
#endif
    }

    // ── Attention compute (per-head loop) ────────────────────────
    // Allocate per-head intermediates from the arena. Reused per head.
    float* attn_out = arena_f32(arena, (size_t)n_q_dim * n_seq);
    if (!attn_out) return -2;

    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int n_kv_total = n_pos_full;

    // Scratch for scores + mask. K_full/V_full are now lazily
    // materialised only on the calibration pass and the fused-path
    // fallback (sqfree/hier/GPU caches), so the common shadow-cache
    // case allocates 0 bytes for K_full/V_full and skips the full
    // [n_kv_total, head_dim, n_head_kv] decompress entirely.
    std::vector<float> scores((size_t)n_seq * n_kv_total);
    std::vector<float> mask_buf((size_t)n_seq * n_kv_total, 0.0f);
    std::vector<float> K_full;
    std::vector<float> V_full;
    std::vector<float> K_all_owned, V_all_owned;
    const float* K_all_ptr = nullptr;
    const float* V_all_ptr = nullptr;
    bool fallback_inited = false;
    auto ensure_fallback = [&]() {
        if (fallback_inited) return true;
        if (kv.calibrate_pass) {
            // Calibration pass: K/V from local just-computed buffers.
            // Layout already matches kv->read's [n_kv_total, n_head_kv,
            // head_dim] flat — n_pos_past was 0 so n_kv_total == n_seq.
            K_all_ptr = K;
            V_all_ptr = V;
        } else if (kv.kv->read(kv.layer_idx, n_kv_total,
                                K_all_owned, V_all_owned)) {
            K_all_ptr = K_all_owned.data();
            V_all_ptr = V_all_owned.data();
        } else {
            std::fprintf(stderr,
                "[sp_native] attn: KvCache::read failed (layer=%d kv_len=%d)\n",
                kv.layer_idx, n_kv_total);
            return false;
        }
        K_full.assign((size_t)n_kv_total * head_dim, 0.0f);
        V_full.assign((size_t)n_kv_total * head_dim, 0.0f);
        fallback_inited = true;
        return true;
    };

    // Causal mask: for each query position (n_pos_past + s), block
    // out future positions. Mask is +0 where allowed, -INF where
    // blocked, ADDED to scores before softmax.
    for (int s = 0; s < n_seq; ++s) {
        const int q_pos = n_pos_past + s;
        for (int kp = 0; kp < n_kv_total; ++kp) {
            mask_buf[(size_t)s * n_kv_total + kp] =
                (kp <= q_pos) ? 0.0f : -INFINITY;
        }
    }

    const int gqa_ratio = n_head / n_head_kv;

    for (int h = 0; h < n_head; ++h) {
        const int hk = h / gqa_ratio;     // shared-KV head index

        // Q for this head: [n_seq, head_dim] view into Q.
        // We need a contiguous Q_h buffer for the matmul.
        std::vector<float> Q_h((size_t)n_seq * head_dim);
        for (int s = 0; s < n_seq; ++s) {
            const float* q_src = Q
                + (size_t)s * n_q_dim
                + (size_t)h * head_dim;
            std::memcpy(Q_h.data() + (size_t)s * head_dim,
                        q_src, (size_t)head_dim * sizeof(float));
        }

        // ── KQ matmul: Phase 1.6 fused decompress+matmul (CPU) ──
        // Per-row decompress directly out of the shadow cache slot
        // (band_dequantize → mobius/var unreorder → VHT2 self-inverse)
        // → dot against every Q row. Compressed K bytes (~10× smaller
        // than fp32) stay in L2; ALU-bound rather than bandwidth-bound.
        // Same algorithm as llama_sp_fused_kq.cpp (the validated CPU
        // path that produced the 3.58× spec-decode win on S22U).
        // Skipped on the calibration pass since the cache hasn't been
        // written with the calibrated bytes yet — falls through to
        // sp_matmul_f32 against the local K.
        int kq_rc = -1;
        if (!kv.calibrate_pass && kv.kv) {
            if (kv.kv->kq_fused_cpu(kv.layer_idx, hk, n_kv_total,
                                    Q_h.data(), n_seq,
                                    scores.data())) {
                kq_rc = 0;
            }
        }
        if (kq_rc != 0) {
            // Fallback: materialise K_full from kv->read (or local K
            // on calibration), then run sp_matmul / kq_dispatch.
            if (!ensure_fallback()) return -1;
            for (int p = 0; p < n_kv_total; ++p) {
                const float* k_src = K_all_ptr
                    + (size_t)(p * n_head_kv + hk) * head_dim;
                std::memcpy(K_full.data() + (size_t)p * head_dim,
                            k_src, (size_t)head_dim * sizeof(float));
            }
            if (layer.kq_dispatch) {
                kq_rc = layer.kq_dispatch(layer.kq_dispatch_userdata,
                                           Q_h.data(), K_full.data(),
                                           n_seq, head_dim, n_kv_total,
                                           scores.data());
            }
            if (kq_rc != 0) {
                sp_matmul_f32(Q_h.data(), K_full.data(),
                              n_seq, head_dim, n_kv_total, scores.data());
            }
        }

        // Softmax with scale + mask, row-wise over n_kv_total.
        sp_softmax_f32_rows(scores.data(), mask_buf.data(),
                            n_kv_total, n_seq, scale, scores.data());

        // ── V dot: Phase 1.6 fused decompress+accumulate (CPU) ──
        // Symmetric to kq_fused_cpu. Per-row decompress V bytes from
        // the shadow cache slot, accumulate into attn_out scaled by
        // the softmax weights for every Q row before moving to the
        // next position. attn_out[s, h*hd + d] gets the result.
        // Skipped on calibration pass (uses local V_full instead).
        bool v_done = false;
        if (!kv.calibrate_pass && kv.kv) {
            // Zero this head's slice of attn_out before the
            // accumulating fused dot.
            for (int s = 0; s < n_seq; ++s) {
                std::memset(attn_out + (size_t)s * n_q_dim
                                     + (size_t)h * head_dim,
                            0, (size_t)head_dim * sizeof(float));
            }
            // v_dot_fused_cpu writes into a contiguous [n_q, head_dim]
            // buffer. Our attn_out is [n_seq, n_q_dim] strided, so we
            // use a small scratch and scatter-back. For the common
            // n_seq=1 (decode) case, scratch is the head's slice
            // directly — no scatter cost.
            if (n_seq == 1) {
                if (kv.kv->v_dot_fused_cpu(kv.layer_idx, hk, n_kv_total,
                                            scores.data(), 1,
                                            attn_out + (size_t)h * head_dim)) {
                    v_done = true;
                }
            } else {
                std::vector<float> attn_h((size_t)n_seq * head_dim, 0.0f);
                if (kv.kv->v_dot_fused_cpu(kv.layer_idx, hk, n_kv_total,
                                            scores.data(), n_seq,
                                            attn_h.data())) {
                    for (int s = 0; s < n_seq; ++s) {
                        std::memcpy(attn_out + (size_t)s * n_q_dim
                                              + (size_t)h * head_dim,
                                    attn_h.data() + (size_t)s * head_dim,
                                    (size_t)head_dim * sizeof(float));
                    }
                    v_done = true;
                }
            }
        }
        if (!v_done) {
            // Fallback: materialise V_full and run the manual dot.
            if (!ensure_fallback()) return -1;
            for (int p = 0; p < n_kv_total; ++p) {
                const float* v_src = V_all_ptr
                    + (size_t)(p * n_head_kv + hk) * head_dim;
                std::memcpy(V_full.data() + (size_t)p * head_dim,
                            v_src, (size_t)head_dim * sizeof(float));
            }
            for (int s = 0; s < n_seq; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int p = 0; p < n_kv_total; ++p) {
                        acc += (double)scores[(size_t)s * n_kv_total + p]
                             * (double)V_full[(size_t)p * head_dim + d];
                    }
                    attn_out[(size_t)s * n_q_dim
                           + (size_t)h * head_dim + d] = (float)acc;
                }
            }
        }
    }

    // ── wo projection ────────────────────────────────────────────
    // attn_out is [n_q_dim, n_seq] in our memory layout (each seq
    // contiguous over n_q_dim). wo: [n_embd, n_q_dim] → out[n_embd,
    // n_seq]. Same matmul shape as Q/K/V proj just with different
    // dims. lhs[n_seq, n_q_dim], W[n_embd, n_q_dim], out[n_seq, n_embd].
    if (matmul_fp32_lhs(attn_out, layer.wo, layer.wo_dtype,
                         n_seq, n_q_dim, n_embd, out_residual) != 0) return -3;
    sp_bias_add_f32_rows(out_residual, layer.bo, n_embd, n_seq);
    return 0;
}

// ─────────────────────────────────────────────────────────────────
// FFN block (Qwen2 SwiGLU)
// ─────────────────────────────────────────────────────────────────
//
// xn   = rms_norm(x, ffn_norm)
// gate = ffn_gate @ xn        → [n_ff, n_seq]
// up   = ffn_up   @ xn        → [n_ff, n_seq]
// h    = silu(gate) * up
// out  = ffn_down @ h         → [n_embd, n_seq]
int forward_native_ffn(const ForwardNativeLayer&   layer,
                       const ForwardNativeHparams& hp,
                       const float*  x,
                       int           n_seq,
                       sp_arena&     arena,
                       float*        out_residual) {
    const int n_embd = hp.n_embd;
    const int n_ff   = hp.n_ff;

    float* xn   = arena_f32(arena, (size_t)n_embd * n_seq);
    float* gate = arena_f32(arena, (size_t)n_ff   * n_seq);
    float* up   = arena_f32(arena, (size_t)n_ff   * n_seq);
    float* h    = arena_f32(arena, (size_t)n_ff   * n_seq);
    if (!xn || !gate || !up || !h) return -2;

    sp_rms_norm_f32_rows(x, layer.ffn_norm,
                         n_embd, n_seq, hp.rms_norm_eps, xn);

    if (matmul_fp32_lhs(xn, layer.ffn_gate, layer.ffn_gate_dtype,
                         n_seq, n_embd, n_ff, gate) != 0) return -3;
    if (matmul_fp32_lhs(xn, layer.ffn_up, layer.ffn_up_dtype,
                         n_seq, n_embd, n_ff, up) != 0) return -3;

    sp_silu_mul_f32(gate, up, n_seq * n_ff, h);

    if (matmul_fp32_lhs(h, layer.ffn_down, layer.ffn_down_dtype,
                         n_seq, n_ff, n_embd, out_residual) != 0) return -3;
    return 0;
}

// ─────────────────────────────────────────────────────────────────
// Full layer step: attention + FFN with residual adds.
// ─────────────────────────────────────────────────────────────────
int forward_native_layer(const ForwardNativeLayer&   layer,
                         const ForwardNativeHparams& hp,
                         const float*  x_in,
                         const int32_t* pos,
                         int           n_seq,
                         ForwardNativeKv& kv,
                         sp_arena&     arena,
                         float*        out) {
    const int n_embd = hp.n_embd;
    const size_t total = (size_t)n_embd * n_seq;

    // Mid buffer for x + attn_residual (= input to FFN).
    float* x_mid = arena_f32(arena, total);
    if (!x_mid) return -2;

    // Attention residual contribution.
    float* attn_res = arena_f32(arena, total);
    if (!attn_res) return -2;
    int rc = forward_native_attention(layer, hp, x_in, pos, n_seq,
                                       kv, arena, attn_res);
    if (rc != 0) return rc;

    // x_mid = x_in + attn_res
    for (size_t i = 0; i < total; ++i) x_mid[i] = x_in[i] + attn_res[i];

    // FFN residual contribution.
    float* ffn_res = arena_f32(arena, total);
    if (!ffn_res) return -2;
    rc = forward_native_ffn(layer, hp, x_mid, n_seq, arena, ffn_res);
    if (rc != 0) return rc;

    // out = x_mid + ffn_res
    for (size_t i = 0; i < total; ++i) out[i] = x_mid[i] + ffn_res[i];

    return 0;
}

}  // namespace sp::engine
