// ForwardNativeContext — see forward_native_context.h.

#include "forward_native_context.h"

#include "forward_native.h"
#include "gguf_loader.h"
#include "llama_weights.h"
#include "sp_kernels_cpu.h"
#include "sp_quant.h"
#include "sp_tensor.h"
#include "sp_threadpool.h"

#include "ggml.h"   // ONLY for reading ggml_tensor metadata + data ptr

#if defined(SP_ENGINE_WITH_QNN)
#include "sp_llama_qnn.h"   // QNN HTP dispatch wrapper (matmul cache)
#include "QnnTypes.h"       // QNN_DATATYPE_FLOAT_16
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sp::engine {

// ─────────────────────────────────────────────────────────────────
// ggml_tensor → sp_dtype
// ─────────────────────────────────────────────────────────────────
static sp_dtype to_sp_dtype(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return sp_dtype::F32;
        case GGML_TYPE_F16:  return sp_dtype::F16;
        case GGML_TYPE_Q4_K: return sp_dtype::Q4_K;
        case GGML_TYPE_Q5_K: return sp_dtype::Q5_K;
        case GGML_TYPE_Q6_K: return sp_dtype::Q6_K;
        case GGML_TYPE_Q8_0: return sp_dtype::Q8_0;
        default:             return sp_dtype::UNDEFINED;
    }
}

// ─────────────────────────────────────────────────────────────────
// Token embedding lookup. Reads row `id` from the embedding table
// into `out_fp32[n_embd]`.
//
// Native fast paths: F32 (memcpy), F16 (sp_fp16_to_fp32), Q5_K
// (sp_dequant_q5_K_to_f32 — bit-exact). For any other quant, we fall
// back to ggml's public type_traits.to_float — that's a one-shot
// per-token lookup so the perf cost is negligible. Native fast paths
// for Q4_K / Q6_K / Q8_0 added when we hit a benchmark that needs
// them.
// ─────────────────────────────────────────────────────────────────
static bool embed_lookup(const ggml_tensor* tok_embd,
                          int32_t id, int n_embd,
                          float* out_fp32) {
    if (!tok_embd || !tok_embd->data) return false;
    const sp_dtype dt = to_sp_dtype(tok_embd->type);
    const auto* row_bytes_base = (const uint8_t*)tok_embd->data;
    const size_t row_stride = tok_embd->nb[1];
    const uint8_t* row = row_bytes_base + (size_t)id * row_stride;

    switch (dt) {
        case sp_dtype::F32:
            std::memcpy(out_fp32, row, (size_t)n_embd * sizeof(float));
            return true;
        case sp_dtype::F16:
            for (int i = 0; i < n_embd; ++i) {
                out_fp32[i] = sp_fp16_to_fp32(((const uint16_t*)row)[i]);
            }
            return true;
        case sp_dtype::Q5_K: {
            const int n_blocks = n_embd / SP_QK_K;
            sp_dequant_q5_K_to_f32((const sp_block_q5_K*)row,
                                    out_fp32, (size_t)n_blocks);
            return true;
        }
        default: {
            // ggml-fallback for Q4_K / Q6_K / Q8_0 / etc. Public API
            // type_traits.to_float reads a flat byte buffer of one
            // row and writes n_embd fp32 values.
            const struct ggml_type_traits* tt =
                ggml_get_type_traits((ggml_type)tok_embd->type);
            if (!tt || !tt->to_float) {
                std::fprintf(stderr,
                    "[sp_native] embed_lookup: dtype %d has no to_float\n",
                    tok_embd->type);
                return false;
            }
            tt->to_float(row, out_fp32, (int64_t)n_embd);
            return true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// "Per-row gather" matmul for the LM head when output is quantized.
// Reuses sp_matmul_f32_q5k for Q5_K; fp32 / fp16 fallback otherwise.
// ─────────────────────────────────────────────────────────────────
// Universal "dequantize then fp32-matmul" fallback. Used for the LM
// head and for any layer weight whose dtype doesn't yet have a native
// fused dequant+matmul kernel. Dequants the entire weight matrix to
// fp32 once via ggml's public type_traits.to_float, then runs
// sp_matmul_f32. Slower than native fused per-row dequant (no L1
// reuse), but trivially correct for any GGUF quant ggml supports.
static int matmul_via_ggml_dequant(const ggml_tensor* W,
                                    const float* x,
                                    int m, int k, int n,
                                    float* out) {
    const struct ggml_type_traits* tt =
        ggml_get_type_traits((ggml_type)W->type);
    if (!tt || !tt->to_float) {
        std::fprintf(stderr,
            "[sp_native] matmul fallback: dtype %d has no to_float\n",
            W->type);
        return -1;
    }
    std::vector<float> Wf((size_t)n * k);
    tt->to_float(W->data, Wf.data(), (int64_t)n * k);
    sp_matmul_f32(x, Wf.data(), m, k, n, out);
    return 0;
}

static int lm_head_apply(const ggml_tensor* W,
                          const float* x,         // [m, n_embd]
                          int m, int n_embd, int n_vocab,
                          float* logits) {
    if (!W || !W->data) {
        std::fprintf(stderr, "[sp_native] lm_head: weight ptr null\n");
        return -1;
    }
    const sp_dtype dt = to_sp_dtype(W->type);
    switch (dt) {
        case sp_dtype::F32:
            sp_matmul_f32(x, (const float*)W->data, m, n_embd, n_vocab, logits);
            return 0;
        case sp_dtype::F16: {
            std::vector<float> Wf((size_t)n_vocab * n_embd);
            const auto* src = (const uint16_t*)W->data;
            for (size_t i = 0; i < Wf.size(); ++i) Wf[i] = sp_fp16_to_fp32(src[i]);
            sp_matmul_f32(x, Wf.data(), m, n_embd, n_vocab, logits);
            return 0;
        }
        case sp_dtype::Q5_K:
            sp_matmul_f32_q5k(x, W->data, m, n_embd, n_vocab, logits);
            return 0;
        default:
            return matmul_via_ggml_dequant(W, x, m, n_embd, n_vocab, logits);
    }
}

#if defined(SP_ENGINE_WITH_QNN)
// ─────────────────────────────────────────────────────────────────
// Round n up to the next bucket boundary. Power-of-2 schedule with a
// minimum of 8 — collapses every distinct n_kv_total to one of
// ~12 buckets (8/16/32/.../4096/8192), so the QNN runtime graph
// gets compiled at most ~12 times per session instead of once per
// decode step. Garbage from zero-padded K positions is masked out
// by the CPU softmax that runs after the QNN call (the caller's
// causal mask already covers positions ≥ n_kv_total as -INF).
// ─────────────────────────────────────────────────────────────────
static inline int qnn_kv_bucket(int n) {
    if (n <= 8) return 8;
    int b = 16;
    while (b < n) b <<= 1;
    return b;
}

// ─────────────────────────────────────────────────────────────────
// QNN KQ-matmul dispatch shim.
//
// forward_native_attention's hook signature is fp32 in / fp32 out.
// QNN HTP wants fp16 buffers. The shim:
//   1. Buckets n_kv_total UP to the next power of 2 (≥8) so the QNN
//      cache reuses one finalized graph across many real n_kv values.
//   2. Q[n_seq, head_dim] fp32 → fp16 scratch.
//   3. K[n_kv_total, head_dim] fp32 → transposed [head_dim, bucket_n]
//      fp16 scratch, ZERO-PADDED in columns [n_kv_total, bucket_n).
//      sp_llama_qnn_matmul does plain MatMul A@B; we hand K^T.
//   4. dispatch → fp16 [n_seq, bucket_n] in scratch.
//   5. fp16 → fp32 ONLY for the first n_kv_total cols, written into
//      caller's `scores` buffer. Padded cols are discarded.
//
// Returns 0 on success, non-zero to fall through to CPU sp_matmul_f32.
// ─────────────────────────────────────────────────────────────────
static int qnn_kq_dispatch_shim(void* userdata,
                                 const float* Q,
                                 const float* K,
                                 int n_seq, int head_dim, int n_kv_total,
                                 float* scores) {
    auto* cache = (sp_llama_qnn_matmul_cache*)userdata;
    if (!cache) return -1;

    const int n_bucket = qnn_kv_bucket(n_kv_total);

    // Per-thread scratch — small enough to live on the heap as
    // thread_local; resized when shape grows.
    static thread_local std::vector<uint16_t> q_fp16, kt_fp16, out_fp16;

    const size_t q_n  = (size_t)n_seq * head_dim;
    const size_t kt_n = (size_t)head_dim * n_bucket;
    const size_t o_n  = (size_t)n_seq * n_bucket;

    if (q_fp16.size()   < q_n)  q_fp16.resize(q_n);
    if (kt_fp16.size()  < kt_n) kt_fp16.resize(kt_n);
    if (out_fp16.size() < o_n)  out_fp16.resize(o_n);

    // Q: just narrow.
    for (size_t i = 0; i < q_n; ++i) q_fp16[i] = sp_fp32_to_fp16(Q[i]);

    // K^T: transpose [n_kv_total, head_dim] → [head_dim, n_bucket].
    // Real columns [0, n_kv_total) hold the transposed K bytes.
    // Padded columns [n_kv_total, n_bucket) get zero — those produce
    // zero dot-products which (after no scale + no mask in QNN)
    // become 0.0f scores; the CPU softmax-with-mask that runs after
    // this call uses an n_kv_total-sized mask buffer so it never
    // sees the padded columns anyway.
    const uint16_t fp16_zero = 0;
    for (int d = 0; d < head_dim; ++d) {
        uint16_t* kt_row = kt_fp16.data() + (size_t)d * n_bucket;
        for (int p = 0; p < n_kv_total; ++p) {
            kt_row[p] = sp_fp32_to_fp16(K[(size_t)p * head_dim + d]);
        }
        for (int p = n_kv_total; p < n_bucket; ++p) kt_row[p] = fp16_zero;
    }

    uint64_t exec_us = 0;
    int rc = sp_llama_qnn_matmul_dispatch(cache,
                                          (uint32_t)n_seq,
                                          (uint32_t)head_dim,
                                          (uint32_t)n_bucket,
                                          q_fp16.data(),  q_n  * 2,
                                          kt_fp16.data(), kt_n * 2,
                                          out_fp16.data(), o_n * 2,
                                          &exec_us);
    if (rc != 0) return rc;

    // Widen output back to fp32 — only the first n_kv_total cols
    // per row; padded cols are discarded. Caller's `scores` is
    // [n_seq, n_kv_total], not the bucketed shape.
    for (int s = 0; s < n_seq; ++s) {
        const uint16_t* src = out_fp16.data() + (size_t)s * n_bucket;
        float*          dst = scores          + (size_t)s * n_kv_total;
        for (int p = 0; p < n_kv_total; ++p) dst[p] = sp_fp16_to_fp32(src[p]);
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────
// QNN dense weight matmul dispatch shim. Phase 4.13.
//
// out[M, N] = lhs[M, K] @ W_fp16[K, N] on V69 HTP. W_fp16 is the
// pre-baked transposed weight (allocated via sp_qnn_alloc_persistent
// at bind time so HTP reads it without per-call marshal copies).
// lhs is fresh per-call: convert fp32 → fp16 into a thread-local
// scratch, dispatch, widen output back to fp32.
//
// Per-shape graphFinalize cost (~50-100 ms) hits once; subsequent
// dispatches are ~330 µs. The seven dense weights × {prefill, decode}
// shapes cap the unique-shape count at ~14 per session.
// ─────────────────────────────────────────────────────────────────
static int qnn_mm_dispatch_shim(void* userdata,
                                 const float*    lhs,
                                 const uint16_t* W_fp16,
                                 int M, int K, int N,
                                 float* out) {
    auto* cache = (sp_llama_qnn_matmul_cache*)userdata;
    if (!cache || !W_fp16) return -1;

    static thread_local std::vector<uint16_t> a_fp16, c_fp16;
    const size_t a_n = (size_t)M * K;
    const size_t c_n = (size_t)M * N;
    if (a_fp16.size() < a_n) a_fp16.resize(a_n);
    if (c_fp16.size() < c_n) c_fp16.resize(c_n);

    // Narrow lhs → fp16. Hot path; could NEON later, but the
    // dispatch itself is the dominant cost.
    for (size_t i = 0; i < a_n; ++i) a_fp16[i] = sp_fp32_to_fp16(lhs[i]);

    uint64_t exec_us = 0;
    int rc = sp_llama_qnn_matmul_dispatch(cache,
                                          (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                          a_fp16.data(), a_n * 2,
                                          W_fp16,        (size_t)K * N * 2,
                                          c_fp16.data(), c_n * 2,
                                          &exec_us);
    if (rc != 0) return rc;

    for (size_t i = 0; i < c_n; ++i) out[i] = sp_fp16_to_fp32(c_fp16[i]);
    return 0;
}
#endif  // SP_ENGINE_WITH_QNN

#if defined(SP_ENGINE_WITH_QNN)
// ─────────────────────────────────────────────────────────────────
// Dequant a GGUF weight tensor to fp16, transposed to QNN's [K, N]
// layout. GGUF stores W as [n_embd_in (inner = K), n_features_out
// (outer = N)] — i.e., physically [N, K] row-major. QNN's MatMul
// expects right-hand B[K, N]. So we dequant to fp32 [N, K] using
// ggml's type_traits, then permute into fp16 [K, N].
//
// Returns true on success. out_fp16 is sized to K*N elements.
// ─────────────────────────────────────────────────────────────────
static bool dequant_transpose_to_fp16(const ggml_tensor* W,
                                       std::vector<uint16_t>& out_fp16) {
    if (!W || !W->data) return false;
    const int K = (int)W->ne[0];   // inner dim — input features
    const int N = (int)W->ne[1];   // outer dim — output features
    if (K <= 0 || N <= 0) return false;

    const struct ggml_type_traits* tt =
        ggml_get_type_traits((ggml_type)W->type);
    if (!tt || !tt->to_float) {
        std::fprintf(stderr,
            "[sp_native] dequant_transpose: dtype %d has no to_float\n",
            W->type);
        return false;
    }

    // Step 1: full-matrix fp32 dequant in [N, K] order (GGUF native).
    std::vector<float> Wf((size_t)N * K);
    tt->to_float(W->data, Wf.data(), (int64_t)N * K);

    // Step 2: transpose-then-narrow into fp16 [K, N].
    out_fp16.assign((size_t)K * N, 0);
    for (int n = 0; n < N; ++n) {
        const float* row = Wf.data() + (size_t)n * K;
        for (int k = 0; k < K; ++k) {
            out_fp16[(size_t)k * N + n] = sp_fp32_to_fp16(row[k]);
        }
    }
    return true;
}
#endif  // SP_ENGINE_WITH_QNN

// ─────────────────────────────────────────────────────────────────
// Build a ForwardNativeLayer view over a LlamaLayer's ggml tensors.
// All pointers point into the GGUF mmap (or backend-resident copy);
// no ownership transferred.
// ─────────────────────────────────────────────────────────────────
static ForwardNativeLayer bind_layer(const LlamaLayer& L) {
    ForwardNativeLayer out{};
    out.attn_norm    = L.attn_norm    ? (const float*)L.attn_norm->data  : nullptr;
    out.ffn_norm     = L.ffn_norm     ? (const float*)L.ffn_norm->data   : nullptr;
    out.attn_q_norm  = L.attn_q_norm  ? (const float*)L.attn_q_norm->data : nullptr;
    out.attn_k_norm  = L.attn_k_norm  ? (const float*)L.attn_k_norm->data : nullptr;
    if (L.wq) { out.wq = L.wq->data; out.wq_dtype = to_sp_dtype(L.wq->type); }
    if (L.wk) { out.wk = L.wk->data; out.wk_dtype = to_sp_dtype(L.wk->type); }
    if (L.wv) { out.wv = L.wv->data; out.wv_dtype = to_sp_dtype(L.wv->type); }
    if (L.wo) { out.wo = L.wo->data; out.wo_dtype = to_sp_dtype(L.wo->type); }
    out.bq = L.bq ? (const float*)L.bq->data : nullptr;
    out.bk = L.bk ? (const float*)L.bk->data : nullptr;
    out.bv = L.bv ? (const float*)L.bv->data : nullptr;
    out.bo = L.bo ? (const float*)L.bo->data : nullptr;
    if (L.ffn_gate) { out.ffn_gate = L.ffn_gate->data; out.ffn_gate_dtype = to_sp_dtype(L.ffn_gate->type); }
    if (L.ffn_up)   { out.ffn_up   = L.ffn_up->data;   out.ffn_up_dtype   = to_sp_dtype(L.ffn_up->type); }
    if (L.ffn_down) { out.ffn_down = L.ffn_down->data; out.ffn_down_dtype = to_sp_dtype(L.ffn_down->type); }
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Impl
// ─────────────────────────────────────────────────────────────────

struct ForwardNativeContext::Impl {
    const Model*        model    = nullptr;
    const LlamaWeights* weights  = nullptr;
    ForwardNativeHparams hp      = {};
    int n_layer  = 0;
    int n_vocab  = 0;
    int max_seq  = 4096;     // cap; configurable later

    // Per-layer bound views.
    std::vector<ForwardNativeLayer> layers_bound;

    // Per-layer KV state. Each layer owns max_seq * n_head_kv * head_dim
    // fp16 K and V slabs.
    std::vector<ForwardNativeKv> kv;
    std::vector<uint16_t>        kv_storage;  // owned: K then V per layer

    // LM head + final norm pointers.
    const ggml_tensor* tok_embd    = nullptr;
    const ggml_tensor* output_norm = nullptr;
    const ggml_tensor* output      = nullptr;

    sp_arena arena;
    int n_pos = 0;     // current cache fill

#if defined(SP_ENGINE_WITH_QNN)
    // QNN HTP matmul dispatch cache. Lazy-init when SHANNON_PRIME_QNN=1
    // is set in env at context create. Shape-keyed on (M=n_seq, K=head_dim,
    // N=n_kv_total) — n_kv_total grows as decode progresses, so each
    // step past the prefill creates a new shape entry. Phase 2.5
    // graphFinalize cost ~50ms once per shape, then ~330µs per dispatch.
    sp_llama_qnn_matmul_cache* qnn_mm_cache = nullptr;
    bool qnn_active = false;

    // Phase 4.13: per-layer fp16 weight bake for HTP dispatch.
    // Each entry is the transposed [K, N] fp16 weight matching the
    // GGUF-native [N, K] order — QNN's MatMul expects the right
    // operand in [K, N]. Storage is plain heap (std::vector); could
    // be promoted to rpcmem-ION via sp_qnn_alloc_persistent later
    // for zero-copy on the FastRPC bridge.
    struct LayerFp16 {
        std::vector<uint16_t> wq, wk, wv, wo;
        std::vector<uint16_t> ffn_gate, ffn_up, ffn_down;
    };
    std::vector<LayerFp16> layers_fp16;
#endif
};

std::unique_ptr<ForwardNativeContext> ForwardNativeContext::create(
        const Model& model,
        const LlamaWeights& weights) {

    // Native path supports the dense Llama / Qwen2 / Qwen3 standard
    // attention layer shape. MoE / GDN layers go through the ggml
    // path (caller checks model.arch_name() and routes).
    for (const auto& L : weights.layers()) {
        if (L.kind != LlamaLayerKind::STANDARD) {
            std::fprintf(stderr,
                "[sp_native] context: layer kind=%d not supported by "
                "native path yet — fall back to ggml ForwardContext\n",
                (int)L.kind);
            return nullptr;
        }
    }

    auto ctx = std::unique_ptr<ForwardNativeContext>(new ForwardNativeContext());
    auto& I = *ctx->impl_;
    I.model   = &model;
    I.weights = &weights;
    I.n_layer = weights.n_layer();

    // Hparams from Model (gguf_loader stores hparams per arch).
    // Fall back to ggml-tensor shape inspection where Model accessors
    // aren't there yet.
    I.hp.rms_norm_eps    = 1e-5f;
    I.hp.rope_freq_base  = 10000.0f;
    I.hp.rope_freq_scale = 1.0f;

    if (!weights.layers().empty() && weights.layers()[0].wq) {
        const ggml_tensor* wq = weights.layers()[0].wq;
        // wq shape: [n_embd, n_head*head_dim] in GGUF order.
        // ggml ne[0]=n_embd (inner), ne[1]=n_q_dim (output rows).
        I.hp.n_embd = (int)wq->ne[0];
    }
    if (weights.tok_embd) {
        // tok_embd shape: [n_embd, n_vocab].
        I.n_vocab = (int)weights.tok_embd->ne[1];
    }
    if (!weights.layers().empty() && weights.layers()[0].wk) {
        const ggml_tensor* wk = weights.layers()[0].wk;
        // wk output rows = n_head_kv * head_dim. We need n_head_kv from
        // hparams; head_dim deducible later. Pull from Model.
    }

    // Read concrete hparams from Model. GGUF stores hparams under an
    // arch-specific prefix (e.g. "qwen2.attention.head_count" for
    // qwen2, "qwen3.attention.head_count" for qwen3). Probe in order:
    // <arch>.* first, then llama.* fallback. Different arches that
    // share the standard llama-family layout (qwen2/qwen3/phi3/...)
    // all use the same key suffix names.
    auto get_i = [&](const char* suffix, int64_t fb) -> int64_t {
        const std::string a = model.architecture() + "." + suffix;
        if (model.find_key(a) >= 0) return model.get_i64(a, fb);
        const std::string l = std::string("llama.") + suffix;
        return model.get_i64(l, fb);
    };
    auto get_f = [&](const char* suffix, double fb) -> double {
        const std::string a = model.architecture() + "." + suffix;
        if (model.find_key(a) >= 0) return model.get_f64(a, fb);
        const std::string l = std::string("llama.") + suffix;
        return model.get_f64(l, fb);
    };

    I.hp.n_head        = (int)get_i("attention.head_count", 0);
    I.hp.n_head_kv     = (int)get_i("attention.head_count_kv", 0);
    I.hp.n_ff          = (int)get_i("feed_forward_length", 0);
    I.hp.head_dim      = (I.hp.n_head > 0) ? (I.hp.n_embd / I.hp.n_head) : 0;
    I.hp.n_rot         = I.hp.head_dim;   // standard RoPE: rotate all of head_dim
    I.hp.rope_freq_base = (float)get_f("rope.freq_base", 10000.0);
    I.hp.rms_norm_eps   = (float)get_f("attention.layer_norm_rms_epsilon", 1e-5);

    // Some Qwen3 models override head_dim explicitly (when n_embd is
    // not exactly n_head * head_dim — e.g. Qwen3-4B has n_embd=2560,
    // n_head=32, but head_dim=128 not 80). Prefer the GGUF override
    // when present.
    int64_t hd_override = get_i("attention.key_length", 0);
    if (hd_override > 0) {
        I.hp.head_dim = (int)hd_override;
        I.hp.n_rot    = (int)hd_override;
    }

    if (I.hp.n_embd <= 0 || I.hp.n_head <= 0 || I.hp.n_head_kv <= 0
        || I.hp.head_dim <= 0 || I.hp.n_ff <= 0 || I.n_vocab <= 0) {
        std::fprintf(stderr,
            "[sp_native] context: failed to resolve hparams — "
            "n_embd=%d n_head=%d n_head_kv=%d head_dim=%d n_ff=%d n_vocab=%d\n",
            I.hp.n_embd, I.hp.n_head, I.hp.n_head_kv, I.hp.head_dim,
            I.hp.n_ff, I.n_vocab);
        return nullptr;
    }

    std::fprintf(stderr,
        "[sp_native] hparams: n_embd=%d n_head=%d n_head_kv=%d head_dim=%d "
        "n_ff=%d n_vocab=%d n_layer=%d rope_base=%.0f rms_eps=%.1e\n",
        I.hp.n_embd, I.hp.n_head, I.hp.n_head_kv, I.hp.head_dim,
        I.hp.n_ff, I.n_vocab, I.n_layer,
        (double)I.hp.rope_freq_base, (double)I.hp.rms_norm_eps);

    // Bind layers.
    I.layers_bound.reserve(I.n_layer);
    for (const auto& L : weights.layers()) I.layers_bound.push_back(bind_layer(L));

#if defined(SP_ENGINE_WITH_QNN)
    // ── QNN HTP KQ matmul dispatch (Phase 4.5) ──────────────────────
    // When SHANNON_PRIME_QNN=1 in env, allocate the matmul cache and
    // install the shim on every bound layer's kq_dispatch hook. The
    // attention loop then routes the per-head Q@K^T matmul to V69 HTP
    // via sp_qnn_runtime_matmul_create (validated standalone at 238 µs
    // for 256² fp32 / 187 µs fp16). Phase 2.6b's ION-backed persistent
    // K plumbing engages here for real for the first time, since this
    // is a clean direct-call site (no map_custom2 / thread_local /
    // ggml worker fan-out around it).
    {
        const char* v = std::getenv("SHANNON_PRIME_QNN");
        if (v && v[0] == '1') {
            I.qnn_mm_cache = sp_llama_qnn_matmul_cache_create();
            if (I.qnn_mm_cache) {
                for (auto& L : I.layers_bound) {
                    L.kq_dispatch = &qnn_kq_dispatch_shim;
                    L.kq_dispatch_userdata = I.qnn_mm_cache;
                }
                I.qnn_active = true;
                std::fprintf(stderr,
                    "[sp_native] QNN HTP dispatch active — per-head KQ "
                    "matmul routed to V69 HTP via sp_qnn runtime graph\n");
            } else {
                std::fprintf(stderr,
                    "[sp_native] SHANNON_PRIME_QNN=1 set but matmul cache "
                    "create failed — staying on CPU sp_matmul_f32\n");
            }
        }
    }

    // ── Phase 4.13: bake fp16 weight matmuls for HTP dispatch ──────
    // For Qwen2.5-Coder-3B Q5_K_M: 7 weights × 36 layers, dequant from
    // Q5_K to fp16 transposed. Per-layer bytes (head_dim=128, n_embd=
    // 2048, n_kv_dim=256, n_q_dim=2048, n_ff=11008):
    //   wq:       2048*2048*2  =   8 MB
    //   wk:       2048*256*2   =   1 MB
    //   wv:       2048*256*2   =   1 MB
    //   wo:       2048*2048*2  =   8 MB
    //   ffn_gate: 2048*11008*2 =  43 MB
    //   ffn_up:   2048*11008*2 =  43 MB
    //   ffn_down: 11008*2048*2 =  43 MB
    //                            ───────
    //                            147 MB / layer × 36 = ~5.3 GB.
    // Tight on 8 GB device but fits with KV + activations. Skipped
    // when QNN inactive — CPU fallback still owns the originals.
    if (I.qnn_active) {
        // Memory budget: cap fp16 weight bake so we don't OOM the device.
        // SHANNON_PRIME_FP16_BUDGET_MB env var (default 4096 = 4 GB —
        // leaves headroom for KV cache + activations + sp_qnn graphs +
        // OS on an 8 GB phone). Set to 0 to disable bake entirely (HTP
        // dispatch off, CPU stays).
        size_t budget_bytes = (size_t)4096 * 1024 * 1024;
        if (const char* s = std::getenv("SHANNON_PRIME_FP16_BUDGET_MB")) {
            const long long mb = std::atoll(s);
            if (mb >= 0) budget_bytes = (size_t)mb * 1024 * 1024;
        }

        I.layers_fp16.resize(I.n_layer);
        std::fprintf(stderr,
            "[sp_native] baking fp16 weights for HTP "
            "(budget=%.2f GB) ",
            (double)budget_bytes / (1024.0 * 1024.0 * 1024.0));
        size_t total_bytes = 0;
        int    layers_baked = 0, layers_partial = 0;
        for (int Li = 0; Li < I.n_layer; ++Li) {
            const auto& L = weights.layers()[Li];
            auto& F      = I.layers_fp16[(size_t)Li];
            auto& B      = I.layers_bound[(size_t)Li];
            const struct {
                const ggml_tensor*    src;
                std::vector<uint16_t>* dst;
                const uint16_t**       layer_ptr;
            } weights_to_bake[] = {
                { L.wq,       &F.wq,       &B.wq_fp16       },
                { L.wk,       &F.wk,       &B.wk_fp16       },
                { L.wv,       &F.wv,       &B.wv_fp16       },
                { L.wo,       &F.wo,       &B.wo_fp16       },
                { L.ffn_gate, &F.ffn_gate, &B.ffn_gate_fp16 },
                { L.ffn_up,   &F.ffn_up,   &B.ffn_up_fp16   },
                { L.ffn_down, &F.ffn_down, &B.ffn_down_fp16 },
            };
            int   weights_in_layer = 0;
            for (const auto& w : weights_to_bake) {
                if (!w.src) continue;
                const size_t need = (size_t)w.src->ne[0] * w.src->ne[1] * 2;
                if (total_bytes + need > budget_bytes) {
                    // Budget exhausted — leave this and remaining
                    // weights on the CPU path (W_fp16 stays nullptr,
                    // weight_matmul falls through to sp_matmul_f32_q5k).
                    continue;
                }
                try {
                    if (!dequant_transpose_to_fp16(w.src, *w.dst)) {
                        continue;
                    }
                } catch (const std::bad_alloc&) {
                    std::fprintf(stderr,
                        "\n[sp_native] OOM during fp16 bake at layer %d — "
                        "remaining weights stay on CPU path\n", Li);
                    // Hard cap: prevent further attempts.
                    budget_bytes = total_bytes;
                    break;
                }
                *w.layer_ptr = w.dst->data();
                total_bytes += w.dst->size() * sizeof(uint16_t);
                ++weights_in_layer;
            }
            // Wire mm_dispatch on this layer regardless — null
            // W_fp16 entries fall through cleanly.
            B.mm_dispatch          = &qnn_mm_dispatch_shim;
            B.mm_dispatch_userdata = I.qnn_mm_cache;
            if (weights_in_layer == 7)      ++layers_baked;
            else if (weights_in_layer > 0)  ++layers_partial;
            std::fputc(weights_in_layer == 7 ? '.' : (weights_in_layer ? 'p' : '_'),
                       stderr);
        }
        std::fprintf(stderr,
            "\n[sp_native] HTP weight bake: %.2f GB used, %d/%d layers full, "
            "%d partial (rest stay on CPU)\n",
            (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
            layers_baked, I.n_layer, layers_partial);
    }
#endif

    // Allocate KV storage for all layers. Per layer: max_seq * n_head_kv
    // * head_dim fp16 each for K and V.
    const size_t per_layer_kv = (size_t)I.max_seq * I.hp.n_head_kv * I.hp.head_dim;
    I.kv_storage.assign((size_t)I.n_layer * 2 * per_layer_kv, 0);
    I.kv.resize(I.n_layer);
    for (int L = 0; L < I.n_layer; ++L) {
        I.kv[L].k_cache = I.kv_storage.data() + (size_t)L * 2 * per_layer_kv;
        I.kv[L].v_cache = I.kv_storage.data() + (size_t)L * 2 * per_layer_kv + per_layer_kv;
        I.kv[L].max_seq = I.max_seq;
        I.kv[L].n_pos   = 0;
    }

    // Top-level pointers.
    I.tok_embd    = weights.tok_embd;
    I.output_norm = weights.output_norm;
    I.output      = weights.output ? weights.output : weights.tok_embd; // tied weights

    // Reserve arena. Per-step worst case for prefill of n_seq tokens
    // through one layer is roughly:
    //   xn + Q + K + V + attn_out ≈ (5 * n_q_dim + n_embd) * n_seq * 4B
    // plus FFN scratch: 4 * n_ff * n_seq * 4B.
    // Round generously — 64 MB for the working set; we reset the arena
    // between layers so peak is per-layer.
    I.arena.reserve(64 * 1024 * 1024);

    // Persistent thread pool for sp_matmul_f32 / sp_matmul_f32_q5k
    // partitioning. SP_ENGINE_THREADS env (or 4 by default) sets the
    // worker count. Init is idempotent — second context creation in
    // the same process reuses the existing pool.
    {
        int n_threads = 4;
        if (const char* s = std::getenv("SP_ENGINE_THREADS")) {
            int v = std::atoi(s);
            if (v > 0 && v <= 64) n_threads = v;
        }
        sp_threadpool_init(n_threads);
    }

    return ctx;
}

ForwardNativeContext::ForwardNativeContext()
    : impl_(std::make_unique<Impl>()) {}

ForwardNativeContext::~ForwardNativeContext() {
#if defined(SP_ENGINE_WITH_QNN)
    if (impl_ && impl_->qnn_mm_cache) {
        sp_llama_qnn_matmul_cache_destroy(&impl_->qnn_mm_cache);
    }
#endif
}

int ForwardNativeContext::n_layer()   const { return impl_->n_layer; }
int ForwardNativeContext::n_embd()    const { return impl_->hp.n_embd; }
int ForwardNativeContext::n_vocab()   const { return impl_->n_vocab; }
int ForwardNativeContext::head_dim()  const { return impl_->hp.head_dim; }
int ForwardNativeContext::n_head_kv() const { return impl_->hp.n_head_kv; }
int ForwardNativeContext::max_seq()   const { return impl_->max_seq; }

void ForwardNativeContext::reset() {
    auto& I = *impl_;
    for (auto& kv : I.kv) kv.n_pos = 0;
    I.n_pos = 0;
}

// ─────────────────────────────────────────────────────────────────
// Run the layer stack on n_seq tokens already embedded in `x_io`,
// updating positions & KV. Final output stays in x_io.
// ─────────────────────────────────────────────────────────────────
static int run_layers(ForwardNativeContext::Impl& I,
                       const int32_t* pos, int n_seq,
                       float* x_io) {
    const int n_embd = I.hp.n_embd;
    const size_t total = (size_t)n_embd * n_seq;
    std::vector<float> x_next(total);

    for (int L = 0; L < I.n_layer; ++L) {
        I.arena.reset();
        int rc = forward_native_layer(I.layers_bound[L], I.hp,
                                       x_io, pos, n_seq,
                                       I.kv[L], I.arena, x_next.data());
        if (rc != 0) {
            std::fprintf(stderr,
                "[sp_native] layer %d failed (rc=%d)\n", L, rc);
            return rc;
        }
        std::memcpy(x_io, x_next.data(), total * sizeof(float));
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────
// Apply final RMSNorm + LM head to the LAST sequence position only,
// writing logits[n_vocab].
// ─────────────────────────────────────────────────────────────────
static int finalize_logits(ForwardNativeContext::Impl& I,
                            const float* x, int n_seq,
                            std::vector<float>& logits, int& n_vocab_out) {
    const int n_embd  = I.hp.n_embd;
    const int n_vocab = I.n_vocab;
    if (!I.output_norm || !I.output) {
        std::fprintf(stderr,
            "[sp_native] finalize: output_norm or output ptr null\n");
        return -1;
    }
    const float* x_last = x + (size_t)(n_seq - 1) * n_embd;
    std::vector<float> x_normed((size_t)n_embd);
    sp_rms_norm_f32(x_last, (const float*)I.output_norm->data,
                     n_embd, I.hp.rms_norm_eps, x_normed.data());

    logits.assign((size_t)n_vocab, 0.0f);
    int rc = lm_head_apply(I.output, x_normed.data(),
                            /*m=*/1, n_embd, n_vocab, logits.data());
    if (rc != 0) return rc;
    n_vocab_out = n_vocab;
    return 0;
}

bool ForwardNativeContext::prefill(const std::vector<int32_t>& ids,
                                    std::vector<float>& last_logits,
                                    int& n_vocab_out) {
    auto& I = *impl_;
    if (ids.empty()) return false;

    // Reset KV between prompts (caller's choice; matches ForwardContext
    // behavior).
    reset();

    const int n_seq  = (int)ids.size();
    const int n_embd = I.hp.n_embd;

    // Embed all tokens.
    std::vector<float> x((size_t)n_embd * n_seq);
    for (int s = 0; s < n_seq; ++s) {
        if (!embed_lookup(I.tok_embd, ids[s], n_embd,
                           x.data() + (size_t)s * n_embd)) {
            return false;
        }
    }

    // Position ids.
    std::vector<int32_t> pos(n_seq);
    for (int s = 0; s < n_seq; ++s) pos[s] = I.n_pos + s;

    // Layer stack.
    int rc = run_layers(I, pos.data(), n_seq, x.data());
    if (rc != 0) return false;
    I.n_pos += n_seq;

    // Final norm + LM head on last position.
    return finalize_logits(I, x.data(), n_seq,
                            last_logits, n_vocab_out) == 0;
}

bool ForwardNativeContext::decode(int32_t tok,
                                    std::vector<float>& step_logits,
                                    int& n_vocab_out) {
    auto& I = *impl_;
    const int n_embd = I.hp.n_embd;

    std::vector<float> x((size_t)n_embd);
    if (!embed_lookup(I.tok_embd, tok, n_embd, x.data())) return false;

    int32_t pos = I.n_pos;
    int rc = run_layers(I, &pos, /*n_seq=*/1, x.data());
    if (rc != 0) return false;
    I.n_pos += 1;

    return finalize_logits(I, x.data(), /*n_seq=*/1,
                            step_logits, n_vocab_out) == 0;
}

}  // namespace sp::engine
