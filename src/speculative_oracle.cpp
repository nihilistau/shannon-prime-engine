// Shannon-Prime Engine — Speculative Oracle implementation (Phase 8)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Builds a thin oracle layer on top of ForwardNativeContext:
//   - load()          → Model::load + LlamaWeights::load + ForwardNativeContext::create
//   - prefill()       → ForwardNativeContext::prefill
//   - step()          → ForwardNativeContext::decode + NEON argmax
//   - predict_multi() → save KV pos, step N times, restore KV pos
//   - accept()        → advance committed pos, step from last accepted token
//   - resync()        → step oracle forward with the correct token
//   - moe_gate_topk() → NEON softmax over ffn_gate_inp weights
//
// KV snapshot strategy: ForwardNativeContext::reset() replaces the entire
// KV state. For speculative lookahead we don't want full reset — we just
// need to roll back the KV write pointer to the saved position. Since
// ForwardNativeContext doesn't expose raw KV, we use a "rewind" approach:
// snapshot the committed position, run N draft steps, then call reset()
// and replay the committed history to restore exact state. This costs
// (committed_pos) oracle steps on a mismatch. For short contexts (< 512)
// this is fast; for longer contexts Phase 8B will add a KV slice API.

#include "speculative_oracle.h"
#include "forward_native_context.h"
#include "gguf_loader.h"
#include "llama_weights.h"
#include "sp_kernels_cpu.h"

// ggml_tensor::ne[] needs the full struct definition.
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
  #include <arm_neon.h>
  #define SP_HAS_NEON 1
#else
  #define SP_HAS_NEON 0
#endif

namespace sp::engine {

// ─── NEON argmax over a fp32 logit vector ─────────────────────────────────
static int fp32_argmax(const float* v, int n) {
    if (n <= 0) return 0;
    int   best_idx = 0;
    float best_val = v[0];
#if SP_HAS_NEON
    // 4-way reduction. Each iteration compares 4 candidates; track per-lane
    // maximums and their lane indices.
    // We accumulate the global max in scalar after the SIMD prefix.
    int i = 1;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vec = vld1q_f32(v + i);
        float vals[4];
        vst1q_f32(vals, vec);
        for (int k = 0; k < 4; ++k) {
            if (vals[k] > best_val) { best_val = vals[k]; best_idx = i + k; }
        }
    }
    for (; i < n; ++i) {
        if (v[i] > best_val) { best_val = v[i]; best_idx = i; }
    }
#else
    for (int i = 1; i < n; ++i) {
        if (v[i] > best_val) { best_val = v[i]; best_idx = i; }
    }
#endif
    return best_idx;
}

// ─── NEON softmax in-place over a fp32 vector ─────────────────────────────
// Used by moe_gate_topk (Phase 9 stub). Suppressed-unused warning via attribute.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced local function
#endif
static void fp32_softmax_inplace(float* v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { v[i] = std::exp(v[i] - mx); sum += v[i]; }
    const float inv = 1.f / (sum + 1e-9f);
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

// ─── SpOracle::Impl ───────────────────────────────────────────────────────

struct SpOracle::Impl {
    // Loaded model assets — lifetime: same as Impl.
    std::unique_ptr<Model>                 model;
    std::unique_ptr<LlamaWeights>          weights;
    std::unique_ptr<ForwardNativeContext>  ctx;

    // History of all tokens fed to the oracle since the last reset/prefill.
    // Used to replay KV state after a speculative draft is discarded.
    std::vector<int32_t> history;   // [0..committed_pos-1]
    int committed_pos = 0;          // number of tokens whose KV is committed

    // Whether the oracle is initialized and ready.
    bool ready_ = false;

    // Working buffer for logits (reused across calls).
    std::vector<float> logits_buf;
    int vocab_ = 0;
    int n_vocab_out_ = 0;

    // MoE gating: pointer into LlamaLayer::ffn_gate_inp (first MoE layer found).
    // nullptr for dense models. Shape [n_embd, n_experts] row-major fp32/q5k.
    const ggml_tensor* gate_inp_tensor = nullptr;
    int n_moe_experts = 0;

    // Stats.
    int n_total_ = 0;
    int n_hits_  = 0;

    // Run the oracle forward one token, update logits_buf, return predicted next.
    // pos is the absolute position of `tok` in the sequence.
    int32_t run_one(int32_t tok) {
        if (!ctx) return -1;
        bool ok = ctx->decode(tok, logits_buf, n_vocab_out_);
        if (!ok || n_vocab_out_ <= 0) return -1;
        return fp32_argmax(logits_buf.data(), n_vocab_out_);
    }
};

// ─────────────────────────────────────────────────────────────────────────
// SpOracle
// ─────────────────────────────────────────────────────────────────────────

SpOracle::SpOracle()  : impl_(std::make_unique<Impl>()) {}
SpOracle::~SpOracle() = default;

int SpOracle::load(const char* gguf_path, int max_ctx) {
    auto& I = *impl_;
    I.ready_ = false;

    if (!gguf_path || gguf_path[0] == '\0') {
        std::fprintf(stderr, "[sp_oracle] load: empty path\n");
        return -1;
    }

    // ── Step 1: parse GGUF metadata ───────────────────────────────────
    I.model = Model::load(gguf_path);
    if (!I.model) {
        std::fprintf(stderr, "[sp_oracle] Model::load failed: %s\n", gguf_path);
        return -1;
    }

    std::fprintf(stderr, "[sp_oracle] loaded: arch=%s n_layer=%u n_embd=%u "
                 "vocab=%u ctx=%u\n",
                 I.model->architecture().c_str(),
                 I.model->n_layer(),
                 I.model->n_embd(),
                 I.model->vocab_size(),
                 I.model->context_length());

    // ── Step 2: bind weight tensors (CPU mmap, zero-copy) ─────────────
    I.weights = LlamaWeights::load(*I.model);
    if (!I.weights) {
        std::fprintf(stderr, "[sp_oracle] LlamaWeights::load failed\n");
        I.model.reset();
        return -1;
    }

    // ── Step 3: create native forward context ─────────────────────────
    I.ctx = ForwardNativeContext::create(*I.model, *I.weights);
    if (!I.ctx) {
        std::fprintf(stderr, "[sp_oracle] ForwardNativeContext::create failed "
                     "(arch not supported by native path — MoE/GDN?)\n");
        I.weights.reset();
        I.model.reset();
        return -2;
    }

    I.vocab_ = I.ctx->n_vocab();
    I.logits_buf.resize((size_t)I.vocab_, 0.f);

    // ── Step 4: probe for MoE gating tensor (Phase 9 prep) ────────────
    // Look for ffn_gate_inp in the first layer that has it.
    I.gate_inp_tensor = nullptr;
    I.n_moe_experts   = 0;
    for (int li = 0; li < I.weights->n_layer(); ++li) {
        const LlamaLayer& lay = I.weights->layers()[li];
        if (lay.ffn_gate_inp) {
            I.gate_inp_tensor  = lay.ffn_gate_inp;
            // Tensor shape: [n_embd, n_experts] — experts is along dim 1.
            I.n_moe_experts = (int)I.gate_inp_tensor->ne[1];
            std::fprintf(stderr, "[sp_oracle] MoE gating found: layer %d, "
                         "%d experts\n", li, I.n_moe_experts);
            break;
        }
    }
    if (!I.gate_inp_tensor)
        std::fprintf(stderr, "[sp_oracle] dense model — MoE gating not available\n");

    I.history.clear();
    I.committed_pos = 0;
    I.ready_ = true;
    (void)max_ctx;   // ForwardNativeContext owns the KV; max_ctx noted for future API

    std::fprintf(stderr, "[sp_oracle] ready (vocab=%d n_layer=%d n_embd=%d)\n",
                 I.vocab_, I.ctx->n_layer(), I.ctx->n_embd());
    return 0;
}

bool SpOracle::prefill(const std::vector<int32_t>& prompt_ids) {
    auto& I = *impl_;
    if (!I.ready_ || !I.ctx) return false;

    int n_vocab_out = 0;
    bool ok = I.ctx->prefill(prompt_ids, I.logits_buf, n_vocab_out);
    if (!ok) {
        std::fprintf(stderr, "[sp_oracle] prefill failed\n");
        return false;
    }
    I.n_vocab_out_ = n_vocab_out;

    // Commit the prompt to history so we can replay if needed.
    I.history = prompt_ids;
    I.committed_pos = (int)prompt_ids.size();

    return true;
}

int32_t SpOracle::step(int32_t tok) {
    auto& I = *impl_;
    if (!I.ready_) return -1;

    int32_t pred = I.run_one(tok);

    // Advance committed history.
    I.history.push_back(tok);
    I.committed_pos++;

    return pred;
}

int SpOracle::predict_multi(int n, int32_t* out_draft) {
    auto& I = *impl_;
    if (!I.ready_ || n <= 0) return 0;
    if (!out_draft) return 0;

    // We draft by running the oracle forward n times from the current
    // committed state. The first draft token is the oracle's current
    // prediction (already computed at the last step() call or prefill).
    // After drafting, we roll back KV to committed_pos by resetting the
    // context and replaying history.
    //
    // For efficiency when n==1, we skip the replay entirely.
    //
    // Draft loop: start from the last token in committed history.
    // The oracle's KV currently includes history[0..committed_pos-1].

    // The current logits_buf already holds the prediction after
    // committed_pos tokens. Token 0 of the draft is that prediction.
    if (I.logits_buf.empty() || I.n_vocab_out_ <= 0) return 0;

    int32_t cur_pred = fp32_argmax(I.logits_buf.data(), I.n_vocab_out_);
    int n_drafted = 0;

    // Temporary KV state advance (speculative): run n draft steps.
    // We feed each draft prediction as the input to the next step.
    std::vector<int32_t> draft_seq;
    draft_seq.reserve(n);
    draft_seq.push_back(cur_pred);
    n_drafted = 1;

    for (int d = 1; d < n; ++d) {
        int32_t next_pred = I.run_one(cur_pred);
        if (next_pred < 0) break;
        cur_pred = next_pred;
        draft_seq.push_back(cur_pred);
        n_drafted++;
    }

    std::copy(draft_seq.begin(), draft_seq.end(), out_draft);

    // Roll back speculative KV to committed state.
    // Replay history[0..committed_pos-1] to restore oracle's KV.
    if (n_drafted > 0) {
        I.ctx->reset();
        if (!I.history.empty()) {
            int n_vocab_unused = 0;
            // Replay all committed tokens: prefill(history[0..committed_pos-1])
            // For efficiency, use prefill() for the bulk, then decode for the last.
            if (I.committed_pos > 0) {
                std::vector<int32_t> replay(I.history.begin(),
                                            I.history.begin() + I.committed_pos);
                bool ok = I.ctx->prefill(replay, I.logits_buf, n_vocab_unused);
                if (!ok) {
                    // Catastrophic failure — oracle is desynced. Mark not-ready.
                    std::fprintf(stderr, "[sp_oracle] predict_multi: replay failed "
                                 "after draft, oracle disabled\n");
                    I.ready_ = false;
                    return 0;
                }
                I.n_vocab_out_ = n_vocab_unused;
            }
        }
    }

    return n_drafted;
}

void SpOracle::accept(int accepted, const int32_t* verified_tokens) {
    auto& I = *impl_;
    if (!I.ready_ || accepted <= 0) return;

    // The HTP verified `accepted` tokens. Advance oracle KV by stepping
    // through the accepted token IDs so it stays in sync.
    // After the loop, oracle's logits reflect the prediction after
    // verified_tokens[accepted-1].
    for (int i = 0; i < accepted; ++i) {
        int32_t tok = verified_tokens[i];
        I.run_one(tok);
        I.history.push_back(tok);
        I.committed_pos++;
    }
}

void SpOracle::resync(int32_t correct_tok) {
    auto& I = *impl_;
    if (!I.ready_) return;

    // Feed the correct token (the first token where draft diverged from HTP).
    I.run_one(correct_tok);
    I.history.push_back(correct_tok);
    I.committed_pos++;
}

void SpOracle::record_batch(const int32_t* draft, const int32_t* verified, int n) {
    auto& I = *impl_;
    for (int i = 0; i < n; ++i) {
        I.n_total_++;
        if (draft[i] == verified[i]) I.n_hits_++;
    }
}

int   SpOracle::n_total()  const { return impl_->n_total_; }
int   SpOracle::n_hits()   const { return impl_->n_hits_; }
float SpOracle::accuracy() const {
    if (impl_->n_total_ == 0) return 0.f;
    return (float)impl_->n_hits_ / (float)impl_->n_total_;
}
void SpOracle::reset_stats() { impl_->n_total_ = 0; impl_->n_hits_ = 0; }

void SpOracle::reset() {
    auto& I = *impl_;
    if (I.ctx) I.ctx->reset();
    I.history.clear();
    I.committed_pos = 0;
    I.n_vocab_out_  = 0;
    std::fill(I.logits_buf.begin(), I.logits_buf.end(), 0.f);
}

bool SpOracle::ready()      const { return impl_->ready_; }
int  SpOracle::vocab_size() const { return impl_->vocab_; }

int SpOracle::moe_gate_topk(int n_experts, int k,
                             int* out_ids, float* out_scores) {
    auto& I = *impl_;
    if (!I.gate_inp_tensor || I.n_moe_experts <= 0) return 0;
    if (k <= 0 || !out_ids) return 0;

    // The gate input tensor has shape [n_embd, n_experts].
    // We compute gate_scores = h @ gate_inp^T where h is the oracle's
    // last hidden state (current logits_buf before lm_head projection).
    //
    // ForwardNativeContext doesn't expose the raw hidden state — only
    // post-lm_head logits. Phase 9 will add a get_hidden_state() API.
    // For now: approximate by using the logit vector itself (re-normalised
    // to n_embd space is not meaningful here, so we return 0 until the
    // hidden state API is wired).
    //
    // TODO(Phase 9): expose ForwardNativeContext::last_hidden_state()
    //   and compute proper gate scores.
    (void)n_experts;
    (void)k;
    (void)out_ids;
    (void)out_scores;

    std::fprintf(stderr, "[sp_oracle] moe_gate_topk: hidden-state API not yet "
                 "wired (Phase 9). Returning 0.\n");
    return 0;
}

} // namespace sp::engine
