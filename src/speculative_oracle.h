// Shannon-Prime Engine — Speculative Oracle (Phase 8: NEON Oracle)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Wraps ForwardNativeContext to drive a small "draft model" on ARM NEON
// for speculative token prediction. The oracle runs on the Prime cores
// while the HTP is finishing the previous step; its predictions feed the
// speculative-decode verification batch in QnnBinSession::generate().
//
// Design contract:
//   1. Load a GGUF draft model (any llama-family arch, typically small:
//      Qwen2-0.5B-Instruct-Q5_K_M ~390 MB, ~50 tok/s on Cortex-A78).
//   2. prefill() the oracle with the same prompt as the main model so
//      its KV state is aligned.
//   3. Each decode step: predict_multi() drafts up to SP_ORACLE_DRAFT_N
//      tokens. QnnBinSession runs a batched HTP verify pass. Accepted
//      tokens advance the oracle via accept(); a mismatch calls resync()
//      with the correct token.
//   4. record_result() / accuracy() measure draft hit rate; logged at
//      the end of generate().
//
// MoE gating (Phase 9 prep):
//   If the draft model has ffn_gate_inp weights, moe_gate_topk() returns
//   the predicted top-k expert indices from the draft hidden state. Phase 9
//   feeds these to Halide DMA to pre-stream expert weights from UFS.
//   For dense models (Qwen3-4B), moe_gate_topk() is a no-op (returns 0).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace sp::engine {

// Default number of speculative tokens to draft per step.
// Tuned for Qwen2-0.5B @ 50 tok/s oracle vs Qwen3-4B @ 15 tok/s main:
// drafting 3 tokens costs ~60 ms oracle time; a batch-verify of 3 tokens
// on HTP costs roughly the same as 1 single-token step (~15 ms) because
// the attention KQ computation dominates and scales O(n_kv), not O(n_seq)
// for small n_seq. With 70% oracle accuracy this gives ~2.1× effective speed.
static constexpr int SP_ORACLE_DRAFT_N = 3;

class SpOracle {
public:
    SpOracle();
    ~SpOracle();

    // Load a draft model from a GGUF file.
    //
    // gguf_path — path to any llama-family GGUF (q5k, q8_0, fp16, etc.).
    //   The model can be a different arch/size than the main model.
    //   Returns 0 on success, -1 on load failure, -2 if the arch isn't
    //   supported by ForwardNativeContext yet (MoE / GDN fall through).
    //
    // max_ctx — KV cache depth for the oracle. Set to match the main
    //   model's cl (context length) or a smaller value to save memory.
    int load(const char* gguf_path, int max_ctx = 2048);

    // Warm the oracle KV state by running the prompt through it.
    // Must be called with the same token sequence as the main model
    // before the first decode step. Resets any previous KV state.
    // Returns true on success.
    bool prefill(const std::vector<int32_t>& prompt_ids);

    // Single-step oracle decode: advance the oracle's KV state with
    // `tok` and return the predicted NEXT token (the token after `tok`).
    // Returns -1 if the oracle is not loaded or encounters an error.
    //
    // This updates the oracle's internal KV cache — call accept() or
    // resync() to keep it aligned with the verified output.
    int32_t step(int32_t tok);

    // Draft n tokens speculatively without advancing the committed KV
    // state. The committed position is saved before drafting and
    // restored via a lightweight state snapshot.
    //
    // On return, out_draft[0..n_out-1] contains the predicted token IDs
    // (out_draft[0] = prediction after current token, etc.).
    // Returns the number of tokens actually drafted (may be < n if EOS
    // is predicted or an error occurs). Caller allocates out_draft[n].
    int predict_multi(int n, int32_t* out_draft);

    // Advance the oracle's committed KV state by accepting `accepted`
    // tokens from the verified prefix. Called after the HTP verify pass
    // returns `accepted` matching tokens.
    //
    // If accepted == n_drafted, the last draft step was re-run internally
    // to generate the next speculative batch; call predict_multi() for
    // the following step.
    void accept(int accepted, const int32_t* verified_tokens);

    // Resync after a mismatch: discard the draft state and feed the
    // correct token `correct_tok` to get back in sync. The oracle will
    // step forward from this position on the next predict_multi() call.
    void resync(int32_t correct_tok);

    // ── Accuracy stats ────────────────────────────────────────────────
    // Record whether a draft token was correct (batch call — pass the
    // full draft array and the verified outputs from HTP).
    void record_batch(const int32_t* draft, const int32_t* verified, int n);

    int   n_total()  const;      // total tokens drafted and verified
    int   n_hits()   const;      // correctly predicted tokens
    float accuracy() const;      // n_hits / n_total, or 0 if none

    void reset_stats();

    // ── State ─────────────────────────────────────────────────────────
    // Reset KV state and stats. Call before a new conversation.
    void reset();

    // True if the oracle loaded successfully and is ready to use.
    bool ready() const;

    // Vocabulary size of the draft model.
    int vocab_size() const;

    // ── MoE gating (Phase 9 prep) ─────────────────────────────────────
    // Given the oracle's current draft hidden state h [n_embd], compute
    // softmax gate scores over n_experts experts and return the top-k
    // expert indices in out_ids[0..k-1] (sorted by score, descending).
    // out_scores[0..k-1] filled with the corresponding softmax weights.
    //
    // Uses ffn_gate_inp weights from the draft model's last completed
    // layer. For dense models (no MoE FFN), returns 0. The result is
    // only valid after a step() or predict_multi() call that completed
    // at least one full layer.
    //
    // Phase 9: these indices drive sp_halide_dma_prefetch_expert() to
    // stream the top-k expert weights from UFS before HTP needs them.
    int moe_gate_topk(int n_experts, int k,
                      int* out_ids, float* out_scores);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
