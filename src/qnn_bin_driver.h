// Shannon-Prime Engine — QNN context-binary driver (Phase 5).
// Copyright (C) 2026 Ray Daniels. AGPLv3.
//
// Loads V69-compiled QNN context binaries (the AI Hub-exported
// Qwen3-4B w4a16 .bins past-Claude validated at 65.8 t/s sustained)
// and exposes them to Engine::generate as the forward pass. Replaces
// forward_native's hand-rolled per-op dispatch with the whole-model
// graph that already lives in the .bins.
//
// Pattern (mirrors lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/
// test_sp_qnn_prefill_batch.c):
//   per 128-token chunk:
//     load split 1   → set inputs    → exec → save residual_out
//     load split 2   → set residual  → exec → save residual_out
//     destroy split 1+2 (V69 HTP context budget = ~1.5 GB)
//     load split 3   → set residual  → exec → save residual_out
//     load split 4   → set residual  → exec → read logits
//     destroy split 3+4
//
// The 4-split topology is fixed by what AI Hub gave us; future
// shareResources (V73+) collapses this to a single resident context.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sp::engine {

// Bench harness: load + exec the 4 splits over n_chunks×128 tokens,
// report wall-clock and steady-state t/s. Doesn't feed real token
// data — buffers are zero-initialized; this is the same "is the
// pipeline alive end-to-end" check that test_sp_qnn_prefill_batch.c
// runs but called from inside sp-engine. Returns 0 on success.
int qnn_bin_prefill_bench(const std::vector<std::string>& split_paths,
                          int n_chunks);

// Schema dump: load each split, enumerate inputs/outputs via
// sp_qnn_get_io_info, print {name, dtype, rank, dims, bytes} for
// every tensor. The output of this is what Phase 5.1 needs to wire
// real prompts in (we have to identify which tensor is tokens,
// which is position_ids, which is the residual stream, which is
// the KV cache). Returns 0 on success.
int qnn_bin_schema_dump(const std::vector<std::string>& split_paths);

// Phase 5.2: real prompt → tokens → split chain → sample
// argmax(logits[last_real_position]). Validates the .bin pipeline
// produces a coherent next token for an actual prompt (not
// zero-init buffers). One prefill chunk (≤128 tokens), no decode
// loop yet — that's Phase 5.3 once AR=1 .bins are exported.
//
//   prompt_tokens — int32 tokenised prompt (truncated to 128 if
//                   longer, padded with 0s if shorter).
//   ar            — activation rank baked into the .bins (128).
//   cl            — context length baked into the .bins (2048).
//   head_dim      — KV head dim (128 for Qwen3-4B).
//   n_freq_pairs  — head_dim / 2 (RoPE frequency pairs).
//   rope_base     — RoPE base frequency (1000000 for Qwen3-4B).
//   out_next_token_id — receives argmax over logits[last_pos].
//
// Returns 0 on success.
int qnn_bin_generate_one(const std::vector<std::string>& split_paths,
                          const std::vector<int32_t>& prompt_tokens,
                          int   ar,
                          int   cl,
                          int   head_dim,
                          float rope_base,
                          int*  out_next_token_id);

// ─────────────────────────────────────────────────────────────────
// QnnBinSession — persistent HTP context management
// ─────────────────────────────────────────────────────────────────

class SpOracle;  // forward-declared; defined in speculative_oracle.h

class QnnBinSession {
public:
    QnnBinSession();
    ~QnnBinSession();

    // Load the 4 splits and keep them resident in HTP memory.
    int load(const std::vector<std::string>& split_paths,
             int ar = 128, int cl = 2048, int hd = 2560,
             float rope_base = 1000000.0f);

    // Attach a speculative oracle for Phase 8 draft-token prediction.
    // The oracle must outlive all subsequent generate() calls.
    // Pass nullptr to disable oracle (default — no speculative decode).
    // Oracle must already have been prefilled with the session prompt
    // before generate() is called (QnnBinSession calls prefill internally
    // when an oracle is attached and generate() is invoked).
    void set_oracle(SpOracle* oracle);

    // Run a full generation loop.
    // If an oracle is attached (set_oracle), uses speculative decoding:
    //   - Oracle drafts SP_ORACLE_DRAFT_N tokens per step
    //   - HTP verifies the draft in a single batched forward pass
    //   - Accepted tokens advance both the HTP KV and the oracle
    // Accuracy stats are printed to stderr after generation completes.
    int generate(const std::vector<int32_t>& prompt_ids,
                 int n_predict,
                 std::vector<int32_t>& out_ids);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sp::engine
