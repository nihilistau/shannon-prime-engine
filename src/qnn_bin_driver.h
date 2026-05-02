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

}  // namespace sp::engine
