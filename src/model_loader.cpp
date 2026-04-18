// Shannon-Prime Engine — GGUF model loader (skeleton)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Wraps ggml's GGUF loader so the rest of the engine sees a typed
// Model object rather than raw KV pairs + tensor handles. Uses
// `vendor/ggml/gguf.h` directly (MIT), with no dependency on
// llama.cpp's model-specific glue — we do our own per-architecture
// tensor binding.

#include "shannon_prime.h"

namespace sp::engine {

// Placeholder. Real loader lands alongside the first model architecture
// (Qwen / Llama family) — see roadmap.

} // namespace sp::engine
