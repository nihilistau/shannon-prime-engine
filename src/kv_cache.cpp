// Shannon-Prime Engine — compressed KV cache management (skeleton)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// The point of owning the engine rather than patching into llama.cpp is
// that THIS file — not someone else's — decides the KV layout. Write path
// is compressed from day one. Read path decompresses only into attention-
// local scratch, not into a shadow fp16 buffer that a host hook has to
// chase.
//
// Filled in as the attention kernel + model loader land.

#include "shannon_prime.h"

namespace sp::engine {

// Placeholder — real implementation will wrap either sp_shadow_cache_t
// (ship path) or sp_sqfree_cache_t (aggressive path) behind a single
// CompressedKVStore interface, selected by Config::sqfree.

} // namespace sp::engine
