// Shannon-Prime Engine — long-term Algebraic Resonance Memory bank
// (Phase 13.B engine glue around lib/shannon-prime/core/sp_arm).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// One bank object per inference context. The bank has one slab per
// (layer, kv_head) — so for Gemma3-1B (26 layers × 1 kv_head) we
// have 26 slabs, each N=256 coefficients × 2 primes × 8 bytes = 4 KB.
// Total bank state: ~104 KB on Gemma3-1B, ~3 MB on a 7B 40-layer model.
//
// Phase 13.B (this file) only implements the WRITE side:
//   when sp_ok_kv_cache_clear is about to drop the active context
//   window, walk every (layer, kv_head, token) in the cache, decode
//   the K and V polynomials to fp32, and bind them into the
//   corresponding slab via sp_arm_bank_write. The forward path does
//   not yet consume the bank — that arrives in Phase 13.C.
//
// Invariant: with SP_ENGINE_MEMORY=1 enabled but no recall hook yet,
// PPL is bit-identical to the phase12-step-e baseline. The bank is
// purely a write-side side-effect.

#pragma once

#include "sp_kv_cache_ok.h"

extern "C" {
#include "../lib/shannon-prime/core/sp_arm.h"
}

#include <cstdint>
#include <vector>

namespace sp::engine {

struct sp_lt_memory {
    // The underlying ARM bank — a thin C struct from the math repo.
    // We own M_q1 / M_q2 storage in std::vectors and hand pointers to
    // the bank at init time.
    sp_arm_bank bank{};

    // Owning storage for the bank slabs.  Sized n_layers*n_kv_head*N.
    std::vector<uint64_t> M_q1_storage;
    std::vector<uint64_t> M_q2_storage;

    // Per-token scratch for the K/V decode (reused across writes).
    std::vector<float>    k_decode_fp32;        // length d
    std::vector<float>    v_decode_fp32;        // length d
    std::vector<uint64_t> write_scratch_4N;     // 4*N uint64
    std::vector<int64_t>  write_int_scratch;    // N int64

    // Layout metadata.
    int n_layers   = 0;
    int n_kv_head  = 0;
    int head_dim   = 0;
    int n_slabs    = 0;     // = n_layers * n_kv_head

    // Diagnostics.
    uint64_t total_writes = 0;     // total (layer,head,token) bindings
    uint64_t total_evictions = 0;  // calls to write_evict
};

// Allocate the bank slabs (one per (layer, kv_head)) and the per-write
// scratch buffers. `head_dim` must be ≤ SP_ARM_RING_N (256).  `delta`
// is the ARM encoder scale; recommended 256 (2^8) to keep accumulator
// headroom across many evictions.
//
// Returns false if the dimensions are invalid or head_dim > N.
bool sp_lt_memory_init(sp_lt_memory& mem,
                        int n_layers, int n_kv_head, int head_dim,
                        double delta);

// Walk the live KV cache and bind every (K_t, V_t) pair into the
// corresponding (layer, kv_head) slab.  Decodes K and V from their
// stored sp_ok_t representation to fp32 per the cache's scale_recip
// and frobenius_scale, then calls sp_arm_bank_write.
//
// Caller is expected to invoke this BEFORE sp_ok_kv_cache_clear
// (or any other eviction).  Reads `cache.cur_len` tokens.
//
// At cache.cur_len == 0 this is a no-op.
//
// Returns false if shapes mismatch the bank's expectations.
bool sp_lt_memory_write_evict(sp_lt_memory& mem,
                                const sp_ok_kv_cache& cache);

// Soft "memory mass" of a specific (layer, kv_head) slab.
// Decodes the bank slab via inverse-NTT and returns the L2² of the
// resulting coefficient polynomial.  Used as a threshold check by
// the recall path in Phase 13.C (a slab below the floor has nothing
// useful to recall).
double sp_lt_memory_slab_norm(const sp_lt_memory& mem,
                               int layer, int kv_head);

}  // namespace sp::engine
