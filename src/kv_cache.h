// Shannon-Prime Engine — compressed KV cache (public API)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Wraps the C-level sp_shadow_cache_t (ship path) or sp_sqfree_cache_t
// (aggressive path) behind a single typed C++ interface. The cache owns
// the per-(layer, head) compressed storage and the scratch buffers; the
// engine writes raw fp32 K/V vectors in and reads reconstructed fp32
// vectors out. Compression happens on the write path by construction.
//
// Stage 4: the cache is a passive store. Stage 5 will swap the inline
// K/V projection in build_block for read-from-cache so decode can run
// over a real compressed history.

#pragma once

#include "engine.h"

#include <memory>
#include <string>
#include <vector>

namespace sp::engine {

class KvCache {
public:
    // Allocate a cache sized for `n_layer × n_head_kv × max_seq` slots.
    // `cfg.sqfree`, `cfg.spinor`, `cfg.mobius`, `cfg.k_bits_csv`,
    // `cfg.v_bits_csv`, `cfg.residual_bits` select the compression.
    // Returns nullptr on init failure (bad bit allocation, OOM, etc.).
    static std::unique_ptr<KvCache> create(int n_layer, int n_head_kv,
                                           int head_dim, int max_seq,
                                           const Config& cfg);

    ~KvCache();
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

    // Write a contiguous batch of n_tokens K and V vectors at sequence
    // positions [pos_offset, pos_offset + n_tokens) for the given layer.
    //
    // Layout (matches what ggml_backend_tensor_get returns for a
    // [head_dim, n_head_kv, n_tokens] tensor):
    //   K_flat[(q * n_head_kv + h) * head_dim + d]
    //
    // Returns false if positions overflow max_seq.
    bool write(int layer, int pos_offset, int n_tokens,
               const float* K_flat, const float* V_flat);

    // Read positions [0, kv_len) for the layer back into K_out / V_out
    // using the same layout. Buffers are resized as needed.
    bool read(int layer, int kv_len,
              std::vector<float>& K_out,
              std::vector<float>& V_out) const;

    // --- adaptive calibration ---
    //
    // Calibration feeds raw KV vectors through the spectral transform
    // and accumulates per-coefficient variance. calibrate_end() rebuilds
    // the internal masks so write/read use variance-ranked ordering
    // (sqfree: Knight mask with L/2 skeleton; ship: variance-ranked
    // reorder into banded quantizer).
    //
    // Feed ALL calibration vectors between begin/end. Typical use: feed
    // K vectors from the first forward pass (warmup), then end.
    bool calibrate_begin();
    void calibrate_feed(const float* vec);
    bool calibrate_end();
    bool is_calibrated() const;

    // --- diagnostics / introspection ---
    int  n_layer()           const;
    int  n_head_kv()         const;
    int  head_dim()          const;
    int  max_seq()           const;
    bool is_sqfree()         const;
    float compression_ratio() const;
    std::string describe()    const;

private:
    KvCache();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
