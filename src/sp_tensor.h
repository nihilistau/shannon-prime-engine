// Shannon-Prime Engine — minimal native tensor + arena.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 4 of the engine pivot: replace ggml entirely for the forward
// pass. ggml stays available in the build for tests/benches that
// haven't been ported yet, but `forward_native.cpp` (when
// SP_ENGINE_NATIVE=1) uses ONLY this header for tensor metadata, plus
// sp_quant for dequant and sp_kernels_cpu/Hexagon/sp_qnn for compute.
//
// What this is:
//   - sp_dtype: minimal datatype enum (matches GGUF's enum values for
//     the formats we care about so weight binding is a memcpy).
//   - sp_tensor: shape (≤4 dims) + strides + dtype + raw data ptr.
//     No graph node, no allocator-of-record, no scheduler. Just a
//     descriptor over a flat byte buffer.
//   - sp_arena: bump allocator over a single big buffer. Used for
//     layer-step intermediates (Q/K/V projections, attention output,
//     FFN intermediate). One arena per ForwardContext, reset between
//     layers within a step. Ownership: arena owns its buffer, caller
//     owns the arena.
//
// Why not just use ggml_tensor: ggml_tensor is tied to ggml_context
// allocation and graph membership. We want raw descriptors that the
// kernels read. Our kernels take (data, shape, strides) directly so
// there's nothing to inherit from a framework.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sp::engine {

// Datatype enum — values match GGUF's GGML_TYPE_* numerically so we
// can bind weights from GGUF metadata without translation. Only the
// types we actually need for Qwen2/3 inference are defined.
enum class sp_dtype : int32_t {
    F32  = 0,
    F16  = 1,
    Q4_K = 12,   // GGML_TYPE_Q4_K
    Q5_K = 13,   // GGML_TYPE_Q5_K
    Q6_K = 14,   // GGML_TYPE_Q6_K
    Q8_0 = 8,    // GGML_TYPE_Q8_0
    UNDEFINED = -1,
};

// Bytes per element (or per quantized BLOCK for the K-quant types).
// For K-quants the natural unit is the 256-element super-block; the
// caller multiplies by (numel/256) to get total bytes.
struct sp_dtype_traits {
    bool   is_quantized;
    size_t block_elements;   // elements per packed block (256 for K-quants)
    size_t block_bytes;      // bytes per packed block
};

inline sp_dtype_traits sp_dtype_info(sp_dtype t) {
    switch (t) {
        case sp_dtype::F32:  return { false, 1,   4 };
        case sp_dtype::F16:  return { false, 1,   2 };
        case sp_dtype::Q8_0: return { true,  32,  34 };   // 32 quants + fp16 scale
        case sp_dtype::Q4_K: return { true,  256, 144 };  // K-quant block size
        case sp_dtype::Q5_K: return { true,  256, 176 };  // K-quant block size
        case sp_dtype::Q6_K: return { true,  256, 210 };  // K-quant block size
        default:             return { false, 0,   0 };
    }
}

// Total bytes for a contiguous tensor of `numel` elements at dtype t.
inline size_t sp_dtype_byte_size(sp_dtype t, size_t numel) {
    sp_dtype_traits info = sp_dtype_info(t);
    if (info.block_elements == 0) return 0;
    if (info.is_quantized) {
        // numel must be a multiple of block_elements for quantized types.
        const size_t nblocks = (numel + info.block_elements - 1) / info.block_elements;
        return nblocks * info.block_bytes;
    }
    return numel * info.block_bytes;
}

// Tensor descriptor. Up to 4 dimensions, contiguous-row layout by default.
// `strides` are in BYTES (matches ggml convention) so we can describe
// non-contiguous views without needing element-size juggling.
//
// Convention: shape[0] is the inner-most (contiguous-stride) dim,
// shape[3] is the outer-most. Same as ggml.
struct sp_tensor {
    static constexpr int MAX_DIMS = 4;

    sp_dtype dtype       = sp_dtype::UNDEFINED;
    int      n_dims      = 0;
    int64_t  shape[MAX_DIMS]  = {1, 1, 1, 1};
    size_t   strides[MAX_DIMS] = {0, 0, 0, 0};
    void*    data        = nullptr;
    const char* name     = nullptr;   // optional, weak ptr

    // Total element count.
    inline int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < n_dims; ++i) n *= shape[i];
        return n;
    }
    // Total byte size (assumes contiguous; for non-contig views this is
    // the size IF the tensor were contiguous — useful for arena alloc).
    inline size_t nbytes_contig() const {
        return sp_dtype_byte_size(dtype, (size_t)numel());
    }
    // True if strides imply a row-major contiguous layout matching shape.
    bool is_contiguous() const;

    // Set up shape + compute contiguous strides for a given dtype.
    // Caller still has to assign `data` after this (typically from arena).
    void reset(sp_dtype t, int nd, const int64_t s[MAX_DIMS]);
};

// Bump-allocator arena. Backed by a single contiguous buffer; allocations
// are aligned to 64 bytes (matches ARM cache line + most NEON intrinsic
// alignment requirements). `reset()` rewinds without freeing.
class sp_arena {
public:
    sp_arena() = default;
    explicit sp_arena(size_t bytes) { reserve(bytes); }
    ~sp_arena();

    sp_arena(const sp_arena&)            = delete;
    sp_arena& operator=(const sp_arena&) = delete;
    sp_arena(sp_arena&& o) noexcept;
    sp_arena& operator=(sp_arena&& o) noexcept;

    // Reserve `bytes` of backing storage. Idempotent if current buffer
    // is already at least that large.
    void reserve(size_t bytes);

    // Aligned allocation; returns nullptr if the request would overflow
    // the reserved buffer (callers should over-reserve generously).
    void* alloc(size_t bytes, size_t alignment = 64);

    // Allocate space for a tensor and bind it: returns false if there
    // isn't room (caller should reserve more or split into smaller
    // sub-graphs). The tensor's shape/strides/dtype must already be set
    // via sp_tensor::reset(); this only assigns `data`.
    bool alloc_tensor(sp_tensor& t);

    // Rewind to empty without freeing the backing buffer.
    void reset() { used_ = 0; }

    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }
    size_t remaining() const { return capacity_ - used_; }

private:
    uint8_t* buf_      = nullptr;
    size_t   capacity_ = 0;
    size_t   used_     = 0;
};

}  // namespace sp::engine
