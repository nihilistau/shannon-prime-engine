// Shannon-Prime Engine — O_K-coordinate tensor.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 1.2 of the Theory-First engine. Parallel to sp_tensor but every
// element is an sp_ok_t (a + b*omega where omega = (1+sqrt(-163))/2,
// omega^2 = omega - 41). All hot-path arithmetic on these tensors is
// exact integer math via lib/shannon-prime/core/sp_ok_arith.h.
//
// Why a separate type from sp_tensor:
//   - element size: 16 bytes (two int64), not 2/4/8
//   - layout: AoS (interleaved a,b) for now; SoA path reserved for
//     a Phase 3 CUDA optimization (stride-1 coalesced loads)
//   - operations: all Frobenius / Sato-Tate dispatch lives here
//   - convertibility: per-tensor scale `S` for fp16 <-> O_K round-trip
//     (see sp_ok_encode.h)
//
// The convention follows sp_tensor: shape[0] is the inner-most
// contiguous dim; up to 4 dims; strides in BYTES so non-contiguous
// views work without element-size juggling.

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_ok_q8.h"
}

namespace sp::engine {

// -----------------------------------------------------------------------
// sp_ok_tensor — O_K-element tensor descriptor.
// -----------------------------------------------------------------------
struct sp_ok_tensor {
    static constexpr int MAX_DIMS = 4;

    sp_ok_t* data           = nullptr;          // n_rows * n_cols * ... sp_ok_t elements
    int      n_dims         = 0;
    int64_t  shape[MAX_DIMS]   = {1, 1, 1, 1};
    size_t   strides[MAX_DIMS] = {0, 0, 0, 0};  // BYTES
    const char* name        = nullptr;          // weak, optional

    // Per-tensor scale used by the fp16<->O_K encoding (sp_ok_encode.h).
    // For pure-integer tensors not bridging to fp16, leave as 1.
    // Encoding: data[i].a = (int64_t)round(fp16_value[i] * scale_recip);
    // Decoding: fp16_value[i] = (float)data[i].a / scale_recip.
    int64_t  scale_recip    = 1;

    // After Frobenius application, the scale_recip is multiplied by
    // |phi_p^k_a|: |-p|^(k/2) for inert, |Tr(pi^k)| or N(pi)^(k/2) for split.
    // Tracked so decode produces the original fp16 values (Theorem 4
    // commutativity guarantees this is exact within the int64 dynamic range).
    int64_t  frobenius_scale = 1;

    // ----- shape / size ---------------------------------------------------
    inline int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < n_dims; ++i) n *= shape[i];
        return n;
    }
    inline size_t nbytes_contig() const {
        return (size_t)numel() * sizeof(sp_ok_t);
    }
    bool is_contiguous() const;

    // Set up shape + compute contiguous strides. Caller assigns data
    // afterwards (typically from an arena).
    void reset(int nd, const int64_t s[MAX_DIMS]);

    // Element access (contiguous-row-major assumption).
    inline sp_ok_t& at(int64_t i)                       { return data[i]; }
    inline sp_ok_t& at(int64_t i, int64_t j)            { return data[i * shape[0] + j]; }
    inline const sp_ok_t& at(int64_t i) const           { return data[i]; }
    inline const sp_ok_t& at(int64_t i, int64_t j) const { return data[i * shape[0] + j]; }
};

// -----------------------------------------------------------------------
// sp_ok_arena — bump allocator for sp_ok_tensor backing storage.
// Aligned to 64 bytes (matches sp_arena convention).
// -----------------------------------------------------------------------
class sp_ok_arena {
public:
    sp_ok_arena() = default;
    explicit sp_ok_arena(size_t bytes) { reserve(bytes); }
    ~sp_ok_arena();

    sp_ok_arena(const sp_ok_arena&)            = delete;
    sp_ok_arena& operator=(const sp_ok_arena&) = delete;
    sp_ok_arena(sp_ok_arena&& o) noexcept;
    sp_ok_arena& operator=(sp_ok_arena&& o) noexcept;

    // Reserve at least `bytes` of backing storage. Idempotent.
    void reserve(size_t bytes);

    // Aligned allocation. Returns nullptr if it would overflow.
    void* alloc(size_t bytes, size_t alignment = 64);

    // Allocate space for a tensor whose shape is already set. Returns
    // false if the arena doesn't have room.
    bool alloc_tensor(sp_ok_tensor& t);

    // Phase 12 Step B: packed-int8 ring elements. The descriptor's numel
    // and shape come from the matching sp_ok_tensor; this call sizes the
    // backing storage at numel * sizeof(sp_ok_q8_t) = 2 * numel bytes and
    // returns the descriptor with `data` pointing into the arena. Per-tensor
    // metadata (q8_shift, scale_recip, frobenius_scale, p, k) is the
    // caller's responsibility — typically set by the encoder. Returns
    // false on arena exhaustion. */
    bool alloc_tensor_q8(sp_ok_q8_tensor& t, size_t numel);

    void   reset()              { used_ = 0; }
    size_t capacity() const     { return capacity_; }
    size_t used()     const     { return used_; }
    size_t remaining() const    { return capacity_ - used_; }

private:
    uint8_t* buf_      = nullptr;
    size_t   capacity_ = 0;
    size_t   used_     = 0;
};

// -----------------------------------------------------------------------
// Helper functions on sp_ok_tensor.
// -----------------------------------------------------------------------

// In-place element-wise multiplication by a scalar sp_ok element.
//   t[i] = t[i] * scalar
void sp_ok_tensor_scalar_mul(sp_ok_tensor& t, sp_ok_t scalar);

// In-place element-wise addition.
//   t[i] = t[i] + other[i]
// Shapes, scale_recip, AND frobenius_scale must all match — otherwise
// the addition is semantically invalid (Theorem 4 cancellation only
// works when both operands share the same scale). Returns false on
// any mismatch and leaves `t` unchanged.
//
// For residual stream + (Wo @ attn_out) where the projections produce
// pi^k-scaled outputs, the caller must first decode to fp32 (via
// sp_matmul_ok_to_fp32) and add in fp32 -- DO NOT use this function
// across mismatched scales.
bool sp_ok_tensor_add_inplace(sp_ok_tensor& t, const sp_ok_tensor& other);

// In-place negation.
void sp_ok_tensor_negate(sp_ok_tensor& t);

// Sum of norms -- useful as a sanity-check invariant (norm scales
// predictably under Frobenius, see test_sp_frobenius.cpp).
int64_t sp_ok_tensor_sum_norms(const sp_ok_tensor& t);

}  // namespace sp::engine
