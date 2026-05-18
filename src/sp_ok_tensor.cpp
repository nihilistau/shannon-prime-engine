// Shannon-Prime Engine — O_K-coordinate tensor (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_ok_tensor.h"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

// Phase 2.3b iter 5: sp_ok_arena's backing storage on Windows uses
// VirtualAlloc directly so multi-GB allocations don't go through the
// CRT heap. The CRT heap fragments badly after ~10 GB of large allocs
// and refuses further requests even when total free RAM is sufficient
// (observed at L=24/26 on Gemma3-1B with 18.8 GB free). VirtualAlloc
// reserves directly from the OS page allocator, sidestepping the
// fragmentation. POSIX path keeps std::malloc.
namespace {
inline void* sp_ok_arena_alloc_bytes(size_t bytes) {
#ifdef _WIN32
    void* p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE,
                            PAGE_READWRITE);
    return p;
#else
    return std::malloc(bytes);
#endif
}
inline void sp_ok_arena_free_bytes(void* p, size_t bytes) {
#ifdef _WIN32
    if (p) VirtualFree(p, 0, MEM_RELEASE);
    (void)bytes;
#else
    (void)bytes;
    std::free(p);
#endif
}
}  // anon namespace

namespace sp::engine {

// =========================================================================
// sp_ok_tensor
// =========================================================================

void sp_ok_tensor::reset(int nd, const int64_t s[MAX_DIMS]) {
    n_dims = nd;
    for (int i = 0; i < MAX_DIMS; ++i) {
        shape[i] = (i < nd) ? s[i] : 1;
    }
    // Compute contiguous strides in bytes (innermost = sizeof(sp_ok_t)).
    size_t st = sizeof(sp_ok_t);
    for (int i = 0; i < nd; ++i) {
        strides[i] = st;
        st *= (size_t)shape[i];
    }
    for (int i = nd; i < MAX_DIMS; ++i) {
        strides[i] = st;
    }
    scale_recip = 1;
    frobenius_scale = 1;
}

bool sp_ok_tensor::is_contiguous() const {
    size_t expected = sizeof(sp_ok_t);
    for (int i = 0; i < n_dims; ++i) {
        if (strides[i] != expected) return false;
        expected *= (size_t)shape[i];
    }
    return true;
}

// =========================================================================
// sp_ok_arena
// =========================================================================

sp_ok_arena::~sp_ok_arena() {
    sp_ok_arena_free_bytes(buf_, capacity_);
    buf_      = nullptr;
    capacity_ = 0;
    used_     = 0;
}

sp_ok_arena::sp_ok_arena(sp_ok_arena&& o) noexcept
    : buf_(o.buf_), capacity_(o.capacity_), used_(o.used_) {
    o.buf_ = nullptr;
    o.capacity_ = 0;
    o.used_ = 0;
}

sp_ok_arena& sp_ok_arena::operator=(sp_ok_arena&& o) noexcept {
    if (this != &o) {
        std::free(buf_);
        buf_ = o.buf_;
        capacity_ = o.capacity_;
        used_ = o.used_;
        o.buf_ = nullptr;
        o.capacity_ = 0;
        o.used_ = 0;
    }
    return *this;
}

void sp_ok_arena::reserve(size_t bytes) {
    if (bytes <= capacity_) return;
    // Round up to 64-byte multiple for alignment (VirtualAlloc rounds
    // further up to page boundary, which is fine).
    bytes = (bytes + 63) & ~(size_t)63;
    uint8_t* nb = static_cast<uint8_t*>(sp_ok_arena_alloc_bytes(bytes));
    if (!nb) throw std::bad_alloc{};
    if (buf_) {
        std::memcpy(nb, buf_, used_);
        sp_ok_arena_free_bytes(buf_, capacity_);
    }
    buf_      = nb;
    capacity_ = bytes;
}

void* sp_ok_arena::alloc(size_t bytes, size_t alignment) {
    if (!buf_) return nullptr;
    // Align used_ up to `alignment`.
    size_t mis = used_ & (alignment - 1);
    size_t pad = mis ? (alignment - mis) : 0;
    if (used_ + pad + bytes > capacity_) return nullptr;
    used_ += pad;
    void* p = buf_ + used_;
    used_ += bytes;
    return p;
}

bool sp_ok_arena::alloc_tensor(sp_ok_tensor& t) {
    size_t bytes = t.nbytes_contig();
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.data = static_cast<sp_ok_t*>(p);
    return true;
}

bool sp_ok_arena::alloc_tensor_q8(sp_ok_q8_tensor& t, size_t numel) {
    if (numel == 0) {
        t.data  = nullptr;
        t.numel = 0;
        return true;
    }
    size_t bytes = numel * sizeof(sp_ok_q8_t);
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.data           = static_cast<sp_ok_q8_t*>(p);
    t.numel          = numel;
    t.q8_shift       = 0;
    t.scale_recip    = 1;
    t.frobenius_scale = 1;
    t.frobenius_p    = 0;
    t.frobenius_k    = 0;
    return true;
}

bool sp_ok_arena::alloc_tensor_block_q8(sp_ok_block_q8_tensor& t, size_t numel) {
    if (numel == 0) {
        t.blocks   = nullptr;
        t.numel    = 0;
        t.n_blocks = 0;
        return true;
    }
    if ((numel % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t n_blocks = numel / SP_OK_BLOCK_SIZE;
    const size_t bytes    = n_blocks * sizeof(sp_ok_q8_block_t);
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.blocks      = static_cast<sp_ok_q8_block_t*>(p);
    t.numel       = numel;
    t.n_blocks    = n_blocks;
    t.frobenius_p = 0;
    t.frobenius_k = 0;
    t.reserved    = 0;
    return true;
}

bool sp_ok_arena::alloc_tensor_block_q4_1(sp_ok_block_q4_1_tensor& t, size_t numel) {
    if (numel == 0) {
        t.blocks   = nullptr;
        t.numel    = 0;
        t.n_blocks = 0;
        return true;
    }
    if ((numel % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t n_blocks = numel / SP_OK_BLOCK_SIZE;
    const size_t bytes    = n_blocks * sizeof(sp_ok_q4_1_block_t);
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.blocks      = static_cast<sp_ok_q4_1_block_t*>(p);
    t.numel       = numel;
    t.n_blocks    = n_blocks;
    t.frobenius_p = 0;
    t.frobenius_k = 0;
    t.reserved    = 0;
    return true;
}

bool sp_ok_arena::alloc_tensor_block_q4(sp_ok_block_q4_tensor& t, size_t numel) {
    if (numel == 0) {
        t.blocks   = nullptr;
        t.numel    = 0;
        t.n_blocks = 0;
        return true;
    }
    if ((numel % SP_OK_BLOCK_SIZE) != 0) return false;
    const size_t n_blocks = numel / SP_OK_BLOCK_SIZE;
    const size_t bytes    = n_blocks * sizeof(sp_ok_q4_block_t);
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.blocks      = static_cast<sp_ok_q4_block_t*>(p);
    t.numel       = numel;
    t.n_blocks    = n_blocks;
    t.frobenius_p = 0;
    t.frobenius_k = 0;
    t.reserved    = 0;
    return true;
}

bool sp_ok_arena::alloc_tensor_q4(sp_ok_q4_tensor& t, size_t numel) {
    if (numel == 0) {
        t.data  = nullptr;
        t.numel = 0;
        return true;
    }
    size_t bytes = numel * sizeof(sp_ok_q4_t);
    void* p = alloc(bytes, 64);
    if (!p) return false;
    t.data           = static_cast<sp_ok_q4_t*>(p);
    t.numel          = numel;
    t.q4_shift       = 0;
    t.scale_recip    = 1;
    t.frobenius_scale = 1;
    t.frobenius_p    = 0;
    t.frobenius_k    = 0;
    return true;
}

// =========================================================================
// Helper functions
// =========================================================================

void sp_ok_tensor_scalar_mul(sp_ok_tensor& t, sp_ok_t scalar) {
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = sp_ok_mul(t.data[i], scalar);
    }
}

bool sp_ok_tensor_add_inplace(sp_ok_tensor& t, const sp_ok_tensor& other) {
    int64_t n = t.numel();
    int64_t no = other.numel();
    if (no != n) return false;
    if (t.scale_recip      != other.scale_recip)      return false;
    if (t.frobenius_scale  != other.frobenius_scale)  return false;
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = sp_ok_add(t.data[i], other.data[i]);
    }
    return true;
}

void sp_ok_tensor_negate(sp_ok_tensor& t) {
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = sp_ok_neg(t.data[i]);
    }
}

int64_t sp_ok_tensor_sum_norms(const sp_ok_tensor& t) {
    int64_t n = t.numel();
    int64_t s = 0;
    for (int64_t i = 0; i < n; ++i) {
        s += sp_ok_norm(t.data[i]);
    }
    return s;
}

}  // namespace sp::engine
