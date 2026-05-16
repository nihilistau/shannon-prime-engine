// Shannon-Prime Engine — O_K-coordinate tensor (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_ok_tensor.h"

#include <cstdlib>
#include <cstring>
#include <new>

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
    std::free(buf_);
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
    // Round up to 64-byte multiple for alignment.
    bytes = (bytes + 63) & ~(size_t)63;
    uint8_t* nb = static_cast<uint8_t*>(std::malloc(bytes));
    if (!nb) throw std::bad_alloc{};
    if (buf_) {
        std::memcpy(nb, buf_, used_);
        std::free(buf_);
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

// =========================================================================
// Helper functions
// =========================================================================

void sp_ok_tensor_scalar_mul(sp_ok_tensor& t, sp_ok_t scalar) {
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = sp_ok_mul(t.data[i], scalar);
    }
}

void sp_ok_tensor_add_inplace(sp_ok_tensor& t, const sp_ok_tensor& other) {
    int64_t n = t.numel();
    int64_t no = other.numel();
    if (no != n) return;  // silent shape mismatch — caller should check
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = sp_ok_add(t.data[i], other.data[i]);
    }
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
