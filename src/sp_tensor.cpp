// sp_tensor implementation. See sp_tensor.h for the why.

#include "sp_tensor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace sp::engine {

// Platform-portable aligned alloc/free. std::aligned_alloc is C++17
// but MSVC didn't ship it until very late; use _aligned_malloc there.
static void* sp_aligned_alloc_impl(size_t alignment, size_t bytes) {
#if defined(_WIN32)
    // Both MSVC and MinGW provide _aligned_malloc.
    return _aligned_malloc(bytes, alignment);
#else
    // POSIX aligned_alloc requires bytes to be a multiple of alignment.
    size_t round = (bytes + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, round);
#endif
}
static void sp_aligned_free_impl(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void sp_tensor::reset(sp_dtype t, int nd, const int64_t s[MAX_DIMS]) {
    dtype  = t;
    n_dims = nd;
    for (int i = 0; i < MAX_DIMS; ++i) {
        shape[i]   = (i < nd) ? s[i] : 1;
        strides[i] = 0;
    }

    // Compute contiguous strides (in bytes). For quantized types,
    // shape[0] (the inner dim) MUST be a multiple of block_elements;
    // otherwise the layout is undefined. The caller is responsible.
    sp_dtype_traits info = sp_dtype_info(t);
    size_t row_bytes;
    if (info.is_quantized) {
        // Inner dim measured in BLOCKS, not elements, for stride math.
        int64_t inner_blocks =
            (shape[0] + (int64_t)info.block_elements - 1)
            / (int64_t)info.block_elements;
        row_bytes  = (size_t)inner_blocks * info.block_bytes;
    } else {
        row_bytes  = (size_t)shape[0] * info.block_bytes;
    }
    strides[0] = info.block_bytes;        // bytes between adjacent elements (block-bytes for quant)
    strides[1] = row_bytes;               // bytes between adjacent rows
    strides[2] = strides[1] * (size_t)shape[1];
    strides[3] = strides[2] * (size_t)shape[2];
}

bool sp_tensor::is_contiguous() const {
    sp_dtype_traits info = sp_dtype_info(dtype);
    if (info.block_elements == 0) return false;
    size_t expected_inner = info.block_bytes;
    if (strides[0] != expected_inner) return false;
    size_t expected;
    if (info.is_quantized) {
        int64_t inner_blocks =
            (shape[0] + (int64_t)info.block_elements - 1)
            / (int64_t)info.block_elements;
        expected = (size_t)inner_blocks * info.block_bytes;
    } else {
        expected = (size_t)shape[0] * info.block_bytes;
    }
    if (strides[1] != expected) return false;
    expected *= (size_t)shape[1];
    if (strides[2] != expected) return false;
    expected *= (size_t)shape[2];
    if (strides[3] != expected) return false;
    return true;
}

// ---------------------------------------------------------------------
// sp_arena
// ---------------------------------------------------------------------

sp_arena::~sp_arena() {
    if (buf_) sp_aligned_free_impl(buf_);
}

sp_arena::sp_arena(sp_arena&& o) noexcept
    : buf_(o.buf_), capacity_(o.capacity_), used_(o.used_) {
    o.buf_ = nullptr; o.capacity_ = 0; o.used_ = 0;
}

sp_arena& sp_arena::operator=(sp_arena&& o) noexcept {
    if (this != &o) {
        if (buf_) sp_aligned_free_impl(buf_);
        buf_ = o.buf_; capacity_ = o.capacity_; used_ = o.used_;
        o.buf_ = nullptr; o.capacity_ = 0; o.used_ = 0;
    }
    return *this;
}

void sp_arena::reserve(size_t bytes) {
    if (bytes <= capacity_) return;
    // Grow to requested size; round up to a 4 KB page boundary so we
    // play nice with mmap/madvise downstream if we ever swap to that.
    constexpr size_t PAGE = 4096;
    size_t new_cap = (bytes + PAGE - 1) & ~(PAGE - 1);
    void* p = sp_aligned_alloc_impl(64, new_cap);
    if (!p) return;
    if (buf_ && used_) std::memcpy(p, buf_, used_);
    if (buf_) sp_aligned_free_impl(buf_);
    buf_ = (uint8_t*)p;
    capacity_ = new_cap;
}

void* sp_arena::alloc(size_t bytes, size_t alignment) {
    // Round used_ up to the alignment boundary, then check fit.
    size_t aligned = (used_ + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > capacity_) return nullptr;
    void* p = buf_ + aligned;
    used_ = aligned + bytes;
    return p;
}

bool sp_arena::alloc_tensor(sp_tensor& t) {
    size_t need = t.nbytes_contig();
    if (need == 0) return false;
    void* p = alloc(need, 64);
    if (!p) return false;
    t.data = p;
    return true;
}

}  // namespace sp::engine
