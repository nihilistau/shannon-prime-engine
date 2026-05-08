// Shannon-Prime Engine — GGUF loader (implementation)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "gguf_loader.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>

namespace sp::engine {

struct Model::Impl {
    gguf_context* gguf = nullptr;
    ~Impl() {
        if (gguf) gguf_free(gguf);
    }
};

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
static std::string gguf_try_str(gguf_context* g, const char* key,
                                const std::string& fallback = "") {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    const gguf_type t = gguf_get_kv_type(g, id);
    if (t != GGUF_TYPE_STRING) return fallback;
    return std::string(gguf_get_val_str(g, id));
}

static int64_t gguf_try_i64(gguf_context* g, const char* key,
                            int64_t fallback = 0) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  return (int64_t)gguf_get_val_u8 (g, id);
        case GGUF_TYPE_INT8:   return (int64_t)gguf_get_val_i8 (g, id);
        case GGUF_TYPE_UINT16: return (int64_t)gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return (int64_t)gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32: return (int64_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (int64_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int64_t)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return          gguf_get_val_i64(g, id);
        default:               return fallback;
    }
}

static double gguf_try_f64(gguf_context* g, const char* key,
                           double fallback = 0.0) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_FLOAT32: return (double)gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64: return          gguf_get_val_f64(g, id);
        default:                return fallback;
    }
}

// ------------------------------------------------------------------
// load()
// ------------------------------------------------------------------
std::unique_ptr<Model> Model::load(const std::string& path) {
    gguf_init_params params = {};
    params.no_alloc = true;     // don't load tensor data eagerly
    params.ctx      = nullptr;

    gguf_context* g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        std::fprintf(stderr, "[sp-engine] failed to open GGUF: %s\n", path.c_str());
        return nullptr;
    }

    auto m = std::unique_ptr<Model>(new Model());
    m->impl_->gguf = g;
    m->path_         = path;
    m->gguf_version_ = gguf_get_version(g);
    m->n_kv_         = gguf_get_n_kv(g);
    m->n_tensors_    = gguf_get_n_tensors(g);

    m->arch_ = gguf_try_str(g, "general.architecture");
    m->name_ = gguf_try_str(g, "general.name", "(unnamed)");
    if (m->arch_.empty()) {
        std::fprintf(stderr,
            "[sp-engine] GGUF missing 'general.architecture' — not a model file?\n");
        return nullptr;
    }

    // llama-family share most hparam keys under "<arch>.*" prefix.
    const std::string p = m->arch_ + ".";
    m->vocab_size_      = (uint32_t)gguf_try_i64(g, (p + "vocab_size").c_str(), 0);
    if (m->vocab_size_ == 0) {
        // Some GGUFs store it under the tokenizer section instead.
        m->vocab_size_ = (uint32_t)gguf_try_i64(g, "tokenizer.ggml.vocab_size", 0);
    }
    m->n_layer_         = (uint32_t)gguf_try_i64(g, (p + "block_count").c_str(), 0);
    m->n_embd_          = (uint32_t)gguf_try_i64(g, (p + "embedding_length").c_str(), 0);
    m->n_head_          = (uint32_t)gguf_try_i64(g, (p + "attention.head_count").c_str(), 0);
    m->n_head_kv_       = (uint32_t)gguf_try_i64(g, (p + "attention.head_count_kv").c_str(), m->n_head_);
    m->rope_dim_count_  = (uint32_t)gguf_try_i64(g, (p + "rope.dimension_count").c_str(), 0);
    m->rope_freq_base_  = (float)   gguf_try_f64(g, (p + "rope.freq_base").c_str(), 10000.0);
    m->context_length_  = (uint32_t)gguf_try_i64(g, (p + "context_length").c_str(), 0);

    // Cache tensor names for quick lookup.
    m->tensor_names_.reserve((size_t)m->n_tensors_);
    for (int64_t i = 0; i < m->n_tensors_; ++i) {
        m->tensor_names_.emplace_back(gguf_get_tensor_name(g, i));
    }

    return m;
}

// ------------------------------------------------------------------
// Accessors
// ------------------------------------------------------------------
uint32_t Model::head_dim() const {
    if (n_head_ == 0) return 0;
    // Prefer the explicit key if the arch ships it; otherwise derive.
    int64_t id = gguf_find_key(impl_->gguf, (arch_ + ".attention.key_length").c_str());
    if (id >= 0) return (uint32_t)gguf_try_i64(impl_->gguf, (arch_ + ".attention.key_length").c_str(), 0);
    return n_embd_ / n_head_;
}

int64_t Model::find_key(const std::string& key) const {
    return gguf_find_key(impl_->gguf, key.c_str());
}
std::string Model::get_str(const std::string& key, const std::string& fallback) const {
    return gguf_try_str(impl_->gguf, key.c_str(), fallback);
}
int64_t Model::get_i64(const std::string& key, int64_t fallback) const {
    return gguf_try_i64(impl_->gguf, key.c_str(), fallback);
}
double Model::get_f64(const std::string& key, double fallback) const {
    return gguf_try_f64(impl_->gguf, key.c_str(), fallback);
}

std::vector<int32_t> Model::get_i32_array(const std::string& key) const {
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(impl_->gguf, key.c_str());
    if (id < 0) return out;
    if (gguf_get_kv_type(impl_->gguf, id) != GGUF_TYPE_ARRAY) return out;

    const gguf_type et = gguf_get_arr_type(impl_->gguf, id);
    const size_t    n  = gguf_get_arr_n   (impl_->gguf, id);
    const void*     d  = gguf_get_arr_data(impl_->gguf, id);
    if (!d || n == 0) return out;

    out.resize(n);
    switch (et) {
        case GGUF_TYPE_INT8:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const int8_t*)d)[i];
            break;
        case GGUF_TYPE_UINT8:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const uint8_t*)d)[i];
            break;
        case GGUF_TYPE_INT16:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const int16_t*)d)[i];
            break;
        case GGUF_TYPE_UINT16:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const uint16_t*)d)[i];
            break;
        case GGUF_TYPE_INT32:
            std::memcpy(out.data(), d, n * sizeof(int32_t));
            break;
        case GGUF_TYPE_UINT32:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const uint32_t*)d)[i];
            break;
        case GGUF_TYPE_INT64:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const int64_t*)d)[i];
            break;
        case GGUF_TYPE_UINT64:
            for (size_t i = 0; i < n; ++i) out[i] = (int32_t)((const uint64_t*)d)[i];
            break;
        default:
            out.clear();
            break;
    }
    return out;
}

Model::TensorInfo Model::tensor_info(size_t i) const {
    TensorInfo ti;
    if (i >= tensor_names_.size()) return ti;
    ti.name       = tensor_names_[i];
    ti.type       = (int32_t)gguf_get_tensor_type(impl_->gguf, (int64_t)i);
    ti.n_bytes    = gguf_get_tensor_size(impl_->gguf, (int64_t)i);
    // gguf_get_tensor_size returns bytes; we don't have a cheap way to
    // recover element count without the shape, which lives in the ggml
    // metadata tensor objects (not plain gguf KV). Leave n_elements 0
    // until the ggml-ctx load path is wired.
    ti.n_elements = 0;
    return ti;
}

int64_t Model::find_tensor(const std::string& name) const {
    return gguf_find_tensor(impl_->gguf, name.c_str());
}

void* Model::_gguf_context_opaque() const {
    return reinterpret_cast<void*>(impl_->gguf);
}

// ------------------------------------------------------------------
// Pretty-print
// ------------------------------------------------------------------
void Model::print_summary(std::FILE* f) const {
    std::fprintf(f, "GGUF: %s\n", path_.c_str());
    std::fprintf(f, "  version:       %u\n", gguf_version_);
    std::fprintf(f, "  architecture:  %s\n", arch_.c_str());
    std::fprintf(f, "  name:          %s\n", name_.c_str());
    std::fprintf(f, "  n_kv:          %lld\n", (long long)n_kv_);
    std::fprintf(f, "  n_tensors:     %lld\n", (long long)n_tensors_);
    std::fprintf(f, "  ----\n");
    std::fprintf(f, "  n_layer:         %u\n", n_layer_);
    std::fprintf(f, "  n_embd:          %u\n", n_embd_);
    std::fprintf(f, "  n_head:          %u\n", n_head_);
    std::fprintf(f, "  n_head_kv:       %u (GQA ratio %.0fx)\n",
                 n_head_kv_, n_head_kv_ ? (double)n_head_ / n_head_kv_ : 0.0);
    std::fprintf(f, "  head_dim:        %u\n", head_dim());
    std::fprintf(f, "  vocab_size:      %u\n", vocab_size_);
    std::fprintf(f, "  context_length:  %u\n", context_length_);
    std::fprintf(f, "  rope_dim_count:  %u\n", rope_dim_count_);
    std::fprintf(f, "  rope_freq_base:  %.1f\n", rope_freq_base_);
}

} // namespace sp::engine
