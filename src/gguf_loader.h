// Shannon-Prime Engine — GGUF loader (public API)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace sp::engine {

// A minimal typed view over a GGUF model file. Wraps ggml's gguf_context
// so the rest of the engine doesn't have to pattern-match raw KV IDs.
//
// Supports llama-family architectures today: llama, qwen2, qwen3, phi3,
// mistral. Other archs parse but only the shared hparams are populated;
// arch-specific fields may be zero — each model binding fills them in.
class Model {
public:
    // Load a GGUF file. Reads all KV metadata immediately; tensor data
    // stays mmapped and is lazily pulled via tensor() on demand.
    // Returns an empty unique_ptr on failure (bad file, unsupported
    // version, etc); writes a diagnostic to stderr.
    static std::unique_ptr<Model> load(const std::string& path);

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // --- Architecture / metadata ---------------------------------------
    const std::string& path()          const { return path_; }
    const std::string& architecture()  const { return arch_; }
    const std::string& name()          const { return name_; }
    uint32_t           gguf_version()  const { return gguf_version_; }
    int64_t            n_kv()          const { return n_kv_; }
    int64_t            n_tensors()     const { return n_tensors_; }

    // --- Shared llama-family hparams -----------------------------------
    uint32_t vocab_size()       const { return vocab_size_; }
    uint32_t n_layer()          const { return n_layer_; }
    uint32_t n_embd()           const { return n_embd_; }
    uint32_t n_head()           const { return n_head_; }
    uint32_t n_head_kv()        const { return n_head_kv_; }
    uint32_t head_dim()         const;   // derived: n_embd / n_head if not stored
    uint32_t rope_dim_count()   const { return rope_dim_count_; }
    float    rope_freq_base()   const { return rope_freq_base_; }
    uint32_t context_length()   const { return context_length_; }

    // --- Raw KV access (escape hatch for arch-specific keys) -----------
    // Returns the KV id, or -1 if the key is not present.
    int64_t     find_key(const std::string& key) const;
    std::string get_str(const std::string& key, const std::string& fallback = "") const;
    int64_t     get_i64(const std::string& key, int64_t fallback = 0) const;
    double      get_f64(const std::string& key, double  fallback = 0.0) const;
    // Reads an int-typed array (accepts i8/i16/i32/i64, u8/u16/u32/u64)
    // and returns it as int32_t. Returns an empty vector if the key is
    // missing or not an integer array. Used for things like qwen35moe's
    // `rope.dimension_sections` (4-element mRoPE section layout).
    std::vector<int32_t> get_i32_array(const std::string& key) const;

    // --- Tensor iteration ----------------------------------------------
    struct TensorInfo {
        std::string name;
        int32_t     type;        // ggml_type enum value
        uint64_t    n_elements;
        uint64_t    n_bytes;
    };
    size_t           tensor_count() const { return tensor_names_.size(); }
    TensorInfo       tensor_info(size_t i) const;
    int64_t          find_tensor(const std::string& name) const;

    // Pretty-print: one-line summary good enough for CLI `info` verb.
    void print_summary(std::FILE* f) const;

    // Escape hatch for companion classes (Vocab, ModelArchBinding, ...)
    // that need direct gguf_context access. Returns a `void*` so the
    // header doesn't pull ggml's types in; callers reinterpret-cast.
    // Stable while the Model object lives.
    void* _gguf_context_opaque() const;

private:
    Model();

    // Opaque gguf_context* under the hood; the header doesn't pull
    // ggml's C API in to keep the public surface clean.
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Cached hparams (filled at load()).
    std::string path_;
    std::string arch_;
    std::string name_;
    uint32_t    gguf_version_    = 0;
    int64_t     n_kv_            = 0;
    int64_t     n_tensors_       = 0;
    uint32_t    vocab_size_      = 0;
    uint32_t    n_layer_         = 0;
    uint32_t    n_embd_          = 0;
    uint32_t    n_head_          = 0;
    uint32_t    n_head_kv_       = 0;
    uint32_t    rope_dim_count_  = 0;
    float       rope_freq_base_  = 10000.0f;
    uint32_t    context_length_  = 0;
    std::vector<std::string> tensor_names_;
};

} // namespace sp::engine
