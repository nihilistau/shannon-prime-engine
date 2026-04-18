// Shannon-Prime Engine — vocabulary reader implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "vocab.h"
#include "gguf_loader.h"

#include "gguf.h"

#include <cstdio>
#include <sstream>

namespace sp::engine {

Vocab::~Vocab() = default;

static const std::string kEmpty;

const std::string& Vocab::token(int32_t id) const {
    if (id < 0 || (size_t)id >= tokens_.size()) return kEmpty;
    return tokens_[(size_t)id];
}

TokenType Vocab::token_type(int32_t id) const {
    if (id < 0 || (size_t)id >= token_types_.size()) return TokenType::UNDEFINED;
    return token_types_[(size_t)id];
}

float Vocab::token_score(int32_t id) const {
    if (id < 0 || (size_t)id >= scores_.size()) return 0.0f;
    return scores_[(size_t)id];
}

int32_t Vocab::find(const std::string& piece) const {
    auto it = by_piece_.find(piece);
    return (it == by_piece_.end()) ? -1 : it->second;
}

int32_t Vocab::merge_rank(const std::string& left, const std::string& right) const {
    std::string key = left;
    key.push_back('\0');
    key.append(right);
    auto it = merge_ranks_.find(key);
    return (it == merge_ranks_.end()) ? -1 : it->second;
}

void Vocab::print_summary(std::FILE* f) const {
    std::fprintf(f, "Tokenizer:\n");
    std::fprintf(f, "  model:  %s\n",   model_.empty() ? "(unknown)" : model_.c_str());
    std::fprintf(f, "  pre:    %s\n",   pre_.empty()   ? "(unknown)" : pre_.c_str());
    std::fprintf(f, "  size:   %zu tokens\n", tokens_.size());
    std::fprintf(f, "  merges: %zu\n", merges_.size());
    std::fprintf(f, "  BOS=%d EOS=%d PAD=%d UNK=%d\n",
                 bos_id_, eos_id_, pad_id_, unk_id_);
}

// ------------------------------------------------------------------
// Load
// ------------------------------------------------------------------
std::unique_ptr<Vocab> Vocab::load(const Model& model) {
    auto* g = reinterpret_cast<gguf_context*>(model._gguf_context_opaque());
    if (!g) return nullptr;

    auto v = std::unique_ptr<Vocab>(new Vocab());

    // Tokenizer family names.
    v->model_ = model.get_str("tokenizer.ggml.model");
    v->pre_   = model.get_str("tokenizer.ggml.pre");

    // Token strings — required.
    int64_t tokens_id = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tokens_id < 0 || gguf_get_kv_type(g, tokens_id) != GGUF_TYPE_ARRAY) {
        std::fprintf(stderr, "[sp-engine] GGUF has no tokenizer.ggml.tokens array\n");
        return nullptr;
    }
    if (gguf_get_arr_type(g, tokens_id) != GGUF_TYPE_STRING) {
        std::fprintf(stderr, "[sp-engine] tokens array is not GGUF_TYPE_STRING\n");
        return nullptr;
    }
    const size_t n_tokens = gguf_get_arr_n(g, tokens_id);
    v->tokens_.reserve(n_tokens);
    v->by_piece_.reserve(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        const char* s = gguf_get_arr_str(g, tokens_id, i);
        v->tokens_.emplace_back(s ? s : "");
        v->by_piece_.emplace(v->tokens_.back(), (int32_t)i);
    }

    // Token types (optional but usually present).
    v->token_types_.resize(n_tokens, TokenType::NORMAL);
    int64_t types_id = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (types_id >= 0
        && gguf_get_kv_type(g, types_id) == GGUF_TYPE_ARRAY
        && gguf_get_arr_type(g, types_id) == GGUF_TYPE_INT32
        && gguf_get_arr_n(g, types_id) == n_tokens) {
        const int32_t* data =
            static_cast<const int32_t*>(gguf_get_arr_data(g, types_id));
        for (size_t i = 0; i < n_tokens; ++i) {
            v->token_types_[i] = static_cast<TokenType>((int8_t)data[i]);
        }
    }

    // Scores (optional; used by SentencePiece flavours).
    v->scores_.resize(n_tokens, 0.0f);
    int64_t scores_id = gguf_find_key(g, "tokenizer.ggml.scores");
    if (scores_id >= 0
        && gguf_get_kv_type(g, scores_id) == GGUF_TYPE_ARRAY
        && gguf_get_arr_type(g, scores_id) == GGUF_TYPE_FLOAT32
        && gguf_get_arr_n(g, scores_id) == n_tokens) {
        const float* data = static_cast<const float*>(gguf_get_arr_data(g, scores_id));
        for (size_t i = 0; i < n_tokens; ++i) v->scores_[i] = data[i];
    }

    // Merges (required for BPE). Each entry is a string of form "left right".
    int64_t merges_id = gguf_find_key(g, "tokenizer.ggml.merges");
    if (merges_id >= 0
        && gguf_get_kv_type(g, merges_id) == GGUF_TYPE_ARRAY
        && gguf_get_arr_type(g, merges_id) == GGUF_TYPE_STRING) {
        const size_t n_merges = gguf_get_arr_n(g, merges_id);
        v->merges_.reserve(n_merges);
        v->merge_ranks_.reserve(n_merges);
        for (size_t i = 0; i < n_merges; ++i) {
            const char* m = gguf_get_arr_str(g, merges_id, i);
            if (!m) continue;
            std::string s(m);
            auto sp = s.find(' ');
            if (sp == std::string::npos) continue;
            std::string left  = s.substr(0, sp);
            std::string right = s.substr(sp + 1);
            v->merges_.emplace_back(left, right);
            std::string key = left;
            key.push_back('\0');
            key.append(right);
            v->merge_ranks_.emplace(std::move(key), (int32_t)i);
        }
    }

    // Special IDs.
    v->bos_id_ = (int32_t)model.get_i64("tokenizer.ggml.bos_token_id", -1);
    v->eos_id_ = (int32_t)model.get_i64("tokenizer.ggml.eos_token_id", -1);
    v->pad_id_ = (int32_t)model.get_i64("tokenizer.ggml.padding_token_id", -1);
    v->unk_id_ = (int32_t)model.get_i64("tokenizer.ggml.unknown_token_id", -1);

    return v;
}

} // namespace sp::engine
