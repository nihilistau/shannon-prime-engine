// Shannon-Prime Engine — vocabulary reader
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Reads tokens, merges, scores, and token-type metadata from a GGUF
// and exposes them as typed C++ containers. The actual BPE encode/
// decode logic sits in a follow-up (tokenizer.{h,cpp}) — this file
// is pure data plumbing so future commits can reuse the same Vocab
// across different tokenizer flavours (BPE, SentencePiece, etc).

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sp::engine {

class Model;  // forward decl

enum class TokenType : int8_t {
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

class Vocab {
public:
    // Load vocab from the GGUF behind `model`. Returns nullptr if the
    // GGUF doesn't carry a tokenizer section or if it's empty.
    static std::unique_ptr<Vocab> load(const Model& model);

    ~Vocab();
    Vocab(const Vocab&) = delete;
    Vocab& operator=(const Vocab&) = delete;

    // --- Sizes ---
    size_t size()       const { return tokens_.size(); }
    size_t n_merges()   const { return merges_.size(); }

    // --- Token lookup ---
    const std::string& token(int32_t id) const;           // empty string if out of range
    TokenType          token_type(int32_t id) const;
    float              token_score(int32_t id) const;
    int32_t            find(const std::string& piece) const; // -1 if absent

    // --- Merge lookup (BPE pair -> merged token id) -------------------
    // Returns the merge rank (0 = highest priority), or -1 if the pair
    // is not a known merge.
    int32_t merge_rank(const std::string& left, const std::string& right) const;

    // --- Special token IDs (may be -1 if not defined) ----------------
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t unk_id() const { return unk_id_; }

    // Which tokenizer model the GGUF says it is ("gpt2", "llama", ...).
    const std::string& pre()   const { return pre_; }
    const std::string& model() const { return model_; }

    // Print a brief summary for the `info` CLI verb.
    void print_summary(std::FILE* f) const;

private:
    Vocab() = default;

    std::vector<std::string>                  tokens_;
    std::vector<TokenType>                    token_types_;
    std::vector<float>                        scores_;
    // Merge-pair → rank (index). Stored as "left\0right" concatenation
    // for a cheap std::string key. Ranks come straight from the GGUF
    // merges array order (earlier = higher priority).
    std::unordered_map<std::string, int32_t>  merge_ranks_;
    // Flat copy of the raw merges for iteration / diagnostics.
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t>  by_piece_;

    std::string pre_;
    std::string model_;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;
};

} // namespace sp::engine
