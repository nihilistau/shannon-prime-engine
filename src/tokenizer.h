// Shannon-Prime Engine — BPE tokenizer
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Byte-Pair-Encoding over a loaded Vocab. Covers the llama-bpe and
// qwen2 pre-tokenizer flavours that the current llama.cpp ships —
// both are GPT-2 style: apply a unicode byte-level remap, then
// merge pairs in rank order. SentencePiece models are not handled
// here (different surface); they'll need their own Tokenizer
// subclass when we broaden model support.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sp::engine {

class Vocab;

class Tokenizer {
public:
    // Create a tokenizer appropriate for vocab.pre() ∈ {"llama-bpe",
    // "qwen2", "default"}. Returns nullptr on unsupported pre().
    static std::unique_ptr<Tokenizer> create(const Vocab& vocab);

    virtual ~Tokenizer() = default;

    // Encode a UTF-8 string to token IDs. Optionally prepends the BOS
    // token. Returns true on success. Unknown bytes fall through to
    // the byte-level fallback that GPT-2 BPE always supports.
    virtual bool encode(const std::string& text,
                        bool add_bos,
                        std::vector<int32_t>& out) const = 0;

    // Decode token IDs back to UTF-8.
    virtual std::string decode(const std::vector<int32_t>& ids) const = 0;

    // Which pre-tokenizer we're implementing ("llama-bpe" / "qwen2").
    virtual const std::string& pre() const = 0;
};

} // namespace sp::engine
