// Shannon-Prime Engine — BPE tokenizer implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Covers the llama-bpe / qwen2 / default pre-tokenizers at "close
// enough for perplexity sanity" fidelity. The canonical llama.cpp
// uses architecture-specific regexes pulled from the GGUF
// tokenizer.ggml.pre field — those land in a follow-up when we need
// byte-for-byte agreement with llama.cpp for cross-validation.
//
// For now the word-splitter only differentiates between letters,
// digits, punctuation, and whitespace. That's good enough for the
// scaffolding milestone (round-trip text through vocab → IDs →
// text) and the bench-parity note will live on any perplexity
// number we publish through this engine until the regex is wired.

#include "tokenizer.h"
#include "vocab.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <utility>

namespace sp::engine {

// -------------------------------------------------------------------
// GPT-2 byte-to-unicode remap (matches Hugging Face's tokenizers crate
// and llama.cpp). Maps the 256 raw bytes to a set of visible unicode
// code points so BPE merges only operate on printable strings.
// -------------------------------------------------------------------
static const std::vector<uint32_t>& byte_to_unicode() {
    static const std::vector<uint32_t> table = [] {
        std::vector<uint32_t> t(256, 0);
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
        std::vector<uint32_t> cs(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back((uint32_t)(256 + n));
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) t[bs[i]] = cs[i];
        return t;
    }();
    return table;
}

static std::string encode_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back((char)cp);
    } else if (cp < 0x800) {
        s.push_back((char)(0xC0 | (cp >> 6)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back((char)(0xE0 | (cp >> 12)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        s.push_back((char)(0xF0 | (cp >> 18)));
        s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    }
    return s;
}

static std::string bytes_to_gpt2(const std::string& in) {
    const auto& table = byte_to_unicode();
    std::string out;
    out.reserve(in.size() * 2);
    for (unsigned char b : in) out += encode_utf8(table[b]);
    return out;
}

// Reverse map: build once, cache.
static const std::vector<int>& unicode_to_byte() {
    static const std::vector<int> rev = [] {
        std::vector<int> r(512, -1);
        const auto& t = byte_to_unicode();
        for (int b = 0; b < 256; ++b) {
            if (t[b] < r.size()) r[t[b]] = b;
        }
        return r;
    }();
    return rev;
}

// -------------------------------------------------------------------
// Minimal pre-tokenizer: split on whitespace / alnum / punctuation
// boundaries. Not byte-for-byte identical to llama.cpp's regex but
// close enough for sanity. Each returned word has a leading space
// preserved (GPT-2 convention).
// -------------------------------------------------------------------
static std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto is_alnum = [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z');
    };
    auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n'; };

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = (unsigned char)text[i];

        // leading-space mode — consume one space, then whatever follows
        if (is_space(c)) {
            cur.push_back(' ');
            ++i;
            if (i < text.size()) {
                unsigned char d = (unsigned char)text[i];
                if (is_alnum(d)) {
                    while (i < text.size() && is_alnum((unsigned char)text[i])) {
                        cur.push_back(text[i]); ++i;
                    }
                } else if (!is_space(d)) {
                    cur.push_back(text[i]); ++i;
                }
            }
            out.push_back(std::move(cur)); cur.clear();
            continue;
        }

        if (is_alnum(c)) {
            while (i < text.size() && is_alnum((unsigned char)text[i])) {
                cur.push_back(text[i]); ++i;
            }
            out.push_back(std::move(cur)); cur.clear();
            continue;
        }

        // single punctuation char / anything else
        cur.push_back(text[i]); ++i;
        out.push_back(std::move(cur)); cur.clear();
    }
    return out;
}

// -------------------------------------------------------------------
// BPE core: merge the symbol list in priority order.
// -------------------------------------------------------------------
struct BpeSym { std::string s; int prev, next; };

static std::vector<std::string> bpe_merge(const std::string& word,
                                          const Vocab& vocab) {
    // Each element is a utf-8 code point (or byte-group after remap);
    // we treat each UTF-8 codepoint as one symbol initially.
    std::vector<BpeSym> syms;
    size_t i = 0;
    while (i < word.size()) {
        // read one UTF-8 code point
        unsigned char c = (unsigned char)word[i];
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        BpeSym sym;
        sym.s = word.substr(i, len);
        sym.prev = (int)syms.size() - 1;
        sym.next = (int)syms.size() + 1;
        syms.push_back(std::move(sym));
        i += len;
    }
    if (syms.empty()) return {};
    syms.back().next = -1;

    struct QItem { int rank; int left; int right; std::string merged; };
    struct QCmp { bool operator()(const QItem& a, const QItem& b) const {
        return a.rank > b.rank;  // lower rank = higher priority
    }};
    std::priority_queue<QItem, std::vector<QItem>, QCmp> q;

    auto push_pair = [&](int l) {
        if (l < 0 || syms[l].next < 0) return;
        int r = syms[l].next;
        int rank = vocab.merge_rank(syms[l].s, syms[r].s);
        if (rank >= 0) {
            q.push({rank, l, r, syms[l].s + syms[r].s});
        }
    };

    for (int k = 0; k < (int)syms.size(); ++k) push_pair(k);

    while (!q.empty()) {
        QItem top = q.top(); q.pop();
        int l = top.left;
        int r = top.right;
        // skip stale entries (symbols were already merged into something else)
        if (syms[l].s.empty() || syms[r].s.empty()) continue;
        if (syms[l].next != r) continue;
        std::string combined = syms[l].s + syms[r].s;
        if (combined != top.merged) continue;

        // merge r into l
        syms[l].s = combined;
        syms[l].next = syms[r].next;
        if (syms[r].next >= 0) syms[syms[r].next].prev = l;
        syms[r].s.clear();

        // re-push neighbouring pairs
        push_pair(syms[l].prev);
        push_pair(l);
    }

    std::vector<std::string> pieces;
    for (int k = 0; k >= 0 && k < (int)syms.size();) {
        if (!syms[k].s.empty()) pieces.push_back(syms[k].s);
        k = syms[k].next;
    }
    return pieces;
}

// -------------------------------------------------------------------
// Concrete tokenizer implementation shared across pre-tokenizer
// flavours (differences live in pretokenize() — one-liner to swap).
// -------------------------------------------------------------------
class BpeTokenizer : public Tokenizer {
public:
    BpeTokenizer(const Vocab& v, std::string pre) : vocab_(v), pre_(std::move(pre)) {}

    bool encode(const std::string& text, bool add_bos,
                std::vector<int32_t>& out) const override {
        out.clear();
        if (add_bos && vocab_.bos_id() >= 0) out.push_back(vocab_.bos_id());

        auto words = pretokenize(text);
        for (const auto& w : words) {
            std::string remap = bytes_to_gpt2(w);
            auto pieces = bpe_merge(remap, vocab_);
            for (const auto& p : pieces) {
                int32_t id = vocab_.find(p);
                if (id < 0) {
                    // Byte-level fallback: break into individual code-points
                    // and look each up.
                    size_t j = 0;
                    while (j < p.size()) {
                        unsigned char c = (unsigned char)p[j];
                        size_t len = 1;
                        if      ((c & 0x80) == 0x00) len = 1;
                        else if ((c & 0xE0) == 0xC0) len = 2;
                        else if ((c & 0xF0) == 0xE0) len = 3;
                        else if ((c & 0xF8) == 0xF0) len = 4;
                        std::string cp = p.substr(j, len);
                        int32_t cid = vocab_.find(cp);
                        if (cid >= 0) out.push_back(cid);
                        else if (vocab_.unk_id() >= 0) out.push_back(vocab_.unk_id());
                        j += len;
                    }
                    continue;
                }
                out.push_back(id);
            }
        }
        return true;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        // Concatenate token strings, then reverse the GPT-2 remap.
        std::string s;
        for (int32_t id : ids) s += vocab_.token(id);
        return gpt2_to_bytes(s);
    }

    const std::string& pre() const override { return pre_; }

private:
    // Walk s as UTF-8, map each code point back to a byte. Unknown
    // code points pass through unchanged (should be rare).
    static std::string gpt2_to_bytes(const std::string& s) {
        const auto& rev = unicode_to_byte();
        std::string out;
        out.reserve(s.size());
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            size_t len = 1;
            uint32_t cp = 0;
            if      ((c & 0x80) == 0x00) { cp = c; len = 1; }
            else if ((c & 0xE0) == 0xC0) { cp = (c & 0x1F) << 6;            len = 2; }
            else if ((c & 0xF0) == 0xE0) { cp = (c & 0x0F) << 12;           len = 3; }
            else if ((c & 0xF8) == 0xF0) { cp = (c & 0x07) << 18;           len = 4; }
            for (size_t k = 1; k < len && i + k < s.size(); ++k) {
                cp |= (uint32_t)((unsigned char)s[i + k] & 0x3F) << (6 * (len - 1 - k));
            }
            if (cp < rev.size() && rev[cp] >= 0) out.push_back((char)rev[cp]);
            else                                 out += s.substr(i, len);
            i += len;
        }
        return out;
    }

    const Vocab& vocab_;
    std::string  pre_;
};

std::unique_ptr<Tokenizer> Tokenizer::create(const Vocab& vocab) {
    const std::string& pre = vocab.pre();
    if (pre == "llama-bpe" || pre == "qwen2" || pre == "default" || pre == "gpt2") {
        return std::make_unique<BpeTokenizer>(vocab, pre);
    }
    std::fprintf(stderr,
        "[sp-engine] unsupported tokenizer pre='%s' "
        "(llama-bpe/qwen2/default/gpt2 supported in this milestone)\n",
        pre.c_str());
    return nullptr;
}

} // namespace sp::engine
