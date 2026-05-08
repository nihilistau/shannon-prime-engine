// Shannon-Prime Engine — tokenizer implementations
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Two flavours live in this file:
//
//   BpeTokenizer  — GPT-2 style byte-level BPE. Dispatched for vocabs
//                   that carry a `merges` array (llama-bpe, qwen2,
//                   gpt2, most "default" pre values whose model is
//                   "gpt2"). Word-splitter differentiates letters,
//                   digits, punctuation, and whitespace — close enough
//                   for perplexity sanity; we'll swap in the per-arch
//                   regex when we need byte-for-byte parity.
//
//   SpmTokenizer  — SentencePiece BPE (the Llama / Gemma family). No
//                   byte-to-unicode remap, no pretokenization. Space
//                   runs normalize to ▁ (U+2581). Greedy-merge by
//                   vocab score (higher score = higher priority).
//                   Unknown codepoints fall through to byte-fallback
//                   <0xNN> tokens.
//
// Dispatch at Tokenizer::create() picks SpmTokenizer when
// tokenizer.ggml.model == "llama" AND the vocab carries no merges
// array (i.e. scores-only). Everything else goes to BpeTokenizer.

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

// -------------------------------------------------------------------
// SentencePiece (Llama/Gemma family) tokenizer.
//
// Algorithm (matches llama.cpp's llm_tokenizer_spm):
//   1. Normalise: prepend ▁ and replace every ASCII space with ▁
//      (U+2581). No GPT-2 byte remap — SPM operates on raw UTF-8.
//   2. Seed a doubly-linked list of symbols, one UTF-8 codepoint each.
//   3. Greedy merge loop driven by a max-heap of adjacent pairs. A
//      pair (A, B) is pushed with priority = vocab.token_score(id(A+B))
//      IF `A+B` is a known vocab entry; higher score wins. After a
//      merge, re-queue the new pair's left/right neighbours. Stale
//      queue entries are skipped via length/offset tracking.
//   4. Emit: for each surviving symbol, look it up in the vocab. If
//      present, emit that id. If absent (single unknown codepoint),
//      emit byte-fallback — one <0xNN> token per UTF-8 byte.
//
// Decode: concatenate token strings, translate ▁ → space, translate
// <0xNN> → raw byte. UTF-8 assembly handled implicitly since bytes
// come out in order.
// -------------------------------------------------------------------
class SpmTokenizer : public Tokenizer {
public:
    SpmTokenizer(const Vocab& v, std::string pre) : vocab_(v), pre_(std::move(pre)) {
        // Precompute byte-fallback token ids: "<0x00>".."<0xFF>" -> id.
        byte_id_.assign(256, -1);
        char buf[8];
        for (int b = 0; b < 256; ++b) {
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            byte_id_[b] = vocab_.find(buf);
        }
    }

    bool encode(const std::string& text, bool add_bos,
                std::vector<int32_t>& out) const override {
        out.clear();
        if (add_bos && vocab_.bos_id() >= 0) out.push_back(vocab_.bos_id());

        // Normalise: prepend ▁ and replace each space run with a single ▁.
        // Newlines and tabs are passed through — the byte-fallback path
        // will handle them if they don't appear as their own tokens.
        static const std::string META = "\xE2\x96\x81"; // U+2581
        std::string norm;
        norm.reserve(text.size() + META.size());
        norm += META;
        bool prev_space = false;
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            if (c == ' ') {
                if (!prev_space) norm += META;
                prev_space = true;
            } else {
                norm.push_back(c);
                prev_space = false;
            }
        }
        if (norm.size() == META.size()) return true;  // empty text

        // Seed symbol list — one UTF-8 codepoint per symbol.
        struct Sym { size_t off; int len; int prev; int next; };
        std::vector<Sym> syms;
        syms.reserve(norm.size());
        {
            int idx = 0;
            size_t i = 0;
            while (i < norm.size()) {
                unsigned char c = (unsigned char)norm[i];
                int cp_len = 1;
                if      ((c & 0x80) == 0x00) cp_len = 1;
                else if ((c & 0xE0) == 0xC0) cp_len = 2;
                else if ((c & 0xF0) == 0xE0) cp_len = 3;
                else if ((c & 0xF8) == 0xF0) cp_len = 4;
                Sym s;
                s.off  = i;
                s.len  = cp_len;
                s.prev = idx - 1;
                s.next = idx + 1;
                syms.push_back(s);
                i += cp_len;
                ++idx;
            }
            if (!syms.empty()) syms.back().next = -1;
        }
        if (syms.empty()) return true;

        // Queue of candidate merges — priority = vocab score (higher wins).
        struct QItem {
            float  score;
            int    left;      // symbol index (left side of the pair)
            int    right;     // symbol index (right side of the pair)
            int    merged_len; // byte length of left+right at enqueue time
        };
        struct QCmp {
            bool operator()(const QItem& a, const QItem& b) const {
                // Max-heap on score, then on position (earlier wins ties).
                if (a.score != b.score) return a.score < b.score;
                return a.left > b.left;
            }
        };
        std::priority_queue<QItem, std::vector<QItem>, QCmp> q;

        auto push_pair = [&](int l) {
            if (l < 0) return;
            int r = syms[l].next;
            if (r < 0) return;
            size_t off = syms[l].off;
            int    len = syms[l].len + syms[r].len;
            std::string piece = norm.substr(off, (size_t)len);
            int32_t id = vocab_.find(piece);
            if (id < 0) return;
            float sc = vocab_.token_score(id);
            q.push({sc, l, r, len});
        };

        for (int k = 0; k + 1 < (int)syms.size(); ++k) push_pair(k);

        while (!q.empty()) {
            QItem top = q.top(); q.pop();
            int l = top.left;
            int r = top.right;
            // Skip stale entries: either side merged elsewhere, or the
            // chain no longer reflects the pair's byte length.
            if (syms[l].len == 0 || syms[r].len == 0) continue;
            if (syms[l].next != r) continue;
            if (syms[l].len + syms[r].len != top.merged_len) continue;

            // Merge r into l: extend l to cover both codepoints, drop r.
            syms[l].len += syms[r].len;
            syms[l].next = syms[r].next;
            if (syms[r].next >= 0) syms[syms[r].next].prev = l;
            syms[r].len = 0;

            // Re-queue the pairs touching the merged symbol.
            push_pair(syms[l].prev);
            push_pair(l);
        }

        // Emit surviving symbols.
        for (int k = 0; k >= 0 && k < (int)syms.size(); k = syms[k].next) {
            if (syms[k].len == 0) continue;
            std::string piece = norm.substr(syms[k].off, (size_t)syms[k].len);
            int32_t id = vocab_.find(piece);
            if (id >= 0) {
                out.push_back(id);
                continue;
            }
            // Byte-fallback: emit one <0xNN> per UTF-8 byte.
            for (int b = 0; b < syms[k].len; ++b) {
                unsigned char byte = (unsigned char)norm[syms[k].off + b];
                int32_t bid = byte_id_[byte];
                if (bid >= 0) {
                    out.push_back(bid);
                } else if (vocab_.unk_id() >= 0) {
                    out.push_back(vocab_.unk_id());
                }
            }
        }
        return true;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string out;
        out.reserve(ids.size() * 4);
        bool first_emitted_word = true;
        for (int32_t id : ids) {
            const std::string& t = vocab_.token(id);
            if (t.empty()) continue;
            // Skip control tokens (<bos>, <eos>, <pad>, <unk>, ...).
            if (vocab_.token_type(id) == TokenType::CONTROL) continue;
            // Byte-fallback token: "<0xNN>" -> raw byte.
            if (vocab_.token_type(id) == TokenType::BYTE
                && t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x'
                && t[5] == '>') {
                auto hex = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return 0;
                };
                int hi = hex(t[3]);
                int lo = hex(t[4]);
                out.push_back((char)((hi << 4) | lo));
                continue;
            }
            // Normal piece: translate ▁ to space. The SPM convention is
            // that the leading ▁ of the first word is emitted as a space
            // too — matches llama.cpp / HF `SentencePieceBPETokenizer`.
            // We drop that leading space only if it's the very first
            // word emitted (so "▁Hello▁World" decodes to "Hello World").
            size_t i = 0;
            while (i < t.size()) {
                // Look for the 3-byte ▁ (U+2581 = E2 96 81).
                if (i + 2 < t.size()
                    && (unsigned char)t[i]     == 0xE2
                    && (unsigned char)t[i + 1] == 0x96
                    && (unsigned char)t[i + 2] == 0x81) {
                    if (!first_emitted_word) out.push_back(' ');
                    first_emitted_word = false;
                    i += 3;
                } else {
                    out.push_back(t[i]);
                    ++i;
                    first_emitted_word = false;
                }
            }
        }
        return out;
    }

    const std::string& pre() const override { return pre_; }

private:
    const Vocab&         vocab_;
    std::string          pre_;
    std::vector<int32_t> byte_id_;  // byte value -> <0xNN> token id (-1 if absent)
};

std::unique_ptr<Tokenizer> Tokenizer::create(const Vocab& vocab) {
    const std::string& pre   = vocab.pre();
    const std::string& model = vocab.model();

    // SentencePiece: tokenizer.ggml.model == "llama" → greedy-merge by
    // score. Covers Llama-1/2, Gemma, Mistral (v1), and Gemma-family
    // finetunes like functiongemma. In GGUF, SPM vocabs never ship a
    // real merges array (the merge priorities are inherent in the
    // token scores); if one somehow appears it's ignored here.
    if (model == "llama") {
        return std::make_unique<SpmTokenizer>(vocab, pre);
    }

    // GPT-2 byte-level BPE: llama-3 / qwen2 / qwen3 / qwen35 / gpt2 / dbrx
    // and anything else that ships merges. qwen35 (Qwen3.6-MoE) uses the
    // same byte-level BPE + pretok regex as qwen2 — the `pre` field
    // just rev-tags the merge table version. dbrx (phi-4 GGUFs) uses the
    // standard GPT-2 byte-level BPE + regex; calibration purposes only
    // need tokenizer self-consistency (baseline and candidate runs
    // tokenise the corpus identically), not bit-for-bit agreement with
    // llama.cpp's reference tokenizer.
    if (pre == "llama-bpe" || pre == "qwen2" || pre == "qwen35" ||
        pre == "default"   || pre == "gpt2"  || pre == "dbrx") {
        return std::make_unique<BpeTokenizer>(vocab, pre);
    }
    std::fprintf(stderr,
        "[sp-engine] unsupported tokenizer model='%s' pre='%s' "
        "(BPE llama-bpe/qwen2/qwen35/default/gpt2/dbrx and SPM llama supported)\n",
        model.c_str(), pre.c_str());
    return nullptr;
}

} // namespace sp::engine
