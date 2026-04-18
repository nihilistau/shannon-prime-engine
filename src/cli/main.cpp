// Shannon-Prime Engine — sp-engine CLI
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"
#include "forward.h"
#include "gguf_loader.h"
#include "llama_weights.h"
#include "tokenizer.h"
#include "vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char* prog) {
    std::fprintf(stderr,
        "sp-engine — Shannon-Prime reference inference engine (scaffolding)\n"
        "\n"
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  version              Print version.\n"
        "  banner               Print banner + loaded submodule SHAs (sanity).\n"
        "  info --model <gguf>  Load a GGUF and print hparams + tensor summary.\n"
        "  encode --model <gguf> <text>  Tokenise text to IDs.\n"
        "  decode --model <gguf> <id1> [id2 ...]  Decode IDs to text.\n"
        "  embed  --model <gguf> <text>  Encode + run token-embedding lookup.\n"
        "  block1 --model <gguf> <text>  Run layer-0 transformer block forward.\n"
        "  logits --model <gguf> <text>  Run full forward pass, print logit stats.\n"
        "  perplexity <args>    (not yet implemented)\n"
        "  run <args>           (not yet implemented)\n"
        "\n"
        "Options:\n"
        "  --model <path.gguf>\n"
        "  --ctx <n>            default 2048\n"
        "  --sqfree             enable sqfree + Knight skeleton\n"
        "  --spinor             enable SU(2) sheet bit (requires --sqfree)\n"
        "  --no-mobius          disable ship-path Möbius reorder\n"
        "  --k-bits <csv>       K band bits, e.g. 5,5,4,3\n"
        "  --v-bits <csv>       V band bits, default 3\n"
        "  --residual-bits <n>  sqfree residual bits, default 3\n"
        "\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string cmd = argv[1];

    if (cmd == "version") {
        std::printf("sp-engine 0.1.0 (scaffolding)\n");
        return 0;
    }

    if (cmd == "banner") {
        std::printf("Shannon-Prime Engine — scaffolding build\n");
        std::printf("  linked: shannon-prime core (AGPLv3)\n");
        std::printf("  linked: ggml (MIT)\n");
        std::printf("  status: pre-alpha, no inference path implemented\n");
        return 0;
    }

    // Flag parser — extracts known flags and stashes positional args in `rest`.
    // Per-command handlers below can consume those positionals however they like.
    sp::engine::Config cfg;
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* key, std::string& dst) {
            if (a == key && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        if      (a == "--sqfree")    cfg.sqfree = true;
        else if (a == "--spinor")    cfg.spinor = true;
        else if (a == "--no-mobius") cfg.mobius = false;
        else if (next("--model",   cfg.model_path)) {}
        else if (next("--k-bits",  cfg.k_bits_csv)) {}
        else if (next("--v-bits",  cfg.v_bits_csv)) {}
        else if (a == "--ctx" && i + 1 < argc)           cfg.n_ctx = std::atoi(argv[++i]);
        else if (a == "--residual-bits" && i + 1 < argc) cfg.residual_bits = std::atoi(argv[++i]);
        else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 2;
        }
        else rest.push_back(std::move(a));
    }

    if (cmd == "info") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "info requires --model <path.gguf>\n");
            return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        m->print_summary(stdout);

        // Also show the first few tensors so the user can spot-check the
        // layout without loading a full inspection tool.
        std::printf("\n  first %d tensors:\n",
                    (int)std::min<size_t>(m->tensor_count(), 8));
        for (size_t i = 0; i < m->tensor_count() && i < 8; ++i) {
            auto ti = m->tensor_info(i);
            std::printf("    [%3zu] %-48s type=%d size=%llu B\n",
                        i, ti.name.c_str(), ti.type,
                        (unsigned long long)ti.n_bytes);
        }
        if (m->tensor_count() > 8) {
            std::printf("    ... (%zu total)\n", m->tensor_count());
        }

        std::printf("\n");
        auto v = sp::engine::Vocab::load(*m);
        if (v) {
            v->print_summary(stdout);
            if (v->size() >= 3) {
                std::printf("  sample: [0]=%-12s [1]=%-12s [2]=%-12s\n",
                            v->token(0).c_str(), v->token(1).c_str(), v->token(2).c_str());
            }
        } else {
            std::printf("Tokenizer: (no vocab section in this GGUF)\n");
        }

        std::printf("\n");
        auto w = sp::engine::LlamaWeights::load(*m);
        if (w) {
            w->print_summary(stdout);
        } else {
            std::printf("Weights: (arch binding failed — unsupported arch or missing tensor)\n");
        }
        return 0;
    }

    if (cmd == "encode" || cmd == "decode") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "%s requires --model <path.gguf>\n", cmd.c_str());
            return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        if (!v) { std::fprintf(stderr, "no vocab\n"); return 3; }
        auto tk = sp::engine::Tokenizer::create(*v);
        if (!tk) return 4;

        if (cmd == "encode") {
            std::string text;
            for (size_t i = 0; i < rest.size(); ++i) {
                if (i) text.push_back(' ');
                text += rest[i];
            }
            std::vector<int32_t> ids;
            tk->encode(text, /*add_bos=*/true, ids);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::printf("%s%d", i ? " " : "", ids[i]);
            }
            std::printf("\n");
            std::fprintf(stderr, "(%zu tokens)\n", ids.size());
            return 0;
        }

        // decode
        std::vector<int32_t> ids;
        for (const auto& s : rest) ids.push_back(std::atoi(s.c_str()));
        std::string out = tk->decode(ids);
        std::printf("%s\n", out.c_str());
        return 0;
    }

    if (cmd == "embed") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "embed requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        auto fc = sp::engine::ForwardContext::create(*m, *W);
        if (!fc) return 4;

        std::vector<float> emb;
        int n_embd = 0;
        if (!fc->embed(ids, emb, n_embd)) {
            std::fprintf(stderr, "embed failed\n"); return 5;
        }

        // Print a summary: shape, a few values at the start / middle / end,
        // and the mean + std of the whole block for sanity.
        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        for (float f : emb) { sum += f; sumsq += (double)f * f; }
        double mean = sum / emb.size();
        double var  = (sumsq / emb.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_embd=%d  n_elems=%zu\n", n, n_embd, emb.size());
        std::printf("mean=%.6f  std=%.6f  min=%.6f  max=%.6f\n",
                    mean, stdv,
                    *std::min_element(emb.begin(), emb.end()),
                    *std::max_element(emb.begin(), emb.end()));
        std::printf("emb[0][:4]   = %+.6f %+.6f %+.6f %+.6f\n",
                    emb[0], emb[1], emb[2], emb[3]);
        if (n >= 2) {
            std::printf("emb[1][:4]   = %+.6f %+.6f %+.6f %+.6f\n",
                        emb[n_embd], emb[n_embd+1], emb[n_embd+2], emb[n_embd+3]);
        }
        return 0;
    }

    if (cmd == "block1") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "block1 requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        auto fc = sp::engine::ForwardContext::create(*m, *W, /*ctx_size_bytes=*/256 * 1024 * 1024);
        if (!fc) return 4;

        std::vector<float> out;
        int n_embd = 0;
        if (!fc->forward_one_block(ids, out, n_embd)) {
            std::fprintf(stderr, "forward_one_block failed\n"); return 5;
        }

        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        int n_nan = 0;
        for (float f : out) {
            if (std::isnan(f) || std::isinf(f)) { n_nan++; continue; }
            sum += f; sumsq += (double)f * f;
        }
        double mean = sum / out.size();
        double var  = (sumsq / out.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_embd=%d  n_elems=%zu  n_nan=%d\n",
                    n, n_embd, out.size(), n_nan);
        std::printf("mean=%.6f  std=%.6f\n", mean, stdv);
        std::printf("out[0][:4]  = %+.6f %+.6f %+.6f %+.6f\n",
                    out[0], out[1], out[2], out[3]);
        if (n >= 2) {
            std::printf("out[-1][:4] = %+.6f %+.6f %+.6f %+.6f\n",
                        out[(size_t)(n-1) * n_embd + 0], out[(size_t)(n-1) * n_embd + 1],
                        out[(size_t)(n-1) * n_embd + 2], out[(size_t)(n-1) * n_embd + 3]);
        }
        return 0;
    }

    if (cmd == "logits") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "logits requires --model <path.gguf>\n"); return 1;
        }
        auto m = sp::engine::Model::load(cfg.model_path);
        if (!m) return 2;
        auto v = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::string text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) text.push_back(' ');
            text += rest[i];
        }
        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);

        auto fc = sp::engine::ForwardContext::create(*m, *W, /*ctx_size_bytes=*/1024 * 1024 * 1024);
        if (!fc) return 4;

        std::vector<float> logits;
        int n_vocab = 0;
        if (!fc->forward_full(ids, logits, n_vocab)) {
            std::fprintf(stderr, "forward_full failed\n"); return 5;
        }

        const int n = (int)ids.size();
        double sum = 0, sumsq = 0;
        int n_nan = 0;
        for (float f : logits) {
            if (std::isnan(f) || std::isinf(f)) { n_nan++; continue; }
            sum += f; sumsq += (double)f * f;
        }
        double mean = sum / logits.size();
        double var  = (sumsq / logits.size()) - mean * mean;
        double stdv = var > 0 ? std::sqrt(var) : 0.0;

        std::printf("n_tokens=%d  n_vocab=%d  n_elems=%zu  n_nan=%d\n",
                    n, n_vocab, logits.size(), n_nan);
        std::printf("mean=%.6f  std=%.6f  min=%.6f  max=%.6f\n",
                    mean, stdv,
                    *std::min_element(logits.begin(), logits.end()),
                    *std::max_element(logits.begin(), logits.end()));

        // Last-row argmax: the next-token prediction for the prompt.
        const float* last = logits.data() + (size_t)(n - 1) * n_vocab;
        int arg = 0;
        float best = last[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (last[i] > best) { best = last[i]; arg = i; }
        }
        std::printf("argmax(last) = %d  logit=%+.4f  token=\"%s\"\n",
                    arg, best,
                    (v && arg >= 0 && (size_t)arg < v->size()) ? v->token(arg).c_str() : "?");

        // Top-5 of the last row for a sanity sample.
        std::vector<std::pair<float,int>> topv;
        topv.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) topv.emplace_back(last[i], i);
        std::partial_sort(topv.begin(), topv.begin() + 5, topv.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        std::printf("top5:");
        for (int i = 0; i < 5; ++i) {
            int id = topv[i].second;
            std::printf("  [%d %s %+.3f]",
                        id,
                        (v && (size_t)id < v->size()) ? v->token(id).c_str() : "?",
                        topv[i].first);
        }
        std::printf("\n");
        return 0;
    }

    if (cmd == "perplexity" || cmd == "run") {
        sp::engine::Engine engine;
        int rc = engine.load(cfg);
        if (rc != 0) return rc;
        if (cmd == "perplexity") {
            std::fprintf(stderr, "(perplexity scaffold: not yet implemented)\n");
            return 3;
        }
        std::string out;
        engine.generate("", 0, out);
        std::printf("%s\n", out.c_str());
        return 0;
    }

    usage(argv[0]);
    return 1;
}
