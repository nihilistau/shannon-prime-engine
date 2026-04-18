// Shannon-Prime Engine — sp-engine CLI
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"
#include "forward.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "llama_weights.h"
#include "prime_pe.h"
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
        "  kv_smoke [--sqfree] [--head-dim N] [--n-tokens N]\n"
        "                       Push synthetic K/V through compressed cache, report\n"
        "                       compression ratio + per-head correlation.\n"
        "  prefill --model <gguf> [--sqfree] <text>\n"
        "                       Run prefill, push real RoPE'd K/V through KvCache,\n"
        "                       report per-layer correlation (cache vs uncompressed).\n"
        "  chat --model <gguf> [--n-predict N] [--sqfree] <prompt>\n"
        "                       Greedy generation: prefill prompt, decode N tokens\n"
        "                       reading past K/V from the compressed cache.\n"
        "  perplexity --model <gguf> [--ctx N] [--chunks N] <textfile>\n"
        "                       Compute baseline PPL over a UTF-8 text file in\n"
        "                       contiguous N-token chunks. Hook-free reference.\n"
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
        "\n"
        "PrimePE-RoPE-ALiBi:\n"
        "  --pe-mode <name>     standard|primepe|primepe_alibi|alibi (default: standard)\n"
        "  --pe-alpha <f>       blend factor 0..1 (default: 0.0 = identity)\n"
        "  --pe-tier  <n>       0 = composite lattice, 1 = prime generators\n"
        "\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string cmd = argv[1];

    if (cmd == "version") {
        std::printf("sp-engine 0.1.0 (scaffolding)\n");
        return 0;
    }

    // perplexity handles its own argv (positional textfile + --ctx/--chunks)
    // so the strict --flag pass doesn't reject the textfile path.
    if (cmd == "perplexity") {
        sp::engine::Config pc;
        int  n_ctx    = 512;
        int  n_chunks = 0;     // 0 = all
        std::string textfile;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model"   && i + 1 < argc) pc.model_path = argv[++i];
            else if (a == "--ctx"     && i + 1 < argc) n_ctx         = std::atoi(argv[++i]);
            else if (a == "--chunks"  && i + 1 < argc) n_chunks      = std::atoi(argv[++i]);
            else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "perplexity: unknown flag %s\n", a.c_str()); return 2;
            }
            else { textfile = a; }
        }
        if (pc.model_path.empty()) {
            std::fprintf(stderr, "perplexity requires --model <path.gguf>\n"); return 1;
        }
        if (textfile.empty()) {
            std::fprintf(stderr, "perplexity requires a UTF-8 text file path\n"); return 1;
        }

        auto m  = sp::engine::Model::load(pc.model_path);
        if (!m) return 2;
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W  = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        // Read whole file. Binary mode so byte-perfect with what llama.cpp's
        // perplexity sees from wiki.test.raw.
        std::FILE* fp = std::fopen(textfile.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "cannot open %s\n", textfile.c_str()); return 4;
        }
        std::fseek(fp, 0, SEEK_END);
        const size_t fsize = (size_t)std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        std::string text((size_t)fsize, '\0');
        if (std::fread(text.data(), 1, fsize, fp) != fsize) {
            std::fclose(fp);
            std::fprintf(stderr, "short read on %s\n", textfile.c_str()); return 4;
        }
        std::fclose(fp);

        std::vector<int32_t> all_ids;
        tk->encode(text, /*add_bos=*/true, all_ids);
        std::fprintf(stderr, "[sp-engine] tokenised %zu bytes -> %zu tokens\n",
                     fsize, all_ids.size());

        // Estimate scratch: full forward at n=ctx for an n_layer=36 / 8B model
        // wants ~1 GB. The 1 GB ctx_size we pass to ForwardContext::create
        // covers up to ctx ~ 1024 on Qwen3-8B comfortably.
        auto fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024);
        if (!fc) return 5;

        const int n_vocab_local = fc->n_vocab();
        const int total_chunks  = (int)(all_ids.size() / (size_t)n_ctx);
        const int eval_chunks   = (n_chunks > 0 && n_chunks < total_chunks)
                                    ? n_chunks : total_chunks;
        if (eval_chunks <= 0) {
            std::fprintf(stderr, "text too short for ctx=%d (have %zu tokens)\n",
                         n_ctx, all_ids.size()); return 6;
        }
        std::fprintf(stderr,
            "[sp-engine] perplexity: n_ctx=%d  total_chunks=%d  eval=%d  n_vocab=%d\n",
            n_ctx, total_chunks, eval_chunks, n_vocab_local);

        double total_nll        = 0.0;
        long long total_evalled = 0;

        std::vector<float> logits;
        std::vector<int32_t> chunk((size_t)n_ctx);
        const int32_t bos = v->bos_id();

        for (int c = 0; c < eval_chunks; ++c) {
            for (int t = 0; t < n_ctx; ++t) {
                chunk[(size_t)t] = all_ids[(size_t)(c * n_ctx + t)];
            }
            // Match llama-perplexity: every chunk's first token is BOS for
            // the forward pass (resets the "context"), even mid-document.
            // Targets at chunk positions [first+1, n_ctx) come from the
            // unchanged all_ids array, so position 0 being BOS only
            // affects what the model sees, not what we're scoring.
            if (bos >= 0) chunk[0] = bos;
            int n_vocab_out = 0;
            if (!fc->forward_full(chunk, logits, n_vocab_out)) {
                std::fprintf(stderr, "forward_full failed at chunk %d\n", c); return 7;
            }
            // Cross-entropy at every position predicting the next token in
            // the chunk. We use the half-context warmup convention so the
            // first half of the chunk is "context only" — matches
            // llama-perplexity's default behaviour.
            const int first_eval = n_ctx / 2;
            for (int i = first_eval; i < n_ctx - 1; ++i) {
                const float* row = logits.data() + (size_t)i * n_vocab_out;
                const int32_t target = chunk[(size_t)(i + 1)];
                // log-softmax via logsumexp for numerical stability.
                float mx = row[0];
                for (int k = 1; k < n_vocab_out; ++k) if (row[k] > mx) mx = row[k];
                double s = 0.0;
                for (int k = 0; k < n_vocab_out; ++k) s += std::exp((double)(row[k] - mx));
                const double lse = (double)mx + std::log(s);
                const double nll = lse - (double)row[target];
                total_nll += nll;
                total_evalled += 1;
            }
            const double running_ppl = std::exp(total_nll / (double)total_evalled);
            std::fprintf(stderr, "  chunk %3d/%d  PPL_running=%.4f\n",
                         c + 1, eval_chunks, running_ppl);
        }
        const double mean_nll = total_nll / (double)total_evalled;
        const double ppl      = std::exp(mean_nll);
        std::printf("PPL = %.4f  (over %lld tokens, %d chunks at ctx=%d)\n",
                    ppl, total_evalled, eval_chunks, n_ctx);
        return 0;
    }

    // chat handles its own argv so the strict --flag pass doesn't reject
    // --n-predict and so on. Kept above the global parser for the same
    // reason kv_smoke is.
    if (cmd == "chat") {
        sp::engine::Config cc;
        int  n_predict = 32;
        bool naive     = false;
        bool debug_decode = false;
        std::string text;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--sqfree")       cc.sqfree = true;
            else if (a == "--spinor")       { cc.spinor = true; cc.sqfree = true; }
            else if (a == "--no-mobius")    cc.mobius  = false;
            else if (a == "--naive")        naive      = true;
            else if (a == "--debug-decode") debug_decode = true;
            else if (a == "--model"     && i + 1 < argc) cc.model_path = argv[++i];
            else if (a == "--k-bits"    && i + 1 < argc) cc.k_bits_csv = argv[++i];
            else if (a == "--v-bits"    && i + 1 < argc) cc.v_bits_csv = argv[++i];
            else if (a == "--n-predict" && i + 1 < argc) n_predict     = std::atoi(argv[++i]);
            else if (a == "--ctx"       && i + 1 < argc) cc.n_ctx      = std::atoi(argv[++i]);
            else if (a == "--pe-mode"   && i + 1 < argc) {
                std::string m = argv[++i];
                if      (m == "standard")      cc.pe_mode = sp::engine::Config::PeMode::Standard;
                else if (m == "primepe")       cc.pe_mode = sp::engine::Config::PeMode::PrimePe;
                else if (m == "primepe_alibi") cc.pe_mode = sp::engine::Config::PeMode::PrimePeAlibi;
                else if (m == "alibi")         cc.pe_mode = sp::engine::Config::PeMode::AlibiOnly;
            }
            else if (a == "--pe-alpha"  && i + 1 < argc) cc.pe_alpha = (float)std::atof(argv[++i]);
            else if (a == "--pe-tier"   && i + 1 < argc) cc.pe_tier  = std::atoi(argv[++i]);
            else if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
                std::fprintf(stderr, "chat: unknown flag %s\n", a.c_str()); return 2;
            }
            else {
                if (!text.empty()) text.push_back(' ');
                text += a;
            }
        }
        if (cc.model_path.empty()) {
            std::fprintf(stderr, "chat requires --model <path.gguf>\n"); return 1;
        }
        if (text.empty()) {
            std::fprintf(stderr, "chat requires a prompt\n"); return 1;
        }

        auto m = sp::engine::Model::load(cc.model_path);
        if (!m) return 2;
        auto v  = sp::engine::Vocab::load(*m);
        auto tk = v ? sp::engine::Tokenizer::create(*v) : nullptr;
        auto W  = sp::engine::LlamaWeights::load(*m);
        if (!tk || !W) return 3;

        std::vector<int32_t> ids;
        tk->encode(text, /*add_bos=*/true, ids);
        const int n_prompt = (int)ids.size();
        const int max_seq  = n_prompt + n_predict + 4;

        sp::engine::PeSettings pe{cc.pe_mode, cc.pe_alpha, cc.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe);
        if (!fc) return 4;

        auto kv = sp::engine::KvCache::create(fc->n_layer(),
                                               (int)m->n_head_kv(),
                                               (int)m->head_dim(),
                                               max_seq, cc);
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 5; }
        std::fprintf(stderr, "[sp-engine] %s\n", kv->describe().c_str());
        std::fprintf(stderr, "[sp-engine] PE: %s\n",
                     sp::engine::prime_pe_describe(cc.pe_mode, cc.pe_alpha, cc.pe_tier).c_str());

        fc->bind_cache(kv.get());

        std::vector<float> last_logits;
        int n_vocab = 0;
        std::vector<int32_t> running = ids;  // for --naive path
        if (naive) {
            std::vector<float> all;
            if (!fc->forward_full(running, all, n_vocab)) {
                std::fprintf(stderr, "naive forward failed\n"); return 6;
            }
            last_logits.assign(all.end() - n_vocab, all.end());
        } else {
            if (!fc->prefill(ids, last_logits, n_vocab)) {
                std::fprintf(stderr, "prefill failed\n"); return 6;
            }
        }

        // Print the prompt as-is so the user sees what we tokenised.
        std::printf("%s", text.c_str());
        std::fflush(stdout);

        std::string out_so_far;
        for (int step = 0; step < n_predict; ++step) {
            int arg = 0;
            float best = last_logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (last_logits[(size_t)i] > best) { best = last_logits[(size_t)i]; arg = i; }
            }
            std::vector<int32_t> one = { arg };
            std::string piece = tk->decode(one);
            std::printf("%s", piece.c_str());
            std::fflush(stdout);

            if (step + 1 == n_predict) break;
            if (naive) {
                running.push_back(arg);
                std::vector<float> all;
                if (!fc->forward_full(running, all, n_vocab)) {
                    std::fprintf(stderr, "\nnaive forward failed at step %d\n", step); return 7;
                }
                last_logits.assign(all.end() - n_vocab, all.end());
            } else {
                std::vector<float> dbg_K, dbg_X;
                std::vector<float>* dbg_K_ptr = debug_decode ? &dbg_K : nullptr;
                std::vector<float>* dbg_X_ptr = debug_decode ? &dbg_X : nullptr;
                if (!fc->decode(arg, last_logits, n_vocab, dbg_K_ptr, dbg_X_ptr)) {
                    std::fprintf(stderr, "\ndecode failed at step %d\n", step); return 7;
                }
                if (debug_decode) {
                    running.push_back(arg);
                    std::vector<float> ref_logits, ref_X;
                    int rn = 0;
                    std::vector<std::vector<float>> ref_K, ref_V;
                    if (!fc->forward_full(running, ref_logits, rn, &ref_K, &ref_V, &ref_X)) {
                        std::fprintf(stderr, "\nref forward failed at step %d\n", step); return 8;
                    }
                    const int n_kv     = (int)m->n_head_kv();
                    const int hd       = (int)m->head_dim();
                    const int n_embd   = fc->n_embd();
                    const int last_pos = (int)running.size() - 1;
                    const float* ref_K_last = ref_K[0].data() + (size_t)last_pos * n_kv * hd;
                    const float* ref_X_last = ref_X.data()    + (size_t)last_pos * n_embd;
                    auto corr = [&](const float* a, const float* b, int len) {
                        double ma = 0, mb = 0;
                        for (int i = 0; i < len; ++i) { ma += a[i]; mb += b[i]; }
                        ma /= len; mb /= len;
                        double sxy = 0, sxx = 0, syy = 0;
                        for (int i = 0; i < len; ++i) {
                            double da = a[i] - ma, db = b[i] - mb;
                            sxy += da * db; sxx += da * da; syy += db * db;
                        }
                        double d = std::sqrt(sxx * syy);
                        return d > 0 ? (float)(sxy / d) : 0.0f;
                    };
                    auto magn = [](const float* a, int len) {
                        double s = 0; for (int i = 0; i < len; ++i) s += a[i]*a[i];
                        return std::sqrt(s / len);
                    };
                    float k_corr_total = corr(dbg_K.data(), ref_K_last, n_kv * hd);
                    float x_corr       = corr(dbg_X.data(), ref_X_last, n_embd);
                    float dec_x_mag    = magn(dbg_X.data(), n_embd);
                    float ref_x_mag    = magn(ref_X_last, n_embd);
                    int ref_arg = 0; float ref_best = ref_logits[ref_logits.size() - n_vocab];
                    for (int i = 1; i < n_vocab; ++i) {
                        float v = ref_logits[ref_logits.size() - n_vocab + i];
                        if (v > ref_best) { ref_best = v; ref_arg = i; }
                    }
                    int dec_arg = 0; float dec_best = last_logits[0];
                    for (int i = 1; i < n_vocab; ++i) {
                        if (last_logits[(size_t)i] > dec_best) { dec_best = last_logits[(size_t)i]; dec_arg = i; }
                    }
                    std::fprintf(stderr,
                        "\n[debug step %d past_n=%d] L0 K corr=%.4f  L0 X corr=%.4f  X rms dec=%.3f ref=%.3f  "
                        "dec argmax=%d (%+.3f)  ref argmax=%d (%+.3f)\n",
                        step, fc->kv_pos() - 1, k_corr_total, x_corr,
                        dec_x_mag, ref_x_mag,
                        dec_arg, dec_best, ref_arg, ref_best);
                }
            }
        }
        std::printf("\n");
        std::fprintf(stderr, "[sp-engine] kv_pos=%d  (prompt=%d, generated=%d)\n",
                     fc->kv_pos(), n_prompt, n_predict);
        return 0;
    }

    // kv_smoke handles its own argv parsing so the strict --flag check below
    // doesn't reject its tweakable knobs (--head-dim, --n-tokens, etc.).
    if (cmd == "kv_smoke") {
        sp::engine::Config kvc;
        int hd = 128, n_tokens = 32, n_head_kv = 4, n_layer = 2;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--sqfree")    kvc.sqfree = true;
            else if (a == "--spinor")    { kvc.spinor = true; kvc.sqfree = true; }
            else if (a == "--no-mobius") kvc.mobius = false;
            else if (a == "--head-dim"  && i + 1 < argc) hd        = std::atoi(argv[++i]);
            else if (a == "--n-tokens"  && i + 1 < argc) n_tokens  = std::atoi(argv[++i]);
            else if (a == "--n-head-kv" && i + 1 < argc) n_head_kv = std::atoi(argv[++i]);
            else if (a == "--n-layer"   && i + 1 < argc) n_layer   = std::atoi(argv[++i]);
            else if (a == "--k-bits"    && i + 1 < argc) kvc.k_bits_csv = argv[++i];
            else if (a == "--v-bits"    && i + 1 < argc) kvc.v_bits_csv = argv[++i];
            else if (a == "--residual-bits" && i + 1 < argc) kvc.residual_bits = std::atoi(argv[++i]);
            else { std::fprintf(stderr, "kv_smoke: unknown arg %s\n", a.c_str()); return 2; }
        }

        auto kv = sp::engine::KvCache::create(n_layer, n_head_kv, hd, n_tokens, kvc);
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 2; }
        std::fprintf(stderr, "[sp-engine] %s\n", kv->describe().c_str());

        const size_t n_elems = (size_t)n_tokens * n_head_kv * hd;
        std::vector<float> K(n_elems), V(n_elems);
        uint64_t s = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)hd << 32) ^ (uint64_t)n_tokens;
        auto next = [&]() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint32_t u1 = (uint32_t)(s & 0xFFFFFFFFULL);
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint32_t u2 = (uint32_t)(s & 0xFFFFFFFFULL);
            float r1 = (u1 + 1.0f) / 4294967297.0f;
            float r2 = (u2 + 0.0f) / 4294967296.0f;
            return std::sqrt(-2.0f * std::log(r1)) * std::cos(6.2831853f * r2);
        };
        for (size_t i = 0; i < n_elems; ++i) K[i] = next();
        for (size_t i = 0; i < n_elems; ++i) V[i] = next();

        for (int L = 0; L < n_layer; ++L) {
            if (!kv->write(L, 0, n_tokens, K.data(), V.data())) {
                std::fprintf(stderr, "kv->write layer %d failed\n", L); return 3;
            }
        }
        std::vector<float> Krec, Vrec;
        if (!kv->read(0, n_tokens, Krec, Vrec)) {
            std::fprintf(stderr, "kv->read failed\n"); return 4;
        }

        auto corr = [&](const float* a, const float* b, int n) {
            double ma = 0, mb = 0;
            for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
            ma /= n; mb /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; ++i) {
                double da = a[i] - ma, db = b[i] - mb;
                sxy += da * db; sxx += da * da; syy += db * db;
            }
            const double denom = std::sqrt(sxx * syy);
            return (denom > 0) ? (float)(sxy / denom) : 0.0f;
        };

        double k_sum = 0, v_sum = 0;
        float  k_min = 1.0f, v_min = 1.0f;
        const int per = n_head_kv * n_tokens;
        for (int q = 0; q < n_tokens; ++q) {
            for (int h = 0; h < n_head_kv; ++h) {
                const float* k0 = K.data()    + (size_t)(q * n_head_kv + h) * hd;
                const float* k1 = Krec.data() + (size_t)(q * n_head_kv + h) * hd;
                const float* v0 = V.data()    + (size_t)(q * n_head_kv + h) * hd;
                const float* v1 = Vrec.data() + (size_t)(q * n_head_kv + h) * hd;
                float kc = corr(k0, k1, hd);
                float vc = corr(v0, v1, hd);
                k_sum += kc; v_sum += vc;
                if (kc < k_min) k_min = kc;
                if (vc < v_min) v_min = vc;
            }
        }
        std::printf("K corr: mean=%.4f  min=%.4f  (over %d vectors, hd=%d)\n",
                    k_sum / per, k_min, per, hd);
        std::printf("V corr: mean=%.4f  min=%.4f\n", v_sum / per, v_min);
        std::printf("compression ratio = %.2fx\n", kv->compression_ratio());
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
        else if (a == "--pe-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "standard")      cfg.pe_mode = sp::engine::Config::PeMode::Standard;
            else if (m == "primepe")       cfg.pe_mode = sp::engine::Config::PeMode::PrimePe;
            else if (m == "primepe_alibi") cfg.pe_mode = sp::engine::Config::PeMode::PrimePeAlibi;
            else if (m == "alibi")         cfg.pe_mode = sp::engine::Config::PeMode::AlibiOnly;
            else { std::fprintf(stderr, "bad --pe-mode: %s\n", m.c_str()); return 2; }
        }
        else if (a == "--pe-alpha" && i + 1 < argc) cfg.pe_alpha = (float)std::atof(argv[++i]);
        else if (a == "--pe-tier"  && i + 1 < argc) cfg.pe_tier  = std::atoi(argv[++i]);
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

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 512*1024*1024, pe);
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

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W,
                      /*ctx_size_bytes=*/256 * 1024 * 1024, pe);
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

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        std::fprintf(stderr, "[sp-engine] PE: %s\n",
                     sp::engine::prime_pe_describe(cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier).c_str());
        auto fc = sp::engine::ForwardContext::create(*m, *W,
                      /*ctx_size_bytes=*/1024 * 1024 * 1024, pe);
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

    if (cmd == "prefill") {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "prefill requires --model <path.gguf>\n"); return 1;
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
        const int n = (int)ids.size();
        if (n == 0) { std::fprintf(stderr, "prefill: empty text\n"); return 4; }

        sp::engine::PeSettings pe{cfg.pe_mode, cfg.pe_alpha, cfg.pe_tier};
        auto fc = sp::engine::ForwardContext::create(*m, *W, 1024 * 1024 * 1024, pe);
        if (!fc) return 5;

        std::vector<float> logits;
        int n_vocab = 0;
        std::vector<std::vector<float>> Ks, Vs;
        if (!fc->forward_full(ids, logits, n_vocab, &Ks, &Vs)) {
            std::fprintf(stderr, "prefill: forward_full failed\n"); return 6;
        }

        const int n_layer   = fc->n_layer();
        const int head_dim  = (int)m->head_dim();
        const int n_head_kv = (int)m->n_head_kv();

        auto kv = sp::engine::KvCache::create(n_layer, n_head_kv, head_dim, n, cfg);
        if (!kv) { std::fprintf(stderr, "KvCache::create failed\n"); return 7; }
        std::fprintf(stderr, "[sp-engine] %s\n", kv->describe().c_str());

        // Optional: calibrate the cache using this prefill's K vectors
        // before compressed writes. SP_CALIBRATE=1 enables it — matches
        // what ForwardContext::prefill does automatically under chat.
        if (std::getenv("SP_CALIBRATE")) {
            if (kv->calibrate_begin()) {
                for (int L = 0; L < n_layer; ++L) {
                    const float* Kd = Ks[(size_t)L].data();
                    for (int q = 0; q < n; ++q) {
                        for (int h = 0; h < n_head_kv; ++h) {
                            kv->calibrate_feed(Kd + (size_t)(q * n_head_kv + h) * head_dim);
                        }
                    }
                }
                kv->calibrate_end();
                std::fprintf(stderr, "[sp-engine] calibrated=%s\n",
                             kv->is_calibrated() ? "yes" : "no");
            }
        }

        // Push every layer's captured K/V through the compressed cache.
        for (int L = 0; L < n_layer; ++L) {
            if (!kv->write(L, /*pos_offset=*/0, n, Ks[(size_t)L].data(), Vs[(size_t)L].data())) {
                std::fprintf(stderr, "kv->write layer %d failed\n", L); return 8;
            }
        }

        auto corr = [](const float* a, const float* b, int len) {
            double ma = 0, mb = 0;
            for (int i = 0; i < len; ++i) { ma += a[i]; mb += b[i]; }
            ma /= len; mb /= len;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < len; ++i) {
                double da = a[i] - ma, db = b[i] - mb;
                sxy += da * db; sxx += da * da; syy += db * db;
            }
            double denom = std::sqrt(sxx * syy);
            return (denom > 0) ? (float)(sxy / denom) : 0.0f;
        };

        // Read every layer back, compute mean K corr / V corr per layer.
        std::printf("layer  K_corr   V_corr  K_min   V_min\n");
        double overall_k = 0, overall_v = 0;
        for (int L = 0; L < n_layer; ++L) {
            std::vector<float> Krec, Vrec;
            if (!kv->read(L, n, Krec, Vrec)) {
                std::fprintf(stderr, "kv->read layer %d failed\n", L); return 9;
            }
            double k_sum = 0, v_sum = 0;
            float  k_min = 1.0f, v_min = 1.0f;
            const int per = n_head_kv * n;
            for (int q = 0; q < n; ++q) {
                for (int h = 0; h < n_head_kv; ++h) {
                    const float* k0 = Ks[(size_t)L].data() + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* k1 = Krec.data()           + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* v0 = Vs[(size_t)L].data() + (size_t)(q * n_head_kv + h) * head_dim;
                    const float* v1 = Vrec.data()           + (size_t)(q * n_head_kv + h) * head_dim;
                    float kc = corr(k0, k1, head_dim);
                    float vc = corr(v0, v1, head_dim);
                    k_sum += kc; v_sum += vc;
                    if (kc < k_min) k_min = kc;
                    if (vc < v_min) v_min = vc;
                }
            }
            float km = (float)(k_sum / per), vm = (float)(v_sum / per);
            overall_k += km; overall_v += vm;
            // Show a sample of layers to keep the table small.
            if (L < 4 || L == n_layer - 1 || L == n_layer / 2) {
                std::printf("%3d   %6.4f  %6.4f  %6.4f  %6.4f\n", L, km, vm, k_min, v_min);
            }
        }
        std::printf("---\nmean over %d layers: K_corr=%.4f  V_corr=%.4f  compression=%.2fx\n",
                    n_layer, overall_k / n_layer, overall_v / n_layer,
                    kv->compression_ratio());
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
