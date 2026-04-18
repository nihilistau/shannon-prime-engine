// Shannon-Prime Engine — sp-engine CLI
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "engine.h"
#include "gguf_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

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

    // Stub argparse for the other commands. Real one lands with the
    // first functional command.
    sp::engine::Config cfg;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* key, std::string& dst) {
            if (a == key && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        if (a == "--sqfree")         cfg.sqfree = true;
        else if (a == "--spinor")    cfg.spinor = true;
        else if (a == "--no-mobius") cfg.mobius = false;
        else if (next("--model",   cfg.model_path)) {}
        else if (next("--k-bits",  cfg.k_bits_csv)) {}
        else if (next("--v-bits",  cfg.v_bits_csv)) {}
        else if (a == "--ctx" && i + 1 < argc)           cfg.n_ctx = std::atoi(argv[++i]);
        else if (a == "--residual-bits" && i + 1 < argc) cfg.residual_bits = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
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
