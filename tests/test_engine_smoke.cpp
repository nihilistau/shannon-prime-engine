// Shannon-Prime Engine — Engine library API smoke test.
//
// Default (no model file): exercises Engine::load()'s scaffold fallback,
// asserts that generate/perplexity both refuse to run without a loaded
// model. Good enough for CI without assets.
//
// With SP_ENGINE_TEST_MODEL pointing at a GGUF: loads a real model via
// the public Engine API and runs a short greedy generate to verify the
// end-to-end library binding compiles, links, and produces some output.

#include "engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* model_env = std::getenv("SP_ENGINE_TEST_MODEL");

    // --- Scaffold pass: no model path ----------------------------------
    {
        sp::engine::Config cfg;
        cfg.model_path = "";
        cfg.n_ctx      = 128;

        sp::engine::Engine engine;
        int rc = engine.load(cfg);
        if (rc != 0) {
            std::fprintf(stderr, "scaffold load returned %d (expected 0)\n", rc);
            return 1;
        }
        std::string out;
        int gr = engine.generate("hi", 4, out);
        if (gr == 0) {
            std::fprintf(stderr, "generate without model succeeded (expected failure)\n");
            return 1;
        }
        float ppl = engine.perplexity("/dev/null", 0, false);
        if (ppl >= 0.0f) {
            std::fprintf(stderr, "perplexity without model returned %.4f (expected <0)\n", ppl);
            return 1;
        }
    }

    // --- Live pass: real GGUF if the caller provided one ---------------
    if (model_env && *model_env) {
        sp::engine::Config cfg;
        cfg.model_path = model_env;
        cfg.n_ctx      = 128;
        sp::engine::Engine engine;
        int rc = engine.load(cfg);
        if (rc != 0) {
            std::fprintf(stderr, "engine.load(%s) returned %d\n", model_env, rc);
            return 1;
        }
        std::string out;
        int gr = engine.generate("Hello", /*n_predict=*/4, out);
        if (gr != 0) {
            std::fprintf(stderr, "engine.generate returned %d\n", gr);
            return 1;
        }
        std::printf("engine_smoke(live): generate -> %zu bytes\n", out.size());
    } else {
        std::printf("engine_smoke: no SP_ENGINE_TEST_MODEL, scaffold-only pass.\n");
    }

    std::printf("engine_smoke: OK\n");
    return 0;
}
