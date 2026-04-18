// Shannon-Prime Engine — scaffolding smoke test
// Verifies the build pulls in both shannon-prime core and ggml, and that
// Engine::load() / generate() / perplexity() are linked.

#include "engine.h"

#include <cstdio>
#include <cstdlib>

int main() {
    sp::engine::Config cfg;
    cfg.model_path = "";
    cfg.n_ctx = 128;

    sp::engine::Engine engine;
    int rc = engine.load(cfg);
    if (rc != 0) { std::fprintf(stderr, "load returned %d\n", rc); return 1; }

    std::string out;
    (void)engine.generate("", 0, out);
    float ppl = engine.perplexity("/dev/null", 0, false);
    (void)ppl;

    std::printf("engine_smoke: OK (scaffolding-level checks passed)\n");
    return 0;
}
