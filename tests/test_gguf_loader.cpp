// Shannon-Prime Engine — GGUF loader test
//
// If SP_ENGINE_TEST_MODEL env var points at a GGUF file, loads it and
// checks minimum fields are non-zero. Otherwise skips with SUCCESS
// (CI can't ship model files, so the test only exercises the API
// surface on the machines that have real assets).

#include "gguf_loader.h"
#include <cstdio>
#include <cstdlib>

int main() {
    const char* path = std::getenv("SP_ENGINE_TEST_MODEL");
    if (!path || !*path) {
        std::printf("test_gguf_loader: no SP_ENGINE_TEST_MODEL set, skipping.\n");
        return 0;
    }

    auto m = sp::engine::Model::load(path);
    if (!m) { std::fprintf(stderr, "Model::load returned null\n"); return 1; }

    int ok = 1;
    if (m->architecture().empty()) { std::fprintf(stderr, "arch empty\n");  ok = 0; }
    if (m->n_layer()  == 0)        { std::fprintf(stderr, "n_layer=0\n");   ok = 0; }
    if (m->n_embd()   == 0)        { std::fprintf(stderr, "n_embd=0\n");    ok = 0; }
    if (m->n_head()   == 0)        { std::fprintf(stderr, "n_head=0\n");    ok = 0; }
    if (m->head_dim() == 0)        { std::fprintf(stderr, "head_dim=0\n");  ok = 0; }
    if (m->n_tensors() <= 0)       { std::fprintf(stderr, "n_tensors<=0\n"); ok = 0; }

    m->print_summary(stdout);
    return ok ? 0 : 1;
}
