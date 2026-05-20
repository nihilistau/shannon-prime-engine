/* Strike 4 — A510 Prefetch Worker parity / latency test.
 *
 * Validates:
 *   1. Predict-then-get is a HIT (worker pre-filled the slot).
 *   2. Get without predict is a MISS (synchronous fetch on caller).
 *   3. Slot eviction is LRU-correct when n_slots is exceeded.
 *   4. Latency hiding: with a "slow" backend (artificial 10ms delay),
 *      predict-then-(wait)-then-get returns faster than cold get.
 *   5. Returned bytes are correct (per-layer signature check).
 *   6. Concurrent predict + get from multiple caller threads doesn't
 *      corrupt slot state.
 *
 * Backend used: in-memory synthetic data. Layer i's bytes are filled
 * with the byte pattern (i + 13) repeated, so a corruption is trivial
 * to detect.
 */

extern "C" {
#include "../src/sp_oracle_prefetch.h"
}

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TE { const char *name; void (*fn)(); };
static std::vector<TE> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

/* ─── Synthetic backend ────────────────────────────────────────────── */

struct FakeBackend {
    size_t   bytes_per_layer;
    int      sleep_ms;     /* per-read artificial latency (0 = none) */
    std::atomic<int> reads{0};
};

static size_t fake_read(void* user, int layer_idx, void* dst, size_t size) {
    auto* fb = static_cast<FakeBackend*>(user);
    if (size != fb->bytes_per_layer) return 0;
    if (fb->sleep_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(fb->sleep_ms));
    }
    uint8_t signature = (uint8_t)(layer_idx + 13);
    std::memset(dst, signature, size);
    fb->reads.fetch_add(1);
    return size;
}

static bool check_layer_bytes(const void* p, int layer_idx, size_t size) {
    const uint8_t* b = (const uint8_t*)p;
    uint8_t signature = (uint8_t)(layer_idx + 13);
    for (size_t i = 0; i < size; ++i) {
        if (b[i] != signature) return false;
    }
    return true;
}

/* Test 1 — predict → get is a hit. */
TEST(predict_then_get_is_hit) {
    FakeBackend fb{4096, 5};  /* 4 KB, 5ms artificial delay per fetch */
    sp_oracle_backend be{fake_read, &fb, fb.bytes_per_layer};

    auto* c = sp_oracle_prefetch_create(&be, 2);
    ASSERT(c != nullptr);

    sp_oracle_prefetch_predict(c, 7);
    /* Give the worker time to complete the fetch. */
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    const void* p = sp_oracle_prefetch_get(c, 7);
    ASSERT(p != nullptr);
    ASSERT(check_layer_bytes(p, 7, fb.bytes_per_layer));

    sp_oracle_prefetch_stats st;
    sp_oracle_prefetch_get_stats(c, &st);
    std::fprintf(stderr, "  [info] preds=%llu prefetches=%llu hits=%llu misses=%llu\n",
                 (unsigned long long)st.predictions,
                 (unsigned long long)st.prefetches,
                 (unsigned long long)st.hits,
                 (unsigned long long)st.misses);
    ASSERT(st.hits == 1);
    ASSERT(st.misses == 0);

    sp_oracle_prefetch_destroy(c);
}

/* Test 2 — get without predict is a miss but data is correct. */
TEST(cold_get_is_miss_but_correct) {
    FakeBackend fb{1024, 0};
    sp_oracle_backend be{fake_read, &fb, fb.bytes_per_layer};

    auto* c = sp_oracle_prefetch_create(&be, 2);
    ASSERT(c != nullptr);

    const void* p = sp_oracle_prefetch_get(c, 42);
    ASSERT(p != nullptr);
    ASSERT(check_layer_bytes(p, 42, fb.bytes_per_layer));

    sp_oracle_prefetch_stats st;
    sp_oracle_prefetch_get_stats(c, &st);
    ASSERT(st.hits == 0);
    ASSERT(st.misses == 1);

    sp_oracle_prefetch_destroy(c);
}

/* Test 3 — LRU eviction with 2 slots: predict 0, 1, 2 → slot for 0 is
 * evicted (oldest), get(2) hits, get(1) hits, get(0) misses. */
TEST(lru_eviction_correct) {
    FakeBackend fb{256, 2};
    sp_oracle_backend be{fake_read, &fb, fb.bytes_per_layer};

    auto* c = sp_oracle_prefetch_create(&be, 2);
    ASSERT(c != nullptr);

    sp_oracle_prefetch_predict(c, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    sp_oracle_prefetch_predict(c, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    sp_oracle_prefetch_predict(c, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    /* After three predictions in a 2-slot cache, layer 0 is gone. */
    const void* p2 = sp_oracle_prefetch_get(c, 2);  ASSERT(check_layer_bytes(p2, 2, fb.bytes_per_layer));
    const void* p1 = sp_oracle_prefetch_get(c, 1);  ASSERT(check_layer_bytes(p1, 1, fb.bytes_per_layer));
    const void* p0 = sp_oracle_prefetch_get(c, 0);  ASSERT(check_layer_bytes(p0, 0, fb.bytes_per_layer));

    sp_oracle_prefetch_stats st;
    sp_oracle_prefetch_get_stats(c, &st);
    std::fprintf(stderr, "  [info] hits=%llu misses=%llu evictions=%llu\n",
                 (unsigned long long)st.hits,
                 (unsigned long long)st.misses,
                 (unsigned long long)st.evictions);
    ASSERT(st.hits == 2);     /* 2 and 1 stayed */
    ASSERT(st.misses == 1);   /* 0 was evicted */
    ASSERT(st.evictions >= 1);

    sp_oracle_prefetch_destroy(c);
}

/* Test 4 — latency hiding: predicted gets are faster than cold gets. */
TEST(latency_hiding_measurable) {
    FakeBackend fb{8192, 30};  /* 30ms per read — the "slow UFS" simulation */
    sp_oracle_backend be{fake_read, &fb, fb.bytes_per_layer};

    auto* c = sp_oracle_prefetch_create(&be, 2);
    ASSERT(c != nullptr);

    /* Cold get baseline. */
    auto t0 = std::chrono::steady_clock::now();
    (void)sp_oracle_prefetch_get(c, 100);
    auto t1 = std::chrono::steady_clock::now();
    auto cold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    /* Predicted get. Simulate the realistic timing: predict fires, then
     * the caller does ~25ms of compute, then asks for the layer. */
    sp_oracle_prefetch_predict(c, 101);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));  /* "compute" */
    auto t2 = std::chrono::steady_clock::now();
    (void)sp_oracle_prefetch_get(c, 101);
    auto t3 = std::chrono::steady_clock::now();
    auto warm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::fprintf(stderr, "  [info] cold get=%lld ms, warm get=%lld ms\n",
                 (long long)cold_ms, (long long)warm_ms);
    /* Warm should be substantially faster — within a few ms vs 30+ ms cold. */
    ASSERT(cold_ms >= 25);
    ASSERT(warm_ms < cold_ms / 2);  /* at least 2x speedup */

    sp_oracle_prefetch_destroy(c);
}

/* Test 5 — concurrent caller + Oracle. One thread produces predictions
 * (simulating the A710 Oracle); another consumes via get (the X2
 * orchestrator dispatching to HTP). Verify all reads return correct
 * signature bytes and no slot state corrupts. */
TEST(concurrent_oracle_consumer) {
    FakeBackend fb{2048, 1};
    sp_oracle_backend be{fake_read, &fb, fb.bytes_per_layer};

    auto* c = sp_oracle_prefetch_create(&be, 4);
    ASSERT(c != nullptr);

    constexpr int N_LAYERS = 50;
    std::atomic<bool> oracle_done{false};

    std::thread oracle([&] {
        for (int i = 0; i < N_LAYERS; ++i) {
            sp_oracle_prefetch_predict(c, i);
            std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
        oracle_done.store(true);
    });

    int correct = 0, wrong = 0;
    for (int i = 0; i < N_LAYERS; ++i) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        const void* p = sp_oracle_prefetch_get(c, i);
        if (p && check_layer_bytes(p, i, fb.bytes_per_layer)) ++correct;
        else ++wrong;
    }
    oracle.join();

    sp_oracle_prefetch_stats st;
    sp_oracle_prefetch_get_stats(c, &st);
    std::fprintf(stderr, "  [info] correct=%d wrong=%d hits=%llu misses=%llu\n",
                 correct, wrong,
                 (unsigned long long)st.hits,
                 (unsigned long long)st.misses);
    ASSERT(wrong == 0);
    ASSERT(correct == N_LAYERS);

    sp_oracle_prefetch_destroy(c);
}

int main() {
    std::fprintf(stderr, "test_sp_oracle_prefetch: %zu tests\n", g_tests.size());
    for (auto& t : g_tests) {
        std::fprintf(stderr, "  %s ...\n", t.name);
        t.fn();
    }
    if (g_fail) {
        std::fprintf(stderr, "FAILED %d assertions\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "all tests passed\n");
    return 0;
}
