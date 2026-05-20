/* sp_oracle_telemetry — Strike 4 device-side latency measurement.
 *
 * Pushes a binary file to the device, creates the prefetch coordinator
 * pointing at it via pread, pins the worker to the A510 silver cluster,
 * fires a dummy Oracle that predicts layer i+1 every 20 ms, and reads
 * each layer in sequence. Measures cold vs warm get latency.
 *
 * Usage (on S22 Ultra after adb push):
 *   /data/local/tmp/sp_oracle_telemetry <path-to-model.bin> [bytes_per_layer] [n_layers]
 *
 * Default config: 2 MB per layer × 28 layers = ~56 MB scan. UFS 3.1
 * sustained read ~2.1 GB/s should land each layer in ~1 ms; the 20 ms
 * Oracle cadence is the headroom budget.
 *
 * Output: stats summary + per-layer latency CSV to stderr.
 */

extern "C" {
#include "../src/sp_oracle_prefetch.h"
}

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#  include <fcntl.h>
#  include <unistd.h>
#  include <pthread.h>
#  include <sched.h>
#else
#  include <fcntl.h>
#  include <io.h>
#  define O_RDONLY _O_RDONLY
#  define open _open
#  define close _close
#endif

static double now_ms() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/data/local/tmp/model.bin";
    size_t bytes_per_layer = (argc > 2) ? (size_t)std::strtoull(argv[2], nullptr, 10) : (2u * 1024u * 1024u);
    int n_layers = (argc > 3) ? std::atoi(argv[3]) : 28;
    int n_slots  = (argc > 4) ? std::atoi(argv[4]) : 8;
    int use_rpcmem = (argc > 5) ? std::atoi(argv[5]) : 0;  /* 1 = ION-backed slots */

    std::fprintf(stderr, "sp_oracle_telemetry: %s, %zu B/layer × %d layers (%.1f MB total)\n",
                 path, bytes_per_layer, n_layers,
                 (double)(bytes_per_layer * (size_t)n_layers) / (1024.0 * 1024.0));

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "ERROR: open(%s) failed\n", path);
        return 1;
    }

    sp_oracle_backend be = sp_oracle_prefetch_pread_backend(fd, /*base_offset=*/0, bytes_per_layer);
    sp_oracle_prefetch* c = nullptr;
    if (use_rpcmem) {
        c = sp_oracle_prefetch_create_rpcmem(&be, n_slots);
    } else {
        c = sp_oracle_prefetch_create(&be, n_slots);
    }
    std::fprintf(stderr, "[config] n_slots=%d use_rpcmem=%d rpcmem_active=%d\n",
                 n_slots, use_rpcmem,
                 c ? sp_oracle_prefetch_slots_are_rpcmem(c) : 0);
    if (!c) {
        std::fprintf(stderr, "ERROR: sp_oracle_prefetch_create failed\n");
        close(fd);
        return 1;
    }

    /* Pin worker to A510 silver cluster (cores 0-3 on Snapdragon 8 Gen 1).
     * On non-POSIX targets this is a no-op (returns -1). */
    int silver[] = {0, 1, 2, 3};
    int aff_rc = sp_oracle_prefetch_pin_worker_to_cpus(c, silver, 4);
    std::fprintf(stderr, "[affinity] worker pinned to A510 silver cluster: rc=%d\n", aff_rc);

    /* Pin self (the orchestrator / "X2 prime") to core 7. */
#if defined(__linux__) || defined(__ANDROID__)
    cpu_set_t self_set;
    CPU_ZERO(&self_set);
    CPU_SET(7, &self_set);
    if (sched_setaffinity(0, sizeof(self_set), &self_set) == 0) {
        std::fprintf(stderr, "[affinity] main thread pinned to X2 prime (core 7)\n");
    }
#endif

    /* Dummy Oracle: predict layer i+1 every 20 ms in a separate thread.
     * In the production wiring this would be the A710 NEON draft model
     * emitting per-token layer-access predictions. */
    std::thread oracle([&] {
#if defined(__linux__) || defined(__ANDROID__)
        cpu_set_t gold_set;
        CPU_ZERO(&gold_set);
        CPU_SET(4, &gold_set);
        CPU_SET(5, &gold_set);
        CPU_SET(6, &gold_set);
        sched_setaffinity(0, sizeof(gold_set), &gold_set);
#endif
        for (int i = 0; i < n_layers; ++i) {
            sp_oracle_prefetch_predict(c, i);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    /* The consumer reads layers in order with a 25 ms inter-layer compute
     * stand-in. Each call to get() either hits (worker pre-fetched) or
     * blocks waiting for the in-flight read to complete. */
    std::vector<double> layer_ms(n_layers, 0.0);
    double total = 0.0;
    for (int i = 0; i < n_layers; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        double t0 = now_ms();
        const void* p = sp_oracle_prefetch_get(c, i);
        double t1 = now_ms();
        if (!p) {
            std::fprintf(stderr, "ERROR: get(%d) returned null\n", i);
            break;
        }
        layer_ms[i] = t1 - t0;
        total += t1 - t0;
    }

    oracle.join();

    sp_oracle_prefetch_stats st;
    sp_oracle_prefetch_get_stats(c, &st);
    std::fprintf(stderr, "\n=== Strike 4 telemetry ===\n");
    std::fprintf(stderr, "predictions = %llu\n", (unsigned long long)st.predictions);
    std::fprintf(stderr, "prefetches  = %llu\n", (unsigned long long)st.prefetches);
    std::fprintf(stderr, "hits        = %llu  (%.1f%%)\n",
                 (unsigned long long)st.hits, 100.0 * (double)st.hits / (double)n_layers);
    std::fprintf(stderr, "misses      = %llu\n", (unsigned long long)st.misses);
    std::fprintf(stderr, "evictions   = %llu\n", (unsigned long long)st.evictions);
    std::fprintf(stderr, "total get   = %.2f ms (avg %.3f ms/layer)\n",
                 total, total / (double)n_layers);

    /* Per-layer CSV — useful for plotting hit/miss latency distributions. */
    std::fprintf(stderr, "\nlayer,get_ms\n");
    for (int i = 0; i < n_layers; ++i) {
        std::fprintf(stderr, "%d,%.3f\n", i, layer_ms[i]);
    }

    sp_oracle_prefetch_destroy(c);
    close(fd);
    return 0;
}
