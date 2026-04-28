// Shannon-Prime Engine -- v3 partial-load IO bench harness.
//
// Times three KvCache disk-load paths to measure the IO win of the
// per-band-major v3 format from phase 2 of the disk-tier architecture:
//
//   1. load_from_disk()                  full read (all bands)
//   2. load_from_disk_partial(max=1)     band 0 only (Granite tier)
//   3. load_from_disk_partial(max=2)     bands 0+1 (Sand tier early)
//   4. load_from_disk_partial(max=3)     bands 0+1+2
//
// Output is one CSV row per configuration written to stdout, ready to
// drop into archive/eval/ alongside other bench logs:
//
//   config,n_runs,bytes_per_run,total_ms,mean_ms_per_run,bandwidth_MBps
//   full,5,30461952,142.34,28.47,1018.5
//   partial-1,5,8810496,18.21,3.64,2304.7
//   ...
//
// The actual numbers depend heavily on disk type:
//   - SATA SSD       ~500 MB/s peak           (cheap, plentiful)
//   - NVMe PCIe 3.0  ~3500 MB/s peak          (typical laptop)
//   - NVMe PCIe 4.0  ~7000 MB/s peak          (modern desktop)
//   - Optane         ~2500 MB/s + 5us latency (former Intel server tier)
//   - UFS 3.1 mobile ~1700 MB/s peak (Galaxy S22 Ultra et al.)
//
// To measure cold-cache performance on Windows, drop the OS file cache
// between runs with: `RAMMap64.exe -E` (from Sysinternals). On Linux:
// `echo 3 | sudo tee /proc/sys/vm/drop_caches`. Without that, the second
// run hits the OS page cache and reports memory bandwidth instead of
// disk bandwidth -- still useful but a different number.

#include "engine.h"
#include "kv_cache.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Bench configuration -- big enough that one load is millisecond-scale,
// small enough to fit on a laptop without filling RAM.
//   12 layers x 8 heads x 2048 positions x 76 bytes/vec ~= 14.6 MB per K
//   x2 (K+V) per layer = ~29 MB total file size on disk
struct BenchConfig {
    int n_layer    = 12;
    int n_head_kv  = 8;
    int head_dim   = 128;
    int max_seq    = 2048;
    int n_pos      = 2048;
    int n_runs     = 5;          // repeat each load path this many times
    int n_warmup   = 1;           // discard the first run as warmup
};

static float test_value(int layer, int pos, int head, int d, int head_dim) {
    const float phase = 0.013f * (float)layer
                      + 0.071f * (float)pos
                      + 0.029f * (float)head
                      + (2.0f * 3.14159265f / (float)head_dim) * (float)d;
    return 0.4f * std::sin(phase);
}

static void fill_layer(int layer, int n_head_kv, int head_dim,
                       int n_tokens, std::vector<float>& K,
                       std::vector<float>& V) {
    K.resize((size_t)n_tokens * n_head_kv * head_dim);
    V.resize(K.size());
    for (int q = 0; q < n_tokens; ++q) {
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t i = ((size_t)q * n_head_kv + h) * head_dim + d;
                K[i] = test_value(layer, q, h,         d, head_dim);
                V[i] = test_value(layer, q, h + 1000, d, head_dim);
            }
        }
    }
}

// Compute approximate bytes that load_from_disk_partial(max_bands) reads
// from the K side (V is similar shape; we just K x 2 for the "both" case).
// Per-band record bytes: 2 (fp16 scale) + ceil(band_size * bits / 8).
static size_t bytes_per_band_record(int band_size, int bits) {
    return 2u + (size_t)((band_size * bits + 7) / 8);
}

static size_t expected_bytes_partial(const BenchConfig& cfg, int max_bands) {
    // Match the K bands the test cache uses (5/5/4/3 with hd=128 -> bands of 32 each)
    const int hd_bands = 4;
    const int band_size = cfg.head_dim / hd_bands;
    const int bits[4] = {5, 5, 4, 3};
    int eff = (max_bands < 0) ? 0 : (max_bands > hd_bands ? hd_bands : max_bands);
    size_t per_pos_bytes_K = 0;
    size_t per_pos_bytes_V = 0;
    for (int b = 0; b < eff; ++b) {
        per_pos_bytes_K += bytes_per_band_record(band_size, bits[b]);
    }
    // V is flat 3-bit, single band: 2 + ceil(128*3/8) = 50
    if (eff > 0) per_pos_bytes_V = 2u + (size_t)((cfg.head_dim * 3 + 7) / 8);
    size_t bytes_per_layer =
        (per_pos_bytes_K + per_pos_bytes_V) * cfg.n_head_kv * cfg.n_pos;
    // + 64 byte headers x 2 (K + V file)
    bytes_per_layer += 128;
    return bytes_per_layer * cfg.n_layer;
}

struct Result {
    std::string config_name;
    int    n_runs_recorded;
    size_t bytes_per_run;
    double total_ms;
    double mean_ms;
    double mean_bandwidth_MBps;
};

int main(int argc, char** argv) {
    using namespace sp::engine;

    BenchConfig cfg;
    bool drop_cache_hint = false;

    // Trivial CLI: --runs N --pos N --layers N --warmup N --drop-cache
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--runs"    && i + 1 < argc) cfg.n_runs   = std::atoi(argv[++i]);
        else if (arg == "--pos"     && i + 1 < argc) cfg.n_pos    = std::atoi(argv[++i]);
        else if (arg == "--layers"  && i + 1 < argc) cfg.n_layer  = std::atoi(argv[++i]);
        else if (arg == "--heads"   && i + 1 < argc) cfg.n_head_kv= std::atoi(argv[++i]);
        else if (arg == "--warmup"  && i + 1 < argc) cfg.n_warmup = std::atoi(argv[++i]);
        else if (arg == "--drop-cache") drop_cache_hint = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: bench_disk_partial [--runs N] [--pos N] [--layers N] "
                        "[--heads N] [--warmup N] [--drop-cache]\n");
            return 0;
        }
    }
    cfg.max_seq = cfg.n_pos > 16 ? cfg.n_pos : 16;

    if (drop_cache_hint) {
        std::fprintf(stderr,
            "[bench] --drop-cache flag noted. The harness does NOT actually drop\n"
            "        the OS cache (that needs admin privileges + a separate tool).\n"
            "        On Windows: run RAMMap64.exe -E between runs.\n"
            "        On Linux:   echo 3 | sudo tee /proc/sys/vm/drop_caches\n"
            "        The numbers below WITHOUT manually dropping cache include\n"
            "        OS-cached reads, which is closer to memory bandwidth than\n"
            "        actual disk bandwidth.\n");
    }

    std::fprintf(stderr,
        "[bench] config: n_layer=%d n_head_kv=%d head_dim=%d n_pos=%d "
        "n_runs=%d (warmup=%d)\n",
        cfg.n_layer, cfg.n_head_kv, cfg.head_dim, cfg.n_pos,
        cfg.n_runs, cfg.n_warmup);

    // ---- Build cache, populate, save ----
    Config sp_cfg;
    sp_cfg.n_ctx        = cfg.max_seq;
    sp_cfg.k_bits_csv   = "5,5,4,3";
    sp_cfg.v_bits_csv   = "3";
    sp_cfg.residual_bits = 3;

    auto cache = KvCache::create(cfg.n_layer, cfg.n_head_kv, cfg.head_dim,
                                  cfg.max_seq, sp_cfg);
    if (!cache) {
        std::fprintf(stderr, "[bench] KvCache::create failed\n");
        return 1;
    }

    for (int il = 0; il < cfg.n_layer; ++il) {
        std::vector<float> K, V;
        fill_layer(il, cfg.n_head_kv, cfg.head_dim, cfg.n_pos, K, V);
        if (!cache->write(il, /*pos_offset=*/0, cfg.n_pos, K.data(), V.data())) {
            std::fprintf(stderr, "[bench] write failed at layer %d\n", il);
            return 1;
        }
    }

    const fs::path tmp_dir = fs::temp_directory_path() / "sp_bench_disk_partial";
    fs::create_directories(tmp_dir);
    const std::string prefix = (tmp_dir / "kv").string();
    const uint64_t model_hash = 0xCAFEBABE12345678ULL;

    if (cache->save_to_disk(prefix, cfg.n_pos, model_hash) != 0) {
        std::fprintf(stderr, "[bench] save_to_disk failed\n");
        return 1;
    }

    // Compute on-disk file sizes for sanity. K and V files per layer.
    size_t total_disk_bytes = 0;
    for (int il = 0; il < cfg.n_layer; ++il) {
        for (const char* suffix : {"k", "v"}) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s.l%d.%s.vht2", prefix.c_str(),
                          il, suffix);
            std::error_code ec;
            uintmax_t sz = fs::file_size(path, ec);
            if (!ec) total_disk_bytes += (size_t)sz;
        }
    }
    std::fprintf(stderr, "[bench] saved cache: %zu bytes on disk across %d layers\n",
                 total_disk_bytes, cfg.n_layer);

    // ---- Time each load configuration ----
    auto time_load = [&](const std::string& name, int max_bands,
                         size_t expected_bytes) -> Result {
        std::vector<double> times_ms;
        times_ms.reserve(cfg.n_runs);

        for (int r = 0; r < cfg.n_runs; ++r) {
            // Fresh cache for each run so we don't measure a no-op.
            auto fresh = KvCache::create(cfg.n_layer, cfg.n_head_kv,
                                         cfg.head_dim, cfg.max_seq, sp_cfg);

            auto t0 = std::chrono::steady_clock::now();
            int rc;
            if (max_bands < 0) {
                rc = fresh->load_from_disk(prefix, model_hash);
            } else {
                rc = fresh->load_from_disk_partial(prefix, model_hash, max_bands);
            }
            auto t1 = std::chrono::steady_clock::now();
            if (rc < 0) {
                std::fprintf(stderr, "[bench] %s run %d failed\n", name.c_str(), r);
                continue;
            }
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            times_ms.push_back(ms);
        }

        // Discard warmup runs.
        size_t recorded_start = (size_t)cfg.n_warmup;
        if (recorded_start > times_ms.size()) recorded_start = times_ms.size();
        double sum = 0.0;
        for (size_t i = recorded_start; i < times_ms.size(); ++i) sum += times_ms[i];
        size_t recorded_n = times_ms.size() - recorded_start;
        double mean_ms = (recorded_n > 0) ? sum / (double)recorded_n : 0.0;
        double bw_mbps = 0.0;
        if (mean_ms > 0.0 && expected_bytes > 0) {
            bw_mbps = ((double)expected_bytes / (1024.0 * 1024.0)) / (mean_ms / 1000.0);
        }

        Result r;
        r.config_name = name;
        r.n_runs_recorded = (int)recorded_n;
        r.bytes_per_run = expected_bytes;
        r.total_ms = sum;
        r.mean_ms = mean_ms;
        r.mean_bandwidth_MBps = bw_mbps;
        return r;
    };

    std::vector<Result> results;
    results.push_back(time_load("partial-1", 1, expected_bytes_partial(cfg, 1)));
    results.push_back(time_load("partial-2", 2, expected_bytes_partial(cfg, 2)));
    results.push_back(time_load("partial-3", 3, expected_bytes_partial(cfg, 3)));
    results.push_back(time_load("full",     -1, total_disk_bytes));

    // ---- Print CSV ----
    std::printf("config,n_runs,bytes_per_run,total_ms,mean_ms_per_run,bandwidth_MBps\n");
    for (const auto& r : results) {
        std::printf("%s,%d,%zu,%.2f,%.2f,%.1f\n",
                    r.config_name.c_str(),
                    r.n_runs_recorded,
                    r.bytes_per_run,
                    r.total_ms,
                    r.mean_ms,
                    r.mean_bandwidth_MBps);
    }

    // Highlight the partial vs full speedup as a friendly footer.
    if (results.size() == 4) {
        const auto& full = results[3];
        std::fprintf(stderr, "\n[bench] speedup vs full load:\n");
        for (int i = 0; i < 3; ++i) {
            double ratio = full.mean_ms > 0 ? full.mean_ms / results[i].mean_ms : 0;
            std::fprintf(stderr, "  %-12s  %.2fx faster  (%zu vs %zu bytes)\n",
                         results[i].config_name.c_str(), ratio,
                         results[i].bytes_per_run, full.bytes_per_run);
        }
    }

    // Cleanup
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    return 0;
}
