// qnn_bin_driver — see qnn_bin_driver.h.
//
// Phase 5.0 / 5.1 — bench + schema dump for AI Hub-compiled V69 QNN
// context binaries. Mirrors the load+exec+destroy pattern from
// test_sp_qnn_prefill_batch.c (Phase 2.4, 65.8 t/s on Qwen3-4B
// w4a16 ar128 cl2048) but called from inside sp-engine so the
// pipeline can be wired into Engine::generate next.

#include "qnn_bin_driver.h"

#include "sp_qnn.h"   // load_binary, get_io_info, execute, init/shutdown

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sp::engine {

namespace {

uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count();
}

// Per-split state: handle, n_in/n_out, owned input/output buffers,
// and the residual-stream tensor index (matched by rank=3 + the
// canonical [1, ar, hidden] shape Qwen3-4B exports use).
struct Split {
    sp_qnn_handle*    h = nullptr;
    size_t            n_in = 0, n_out = 0;
    std::vector<void*> in_bufs, out_bufs;
    std::vector<size_t> in_sz, out_sz;
    int               residual_in_idx  = -1;
    int               residual_out_idx = -1;
};

size_t tensor_bytes(const sp_qnn_tensor_info& t) {
    size_t n = t.bytes_per_element ? t.bytes_per_element : 1;
    for (uint32_t d = 0; d < t.rank; ++d) n *= t.dims[d];
    return n;
}

// Find the rank-3 [1, AR, hidden] residual stream tensor. AR is the
// activation rank (128 for the prefill .bins), hidden is the model
// dim (2560 for Qwen3-4B). Returns -1 if no match.
int find_residual_idx(const sp_qnn_tensor_info* infos, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (infos[i].rank == 3
            && infos[i].dims[0] == 1
            && infos[i].dims[1] == 128
            && infos[i].dims[2] == 2560) {
            return (int)i;
        }
    }
    return -1;
}

// Print the schema for one split's tensors. Pure diagnostic.
void print_io(const char* label,
              const sp_qnn_tensor_info* infos, size_t n,
              int residual_idx) {
    for (size_t i = 0; i < n; ++i) {
        std::fprintf(stderr,
            "    %s[%zu]: name=%-32s dtype=%u bpe=%zu rank=%u dims=[",
            label, i,
            infos[i].name ? infos[i].name : "(null)",
            (unsigned)infos[i].dtype,
            infos[i].bytes_per_element,
            (unsigned)infos[i].rank);
        for (uint32_t d = 0; d < infos[i].rank; ++d) {
            std::fprintf(stderr, "%u%s",
                infos[i].dims[d],
                d + 1 == infos[i].rank ? "" : ", ");
        }
        std::fprintf(stderr, "] bytes=%zu%s\n",
            tensor_bytes(infos[i]),
            (int)i == residual_idx ? "  ← residual" : "");
    }
}

bool load_split(const std::string& path, Split& s) {
    std::memset(&s, 0, sizeof(s));   // zero POD members; vectors stay valid
    s.in_bufs.clear(); s.out_bufs.clear();
    s.in_sz.clear();   s.out_sz.clear();
    s.h = nullptr;
    s.residual_in_idx = s.residual_out_idx = -1;

    if (sp_qnn_load_binary(path.c_str(), nullptr, &s.h) != SP_QNN_OK) {
        std::fprintf(stderr, "[qnn_bin] load_binary failed: %s\n", path.c_str());
        return false;
    }
    const sp_qnn_tensor_info* in_info  = nullptr;
    const sp_qnn_tensor_info* out_info = nullptr;
    sp_qnn_get_io_info(s.h, &s.n_in, &in_info, &s.n_out, &out_info);

    s.in_bufs.assign(s.n_in, nullptr);
    s.in_sz.assign(s.n_in, 0);
    s.out_bufs.assign(s.n_out, nullptr);
    s.out_sz.assign(s.n_out, 0);
    for (size_t i = 0; i < s.n_in; ++i) {
        s.in_sz[i]   = tensor_bytes(in_info[i]);
        s.in_bufs[i] = std::calloc(1, s.in_sz[i]);
    }
    for (size_t i = 0; i < s.n_out; ++i) {
        s.out_sz[i]   = tensor_bytes(out_info[i]);
        s.out_bufs[i] = std::calloc(1, s.out_sz[i]);
    }
    s.residual_in_idx  = find_residual_idx(in_info,  s.n_in);
    s.residual_out_idx = find_residual_idx(out_info, s.n_out);
    return true;
}

void free_split(Split& s) {
    for (void* p : s.in_bufs)  std::free(p);
    for (void* p : s.out_bufs) std::free(p);
    s.in_bufs.clear(); s.out_bufs.clear();
    s.in_sz.clear();   s.out_sz.clear();
    if (s.h) sp_qnn_destroy(&s.h);
    s.h = nullptr;
}

bool exec_split(Split& s, uint64_t* exec_us) {
    return sp_qnn_execute(s.h,
        (const void* const*)s.in_bufs.data(), s.in_sz.data(),
        (void* const*)s.out_bufs.data(),      s.out_sz.data(),
        exec_us) == SP_QNN_OK;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// Schema dump — Phase 5.1.
// ─────────────────────────────────────────────────────────────────
int qnn_bin_schema_dump(const std::vector<std::string>& split_paths) {
    if (sp_qnn_init(nullptr, nullptr) != SP_QNN_OK) {
        std::fprintf(stderr, "[qnn_bin] sp_qnn_init failed\n");
        return -1;
    }
    for (size_t i = 0; i < split_paths.size(); ++i) {
        std::fprintf(stderr, "\n=== split %zu: %s ===\n",
                     i + 1, split_paths[i].c_str());
        Split s;
        if (!load_split(split_paths[i], s)) {
            sp_qnn_shutdown();
            return -2;
        }
        const sp_qnn_tensor_info* in_info  = nullptr;
        const sp_qnn_tensor_info* out_info = nullptr;
        sp_qnn_get_io_info(s.h, &s.n_in, &in_info, &s.n_out, &out_info);

        std::fprintf(stderr, "  inputs (%zu):\n", s.n_in);
        print_io("in ", in_info, s.n_in, s.residual_in_idx);
        std::fprintf(stderr, "  outputs (%zu):\n", s.n_out);
        print_io("out", out_info, s.n_out, s.residual_out_idx);

        free_split(s);
    }
    sp_qnn_shutdown();
    return 0;
}

// ─────────────────────────────────────────────────────────────────
// Prefill bench — Phase 5.0. Mirrors test_sp_qnn_prefill_batch.c
// with zero-initialized buffers; proves the .bin pipeline runs from
// inside sp-engine and reports the same t/s past-Claude measured.
// ─────────────────────────────────────────────────────────────────
int qnn_bin_prefill_bench(const std::vector<std::string>& split_paths,
                          int n_chunks) {
    if (split_paths.size() != 4) {
        std::fprintf(stderr,
            "[qnn_bin] prefill bench expects exactly 4 splits, got %zu\n",
            split_paths.size());
        return -1;
    }
    if (n_chunks < 1) n_chunks = 3;

    if (sp_qnn_init(nullptr, nullptr) != SP_QNN_OK) {
        std::fprintf(stderr, "[qnn_bin] sp_qnn_init failed\n");
        return -2;
    }

    std::fprintf(stderr,
        "=== Phase 5.0 — sp-engine internal QNN prefill bench ===\n"
        "Pattern: load(1)→exec→load(2)→exec→destroy(1,2)\n"
        "         load(3)→exec→load(4)→exec→destroy(3,4)\n"
        "Per chunk: 128 tokens, 4 loads + 4 execs.\n\n");

    std::vector<uint8_t> host_residual;
    size_t host_residual_size = 0;
    std::vector<uint64_t> chunk_total_us((size_t)n_chunks, 0);

    auto carry_residual = [&](Split& src) {
        if (src.residual_out_idx < 0) return;
        const size_t sz = src.out_sz[src.residual_out_idx];
        if (host_residual.size() < sz) host_residual.resize(sz);
        std::memcpy(host_residual.data(),
                    src.out_bufs[src.residual_out_idx], sz);
        host_residual_size = sz;
    };
    auto inject_residual = [&](Split& dst) {
        if (dst.residual_in_idx < 0) return;
        if (host_residual_size != dst.in_sz[dst.residual_in_idx]) return;
        std::memcpy(dst.in_bufs[dst.residual_in_idx],
                    host_residual.data(), host_residual_size);
    };

    int rc = 0;
    for (int c = 0; c < n_chunks; ++c) {
        std::fprintf(stderr, "=== chunk %d (128 tokens) ===\n", c);
        host_residual_size = 0;
        const uint64_t chunk_start = now_us();
        Split a, b;
        uint64_t load_us[4] = {0}, exec_us[4] = {0};

        // Phase A: split 1 + 2 resident together.
        uint64_t t = now_us();
        if (!load_split(split_paths[0], a)) { rc = -3; break; }
        load_us[0] = now_us() - t;
        if (!exec_split(a, &exec_us[0])) { rc = -4; break; }
        carry_residual(a);

        t = now_us();
        if (!load_split(split_paths[1], b)) { rc = -3; break; }
        load_us[1] = now_us() - t;
        inject_residual(b);
        if (!exec_split(b, &exec_us[1])) { rc = -4; break; }
        carry_residual(b);

        free_split(a);
        free_split(b);

        // Phase B: split 3 + 4 resident together.
        t = now_us();
        if (!load_split(split_paths[2], a)) { rc = -3; break; }
        load_us[2] = now_us() - t;
        inject_residual(a);
        if (!exec_split(a, &exec_us[2])) { rc = -4; break; }
        carry_residual(a);

        t = now_us();
        if (!load_split(split_paths[3], b)) { rc = -3; break; }
        load_us[3] = now_us() - t;
        inject_residual(b);
        if (!exec_split(b, &exec_us[3])) { rc = -4; break; }

        free_split(a);
        free_split(b);

        chunk_total_us[(size_t)c] = now_us() - chunk_start;

        const uint64_t load_sum = load_us[0]+load_us[1]+load_us[2]+load_us[3];
        const uint64_t exec_sum = exec_us[0]+exec_us[1]+exec_us[2]+exec_us[3];
        std::fprintf(stderr,
            "  loads: %.0f %.0f %.0f %.0f ms (sum %.0f)\n",
            load_us[0]/1000.0, load_us[1]/1000.0,
            load_us[2]/1000.0, load_us[3]/1000.0,
            load_sum/1000.0);
        std::fprintf(stderr,
            "  execs: %.0f %.0f %.0f %.0f ms (sum %.0f)\n",
            exec_us[0]/1000.0, exec_us[1]/1000.0,
            exec_us[2]/1000.0, exec_us[3]/1000.0,
            exec_sum/1000.0);
        std::fprintf(stderr,
            "  chunk wall: %.0f ms = %.1f tok/sec (128-token chunk)\n",
            chunk_total_us[(size_t)c]/1000.0,
            128.0 * 1e6 / (double)chunk_total_us[(size_t)c]);
    }

    if (rc == 0 && n_chunks >= 2) {
        uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
        for (int c = 1; c < n_chunks; ++c) {
            const uint64_t v = chunk_total_us[(size_t)c];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const uint64_t avg = sum / (uint64_t)(n_chunks - 1);
        std::fprintf(stderr,
            "\n=== steady-state (excl chunk 0) ===\n"
            "  chunk[0]:    %.0f ms\n"
            "  steady min:  %.0f ms = %.1f tok/sec\n"
            "  steady avg:  %.0f ms = %.1f tok/sec\n"
            "  steady max:  %.0f ms = %.1f tok/sec\n",
            chunk_total_us[0]/1000.0,
            mn/1000.0,  128.0 * 1e6 / (double)mn,
            avg/1000.0, 128.0 * 1e6 / (double)avg,
            mx/1000.0,  128.0 * 1e6 / (double)mx);
    }

    sp_qnn_shutdown();
    return rc;
}

}  // namespace sp::engine
