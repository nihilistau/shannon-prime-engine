// qnn_bin_driver — see qnn_bin_driver.h.
//
// Phase 5.0 / 5.1 — bench + schema dump for AI Hub-compiled V69 QNN
// context binaries. Mirrors the load+exec+destroy pattern from
// test_sp_qnn_prefill_batch.c (Phase 2.4, 65.8 t/s on Qwen3-4B
// w4a16 ar128 cl2048) but called from inside sp-engine so the
// pipeline can be wired into Engine::generate next.

#include "qnn_bin_driver.h"

#include "sp_quant.h"         // sp_fp32_to_fp16 / sp_fp16_to_fp32
#include "sp_qnn.h"           // load_binary, get_io_info, execute, init/shutdown

#include <chrono>
#include <cinttypes>
#include <cmath>
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

// fp16 buffer stats — min/max/abs-mean/finite-count. Used to bisect
// where saturation begins along the split chain.
struct ResStats {
    float vmin, vmax, abs_mean;
    size_t n_total, n_inf, n_nan;
};
inline ResStats fp16_stats(const uint8_t* bytes, size_t bytes_len) {
    const size_t n = bytes_len / 2;
    const uint16_t* p = (const uint16_t*)bytes;
    ResStats r{1e30f, -1e30f, 0.0f, n, 0, 0};
    double accum = 0.0;
    size_t finite = 0;
    for (size_t i = 0; i < n; ++i) {
        const float f = sp_fp16_to_fp32(p[i]);
        if (std::isnan(f)) { r.n_nan++; continue; }
        if (std::isinf(f)) { r.n_inf++; continue; }
        if (f < r.vmin) r.vmin = f;
        if (f > r.vmax) r.vmax = f;
        accum += std::fabs((double)f);
        finite++;
    }
    r.abs_mean = finite ? (float)(accum / (double)finite) : 0.0f;
    if (finite == 0) { r.vmin = 0.0f; r.vmax = 0.0f; }
    return r;
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

    auto carry_residual = [&](Split& src, const char* tag) {
        if (src.residual_out_idx < 0) return;
        const size_t sz = src.out_sz[src.residual_out_idx];
        if (host_residual.size() < sz) host_residual.resize(sz);
        std::memcpy(host_residual.data(),
                    src.out_bufs[src.residual_out_idx], sz);
        host_residual_size = sz;
        const ResStats st = fp16_stats(host_residual.data(), sz);
        std::fprintf(stderr,
            "  [%s] residual: min=%.3g max=%.3g abs_mean=%.3g inf=%zu nan=%zu\n",
            tag, st.vmin, st.vmax, st.abs_mean, st.n_inf, st.n_nan);
    };
    auto inject_residual = [&](Split& dst) {
        if (dst.residual_in_idx < 0) return;
        if (host_residual_size != dst.in_sz[dst.residual_in_idx]) return;
        std::memcpy(dst.in_bufs[dst.residual_in_idx],
                    host_residual.data(), host_residual_size);
        // Optional override: fill residual with a uniform fp16 value
        // chosen by SP_QNN_BIN_RES_OVERRIDE (parses as a float).
        // Use this to test "does split 2 actually read the residual?".
        if (const char* v = std::getenv("SP_QNN_BIN_RES_OVERRIDE")) {
            const float f = std::strtof(v, nullptr);
            const uint16_t fp16 = sp_fp32_to_fp16(f);
            uint16_t* p = (uint16_t*)dst.in_bufs[dst.residual_in_idx];
            const size_t n = host_residual_size / 2;
            for (size_t i = 0; i < n; ++i) p[i] = fp16;
            std::fprintf(stderr,
                "  [override] residual filled with %g (%zu fp16)\n", f, n);
        }
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
        carry_residual(a, "split1");

        t = now_us();
        if (!load_split(split_paths[1], b)) { rc = -3; break; }
        load_us[1] = now_us() - t;
        inject_residual(b);
        if (!exec_split(b, &exec_us[1])) { rc = -4; break; }
        carry_residual(b, "split2");

        free_split(a);
        free_split(b);

        // Phase B: split 3 + 4 resident together.
        t = now_us();
        if (!load_split(split_paths[2], a)) { rc = -3; break; }
        load_us[2] = now_us() - t;
        inject_residual(a);
        if (!exec_split(a, &exec_us[2])) { rc = -4; break; }
        carry_residual(a, "split3");

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

// ─────────────────────────────────────────────────────────────────
// Phase 5.2 — real prompt → next token through the .bin chain.
// ─────────────────────────────────────────────────────────────────

namespace {

// Find input/output index by exact name match. Returns -1 if not found.
int find_io_idx(const sp_qnn_tensor_info* infos, size_t n,
                const char* needle) {
    for (size_t i = 0; i < n; ++i) {
        if (infos[i].name && std::strcmp(infos[i].name, needle) == 0) {
            return (int)i;
        }
    }
    return -1;
}

// Find any input/output whose name contains the substring. Used for
// past_key_*_in / past_value_*_in iteration where the layer index is
// embedded in the name.
int find_io_by_substr(const sp_qnn_tensor_info* infos, size_t n,
                       const char* needle) {
    for (size_t i = 0; i < n; ++i) {
        if (infos[i].name && std::strstr(infos[i].name, needle)) {
            return (int)i;
        }
    }
    return -1;
}

// Build precomputed RoPE position_ids_cos/sin tables for the first
// `n_positions` positions. fp16[1, 1, n_positions, n_freq_pairs].
// theta_i(p) = p / rope_base^(2i / head_dim) for i in [0, n_freq_pairs).
void build_position_ids(int n_positions, int n_freq_pairs,
                          float rope_base,
                          std::vector<uint16_t>& cos_out,
                          std::vector<uint16_t>& sin_out) {
    cos_out.assign((size_t)n_positions * n_freq_pairs, 0);
    sin_out.assign((size_t)n_positions * n_freq_pairs, 0);
    const double base = (double)rope_base;
    const int    head_dim = n_freq_pairs * 2;
    for (int p = 0; p < n_positions; ++p) {
        for (int i = 0; i < n_freq_pairs; ++i) {
            const double freq  = std::pow(base, (double)(2 * i) / (double)head_dim);
            const double theta = (double)p / freq;
            cos_out[(size_t)p * n_freq_pairs + i] =
                sp_fp32_to_fp16((float)std::cos(theta));
            sin_out[(size_t)p * n_freq_pairs + i] =
                sp_fp32_to_fp16((float)std::sin(theta));
        }
    }
}

// Build causal attention mask fp16[1, 1, ar, cl] for HF-convention
// hybrid prefill graphs. The 2048-wide K dimension is laid out as
// [past_in_buffer (cl-ar slots) | current_chunk (ar slots)]. Past
// has past_len ≤ cl-ar meaningful positions starting at offset
// (cl - ar - past_len); the rest of the past buffer is junk and
// must be masked. Current chunk lives at [cl-ar, cl) — for query q
// (q ∈ [0, ar)), causal allow covers past meaningful positions plus
// current [cl-ar, cl-ar+q].
//
// First chunk has past_len = 0 → only the current chunk's [cl-ar,
// cl-ar+q] positions are unmasked. My earlier version allowed
// [0, q] which pointed at PAST junk slots → fp16-saturated logits.
void build_causal_mask(int ar, int cl, int n_real_tokens, int past_len,
                        std::vector<uint16_t>& mask_out) {
    // SP_QNN_BIN_MASK_FINITE=1 uses fp16 most-negative-finite (~-65504)
    // instead of -inf. AIMET-quantized attention may handle finite better
    // through the int16 activation path. Default ON since past-Claude's
    // notes + Genie configs both use the finite convention.
    static const bool use_finite = []() {
        const char* v = std::getenv("SP_QNN_BIN_MASK_FINITE");
        return !v || v[0] != '0';   // default ON
    }();
    const uint16_t neg_inf_fp16 = use_finite
        ? (uint16_t)0xFBFF                  // -65504, most-neg finite fp16
        : sp_fp32_to_fp16(-1.0e30f);        // saturates to -inf
    const uint16_t zero_fp16    = sp_fp32_to_fp16(0.0f);
    mask_out.assign((size_t)ar * cl, neg_inf_fp16);
    const int past_max = cl - ar;            // past buffer width
    const int past_off = past_max - past_len; // offset to first meaningful past
    // SP_QNN_BIN_MASK_FULL_AR: when "1", mask covers ALL ar query
    // rows causally (good when padded slots are repeats of real tokens).
    // When "0" (the default), only allow attention for real query
    // positions [0, n_real_tokens) — padded q slots stay all -inf.
    // Genie/HF convention is to mask padded query rows; the .bin's
    // post-attention norm then doesn't pool garbage from those rows.
    static const bool full_ar = []() {
        const char* v = std::getenv("SP_QNN_BIN_MASK_FULL_AR");
        return v && v[0] == '1';
    }();
    const int last_q = full_ar
        ? ar
        : (n_real_tokens > ar ? ar : n_real_tokens);
    for (int q = 0; q < last_q; ++q) {
        for (int k = past_off; k < past_max && k < cl; ++k) {
            mask_out[(size_t)q * cl + k] = zero_fp16;
        }
        for (int k = past_max; k <= past_max + q && k < cl; ++k) {
            mask_out[(size_t)q * cl + k] = zero_fp16;
        }
    }
}

}  // namespace

int qnn_bin_generate_one(const std::vector<std::string>& split_paths,
                          const std::vector<int32_t>& prompt_tokens,
                          int   ar,
                          int   cl,
                          int   head_dim,
                          float rope_base,
                          int*  out_next_token_id) {
    if (split_paths.size() != 4) {
        std::fprintf(stderr,
            "[qnn_bin] generate_one expects 4 splits, got %zu\n",
            split_paths.size());
        return -1;
    }
    if (!out_next_token_id) return -1;
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "[qnn_bin] empty prompt\n");
        return -1;
    }

    if (sp_qnn_init(nullptr, nullptr) != SP_QNN_OK) {
        std::fprintf(stderr, "[qnn_bin] sp_qnn_init failed\n");
        return -2;
    }

    // ── Build static inputs that don't change per chunk ──
    const int n_freq_pairs = head_dim / 2;
    std::vector<uint16_t> pos_cos, pos_sin;
    build_position_ids(ar, n_freq_pairs, rope_base, pos_cos, pos_sin);
    // Diagnostic override: SP_QNN_BIN_NOROPE=1 sets cos=1,sin=0 →
    // RoPE becomes identity. Bisects "is divergence from RoPE
    // encoding or from past_KV quant decode?".
    if (const char* v = std::getenv("SP_QNN_BIN_NOROPE")) {
        if (v[0] == '1') {
            std::fprintf(stderr,
                "[qnn_bin] SP_QNN_BIN_NOROPE=1 — cos=1, sin=0 (no rotation)\n");
            const uint16_t one_fp16 = sp_fp32_to_fp16(1.0f);
            std::fill(pos_cos.begin(), pos_cos.end(), one_fp16);
            std::fill(pos_sin.begin(), pos_sin.end(), (uint16_t)0);
        }
    }

    const int n_real = (int)prompt_tokens.size();
    const int n_real_clamped = n_real > ar ? ar : n_real;
    if (n_real > ar) {
        std::fprintf(stderr,
            "[qnn_bin] warning: prompt has %d tokens, truncating to AR=%d "
            "(multi-chunk prefill is Phase 5.4)\n", n_real, ar);
    }

    std::vector<uint16_t> attn_mask;
    // First chunk: no history.
    build_causal_mask(ar, cl, n_real_clamped, /*past_len=*/0, attn_mask);

    // SP_QNN_BIN_PAD_TOKEN env: pad token id for slots beyond
    // n_real_clamped. Default 151643 (Qwen3 endoftext / pad_token).
    // Genie configs use this convention. Set to "last" to fall back
    // to repeating the last real token (the previous default).
    int32_t pad_id = 151643;
    if (const char* v = std::getenv("SP_QNN_BIN_PAD_TOKEN")) {
        if (std::strcmp(v, "last") == 0 && n_real_clamped > 0) {
            pad_id = prompt_tokens[(size_t)(n_real_clamped - 1)];
        } else {
            const long parsed = std::strtol(v, nullptr, 10);
            if (parsed > 0) pad_id = (int32_t)parsed;
        }
    }
    std::vector<int32_t> input_ids((size_t)ar, pad_id);
    for (int i = 0; i < n_real_clamped; ++i) {
        input_ids[(size_t)i] = prompt_tokens[(size_t)i];
    }
    // SP_QNN_BIN_INPUT_TOKEN: override every input_ids slot with this
    // single token id. Useful to compare embedding lookup behavior
    // across token IDs (e.g. SP_QNN_BIN_INPUT_TOKEN=0 vs 100 vs 1000).
    if (const char* v = std::getenv("SP_QNN_BIN_INPUT_TOKEN")) {
        const long parsed = std::strtol(v, nullptr, 10);
        if (parsed >= 0) {
            std::fill(input_ids.begin(), input_ids.end(), (int32_t)parsed);
            std::fprintf(stderr,
                "[qnn_bin] SP_QNN_BIN_INPUT_TOKEN=%ld — all 128 slots\n",
                parsed);
        }
    }

    // ── SPLIT 1: input_ids → embedding ──────────────────────────
    Split s1;
    if (!load_split(split_paths[0], s1)) { sp_qnn_shutdown(); return -3; }
    const sp_qnn_tensor_info* s1_in_info  = nullptr;
    const sp_qnn_tensor_info* s1_out_info = nullptr;
    sp_qnn_get_io_info(s1.h, &s1.n_in, &s1_in_info, &s1.n_out, &s1_out_info);

    const int s1_input_ids_idx = find_io_idx(s1_in_info, s1.n_in, "input_ids");
    if (s1_input_ids_idx < 0) {
        std::fprintf(stderr, "[qnn_bin] split 1 missing input_ids\n");
        free_split(s1); sp_qnn_shutdown(); return -3;
    }
    std::memcpy(s1.in_bufs[(size_t)s1_input_ids_idx],
                input_ids.data(), s1.in_sz[(size_t)s1_input_ids_idx]);

    // Pre-fill output buffer with a marker pattern (0xCD bytes) so we
    // can tell if exec() actually wrote to our host buffer or just
    // wrote to a device-side buffer that didn't copy back.
    if (s1.residual_out_idx >= 0) {
        std::memset(s1.out_bufs[(size_t)s1.residual_out_idx], 0xCD,
                    s1.out_sz[(size_t)s1.residual_out_idx]);
    }

    uint64_t exec_us = 0;
    if (!exec_split(s1, &exec_us)) {
        std::fprintf(stderr, "[qnn_bin] split 1 exec failed\n");
        free_split(s1); sp_qnn_shutdown(); return -4;
    }
    std::fprintf(stderr, "[qnn_bin] split 1 exec: %.1f ms\n", exec_us / 1000.0);

    // Count how many bytes still equal 0xCD (i.e., were NOT written
    // by the .bin's exec).
    if (s1.residual_out_idx >= 0) {
        const uint8_t* p = (const uint8_t*)s1.out_bufs[(size_t)s1.residual_out_idx];
        const size_t   n = s1.out_sz[(size_t)s1.residual_out_idx];
        size_t unwritten = 0;
        for (size_t i = 0; i < n; ++i) if (p[i] == 0xCD) ++unwritten;
        std::fprintf(stderr,
            "[qnn_bin] split 1 unwritten bytes (still 0xCD): %zu / %zu (%.2f%%)\n",
            unwritten, n, 100.0 * (double)unwritten / (double)n);
    }

    // Carry the embedding (residual) buffer.
    std::vector<uint8_t> residual;
    if (s1.residual_out_idx < 0) {
        std::fprintf(stderr, "[qnn_bin] split 1 has no residual output\n");
        free_split(s1); sp_qnn_shutdown(); return -3;
    }
    {
        const size_t sz = s1.out_sz[(size_t)s1.residual_out_idx];
        residual.assign(sz, 0);
        std::memcpy(residual.data(),
                    s1.out_bufs[(size_t)s1.residual_out_idx], sz);
        const ResStats st = fp16_stats(residual.data(), residual.size());
        std::fprintf(stderr,
            "[qnn_bin] split 1 residual: min=%.3g max=%.3g abs_mean=%.3g "
            "inf=%zu nan=%zu n=%zu\n",
            st.vmin, st.vmax, st.abs_mean, st.n_inf, st.n_nan, st.n_total);
    }
    free_split(s1);

    // ── Helper: run one of the layer-block splits (2/3/4) ───────
    // For each: bind attention_mask, residual_in, position_ids_cos/sin,
    // and zero-init all past_key/past_value inputs (first chunk, no
    // history). Carry residual_out into `residual`. For split 4,
    // additionally read the logits output and pick the argmax at
    // position n_real_clamped - 1.
    auto run_layer_block = [&](const std::string& path, bool is_last,
                                int* picked_token_id) -> int {
        Split s;
        if (!load_split(path, s)) return -3;
        const sp_qnn_tensor_info* in_info  = nullptr;
        const sp_qnn_tensor_info* out_info = nullptr;
        sp_qnn_get_io_info(s.h, &s.n_in, &in_info, &s.n_out, &out_info);

        // Bind attention_mask.
        const int mask_idx = find_io_idx(in_info, s.n_in, "attention_mask");
        if (mask_idx < 0
            || s.in_sz[(size_t)mask_idx] != attn_mask.size() * sizeof(uint16_t)) {
            std::fprintf(stderr,
                "[qnn_bin] %s: attention_mask shape mismatch\n", path.c_str());
            free_split(s); return -3;
        }
        std::memcpy(s.in_bufs[(size_t)mask_idx],
                    attn_mask.data(), s.in_sz[(size_t)mask_idx]);

        // Bind position_ids_cos/sin.
        const int pcos_idx = find_io_idx(in_info, s.n_in, "position_ids_cos");
        const int psin_idx = find_io_idx(in_info, s.n_in, "position_ids_sin");
        if (pcos_idx < 0 || psin_idx < 0) {
            std::fprintf(stderr, "[qnn_bin] %s: position_ids missing\n",
                         path.c_str());
            free_split(s); return -3;
        }
        std::memcpy(s.in_bufs[(size_t)pcos_idx], pos_cos.data(),
                    s.in_sz[(size_t)pcos_idx]);
        std::memcpy(s.in_bufs[(size_t)psin_idx], pos_sin.data(),
                    s.in_sz[(size_t)psin_idx]);

        // Bind residual_in via name search — covers both
        // "..._Add_1_output_0" (block residual chain) and
        // "..._Gather_output_0" (embedding from split 1).
        if (s.residual_in_idx < 0) {
            std::fprintf(stderr,
                "[qnn_bin] %s: no residual input\n", path.c_str());
            free_split(s); return -3;
        }
        const size_t res_in_bytes = s.in_sz[(size_t)s.residual_in_idx];
        if (res_in_bytes != residual.size()) {
            std::fprintf(stderr,
                "[qnn_bin] %s: residual size mismatch %zu vs %zu\n",
                path.c_str(), res_in_bytes, residual.size());
            free_split(s); return -3;
        }
        std::memcpy(s.in_bufs[(size_t)s.residual_in_idx],
                    residual.data(), res_in_bytes);

        // past_key/past_value inputs stay at calloc-zero (first chunk,
        // no history). Inputs are tagged "past_key_*_in" / "past_value_*_in".
        // Nothing to do.

        // Execute.
        uint64_t us = 0;
        if (!exec_split(s, &us)) {
            std::fprintf(stderr,
                "[qnn_bin] %s exec failed\n", path.c_str());
            free_split(s); return -4;
        }
        std::fprintf(stderr, "[qnn_bin] %s exec: %.1f ms\n",
                     path.c_str(), us / 1000.0);

        // Carry residual_out → residual (for next split).
        if (s.residual_out_idx >= 0) {
            const size_t sz = s.out_sz[(size_t)s.residual_out_idx];
            residual.assign(sz, 0);
            std::memcpy(residual.data(),
                        s.out_bufs[(size_t)s.residual_out_idx], sz);
            const ResStats st = fp16_stats(residual.data(), residual.size());
            std::fprintf(stderr,
                "[qnn_bin] %s residual: min=%.3g max=%.3g abs_mean=%.3g "
                "inf=%zu nan=%zu n=%zu\n",
                path.c_str(),
                st.vmin, st.vmax, st.abs_mean, st.n_inf, st.n_nan, st.n_total);
        }

        // For split 4: pick argmax(logits[last_real_position]).
        if (is_last) {
            const int logits_idx = find_io_idx(out_info, s.n_out, "logits");
            if (logits_idx < 0) {
                std::fprintf(stderr,
                    "[qnn_bin] %s: missing logits output\n", path.c_str());
                free_split(s); return -3;
            }
            // logits shape [1, 128, vocab]; bytes = 1 * 128 * vocab * 2.
            const size_t total_bytes = s.out_sz[(size_t)logits_idx];
            const int    vocab = (int)(total_bytes / 2 / (size_t)ar);
            const uint16_t* logits =
                (const uint16_t*)s.out_bufs[(size_t)logits_idx];
            const int last_q = n_real_clamped - 1;
            const uint16_t* row = logits + (size_t)last_q * vocab;
            int    best_id = 0;
            float  best_v  = -1.0e30f;
            for (int v = 0; v < vocab; ++v) {
                const float f = sp_fp16_to_fp32(row[v]);
                if (f > best_v) { best_v = f; best_id = v; }
            }
            *picked_token_id = best_id;
            std::fprintf(stderr,
                "[qnn_bin] logits[%d]: argmax id=%d value=%.4f (vocab=%d)\n",
                last_q, best_id, best_v, vocab);
        }

        free_split(s);
        return 0;
    };

    int rc = run_layer_block(split_paths[1], /*is_last=*/false, nullptr);
    if (rc != 0) { sp_qnn_shutdown(); return rc; }
    rc = run_layer_block(split_paths[2], /*is_last=*/false, nullptr);
    if (rc != 0) { sp_qnn_shutdown(); return rc; }
    rc = run_layer_block(split_paths[3], /*is_last=*/true,
                          out_next_token_id);
    sp_qnn_shutdown();
    return rc;
}

}  // namespace sp::engine
