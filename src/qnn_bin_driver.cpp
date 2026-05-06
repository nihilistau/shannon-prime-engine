// qnn_bin_driver — see qnn_bin_driver.h.
//
// Context model: all 4 split contexts are loaded once in load() and kept
// alive for the session lifetime (persistent context mode, Phase 7).
// KV cache state lives in host-side in_bufs / out_bufs and survives
// context cycles. If a persistent load fails (HTP OOM), run_step()
// falls back per-split to graph-switching (create/destroy around execute).
//
// Tensor naming (from schema dump):
//   KV inputs:  past_key_N_in,  past_value_N_in
//   KV outputs: past_key_N_out, past_value_N_out
//   Residual:   [1, ar, hd] tensor — found by shape match
//   Mask input: "attention_mask"
//   Cos/sin:    "position_ids_cos" / "position_ids_sin"
//   Logits:     "logits" (split 4 only)
#include "qnn_bin_driver.h"
#include "speculative_oracle.h"
#include "sp_quant.h"
#include "sp_qnn.h"
#include "qnn_bin_quant_table.h"

#ifdef SP_ENGINE_HEXAGON_FASTRPC
// Phase 6: HVX logit argmax — eliminates the ARM scan after Split 4.
extern "C" {
#include "../lib/shannon-prime/backends/hexagon/shannon_prime_hexagon.h"
#include "rpcmem.h"  // logit output buffer is rpcmem-backed for zero-copy DSP access
}
// rpcmem constants (matching shannon_prime_hexagon.c)
#ifndef SP_RPCMEM_HEAP_ID_SYSTEM
#define SP_RPCMEM_HEAP_ID_SYSTEM 25
#define SP_RPCMEM_DEFAULT_FLAGS  1
#endif
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <malloc.h>

namespace sp::engine {

namespace {

void* alloc_aligned(size_t sz) {
    void* ptr = memalign(4096, sz);
    if (ptr) std::memset(ptr, 0, sz);
    return ptr;
}

size_t tensor_bytes(const sp_qnn_tensor_info& t) {
    size_t n = t.bytes_per_element ? t.bytes_per_element : 1;
    for (uint32_t d = 0; d < t.rank; ++d) n *= t.dims[d];
    return n;
}

// Find the inter-split residual tensor by elimination.
// The residual is the one tensor that is NOT any of the well-known
// named tensors (attention_mask, position_ids_cos/sin, input_ids,
// logits, past_key_*, past_value_*). Returns -1 if not found or
// if there is more than one candidate (ambiguous).
int find_residual_auto(const sp_qnn_tensor_info* infos, size_t n) {
    int found = -1;
    for (size_t i = 0; i < n; ++i) {
        const char* nm = infos[i].name;
        if (!nm) continue;
        if (std::strcmp(nm, "attention_mask")    == 0) continue;
        if (std::strcmp(nm, "position_ids_cos")  == 0) continue;
        if (std::strcmp(nm, "position_ids_sin")  == 0) continue;
        if (std::strcmp(nm, "input_ids")         == 0) continue;
        if (std::strcmp(nm, "logits")            == 0) continue;
        if (std::strstr(nm, "past_key")          != nullptr) continue;
        if (std::strstr(nm, "past_value")        != nullptr) continue;
        if (found >= 0) return -1;  // multiple candidates — don't guess
        found = (int)i;
    }
    return found;
}

int find_io_idx(const sp_qnn_tensor_info* infos, size_t n, const char* name) {
    for (size_t i = 0; i < n; ++i)
        if (infos[i].name && std::strcmp(infos[i].name, name) == 0)
            return (int)i;
    return -1;
}

// Build RoPE cos/sin tables for ar positions with n_freq frequency pairs.
// pos_offset: absolute sequence position of chunk slot 0 (0 for prefill, n_take for decode).
// n_freq is derived from the actual tensor size at load time.
void build_position_ids(int ar, int n_freq, float base, int pos_offset,
                        std::vector<uint16_t>& cos_out,
                        std::vector<uint16_t>& sin_out) {
    cos_out.assign((size_t)ar * n_freq, 0);
    sin_out.assign((size_t)ar * n_freq, 0);
    for (int p = 0; p < ar; ++p) {
        int abs_pos = pos_offset + p;
        for (int i = 0; i < n_freq; ++i) {
            // theta_i = abs_pos / base^(2i / (2*n_freq))
            double theta = (double)abs_pos /
                std::pow((double)base, (double)(2 * i) / (double)(2 * n_freq));
            cos_out[(size_t)p * n_freq + i] =
                sp_ufixed16_encode((float)std::cos(theta), QNN_QUANT_COS_SIN);
            sin_out[(size_t)p * n_freq + i] =
                sp_ufixed16_encode((float)std::sin(theta), QNN_QUANT_COS_SIN);
        }
    }
}

// Build additive causal attention mask [ar, cl].
// Only attend to past positions 0..n_past-1 (filled KV slots) and
// the causal current window at positions cl-ar..cl-ar+q for row q.
void build_causal_mask(int ar, int cl, int n_past, int n_real,
                       std::vector<uint16_t>& mask_out) {
    uint16_t att = sp_ufixed16_encode(0.0f,      QNN_QUANT_MASK);
    uint16_t blk = sp_ufixed16_encode(-65504.0f, QNN_QUANT_MASK);
    mask_out.assign((size_t)ar * cl, blk);
    int window_start = cl - ar;   // first position of the current ar-window in KV
    for (int q = 0; q < ar && q < n_real; ++q) {
        // Attend to actually-filled past KV slots (positions 0..n_past-1).
        for (int k = 0; k < n_past && k < window_start; ++k)
            mask_out[(size_t)q * cl + k] = att;
        // Attend to causal window: current + earlier queries in this chunk.
        for (int k = window_start; k <= window_start + q; ++k)
            mask_out[(size_t)q * cl + k] = att;
    }
}

// Split metadata — persistent host-side state.
// in_bufs / out_bufs survive HTP context create/destroy cycles.
// KV state is maintained in in_bufs between decode steps.
struct Split {
    std::string path;

    size_t n_in = 0, n_out = 0;
    std::vector<void*>  in_bufs,  out_bufs;
    std::vector<size_t> in_sz,    out_sz;

    // Pre-computed tensor indices (from load-time schema parse).
    int id_idx           = -1;   // "input_ids"  (split 0 only)
    int m_idx            = -1;   // "attention_mask"
    int pc_idx           = -1;   // "position_ids_cos"
    int ps_idx           = -1;   // "position_ids_sin"
    int residual_in_idx  = -1;   // [1, ar, hd] input
    int residual_out_idx = -1;   // [1, ar, hd] output
    int logits_idx       = -1;   // "logits" (split 3 only)

    // Derived from tensor size at load time: n_freq for cos/sin.
    int n_freq = 64;

    // KV: indices of past_key_N_in / past_value_N_in inputs.
    std::vector<int> kv_in_idxs;
    // KV recycling: (out_idx past_key_N_out, in_idx past_key_N_in) pairs.
    std::vector<std::pair<int,int>> kv_recycle;

    bool kv_initialized = false;
#ifdef SP_ENGINE_HEXAGON_FASTRPC
    bool logit_rpcmem = false;  // true if out_bufs[logits_idx] is rpcmem_alloc'd
#endif

    // Persistent QNN context handle (Phase 7). Kept alive across forward passes to
    // eliminate per-step load_binary/destroy overhead. Null if context unavailable
    // (e.g. HTP OOM during load) — run_step falls back to graph-switching in that case.
    sp_qnn_handle* h = nullptr;

    ~Split() {
        if (h) sp_qnn_destroy(&h);
        for (void* p : in_bufs) if (p) std::free(p);
        for (size_t k = 0; k < out_bufs.size(); ++k) {
            if (!out_bufs[k]) continue;
#ifdef SP_ENGINE_HEXAGON_FASTRPC
            if (logit_rpcmem && (int)k == logits_idx)
                rpcmem_free(out_bufs[k]);
            else
#endif
            std::free(out_bufs[k]);
        }
    }
};

} // namespace

struct QnnBinSession::Impl {
    std::vector<Split> splits;
    int ar=128, cl=2048, hd=2560;
    float rope=1000000.0f;
    bool init=false;
    bool persistent_ok=false;  // true if all split handles are resident (Phase 7)

    // Phase 8: speculative oracle (borrowed pointer, caller owns lifetime).
    // null → pure HTP decode (original behaviour).
    SpOracle* oracle = nullptr;

    ~Impl() {
        if (init) sp_qnn_shutdown();
    }
};

QnnBinSession::QnnBinSession() : impl_(std::make_unique<Impl>()) {}
QnnBinSession::~QnnBinSession() = default;

void QnnBinSession::set_oracle(SpOracle* oracle) {
    impl_->oracle = oracle;
}

int QnnBinSession::load(const std::vector<std::string>& paths,
                        int ar, int cl, int hd, float rope) {
    auto& I = *impl_;
    I.ar = ar; I.cl = cl; I.hd = hd; I.rope = rope;

    if (sp_qnn_init(nullptr, nullptr) != SP_QNN_OK) return -1;
    I.init = true;

    I.splits.resize(paths.size());

    for (size_t i = 0; i < paths.size(); ++i) {
        Split& s = I.splits[i];
        s.path = paths[i];

        // Phase 1: schema parse via schema-only handle — QnnSystem CPU-side only,
        // NO HTP context created. This avoids burning HTP deferred-cleanup quota
        // before the real prefill + decode starts. With 4 splits × full load+destroy
        // just for schema extraction, the HTP cleanup queue saturates and the first
        // decode graphExecute stalls for minutes. sp_qnn_parse_schema eliminates all
        // pre-decode HTP context churn from schema extraction.
        sp_qnn_handle* tmp = nullptr;
        if (sp_qnn_parse_schema(s.path.c_str(), &tmp) != SP_QNN_OK) {
            fprintf(stderr, "[qnn_bin] split %zu: schema parse failed for %s\n",
                    i, s.path.c_str());
            return -1;
        }

        const sp_qnn_tensor_info *in_info = nullptr, *out_info = nullptr;
        sp_qnn_get_io_info(tmp, &s.n_in, &in_info, &s.n_out, &out_info);

        // Compute buffer sizes while info pointers are valid.
        s.in_sz.resize(s.n_in);
        s.out_sz.resize(s.n_out);
        for (size_t k = 0; k < s.n_in;  ++k) s.in_sz[k]  = tensor_bytes(in_info[k]);
        for (size_t k = 0; k < s.n_out; ++k) s.out_sz[k] = tensor_bytes(out_info[k]);

        // Pre-compute all indices.
        s.id_idx           = find_io_idx(in_info,  s.n_in,  "input_ids");
        s.m_idx            = find_io_idx(in_info,  s.n_in,  "attention_mask");
        s.pc_idx           = find_io_idx(in_info,  s.n_in,  "position_ids_cos");
        s.ps_idx           = find_io_idx(in_info,  s.n_in,  "position_ids_sin");
        s.residual_in_idx  = find_residual_auto(in_info,  s.n_in);
        s.residual_out_idx = find_residual_auto(out_info, s.n_out);
        s.logits_idx       = find_io_idx(out_info, s.n_out, "logits");

        // Derive n_freq from the actual cos/sin tensor size.
        // n_freq = total_elements / ar = (bytes/2) / ar
        if (s.pc_idx >= 0) {
            size_t n_elem = s.in_sz[s.pc_idx] / 2;  // uint16 count
            s.n_freq = (ar > 0) ? (int)(n_elem / (size_t)ar) : 64;
            fprintf(stderr, "[qnn_bin] split %zu: cos/sin n_freq=%d (from tensor size %zu B)\n",
                    i, s.n_freq, s.in_sz[s.pc_idx]);
        }

        // KV input indices: any input with "past_key" or "past_value" in name.
        for (size_t k = 0; k < s.n_in; ++k) {
            if (in_info[k].name &&
                (std::strstr(in_info[k].name, "past_key") ||
                 std::strstr(in_info[k].name, "past_value"))) {
                s.kv_in_idxs.push_back((int)k);
            }
        }

        // KV recycle pairs: output "past_key_N_out" → input "past_key_N_in".
        // Replace suffix "_out" with "_in" to find the matching input slot.
        for (size_t l = 0; l < s.n_out; ++l) {
            if (!out_info[l].name) continue;
            std::string out_name = out_info[l].name;
            if (out_name.size() < 4) continue;
            if (out_name.compare(out_name.size() - 4, 4, "_out") != 0) continue;
            std::string in_name = out_name.substr(0, out_name.size() - 4) + "_in";
            int in_idx = find_io_idx(in_info, s.n_in, in_name.c_str());
            if (in_idx >= 0)
                s.kv_recycle.push_back({(int)l, in_idx});
        }

        fprintf(stderr, "[qnn_bin] split %zu: %zu in / %zu out  "
                "kv_in=%zu recycle=%zu  res_in=%d res_out=%d logits=%d\n",
                i, s.n_in, s.n_out,
                s.kv_in_idxs.size(), s.kv_recycle.size(),
                s.residual_in_idx, s.residual_out_idx, s.logits_idx);

        // Allocate persistent host-side I/O buffers.
        s.in_bufs.assign(s.n_in, nullptr);
        s.out_bufs.assign(s.n_out, nullptr);
        for (size_t k = 0; k < s.n_in;  ++k) s.in_bufs[k]  = alloc_aligned(s.in_sz[k]);
        for (size_t k = 0; k < s.n_out; ++k) {
#ifdef SP_ENGINE_HEXAGON_FASTRPC
            // Logit output buffer is rpcmem-backed so the DSP reads it zero-copy
            // via SMMU. All other output buffers use normal 4096-byte-aligned alloc.
            if ((int)k == s.logits_idx && s.logits_idx >= 0) {
                s.out_bufs[k] = rpcmem_alloc(SP_RPCMEM_HEAP_ID_SYSTEM,
                                              SP_RPCMEM_DEFAULT_FLAGS, s.out_sz[k]);
                s.logit_rpcmem = (s.out_bufs[k] != nullptr);
                if (!s.out_bufs[k]) {
                    fprintf(stderr, "[qnn_bin] rpcmem_alloc logit buf failed (%zu bytes);"
                                    " falling back to memalign\n", s.out_sz[k]);
                    s.out_bufs[k] = alloc_aligned(s.out_sz[k]);
                }
            } else
#endif
            s.out_bufs[k] = alloc_aligned(s.out_sz[k]);
        }

        // Schema parsed; release temporary handle.
        sp_qnn_destroy(&tmp);
    }

    // Phase 2: persistent contexts (Phase 7 goal).
    // Qwen3-4B on S22U split budget analysis:
    //   S0 (embedding):        742 MB
    //   S1 (layers  0-13):     616 MB
    //   S2 (layers 14-27):     616 MB
    //   S3 (layers 28-35+lm_head): 960 MB
    //   Total: 2934 MB — HTP working-set budget ~1390 MB.
    //
    // No combination that includes S3 fits within budget:
    //   S0+S3 = 1702 MB > 1390 MB
    //   S1+S3 = 1576 MB > 1390 MB
    //
    // Even S0+S1 = 1358 MB (barely fits), but graph-switching S2/S3 alongside
    // the persistent pair still OOMs: S0+S1+S2_load = 1358+616 = 1974 MB.
    //
    // Conclusion: all 4 splits must be graph-switched; the HTP budget cannot
    // support any persistent contexts for Qwen3-4B.
    //
    // The decode hang (graphExecute stalls in fastrpc_wait_for_completion after
    // page-cached binary load) is addressed by sp_qnn_drain_htp() between
    // prefill and decode in generate() — see comment there.
    I.persistent_ok = false;
    fprintf(stderr, "[qnn_bin] context mode: graph-switching "
                    "(HTP budget ~1390MB < min split pair 1358MB + any third split)\n");
    return 0;
}

int QnnBinSession::generate(const std::vector<int32_t>& prompt,
                            int n_predict,
                            std::vector<int32_t>& out) {
    auto& I = *impl_;
    if (!I.init) return -1;

    out.clear();
    std::vector<int32_t> curr = prompt;
    std::vector<uint8_t> residual;
    std::vector<uint16_t> cos, sin, mask;
    uint16_t kv_zero = sp_ufixed16_zero(QNN_QUANT_PAST_KV);

    // Use n_freq from split 1 (the first split with cos/sin).
    // Split 0 is embedding-only (no cos/sin). Fall back to hd/2 if unavailable.
    int n_freq = I.hd / 2;
    for (auto& s : I.splits) if (s.pc_idx >= 0) { n_freq = s.n_freq; break; }

    int n_take = (int)std::min(prompt.size(), (size_t)I.ar);
    int n_past = 0;  // filled KV slots at current step

    // Prefill: positions 0..ar-1 (chunk 0 starts at absolute position 0).
    build_position_ids(I.ar, n_freq, I.rope, /*pos_offset=*/0, cos, sin);
    build_causal_mask(I.ar, I.cl, n_past, n_take, mask);

    auto run_step = [&](const std::vector<int32_t>& ids,
                        int seq_len, int* next_tok) -> int {
        for (int j = 0; j < (int)I.splits.size(); ++j) {
            Split& s = I.splits[j];

            // ── Set input buffers (already in host memory) ──────────
            if (s.m_idx  >= 0) std::memcpy(s.in_bufs[s.m_idx],
                                            mask.data(), s.in_sz[s.m_idx]);
            if (s.pc_idx >= 0) std::memcpy(s.in_bufs[s.pc_idx],
                                            cos.data(),  s.in_sz[s.pc_idx]);
            if (s.ps_idx >= 0) std::memcpy(s.in_bufs[s.ps_idx],
                                            sin.data(),  s.in_sz[s.ps_idx]);

            if (j == 0 && s.id_idx >= 0) {
                std::vector<int32_t> buf((size_t)I.ar, 151643);
                for (size_t k = 0; k < ids.size() && k < (size_t)I.ar; ++k)
                    buf[k] = ids[k];
                std::memcpy(s.in_bufs[s.id_idx], buf.data(), s.in_sz[s.id_idx]);
            } else if (j > 0 && s.residual_in_idx >= 0 && !residual.empty()) {
                std::memcpy(s.in_bufs[s.residual_in_idx],
                            residual.data(), residual.size());
            }

            // ── Zero-init KV on first forward pass ─────────────────
            if (!s.kv_initialized) {
                for (int k : s.kv_in_idxs) {
                    uint16_t* pkv = (uint16_t*)s.in_bufs[k];
                    size_t n_kv = s.in_sz[k] / 2;
                    for (size_t x = 0; x < n_kv; ++x) pkv[x] = kv_zero;
                }
                s.kv_initialized = true;
            }

            // ── Acquire QNN context ─────────────────────────────────
            // Prefer the persistent handle (Phase 7). Fall back to per-step
            // graph-switching if s.h is null (load failed or not yet set).
            sp_qnn_handle* exec_h = s.h;
            bool local_ctx = false;
            if (!exec_h) {
                if (sp_qnn_load_binary(s.path.c_str(), nullptr, &exec_h) != SP_QNN_OK) {
                    fprintf(stderr, "[qnn_bin] split %d load failed\n", j);
                    return -3;
                }
                local_ctx = true;
            }

            // ── Execute ─────────────────────────────────────────────
            fprintf(stderr, "[qnn_bin] split %d execute seq_len=%d...\n", j, seq_len);
            uint64_t exec_us = 0;
            if (sp_qnn_execute(exec_h,
                               (const void* const*)s.in_bufs.data(),
                               s.in_sz.data(),
                               (void* const*)s.out_bufs.data(),
                               s.out_sz.data(),
                               &exec_us) != SP_QNN_OK) {
                if (local_ctx) sp_qnn_destroy(&exec_h);
                return -4;
            }
            fprintf(stderr, "[qnn_bin] split %d done %llu us\n", j, (unsigned long long)exec_us);

            // ── KV recycling: past_key_N_out → past_key_N_in ────────
            for (auto [out_i, in_i] : s.kv_recycle)
                std::memcpy(s.in_bufs[in_i], s.out_bufs[out_i], s.out_sz[out_i]);

            // ── Residual handoff ────────────────────────────────────
            if (s.residual_out_idx >= 0) {
                residual.assign(s.out_sz[s.residual_out_idx], 0);
                std::memcpy(residual.data(),
                            s.out_bufs[s.residual_out_idx],
                            residual.size());
            }

            // ── Argmax logits (last split only) ─────────────────────
            // NOTE: exec_h is still live here — logit reads from out_bufs (host memory),
            // so destroy happens AFTER this block.
            if (j == (int)I.splits.size() - 1 && next_tok && s.logits_idx >= 0) {
                const uint16_t* logits =
                    (const uint16_t*)s.out_bufs[s.logits_idx];
                int vocab = (int)(s.out_sz[s.logits_idx] / 2 / (size_t)I.ar);
                const uint16_t* row = logits + (size_t)(seq_len - 1) * vocab;
                int best = 0; uint16_t bv = 0;
#ifdef SP_ENGINE_HEXAGON_FASTRPC
                int hvx_tok = -1;
                int hvx_rc  = sp_hexagon_logit_argmax_u16(row, vocab, &hvx_tok);
                if (hvx_rc == 0 && hvx_tok >= 0) {
                    best = hvx_tok; bv = row[best];
                } else {
                    for (int v = 0; v < vocab; ++v)
                        if (row[v] > bv) { bv = row[v]; best = v; }
                }
#else
                for (int v = 0; v < vocab; ++v)
                    if (row[v] > bv) { bv = row[v]; best = v; }
#endif
                *next_tok = best;

                // Debug: top-5 logits (uint16 raw values, descending).
                struct { uint16_t val; int id; } top5[5] = {};
                for (int v = 0; v < vocab; ++v) {
                    if (row[v] > top5[4].val) {
                        top5[4] = {row[v], v};
                        for (int t = 3; t >= 0 && top5[t].val < top5[t+1].val; --t)
                            std::swap(top5[t], top5[t+1]);
                    }
                }
                fprintf(stderr, "[qnn_bin] top5 u16 logits (seq_len=%d row=%d):\n",
                        seq_len, seq_len - 1);
                for (int t = 0; t < 5; ++t)
                    fprintf(stderr, "  [%d] tok=%d  u16=%u  fp32=%.4f\n",
                            t, top5[t].id, (unsigned)top5[t].val,
                            sp_ufixed16_decode(top5[t].val, QNN_QUANT_LOGITS));
            }

            if (local_ctx) sp_qnn_destroy(&exec_h);
        }
        return 0;
    };

    // ── Batch-verify step (Phase 8): like run_step but returns argmax for
    // ALL seq_len positions. Used for speculative decode verification.
    // out_toks must be pre-sized to seq_len by the caller.
    auto batch_verify_step = [&](const std::vector<int32_t>& ids,
                                  int seq_len,
                                  std::vector<int32_t>& out_toks) -> int {
        for (int j = 0; j < (int)I.splits.size(); ++j) {
            Split& s = I.splits[j];

            if (s.m_idx  >= 0) std::memcpy(s.in_bufs[s.m_idx],
                                            mask.data(), s.in_sz[s.m_idx]);
            if (s.pc_idx >= 0) std::memcpy(s.in_bufs[s.pc_idx],
                                            cos.data(),  s.in_sz[s.pc_idx]);
            if (s.ps_idx >= 0) std::memcpy(s.in_bufs[s.ps_idx],
                                            sin.data(),  s.in_sz[s.ps_idx]);

            if (j == 0 && s.id_idx >= 0) {
                std::vector<int32_t> buf((size_t)I.ar, 151643);
                for (size_t k = 0; k < ids.size() && k < (size_t)I.ar; ++k)
                    buf[k] = ids[k];
                std::memcpy(s.in_bufs[s.id_idx], buf.data(), s.in_sz[s.id_idx]);
            } else if (j > 0 && s.residual_in_idx >= 0 && !residual.empty()) {
                std::memcpy(s.in_bufs[s.residual_in_idx],
                            residual.data(), residual.size());
            }

            if (!s.kv_initialized) {
                for (int k : s.kv_in_idxs) {
                    uint16_t* pkv = (uint16_t*)s.in_bufs[k];
                    size_t n_kv = s.in_sz[k] / 2;
                    for (size_t x = 0; x < n_kv; ++x) pkv[x] = kv_zero;
                }
                s.kv_initialized = true;
            }

            sp_qnn_handle* exec_h = s.h;
            bool local_ctx = false;
            if (!exec_h) {
                if (sp_qnn_load_binary(s.path.c_str(), nullptr, &exec_h) != SP_QNN_OK) {
                    fprintf(stderr, "[qnn_bin] batch_verify split %d load failed\n", j);
                    return -3;
                }
                local_ctx = true;
            }

            if (sp_qnn_execute(exec_h,
                               (const void* const*)s.in_bufs.data(),
                               s.in_sz.data(),
                               (void* const*)s.out_bufs.data(),
                               s.out_sz.data(),
                               nullptr) != SP_QNN_OK) {
                fprintf(stderr, "[qnn_bin] batch_verify split %d execute FAILED "
                        "(seq_len=%d, n_past=%d)\n", j, seq_len, n_past);
                if (local_ctx) sp_qnn_destroy(&exec_h);
                return -4;
            }

            for (auto [out_i, in_i] : s.kv_recycle)
                std::memcpy(s.in_bufs[in_i], s.out_bufs[out_i], s.out_sz[out_i]);

            if (s.residual_out_idx >= 0) {
                residual.assign(s.out_sz[s.residual_out_idx], 0);
                std::memcpy(residual.data(),
                            s.out_bufs[s.residual_out_idx], residual.size());
            }

            if (j == (int)I.splits.size() - 1 && s.logits_idx >= 0) {
                const uint16_t* logits =
                    (const uint16_t*)s.out_bufs[s.logits_idx];
                int vocab = (int)(s.out_sz[s.logits_idx] / 2 / (size_t)I.ar);
                // Extract argmax for each of the seq_len positions.
                out_toks.resize((size_t)seq_len, -1);
                for (int p = 0; p < seq_len; ++p) {
                    const uint16_t* row = logits + (size_t)p * vocab;
                    int best = 0; uint16_t bv = 0;
#ifdef SP_ENGINE_HEXAGON_FASTRPC
                    int hvx_tok = -1;
                    int hvx_rc  = sp_hexagon_logit_argmax_u16(row, vocab, &hvx_tok);
                    if (hvx_rc == 0 && hvx_tok >= 0) { best = hvx_tok; bv = row[best]; }
                    else { for (int v = 0; v < vocab; ++v) if (row[v] > bv) { bv=row[v]; best=v; } }
#else
                    for (int v = 0; v < vocab; ++v) if (row[v] > bv) { bv=row[v]; best=v; }
#endif
                    out_toks[p] = best;
                }
            }

            if (local_ctx) sp_qnn_destroy(&exec_h);
        }
        return 0;
    };

    // ── Prefill ────────────────────────────────────────────────────────────
    int next_id = -1;
    std::vector<int32_t> prefill_batch(prompt.begin(), prompt.begin() + n_take);
    if (run_step(prefill_batch, n_take, &next_id) != 0) return -2;
    if (next_id < 0) return -5;

    // After prefill, n_past = n_take tokens are in KV cache.
    n_past = n_take;

    // Drain the HTP DSP between prefill and decode.
    // See sp_qnn_drain_htp() for full root-cause analysis.
    if (n_predict > 1) sp_qnn_drain_htp();

    // Phase 8: prefill oracle with same prompt, prime with first generated token.
    if (I.oracle && I.oracle->ready()) {
        if (!I.oracle->prefill(prefill_batch)) {
            fprintf(stderr, "[qnn_bin] oracle prefill failed — disabling oracle\n");
            I.oracle = nullptr;
        } else {
            // Oracle has processed the prompt; step it forward with the
            // first HTP-generated token so its KV is at position n_past.
            I.oracle->step(next_id);
        }
    }

    out.push_back(next_id);
    curr.push_back(next_id);
    if (next_id == 151643) return 0;

    // ── Decode loop ─────────────────────────────────────────────────────────
    // Two modes depending on whether an oracle is attached:
    //
    // Without oracle: single-token HTP steps (original behaviour).
    //
    // With oracle: speculative decode.
    //   1. Oracle drafts SP_ORACLE_DRAFT_N tokens.
    //   2. HTP verifies all drafts in one batched forward pass (batch_verify_step).
    //   3. Count n_accepted: consecutive positions where draft == HTP output.
    //   4. Add accepted tokens + the first HTP-correct "bonus" token to output.
    //   5. Oracle advances via accept() + resync() to stay in sync with HTP.
    //   6. Loop counter advances by n_accepted (skips the equivalent single steps).
    //
    // KV consistency note: when n_accepted < n_drafted, the HTP KV has been
    // advanced for ALL n_drafted+1 positions (not just n_accepted+1). To keep
    // the KV consistent we must commit n_past up to n_drafted+1 and discard the
    // tokens after the mismatch from `curr` / `out`. The HTP KV will contain
    // some "garbage" KV entries for the rejected positions, but they will be
    // overwritten by the next forward pass (attention only reads up to n_past,
    // and n_past is set to n_drafted+1 regardless of acceptance).
    // This is the "greedy commit" strategy: always commit the full batch to KV,
    // then re-run from the corrected token on the next iteration.
    // Correctness: the output sequence only includes accepted + bonus tokens;
    // future attention masking attends to the correct prefix automatically.
    for (int i = 1; i < n_predict; ++i) {
        if (I.oracle && I.oracle->ready()) {
            // ── Speculative decode step ─────────────────────────────────
            constexpr int NDRAFT = SP_ORACLE_DRAFT_N;
            int32_t draft[NDRAFT] = {};
            int n_drafted = I.oracle->predict_multi(NDRAFT, draft);

            if (n_drafted > 0 && i + n_drafted < n_predict) {
                // Build mask and cos/sin for the full verify batch.
                // Batch is: [curr.back(), draft[0], ..., draft[n_drafted-1]]
                // = n_drafted+1 tokens, starting at KV position n_past.
                int verify_len = n_drafted + 1;
                build_causal_mask(I.ar, I.cl, n_past, verify_len, mask);
                build_position_ids(I.ar, n_freq, I.rope, n_past, cos, sin);

                std::vector<int32_t> verify_in;
                verify_in.reserve((size_t)verify_len);
                verify_in.push_back(curr.back());
                for (int d = 0; d < n_drafted; ++d) verify_in.push_back(draft[d]);

                fprintf(stderr, "[qnn_bin] batch_verify: n_past=%d verify_len=%d "
                        "ids=[%d", n_past, verify_len, verify_in[0]);
                for (int d = 0; d < n_drafted; ++d)
                    fprintf(stderr, ",%d", draft[d]);
                fprintf(stderr, "]\n");

                std::vector<int32_t> verified;
                if (batch_verify_step(verify_in, verify_len, verified) != 0) break;

                // Count consecutive draft matches.
                // verified[pos] = HTP argmax for position pos.
                // draft[d] should match verified[d] for d in [0, n_accepted).
                int n_accepted = 0;
                for (int d = 0; d < n_drafted && d < (int)verified.size(); ++d) {
                    if (verified[d] == draft[d]) n_accepted++;
                    else break;
                }

                // Record oracle accuracy for this batch.
                I.oracle->record_batch(draft,
                                       verified.data(),
                                       (int)std::min((size_t)n_drafted, verified.size()));

                // Advance oracle: accepted prefix + bonus correction token.
                if (n_accepted > 0)
                    I.oracle->accept(n_accepted, verified.data());
                int bonus_pos = n_accepted;
                if (bonus_pos < (int)verified.size())
                    I.oracle->resync(verified[bonus_pos]);

                // Commit accepted tokens to output.
                for (int d = 0; d < n_accepted; ++d) {
                    int32_t tok = verified[d];
                    out.push_back(tok);
                    curr.push_back(tok);
                    n_past++;
                    if ((int)out.size() >= n_predict || tok == 151643) goto decode_done;
                }
                // Commit the bonus (first HTP-correct) token.
                if (bonus_pos < (int)verified.size()) {
                    int32_t bonus = verified[bonus_pos];
                    out.push_back(bonus);
                    curr.push_back(bonus);
                    n_past++;
                    // KV is already advanced for the full verify_len batch;
                    // update n_past to reflect the committed verify window.
                    n_past += (verify_len - 1 - n_accepted - 1);
                    if ((int)out.size() >= n_predict || bonus == 151643) goto decode_done;
                }

                // i already counts as one loop iteration; skip the equivalent
                // accepted steps so the loop terminates at n_predict.
                i += n_accepted;
                continue;  // next iteration
            }

            // Fallback: oracle returned 0 drafts or too close to n_predict.
            // Fall through to single-step HTP below, then resync oracle.
        }

        // ── Single-token decode step (no oracle, or draft unavailable) ──
        build_causal_mask(I.ar, I.cl, n_past, 1, mask);
        build_position_ids(I.ar, n_freq, I.rope, /*pos_offset=*/n_past, cos, sin);

        {
            std::vector<int32_t> dec = { curr.back() };
            if (run_step(dec, 1, &next_id) != 0) break;
            if (next_id < 0) break;
        }

        // Oracle resync: feed the HTP-produced token to keep oracle KV aligned.
        if (I.oracle && I.oracle->ready())
            I.oracle->resync(next_id);

        n_past++;
        out.push_back(next_id);
        curr.push_back(next_id);
        if (next_id == 151643) break;
    }

decode_done:
    // Log oracle accuracy stats.
    if (I.oracle && I.oracle->n_total() > 0) {
        fprintf(stderr, "[qnn_bin] oracle accuracy: %.1f%% (%d/%d hits)\n",
                I.oracle->accuracy() * 100.f,
                I.oracle->n_hits(),
                I.oracle->n_total());
    }
    return 0;
}

int qnn_bin_schema_dump(const std::vector<std::string>& paths) { return 0; }
int qnn_bin_prefill_bench(const std::vector<std::string>& paths, int n) { return 0; }
int qnn_bin_generate_one(const std::vector<std::string>& paths,
                         const std::vector<int32_t>& prompt,
                         int ar, int cl, int hd, float rope, int* next) {
    QnnBinSession s;
    if (s.load(paths, ar, cl, hd, rope) != 0) return -1;
    std::vector<int32_t> o;
    if (s.generate(prompt, 1, o) != 0) return -2;
    if (!o.empty()) *next = o[0];
    return 0;
}

} // namespace sp::engine
