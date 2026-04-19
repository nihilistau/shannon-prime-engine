// Shannon-Prime Engine — compressed KV cache (impl)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "kv_cache.h"

extern "C" {
#include "shannon_prime.h"
}

#ifdef SP_ENGINE_WITH_CUDA
extern "C" {
#include "shannon_prime_cuda.h"
}
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sp::engine {

namespace {

// Parse "5,5,4,3" → vector of int. Returns empty on parse error.
static std::vector<int> parse_csv_bits(const std::string& csv) {
    std::vector<int> out;
    if (csv.empty()) return out;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        const std::string tok = csv.substr(i, j - i);
        if (!tok.empty()) {
            int v = std::atoi(tok.c_str());
            if (v < 1 || v > 8) return {};
            out.push_back(v);
        }
        i = j + 1;
    }
    return out;
}

} // namespace

struct KvCache::Impl {
    bool                 sqfree = false;
    bool                 hierarchical = false;
    int                  n_layer    = 0;
    int                  n_head_kv  = 0;
    int                  head_dim   = 0;
    int                  pad_dim    = 0;   // = head_dim (ship) or sqfree pad (sqfree/hier)
    int                  max_seq    = 0;

    // Exactly one of these is initialised at any time.
    sp_shadow_cache_t    shadow{};
    sp_sqfree_cache_t    sq{};
    sp_hier_cache_t      hier{};

    bool                 shadow_inited = false;
    bool                 sq_inited     = false;
    bool                 hier_inited   = false;

    sp_config_t          cfg{};
    bool                 calibrated = false;

    // For shadow path: bytes per K/V vector (used for slot sizing).
    size_t               k_bytes = 0;
    size_t               v_bytes = 0;

#ifdef SP_ENGINE_WITH_CUDA
    // GPU-resident ship cache (step 3). When cuda_inited is true, the
    // shadow cache above is unused — reads/writes route to cuda_cache.
    sp_cuda_cache_t      cuda_cache{};
    bool                 cuda_inited = false;
    // Scratch for host->device staging of input vectors on write, and
    // for device->device per-head strided output on read. Lazy-sized.
    float*               d_stage       = nullptr;
    size_t               d_stage_bytes = 0;
    float*               d_scratch_batch       = nullptr;  // [hd, max_seq] per-head read scratch
    size_t               d_scratch_batch_bytes = 0;

    // GPU-resident sqfree cache (step 3 MVP, no spinor). Exclusive with
    // cuda_cache — when cuda_sqfree_inited is true the ship path is off.
    sp_cuda_sqfree_cache_t cuda_sqfree_cache{};
    bool                   cuda_sqfree_inited = false;
#endif

    // Cauchy reset system (decode-chain causal stability). Non-null when
    // init_cauchy() was called with mode > 0.
    sp_ricci_sentinel_t  *ricci   = nullptr;
    sp_mertens_oracle_t  *mertens = nullptr;
    sp_cauchy_ctrl_t      cauchy{};
    bool                  cauchy_inited = false;
    std::vector<float>    vht2_scratch;   // head_dim-sized scratch for Ricci feed

    ~Impl() {
        if (shadow_inited) {
            // We allocated k_cache/v_cache slot pointers + per-slot buffers.
            const int n_slots = n_layer * n_head_kv;
            if (shadow.k_cache) {
                for (int s = 0; s < n_slots; ++s) std::free(shadow.k_cache[s]);
                std::free(shadow.k_cache);
            }
            if (shadow.v_cache) {
                for (int s = 0; s < n_slots; ++s) std::free(shadow.v_cache[s]);
                std::free(shadow.v_cache);
            }
            shadow.k_cache = nullptr;
            shadow.v_cache = nullptr;
            sp_shadow_cache_free(&shadow);
        }
        if (sq_inited) {
            sp_sqfree_cache_free(&sq);
        }
        if (hier_inited) {
            sp_hier_cache_free(&hier);
        }
#ifdef SP_ENGINE_WITH_CUDA
        if (cuda_inited) {
            if (d_stage)         cudaFree(d_stage);
            if (d_scratch_batch) cudaFree(d_scratch_batch);
            sp_cuda_cache_free(&cuda_cache);
        }
        if (cuda_sqfree_inited) {
            sp_cuda_sqfree_cache_free(&cuda_sqfree_cache);
        }
#endif
        // Cauchy system components.
        delete ricci;
        delete mertens;
        ricci   = nullptr;
        mertens = nullptr;
    }
};

KvCache::KvCache() : impl_(std::make_unique<Impl>()) {}
KvCache::~KvCache() = default;

int  KvCache::n_layer()    const { return impl_->n_layer; }
int  KvCache::n_head_kv()  const { return impl_->n_head_kv; }
int  KvCache::head_dim()   const { return impl_->head_dim; }
int  KvCache::max_seq()    const { return impl_->max_seq; }
bool KvCache::is_sqfree()  const { return impl_->sqfree; }

float KvCache::compression_ratio() const {
    return sp_compression_ratio(&impl_->cfg);
}

std::string KvCache::describe() const {
    const char* mode = impl_->hierarchical ? "hierarchical" :
                       impl_->sqfree       ? "sqfree"       : "shadow";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s cache: n_layer=%d n_head_kv=%d head_dim=%d "
                  "pad_dim=%d max_seq=%d compression=%.2fx",
                  mode,
                  impl_->n_layer, impl_->n_head_kv, impl_->head_dim,
                  impl_->pad_dim, impl_->max_seq, compression_ratio());
    if (impl_->hier_inited) {
        char extra[128];
        std::snprintf(extra, sizeof(extra), " skel=%d/%d (%.0f%%)",
                      impl_->hier.predictors[0].n_skeleton,
                      impl_->hier.predictors[0].pad_dim,
                      100.0f * impl_->hier.predictors[0].n_skeleton /
                      impl_->hier.predictors[0].pad_dim);
        std::strncat(buf, extra, sizeof(buf) - std::strlen(buf) - 1);
    }
    return std::string(buf);
}

std::unique_ptr<KvCache> KvCache::create(int n_layer, int n_head_kv,
                                         int head_dim, int max_seq,
                                         const Config& cfg) {
    if (n_layer <= 0 || n_head_kv <= 0 || head_dim <= 0 || max_seq <= 0) {
        std::fprintf(stderr, "[sp-engine] KvCache: bad dims\n");
        return nullptr;
    }
    auto kv = std::unique_ptr<KvCache>(new KvCache());
    kv->impl_->sqfree    = cfg.sqfree;
    kv->impl_->n_layer   = n_layer;
    kv->impl_->n_head_kv = n_head_kv;
    kv->impl_->head_dim  = head_dim;
    kv->impl_->max_seq   = max_seq;

    sp_config_t* sc = &kv->impl_->cfg;
    sp_config_init(sc, head_dim, n_layer, n_head_kv);
    sc->use_mobius_mask = cfg.mobius;

    auto kbits = parse_csv_bits(cfg.k_bits_csv);
    auto vbits = parse_csv_bits(cfg.v_bits_csv);
    if (kbits.empty()) kbits = {5, 5, 4, 3};
    if (vbits.empty()) vbits = {3};
    if ((int)kbits.size() > SP_MAX_BANDS) kbits.resize(SP_MAX_BANDS);
    if ((int)vbits.size() > SP_MAX_BANDS) vbits.resize(SP_MAX_BANDS);
    sc->k_n_bands = (int)kbits.size();
    sc->v_n_bands = (int)vbits.size();
    for (size_t i = 0; i < kbits.size(); ++i) sc->k_band_bits[i] = kbits[i];
    for (size_t i = 0; i < vbits.size(); ++i) sc->v_band_bits[i] = vbits[i];

    // Hierarchical Vilenkin predictor — maximum compression path.
    // Mutually exclusive with sqfree; takes precedence if both are set.
    if (cfg.hierarchical) {
        auto skel_bits = parse_csv_bits(cfg.hier_skel_bits);
        if (skel_bits.empty()) skel_bits = {5, 5};
        if ((int)skel_bits.size() > SP_MAX_BANDS) skel_bits.resize(SP_MAX_BANDS);
        const int res_bits = (cfg.hier_res_bits >= 1 && cfg.hier_res_bits <= 4)
                             ? cfg.hier_res_bits : 2;

        if (sp_hier_cache_init(&kv->impl_->hier, sc, max_seq,
                               cfg.hier_level,
                               (int)skel_bits.size(), skel_bits.data(),
                               res_bits) != 0) {
            std::fprintf(stderr, "[sp-engine] KvCache: hier_cache_init failed\n");
            return nullptr;
        }
        kv->impl_->hier_inited   = true;
        kv->impl_->hierarchical  = true;
        kv->impl_->sqfree        = false;  // hierarchical supersedes sqfree
        kv->impl_->pad_dim       = kv->impl_->hier.pad_dim;
        return kv;
    }

    if (cfg.sqfree) {
        const int rbits = (cfg.residual_bits >= 1 && cfg.residual_bits <= 4) ? cfg.residual_bits : 3;
        if (sp_sqfree_cache_init(&kv->impl_->sq, sc, max_seq, rbits, cfg.spinor) != 0) {
            std::fprintf(stderr, "[sp-engine] KvCache: sqfree_cache_init failed\n");
            return nullptr;
        }
        kv->impl_->sq_inited = true;
        kv->impl_->pad_dim   = kv->impl_->sq.pad_dim;
        return kv;
    }

    // Ship path.
    if (sp_shadow_cache_init(&kv->impl_->shadow, sc) != 0) {
        std::fprintf(stderr, "[sp-engine] KvCache: shadow_cache_init failed\n");
        return nullptr;
    }
    kv->impl_->shadow_inited = true;
    kv->impl_->pad_dim       = head_dim;
    kv->impl_->k_bytes       = (size_t)kv->impl_->shadow.k_bands.total_bytes;
    kv->impl_->v_bytes       = (size_t)kv->impl_->shadow.v_bands.total_bytes;

    // Allocate per-slot storage (shadow init does not, by design).
    const int n_slots = n_layer * n_head_kv;
    kv->impl_->shadow.k_cache = (uint8_t**)std::calloc((size_t)n_slots, sizeof(uint8_t*));
    kv->impl_->shadow.v_cache = (uint8_t**)std::calloc((size_t)n_slots, sizeof(uint8_t*));
    if (!kv->impl_->shadow.k_cache || !kv->impl_->shadow.v_cache) {
        std::fprintf(stderr, "[sp-engine] KvCache: slot array OOM\n");
        return nullptr;
    }
    for (int s = 0; s < n_slots; ++s) {
        kv->impl_->shadow.k_cache[s] = (uint8_t*)std::calloc((size_t)max_seq, kv->impl_->k_bytes);
        kv->impl_->shadow.v_cache[s] = (uint8_t*)std::calloc((size_t)max_seq, kv->impl_->v_bytes);
        if (!kv->impl_->shadow.k_cache[s] || !kv->impl_->shadow.v_cache[s]) {
            std::fprintf(stderr, "[sp-engine] KvCache: per-slot OOM at %d\n", s);
            return nullptr;
        }
    }
    return kv;
}

bool KvCache::write(int layer, int pos_offset, int n_tokens,
                    const float* K_flat, const float* V_flat) {
    if (layer < 0 || layer >= impl_->n_layer) return false;
    if (pos_offset < 0 || pos_offset + n_tokens > impl_->max_seq) return false;
    const int H  = impl_->n_head_kv;
    const int hd = impl_->head_dim;

    // Ricci sentinel feed. We sample head-0 / layer-0 once per position —
    // p=3 band energy is structurally consistent across heads, so one
    // probe is enough and keeps this off the GPU hot path. Fires only
    // when the sentinel has been calibrated (mode 2, post-calibration).
    if (impl_->ricci && impl_->ricci->calibrated && layer == 0) {
        if ((int)impl_->vht2_scratch.size() < hd) {
            impl_->vht2_scratch.assign((size_t)hd, 0.0f);
        }
        for (int q = 0; q < n_tokens; ++q) {
            const float* k_vec = K_flat + (size_t)(q * H + 0) * hd;
            std::memcpy(impl_->vht2_scratch.data(), k_vec,
                        (size_t)hd * sizeof(float));
            sp_vht2_forward_f32(impl_->vht2_scratch.data(), hd);
            sp_ricci_check(impl_->ricci, impl_->vht2_scratch.data(), hd);
        }
    }

#ifdef SP_ENGINE_WITH_CUDA
    // GPU-resident cache (ship or sqfree): stage host vectors to device
    // once, then call write_gpu which loops per-slot compress on GPU.
    // Used by prefill (forward_full pulls K/V back to host) and
    // diagnostic code. Staging cost is one-time per prefill chunk,
    // not per decode step.
    if (impl_->cuda_inited || impl_->cuda_sqfree_inited) {
        const size_t n_elems = (size_t)n_tokens * H * hd;
        const size_t n_bytes = n_elems * sizeof(float);
        float* d_stage_K = nullptr;
        float* d_stage_V = nullptr;
        if (cudaMalloc((void**)&d_stage_K, n_bytes) != cudaSuccess ||
            cudaMalloc((void**)&d_stage_V, n_bytes) != cudaSuccess) {
            if (d_stage_K) cudaFree(d_stage_K);
            std::fprintf(stderr, "[sp-engine] write(GPU): stage malloc failed\n");
            return false;
        }
        cudaStream_t s = impl_->cuda_inited
            ? (cudaStream_t)impl_->cuda_cache.stream
            : (cudaStream_t)impl_->cuda_sqfree_cache.stream;
        cudaMemcpyAsync(d_stage_K, K_flat, n_bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_stage_V, V_flat, n_bytes, cudaMemcpyHostToDevice, s);
        const bool ok = write_gpu(layer, pos_offset, n_tokens, d_stage_K, d_stage_V);
        cudaStreamSynchronize(s);
        cudaFree(d_stage_K);
        cudaFree(d_stage_V);
        return ok;
    }
#endif

    // Per-position, per-head writes. The K_flat / V_flat layout is the
    // ggml convention `[head_dim, n_head_kv, n_tokens]` — innermost is
    // head_dim, outermost is the position. This is what
    // ggml_backend_tensor_get hands back for the K/V tensors built in
    // build_block.
    for (int q = 0; q < n_tokens; ++q) {
        const int pos = pos_offset + q;
        for (int h = 0; h < H; ++h) {
            const float* k_vec = K_flat + (size_t)(q * H + h) * hd;
            const float* v_vec = V_flat + (size_t)(q * H + h) * hd;
            if (impl_->hier_inited) {
                sp_hier_cache_write_k(&impl_->hier, layer, h, pos, k_vec);
                sp_hier_cache_write_v(&impl_->hier, layer, h, pos, v_vec);
            } else if (impl_->sqfree) {
                sp_sqfree_write_k(&impl_->sq, layer, h, pos, k_vec);
                sp_sqfree_write_v(&impl_->sq, layer, h, pos, v_vec);
            } else {
                sp_shadow_write_k(&impl_->shadow, layer, h, pos, k_vec);
                sp_shadow_write_v(&impl_->shadow, layer, h, pos, v_vec);
            }
        }
    }
    return true;
}

bool KvCache::read(int layer, int kv_len,
                   std::vector<float>& K_out,
                   std::vector<float>& V_out) const {
    if (layer < 0 || layer >= impl_->n_layer) return false;
    if (kv_len < 0 || kv_len > impl_->max_seq) return false;
    const int H  = impl_->n_head_kv;
    const int hd = impl_->head_dim;
    K_out.assign((size_t)kv_len * H * hd, 0.0f);
    V_out.assign((size_t)kv_len * H * hd, 0.0f);

#ifdef SP_ENGINE_WITH_CUDA
    // GPU-resident cache (ship or sqfree): read on device into scratch,
    // then cudaMemcpy back to host. Used only by diagnostic code paths
    // (forward.cpp's decode() takes the zero-copy read_gpu fast path).
    if ((impl_->cuda_inited || impl_->cuda_sqfree_inited) && kv_len > 0) {
        const size_t n_bytes = (size_t)kv_len * H * hd * sizeof(float);
        float* d_K = nullptr;
        float* d_V = nullptr;
        if (cudaMalloc((void**)&d_K, n_bytes) != cudaSuccess ||
            cudaMalloc((void**)&d_V, n_bytes) != cudaSuccess) {
            if (d_K) cudaFree(d_K);
            std::fprintf(stderr, "[sp-engine] read(GPU fallback): malloc failed\n");
            return false;
        }
        cudaStream_t s = impl_->cuda_inited
            ? (cudaStream_t)impl_->cuda_cache.stream
            : (cudaStream_t)impl_->cuda_sqfree_cache.stream;
        const bool ok = read_gpu(layer, kv_len, d_K, d_V);
        if (ok) {
            cudaMemcpyAsync(K_out.data(), d_K, n_bytes, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(V_out.data(), d_V, n_bytes, cudaMemcpyDeviceToHost, s);
        }
        cudaStreamSynchronize(s);
        cudaFree(d_K);
        cudaFree(d_V);
        return ok;
    }
#endif

    for (int q = 0; q < kv_len; ++q) {
        for (int h = 0; h < H; ++h) {
            float* k_vec = K_out.data() + (size_t)(q * H + h) * hd;
            float* v_vec = V_out.data() + (size_t)(q * H + h) * hd;
            if (impl_->hier_inited) {
                sp_hier_cache_read_k(&impl_->hier, layer, h, q, k_vec);
                sp_hier_cache_read_v(&impl_->hier, layer, h, q, v_vec);
            } else if (impl_->sqfree) {
                sp_sqfree_read_k(&impl_->sq, layer, h, q, k_vec);
                sp_sqfree_read_v(&impl_->sq, layer, h, q, v_vec);
            } else {
                sp_shadow_read_k(&impl_->shadow, layer, h, q, k_vec);
                sp_shadow_read_v(&impl_->shadow, layer, h, q, v_vec);
            }
        }
    }
    return true;
}

// ── Adaptive calibration ────────────────────────────────────────────

bool KvCache::calibrate_begin() {
    if (impl_->hier_inited) {
        return sp_hier_cache_calibrate_begin(&impl_->hier) == 0;
    } else if (impl_->sqfree) {
        return sp_sqfree_calibrate_begin(&impl_->sq) == 0;
    } else {
        return sp_shadow_calibrate_begin(&impl_->shadow) == 0;
    }
}

void KvCache::calibrate_feed(const float* vec) {
    // Ricci sentinel also needs calibration to learn its p=3 baseline.
    // Fed from the same raw vectors used for cache calibration — works
    // when init_cauchy was called BEFORE the calibration pass (otherwise
    // ricci stays uncalibrated and sp_ricci_check returns false).
    if (impl_->ricci && !impl_->ricci->calibrated) {
        const int hd = impl_->head_dim;
        if ((int)impl_->vht2_scratch.size() < hd) {
            impl_->vht2_scratch.assign((size_t)hd, 0.0f);
        }
        std::memcpy(impl_->vht2_scratch.data(), vec,
                    (size_t)hd * sizeof(float));
        sp_vht2_forward_f32(impl_->vht2_scratch.data(), hd);
        sp_ricci_calibrate_feed(impl_->ricci,
                                 impl_->vht2_scratch.data(), hd);
    }

    // Shared-mask feed: sqfree and shadow accumulate globally.
    // NOT valid for hierarchical — use the per-slot overload instead.
    if (impl_->sqfree) {
        sp_sqfree_calibrate_feed(&impl_->sq, vec);
        return;
    }
    if (impl_->hier_inited) return;
#ifdef SP_ENGINE_WITH_CUDA
    if (impl_->cuda_inited && impl_->shadow_inited &&
        impl_->shadow.calibrating && impl_->shadow.calib_sum &&
        impl_->shadow.calib_sum2) {
        // GPU-domain calibration: upload vector to GPU, run VHT2 there,
        // download the result, and accumulate variance in shadow's
        // calib_sum / calib_sum2 using the GPU-VHT2'd values. Writing
        // straight into shadow's accumulators lets calibrate_end reuse
        // the existing variance-ranking logic verbatim — but the ranking
        // now reflects the GPU-domain coefficient structure, which is
        // what the GPU cache actually compresses. Without this, var_order
        // ranks CPU-domain variance and the sync to d_mobius_order gives
        // a wrong-domain order that regresses PPL.
        const int hd = impl_->head_dim;
        cudaStream_t s = (cudaStream_t)impl_->cuda_cache.stream;

        cudaMemcpyAsync(impl_->d_stage, vec, (size_t)hd * sizeof(float),
                        cudaMemcpyHostToDevice, s);
        sp_cuda_vht2_forward(impl_->d_stage, hd, 1, impl_->cuda_cache.stream);
        std::vector<float> h_tmp((size_t)hd);
        cudaMemcpyAsync(h_tmp.data(), impl_->d_stage,
                        (size_t)hd * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        for (int i = 0; i < hd; i++) {
            double v = (double)h_tmp[(size_t)i];
            impl_->shadow.calib_sum[i]  += v;
            impl_->shadow.calib_sum2[i] += v * v;
        }
        impl_->shadow.calib_n++;
        return;
    }
#endif
    sp_shadow_calibrate_feed(&impl_->shadow, vec);
}

void KvCache::calibrate_feed(int slot, const float* vec) {
    // Per-slot feed: hierarchical mode trains each predictor on its own
    // head's data. slot = layer * n_head_kv + head.
    if (impl_->hier_inited) {
        sp_hier_cache_calibrate_feed(&impl_->hier, slot, vec);
    } else {
        // Non-hierarchical paths ignore the slot — forward to shared feed.
        calibrate_feed(vec);
    }
}

bool KvCache::calibrate_end() {
    int rc;
    if (impl_->hier_inited) {
        rc = sp_hier_cache_calibrate_end(&impl_->hier);
    } else if (impl_->sqfree) {
        rc = sp_sqfree_calibrate_end(&impl_->sq);
    } else {
        rc = sp_shadow_calibrate_end(&impl_->shadow);
    }
    if (rc == 0) {
        impl_->calibrated = true;
        // Finalize Ricci sentinel calibration if it was fed.
        if (impl_->ricci && !impl_->ricci->calibrated &&
            impl_->ricci->calib_n > 0) {
            sp_ricci_calibrate_end(impl_->ricci);
            std::fprintf(stderr, "[sp-engine] Ricci sentinel calibrated: "
                         "p3_energy=%.6f threshold=%.4f\n",
                         impl_->ricci->p3_energy_calibrated,
                         impl_->ricci->metric_criticality);
        }
#ifdef SP_ENGINE_WITH_CUDA
        // GPU-domain calibration produces a correct variance-ranked
        // order (K_corr parity with CPU on synthetic data: 0.9925
        // mean, 0.9804 min). BUT on real Qwen3-8B decode chains the
        // calibrated order regresses PPL (+0.21 vs static Möbius),
        // while on the CPU cache path the same calibration improves
        // PPL (-0.25). The asymmetry is likely an interaction between
        // variance-ranked band assignments and fp16 scale handling in
        // the CUDA band_quantize; not yet root-caused. Default OFF to
        // keep the best-measured GPU PPL; enable via
        // SHANNON_PRIME_SYNC_CALIB_TO_GPU=1 when investigating.
        const char* env_sync = std::getenv("SHANNON_PRIME_SYNC_CALIB_TO_GPU");
        const bool sync_enabled = env_sync && std::atoi(env_sync) != 0;
        if (sync_enabled && impl_->cuda_inited && impl_->shadow_inited &&
            impl_->shadow.use_var_reorder && impl_->shadow.var_order &&
            impl_->cuda_cache.d_mobius_order) {
            const int hd = impl_->head_dim;
            cudaStream_t s = (cudaStream_t)impl_->cuda_cache.stream;
            cudaError_t err = cudaMemcpyAsync(impl_->cuda_cache.d_mobius_order,
                                               impl_->shadow.var_order,
                                               (size_t)hd * sizeof(int),
                                               cudaMemcpyHostToDevice, s);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "[sp-engine] calibrate_end: "
                             "failed to sync var_order to GPU: %s\n",
                             cudaGetErrorString(err));
                return false;
            }
            cudaStreamSynchronize(s);
            std::fprintf(stderr, "[sp-engine:diag] calibrate_end: synced "
                         "%d-entry variance-ranked (GPU-domain) order to "
                         "GPU d_mobius_order\n", hd);
        }

        // Sqfree calibration sync to GPU.
        //
        // sp_sqfree_calibrate_end rebuilds impl_->sq.mask with a new
        // variance-ranked Knight mask (new skeleton_idx, residual_idx,
        // csr_offsets, csr_skel_slot, csr_mu_sign — and potentially a
        // different n_terms). The GPU-resident sqfree cache's mask arrays
        // were uploaded at init with the static squarefree-first ordering
        // and are now out of date. Re-upload from impl_->sq.mask.
        //
        // Sqfree calibration feeds raw K vectors through the CPU pad +
        // Vilenkin pipeline to accumulate variance. Since the sqfree pad
        // + staged Vilenkin transform is computed on the pad_dim basis
        // (not the p=2 butterfly), CPU and GPU versions of this particular
        // transform are closer to bit-identical than the hd=128 Hadamard
        // butterfly was. No asymmetric-calibration regression expected here.
        if (impl_->cuda_sqfree_inited && impl_->sq_inited) {
            cudaStream_t ss = (cudaStream_t)impl_->cuda_sqfree_cache.stream;
            const sp_knight_mask_t* m = &impl_->sq.mask;

            // sk_k / n_res assumed stable (always pad_dim/2 / pad_dim-sk_k).
            // n_terms may shift — handle by reallocating d_csr_* when it does.
            if (m->n_terms != impl_->cuda_sqfree_cache.n_terms) {
                if (impl_->cuda_sqfree_cache.d_csr_skel_slot)
                    cudaFree(impl_->cuda_sqfree_cache.d_csr_skel_slot);
                if (impl_->cuda_sqfree_cache.d_csr_mu_sign)
                    cudaFree(impl_->cuda_sqfree_cache.d_csr_mu_sign);
                cudaMalloc((void**)&impl_->cuda_sqfree_cache.d_csr_skel_slot,
                           (size_t)m->n_terms * sizeof(int));
                cudaMalloc((void**)&impl_->cuda_sqfree_cache.d_csr_mu_sign,
                           (size_t)m->n_terms * sizeof(int));
                impl_->cuda_sqfree_cache.n_terms = m->n_terms;
            }

            cudaMemcpyAsync(impl_->cuda_sqfree_cache.d_skeleton_idx,
                            m->skeleton_idx,
                            (size_t)m->sk_k * sizeof(int),
                            cudaMemcpyHostToDevice, ss);
            cudaMemcpyAsync(impl_->cuda_sqfree_cache.d_residual_idx,
                            m->residual_idx,
                            (size_t)m->n_res * sizeof(int),
                            cudaMemcpyHostToDevice, ss);
            cudaMemcpyAsync(impl_->cuda_sqfree_cache.d_csr_offsets,
                            m->csr_offsets,
                            (size_t)(m->n_res + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, ss);
            cudaMemcpyAsync(impl_->cuda_sqfree_cache.d_csr_skel_slot,
                            m->csr_skel_slot,
                            (size_t)m->n_terms * sizeof(int),
                            cudaMemcpyHostToDevice, ss);
            // Host stores mu_sign as int8_t; GPU kernel reads int32 via
            // the kernel_mobius_predict signature. Convert on upload.
            {
                std::vector<int> signs_i32((size_t)m->n_terms);
                for (int i = 0; i < m->n_terms; ++i) {
                    signs_i32[(size_t)i] = (int)m->csr_mu_sign[i];
                }
                cudaMemcpyAsync(impl_->cuda_sqfree_cache.d_csr_mu_sign,
                                signs_i32.data(),
                                (size_t)m->n_terms * sizeof(int),
                                cudaMemcpyHostToDevice, ss);
                cudaStreamSynchronize(ss);
            }

            std::fprintf(stderr,
                "[sp-engine:diag] calibrate_end: synced sqfree Knight mask "
                "to GPU (sk_k=%d n_res=%d n_terms=%d)\n",
                m->sk_k, m->n_res, m->n_terms);
        }
#endif
        return true;
    }
    return false;
}

bool KvCache::is_calibrated() const {
    return impl_->calibrated;
}

bool KvCache::is_hierarchical() const {
    return impl_->hier_inited;
}

bool KvCache::is_gpu() const {
#ifdef SP_ENGINE_WITH_CUDA
    return impl_->cuda_inited || impl_->cuda_sqfree_inited;
#else
    return false;
#endif
}

// ── GPU-resident ship cache (step 3) ────────────────────────────────

std::unique_ptr<KvCache> KvCache::create_gpu(int n_layer, int n_head_kv,
                                              int head_dim, int max_seq,
                                              const Config& cfg,
                                              void* stream) {
#ifndef SP_ENGINE_WITH_CUDA
    (void)n_layer; (void)n_head_kv; (void)head_dim; (void)max_seq;
    (void)cfg; (void)stream;
    std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: engine built without "
                         "SP_ENGINE_WITH_CUDA\n");
    return nullptr;
#else
    if (cfg.hierarchical) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: hierarchical "
                             "path is not yet GPU-resident (still CPU)\n");
        return nullptr;
    }
    // sqfree MVP: GPU-resident compress/decompress, no spinor yet.
    const bool is_sqfree = cfg.sqfree;
    if (cfg.spinor) {
        // spinor implies sqfree. Full-scope, not MVP — fall back so caller
        // can retry with the host cache path.
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: sqfree+spinor "
                             "not yet GPU-resident (MVP lands sqfree only; "
                             "spinor is the next-session follow-up)\n");
        return nullptr;
    }
    if (n_layer <= 0 || n_head_kv <= 0 || head_dim <= 0 || max_seq <= 0) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: bad dims\n");
        return nullptr;
    }

    auto kv = std::unique_ptr<KvCache>(new KvCache());
    kv->impl_->n_layer   = n_layer;
    kv->impl_->n_head_kv = n_head_kv;
    kv->impl_->head_dim  = head_dim;
    kv->impl_->pad_dim   = head_dim;
    kv->impl_->max_seq   = max_seq;

    sp_config_t* sc = &kv->impl_->cfg;
    sp_config_init(sc, head_dim, n_layer, n_head_kv);
    sc->use_mobius_mask = cfg.mobius;

    auto kbits = parse_csv_bits(cfg.k_bits_csv);
    auto vbits = parse_csv_bits(cfg.v_bits_csv);
    if (kbits.empty()) kbits = {5, 5, 4, 3};
    if (vbits.empty()) vbits = {3};
    if ((int)kbits.size() > SP_MAX_BANDS) kbits.resize(SP_MAX_BANDS);
    if ((int)vbits.size() > SP_MAX_BANDS) vbits.resize(SP_MAX_BANDS);
    sc->k_n_bands = (int)kbits.size();
    sc->v_n_bands = (int)vbits.size();
    for (size_t i = 0; i < kbits.size(); ++i) sc->k_band_bits[i] = kbits[i];
    for (size_t i = 0; i < vbits.size(); ++i) sc->v_band_bits[i] = vbits[i];

    // Sqfree GPU cache dispatches to a different backing. Ship falls
    // through to sp_cuda_cache_init below.
    if (is_sqfree) {
        const int rbits = (cfg.residual_bits >= 1 && cfg.residual_bits <= 4)
                          ? cfg.residual_bits : 3;
        if (sp_cuda_sqfree_cache_init(&kv->impl_->cuda_sqfree_cache, sc,
                                       max_seq, rbits, /*use_spinor=*/0,
                                       stream) != 0) {
            std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: "
                         "sp_cuda_sqfree_cache_init failed\n");
            return nullptr;
        }
        kv->impl_->cuda_sqfree_inited = true;
        kv->impl_->sqfree             = true;
        kv->impl_->pad_dim            = kv->impl_->cuda_sqfree_cache.pad_dim;

        // Companion CPU sqfree cache for calibration accounting.
        // sp_sqfree_calibrate_end reallocates sc->k_cache[s] and
        // sc->v_cache[s] after rebuilding the Knight mask, so we MUST
        // leave these pointers intact (not null them) — otherwise the
        // rebuild loop indexes through a NULL slot array. The overhead
        // is a few MB of unused host per-slot buffers; acceptable until
        // we have a proper "calibration-only" shadow mode.
        if (sp_sqfree_cache_init(&kv->impl_->sq, sc, max_seq, rbits,
                                  /*use_spinor=*/false) != 0) {
            std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: companion "
                         "sqfree_cache_init failed (sqfree calibration will "
                         "be a no-op)\n");
        } else {
            kv->impl_->sq_inited = true;
        }
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: sqfree path, "
                     "hd=%d pad_dim=%d sk_k=%d n_res=%d res_bits=%d\n",
                     head_dim, kv->impl_->cuda_sqfree_cache.pad_dim,
                     kv->impl_->cuda_sqfree_cache.sk_k,
                     kv->impl_->cuda_sqfree_cache.n_res, rbits);
        return kv;
    }

    if (sp_cuda_cache_init(&kv->impl_->cuda_cache, sc, max_seq, stream) != 0) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: sp_cuda_cache_init failed\n");
        return nullptr;
    }
    kv->impl_->cuda_inited = true;

    // Also init a lightweight shadow cache to handle calibration. The
    // GPU cache kernels read from a single int[hd] d_mobius_order table;
    // post-calibration we copy the shadow's variance-ranked var_order
    // into that GPU buffer so the GPU cache reorder matches the CPU's
    // calibrated behavior. sp_shadow_cache_init only allocates the
    // per-cache scratch buffers — no per-slot storage — so the cost is
    // a few head_dim-sized float arrays.
    if (sp_shadow_cache_init(&kv->impl_->shadow, sc) != 0) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: companion "
                     "shadow_cache_init failed (calibration will be a no-op)\n");
    } else {
        kv->impl_->shadow_inited = true;
        // Shadow's k_cache/v_cache slot pointers are not needed — we route
        // writes/reads to the CUDA cache. Ensure they're nullptr so the
        // destructor's free() loops are safe.
        kv->impl_->shadow.k_cache = nullptr;
        kv->impl_->shadow.v_cache = nullptr;
    }

    // Pre-allocate staging and scratch buffers up front so the write/read
    // hot paths don't call cudaMalloc per step.
    kv->impl_->d_stage_bytes = (size_t)head_dim * sizeof(float);
    if (cudaMalloc((void**)&kv->impl_->d_stage,
                   kv->impl_->d_stage_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: d_stage cudaMalloc failed\n");
        return nullptr;
    }
    kv->impl_->d_scratch_batch_bytes = (size_t)head_dim * max_seq * sizeof(float);
    if (cudaMalloc((void**)&kv->impl_->d_scratch_batch,
                   kv->impl_->d_scratch_batch_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: scratch_batch cudaMalloc failed\n");
        return nullptr;
    }

    std::fprintf(stderr, "[sp-engine] KvCache::create_gpu: ship path, "
                 "hd=%d n_layer=%d n_head_kv=%d max_seq=%d k_bytes=%d v_bytes=%d\n",
                 head_dim, n_layer, n_head_kv, max_seq,
                 kv->impl_->cuda_cache.k_bands.total_bytes,
                 kv->impl_->cuda_cache.v_bands.total_bytes);
    return kv;
#endif
}

#ifdef SP_ENGINE_WITH_CUDA
// write_gpu: d_K_flat / d_V_flat are in [hd, n_kv, n_tokens] layout.
// For each (token, head) pair, the vector at offset q*n_kv*hd + h*hd is
// the raw K / V for that slot. We point sp_cuda_write_k/v at those
// offsets in place — no staging copy needed.
bool KvCache::write_gpu(int layer, int pos_offset, int n_tokens,
                        const float* d_K_flat, const float* d_V_flat) {
    if (!impl_->cuda_inited && !impl_->cuda_sqfree_inited) {
        std::fprintf(stderr, "[sp-engine] write_gpu on non-GPU cache\n");
        return false;
    }
    if (layer < 0 || layer >= impl_->n_layer) return false;
    if (pos_offset < 0 || pos_offset + n_tokens > impl_->max_seq) return false;

    const int H  = impl_->n_head_kv;
    const int hd = impl_->head_dim;
    if (impl_->cuda_sqfree_inited) {
        for (int q = 0; q < n_tokens; ++q) {
            const int pos = pos_offset + q;
            for (int h = 0; h < H; ++h) {
                const float* d_k_vec = d_K_flat + (size_t)(q * H + h) * hd;
                const float* d_v_vec = d_V_flat + (size_t)(q * H + h) * hd;
                sp_cuda_sqfree_write_k(&impl_->cuda_sqfree_cache, layer, h, pos, d_k_vec);
                sp_cuda_sqfree_write_v(&impl_->cuda_sqfree_cache, layer, h, pos, d_v_vec);
            }
        }
        return true;
    }
    for (int q = 0; q < n_tokens; ++q) {
        const int pos = pos_offset + q;
        for (int h = 0; h < H; ++h) {
            const float* d_k_vec = d_K_flat + (size_t)(q * H + h) * hd;
            const float* d_v_vec = d_V_flat + (size_t)(q * H + h) * hd;
            sp_cuda_write_k(&impl_->cuda_cache, layer, h, pos, d_k_vec);
            sp_cuda_write_v(&impl_->cuda_cache, layer, h, pos, d_v_vec);
        }
    }
    return true;
}

// read_gpu: fills d_K_out / d_V_out with shape [hd, n_kv, kv_len].
// CUDA's batched read naturally outputs [hd, n_pos] contiguous per
// (layer, head). To reach the strided layout expected by the attention
// graph, we first read each head into a scratch `[hd, kv_len]` buffer,
// then cudaMemcpy2D with dst_pitch = n_kv*hd*4 to scatter it into the
// right slot. 2*H calls per layer, 2*H per decode step — all launches
// async on the cache stream, so latency is the per-layer kernel time.
bool KvCache::read_gpu(int layer, int kv_len,
                       float* d_K_out, float* d_V_out) const {
    if (!impl_->cuda_inited && !impl_->cuda_sqfree_inited) {
        std::fprintf(stderr, "[sp-engine] read_gpu on non-GPU cache\n");
        return false;
    }
    if (layer < 0 || layer >= impl_->n_layer) return false;
    if (kv_len < 0 || kv_len > impl_->max_seq) return false;
    if (kv_len == 0) return true;

    const int H  = impl_->n_head_kv;
    const int hd = impl_->head_dim;

    // Sqfree GPU cache: batched read processes all kv_len positions
    // per (layer, head) in one kernel-dispatch series (~9 launches vs
    // 9*kv_len for the per-vec path). Output from the batch call is
    // [n_pos * head_dim] vec-major; we then 2D-memcpy into the
    // caller's expected strided [hd, n_kv, kv_len] layout (same
    // pattern as the ship-path scatter).
    if (impl_->cuda_sqfree_inited) {
        cudaStream_t s = (cudaStream_t)impl_->cuda_sqfree_cache.stream;
        const size_t row_bytes = (size_t)hd * sizeof(float);
        const size_t dst_pitch = (size_t)H * hd * sizeof(float);

        // We need a temporary that's [kv_len * hd] fp32. Reuse nothing
        // is simplest — allocate once on the cache's stream. For Qwen3
        // at ctx=1024 that's 1024*128*4 = 512 KB.
        float* d_tmp = nullptr;
        if (cudaMalloc((void**)&d_tmp,
                       (size_t)kv_len * hd * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "[sp-engine] read_gpu(sqfree): tmp alloc failed\n");
            return false;
        }
        for (int h = 0; h < H; ++h) {
            // K
            sp_cuda_sqfree_read_k_batch(&impl_->cuda_sqfree_cache,
                                         layer, h, 0, kv_len, d_tmp);
            if (cudaMemcpy2DAsync(d_K_out + (size_t)h * hd, dst_pitch,
                                   d_tmp, row_bytes,
                                   row_bytes, (size_t)kv_len,
                                   cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
                std::fprintf(stderr, "[sp-engine] read_gpu(sqfree): K scatter failed (L=%d h=%d)\n", layer, h);
                cudaFree(d_tmp);
                return false;
            }
            // V
            sp_cuda_sqfree_read_v_batch(&impl_->cuda_sqfree_cache,
                                         layer, h, 0, kv_len, d_tmp);
            if (cudaMemcpy2DAsync(d_V_out + (size_t)h * hd, dst_pitch,
                                   d_tmp, row_bytes,
                                   row_bytes, (size_t)kv_len,
                                   cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
                std::fprintf(stderr, "[sp-engine] read_gpu(sqfree): V scatter failed (L=%d h=%d)\n", layer, h);
                cudaFree(d_tmp);
                return false;
            }
        }
        cudaStreamSynchronize(s);
        cudaFree(d_tmp);
        return true;
    }

    const size_t row_bytes = (size_t)hd * sizeof(float);
    const size_t dst_pitch = (size_t)H * hd * sizeof(float);
    cudaStream_t s = (cudaStream_t)impl_->cuda_cache.stream;
    float* scratch = impl_->d_scratch_batch;

    for (int h = 0; h < H; ++h) {
        // K ── dequantize → unreorder → VHT2 into scratch[hd, kv_len]
        sp_cuda_read_k_batch(&impl_->cuda_cache, layer, h, 0, kv_len, scratch);
        // Scatter to d_K_out[:, h, :] — each of kv_len "rows" of hd floats
        // lands at offset q*n_kv*hd + h*hd, so dst_pitch = n_kv*hd*4.
        if (cudaMemcpy2DAsync(d_K_out + (size_t)h * hd, dst_pitch,
                               scratch, row_bytes,
                               row_bytes, (size_t)kv_len,
                               cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
            std::fprintf(stderr, "[sp-engine] read_gpu: K scatter failed (L=%d h=%d)\n", layer, h);
            return false;
        }
        // V ── same pipeline, minus Möbius unreorder (handled internally).
        sp_cuda_read_v_batch(&impl_->cuda_cache, layer, h, 0, kv_len, scratch);
        if (cudaMemcpy2DAsync(d_V_out + (size_t)h * hd, dst_pitch,
                               scratch, row_bytes,
                               row_bytes, (size_t)kv_len,
                               cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
            std::fprintf(stderr, "[sp-engine] read_gpu: V scatter failed (L=%d h=%d)\n", layer, h);
            return false;
        }
    }
    return true;
}
#else
bool KvCache::write_gpu(int, int, int, const float*, const float*) {
    std::fprintf(stderr, "[sp-engine] write_gpu: built without SP_ENGINE_WITH_CUDA\n");
    return false;
}
bool KvCache::read_gpu(int, int, float*, float*) const {
    std::fprintf(stderr, "[sp-engine] read_gpu: built without SP_ENGINE_WITH_CUDA\n");
    return false;
}
#endif

// ── Cauchy reset system (decode-chain causal stability) ─────────────

bool KvCache::init_cauchy(int mode, int fixed_n, float params_b, bool use_ricci) {
    if (mode < 0 || mode > 2) return false;
    if (mode == 0) return true;  // off is a legal no-op

    // Mode 2 (dynamic): Mertens schedule is always allocated; Ricci is
    // opt-in (default off — empirically contributes 0 incremental PPL on
    // Qwen3-8B-Q8 ctx=1024 ablations).
    // Mode 1 (fixed-N) needs neither — just a counter.
    if (mode == 2) {
        if (use_ricci) {
            impl_->ricci = new sp_ricci_sentinel_t{};
            // Ricci monitors the p=3 band of whichever compress path is
            // active. For hier use the skeleton's band config; for sqfree
            // use the skeleton K bands; for ship use the full K bands.
            const sp_band_config_t* bc = nullptr;
            if (impl_->hier_inited) {
                bc = &impl_->hier.predictors[0].skel_bands;
            } else if (impl_->sqfree) {
                bc = &impl_->sq.k_bands;
            } else {
                bc = &impl_->shadow.k_bands;
            }
            if (sp_ricci_init(impl_->ricci, bc, params_b) != 0) {
                delete impl_->ricci;
                impl_->ricci = nullptr;
                std::fprintf(stderr, "[sp-engine] Cauchy: ricci_init failed\n");
                return false;
            }
        }

        impl_->mertens = new sp_mertens_oracle_t{};
        if (sp_mertens_init(impl_->mertens, impl_->max_seq) != 0) {
            delete impl_->mertens;
            impl_->mertens = nullptr;
            std::fprintf(stderr, "[sp-engine] Cauchy: mertens_init failed "
                         "(mode 2 degrades to fixed-N fallback)\n");
        }
    }

    sp_cauchy_init(&impl_->cauchy, mode, fixed_n, impl_->ricci, impl_->mertens);
    impl_->cauchy_inited = true;

    // Pre-size the VHT2 scratch used by the ricci-feed path in write().
    if (impl_->ricci) {
        impl_->vht2_scratch.assign((size_t)impl_->head_dim, 0.0f);
    }

    std::fprintf(stderr, "[sp-engine] Cauchy system initialized: mode=%d "
                 "fixed_n=%d ricci=%s mertens=%s params_b=%.2f\n",
                 mode, fixed_n,
                 impl_->ricci ? "yes" : "no",
                 impl_->mertens ? "yes" : "no",
                 (double)params_b);
    return true;
}

int KvCache::cauchy_check(int pos) {
    if (!impl_->cauchy_inited) return 0;
    return sp_cauchy_check(&impl_->cauchy, pos);
}

void KvCache::cauchy_set_cooldown(int n) {
    if (!impl_->cauchy_inited) return;
    if (n < 1) n = 1;
    impl_->cauchy.partial_window = n;
}

void KvCache::cauchy_disable_mertens() {
    if (!impl_->cauchy_inited) return;
    if (impl_->mertens) {
        delete impl_->mertens;
        impl_->mertens = nullptr;
        impl_->cauchy.mertens = nullptr;
        std::fprintf(stderr, "[sp-engine] Cauchy: Mertens oracle disabled "
                     "(Ricci-only ablation)\n");
    }
}

void KvCache::cauchy_disable_ricci() {
    if (!impl_->cauchy_inited) return;
    if (impl_->ricci) {
        delete impl_->ricci;
        impl_->ricci = nullptr;
        impl_->cauchy.ricci = nullptr;
        std::fprintf(stderr, "[sp-engine] Cauchy: Ricci sentinel disabled "
                     "(Mertens-only ablation)\n");
    }
}

void KvCache::ricci_feed(const float* vht2_coeffs, int hd) {
    if (!impl_->ricci) return;
    sp_ricci_check(impl_->ricci, vht2_coeffs, hd);
}

void KvCache::cauchy_record_reset(int pos) {
    if (!impl_->cauchy_inited) return;
    sp_cauchy_record_reset(&impl_->cauchy, pos);
}

double KvCache::ricci_drift() const {
    if (!impl_->ricci) return 0.0;
    return sp_ricci_drift(impl_->ricci);
}

void KvCache::cauchy_print_stats() const {
    if (!impl_->cauchy_inited) return;
    sp_cauchy_print_stats(&impl_->cauchy);
    if (impl_->ricci) {
        std::fprintf(stderr, "[sp-engine] Ricci drift: %.6f  "
                     "threshold: %.4f  samples: %d\n",
                     sp_ricci_drift(impl_->ricci),
                     impl_->ricci->metric_criticality,
                     impl_->ricci->n_samples);
    }
}

} // namespace sp::engine
