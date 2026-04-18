// Shannon-Prime Engine — compressed KV cache (impl)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.

#include "kv_cache.h"

extern "C" {
#include "shannon_prime.h"
}

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
    int                  n_layer    = 0;
    int                  n_head_kv  = 0;
    int                  head_dim   = 0;
    int                  pad_dim    = 0;   // = head_dim (ship) or sqfree pad (sqfree)
    int                  max_seq    = 0;

    // Exactly one of these is initialised at any time.
    sp_shadow_cache_t    shadow{};
    sp_sqfree_cache_t    sq{};

    bool                 shadow_inited = false;
    bool                 sq_inited     = false;

    sp_config_t          cfg{};

    // For shadow path: bytes per K/V vector (used for slot sizing).
    size_t               k_bytes = 0;
    size_t               v_bytes = 0;

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
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "%s cache: n_layer=%d n_head_kv=%d head_dim=%d "
                  "pad_dim=%d max_seq=%d compression=%.2fx",
                  impl_->sqfree ? "sqfree" : "shadow",
                  impl_->n_layer, impl_->n_head_kv, impl_->head_dim,
                  impl_->pad_dim, impl_->max_seq, compression_ratio());
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
            if (impl_->sqfree) {
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

    for (int q = 0; q < kv_len; ++q) {
        for (int h = 0; h < H; ++h) {
            float* k_vec = K_out.data() + (size_t)(q * H + h) * hd;
            float* v_vec = V_out.data() + (size_t)(q * H + h) * hd;
            if (impl_->sqfree) {
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

} // namespace sp::engine
