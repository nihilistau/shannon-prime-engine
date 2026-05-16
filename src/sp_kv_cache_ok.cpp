// Shannon-Prime Engine — O_K KV cache (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "sp_kv_cache_ok.h"

#include <cstring>

namespace sp::engine {

bool sp_ok_kv_cache_init(sp_ok_kv_cache& cache,
                          int             n_layers,
                          int             max_len,
                          int             n_kv_head,
                          int             head_dim,
                          int64_t         k_scale_recip,
                          int64_t         v_scale_recip,
                          int64_t         v_frobenius_scale,
                          sp_ok_arena&    arena) {
    if (n_layers <= 0 || max_len <= 0 || n_kv_head <= 0 || head_dim <= 0) {
        return false;
    }
    if (k_scale_recip <= 0 || v_scale_recip <= 0 || v_frobenius_scale == 0) {
        return false;
    }
    cache.n_layers  = n_layers;
    cache.max_len   = max_len;
    cache.n_kv_head = n_kv_head;
    cache.head_dim  = head_dim;
    cache.cur_len   = 0;
    cache.arena     = &arena;
    cache.layers.clear();
    cache.layers.resize(n_layers);

    const int64_t d_kv = (int64_t)n_kv_head * head_dim;
    int64_t shp[4] = { max_len, d_kv, 1, 1 };

    for (int L = 0; L < n_layers; ++L) {
        sp_ok_kv_layer& lyr = cache.layers[L];
        lyr.K.reset(2, shp);
        if (!arena.alloc_tensor(lyr.K)) return false;
        lyr.K.scale_recip     = k_scale_recip;
        lyr.K.frobenius_scale = 1;
        // Zero the storage so unused positions don't pollute attention.
        std::memset(lyr.K.data, 0, lyr.K.nbytes_contig());

        lyr.V.reset(2, shp);
        if (!arena.alloc_tensor(lyr.V)) return false;
        lyr.V.scale_recip     = v_scale_recip;
        lyr.V.frobenius_scale = v_frobenius_scale;
        std::memset(lyr.V.data, 0, lyr.V.nbytes_contig());
    }
    return true;
}

static bool kv_append_one(sp_ok_tensor&       dst,
                           const sp_ok_tensor& src,
                           int                 dst_offset_inner,
                           int                 n_new_tokens) {
    if (dst.data == nullptr || src.data == nullptr) return false;
    if (dst.n_dims < 2 || src.n_dims < 2) return false;
    const int64_t T_dst = dst.shape[0];
    const int64_t F_dst = dst.shape[1];
    const int64_t T_src = src.shape[0];
    const int64_t F_src = src.shape[1];
    if (F_dst != F_src) return false;
    if (T_src != (int64_t)n_new_tokens) return false;
    if ((int64_t)dst_offset_inner + n_new_tokens > T_dst) return false;
    if (dst.scale_recip != src.scale_recip) return false;
    if (dst.frobenius_scale != src.frobenius_scale) return false;

    // Copy per-feature row from src[feature * T_src + t] to
    //                          dst[feature * T_dst + (dst_offset_inner + t)]
    for (int64_t f = 0; f < F_dst; ++f) {
        sp_ok_t* d_row = dst.data + f * T_dst + dst_offset_inner;
        const sp_ok_t* s_row = src.data + f * T_src;
        std::memcpy(d_row, s_row, (size_t)n_new_tokens * sizeof(sp_ok_t));
    }
    return true;
}

bool sp_ok_kv_cache_append_layer(sp_ok_kv_cache&     cache,
                                  int                 layer_idx,
                                  const sp_ok_tensor& K_new,
                                  const sp_ok_tensor& V_new,
                                  int                 n_new_tokens) {
    if (layer_idx < 0 || layer_idx >= cache.n_layers) return false;
    if (n_new_tokens <= 0) return false;
    if (cache.cur_len + n_new_tokens > cache.max_len) return false;

    sp_ok_kv_layer& lyr = cache.layers[layer_idx];
    if (!kv_append_one(lyr.K, K_new, cache.cur_len, n_new_tokens)) return false;
    if (!kv_append_one(lyr.V, V_new, cache.cur_len, n_new_tokens)) return false;
    return true;
}

static sp_ok_tensor make_view(const sp_ok_tensor& src, int valid_len) {
    sp_ok_tensor v;
    v.data           = src.data;
    v.n_dims         = src.n_dims;
    v.shape[0]       = valid_len;
    v.shape[1]       = src.shape[1];
    v.shape[2]       = src.shape[2];
    v.shape[3]       = src.shape[3];
    // Strides: feature stride is src.shape[0] (not valid_len). Caller
    // accessing data[f * shape[0] + t] would mis-index; sp_attention reads
    // data[(feature) * T + t] where T = v.shape[0] = valid_len. To keep the
    // view consistent without restriding, we expose the FULL T_max as
    // shape[0] and let the caller scan only valid_len.
    //
    // BUT — to make the view "look like" a {valid_len, F} tensor for
    // sp_attention's k.shape[0] reading, we need to either:
    //   (a) pass valid_len in shape[0] and have attention use that
    //       (but the data layout still uses T_max as the stride!), or
    //   (b) physically compact the data (slow), or
    //   (c) carry an explicit stride.
    //
    // sp_attention reads k.data[(kv_h*head_dim + d) * T + t] with T = k.shape[0].
    // If we lie about shape[0] = valid_len while the actual stride is T_max,
    // the index math is wrong.
    //
    // Solution: set shape[0] = T_max (the actual stride) and let the caller
    // pass valid_len SEPARATELY. We expose a different accessor below.
    v.shape[0]       = src.shape[0];  // full max_len — this IS the stride
    v.strides[0]     = src.strides[0];
    v.strides[1]     = src.strides[1];
    v.strides[2]     = src.strides[2];
    v.strides[3]     = src.strides[3];
    v.scale_recip    = src.scale_recip;
    v.frobenius_scale = src.frobenius_scale;
    (void)valid_len;
    return v;
}

sp_ok_tensor sp_ok_kv_cache_view_k(const sp_ok_kv_cache& cache, int layer_idx) {
    if (layer_idx < 0 || layer_idx >= cache.n_layers) {
        return sp_ok_tensor{};
    }
    return make_view(cache.layers[layer_idx].K, cache.cur_len);
}

sp_ok_tensor sp_ok_kv_cache_view_v(const sp_ok_kv_cache& cache, int layer_idx) {
    if (layer_idx < 0 || layer_idx >= cache.n_layers) {
        return sp_ok_tensor{};
    }
    return make_view(cache.layers[layer_idx].V, cache.cur_len);
}

}  // namespace sp::engine
