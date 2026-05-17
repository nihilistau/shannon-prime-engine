// Shannon-Prime Engine — long-term ARM memory bank (impl).
// Phase 13.B engine glue. See sp_lt_memory.h.

#include "sp_lt_memory.h"

#include <cstdio>
#include <cstring>

extern "C" {
#include "../lib/shannon-prime/core/sp_arm.h"
}

namespace sp::engine {

bool sp_lt_memory_init(sp_lt_memory& mem,
                        int n_layers, int n_kv_head, int head_dim,
                        double delta) {
    if (n_layers <= 0 || n_kv_head <= 0 || head_dim <= 0) return false;
    if (head_dim > SP_ARM_RING_N) {
        std::fprintf(stderr,
            "[sp_lt_memory] head_dim=%d exceeds SP_ARM_RING_N=%d — "
            "ring degree too small for this model\n",
            head_dim, (int)SP_ARM_RING_N);
        return false;
    }

    mem.n_layers  = n_layers;
    mem.n_kv_head = n_kv_head;
    mem.head_dim  = head_dim;
    mem.n_slabs   = n_layers * n_kv_head;

    const size_t slab_words = (size_t)mem.n_slabs * (size_t)SP_ARM_RING_N;
    mem.M_q1_storage.assign(slab_words, 0);
    mem.M_q2_storage.assign(slab_words, 0);

    sp_arm_bank_init(&mem.bank,
                      mem.M_q1_storage.data(),
                      mem.M_q2_storage.data(),
                      mem.n_slabs, head_dim, delta);

    /* Per-write scratch buffers reused across all write calls. */
    mem.k_decode_fp32.assign((size_t)head_dim,    0.0f);
    mem.v_decode_fp32.assign((size_t)head_dim,    0.0f);
    mem.write_scratch_4N.assign((size_t)4 * SP_ARM_RING_N, 0);
    mem.write_int_scratch.assign((size_t)SP_ARM_RING_N,    0);

    mem.total_writes    = 0;
    mem.total_evictions = 0;

    std::fprintf(stderr,
        "[sp_lt_memory] init: n_layers=%d n_kv_head=%d head_dim=%d "
        "n_slabs=%d  bank bytes=%.1f KB  delta=%.0f\n",
        n_layers, n_kv_head, head_dim, mem.n_slabs,
        (double)(slab_words * sizeof(uint64_t) * 2) / 1024.0,
        delta);
    return true;
}

bool sp_lt_memory_write_evict(sp_lt_memory& mem,
                                const sp_ok_kv_cache& cache) {
    if (mem.n_slabs == 0) return false;
    if (cache.n_layers != mem.n_layers) {
        std::fprintf(stderr,
            "[sp_lt_memory] write_evict: cache.n_layers=%d but "
            "mem.n_layers=%d\n",
            cache.n_layers, mem.n_layers);
        return false;
    }
    if (cache.cur_len <= 0) return true;   /* nothing to evict */
    if (cache.n_kv_head != mem.n_kv_head) {
        std::fprintf(stderr,
            "[sp_lt_memory] write_evict: cache.n_kv_head=%d but "
            "mem.n_kv_head=%d\n",
            cache.n_kv_head, mem.n_kv_head);
        return false;
    }
    if (cache.head_dim != mem.head_dim) {
        std::fprintf(stderr,
            "[sp_lt_memory] write_evict: cache.head_dim=%d but "
            "mem.head_dim=%d\n",
            cache.head_dim, mem.head_dim);
        return false;
    }

    const int n_layers  = mem.n_layers;
    const int n_kv_head = mem.n_kv_head;
    const int head_dim  = mem.head_dim;
    const int n_tokens  = cache.cur_len;

    for (int L = 0; L < n_layers; ++L) {
        const sp_ok_tensor& K_layer = cache.layers[(size_t)L].K;
        const sp_ok_tensor& V_layer = cache.layers[(size_t)L].V;

        const double k_div =
            (double)K_layer.scale_recip * (double)K_layer.frobenius_scale;
        const double v_div =
            (double)V_layer.scale_recip * (double)V_layer.frobenius_scale;
        if (k_div == 0.0 || v_div == 0.0) {
            std::fprintf(stderr,
                "[sp_lt_memory] L%d: zero divisor (K_div=%g V_div=%g) "
                "— layer skipped\n",
                L, k_div, v_div);
            continue;
        }
        const int64_t T_stride = K_layer.shape[0];   /* = n_ctx */

        for (int kv_h = 0; kv_h < n_kv_head; ++kv_h) {
            const int slab = L * n_kv_head + kv_h;
            for (int t = 0; t < n_tokens; ++t) {
                /* Decode K[kv_h, :, t] and V[kv_h, :, t] from sp_ok_t
                 * to fp32. Cache layout: data[(kv_h*head_dim + d) *
                 * T_stride + t], col-major-by-t. */
                for (int d = 0; d < head_dim; ++d) {
                    const int64_t f = (int64_t)kv_h * head_dim + d;
                    const int64_t idx = f * T_stride + (int64_t)t;
                    mem.k_decode_fp32[(size_t)d] =
                        (float)((double)K_layer.data[idx].a / k_div);
                    mem.v_decode_fp32[(size_t)d] =
                        (float)((double)V_layer.data[idx].a / v_div);
                }
                sp_arm_bank_write(&mem.bank, slab,
                                    mem.k_decode_fp32.data(),
                                    mem.v_decode_fp32.data(),
                                    mem.write_scratch_4N.data(),
                                    mem.write_int_scratch.data());
                mem.total_writes += 1;
            }
        }
    }
    mem.total_evictions += 1;
    return true;
}

double sp_lt_memory_slab_norm(const sp_lt_memory& mem,
                               int layer, int kv_head) {
    if (layer < 0 || layer >= mem.n_layers) return 0.0;
    if (kv_head < 0 || kv_head >= mem.n_kv_head) return 0.0;
    const int slab = layer * mem.n_kv_head + kv_head;
    /* sp_arm_bank_norm needs 2*N uint64 scratch. We can reuse the
     * write_scratch_4N (only need half). The function is const-ish
     * in semantics (doesn't mutate the bank) but takes a non-const
     * scratch pointer; cast away const on the scratch member. */
    auto& mut_scratch = const_cast<sp_lt_memory&>(mem).write_scratch_4N;
    return sp_arm_bank_norm(&mem.bank, slab, mut_scratch.data());
}

}  // namespace sp::engine
