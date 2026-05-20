/* sp_friedman_kv_hook.h — Phase 4: engine-side KV-write hook.
 *
 * Wraps sp_friedman_cache_t into a per-layer-per-head structure with
 * the lifecycle the engine forward pass expects: init at model load,
 * observe/policy at each KV append, destroy at engine shutdown.
 *
 * Modes:
 *   OBSERVER — the cache runs, records decisions, but never gates
 *              the underlying KV write.  Used for measurement runs
 *              (Phase 4 T2.2 — eviction rate vs WikiText-103) without
 *              risking PPL drift.
 *   POLICY   — the cache's decision becomes the engine's decision:
 *              SP_FRIEDMAN_EVICTED  -> skip the KV write, attention
 *                                      later sees fewer cache slots.
 *              ADMITTED / REPLACED  -> normal KV write proceeds.
 *
 * Integration point (engine forward pass, after K_new is finalized
 * for a token in fp32 and just before sp_ok_kv_cache_append_layer):
 *
 *   for each layer L, each KV head h:
 *       const float *K_fp32 = K_new[L][h];   // head_dim floats
 *       sp_friedman_decision d =
 *           sp_friedman_kv_hook_observe(&hook, L, h, K_fp32, pos);
 *       if (mode == POLICY && d == SP_FRIEDMAN_EVICTED) {
 *           continue;   // skip K/V write for this token at this head
 *       }
 *       // ... existing append path ...
 *
 * Currently *not* wired into sp_forward.cpp — the encoder needs the
 * Phase-4 resolution remediation (probe T4_RES_PROBE found embed rate
 * ≈ 0 on N(0,I) clusters; see SESSION-STATE-friedman-4.md) before
 * the hook can be enabled in policy mode.
 *
 * Copyright (C) 2026 Ray Daniels.  AGPLv3 / commercial.
 */

#ifndef SP_FRIEDMAN_KV_HOOK_H
#define SP_FRIEDMAN_KV_HOOK_H

#include "sp_friedman_cache.h"
#include "sp_kste.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SP_FRIEDMAN_MODE_OFF      = 0,   /* hook is a no-op; no allocation     */
    SP_FRIEDMAN_MODE_OBSERVER = 1,   /* runs; records; never gates writes  */
    SP_FRIEDMAN_MODE_POLICY   = 2    /* runs; decisions gate KV writes     */
} sp_friedman_mode;

/* Per-engine hook.  Owns one sp_friedman_cache_t per (layer, head). */
typedef struct {
    sp_friedman_mode      mode;
    int                   n_layers;
    int                   n_heads;        /* KV-heads count                */
    int                   head_dim;
    int                   capacity;       /* per-cache slot capacity       */

    sp_kste_ctx           encoder_ctx;    /* shared across all caches      */
    sp_friedman_cache_t  *caches;         /* size = n_layers * n_heads     */
    float                *scratch;        /* 3 * head_dim, shared          */

    int                   initialized;
} sp_friedman_kv_hook_t;

/* Construct: allocate per-(layer, head) caches at the given capacity.
 * Returns 1 on success, 0 on OOM or invalid args.  Off-mode is a no-op
 * construct that simply records the configuration and allocates nothing
 * — caller can switch to OBSERVER/POLICY later with hook_set_mode. */
int  sp_friedman_kv_hook_init(sp_friedman_kv_hook_t *hook,
                              sp_friedman_mode       mode,
                              int                    n_layers,
                              int                    n_heads,
                              int                    head_dim,
                              int                    capacity);

void sp_friedman_kv_hook_destroy(sp_friedman_kv_hook_t *hook);

/* Change mode at runtime.  When transitioning OFF -> {OBSERVER,POLICY},
 * caches are allocated lazily.  When transitioning to OFF, caches are
 * destroyed.  Returns 1 on success. */
int  sp_friedman_kv_hook_set_mode(sp_friedman_kv_hook_t *hook,
                                  sp_friedman_mode       mode);

/* Observe / decide on a single K vector.  Returns the sieve's decision
 * (SP_FRIEDMAN_EVICTED / ADMITTED / REPLACED).  In OBSERVER mode the
 * decision is informational only — caller proceeds with KV write
 * regardless.  In POLICY mode the caller MUST skip the KV write iff
 * the decision is SP_FRIEDMAN_EVICTED.  In OFF mode returns
 * SP_FRIEDMAN_ADMITTED unconditionally (no allocation, no work).
 *
 * `K` is head_dim floats — the post-RoPE K vector for this (layer, head,
 * pos).  Frobenius scale must be 1 (post-RMSNorm boundary); the
 * encoder is invariant to positive rescales but the variance fallback
 * key is not.
 *
 * Thread safety: each (layer, head) cache is independent; the caller
 * is responsible for serialising appends to the same (layer, head). */
sp_friedman_decision sp_friedman_kv_hook_observe(
    sp_friedman_kv_hook_t *hook,
    int                    layer,
    int                    head,
    const float           *K,
    int32_t                pos);

/* Aggregate counters across all per-(layer, head) caches.  Used by
 * the CLI to print T2.2-style eviction-rate summaries. */
void sp_friedman_kv_hook_stats(const sp_friedman_kv_hook_t *hook,
                               uint64_t *inserts_total,
                               uint64_t *evictions,
                               uint64_t *admissions,
                               uint64_t *replacements,
                               uint64_t *full_embeds);

#ifdef __cplusplus
}
#endif

#endif /* SP_FRIEDMAN_KV_HOOK_H */
