/* sp_friedman_kv_hook.cpp — Phase 4 engine-side KV hook.
 *
 * See sp_friedman_kv_hook.h for the API and integration contract.
 * Pure C++17, no SIMD, no global state.  Each per-(layer, head) cache
 * is independent so callers can append-in-parallel across heads.
 */

#include "sp_friedman_kv_hook.h"

#include <cstdlib>
#include <cstring>

/* ---------- Helpers ---------------------------------------------------- */

static inline int idx_lh(int layer, int head, int n_heads)
{
    return layer * n_heads + head;
}

static int alloc_caches(sp_friedman_kv_hook_t *hook)
{
    int total = hook->n_layers * hook->n_heads;
    hook->caches = (sp_friedman_cache_t *)std::calloc(
        (size_t)total, sizeof(sp_friedman_cache_t));
    if (!hook->caches) return 0;
    for (int i = 0; i < total; ++i) {
        if (!sp_friedman_cache_init(&hook->caches[i], hook->capacity)) {
            for (int j = 0; j < i; ++j) {
                sp_friedman_cache_destroy(&hook->caches[j]);
            }
            std::free(hook->caches);
            hook->caches = nullptr;
            return 0;
        }
    }
    return 1;
}

static void free_caches(sp_friedman_kv_hook_t *hook)
{
    if (!hook->caches) return;
    int total = hook->n_layers * hook->n_heads;
    for (int i = 0; i < total; ++i) {
        sp_friedman_cache_destroy(&hook->caches[i]);
    }
    std::free(hook->caches);
    hook->caches = nullptr;
}

/* ---------- Ctor / dtor ----------------------------------------------- */

int sp_friedman_kv_hook_init(sp_friedman_kv_hook_t *hook,
                             sp_friedman_mode       mode,
                             int                    n_layers,
                             int                    n_heads,
                             int                    head_dim,
                             int                    capacity)
{
    if (!hook || n_layers <= 0 || n_heads <= 0 || capacity <= 0) return 0;
    std::memset(hook, 0, sizeof(*hook));
    hook->mode      = mode;
    hook->n_layers  = n_layers;
    hook->n_heads   = n_heads;
    hook->head_dim  = head_dim;
    hook->capacity  = capacity;

    if (mode == SP_FRIEDMAN_MODE_OFF) {
        hook->initialized = 1;
        return 1;
    }

    if (!sp_kste_ctx_init(&hook->encoder_ctx, head_dim)) return 0;

    hook->scratch = (float *)std::calloc((size_t)(3 * head_dim), sizeof(float));
    if (!hook->scratch) {
        sp_kste_ctx_destroy(&hook->encoder_ctx);
        return 0;
    }

    if (!alloc_caches(hook)) {
        std::free(hook->scratch);
        hook->scratch = nullptr;
        sp_kste_ctx_destroy(&hook->encoder_ctx);
        return 0;
    }

    hook->initialized = 1;
    return 1;
}

void sp_friedman_kv_hook_destroy(sp_friedman_kv_hook_t *hook)
{
    if (!hook || !hook->initialized) return;
    free_caches(hook);
    if (hook->scratch) {
        std::free(hook->scratch);
        hook->scratch = nullptr;
    }
    if (hook->mode != SP_FRIEDMAN_MODE_OFF) {
        sp_kste_ctx_destroy(&hook->encoder_ctx);
    }
    hook->initialized = 0;
}

int sp_friedman_kv_hook_set_mode(sp_friedman_kv_hook_t *hook,
                                 sp_friedman_mode       mode)
{
    if (!hook || !hook->initialized) return 0;
    if (mode == hook->mode) return 1;

    /* OFF -> active: lazily allocate. */
    if (hook->mode == SP_FRIEDMAN_MODE_OFF &&
        mode      != SP_FRIEDMAN_MODE_OFF) {
        if (!sp_kste_ctx_init(&hook->encoder_ctx, hook->head_dim)) return 0;
        hook->scratch = (float *)std::calloc(
            (size_t)(3 * hook->head_dim), sizeof(float));
        if (!hook->scratch) {
            sp_kste_ctx_destroy(&hook->encoder_ctx);
            return 0;
        }
        if (!alloc_caches(hook)) {
            std::free(hook->scratch);
            hook->scratch = nullptr;
            sp_kste_ctx_destroy(&hook->encoder_ctx);
            return 0;
        }
        hook->mode = mode;
        return 1;
    }

    /* active -> OFF: free. */
    if (hook->mode != SP_FRIEDMAN_MODE_OFF &&
        mode      == SP_FRIEDMAN_MODE_OFF) {
        free_caches(hook);
        if (hook->scratch) { std::free(hook->scratch); hook->scratch = nullptr; }
        sp_kste_ctx_destroy(&hook->encoder_ctx);
        hook->mode = mode;
        return 1;
    }

    /* OBSERVER <-> POLICY: same allocation, just flip the flag. */
    hook->mode = mode;
    return 1;
}

/* ---------- The hot path ---------------------------------------------- */

sp_friedman_decision sp_friedman_kv_hook_observe(
    sp_friedman_kv_hook_t *hook,
    int                    layer,
    int                    head,
    const float           *K,
    int32_t                pos)
{
    if (!hook || !hook->initialized || hook->mode == SP_FRIEDMAN_MODE_OFF
        || !K) {
        return SP_FRIEDMAN_ADMITTED;
    }
    if (layer < 0 || layer >= hook->n_layers ||
        head  < 0 || head  >= hook->n_heads) {
        return SP_FRIEDMAN_ADMITTED;
    }

    sp_kste_tree T;
    float        skel_var = 0.0f;
    if (!sp_kste_encode_ex(&T, K, &hook->encoder_ctx, hook->scratch,
                           &skel_var)) {
        return SP_FRIEDMAN_ADMITTED;
    }

    int i = idx_lh(layer, head, hook->n_heads);
    return sp_friedman_cache_insert(&hook->caches[i], &T, skel_var, pos);
}

/* ---------- Stats aggregator ----------------------------------------- */

void sp_friedman_kv_hook_stats(const sp_friedman_kv_hook_t *hook,
                               uint64_t *inserts_total,
                               uint64_t *evictions,
                               uint64_t *admissions,
                               uint64_t *replacements,
                               uint64_t *full_embeds)
{
    uint64_t a_ins = 0, a_evt = 0, a_adm = 0, a_rep = 0, a_emb = 0;
    if (hook && hook->initialized && hook->caches) {
        int total = hook->n_layers * hook->n_heads;
        for (int i = 0; i < total; ++i) {
            a_ins += hook->caches[i].inserts_total;
            a_evt += hook->caches[i].evictions;
            a_adm += hook->caches[i].admissions;
            a_rep += hook->caches[i].replacements;
            a_emb += hook->caches[i].full_embeds;
        }
    }
    if (inserts_total) *inserts_total = a_ins;
    if (evictions)     *evictions     = a_evt;
    if (admissions)    *admissions    = a_adm;
    if (replacements)  *replacements  = a_rep;
    if (full_embeds)   *full_embeds   = a_emb;
}
