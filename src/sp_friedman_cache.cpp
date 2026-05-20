/* sp_friedman_cache.cpp — Phase 3: KSTE-keyed sieve cache.
 *
 * See sp_friedman_cache.h for the API and invariants.  Pure CPU code,
 * no SIMD, no malloc on the hot path beyond the initial slot array.
 *
 * Performance budget: < 50 µs per token at capacity 4096 on CPU
 * (roadmap §3 exit criterion).  Phase 5 adds the label-multiset
 * pre-filter to amortise the O(N) embed loop; Phase 3 ships the
 * naive linear scan for correctness.
 */

#include "sp_friedman_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

/* ---------- Constructors ------------------------------------------------- */

int sp_friedman_cache_init(sp_friedman_cache_t *c, int capacity)
{
    if (!c || capacity <= 0) return 0;
    std::memset(c, 0, sizeof(*c));
    c->slots = (sp_friedman_slot_t *)std::calloc(
        (size_t)capacity, sizeof(sp_friedman_slot_t));
    if (!c->slots) return 0;
    c->capacity = capacity;
    return 1;
}

void sp_friedman_cache_destroy(sp_friedman_cache_t *c)
{
    if (!c) return;
    if (c->slots) {
        std::free(c->slots);
        c->slots = nullptr;
    }
    c->capacity = 0;
    c->count    = 0;
}

/* ---------- Insert ------------------------------------------------------- */

sp_friedman_decision sp_friedman_cache_insert(
    sp_friedman_cache_t *c,
    const sp_kste_tree  *tree,
    float                skel_var,
    int32_t              pos)
{
    if (!c || !c->slots || !tree) return SP_FRIEDMAN_EVICTED;
    c->inserts_total++;

    /* Pre-compute Tier-0 and Tier-1 signatures for the candidate once. */
    sp_kste_signature_t Q_sig0 = sp_kste_compute_signature(tree);
    sp_kste_anc_sig_t   Q_sig1;
    sp_kste_compute_anc_sig(tree, &Q_sig1);

    /* Subsumption decision: dual-tier dominance.
     *
     * Phase 4b architectural finding (see T4_RES_PROBE.json + SESSION-
     * STATE-friedman-4.md): strict Kruskal homeomorphic embedding is
     * too rigid for naturally-noised K vectors — even cos=0.995 pairs
     * almost never produce trees in a strict sub-tree relation.  But
     * the Tier-0 + Tier-1 dominance signatures DO discriminate cleanly
     * at the near-duplicate regime: 19% of cos>0.995 pairs are caught
     * vs 1% of unrelated pairs (17× separation).
     *
     * The signatures encode *necessary conditions* for embedding — the
     * structural fingerprint a K-tree must satisfy if it is to contain
     * Q.  Used as a STANDALONE equivalence relation, dominance is the
     * operationally-correct semantic operator: K dominates Q iff K's
     * label / depth / ancestor-pair counts all bound Q's.  Two trees
     * whose signatures co-dominate are members of the same structural
     * equivalence class — exactly what the sieve wants. */
    for (int i = 0; i < c->count; ++i) {
        c->slot_tests++;
        sp_friedman_slot_t *s = &c->slots[i];
        if (!sp_kste_sig_dominates(s->sig0, Q_sig0)) continue;
        c->tier1_tests++;
        if (!sp_kste_anc_sig_dominates(&s->sig1, &Q_sig1)) continue;
        /* Subsumed by signature dominance — no full embed needed. */
        c->full_embeds++;          /* repurposed as the "subsumption hits" counter */
        c->evictions++;
        return SP_FRIEDMAN_EVICTED;
    }

    /* Novel: admit.  If room, append; otherwise displace the slot with
     * the lowest skel_var (T2.8 Knight-Skeleton fallback). */
    int slot_idx;
    sp_friedman_decision decision;
    if (c->count < c->capacity) {
        slot_idx = c->count++;
        decision = SP_FRIEDMAN_ADMITTED;
        c->admissions++;
    } else {
        int min_idx = 0;
        float min_var = c->slots[0].skel_var;
        for (int i = 1; i < c->count; ++i) {
            if (c->slots[i].skel_var < min_var) {
                min_var = c->slots[i].skel_var;
                min_idx = i;
            }
        }
        slot_idx = min_idx;
        decision = SP_FRIEDMAN_REPLACED;
        c->replacements++;
    }

    sp_friedman_slot_t *s = &c->slots[slot_idx];
    s->tree     = *tree;          /* sp_kste_tree is POD, 64 B copy        */
    s->sig0     = Q_sig0;
    s->sig1     = Q_sig1;
    s->skel_var = skel_var;
    s->pos      = pos;
    s->gen      = c->gen_counter++;
    return decision;
}

double sp_friedman_cache_eviction_rate(const sp_friedman_cache_t *c)
{
    if (!c || c->inserts_total == 0) return 0.0;
    return (double)c->evictions / (double)c->inserts_total;
}

/* ---------- Structural predicates --------------------------------------- */

int sp_predicate_anchor_count(const sp_kste_tree *T)
{
    if (!T) return 0;
    int n = 0;
    for (int i = 1; i < T->node_count; ++i) {
        if (sp_kste_unpack_parent(T->parents, i) == 0u &&
            sp_kste_unpack_label (T->labels,  i) == SP_KSTE_LBL_A) ++n;
    }
    return n;
}

int sp_predicate_label_b_count(const sp_kste_tree *T)
{
    if (!T) return 0;
    int n = 0;
    for (int i = 1; i < T->node_count; ++i) {
        if (sp_kste_unpack_label(T->labels, i) == SP_KSTE_LBL_B) ++n;
    }
    return n;
}

int sp_predicate_label_c_count(const sp_kste_tree *T)
{
    if (!T) return 0;
    int n = 0;
    for (int i = 1; i < T->node_count; ++i) {
        if (sp_kste_unpack_label(T->labels, i) == SP_KSTE_LBL_C) ++n;
    }
    return n;
}

int sp_predicate_node_count(const sp_kste_tree *T)
{
    return T ? (int)T->node_count : 0;
}

/* Max depth via parent-walk per node (correct because parent[i] < i for
 * all i, so we can fill a depth array in one pass). */
int sp_predicate_max_depth(const sp_kste_tree *T)
{
    if (!T) return 0;
    int depth[SP_KSTE_MAX_NODES];
    depth[0] = 0;
    int max_depth = 0;
    for (int i = 1; i < T->node_count; ++i) {
        int p = sp_kste_unpack_parent(T->parents, i);
        if (p < 0 || p >= i) { depth[i] = 0; continue; }
        depth[i] = depth[p] + 1;
        if (depth[i] > max_depth) max_depth = depth[i];
    }
    return max_depth;
}

/* ---------- Extended-Domain Reduction check ----------------------------- */

/* Paper III §11: phi(v) holds  ⇒  phi*(v) holds, where phi* relativises
 * phi to the active-window RO subset.  Concretely: if phi(v) reports a
 * value X, then there must exist at least one RO slot whose phi-value
 * is >= X, witnessing that v's structural property is consistent with
 * what RO sees.  This captures "the active window covers v's claim".
 *
 * The check is primitive-recursive: an O(|RO|) scan, no recursion. */
int sp_extended_reduction_check(
    const sp_friedman_cache_t *cache,
    int                        ro_count,
    const sp_kste_tree        *v,
    sp_predicate_t             phi)
{
    if (!cache || !v || !phi) return 0;
    int phi_v = phi(v);

    /* Trivial reduction: phi(v) == 0 always restricts cleanly. */
    if (phi_v <= 0) return 1;

    int ro_lo = cache->count - ro_count;
    if (ro_lo < 0) ro_lo = 0;

    /* Scan RO slots; if any satisfies phi >= phi(v), the reduction
     * witness is found. */
    for (int i = ro_lo; i < cache->count; ++i) {
        if (phi(&cache->slots[i].tree) >= phi_v) return 1;
    }
    return 0;
}

/* ---------- Choice operator F ------------------------------------------- */

int sp_kste_compare(const sp_kste_tree *a, const sp_kste_tree *b)
{
    if (a == b) return 0;
    if (!a) return -1;
    if (!b) return  1;
    /* Compare packed bytes in canonical order: node_count first (so
     * smaller trees come first), then labels, then parents.  This is
     * a total order on T_{60,3} that is identical across platforms. */
    if (a->node_count != b->node_count)
        return (int)a->node_count - (int)b->node_count;
    int r = std::memcmp(a->labels,  b->labels,  sizeof(a->labels));
    if (r) return r;
    return std::memcmp(a->parents, b->parents, sizeof(a->parents));
}

const sp_kste_tree *sp_kste_select_canonical(
    const sp_kste_tree *const *candidates,
    int                        n_candidates)
{
    if (!candidates || n_candidates <= 0) return nullptr;
    const sp_kste_tree *best = candidates[0];
    for (int i = 1; i < n_candidates; ++i) {
        if (sp_kste_compare(candidates[i], best) < 0) best = candidates[i];
    }
    return best;
}
