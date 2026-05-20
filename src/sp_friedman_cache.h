/* sp_friedman_cache.h — Phase 3: KSTE-keyed sieve cache.
 *
 * Paper III §3 defines the Friedman sieve as an admission policy:
 * a new tree T_t is added iff it is NOT homeomorphically embeddable
 * into any cached tree.  Paper IV §5 specifies the engine-side
 * data structure.
 *
 * Invariants:
 *   1. Termination (T2.1).  Cache size is bounded by the antichain
 *      number of T_{60,3}.  Empirically the cache plateaus well before
 *      the 4096-slot capacity even on 100 000 random insertions.
 *   2. Closure axiom (T2.5).  big_subset(slot_i) AND big_subset(slot_j)
 *      ⇒ their intersection is also a big_subset, by the structural
 *      definition of "big" used in §10.4 of the test suite.
 *   3. Eviction on subsumption (T2.6).  T_new ⊑ T_existing  ⇒
 *      T_new is rejected (SP_FRIEDMAN_EVICTED).
 *   4. Admission on novelty (T2.7).  T_new ⊀ T_i ∀i  ⇒  T_new admitted.
 *   5. Knight-Skeleton fallback (T2.8).  Cache full + novel arrival  ⇒
 *      replace the slot with the lowest skel_var.
 *   6. Extended-Domain Reduction (T2.12).  For any structural predicate
 *      ϕ and any canonical witness v = F(A) admitted in the cache,
 *      ϕ(v) ⇒ ϕ*(v) where ϕ* is the relativisation of ϕ to the
 *      active-window subset RO.  Primitive-recursively checkable.
 *
 * Engine-side code lives under shannon-prime-engine/src/.  Wire flag:
 * `SP_FRIEDMAN_SIEVE` (CMake option).  No __int128, no global state.
 *
 * Copyright (C) 2026 Ray Daniels.  AGPLv3 / commercial.
 */

#ifndef SP_FRIEDMAN_CACHE_H
#define SP_FRIEDMAN_CACHE_H

#include "sp_kste.h"   /* sp_kste_tree, sp_kste_embed                 */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Slot --------------------------------------------------------- */

/* One cache slot.  Phase 3 stores the tree, the Knight-Skeleton variance
 * (fallback eviction key), the originating token position, and a
 * monotonic generation counter for LRU-style debugging.  The Spinor
 * block and the transitive-closure mask from Paper IV §5.1 are NOT
 * stored yet — they are read-path artefacts that the polynomial-ring
 * scorer in Phase 5/6 will fold back in.  This deferral keeps Phase 3
 * a pure admission-policy layer. */
typedef struct {
    sp_kste_tree         tree;     /* 64 B — packed labelled tree            */
    sp_kste_signature_t  sig0;     /*  8 B — Tier-0 dominance signature      */
    sp_kste_anc_sig_t    sig1;     /* 16 B — Tier-1 ancestor-pair multiset   */
    float                skel_var; /*  4 B — Knight-Skeleton variance        */
    int32_t              pos;      /*  4 B — originating token position      */
    int32_t              gen;      /*  4 B — monotonic admission generation  */
    int32_t              _pad[5];  /* 20 B — pad to 128 B cache-line target  */
} sp_friedman_slot_t;
/* sizeof = 80 B; 64-byte aligned at the cache-allocation boundary,
 * which keeps the predicted hot fields (tree.labels + skel_var)
 * on a single cache line. */

/* ---------- Cache -------------------------------------------------------- */

typedef enum {
    SP_FRIEDMAN_EVICTED   = 0,  /* subsumed by an existing slot — not added */
    SP_FRIEDMAN_ADMITTED  = 1,  /* novel, added to a fresh slot             */
    SP_FRIEDMAN_REPLACED  = 2   /* novel; displaced a low-variance slot     */
} sp_friedman_decision;

typedef struct {
    sp_friedman_slot_t *slots;
    int32_t             capacity;
    int32_t             count;
    int32_t             gen_counter;
    int32_t             _pad;

    /* Cumulative counters across the cache's lifetime. */
    uint64_t            inserts_total;
    uint64_t            evictions;       /* SP_FRIEDMAN_EVICTED returns     */
    uint64_t            admissions;      /* SP_FRIEDMAN_ADMITTED returns    */
    uint64_t            replacements;    /* SP_FRIEDMAN_REPLACED returns    */

    /* Phase 5: layered-filter counters (per slot tested during inserts). */
    uint64_t            slot_tests;      /* total Tier-0 dominance checks   */
    uint64_t            tier1_tests;     /* survived Tier-0, went to Tier-1 */
    uint64_t            full_embeds;     /* survived Tier-1, ran full embed */
} sp_friedman_cache_t;

/* Construct / destroy. */
int  sp_friedman_cache_init   (sp_friedman_cache_t *c, int capacity);
void sp_friedman_cache_destroy(sp_friedman_cache_t *c);

/* Insert a new tree.  Embedding tests run against every existing slot
 * in the order they were admitted (oldest first).  Returns:
 *   SP_FRIEDMAN_EVICTED   if `tree` ⊑ slot for some slot,
 *   SP_FRIEDMAN_ADMITTED  if `tree` is novel and there was room,
 *   SP_FRIEDMAN_REPLACED  if `tree` is novel but cache was full; the
 *                         slot with the lowest skel_var is overwritten.
 *
 * `tree` is copied (deep) into the cache; the caller may reuse the
 * buffer immediately. */
sp_friedman_decision sp_friedman_cache_insert(
    sp_friedman_cache_t *c,
    const sp_kste_tree  *tree,
    float                skel_var,
    int32_t              pos);

/* Eviction rate (admissions + replacements) / inserts_total. */
double sp_friedman_cache_eviction_rate(const sp_friedman_cache_t *c);

/* ---------- Axiomatic layer (Paper III §11) ------------------------------ */

/* Predicates take a tree and return an integer-valued structural
 * quantity (count, depth, etc.).  Used by sp_extended_reduction_check
 * to verify the ED-Reduction invariant (T2.12). */
typedef int (*sp_predicate_t)(const sp_kste_tree *T);

/* Built-in predicates from the test suite (T2.12 §10.4). */
int sp_predicate_anchor_count(const sp_kste_tree *T);
int sp_predicate_label_b_count(const sp_kste_tree *T);
int sp_predicate_label_c_count(const sp_kste_tree *T);
int sp_predicate_node_count(const sp_kste_tree *T);
int sp_predicate_max_depth(const sp_kste_tree *T);

/* The Extended-Domain Reduction check.  Given a canonical witness v
 * already admitted to the cache and a predicate phi, verify that the
 * value of phi on v is consistent with phi restricted to the most
 * recent `ro_count` slots (the active-window subset RO from Paper III
 * §11.1).  Returns 1 iff the implication phi(v) ⇒ phi*(v) holds (i.e.
 * phi(v) is bounded above by the maximum of phi over RO).
 *
 * The check is primitive-recursive: a finite scan of cache_RO. */
int sp_extended_reduction_check(
    const sp_friedman_cache_t *cache,
    int                        ro_count,    /* size of active window      */
    const sp_kste_tree        *v,           /* canonical witness          */
    sp_predicate_t             phi);

/* ---------- Choice operator F (Paper IV §10) ----------------------------- */

/* Packed-byte lexicographic order — the canonical total order ≺_F. */
int sp_kste_compare(const sp_kste_tree *a, const sp_kste_tree *b);

/* Select the ≺_F-minimum tree from a list of candidates.  Returns a
 * pointer into the input array, NULL if n_candidates == 0. */
const sp_kste_tree *sp_kste_select_canonical(
    const sp_kste_tree *const *candidates,
    int                        n_candidates);

#ifdef __cplusplus
}
#endif

#endif /* SP_FRIEDMAN_CACHE_H */
