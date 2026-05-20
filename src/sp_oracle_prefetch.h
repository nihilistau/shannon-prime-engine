/* sp_oracle_prefetch.h — Strike 4: NEON Oracle + UFS prefetch pipeline.
 *
 * The Oracle (a ~100M-param speculative draft model on the A710 gold
 * cluster) predicts upcoming layer accesses. Those predictions feed an
 * A510-silver-cluster worker that streams Band 2/3 pages from UFS into
 * rpcmem-pinned slots before HTP needs them. By the time the HTP
 * dispatch reaches the predicted layer, the bytes are already SMMU-
 * visible — UFS latency disappears under steady-state execution.
 *
 * This module is the COORDINATOR. The Oracle plugs in via predictions
 * (push); the consumer (attention / FFN dispatch) plugs in via reads
 * (pull). The backend abstraction lets us test with file I/O on host
 * and swap in rpcmem on device without changing call sites.
 *
 * Implementation: C public API, C++17 internals (std::thread + condvar).
 * The math submodule stays C-only; this lives in the engine layer.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#ifndef SP_ORACLE_PREFETCH_H
#define SP_ORACLE_PREFETCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend: a read function the worker invokes to populate a slot.
 * On host, point this at a fopen/pread of a binary file.
 * On device, point it at a rpcmem-backed read that DMA-copies UFS pages
 * into a registered SMMU buffer (no marshal copy, SMMU-visible to HTP).
 *
 * `read` returns bytes read (must equal `size` for success), or 0 on
 * error. It runs on the prefetch worker thread, so it may block — that's
 * the whole point. */
typedef struct sp_oracle_backend {
    size_t (*read)(void* user_data, int layer_idx, void* dst, size_t size);
    void*  user_data;
    size_t bytes_per_layer;
} sp_oracle_backend;

/* Opaque handle. */
typedef struct sp_oracle_prefetch sp_oracle_prefetch;

/* Create a coordinator. n_slots is the double-buffer depth (2 is standard;
 * 3 for triple-buffer on devices with slow UFS).
 * Returns NULL on allocation failure or invalid args. */
sp_oracle_prefetch* sp_oracle_prefetch_create(const sp_oracle_backend* backend,
                                              int n_slots);

/* Create a coordinator whose slot buffers are ION/DMA-BUF backed on
 * Linux/Android, mapping to physical pages that the SMMU can route to
 * both ARM and the Hexagon cDSP without a marshal copy. UFS reads land
 * directly in pages that HTP can DMA from.
 *
 * Allocation path: /dev/dma_heap/system (Android 12+, preferred) with
 * legacy /dev/ion fallback. Page-aligned. Caller must subsequently call
 * sp_oracle_prefetch_register_qnn() once a QNN context is available to
 * upgrade the slots to QnnMem-registered handles.
 *
 * On Windows or non-POSIX hosts this transparently falls back to malloc
 * and returns a coordinator that's functionally identical except for
 * zero-copy guarantees. Check sp_oracle_prefetch_slots_are_rpcmem() to
 * detect which path is active. */
sp_oracle_prefetch* sp_oracle_prefetch_create_rpcmem(const sp_oracle_backend* backend,
                                                     int n_slots);

/* Returns 1 if the coordinator's slot buffers are ION/DMA-BUF backed,
 * 0 if they're plain malloc'd. Informational — the public API is
 * identical in both cases. */
int sp_oracle_prefetch_slots_are_rpcmem(const sp_oracle_prefetch* c);

/* Get the DMA-BUF file descriptor for slot `slot_idx`. Returns -1 if
 * not ION-backed or out of range. Used by the QNN registration helper
 * to construct Qnn_MemDescriptor_t with type=ION + fd. */
int sp_oracle_prefetch_slot_dmabuf_fd(const sp_oracle_prefetch* c, int slot_idx);

/* Number of slots in the coordinator. */
int sp_oracle_prefetch_num_slots(const sp_oracle_prefetch* c);

/* Shut down and free. Joins the worker, frees all slot memory. */
void sp_oracle_prefetch_destroy(sp_oracle_prefetch* c);

/* Push an Oracle prediction: "we will read layer_idx soon".
 * Non-blocking — enqueues the prediction and returns immediately.
 * The worker fetches in the background. Duplicate predictions for a
 * layer that's already cached or in-flight are silently dropped. */
void sp_oracle_prefetch_predict(sp_oracle_prefetch* c, int layer_idx);

/* Synchronously retrieve layer_idx's bytes. Returns a pointer to a
 * borrowed slot buffer of size backend->bytes_per_layer.
 *
 * Fast path (hit): the worker pre-fetched this layer — returns immediately.
 * Slow path (miss): blocks until the fetch completes (worker drives it
 *                   to the head of the queue, or runs synchronously if
 *                   the queue is empty).
 *
 * The returned pointer is valid until the next call that would evict
 * this slot — practically, valid until the caller has consumed it for
 * the current layer dispatch. Caller must not free.
 *
 * Returns NULL on backend error. */
const void* sp_oracle_prefetch_get(sp_oracle_prefetch* c, int layer_idx);

/* Observability — populated atomically. */
typedef struct {
    uint64_t predictions;   /* sp_oracle_prefetch_predict invocations */
    uint64_t prefetches;    /* worker-initiated reads completed */
    uint64_t hits;          /* sp_oracle_prefetch_get found pre-fetched data */
    uint64_t misses;        /* sp_oracle_prefetch_get had to wait for I/O */
    uint64_t evictions;     /* slot replaced before being read */
} sp_oracle_prefetch_stats;

void sp_oracle_prefetch_get_stats(const sp_oracle_prefetch* c,
                                   sp_oracle_prefetch_stats* out);

/* ─── Thread affinity (POSIX only — Android, Linux) ──────────────── */

/* Returns the worker's native thread handle. On Linux/Android this is a
 * pthread_t suitable for pthread_setaffinity_np(). On Windows it returns
 * the std::thread::native_handle() (a HANDLE), so callers can mix it
 * with SetThreadAffinityMask. Returns 0 if the worker is not running. */
uintptr_t sp_oracle_prefetch_get_worker_native_handle(const sp_oracle_prefetch* c);

/* Pin the worker thread to the given CPU set. On the S22 Ultra:
 *   silver (A510, energy):  cpus = {0,1,2,3}, n = 4
 *   gold   (A710, perf):    cpus = {4,5,6}, n = 3
 *   prime  (X2, peak):      cpus = {7}, n = 1
 *
 * Returns 0 on success, -1 on platform-not-supported or pthread error.
 * On Windows this returns -1 (caller can use SetThreadAffinityMask on
 * the native handle directly if needed for testing). */
int sp_oracle_prefetch_pin_worker_to_cpus(sp_oracle_prefetch* c,
                                          const int* cpus,
                                          int n_cpus);

/* ─── Built-in backends ───────────────────────────────────────────── */

/* POSIX pread-from-fd backend. The fd must remain open for the
 * coordinator's lifetime. `base_offset` is added to every read so a
 * single large model file can host multiple tensors with caller-known
 * absolute layouts. Reads at offset = base_offset + layer_idx * bytes_per_layer.
 *
 * Cross-platform: uses pread on POSIX, ReadFile + OVERLAPPED on Windows. */
sp_oracle_backend sp_oracle_prefetch_pread_backend(int fd,
                                                   long long base_offset,
                                                   size_t bytes_per_layer);

#ifdef __cplusplus
}
#endif

#endif /* SP_ORACLE_PREFETCH_H */
