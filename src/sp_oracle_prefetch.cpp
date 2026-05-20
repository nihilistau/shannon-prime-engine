/* sp_oracle_prefetch.cpp — Strike 4 implementation.
 *
 * Single worker thread, N-slot LRU buffer, mutex-guarded prediction queue.
 *
 * Thread topology (on S22 Ultra):
 *   - Caller threads (HTP dispatch, Oracle producer) run on X2 / A710 cores.
 *   - This worker thread is intended to be affinity-pinned to the A510
 *     silver cluster (cores 0-3) by the caller via sched_setaffinity
 *     after sp_oracle_prefetch_create. We expose a hook for that via
 *     a thread-id getter (future) — for v1 the caller pins itself before
 *     constructing the coordinator, and the worker inherits affinity.
 *     The clean affinity API is Strike 4.5.
 *
 * Slot eviction: LRU by version counter. Each slot tracks the version
 * at which it was last touched; predict() bumps the version on the
 * filled slot, get() bumps it on hit. The lowest-version slot is the
 * eviction candidate when a new prefetch can't reuse an existing slot.
 *
 * Miss handling: sp_oracle_prefetch_get scans for the layer; if absent,
 * it bypasses the worker queue and calls backend->read directly on the
 * caller's thread (still correct, just no latency hiding). This avoids
 * a deadlock where the worker is stalled on a slow read and the caller
 * needs a different layer urgently.
 */

#include "sp_oracle_prefetch.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  include <sys/types.h>
#  include <sys/ioctl.h>
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <errno.h>
#  define SP_OPF_POSIX 1
#  define SP_OPF_ION_AVAILABLE 1
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  define SP_OPF_WIN 1
#else
#  define SP_OPF_NONE 1
#endif

/* ─── ION / DMA-BUF allocation (Linux/Android) ──────────────────────
 * Mirrors the math repo's backends/hexagon/sp_mem.c logic but inlined
 * here to avoid the layering dependency. /dev/dma_heap/system is the
 * preferred path (Android 12+); /dev/ion is the legacy fallback.
 * Pages allocated here are SMMU-mappable — once registered with QNN
 * they're visible to the Hexagon cDSP for zero-copy DMA. */
#if defined(SP_OPF_ION_AVAILABLE)
namespace {

#ifndef DMA_HEAP_IOCTL_ALLOC
struct sp_opf_dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define SP_OPF_DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct sp_opf_dma_heap_allocation_data)
#endif

static inline size_t sp_opf_page_align(size_t n) { return (n + 4095u) & ~size_t{4095u}; }

/* Allocate `bytes` of DMA-BUF-backed memory, mmap into the caller's
 * address space. Returns 0 on success, populates *ptr_out / *fd_out /
 * *size_out with the page-aligned allocation. */
static int sp_opf_ion_alloc(size_t bytes, void** ptr_out, int* fd_out, size_t* size_out) {
    if (!ptr_out || !fd_out || !size_out) return -1;
    *ptr_out = nullptr; *fd_out = -1; *size_out = 0;

    size_t aligned = sp_opf_page_align(bytes);
    int fd = -1;

    /* Try /dev/dma_heap/system (Android 12+). */
    int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap_fd >= 0) {
        struct sp_opf_dma_heap_allocation_data alloc;
        std::memset(&alloc, 0, sizeof(alloc));
        alloc.len = aligned;
        alloc.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heap_fd, SP_OPF_DMA_HEAP_IOCTL_ALLOC, &alloc) == 0) {
            fd = (int)alloc.fd;
        }
        close(heap_fd);
    }
    if (fd < 0) {
        /* No ION/dma-heap available — caller will fall back to malloc. */
        return -1;
    }

    void* p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return -1;
    }
    *ptr_out = p;
    *fd_out = fd;
    *size_out = aligned;
    return 0;
}

/* Teardown order: munmap → close(fd). When QnnMem registration is added,
 * the caller invokes the unregister callback before this is called. */
static void sp_opf_ion_free(void* ptr, int fd, size_t size) {
    if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
    if (fd >= 0) close(fd);
}

} /* anonymous namespace */
#endif /* SP_OPF_ION_AVAILABLE */

struct Slot {
    int       layer_idx = -1;       /* -1 = empty */
    void*     data      = nullptr;  /* size = backend.bytes_per_layer */
    uint64_t  version   = 0;        /* LRU clock */
    bool      in_flight = false;    /* worker currently filling */
    /* When the coordinator is ION-backed these are populated; otherwise -1/0.
     * The DMA-BUF fd is what callers register with QnnMem_register to make
     * the slot's pages visible to the Hexagon cDSP via the SMMU. */
    int       dmabuf_fd = -1;
    size_t    page_size = 0;        /* page-aligned allocation size */
};

struct sp_oracle_prefetch {
    sp_oracle_backend       backend;
    std::vector<Slot>       slots;
    bool                    rpcmem_backed = false;  /* slots from ION/dma-heap */

    std::mutex              mu;
    std::condition_variable cv_predict;   /* worker waits on new predictions */
    std::condition_variable cv_done;      /* getters wait on slot ready */
    std::queue<int>         pending;      /* layer indices to fetch */
    std::atomic<uint64_t>   version{1};

    std::thread             worker;
    std::atomic<bool>       shutdown{false};

    /* Stats. Counter increments are under the mutex (cheap enough). */
    sp_oracle_prefetch_stats stats{};

    /* Linux/Android TID of the worker, captured at thread start so the
     * caller can sched_setaffinity it. Zero until the worker is running. */
    std::atomic<long> worker_tid{0};

    /* Find a slot already holding `layer_idx`. Returns nullptr if absent. */
    Slot* find_slot_locked(int layer_idx) {
        for (auto& s : slots) {
            if (s.layer_idx == layer_idx && s.data && !s.in_flight) return &s;
        }
        return nullptr;
    }
    /* Find the LRU slot that's not in flight. */
    Slot* pick_eviction_locked() {
        Slot* victim = nullptr;
        uint64_t best = UINT64_MAX;
        for (auto& s : slots) {
            if (s.in_flight) continue;
            if (s.version < best) { best = s.version; victim = &s; }
        }
        return victim;
    }
};

static void worker_loop(sp_oracle_prefetch* c) {
#if defined(SP_OPF_POSIX)
    c->worker_tid.store((long)gettid());
#endif
    while (true) {
        int next_layer = -1;
        Slot* target = nullptr;
        {
            std::unique_lock<std::mutex> lk(c->mu);
            c->cv_predict.wait(lk, [c] {
                return c->shutdown.load() || !c->pending.empty();
            });
            if (c->shutdown.load() && c->pending.empty()) return;

            next_layer = c->pending.front();
            c->pending.pop();

            /* Already cached? Drop this prediction. */
            if (c->find_slot_locked(next_layer)) {
                continue;
            }
            /* Already in flight (worker is double-scheduled)? Drop. */
            for (auto& s : c->slots) {
                if (s.in_flight && s.layer_idx == next_layer) {
                    target = nullptr;  /* signal to skip */
                    next_layer = -2;
                    break;
                }
            }
            if (next_layer == -2) continue;

            target = c->pick_eviction_locked();
            if (!target) {
                /* All slots in flight — extremely rare. Re-queue and yield. */
                c->pending.push(next_layer);
                lk.unlock();
                std::this_thread::yield();
                continue;
            }
            if (target->layer_idx != -1) {
                c->stats.evictions++;
            }
            target->in_flight = true;
            target->layer_idx = next_layer;
        }

        /* Run the backend read OUTSIDE the mutex so other threads can hit
         * cached slots while we fetch. */
        size_t got = c->backend.read(c->backend.user_data, next_layer,
                                      target->data, c->backend.bytes_per_layer);
        bool ok = (got == c->backend.bytes_per_layer);

        {
            std::lock_guard<std::mutex> lk(c->mu);
            target->in_flight = false;
            if (ok) {
                target->version = c->version.fetch_add(1);
                c->stats.prefetches++;
            } else {
                /* Backend failure — clear the slot to avoid serving bad data. */
                target->layer_idx = -1;
            }
        }
        c->cv_done.notify_all();
    }
}

extern "C" {

/* Shared internal: validate args, allocate the coordinator struct, and
 * spin up the worker. Slot allocation is the caller's responsibility
 * (different per allocator strategy). */
static sp_oracle_prefetch* sp_opf_create_shell(const sp_oracle_backend* backend,
                                               int n_slots) {
    if (!backend || !backend->read || backend->bytes_per_layer == 0) return nullptr;
    if (n_slots < 1 || n_slots > 64) return nullptr;
    auto* c = new (std::nothrow) sp_oracle_prefetch();
    if (!c) return nullptr;
    c->backend = *backend;
    c->slots.resize((size_t)n_slots);
    return c;
}

sp_oracle_prefetch* sp_oracle_prefetch_create(const sp_oracle_backend* backend,
                                              int n_slots)
{
    auto* c = sp_opf_create_shell(backend, n_slots);
    if (!c) return nullptr;
    for (auto& s : c->slots) {
        s.data = std::malloc(backend->bytes_per_layer);
        if (!s.data) {
            for (auto& s2 : c->slots) if (s2.data) std::free(s2.data);
            delete c;
            return nullptr;
        }
    }
    c->rpcmem_backed = false;
    c->worker = std::thread(worker_loop, c);
    return c;
}

sp_oracle_prefetch* sp_oracle_prefetch_create_rpcmem(const sp_oracle_backend* backend,
                                                     int n_slots)
{
#if defined(SP_OPF_ION_AVAILABLE)
    auto* c = sp_opf_create_shell(backend, n_slots);
    if (!c) return nullptr;
    bool all_ion = true;
    for (auto& s : c->slots) {
        if (sp_opf_ion_alloc(backend->bytes_per_layer,
                             &s.data, &s.dmabuf_fd, &s.page_size) != 0) {
            all_ion = false;
            break;
        }
    }
    if (!all_ion) {
        /* Roll back any ION allocations and fall through to malloc. */
        for (auto& s : c->slots) {
            if (s.dmabuf_fd >= 0) {
                sp_opf_ion_free(s.data, s.dmabuf_fd, s.page_size);
                s.data = nullptr; s.dmabuf_fd = -1; s.page_size = 0;
            }
        }
        for (auto& s : c->slots) {
            s.data = std::malloc(backend->bytes_per_layer);
            if (!s.data) {
                for (auto& s2 : c->slots) if (s2.data) std::free(s2.data);
                delete c;
                return nullptr;
            }
        }
        c->rpcmem_backed = false;
        std::fprintf(stderr,
            "[sp_oracle_prefetch] ION/dma-heap not available — fell back to malloc\n");
    } else {
        c->rpcmem_backed = true;
        std::fprintf(stderr,
            "[sp_oracle_prefetch] %d slots × %zu B allocated via DMA-BUF (SMMU-mappable)\n",
            n_slots, backend->bytes_per_layer);
    }
    c->worker = std::thread(worker_loop, c);
    return c;
#else
    /* No ION support on this build — caller gets malloc-backed coordinator
     * silently. Public API is unchanged; rpcmem_backed = 0 in stats. */
    return sp_oracle_prefetch_create(backend, n_slots);
#endif
}

void sp_oracle_prefetch_destroy(sp_oracle_prefetch* c) {
    if (!c) return;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        c->shutdown.store(true);
    }
    c->cv_predict.notify_all();
    if (c->worker.joinable()) c->worker.join();
#if defined(SP_OPF_ION_AVAILABLE)
    if (c->rpcmem_backed) {
        for (auto& s : c->slots) {
            if (s.dmabuf_fd >= 0) {
                sp_opf_ion_free(s.data, s.dmabuf_fd, s.page_size);
            }
        }
    } else
#endif
    {
        for (auto& s : c->slots) if (s.data) std::free(s.data);
    }
    delete c;
}

int sp_oracle_prefetch_slots_are_rpcmem(const sp_oracle_prefetch* c) {
    if (!c) return 0;
    return c->rpcmem_backed ? 1 : 0;
}

int sp_oracle_prefetch_slot_dmabuf_fd(const sp_oracle_prefetch* c, int slot_idx) {
    if (!c || slot_idx < 0 || (size_t)slot_idx >= c->slots.size()) return -1;
    return c->slots[(size_t)slot_idx].dmabuf_fd;
}

int sp_oracle_prefetch_num_slots(const sp_oracle_prefetch* c) {
    if (!c) return 0;
    return (int)c->slots.size();
}

void sp_oracle_prefetch_predict(sp_oracle_prefetch* c, int layer_idx) {
    if (!c || layer_idx < 0) return;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        /* Already cached? Skip. */
        if (c->find_slot_locked(layer_idx)) {
            c->stats.predictions++;
            return;
        }
        c->pending.push(layer_idx);
        c->stats.predictions++;
    }
    c->cv_predict.notify_one();
}

const void* sp_oracle_prefetch_get(sp_oracle_prefetch* c, int layer_idx) {
    if (!c || layer_idx < 0) return nullptr;
    {
        std::unique_lock<std::mutex> lk(c->mu);
        /* Fast path — already pre-fetched. */
        if (auto* s = c->find_slot_locked(layer_idx)) {
            s->version = c->version.fetch_add(1);
            c->stats.hits++;
            return s->data;
        }
        /* If the worker is already fetching this layer, wait for it.
         * This is still a hit semantically — the prediction landed,
         * we just arrived a tad early. */
        for (auto& s : c->slots) {
            if (s.in_flight && s.layer_idx == layer_idx) {
                c->cv_done.wait(lk, [&] {
                    return !s.in_flight && s.layer_idx == layer_idx;
                });
                s.version = c->version.fetch_add(1);
                c->stats.hits++;
                return s.data;
            }
        }
        /* Cold miss: fetch synchronously on the caller's thread.
         * Pick an LRU victim, mark in-flight (so a concurrent get for
         * the same layer waits), drop the lock, fetch, retake the lock. */
        Slot* target = c->pick_eviction_locked();
        if (!target) {
            /* All slots are in flight for other layers — wait for any to free. */
            c->cv_done.wait(lk, [&] {
                for (auto& s : c->slots) if (!s.in_flight) return true;
                return false;
            });
            target = c->pick_eviction_locked();
            if (!target) return nullptr;  /* shouldn't reach */
        }
        if (target->layer_idx != -1) c->stats.evictions++;
        target->layer_idx = layer_idx;
        target->in_flight = true;
        c->stats.misses++;

        lk.unlock();
        size_t got = c->backend.read(c->backend.user_data, layer_idx,
                                      target->data, c->backend.bytes_per_layer);
        bool ok = (got == c->backend.bytes_per_layer);
        lk.lock();
        target->in_flight = false;
        if (ok) {
            target->version = c->version.fetch_add(1);
            c->cv_done.notify_all();
            return target->data;
        } else {
            target->layer_idx = -1;
            c->cv_done.notify_all();
            return nullptr;
        }
    }
}

void sp_oracle_prefetch_get_stats(const sp_oracle_prefetch* c,
                                   sp_oracle_prefetch_stats* out) {
    if (!c || !out) return;
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(c->mu));
    *out = c->stats;
}

/* ─── Thread affinity ──────────────────────────────────────────────── */

uintptr_t sp_oracle_prefetch_get_worker_native_handle(const sp_oracle_prefetch* c) {
    if (!c) return 0;
    auto* mc = const_cast<sp_oracle_prefetch*>(c);
    if (!mc->worker.joinable()) return 0;
#if defined(SP_OPF_POSIX)
    return (uintptr_t)mc->worker.native_handle();
#elif defined(SP_OPF_WIN)
    return (uintptr_t)mc->worker.native_handle();
#else
    return 0;
#endif
}

int sp_oracle_prefetch_pin_worker_to_cpus(sp_oracle_prefetch* c,
                                          const int* cpus,
                                          int n_cpus) {
    if (!c || !cpus || n_cpus <= 0) return -1;
#if defined(SP_OPF_POSIX)
    if (!c->worker.joinable()) return -1;
    /* Spin briefly waiting for the worker to publish its TID. The thread
     * sets it at the very top of worker_loop, so this is microseconds at
     * most. Bail after ~10 ms to avoid a stuck-thread hang. */
    long tid = 0;
    for (int i = 0; i < 100; ++i) {
        tid = c->worker_tid.load();
        if (tid != 0) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (tid == 0) return -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n_cpus; ++i) {
        if (cpus[i] < 0 || cpus[i] >= (int)sizeof(set) * 8) continue;
        CPU_SET((size_t)cpus[i], &set);
    }
    /* Android bionic exposes sched_setaffinity but not pthread_setaffinity_np.
     * sched_setaffinity uses the TID (which is the same as PID on Linux for
     * a single-threaded process — but for our worker thread it's the kernel
     * task ID we captured via gettid()). */
    int rc = sched_setaffinity((pid_t)tid, sizeof(set), &set);
    return (rc == 0) ? 0 : -1;
#else
    /* Windows: caller can SetThreadAffinityMask on the native handle.
     * Returning -1 to signal "not supported via this API". */
    (void)cpus; (void)n_cpus;
    return -1;
#endif
}

/* ─── Built-in pread backend ─────────────────────────────────────── */

namespace {
struct PreadCtx {
    int       fd;
    long long base_offset;
    size_t    bytes_per_layer;
};

#if defined(SP_OPF_POSIX)
size_t pread_backend_read(void* user, int layer_idx, void* dst, size_t size) {
    auto* p = static_cast<PreadCtx*>(user);
    if (size != p->bytes_per_layer) return 0;
    off_t off = (off_t)(p->base_offset + (long long)layer_idx * (long long)p->bytes_per_layer);
    size_t total = 0;
    uint8_t* out = static_cast<uint8_t*>(dst);
    while (total < size) {
        ssize_t r = pread(p->fd, out + total, size - total, off + (off_t)total);
        if (r <= 0) return total;
        total += (size_t)r;
    }
    return total;
}
#elif defined(SP_OPF_WIN)
size_t pread_backend_read(void* user, int layer_idx, void* dst, size_t size) {
    auto* p = static_cast<PreadCtx*>(user);
    if (size != p->bytes_per_layer) return 0;
    LARGE_INTEGER offset;
    offset.QuadPart = p->base_offset + (long long)layer_idx * (long long)p->bytes_per_layer;
    OVERLAPPED ov{};
    ov.Offset     = offset.LowPart;
    ov.OffsetHigh = (DWORD)offset.HighPart;
    DWORD got = 0;
    HANDLE h = (HANDLE)_get_osfhandle(p->fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    BOOL ok = ReadFile(h, dst, (DWORD)size, &got, &ov);
    if (!ok && GetLastError() != ERROR_HANDLE_EOF) return 0;
    return (size_t)got;
}
#else
size_t pread_backend_read(void*, int, void*, size_t) { return 0; }
#endif

/* PreadCtx storage: caller-owned via a heap allocation we leak (a single
 * backend per coordinator, freed by the coordinator's destroy). This
 * keeps the public API a flat struct without lifetime entanglement. */
PreadCtx* alloc_pread_ctx(int fd, long long base_offset, size_t bytes_per_layer) {
    auto* p = new (std::nothrow) PreadCtx;
    if (!p) return nullptr;
    p->fd = fd;
    p->base_offset = base_offset;
    p->bytes_per_layer = bytes_per_layer;
    return p;
}
} /* anonymous namespace */

sp_oracle_backend sp_oracle_prefetch_pread_backend(int fd,
                                                   long long base_offset,
                                                   size_t bytes_per_layer) {
    sp_oracle_backend be{};
    be.read = pread_backend_read;
    be.user_data = alloc_pread_ctx(fd, base_offset, bytes_per_layer);
    be.bytes_per_layer = bytes_per_layer;
    return be;
}

} /* extern "C" */
