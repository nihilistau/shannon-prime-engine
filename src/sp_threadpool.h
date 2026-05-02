// Shannon-Prime Engine — minimal persistent thread pool.
// Copyright (C) 2026 Ray Daniels. AGPLv3.
//
// Phase 4.6 perf foundation: parallel-for over a fixed-size pool of
// worker threads. Workers are persistent (created once at startup,
// joined at shutdown) so per-matmul dispatch overhead is just
// condition-variable signaling — order of microseconds, vs ~50 µs to
// spawn-then-join a fresh std::thread on Android.
//
// Used by sp_kernels_cpu's matmul kernels to partition output rows /
// columns across cores. SP_ENGINE_THREADS env (or 4 by default)
// picks the worker count.

#pragma once

#include <cstddef>
#include <functional>

namespace sp::engine {

// One-time init. n_threads ≤ 1 disables threading (fall through to
// inline call). Idempotent — repeat calls with the same n are no-ops.
// Must be called from the driver thread before any sp_parallel_for
// is invoked.
void sp_threadpool_init(int n_threads);

// Tear down the worker threads cleanly. Optional — also runs at
// process exit via atexit().
void sp_threadpool_shutdown();

// Active worker count (0 if uninitialized, 1+ once initialized). The
// driver thread itself counts as one of the workers — sp_parallel_for
// always uses (active_workers - 1) extra threads + the calling thread.
int  sp_threadpool_n_threads();

// Run `task(thread_id)` for thread_id in [0, n_threads()). Splits
// work across the pool + driver thread; returns when ALL slices are
// done. Caller-supplied `task` MUST partition its work using the
// thread_id (typically by computing [start, end) row range).
//
// Cheap if n_threads() == 1 — just calls task(0) inline. No
// thread synchronisation overhead in the single-threaded case.
void sp_parallel_for(const std::function<void(int thread_id)>& task);

}  // namespace sp::engine
