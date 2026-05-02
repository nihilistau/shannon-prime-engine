// sp_threadpool — persistent N-thread fork/join. See sp_threadpool.h.

#include "sp_threadpool.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace sp::engine {

namespace {

struct Pool {
    int                              n_threads = 0;
    std::vector<std::thread>         workers;       // n_threads-1 entries
    std::atomic<bool>                shutdown{false};

    // Task handoff. The driver sets `task` and `task_epoch`, signals
    // `start_cv`. Each worker wakes, runs task(its_id), atomically
    // increments `done_count`, signals `done_cv`. Driver waits for
    // done_count == n_threads-1, then runs task(0) on its own thread.
    std::mutex                          mu;
    std::condition_variable             start_cv;
    std::condition_variable             done_cv;
    const std::function<void(int)>*     task = nullptr;
    uint64_t                            task_epoch = 0;
    std::atomic<int>                    done_count{0};

    static Pool& inst() { static Pool p; return p; }
};

}  // anon

static void worker_main(int worker_id) {
    Pool& p = Pool::inst();
    uint64_t last_epoch = 0;
    while (true) {
        // Wait for a new task or shutdown signal.
        std::unique_lock<std::mutex> lk(p.mu);
        p.start_cv.wait(lk, [&] {
            return p.shutdown.load(std::memory_order_acquire) ||
                   p.task_epoch != last_epoch;
        });
        if (p.shutdown.load(std::memory_order_acquire)) return;
        const auto* t = p.task;
        last_epoch = p.task_epoch;
        lk.unlock();

        if (t) (*t)(worker_id);

        // Mark this slice done; wake driver if it was the last.
        const int prev = p.done_count.fetch_add(1, std::memory_order_acq_rel);
        if (prev == p.n_threads - 2) {  // we're the last worker
            std::lock_guard<std::mutex> g(p.mu);
            p.done_cv.notify_one();
        }
    }
}

void sp_threadpool_init(int n_threads) {
    Pool& p = Pool::inst();
    if (n_threads <= 1) {
        // No-op pool. n_threads = 1 means "driver thread only".
        if (p.n_threads == 1) return;
        sp_threadpool_shutdown();
        p.n_threads = 1;
        return;
    }
    if (p.n_threads == n_threads) return;
    sp_threadpool_shutdown();
    p.n_threads = n_threads;
    p.shutdown.store(false, std::memory_order_release);
    p.task_epoch = 0;
    p.done_count.store(0, std::memory_order_relaxed);
    p.workers.reserve((size_t)n_threads - 1);
    for (int i = 1; i < n_threads; ++i) {
        p.workers.emplace_back(worker_main, i);
    }
    std::fprintf(stderr,
        "[sp_threadpool] %d threads ready (1 driver + %d workers)\n",
        n_threads, n_threads - 1);
    static bool atexit_done = false;
    if (!atexit_done) {
        atexit_done = true;
        std::atexit([]() { sp_threadpool_shutdown(); });
    }
}

void sp_threadpool_shutdown() {
    Pool& p = Pool::inst();
    if (p.workers.empty()) {
        p.n_threads = 0;
        return;
    }
    {
        std::lock_guard<std::mutex> g(p.mu);
        p.shutdown.store(true, std::memory_order_release);
        p.start_cv.notify_all();
    }
    for (auto& t : p.workers) {
        if (t.joinable()) t.join();
    }
    p.workers.clear();
    p.n_threads = 0;
}

int sp_threadpool_n_threads() {
    return Pool::inst().n_threads;
}

void sp_parallel_for(const std::function<void(int)>& task) {
    Pool& p = Pool::inst();
    if (p.n_threads <= 1) {
        task(0);
        return;
    }
    // Hand work to workers and run slice 0 ourselves.
    {
        std::lock_guard<std::mutex> g(p.mu);
        p.task = &task;
        p.task_epoch++;
        p.done_count.store(0, std::memory_order_release);
        p.start_cv.notify_all();
    }
    task(0);

    // Wait for all (n_threads-1) workers to finish.
    {
        std::unique_lock<std::mutex> lk(p.mu);
        p.done_cv.wait(lk, [&] {
            return p.done_count.load(std::memory_order_acquire) == p.n_threads - 1;
        });
        p.task = nullptr;
    }
}

}  // namespace sp::engine
