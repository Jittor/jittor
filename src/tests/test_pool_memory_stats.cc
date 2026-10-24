#include "codegen/opt/kernel_ir.h"
#include "mem/pool_memory_stats.h"
#include <atomic>
#include <thread>
#include <vector>

namespace jittor {

JIT_TEST(pool_memory_event_peaks) {
    PoolMemoryStats ledger;
    int first, second, host, other;
    ledger.update(&first, 0, 100, 200);
    ledger.update(&second, 0, 300, 500);
    ledger.update(&host, -1, 9000, 9000);
    ledger.update(&other, 1, 1000, 2000);
    ledger.update(&first, 0, 0, 200);
    ledger.update(&second, 0, 0, 0);
    auto stats = ledger.read(0);
    ASSERTop(stats.allocated, ==, 0);
    ASSERTop(stats.reserved, ==, 200);
    ASSERTop(stats.peak_allocated, ==, 400);
    ASSERTop(stats.peak_reserved, ==, 700);
    stats = ledger.read(0, true);
    ASSERTop(stats.peak_allocated, ==, 0);
    ASSERTop(stats.peak_reserved, ==, 200);
    ASSERTop(ledger.read(1).peak_allocated, ==, 1000);
    ledger.update(&first, 0, 50, 200);
    ledger.forget(&first);
    stats = ledger.read(0);
    ASSERTop(stats.allocated, ==, 0);
    ASSERTop(stats.reserved, ==, 0);
    ASSERTop(stats.peak_allocated, ==, 50);
    ledger.unsupported(0);
    bool rejected = false;
    try { ledger.read(0); } catch (const std::runtime_error&) { rejected = true; }
    ASSERT(rejected);

    // Distinct pools share a device. Peaks are max(sum(current)), not the
    // sum of independent per-pool maxima, and never include another device.
    PoolMemoryStats staggered;
    staggered.update(&first, 0, 100, 200);
    staggered.update(&first, 0, 0, 0);
    staggered.update(&second, 0, 300, 500);
    ASSERTop(staggered.read(0).peak_allocated, ==, 300);
    ASSERTop(staggered.read(0).peak_reserved, ==, 500);

    PoolMemoryStats concurrent;
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([&] {
        int pool;
        concurrent.update(&pool, 0, 100, 200);
        ++ready;
        while (ready.load() != 4) std::this_thread::yield();
        concurrent.update(&pool, 0, 0, 200);
        concurrent.forget(&pool);
    });
    for (auto& thread : threads) thread.join();
    ASSERTop(concurrent.read(0).allocated, ==, 0);
    ASSERTop(concurrent.read(0).reserved, ==, 0);
    ASSERTop(concurrent.read(0).peak_allocated, ==, 400);
    ASSERTop(concurrent.read(0).peak_reserved, ==, 800);
}

} // namespace jittor
