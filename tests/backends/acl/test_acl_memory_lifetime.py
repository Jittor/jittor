"""Real-device raw allocator release ordering, isolated from the pytest parent.

The C++ probe queues a blocked native callback followed by a copy that reads
the allocation being freed. An independent C++ thread calls the production
backend's raw allocator. Free must remain blocked until the callback is
released; the copied pattern and caller-device restoration are then checked.
An unfixed backend exits immediately on premature free, before permitting
the pending copy to access invalid memory. Run each kind in a fresh process:

  python -m pytest -q tests/backends/acl/test_acl_memory_lifetime.py

Each isolated child compiles a small host binding. It does
not replace BackendOps or mock synchronization/free functions.
"""

import argparse
import json
from pathlib import Path

import pytest

import jittor as jt
import jittor_utils


SOURCE = r'''
#include <acl/acl.h>
#include "runtime/backend.h"
#include "mem/allocator.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace jittor {
namespace {
struct CopyGate {
    std::atomic<bool> entered{false}, release{false};
};
void wait_for_release(void* opaque) {
    auto& gate = *static_cast<CopyGate*>(opaque);
    gate.entered.store(true, std::memory_order_release);
    while (!gate.release.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
void await_flag(const std::atomic<bool>& flag, const char* description) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!flag.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error(description);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}

// @pyjt(run_acl_raw_free_probe)
vector<int> run_acl_raw_free_probe(bool pinned, int owner_device, int caller_device) {
    const auto& ops = backend_ops(BackendId::Acl);
    const int original_device = ops.current_device();
    ops.set_device(owner_device);
    const Device owner{BackendId::Acl, owner_device};
    const Device host{BackendId::Cpu, 0};
    auto kind = pinned ? BackendMemoryKind::Pinned : BackendMemoryKind::Device;
    Allocator* source_allocator = backend_raw_allocator(owner, kind);
    Allocator* destination_allocator = backend_raw_allocator(owner, BackendMemoryKind::Device);
    const size_t bytes = 65536;
    size_t source_token = 0, destination_token = 0;
    void* source = source_allocator->alloc(bytes, source_token);
    void* destination = destination_allocator->alloc(bytes, destination_token);
    std::vector<unsigned char> expected(bytes), actual(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) expected[i] = (i * 37 + 11) % 251;
    if (pinned) std::memcpy(source, expected.data(), bytes);
    else backend_copy(source, owner, expected.data(), host, bytes);

    BackendStream stream{owner, ops.stream_create(owner_device, false)};
    CopyGate gate;
    std::atomic<bool> freeing{false}, freed{false};
    int restored_device = -1;
    std::exception_ptr free_error;
    std::thread free_thread;
    try {
        ops.host_callback(stream, wait_for_release, &gate);
        await_flag(gate.entered, "native copy gate did not start");
        backend_copy_async(destination, owner, source, pinned ? host : owner, bytes, stream);
        free_thread = std::thread([&] {
            try {
                // A new host thread has no current ACL context yet. Bind it
                // directly before exercising the production free boundary;
                // the framework's device-switch API expects an existing one.
                if (aclrtSetDevice(caller_device) != ACL_SUCCESS)
                    throw std::runtime_error("could not bind free-thread ACL context");
                freeing.store(true, std::memory_order_release);
                source_allocator->free(source, bytes, source_token);
                if (aclrtGetDevice(&restored_device) != ACL_SUCCESS)
                    throw std::runtime_error("could not query free-thread ACL context");
            } catch (...) {
                free_error = std::current_exception();
            }
            freed.store(true, std::memory_order_release);
        });
        await_flag(freeing, "free thread did not reach the release boundary");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (freed.load(std::memory_order_acquire)) {
            // The queued copy still waits behind the gate. Never release it
            // onto memory the failed implementation has already unmapped.
            std::fprintf(stderr, "RAW_FREE_PREMATURE: production allocator returned before queued use\n");
            std::fflush(stderr);
            std::_Exit(3);
        }
        gate.release.store(true, std::memory_order_release);
        await_flag(freed, "raw free did not finish after releasing the gate");
        free_thread.join();
        if (free_error) std::rethrow_exception(free_error);
        ops.stream_synchronize(stream);
        backend_copy(actual.data(), host, destination, owner, bytes);
        if (actual != expected) throw std::runtime_error("copy read a released or corrupt source");
        if (restored_device != caller_device) throw std::runtime_error("raw free changed its caller's device");
    } catch (...) {
        gate.release.store(true, std::memory_order_release);
        if (free_thread.joinable()) free_thread.join();
        throw;
    }
    ops.stream_destroy(stream);
    destination_allocator->free(destination, bytes, destination_token);
    if (original_device >= 0) ops.set_device(original_device);
    return {1, 1, restored_device};
}
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("device", "pinned"), required=True)
    parser.add_argument("--owner-device", type=int, default=0)
    parser.add_argument("--caller-device", type=int)
    args = parser.parse_args()
    with jt.runtime.scope(use_cuda=1, use_acl=1):
        count = jt.core.backend_device_count("acl")
        if count < 1:
            raise RuntimeError("a real ACL device is required")
        caller = args.caller_device
        if caller is None:
            caller = 1 if count > 1 and args.owner_device == 0 else args.owner_device
        if not (0 <= args.owner_device < count and 0 <= caller < count):
            raise ValueError("device index outside the visible ACL device range")
        probe = jittor_utils.compile_module(SOURCE, jt.compiler.cc_flags)
        result = probe.run_acl_raw_free_probe(args.kind == "pinned", args.owner_device, caller)
        assert list(result) == [1, 1, caller], result
        print("RAW_FREE_OK", json.dumps({"kind": args.kind, "owner_device": args.owner_device,
                                         "caller_device": caller, "result": list(result)}), flush=True)


@pytest.mark.parametrize("kind", ["device", "pinned"])
def test_raw_free_waits_for_queued_use_and_preserves_data(kind):
    from _helpers import capability
    from _helpers.child_process import run_python_child
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for asynchronous memory lifetime")
    result = run_python_child(
        [str(Path(__file__).resolve()), "--kind", kind],
        without_torch_mode=True, crash_isolated=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "RAW_FREE_OK" in result.stdout, result.stdout


if __name__ == "__main__":
    main()
