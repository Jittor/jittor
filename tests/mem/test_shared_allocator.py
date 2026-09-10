"""Shared ownership must preserve the selected raw/caching allocator policy."""
import gc

import jittor as jt
import numpy as np
import pytest


@pytest.mark.parametrize("flags", [
    {},
    {"use_sfrl_allocator": 0},
    {"use_nfef_allocator": 1},
    {"use_stat_allocator": 1, "use_sfrl_allocator": 0},
    {"use_stat_allocator": 2},
])
def test_materialized_views_keep_aliases_and_lifetime(flags):
    with jt.flag_scope(use_cuda=0, **flags):
        for trial in range(4):
            # array alone uses the fixed host pool; an executed arithmetic
            # output exercises the allocator selected by the runtime flags.
            expected = np.arange(8, dtype="float32").reshape(4, 2) + trial * 100
            source = jt.array(expected) * 1.0
            source.sync()
            offset_view = source[1::2, :1]
            expanded = source[:, :1].expand(4, 3)
            offset_view.sync()
            expanded.sync()
            assert offset_view._storage_address == source._storage_address + 8
            assert expanded._storage_address == source._storage_address
            np.testing.assert_array_equal(offset_view.numpy(), expected[1::2, :1])
            np.testing.assert_array_equal(expanded.numpy(), expected[:, :1].repeat(3, axis=1))
            expected += 7
            source.assign(jt.array(expected))
            np.testing.assert_array_equal(offset_view.numpy(), expected[1::2, :1])
            del source, expanded
            gc.collect()
            # Same-sized allocations must not recycle a still-live view's
            # storage, even after the original owner has been released.
            poison = [jt.array(np.full((4, 2), -98765, "float32")) * 1.0 for _ in range(8)]
            jt.sync_all()
            np.testing.assert_array_equal(offset_view.numpy(), expected[1::2, :1])
            del poison, offset_view
            gc.collect()


def test_shared_allocator_releases_original_tuple_after_last_owner():
    with jt.flag_scope(use_cuda=0):
        result = jt.code([1], "int32", [], cpu_header=r'''
#include "mem/allocator/shared_allocator.h"
#include <thread>
struct SharedTestAllocator : jittor::Allocator {
    char storage[64];
    size_t allocations = 0, releases = 0;
    const char* name() const override { return "shared-test"; }
    jittor::uint64 flags() const override { return _aligned; }
    int device() const override { return 9; }
    void* alloc(size_t size, size_t& token) override {
        token = 123 + ++allocations;
        return size ? storage : nullptr;
    }
    void free(void* ptr, size_t size, const size_t& token) override {
        CHECK(ptr == storage && size == 64 && token == 123 + allocations);
        ++releases;
    }
};
''', cpu_src=r'''
SharedTestAllocator raw;
jittor::SharedAllocator shared;
shared.setup(&raw);
CHECK(shared.flags() == raw.flags() && shared.device() == 9 && shared.can_share());
size_t token;
auto* ptr = static_cast<char*>(shared.alloc(64, token));
CHECK(shared.share_with(32, token));
CHECK(!shared.share_with(65, token));
shared.free(ptr, 64, token);
shared.gc();
CHECK(raw.releases == 0 && shared.used_memory == 64);
std::vector<std::thread> threads;
for (int i=0; i<4; ++i) threads.emplace_back([&]() {
    for (int j=0; j<100; ++j) {
        CHECK(shared.share_with(16, token));
        shared.free(ptr+16, 16, token);
    }
});
for (auto& thread : threads) thread.join();
CHECK(raw.releases == 0);
shared.free(ptr+32, 32, token);
CHECK(raw.releases == 1 && shared.used_memory == 0);
size_t next;
ptr = static_cast<char*>(shared.alloc(64, next));
CHECK(next != token);
shared.free(ptr, 64, next);
CHECK(raw.releases == 2);
CHECK(shared.alloc(0, next) == nullptr && next == 0);
CHECK(shared.share_with(0, next));
shared.free(nullptr, 0, next);
out0_p[0] = 1;
''')
        np.testing.assert_array_equal(result.numpy(), [1])


@pytest.mark.parametrize("backend", ["cpu", "npu"])
@pytest.mark.parametrize("flags", [
    {},
    {"use_sfrl_allocator": 0},
    {"use_nfef_allocator": 1},
    {"use_stat_allocator": 1, "use_sfrl_allocator": 0},
    {"use_stat_allocator": 2},
])
def test_reshape_alias_lifetime_on_selected_backend(backend, flags):
    from _helpers.capability import require_accelerator
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if backend == "npu":
        require_accelerator("acl")
    with jt.flag_scope(use_cuda=int(backend == "npu"), **flags), forbid_backend_fallbacks():
        for trial in range(4):
            expected = np.arange(8, dtype="float32") + trial * 100
            source = jt.array(expected) + 1.0
            source.sync()
            alias = source.reshape(2, 4)
            alias.sync()
            assert source.location() == ("device" if backend == "npu" else "cpu")
            assert alias.location() == source.location()
            assert alias._storage_address == source._storage_address
            # Observe aliasing before readback can migrate any storage.
            del source
            gc.collect()
            poison = [jt.array(np.full(8, -98765, "float32")) + 1.0 for _ in range(8)]
            jt.sync_all()
            np.testing.assert_array_equal(alias.numpy(), (expected + 1).reshape(2, 4))
            del poison, alias
            gc.collect()
