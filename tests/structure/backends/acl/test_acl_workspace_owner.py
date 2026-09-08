"""Execute the real workspace owner against a host-only CANN/allocator model."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]


def test_acl_workspace_is_device_owned_and_retryable(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the ACL workspace host contract")
    headers = {
        "core/common.h": r'''
#pragma once
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#define EXTERN_LIB
struct ThrowingLog {
    std::ostringstream stream;
    template<class T> ThrowingLog& operator<<(const T& value) {
        stream << value; return *this;
    }
    ~ThrowingLog() noexcept(false) { throw std::runtime_error(stream.str()); }
};
#define LOGf ThrowingLog()
#define ASSERT(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
''',
        "acl_runtime.h": r'''
#pragma once
#include "core/common.h"
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
int aclrtGetDevice(int32_t*);
int aclrtSetDevice(int);
int aclrtSynchronizeStream(aclrtStream);
namespace jittor {
aclrtStream acl_current_stream();
int acl_runtime_current_device();
}
''',
        "mem/allocator.h": r'''
#pragma once
#include "core/common.h"
namespace jittor {
struct Allocator {
    virtual int device() const = 0;
    virtual const char* name() const = 0;
    virtual void* alloc(size_t, size_t&) = 0;
    virtual void free(void*, size_t, const size_t&) = 0;
    virtual void gc() = 0;
    virtual ~Allocator() = default;
};
Allocator* get_allocator(int, bool);
}
''',
        "core/executor.h": r'''
#pragma once
#include "mem/allocator.h"
namespace jittor {
struct Executor { Allocator* temp_allocator = nullptr; };
Executor& runtime_executor();
}
''',
    }
    for name, source in headers.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    probe = tmp_path / "probe.cc"
    probe.write_text(r'''
#include "acl_workspace.h"
#include "acl_runtime.h"
#include "core/executor.h"
#include <cassert>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

int current = 0, sync_failure = -1, sdk_queries = 0;
std::vector<std::string> events;
int aclrtGetDevice(int32_t* device) { ++sdk_queries; *device = current; return 0; }
int aclrtSetDevice(int device) { current = device; return 0; }
int aclrtSynchronizeStream(aclrtStream stream) {
    assert(stream == reinterpret_cast<void*>(intptr_t(current + 1)));
    events.push_back("sync" + std::to_string(current));
    return current == sync_failure ? 7 : 0;
}
namespace jittor {
struct Pool : Allocator {
    int id;
    bool fail_alloc = false, fail_free = false, fail_gc = false;
    size_t next = 0;
    std::map<void*, std::pair<size_t, size_t>> blocks;
    explicit Pool(int id) : id(id) {}
    int device() const override { return id; }
    const char* name() const override { return "host-fake-acl"; }
    void* alloc(size_t size, size_t& allocation) override {
        assert(current == id);
        if (fail_alloc) throw std::runtime_error("allocation failure");
        auto* ptr = std::malloc(size);
        allocation = ++next;
        blocks[ptr] = {size, allocation};
        return ptr;
    }
    void free(void* ptr, size_t size, const size_t& allocation) override {
        assert(current == id);
        assert(acl_workspace_address() == nullptr);
        assert(blocks.at(ptr) == std::make_pair(size, allocation));
        events.push_back("free" + std::to_string(id));
        blocks.erase(ptr);
        std::free(ptr);
        if (fail_free) throw std::runtime_error("free failure");
    }
    void gc() override {
        events.push_back("gc" + std::to_string(id));
        if (fail_gc) throw std::runtime_error("gc failure");
    }
};
Pool first(0), second(1);
Executor& runtime_executor() { static Executor executor; return executor; }
Allocator* get_allocator(int device, bool temporary) {
    assert(temporary);
    return device == 0 ? &first : &second;
}
aclrtStream acl_current_stream() {
    return reinterpret_cast<void*>(intptr_t(current + 1));
}
int acl_runtime_current_device() { return current; }
}
void expect_error(const std::function<void()>& call, const std::string& message) {
    bool caught = false;
    try { call(); } catch (const std::runtime_error& error) {
        caught = true;
        assert(std::string(error.what()).find(message) != std::string::npos);
    }
    assert(caught);
}
int main() {
    using namespace jittor;
    release_all_acl_workspaces();
    assert(sdk_queries == 0);
    assert(mallocWorkSpace(0) == nullptr);
    runtime_executor().temp_allocator = &first;
    auto* p0 = mallocWorkSpace(17);
    assert(first.blocks.at(p0).first == 32);
    assert(mallocWorkSpace(31) == p0);
    assert(events.empty());
    current = 1;
    auto* p1 = mallocWorkSpace(50);
    assert(second.blocks.at(p1).first == 64);
    assert(p0 != p1 && acl_workspace_address() == p1);
    current = 0;
    assert(acl_workspace_address() == p0);
    sync_failure = 0;
    expect_error([] { mallocWorkSpace(80); }, "synchronization failed");
    assert(acl_workspace_address() == p0 && first.blocks.size() == 1);
    sync_failure = -1;
    first.fail_alloc = true;
    events.clear();
    expect_error([] { mallocWorkSpace(80); }, "allocation failure");
    assert(acl_workspace_address() == nullptr && first.blocks.empty());
    assert((events == std::vector<std::string>{"sync0", "free0", "gc0"}));
    first.fail_alloc = false;
    mallocWorkSpace(80);
    first.fail_free = true;
    expect_error([] { releaseWorkSpace(); }, "free failure");
    assert(acl_workspace_address() == nullptr && first.blocks.empty());
    first.fail_free = false;
    mallocWorkSpace(80);
    first.fail_gc = true;
    expect_error([] { releaseWorkSpace(); }, "gc failure");
    assert(acl_workspace_address() == nullptr && first.blocks.empty());
    first.fail_gc = false;
    expect_error([] { mallocWorkSpace(std::numeric_limits<uint64_t>::max()); }, "overflow");
    mallocWorkSpace(80);
    sync_failure = 0;
    current = 1;
    release_all_acl_workspaces();
    assert(current == 1);
    assert(first.blocks.size() == 1 && second.blocks.empty());
    sync_failure = -1;
    release_all_acl_workspaces();
    assert(current == 1 && first.blocks.empty() && second.blocks.empty());
    release_all_acl_workspaces();
}
''', encoding="utf-8")
    executable = tmp_path / "probe"
    result = subprocess.run(
        [compiler, "-std=c++14", "-I", str(tmp_path),
         "-I", str(ROOT / "backends/acl/include"),
         str(ROOT / "backends/acl/src/workspace.cc"), str(probe),
         "-o", str(executable)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
