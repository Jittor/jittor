"""ACL BackendOps must publish an ACL-owned allocator, not CUDA pools."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "backends/acl/src/backend.cc"


def test_acl_backend_uses_acl_allocator_adapter():
    text = SOURCE.read_text(encoding="utf-8")
    assert "class AclAllocator final : public Allocator" in text
    assert "Allocator* acl_allocator(int device, BackendMemoryKind kind)" in text
    assert "ops.allocator = acl_allocator;" in text
    assert "ops.allocator = accelerator_allocator_for;" not in text
    assert "ACL allocator does not support managed memory" in text


def test_acl_allocator_preserves_device_identity_and_pinned_host_semantics():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("class AclAllocator final")
    end = text.index("Allocator* acl_allocator", start)
    body = text[start:end]
    assert "kind_ == BackendMemoryKind::Pinned ? 0 : Allocator::_cuda" in body
    assert "kind_ == BackendMemoryKind::Pinned ? -1 : device_" in body
    assert "return allocate_memory(device_, kind_, size);" in body
    assert "free_memory(device_, kind_, pointer);" in body


def test_p2p_refusal_is_device_local_and_only_retries_allocation_errors(tmp_path):
    """Execute the allocator against injected ACL statuses; no device claim."""
    import os
    import shlex
    import subprocess

    text = SOURCE.read_text()
    begin = text.index('namespace {\n// 207001')
    end = text.index('\nvoid free_memory(', begin)
    implementation = text[begin:end]
    prefix = r'''
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>
using aclError = int;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST_P2P = 1;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 2;
enum class BackendMemoryKind { Device, Pinned, Managed };
struct MemoryRecord { int device; BackendMemoryKind kind; };
struct State { std::recursive_mutex mutex; std::map<void*, MemoryRecord> memory; };
State& state() { static State s; return s; }
std::atomic<int> ready_device_count{2};
struct Check { template<class T> Check& operator<<(T) { return *this; } };
#define USER_CHECK(x) ((x) ? Check{} : throw std::runtime_error("invalid kind"))
template<class F> void on_device(int, F f) { f(); }
void check_acl(int code, const char*) { if (code) throw code; }
void report_acl(int code, const char* msg) { check_acl(code, msg); }
int aclrtFreeHost(void*) { return 0; }
int aclrtFree(void*) { return 0; }
std::vector<int> policies;
int p2p_error = 0, host_error = 0, hbm_error = 0;
int aclrtMalloc(void** p, size_t, int policy) {
    policies.push_back(policy);
    *p = reinterpret_cast<void*>(static_cast<uintptr_t>(policies.size() * 32));
    return policy == ACL_MEM_MALLOC_HUGE_FIRST_P2P ? p2p_error : hbm_error;
}
int aclrtMallocHost(void** p, size_t) {
    policies.push_back(3); *p = nullptr; return host_error;
}
'''
    suffix = r'''
int main(int argc, char** argv) {
    if (argc == 2) {
        setenv("JT_ACL_DISABLE_P2P", argv[1], 1);
        allocate_memory(0, BackendMemoryKind::Device, 1024);
        int expected = argv[1][0] == '1' ? 2 : 1;
        assert((policies == std::vector<int>{expected}));
        return 0;
    }
    unsetenv("JT_ACL_DISABLE_P2P");
    p2p_error = 207001;
    allocate_memory(0, BackendMemoryKind::Device, 1024);
    assert((policies == std::vector<int>{1, 2}));
    policies.clear();
    allocate_memory(0, BackendMemoryKind::Device, 1024);
    assert((policies == std::vector<int>{2}));
    policies.clear(); p2p_error = 0;
    allocate_memory(1, BackendMemoryKind::Device, 1024);
    assert((policies == std::vector<int>{1}));
    policies.clear(); p2p_error = 507033;
    try { allocate_memory(2, BackendMemoryKind::Device, 1024); assert(false); }
    catch (int error) { assert(error == 507033); }
    assert((policies == std::vector<int>{1}));
    assert(!p2p_refused_on(2));
    policies.clear(); host_error = 207001;
    try { allocate_memory(3, BackendMemoryKind::Pinned, 1024); assert(false); }
    catch (int error) { assert(error == 207001); }
    assert((policies == std::vector<int>{3}));
    assert(!p2p_refused_on(3));
    policies.clear(); p2p_error = 207001; hbm_error = 42;
    try { allocate_memory(4, BackendMemoryKind::Device, 1024); assert(false); }
    catch (int error) { assert(error == 42); }
    assert((policies == std::vector<int>{1, 2}));
    policies.clear();
    assert(allocate_memory(0, BackendMemoryKind::Device, 0) == nullptr);
    assert(policies.empty());
}
'''
    source = tmp_path / 'allocator.cc'
    binary = tmp_path / 'allocator'
    source.write_text(prefix + implementation + suffix)
    subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17',
                    str(source), '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
    for value in ('', '0', '1'):
        subprocess.run([str(binary), value], check=True)
