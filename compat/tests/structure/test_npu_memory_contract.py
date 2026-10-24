"""Host-only contracts: no Jittor import, JIT, or claim of hardware coverage."""
import ast
import re
from pathlib import Path
import types
import unittest
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "compat/torch/installers/cuda/npu_memory.py"


class TestNPUMemoryContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.core = types.SimpleNamespace(
            current_device=lambda: 1,
            get_device_count=lambda: 2,
            device_pool_memory_stats=lambda index: self.calls.append(("stats", index)) or
                [100 + index, 200 + index, 300 + index, 400 + index],
            device_memory_info=lambda index: self.calls.append(("driver", index)) or [700, 1000],
            reset_device_memory_peaks=lambda index: self.calls.append(("reset", index)),
            gc_device_cache=lambda index: self.calls.append(("device_gc", index)),
        )
        self.jt = types.SimpleNamespace(
            core=self.core, compiler=types.SimpleNamespace(has_acl=True),
            flags=types.SimpleNamespace(use_cuda=0),
            gc=lambda: self.calls.append(("gc",)),
        )
        tree = ast.parse(SOURCE.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Import) and any(alias.name == "jittor" for alias in node.names))]
        self.api = {"jt": self.jt}
        exec(compile(tree, str(SOURCE), "exec"), self.api)

    def test_live_and_distinct_native_peaks(self):
        for name, value in (("memory_allocated", 101), ("memory_reserved", 201),
                            ("max_memory_allocated", 301), ("max_memory_reserved", 401)):
            self.assertEqual(self.api[name](), value)
        self.assertEqual(self.calls, [("stats", 1)] * 4)
        self.assertEqual(self.jt.flags.use_cuda, 0)

    def test_strict_device_selection(self):
        for device, expected in ((0, 0), ("npu:0", 0), ("npu", 1),
                                 (types.SimpleNamespace(type="npu", index=0), 0),
                                 (types.SimpleNamespace(type="npu", index=None), 1)):
            self.assertEqual(self.api["_device_index"](device), expected)
        for device in (True, False, -1, 2, 0.5, "cuda:0", "cpu", "npu:-1", "npu:0junk",
                       types.SimpleNamespace(type="cuda", index=0)):
            with self.subTest(device=device), self.assertRaises((ValueError, TypeError)):
                self.api["memory_allocated"](device)

    def test_driver_info_is_not_total_minus_process_pool(self):
        self.assertEqual(self.api["mem_get_info"]("npu:0"), (700, 1000))
        self.assertEqual(self.calls, [("driver", 0)])
        self.assertEqual(self.core.current_device(), 1)

    def test_no_backend_no_fake_zero(self):
        self.jt.compiler.has_acl = False
        with self.assertRaisesRegex(RuntimeError, "ACL"):
            self.api["memory_allocated"]()

    def test_native_failures_propagate(self):
        def fail(index):
            raise RuntimeError("allocator unsupported")
        self.core.device_pool_memory_stats = fail
        with self.assertRaisesRegex(RuntimeError, "allocator unsupported"):
            self.api["max_memory_allocated"]()
        self.core.device_memory_info = fail
        with self.assertRaisesRegex(RuntimeError, "allocator unsupported"):
            self.api["mem_get_info"]()

    def test_reset_and_cache_are_real_native_calls(self):
        self.api["reset_peak_memory_stats"](0)
        self.api["reset_max_memory_allocated"](1)
        self.api["empty_cache"]()
        self.assertEqual(self.calls, [("reset", 0), ("reset", 1), ("device_gc", 1)])

    def test_cache_errors_propagate(self):
        def fail(index):
            raise RuntimeError("device cache release failed")
        self.core.gc_device_cache = fail
        with self.assertRaisesRegex(RuntimeError, "device cache release failed"):
            self.api["empty_cache"]()

    def test_public_namespace_identity(self):
        npu, modules = types.SimpleNamespace(), {}
        self.api["install"](npu, modules)
        self.assertIs(npu.memory, modules["torch.npu.memory"])
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated",
                     "max_memory_reserved", "mem_get_info", "reset_peak_memory_stats", "empty_cache"):
            self.assertIs(getattr(npu, name), self.api[name])
            self.assertIs(getattr(npu.memory, name), self.api[name])

    def _run_native_contract(self, source):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++ compiler is required for the standalone ledger contract")
        with tempfile.TemporaryDirectory(prefix="jittor-pool-memory-host-") as directory:
            executable = str(Path(directory) / "ledger_test")
            result = subprocess.run(
                [compiler, "-std=c++14", "-pthread", "-I", str(ROOT / "src"),
                 "-x", "c++", "-", "-o", executable], input=source,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_device_collection_scope_without_jittor_jit(self):
        # Execute the actual implementation with observable fake pools, not a
        # copied selection algorithm. This is a host contract, not an ACL test.
        source = (ROOT / "src/mem/allocator.cc").read_text()
        function = re.search(r"void gc_device_cache\(int device\) \{.*?\n\}", source, re.S)
        self.assertIsNotNone(function)
        self._run_native_contract(r'''
#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
using std::vector;
std::mutex allocators_mutex;
struct Check {
    Check(bool value) { if (!value) throw std::runtime_error("invalid device"); }
    template<class T> Check& operator<<(const T&) { return *this; }
};
#define USER_CHECK(value) Check(bool(value))
int get_device_count() { return 2; }
struct Allocator {
    bool accelerator, fail = false;
    int index, calls = 0, live = 100, cached = 200;
    Allocator(bool accelerator, int index) : accelerator(accelerator), index(index) {}
    bool is_cuda() const { return accelerator; }
    int device() const { return index; }
    void gc() {
        // Reclamation must occur outside the registry lock.
        assert(allocators_mutex.try_lock());
        allocators_mutex.unlock();
        ++calls;
        if (fail) throw std::runtime_error("release failed");
        cached = 0;
    }
};
std::map<int, std::unique_ptr<Allocator>> allocators;
''' + function.group(0) + r'''
int main() {
    allocators[0] = std::make_unique<Allocator>(false, -1); // host
    allocators[1] = std::make_unique<Allocator>(false, 0);  // host with device-like index
    allocators[2] = std::make_unique<Allocator>(true, 0);
    allocators[3] = std::make_unique<Allocator>(true, 1);
    gc_device_cache(1);
    for (int i = 0; i < 4; ++i) {
        assert(allocators[i]->calls == (i == 3 ? 1 : 0));
        assert(allocators[i]->cached == (i == 3 ? 0 : 200));
        assert(allocators[i]->live == 100);
    }
    for (int invalid : {-1, 2}) {
        bool failed = false;
        try { gc_device_cache(invalid); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    assert(allocators[2]->calls == 0 && allocators[3]->calls == 1);
    allocators[2]->fail = true;
    bool failed = false;
    try { gc_device_cache(0); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    assert(allocators[2]->calls == 1 && allocators[3]->calls == 1);
}
''')

    def test_native_event_ledger_without_jittor_jit(self):
        source = (ROOT / "src/tests/test_pool_memory_stats.cc").read_text()
        source = source.replace('#include "codegen/opt/kernel_ir.h"', '''
#include <cassert>
#define JIT_TEST(name) void name()
#define ASSERT(condition) assert(condition)
#define ASSERTop(left, op, right) assert((left) op (right))
''')
        source += "\nint main() { jittor::pool_memory_event_peaks(); }\n"
        self._run_native_contract(source)


if __name__ == "__main__":
    unittest.main()
