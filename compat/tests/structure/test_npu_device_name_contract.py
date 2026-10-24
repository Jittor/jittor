"""Host-only provider loading, device routing and native query contracts."""
import ast
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


class TestNPUDeviceNameContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.initializer = types.SimpleNamespace(init_acl_ops=lambda: self.calls.append("init"))
        self.config = types.SimpleNamespace(has_acl=True, cc_flags="fixture flags",
                                          resources={"acl_initializer": self.initializer})
        self.native = types.SimpleNamespace(
            acl_device_name=lambda index: self.calls.append(("query", index)) or "driver-device-%d" % index)
        self.rng_names = ("acl_get_rng_state", "acl_set_rng_state", "acl_get_rng_state_all",
                          "acl_set_rng_state_all", "acl_manual_seed", "acl_manual_seed_all", "acl_initial_seed")
        self.native_rng = types.SimpleNamespace(**{name: mock.Mock(name=name) for name in self.rng_names})
        def compile_binding(source, flags):
            if "@pyjt(acl_device_name)" in source:
                return self.native
            self.assertIn("@pyjt(acl_get_rng_state)", source)
            for name in self.rng_names:
                self.assertIn("@pyjt(" + name + ")", source)
            return self.native_rng
        self.compile = mock.Mock(side_effect=compile_binding)
        self.context = types.SimpleNamespace(config=self.config, native_core=object(),
                                             compile_module=self.compile)
        tree = ast.parse((ROOT / "backends/acl/__init__.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("post_process", "get_device_name")]
        self.provider = {"BuildContext": object}
        exec(compile(tree, "acl_provider_contract", "exec"), self.provider)
        self.compiler = types.SimpleNamespace(build_config=self.config, has_acl=True)
        self.jt = types.SimpleNamespace(compiler=self.compiler,
            core=types.SimpleNamespace(current_device=lambda: 1, get_device_count=lambda: 2))
        memory_tree = ast.parse((ROOT / "compat/torch/installers/cuda/npu_memory.py").read_text())
        memory_tree.body = [node for node in memory_tree.body if not (
            isinstance(node, ast.Import) and any(alias.name == "jittor" for alias in node.names))]
        self.memory = {"jt": self.jt}
        exec(compile(memory_tree, "npu_memory_contract", "exec"), self.memory)
        tree = ast.parse((ROOT / "compat/torch/installers/cuda/npu_device.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.api = {"_device_index": self.memory["_device_index"]}
        exec(compile(tree, "npu_device_contract", "exec"), self.api)
        build = types.ModuleType("jittor.build")
        build.compiler = self.compiler
        acl = types.ModuleType("jittor.backends.acl")
        acl.get_device_name = self.provider["get_device_name"]
        patches = mock.patch.dict(sys.modules, {"jittor.build": build, "jittor.backends.acl": acl})
        patches.start()
        self.addCleanup(patches.stop)

    def test_post_core_binding_is_published_once(self):
        self.assertFalse(hasattr(self.initializer, "_device_info"))
        self.provider["post_process"](self.context)
        self.provider["post_process"](self.context)
        self.assertIs(self.initializer._device_info, self.native)
        self.assertIs(self.initializer._rng_info, self.native_rng)
        self.assertEqual(self.compile.call_count, 2)
        source, flags = self.compile.call_args_list[0].args
        self.assertIn("@pyjt(acl_device_name)", source)
        self.assertEqual(flags, self.config.cc_flags)
        self.initializer._device_info = object()
        with self.assertRaisesRegex(RuntimeError, "resource is invalid"):
            self.provider["post_process"](self.context)
        self.assertEqual(self.compile.call_count, 2)

    def test_rng_binding_missing_any_interface_is_not_published(self):
        for name in self.rng_names:
            with self.subTest(name=name):
                original = getattr(self.native_rng, name)
                delattr(self.native_rng, name)
                try:
                    with self.assertRaisesRegex(RuntimeError, "ACL RNG binding is incomplete"):
                        self.provider["post_process"](self.context)
                    self.assertFalse(hasattr(self.initializer, "_rng_info"))
                    self.assertIs(self.initializer._device_info, self.native)
                finally:
                    setattr(self.native_rng, name, original)
        self.provider["post_process"](self.context)
        self.assertIs(self.initializer._rng_info, self.native_rng)
        calls = self.compile.call_count
        self.initializer._rng_info = object()
        with self.assertRaisesRegex(RuntimeError, "ACL RNG resource is invalid"):
            self.provider["post_process"](self.context)
        self.assertEqual(self.compile.call_count, calls)

    def test_rng_compile_failure_propagates_without_publishing(self):
        self.initializer._device_info = self.native
        self.compile.side_effect = RuntimeError("RNG binding failed")
        with self.assertRaisesRegex(RuntimeError, "RNG binding failed"):
            self.provider["post_process"](self.context)
        self.assertFalse(hasattr(self.initializer, "_rng_info"))
        self.assertIs(self.initializer._device_info, self.native)

    def test_missing_core_and_resources_fail_without_compilation(self):
        self.context.native_core = None
        with self.assertRaisesRegex(RuntimeError, "loaded native core"):
            self.provider["post_process"](self.context)
        self.context.native_core = object()
        self.config.resources.clear()
        with self.assertRaisesRegex(RuntimeError, "initializer"):
            self.provider["post_process"](self.context)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.provider["get_device_name"](0)
        self.compile.assert_not_called()

    def test_failed_compile_is_not_published_or_hidden(self):
        self.compile.side_effect = RuntimeError("binding failed")
        with self.assertRaisesRegex(RuntimeError, "binding failed"):
            self.provider["post_process"](self.context)
        self.assertFalse(hasattr(self.initializer, "_device_info"))
        self.compile.side_effect = None
        self.compile.return_value = object()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.provider["post_process"](self.context)
        self.assertFalse(hasattr(self.initializer, "_device_info"))

    def test_public_npu_parameters_reach_the_selected_provider_index(self):
        self.provider["post_process"](self.context)
        for argument, expected in ((None, 1), (0, 0), ("npu:0", 0), ("npu", 1),
                                   (types.SimpleNamespace(type="npu", index=0), 0)):
            with self.subTest(argument=argument):
                self.assertEqual(self.api["get_device_name"](argument), "driver-device-%d" % expected)
                self.assertEqual(self.calls[-1], ("query", expected))
        for argument in (True, False, -1, 2, "cpu", "cuda:0", 0.5):
            before = list(self.calls)
            with self.subTest(argument=argument), self.assertRaises((ValueError, TypeError)):
                self.api["get_device_name"](argument)
            self.assertEqual(self.calls, before)

    def test_unavailable_backend_and_native_error_are_not_fake_names(self):
        self.compiler.has_acl = self.config.has_acl = False
        self.provider["post_process"](self.context)
        self.compile.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "ACL"):
            self.api["get_device_name"]()
        with self.assertRaisesRegex(RuntimeError, "ACL"):
            self.provider["get_device_name"](0)
        self.compiler.has_acl = self.config.has_acl = True
        self.provider["post_process"](self.context)
        self.native.acl_device_name = mock.Mock(side_effect=RuntimeError("driver name unavailable"))
        with self.assertRaisesRegex(RuntimeError, "driver name unavailable"):
            self.api["get_device_name"](0)

    def test_configure_does_not_bind_an_unloaded_core_symbol(self):
        tree = ast.parse((ROOT / "backends/acl/__init__.py").read_text())
        configure = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "configure")
        self.assertNotIn("acl_device_name", ast.dump(configure))
        bindings = (ROOT / "compat/torch/installers/cuda/bindings.py").read_text()
        self.assertIn("g.npu.get_device_name = npu_get_device_name", bindings)

    def test_native_query_preserves_driver_device_on_success_and_failure(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++ compiler is required for the standalone native contract")
        source = (ROOT / "backends/acl/src/backend.cc").read_text()
        guard = re.search(r"template<class Function>\nvoid on_device\(.*?\n\}", source, re.S)
        query = re.search(r"string acl_device_name\(int device\) \{.*?\n\}", source, re.S)
        self.assertIsNotNone(guard)
        self.assertIsNotNone(query)
        native = r'''
#include <cassert>
#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
using std::string;
int selected = 0, query_device = -1, failure = 0, factory_mode = 0;
struct State { std::recursive_mutex mutex; std::set<int> touched; } owner;
State& state() { return owner; }
void validate_device(int device) {
    if (device < 0 || device >= 2) throw std::runtime_error("invalid device");
}
int aclrtGetDevice(int32_t* result) { *result = selected; return 0; }
int aclrtSetDevice(int device) { selected = device; return 0; }
void check_acl(int status, const char*) { if (status) throw std::runtime_error("driver error"); }
void report_acl(int status, const char*) { assert(status == 0); }
const char* aclrtGetSocName() {
    query_device = selected;
    if (failure == 1) return nullptr;
    if (failure == 2) return "";
    if (failure == 3) throw std::runtime_error("driver failure");
    return selected == 0 ? "real-driver-zero" : "real-driver-one";
}
''' + guard.group(0) + "\n" + query.group(0) + r'''
int main() {
    assert(acl_device_name(1) == "real-driver-one");
    assert(query_device == 1 && selected == 0 && factory_mode == 0);
    assert(acl_device_name(0) == "real-driver-zero");
    assert(query_device == 0 && selected == 0);
    for (failure = 1; failure <= 3; ++failure) {
        bool failed = false;
        try { acl_device_name(1); } catch (const std::runtime_error&) { failed = true; }
        assert(failed && query_device == 1 && selected == 0 && factory_mode == 0);
    }
    for (int device : {-1, 2}) {
        bool failed = false;
        try { acl_device_name(device); } catch (const std::runtime_error&) { failed = true; }
        assert(failed && selected == 0);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="acl-device-name-host-") as directory:
            executable = str(Path(directory) / "query")
            result = subprocess.run([compiler, "-std=c++14", "-x", "c++", "-", "-o", executable],
                                    input=native, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([executable], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
