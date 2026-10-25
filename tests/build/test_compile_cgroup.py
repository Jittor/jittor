# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms in LICENSE.txt.
"""Run directly to test compilation limits without importing/building Jittor."""

import ast
import contextlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
GIB = 1024 ** 3
HOST_MEMORY = 96 * GIB
spec = importlib.util.spec_from_file_location("cgroup_under_test", PYTHON_ROOT / "jittor/build/utils/cgroup.py")
cgroup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgroup)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux cgroup filesystem tests")
class TestCompileCgroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="jittor_cgroup_test_")
        cls.binary = Path(cls.build.name) / "probe"
        cls.compiler = shutil.which("g++") if sys.platform.startswith("linux") else None
        if not cls.compiler:
            return
        source = (ROOT / "src/core/parallel_compiler.cc").read_text()
        start = source.index("int parallel_compile_worker_count(")
        function = source[start:source.index("\n}\n", start)+3]
        # Redirect only /proc inputs; execute the actual dynamic worker cap.
        function = function.replace("limit_memory(mem_info.total_cpu_ram)",
                                    "limit_memory(mem_info.total_cpu_ram, groups, mounts)")
        harness = """#include <iostream>
#include <thread>
#include <limits>
#include <sched.h>
#include "utils/cgroup_memory.h"
using namespace jittor;
struct { int64_t total_cpu_ram; } mem_info;
const char* groups;
const char* mounts;
int cpu_limit;
int cgroup_cpu_limit() { return cpu_limit; }
"""
        harness += function.replace("parallel_compile_worker_count", "linux_threads")
        harness += "\n#undef __linux__\n"
        harness += "int other_cpu_limit() { return std::numeric_limits<int>::max(); }\n"
        harness += function.replace("parallel_compile_worker_count", "other_platform_threads").replace(
            "cgroup_cpu_limit()", "other_cpu_limit()")
        harness += """
int main(int argc, char** argv) {
    mem_info.total_cpu_ram = std::stoll(argv[1]);
    groups = argv[2];
    mounts = argv[3];
    int cap = std::stoi(argv[4]);
    cpu_limit = argc > 5 ? std::stoi(argv[5]) : 128;
    std::cout << cgroup_memory::limit_memory(mem_info.total_cpu_ram, groups, mounts) << " "
              << linux_threads(cap) << " "
              << other_platform_threads(cap);
    if (argc > 7) {
        std::ofstream limit_file(argv[6]);
        limit_file << argv[7];
        limit_file.close();
        std::cout << " " << linux_threads(cap);
    }
    std::cout << std::endl;
}
"""
        cpp = Path(cls.build.name) / "probe.cc"
        cpp.write_text(harness)
        subprocess.run([cls.compiler, "-std=c++14", "-O0", "-Wall", "-Wextra",
                        "-Wno-unused-parameter", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(cls.binary)], check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cgroup_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.groups = self.root / "cgroup"
        self.mountinfo = self.root / "mountinfo"
        self.mount = self.root / "memory mount"
        self.mount.mkdir()

    def configure(self, kind="cgroup2", group="/job", mount_root="/", extra=""):
        membership = "0::" if kind == "cgroup2" else "7:memory,other:"
        self.groups.write_text(membership + group + "\n" + extra)
        options = "rw" if kind == "cgroup2" else "rw,memory,other"
        escaped = str(self.mount).replace("\\", "\\134").replace(" ", "\\040")
        self.mountinfo.write_text("1 0 0:1 {} {} rw shared:1 - {} cgroup {}\n".format(
            mount_root, escaped, kind, options))

    def write(self, relative, value):
        path = self.mount / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value))

    def check_limit(self, expected, cap=16, cpu_limit=128):
        self.assertEqual(cgroup.limit_memory(HOST_MEMORY, self.groups, self.mountinfo), expected)
        if self.compiler:
            output = subprocess.check_output([str(self.binary), str(HOST_MEMORY),
                str(self.groups), str(self.mountinfo), str(cap), str(cpu_limit)], text=True, timeout=5)
            memory, linux_threads, other_threads = map(int, output.split())
            self.assertEqual(memory, expected)
            self.assertEqual(linux_threads, max(1, min(cap, cpu_limit, len(os.sched_getaffinity(0)), expected // (3 * GIB))))
            self.assertEqual(other_threads, max(1, min(cap, os.cpu_count())))

    def test_v2_leaf_and_parent_limit(self):
        self.configure(group="/parent/job")
        self.write("parent/job/memory.max", 9 * GIB)
        self.write("parent/memory.max", 2 * GIB)
        self.write("memory.max", "max")
        self.check_limit(2 * GIB)

    def test_v1_combined_controller_mount(self):
        self.configure(kind="cgroup", group="/parent/job")
        self.write("parent/job/memory.limit_in_bytes", 9 * GIB)
        self.write("parent/memory.limit_in_bytes", 2 * GIB)
        self.write("parent/memory.use_hierarchy", 1)
        self.write("memory.limit_in_bytes", 9223372036854771712)
        self.write("memory.use_hierarchy", 1)
        self.check_limit(2 * GIB)

    def test_v1_nonhierarchical_parent_does_not_limit_child(self):
        self.configure(kind="cgroup", group="/parent/job")
        self.write("parent/job/memory.limit_in_bytes", 9 * GIB)
        self.write("parent/memory.limit_in_bytes", 2 * GIB)
        self.write("parent/memory.use_hierarchy", 0)
        self.check_limit(9 * GIB)

    def test_mount_root_matches_container(self):
        self.configure(group="/docker/container/job", mount_root="/docker/container")
        self.write("job/memory.max", 9 * GIB)
        self.write("memory.max", 6 * GIB)
        self.check_limit(6 * GIB)

    def test_cgroup_namespace_root(self):
        # Membership and mount roots are both relative to the cgroup namespace.
        for kind in ("cgroup", "cgroup2"):
            with self.subTest(kind=kind):
                self.configure(kind=kind, group="/", mount_root="/")
                name = "memory.max" if kind == "cgroup2" else "memory.limit_in_bytes"
                self.write(name, 2 * GIB)
                self.check_limit(2 * GIB)

    def test_namespace_root_ignores_child_bind_mount(self):
        child_mount = self.root / "child_mount"
        child_mount.mkdir()
        for kind in ("cgroup", "cgroup2"):
            self.configure(kind=kind, group="/", mount_root="/")
            name = "memory.max" if kind == "cgroup2" else "memory.limit_in_bytes"
            self.write(name, 24 * GIB)
            self.write("unrelated/" + name, 2 * GIB)
            (child_mount / name).write_text(str(2 * GIB))
            own_record = self.mountinfo.read_text()
            options = "rw" if kind == "cgroup2" else "rw,memory,other"
            child_record = "2 0 0:1 /unrelated {} rw - {} cgroup {}\n".format(
                child_mount, kind, options)
            for child_first in (False, True):
                with self.subTest(kind=kind, child_first=child_first):
                    records = (child_record + own_record if child_first
                               else own_record + child_record)
                    self.mountinfo.write_text(records)
                    self.check_limit(24 * GIB)

    def test_namespace_root_with_only_unrelated_mount_falls_back(self):
        for kind in ("cgroup", "cgroup2"):
            with self.subTest(kind=kind):
                self.configure(kind=kind, group="/", mount_root="/unrelated")
                name = "memory.max" if kind == "cgroup2" else "memory.limit_in_bytes"
                self.write(name, 2 * GIB)
                self.check_limit(HOST_MEMORY)

    def test_unrelated_mount_is_ignored(self):
        self.configure(group="/docker/container2", mount_root="/docker/container")
        self.write("memory.max", 2 * GIB)
        self.check_limit(HOST_MEMORY)

    def test_outside_namespace_is_ignored(self):
        self.configure(group="/../other")
        self.write("memory.max", 2 * GIB)
        self.check_limit(HOST_MEMORY)

    def test_unlimited_invalid_and_larger_than_host(self):
        for value in ("max", "-1", "", "not-a-number", "2048 garbage", "9" * 100,
                      str(9223372036854771712), str(HOST_MEMORY * 2)):
            with self.subTest(value=value):
                self.configure()
                self.write("job/memory.max", value)
                self.check_limit(HOST_MEMORY)

    def test_missing_files_fall_back(self):
        self.check_limit(HOST_MEMORY)
        self.configure()
        self.check_limit(HOST_MEMORY)

    def test_unreadable_leaf_still_checks_parent(self):
        self.configure()
        (self.mount / "job/memory.max").mkdir(parents=True)
        self.write("memory.max", 6 * GIB)
        self.check_limit(6 * GIB)

    def test_zero_limit_preserves_one_worker_minimum(self):
        self.configure()
        self.write("job/memory.max", 0)
        self.check_limit(0)

    def test_existing_user_cap_is_preserved(self):
        self.configure()
        self.write("job/memory.max", 24 * GIB)
        self.check_limit(24 * GIB, cap=2)

    def test_native_worker_count_rechecks_changed_limit(self):
        if not self.compiler:
            self.skipTest("C++ compiler required")
        if len(os.sched_getaffinity(0)) < 2:
            self.skipTest("At least two allowed CPUs required to observe changed worker cap")
        self.configure()
        self.write("job/memory.max", 0)
        output = subprocess.check_output([str(self.binary), str(HOST_MEMORY),
            str(self.groups), str(self.mountinfo), "16", "128",
            str(self.mount / "job/memory.max"), str(24 * GIB)], text=True, timeout=5)
        memory, initial, other, changed = map(int, output.split())
        self.assertEqual(memory, 0)
        self.assertEqual(initial, 1)
        self.assertEqual(changed, min(8, len(os.sched_getaffinity(0))))
        self.assertGreater(changed, initial)

    def test_cpu_quota_is_preserved(self):
        self.configure()
        self.write("job/memory.max", 24 * GIB)
        self.check_limit(24 * GIB, cpu_limit=2)

    def test_hybrid_uses_memory_controller(self):
        self.configure(kind="cgroup", extra="0::/unified\n")
        self.write("job/memory.limit_in_bytes", 2 * GIB)
        self.check_limit(2 * GIB)

    def test_python_dispatch_uses_cgroup_limit_and_preserves_nonlinux(self):
        self.configure()
        self.write("job/memory.max", 2 * GIB)
        tree = ast.parse((PYTHON_ROOT / "jittor/build/utils/__init__.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("get_total_mem", "run_cmds")]
        pool_sizes = []
        class Pool:
            def __init__(self, size, **kwargs):
                pool_sizes.append(size)
            def __enter__(self):
                return self
            def imap_unordered(self, func, args):
                return map(func, args)
        module = ModuleType("compile_memory_probe.cgroup")
        module.limit_memory = lambda memory: cgroup.limit_memory(memory, self.groups, self.mountinfo)
        namespace = dict(__package__="compile_memory_probe", Any=object,
            _main_module_not_reexecuted=contextlib.nullcontext, os=SimpleNamespace(name="posix",
            sysconf=lambda name: 4096 if name == "SC_PAGE_SIZE" else HOST_MEMORY // 4096, environ={}),
            sys=SimpleNamespace(platform="linux"),
            platform=SimpleNamespace(system=lambda: "Linux"), Pool=Pool, pool_size=0,
            mp=SimpleNamespace(current_process=lambda: SimpleNamespace(_config={})),
            pool_initializer=lambda: None, pool_cleanup=lambda: None,
            LOG=SimpleNamespace(i=lambda message: None), do_compile=lambda args: args,
            DelayProgress=lambda *args: SimpleNamespace(update=lambda i: None))
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<compiler-functions>", "exec"), namespace)
        with mock.patch.dict(sys.modules, {module.__name__: module}), mock.patch("atexit.register"):
            self.assertEqual(namespace["get_total_mem"](), 2 * GIB)
            namespace["run_cmds"](["compile-a", "compile-b"], "unused", "unused")
            self.assertEqual(pool_sizes, [1])
            namespace["platform"].system = lambda: "Darwin"
            self.assertEqual(namespace["get_total_mem"](), HOST_MEMORY)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc probe")
    def test_live_python_and_cpp_agree(self):
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        limited = cgroup.limit_memory(physical)
        self.assertGreaterEqual(limited, 0)
        self.assertLessEqual(limited, physical)
        if self.compiler:
            output = subprocess.check_output([str(self.binary), str(physical),
                "/proc/self/cgroup", "/proc/self/mountinfo", "16"], text=True, timeout=5)
            self.assertEqual(int(output.split()[0]), limited)


if __name__ == "__main__":
    unittest.main()
