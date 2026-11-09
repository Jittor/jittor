"""Numeric and dtype contract for C++ extension half/bfloat16 scalars."""
from pathlib import Path
from _helpers.child_process import PYTHON
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest


class TestCppScalarTypes(unittest.TestCase):
    def test_half_bfloat16_conversion_and_dtype_mapping(self):
        compiler = shutil.which("g++")
        from jittor.compat.shim.cpp_extension import _find_pybind_include
        pybind_include = _find_pybind_include()
        if compiler is None or pybind_include is None:
            self.skipTest("requires g++ and pybind11 headers from optional real torch or pybind11")

        repo = Path(__file__).resolve().parents[3]
        includes = repo / "python/jittor/compat/shim/cpp_extension/include"
        source = r"""
#include <torch/extension.h>
#include <cstdio>
static_assert(sizeof(c10::Half) == 2);
static_assert(sizeof(c10::BFloat16) == 2);
static_assert(c10::CppTypeToScalarType<c10::Half>() == c10::ScalarType::Half);
static_assert(c10::CppTypeToScalarType<c10::BFloat16>() == c10::ScalarType::BFloat16);
static_assert(c10::CppTypeToScalarType<float>() == c10::ScalarType::Float);
int main() {
    for (float value : {0.1f, -0.0f, 65504.0f}) {
        c10::Half half(value);
        c10::BFloat16 bfloat(value);
        std::printf("%u %u %.10g %.10g\n",
                    unsigned(half.x), unsigned(bfloat.x),
                    float(half), float(bfloat));
    }
}
"""
        configured = sysconfig.get_config_var("LDLIBRARY")
        # Static-linked Python builds may still ship the embedding shared
        # library. LDLIBRARY alone can name an absent .a archive.
        version = "%d.%d" % sys.version_info[:2]
        libnames = [configured, "libpython%s.so" % version, "libpython%s.dylib" % version]
        libdirs = [Path(sysconfig.get_config_var("LIBDIR") or ""),
                   Path(sys.base_prefix) / "lib",
                   Path(PYTHON).resolve().parents[1] / "lib"]
        libpython = next((d / name for d in libdirs for name in libnames
                          if name and (d / name).is_file()), None)
        if libpython is None:
            self.skipTest("requires the Python shared library")
        libdir = libpython.parent
        include_candidates = [Path(sysconfig.get_path('include')),
                              libdir.parent / "include" / ("python%d.%d" % sys.version_info[:2])]
        python_include = next((p for p in include_candidates if (p / "Python.h").is_file()), None)
        if python_include is None:
            self.skipTest("requires Python development headers")
        with tempfile.TemporaryDirectory(prefix="jt-cpp-scalar-") as temp:
            cpp = Path(temp) / "probe.cpp"
            binary = Path(temp) / "probe"
            cpp.write_text(source)
            result = subprocess.run(
                [compiler, "-std=c++17", "-O1", f"-I{includes}",
                 f"-I{pybind_include}", f"-I{python_include}",
                 str(cpp), str(libpython), f"-Wl,-rpath,{libdir}",
                 "-o", str(binary)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.splitlines(), [
                "11878 15821 0.09997558594 0.1000976562",
                "32768 32768 -0 -0",
                "31743 18304 65504 65536",
            ])
