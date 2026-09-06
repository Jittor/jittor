"""ROCm provider build/layout contracts without Jittor or HIP initialization."""

import importlib.util
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROCM = ROOT / "backends" / "rocm"


def load_build():
    spec = importlib.util.spec_from_file_location("rocm_library_build_test", ROCM / "build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def sdk_fixture(tmp_path):
    sdk = tmp_path / "ROCm SDK"
    for header in ("hipblas/hipblas.h", "rocprim/device/device_scan.hpp"):
        path = sdk / "include" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (sdk / "lib64").mkdir()
    (sdk / "lib64" / "libhipblas.so").touch()
    return sdk


def test_specs_resolve_only_native_source_and_quote_paths(tmp_path):
    build = load_build()
    sdk = sdk_fixture(tmp_path)
    blas = build.library_build(sdk, ROCM, "hipblas")
    scan = build.library_build(sdk, ROCM, "rocprim")
    assert blas.libraries == ("hipblas",)
    assert blas.library_dirs == (str(sdk / "lib64"),)
    assert scan.libraries == scan.library_dirs == ()
    assert "-I" + str(sdk / "include") in shlex.split(blas.extra_flags)
    assert "-Wl,-rpath," + str(sdk / "lib64") in shlex.split(blas.extra_flags)
    for spec in (blas, scan):
        assert spec.sources
        assert all(Path(path).is_file() for path in spec.sources)
        assert all(ROCM / "libraries" in Path(path).parents for path in spec.sources)
        assert all("cuda" not in path for path in spec.sources)
    assert any(path.endswith("scan.cu") for path in scan.sources)
    with pytest.raises(AttributeError):
        blas.name = "cublas"


def test_missing_sdk_and_unknown_provider_fail_before_build(tmp_path):
    build = load_build()
    with pytest.raises(RuntimeError, match="development header"):
        build.library_build(tmp_path, ROCM, "hipblas")
    sdk = sdk_fixture(tmp_path)
    (sdk / "lib64" / "libhipblas.so").unlink()
    with pytest.raises(RuntimeError, match="libhipblas.so"):
        build.library_build(sdk, ROCM, "hipblas")
    for name in ("MIOpen", "rccl", "cublas", "cub"):
        with pytest.raises(NotImplementedError, match="no native ROCm library provider"):
            build.library_build(sdk, ROCM, name)


def load_installer(monkeypatch):
    package = ModuleType("rocm_provider_test")
    package.__path__ = [str(ROCM)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".build", load_build())
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".libraries.install", ROCM / "libraries" / "install.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_install_publishes_real_providers_only_after_both_compile(tmp_path, monkeypatch):
    installer = load_installer(monkeypatch)
    sdk = sdk_fixture(tmp_path)
    calls, published = [], []

    def compile_ops(sources, **kwargs):
        assert not published
        assert kwargs["backend"] == "accelerator"
        assert kwargs["return_module"] is True
        calls.append((sources, kwargs))
        return object()

    context = SimpleNamespace(
        config=SimpleNamespace(resources={"rocm_home": str(sdk)}),
        compile_custom_ops=compile_ops,
        publish_library=lambda name, module: published.append((name, module)))
    result = installer.install_libraries(context)
    assert len(calls) == 2
    assert tuple(published) == result
    assert [name for name, _ in published] == ["hipblas", "rocprim"]


def test_compilation_failure_is_not_published_or_swallowed(tmp_path, monkeypatch):
    installer = load_installer(monkeypatch)
    sdk = sdk_fixture(tmp_path)
    calls, published = [], []

    def compile_ops(sources, **kwargs):
        calls.append(sources)
        if len(calls) == 2:
            raise RuntimeError("HIP compiler failure")
        return object()

    context = SimpleNamespace(
        config=SimpleNamespace(resources={"rocm_home": str(sdk)}),
        compile_custom_ops=compile_ops,
        publish_library=lambda *args: published.append(args))
    with pytest.raises(RuntimeError, match="HIP compiler failure"):
        installer.install_libraries(context)
    assert not published


def test_python_registration_is_narrow_and_uses_native_names(monkeypatch):
    installer = load_installer(monkeypatch)
    calls = []
    dispatch = ModuleType("jittor._runtime.dispatch")
    dispatch.register_kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    installer.install_kernels()
    assert [args[0] for args, _ in calls] == ["matmul", "misc.scan_2d"]
    assert all(args[1] == "rocm_legacy" for args, _ in calls)
    assert calls[0][1]["dtypes"] == {"float32", "float64"}
    assert calls[1][1]["dtypes"] == {"float32", "float64", "int32", "int64"}
    a = SimpleNamespace(ndim=2, dtype="float32")
    assert installer._supports_matmul(a, a)
    assert not installer._supports_matmul(a, SimpleNamespace(ndim=3, dtype="float32"))
    assert not installer._supports_matmul(a, SimpleNamespace(ndim=2, dtype="float64"))


def test_gemm_layout_all_transposes_and_checked_dimensions(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for ROCm row-major GEMM contract")
    source = tmp_path / "gemm_layout.cc"
    source.write_text(r'''
#include "hipblas/gemm_layout.h"
#include <cassert>
#include <vector>
using namespace jittor;
int main() {
    for (bool ta : {false, true}) for (bool tb : {false, true}) {
        const int m=3, n=5, k=4;
        const int ar=ta?k:m, ac=ta?m:k, br=tb?n:k, bc=tb?k:n;
        auto plan=hipblas_gemm_layout(ar, ac, br, bc, ta, tb);
        assert(plan.rows==m && plan.columns==n && plan.inner==k);
        std::vector<double> a(ar*ac), b(br*bc);
        for (unsigned i=0; i<a.size(); ++i) a[i]=int(i%7)-3;
        for (unsigned i=0; i<b.size(); ++i) b[i]=int(i%9)-4;
        for (int i=0; i<m; ++i) for (int j=0; j<n; ++j) {
            double reference=0, column_major=0;
            for (int r=0; r<k; ++r) {
                reference += (ta?a[r*ac+i]:a[i*ac+r]) * (tb?b[j*bc+r]:b[r*bc+j]);
                column_major += (tb?b[r+j*plan.lda]:b[j+r*plan.lda])
                              * (ta?a[i+r*plan.ldb]:a[r+i*plan.ldb]);
            }
            assert(reference==column_major);
            assert(i*n+j==j+i*plan.ldc);
        }
    }
    auto empty=hipblas_gemm_layout(3, 0, 0, 5, false, false);
    assert(empty.inner==0 && empty.ldb==1 && empty.ldc==5);
    auto zero_rows=hipblas_gemm_layout(0, 4, 4, 5, false, false);
    assert(zero_rows.rows==0);
    int rejected=0;
    try { hipblas_gemm_layout(3, 4, 2, 5, false, false); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { hipblas_gemm_layout(INT64_MAX, 4, 4, 5, false, false); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { hipblas_gemm_layout(-1, 4, 4, 5, false, false); }
    catch (const std::invalid_argument&) { ++rejected; }
    assert(rejected==3);
}
''', encoding="utf-8")
    executable = tmp_path / "gemm_layout"
    subprocess.run([compiler, "-std=c++14", "-Wall", "-Werror",
                    "-I" + str(ROCM / "libraries"), str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
