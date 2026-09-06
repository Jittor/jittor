"""Source, installed, and converted backend roots without runtime imports."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def resolver():
    spec = importlib.util.spec_from_file_location(
        "backend_resources_contract", ROOT / "python/jittor_utils/backend_resources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.backend_root


def _package(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('"""Backend package."""\n')
    return path


def test_checkout_root_wins_over_stale_package_cache(tmp_path, resolver):
    package = tmp_path / "repo/python/jittor"
    stale = package / "backends/cuda/__pycache__"
    stale.mkdir(parents=True)
    expected = _package(tmp_path / "repo/backends/cuda")
    assert resolver(package, "cuda") == str(expected)
    _package(stale.parent)
    assert resolver(package, "cuda") == str(expected)


@pytest.mark.parametrize("layout", ["installed/jittor", "cache/acl_jittor"])
def test_installed_and_converted_packages_resolve_locally(tmp_path, resolver, layout):
    package = tmp_path / layout
    expected = _package(package / "backends/cuda")
    assert resolver(package, "cuda") == str(expected)


def test_empty_or_cache_only_directory_is_not_a_backend(tmp_path, resolver):
    package = tmp_path / "installed/jittor"
    (package / "backends/cuda/__pycache__").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="backend resources"):
        resolver(package, "cuda")
    with pytest.raises(ValueError, match="identifier"):
        resolver(package, "../cuda")


@pytest.mark.parametrize("checkout", [True, False])
def test_conversion_mirrors_moved_native_resources_exactly_once(tmp_path, resolver, checkout):
    package = tmp_path / ("repo/python/jittor" if checkout else "installed/jittor")
    (package / "src").mkdir(parents=True)
    (package / "src/core.cc").write_text("CUDA core\n")
    parent = tmp_path / "repo/backends" if checkout else package / "backends"
    cuda = _package(parent / "cuda")
    (cuda / "include").mkdir()
    (cuda / "include/helper_cuda.h").write_text("CUDA helper\n")
    (cuda / "kernels/core").mkdir(parents=True)
    kernel = cuda / "kernels/core/kernel.cc"
    kernel.write_text("CUDA kernel\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "acl_jittor/extern/cuda/ops/obsolete.cc"
    stale.parent.mkdir(parents=True)
    stale.write_text("obsolete CUDA\n")

    class Config(SimpleNamespace):
        def evolve(self, **changes):
            return Config(**dict(vars(self), **changes))

    config = Config(jittor_path=str(package), cache_path=str(cache), resources={},
                    cc_flags=' -I"%s/src" -I"%s/include" ' % (package, cuda))
    path = ROOT / "python/jittor_utils/__init__.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "process_jittor_source")
    namespace = {"os": os, "backend_root": resolver, "LOG": SimpleNamespace(i=lambda *args: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    def transform(source, name, arguments):
        calls.append(arguments["fname"])
        return source.replace("CUDA", "ACL")
    converted = namespace["process_jittor_source"](config, "acl", transform)
    converted_cuda = Path(resolver(converted.jittor_path, "cuda"))
    assert (converted_cuda / "kernels/core/kernel.cc").read_text() == "ACL kernel\n"
    assert (converted_cuda / "include/helper_cuda.h").read_text() == "ACL helper\n"
    assert len(calls) == len(set(calls)) == 3
    assert converted.resources["backend_roots"]["cuda"] == str(converted_cuda)
    assert str(cuda) not in converted.cc_flags
    assert str(converted_cuda) in converted.cc_flags
    assert config.resources == {} and config.jittor_path == str(package)
    assert not stale.exists()
    assert len(list(cache.glob("acl_source_stale_*/extern/cuda/ops/obsolete.cc"))) == 1


def test_cuda_library_inventory_includes_cutt_support_sources(resolver):
    path = ROOT / "python/jittor/compile_extern.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_cuda_library_sources")
    namespace = {"os": os, "backend_root": resolver,
                 "jittor_path": str(ROOT / "python/jittor")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    files = namespace["_cuda_library_sources"]("cutt")
    assert str(ROOT / "backends/cuda/kernels/cutt/cutt_transpose_op.cc") in files
    assert str(ROOT / "backends/cuda/libraries/cutt/include/cutt_wrapper.h") in files
    assert str(ROOT / "backends/cuda/libraries/cutt/src/cutt_wrapper.cc") in files
    assert len(files) == len(set(files))


def test_source_bridge_resolves_canonical_kernel_modules_without_bootstrap():
    with patch.dict(sys.modules):
        for name in tuple(sys.modules):
            if name == "jittor" or name.startswith("jittor."):
                del sys.modules[name]
        package = ModuleType("jittor")
        package.__path__ = [str(ROOT / "python/jittor")]
        sys.modules["jittor"] = package
        expected = {
            "jittor.backends.cuda.kernels.nn.softmax_cuda": "cuda/kernels/nn/softmax_cuda.py",
            "jittor.backends.cuda.kernels.math.gamma": "cuda/kernels/math/gamma.py",
            "jittor.backends.cuda.kernels.pooling.pool2d": "cuda/kernels/pooling/pool2d.py",
            "jittor.backends.acl.kernels.kv_cache": "acl/kernels/kv_cache.py",
        }
        for name, relative in expected.items():
            spec = importlib.util.find_spec(name)
            assert Path(spec.origin) == ROOT / "backends" / relative
        assert "jittor.compiler" not in sys.modules


def test_source_bridge_prefers_checkout_over_stale_package_init(tmp_path, resolver):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / "pyproject.toml").touch()
    package = checkout / "python/jittor"
    bridge = _package(package / "backends")
    (bridge / "__init__.py").write_text(
        (ROOT / "python/jittor/backends/__init__.py").read_text())
    stale = _package(bridge / "cuda")
    current = _package(checkout / "backends/cuda")
    for directory in (stale, current):
        (directory / "probe.py").write_text('"""Import resolution probe."""\n')
    with patch.dict(sys.modules):
        for name in tuple(sys.modules):
            if name == "jittor" or name.startswith("jittor."):
                del sys.modules[name]
        root = ModuleType("jittor")
        root.__path__ = [str(package)]
        sys.modules["jittor"] = root
        spec = importlib.util.find_spec("jittor.backends.cuda.probe")
        assert Path(spec.origin) == current / "probe.py"
        assert Path(sys.modules["jittor.backends.cuda"].__file__) == current / "__init__.py"
        assert Path(resolver(package, "cuda")) == current
        assert "jittor.compiler" not in sys.modules
