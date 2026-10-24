"""Pure host provenance tests; fake packages are fixtures, not tensor runtimes."""
import ast
import importlib.abc
import importlib.machinery
import importlib.metadata
from pathlib import Path
import sys
import sysconfig
import types

import pytest

BOOTSTRAP = Path(__file__).resolve().parents[1] / "runtime/oracle/sitecustomize.py"

def functions(env=None):
    tree = ast.parse(BOOTSTRAP.read_text())
    dependencies = {"_prepare_native_torchvision", "_NativeVisionLoader", "_oracle_find_spec",
                    "_BlockJittorFinder", "require_clean_venv", "require_official_torch"}
    nodes = [n for n in tree.body
             if (isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (n.name.startswith(("_native_vision", "_native_torchvision"))
                      or n.name in dependencies))
             or (isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id.startswith(
                     ("_NATIVE_VISION", "_native_vision_roots")) for t in n.targets))]
    ns = {"importlib": importlib, "Path": Path, "os": types.SimpleNamespace(environ=env or {}),
          "sys": types.SimpleNamespace(modules={}, meta_path=[]), "json": None,
          "_find_spec": lambda *args: "ORIGINAL_FIND_SPEC"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<host-bootstrap-guards>", "exec"), ns)
    return ns


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prefix = (tmp_path / "venv").resolve()
    pure = prefix / "lib/python3.11/site-packages"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").write_text("# fake host-only interpreter path\n")
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    pure.mkdir(parents=True)
    ns = functions({"VERL_ORACLE_NATIVE_TORCHVISION": "1"})
    ns["sys"].prefix = str(prefix)
    ns["sys"].executable = str(prefix / "bin/python")
    specs, distributions = {}, {}
    loader = types.SimpleNamespace(create_module=lambda spec: None, exec_module=lambda mod: None)
    for name, version in ns["_NATIVE_VISION_PAIR"].items():
        pkg = pure / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# host-only package fixture; not executable\n")
        info = pure / (name + "-" + version + ".dist-info")
        info.mkdir()
        (info / "METADATA").write_text("Name: " + name + "\nVersion: " + version + "\n")
        record = Path(info.name) / "METADATA"
        files = [record, Path(name) / "__init__.py"]
        if name == "torchvision":
            (pkg / "_C.so").write_bytes(b"host-only extension provenance fixture; not loadable")
            files.append(Path(name) / "_C.so")
        distributions[name] = types.SimpleNamespace(version=version, files=files,
                                                     locate_file=lambda item, pure=pure: pure / item)
        specs[name] = importlib.machinery.ModuleSpec(name, loader, origin=str(pkg / "__init__.py"),
                                                   is_package=True)
        specs[name].submodule_search_locations = [str(pkg)]
    child = pure / "torchvision/transforms"
    child.mkdir()
    (child / "__init__.py").write_text("# fixture\n")
    specs["torchvision.transforms"] = importlib.machinery.ModuleSpec(
        "torchvision.transforms", loader, origin=str(child / "__init__.py"), is_package=True)
    specs["torchvision.transforms"].submodule_search_locations = [str(child)]
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distributions[name])
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(pure))
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec",
                        staticmethod(lambda name, path=None, target=None: specs.get(name)))
    ns["_find_spec"] = lambda name, package=None: specs.get(name)
    return types.SimpleNamespace(ns=ns, prefix=prefix, pure=pure, specs=specs, distributions=distributions)


def official_stub(f):
    torch = types.ModuleType("torch")
    torch._C = object()
    torch.__version__ = "2.8.0+cu128"
    torch.__file__ = f.specs["torch"].origin
    torch.__spec__ = f.specs["torch"]
    f.ns["sys"].modules["torch"] = torch
    return torch


def test_native_optin_requires_admission_before_discovery():
    ns = functions({"VERL_ORACLE_NATIVE_TORCHVISION": "1"})
    with pytest.raises(SystemExit, match="admission"):
        ns["_oracle_find_spec"]("torchvision")
    with pytest.raises(SystemExit, match="admission"):
        ns["_BlockJittorFinder"]().find_spec("torchvision.transforms")


def test_exact_profile_and_no_framework_identity_mutations():
    ns = functions()
    assert ns["_NATIVE_VISION_ENV"] == "VERL_ORACLE_NATIVE_TORCHVISION"
    assert ns["_NATIVE_VISION_PAIR"] == {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128"}
    tree = ast.parse(BOOTSTRAP.read_text())
    identity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_official_torch")
    required = {n.value for n in ast.walk(identity) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"_C", "_torch_compat_install_context", "jittor", "jittor."}.issubset(required)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr not in ("__version__", "__spec__")
    assert not any(n == "jittor" or n.startswith("jittor.") for n in sys.modules)
    assert "torch" not in sys.modules and "torchvision" not in sys.modules


@pytest.mark.parametrize("mode", [None, "0"])
def test_default_text_only_without_distribution_discovery(mode):
    env = {} if mode is None else {"VERL_ORACLE_NATIVE_TORCHVISION": mode}
    ns = functions(env)
    ns["require_clean_venv"] = lambda: pytest.fail("disabled feature checked venv")
    ns["_prepare_native_torchvision"]()
    assert ns["_oracle_find_spec"]("torchvision.transforms") is None
    assert ns["_oracle_find_spec"]("other") == "ORIGINAL_FIND_SPEC"
    with pytest.raises(ModuleNotFoundError, match="text-only"):
        ns["_BlockJittorFinder"]().find_spec("torchvision")


@pytest.mark.parametrize("mode", ["", "2", "true", "yes", "-1"])
def test_invalid_env_systemexit(mode):
    ns = functions({"VERL_ORACLE_NATIVE_TORCHVISION": mode})
    with pytest.raises(SystemExit, match="absent, 0 or 1"):
        ns["_native_vision_enabled"]()


def test_enabled_all_checked_public_discovery_and_jittor_still_blocked(fixture):
    f = fixture
    f.ns["_prepare_native_torchvision"]()
    assert f.ns["_oracle_find_spec"]("torchvision") is f.specs["torchvision"]
    assert f.ns["_oracle_find_spec"]("torchvision.transforms") is f.specs["torchvision.transforms"]
    spec = f.ns["_BlockJittorFinder"]().find_spec("torchvision.transforms", [str(f.pure / "torchvision")])
    assert isinstance(spec.loader, f.ns["_NativeVisionLoader"])
    for name in ("jittor", "jittor.compat.torch"):
        with pytest.raises(ModuleNotFoundError, match="refuses"):
            f.ns["_BlockJittorFinder"]().find_spec(name)
    # Re-entry reads/validates the same metadata; no fake framework imported.
    f.ns["_prepare_native_torchvision"]()


@pytest.mark.parametrize("kind", ["missing_cfg", "system_site", "prefix", "outside_purelib", "missing_metadata",
                                  "metadata_symlink", "package_symlink", "host_distribution", "torch_version",
                                  "vision_version", "missing_spec", "namespace_spec", "host_spec", "spec_symlink"])
def test_admission_negative_controls(fixture, kind, monkeypatch, tmp_path):
    f = fixture
    if kind == "missing_cfg":
        (f.prefix / "pyvenv.cfg").unlink()
    elif kind == "system_site":
        (f.prefix / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    elif kind == "prefix":
        f.ns["sys"].executable = str(tmp_path / "foreign/bin/python")
    elif kind == "outside_purelib":
        monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path))
    elif kind == "missing_metadata":
        f.distributions["torchvision"].files = []
    elif kind == "metadata_symlink":
        path = f.distributions["torchvision"].locate_file(f.distributions["torchvision"].files[0])
        moved = path.with_name("real-metadata")
        path.rename(moved)
        path.symlink_to(moved)
    elif kind == "package_symlink":
        path = f.pure / "torchvision"
        moved = f.pure / "real-vision"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    elif kind == "host_distribution":
        f.distributions["torchvision"].locate_file = lambda item: tmp_path / item
    elif kind in ("torch_version", "vision_version"):
        f.distributions["torch" if kind == "torch_version" else "torchvision"].version = "0.0"
    elif kind == "missing_spec":
        f.specs["torchvision"] = None
    elif kind == "namespace_spec":
        f.specs["torchvision"].origin = None
    elif kind == "host_spec":
        path = tmp_path / "host.py"
        path.write_text("# host facade fixture\n")
        f.specs["torchvision"].origin = str(path)
    else:
        path = f.pure / "torchvision/__init__.py"
        moved = path.with_name("real.py")
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(SystemExit):
        f.ns["_prepare_native_torchvision"]()


@pytest.mark.parametrize("name", ["torchvision", "torchvision.transforms"])
def test_preloaded_before_admission_rejected_even_claiming_native_path(fixture, name):
    f = fixture
    module = types.ModuleType(name)
    module.__file__ = f.specs[name].origin
    module.__spec__ = f.specs[name]
    module.__version__ = "0.23.0+cu128"
    f.ns["sys"].modules[name] = module
    with pytest.raises(SystemExit, match="preloaded before admission"):
        f.ns["_prepare_native_torchvision"]()


def test_default_rejects_preloaded_vision():
    ns = functions()
    ns["sys"].modules["torchvision"] = types.ModuleType("torchvision")
    with pytest.raises(SystemExit, match="preloaded"):
        ns["_prepare_native_torchvision"]()


@pytest.mark.parametrize("kind", ["shim", "foreign_file", "version", "no_binary", "jittor_contaminated"])
def test_loaded_torch_rejected(fixture, kind, tmp_path):
    f = fixture
    torch = official_stub(f)
    if kind == "shim":
        torch._torch_compat_install_context = object()
    elif kind == "foreign_file":
        path = tmp_path / "torch.py"
        path.write_text("# facade fixture\n")
        torch.__file__ = str(path)
    elif kind == "version":
        torch.__version__ = "2.8.0"
    elif kind == "no_binary":
        del torch._C
    else:
        f.ns["sys"].modules["jittor"] = types.ModuleType("jittor")
    with pytest.raises(SystemExit):
        f.ns["_prepare_native_torchvision"]()


@pytest.mark.parametrize("kind", ["shadow", "symlink", "missing", "search_path"])
def test_submodule_discovery_negative_after_admission(fixture, kind, tmp_path):
    f = fixture
    f.ns["_prepare_native_torchvision"]()
    path = f.pure / "torchvision"
    if kind == "search_path":
        path = tmp_path
    elif kind == "missing":
        f.specs["torchvision.transforms"] = None
    elif kind == "shadow":
        foreign = tmp_path / "facade.py"
        foreign.write_text("# facade fixture\n")
        f.specs["torchvision.transforms"].origin = str(foreign)
    else:
        original = f.pure / "torchvision/transforms/__init__.py"
        moved = original.with_name("actual.py")
        original.rename(moved)
        original.symlink_to(moved)
    with pytest.raises(SystemExit):
        f.ns["_native_torchvision_spec"]("torchvision.transforms", [str(path)])


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError])
def test_loader_known_failures_not_swallowed_by_upstream_except_exception(fixture, error):
    f = fixture
    f.ns["_prepare_native_torchvision"]()
    def fail(module):
        raise error("native import broken")
    loader = f.ns["_NativeVisionLoader"](types.SimpleNamespace(exec_module=fail))
    with pytest.raises(SystemExit, match="import failed"):
        try:
            loader.exec_module(types.ModuleType("torchvision"))
        except Exception:
            pytest.fail("upstream optional import swallowed the native policy failure")


def test_loader_success_and_reentry_source_checks(fixture):
    f = fixture
    f.ns["_prepare_native_torchvision"]()
    official_stub(f)
    module = types.ModuleType("torchvision")
    module.__file__ = f.specs["torchvision"].origin
    module.__spec__ = f.specs["torchvision"]
    module.__version__ = "0.23.0+cu128"
    f.ns["_NativeVisionLoader"](types.SimpleNamespace(exec_module=lambda m: None)).exec_module(module)
    f.ns["sys"].modules["torchvision"] = module
    f.ns["_prepare_native_torchvision"]()
    module.__file__ = f.pure / "torch/__init__.py"
    with pytest.raises(SystemExit):
        f.ns["_prepare_native_torchvision"]()


def test_extension_load_library_path_cannot_bypass_symlink_guard(fixture):
    f = fixture
    path = f.pure / "torchvision/_C.so"
    moved = path.with_name("actual.so")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(SystemExit, match="canonical"):
        f.ns["_prepare_native_torchvision"]()


def test_wrapped_native_loader_retains_resource_source_apis(fixture):
    wrapped = types.SimpleNamespace(get_resource_reader=lambda name: ("reader", name),
                                    get_source=lambda name: ("source", name))
    loader = fixture.ns["_NativeVisionLoader"](wrapped)
    assert loader.get_resource_reader("torchvision") == ("reader", "torchvision")
    assert loader.get_source("torchvision") == ("source", "torchvision")


def test_invalid_profile_validated_at_module_scope_without_framework_imports():
    tree = ast.parse(BOOTSTRAP.read_text())
    assert any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == "_native_vision_enabled"
               for n in tree.body)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_native_torchvision")
    imports = [alias.name for n in ast.walk(function) if isinstance(n, ast.Import) for alias in n.names]
    assert imports == ["importlib.metadata", "sysconfig"]
