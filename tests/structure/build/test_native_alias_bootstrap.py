"""Native import aliases work without loading Jittor or its optional compat tree."""

from pathlib import Path
import subprocess
import sys

from _helpers.child_process import run_python_child


def test_native_import_and_math_do_not_load_optional_compat():
    script = r'''
import importlib.abc
import sys
class RejectCompat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jittor.compat" or fullname.startswith("jittor.compat."):
            raise AssertionError("native startup imported " + fullname)
sys.meta_path.insert(0, RejectCompat())
import jittor as jt
x = jt.array([1., 2.])
assert jt.grad((x*x).sum(), x).numpy().tolist() == [2., 4.]
assert not any(name == "jittor.compat" or name.startswith("jittor.compat.") for name in sys.modules)
print("NATIVE_WITHOUT_COMPAT_OK")
'''
    result = run_python_child(["-c", script], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "NATIVE_WITHOUT_COMPAT_OK" in result.stdout


def test_requested_but_missing_compat_has_an_installation_error():
    script = r'''
import importlib.abc
import os
import sys
class MissingCompat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jittor.compat":
            raise ModuleNotFoundError("optional package absent", name=fullname)
sys.meta_path.insert(0, MissingCompat())
os.environ["JITTOR_TORCH_SHIM"] = "1"
try:
    import jittor
except ModuleNotFoundError as error:
    assert "install the jittor-torch distribution" in str(error), error
else:
    raise AssertionError("missing requested compatibility was ignored")
print("MISSING_COMPAT_REPORTED")
'''
    result = run_python_child(["-c", script], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "MISSING_COMPAT_REPORTED" in result.stdout


def test_legacy_alias_loads_its_optional_provider_on_demand():
    script = r'''
import sys
import jittor as jt
assert "jittor.compat" not in sys.modules
import jittor.torch_compat
assert sys.modules["torch"] is jt
assert jt._torch_compat_install_complete
import jittor.torch_shim
import jittor.compat.shim
assert jittor.torch_shim is jittor.compat.shim
print("LEGACY_PROVIDER_OK")
'''
    result = run_python_child(["-c", script], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "LEGACY_PROVIDER_OK" in result.stdout


def test_native_alias_loader_is_standalone_and_never_imports_compat():
    source = Path(__file__).resolve().parents[3] / "python/jittor/_runtime/import_aliases.py"
    script = r'''
import importlib
import importlib.abc
import importlib.util
import sys
import types

class RejectCompat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jittor.compat" or fullname.startswith("jittor.compat."):
            raise AssertionError("native aliases imported optional compatibility: " + fullname)

sys.meta_path.insert(0, RejectCompat())
spec = importlib.util.spec_from_file_location("standalone_native_aliases", sys.argv[1])
aliases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aliases)
assert "jittor" not in sys.modules
assert not any(name == "jittor.compat" or name.startswith("jittor.compat.")
               for name in aliases.ALIASES.values())
root = types.ModuleType("jittor")
root.__path__ = []
sys.modules["jittor"] = root
legacy = "jittor.depthwise_conv"
canonical_name = aliases.ALIASES[legacy]
canonical = types.ModuleType(canonical_name)
sys.modules[canonical_name] = canonical
aliases.install_aliases(root)
assert aliases.import_alias(legacy) is canonical
assert importlib.import_module(legacy) is canonical
assert root.depthwise_conv is canonical
assert canonical.__name__ == canonical_name
assert not any(name == "jittor.compat" or name.startswith("jittor.compat.")
               for name in sys.modules)
print("native-alias-bootstrap-ok")
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15,
    )
    assert result.returncode == 0, result.stdout
    assert "native-alias-bootstrap-ok" in result.stdout
