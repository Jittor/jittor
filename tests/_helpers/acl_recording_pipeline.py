"""Record actual ACL Python builders without importing the Jittor runtime."""

import importlib.util
import sys
import types

from _helpers.paths import REPO_ROOT


OPS = REPO_ROOT / "backends/acl/kernels/ops"


def build_pipeline(monkeypatch):
    class Tensor:
        def __init__(self, shape, dtype="float32"):
            self.shape, self.dtype = tuple(shape), dtype

        @property
        def ndim(self):
            return len(self.shape)

        def __len__(self):
            return len(self.shape)

        def numel(self):
            import math

            return math.prod(self.shape)

    calls = []
    jt = types.ModuleType("jittor")
    jt.Function = object
    jt.flags = types.SimpleNamespace(no_grad=0)
    jt.empty = jt.zeros = Tensor

    def code(*args, **kwargs):
        if args:
            kwargs["inputs"] = args[2]
        calls.append(kwargs)
        return kwargs.get("outputs") or [
            Tensor(shape, dtype) for shape, dtype in zip(args[0], args[1])
        ]

    jt.code = code
    monkeypatch.setitem(sys.modules, "jittor", jt)
    compiler = types.ModuleType("jittor.compiler")
    jt.compiler = compiler
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    utils = types.ModuleType("jittor_utils")
    utils.env_or_try_find = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "jittor_utils", utils)
    dtypes = types.ModuleType("jittor._core.dtypes")
    dtypes.dtype_name = str
    monkeypatch.setitem(sys.modules, dtypes.__name__, dtypes)
    package = types.ModuleType("acl_attribute_pipeline")
    # Mirror the real jittor.backends.acl.kernels package: ``ops`` is the
    # subpackage the op files live in, so their ``from .._soc import ...``
    # resolves to a real file in this fixture's path instead of escaping the
    # top level.
    package.__path__ = [str(OPS.parent)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    ops_package = types.ModuleType(package.__name__ + ".ops")
    ops_package.__path__ = [str(OPS)]
    monkeypatch.setitem(sys.modules, ops_package.__name__, ops_package)

    def load(name):
        qualified = ops_package.__name__ + "." + name
        if qualified in sys.modules:
            return sys.modules[qualified]
        spec = importlib.util.spec_from_file_location(qualified, OPS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        return module

    # Register imports through monkeypatch so each test restores the full module table.
    for name in ("acl_data", "_attributes", "_code"):
        load(name)
    return load, Tensor, calls
