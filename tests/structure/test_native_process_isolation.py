"""A native test states "the shim is not active", not "torch was never imported".

Five native files used to open with `assert "torch" not in sys.modules`. What
they mean is that jittor's own runtime is under test -- which is what
`_helpers.torch_runtime.torch_shim_loaded()` answers -- but the assertion says
something stronger, and something the tree does not guarantee: several native
tests import an independent PyTorch in-process to compare against, and xdist
shares a worker between files. One such import turned
`tests/ops/test_storage_layout_dtype.py`, `tests/optim/test_native_zero.py` and
`tests/serialization/test_sharded_checkpoint.py` red in the smoke tier
(KI-TEST-018), 46 setup errors from a single cause.

The heuristic cannot be a text scan: the same phrase appears inside subprocess
script sources, where a fresh process really does mean no torch. So what is
pinned here is the predicate those files now call.
"""

import sys
import types

from _helpers import torch_runtime


def test_an_independent_torch_is_not_the_shim(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert torch_runtime.torch_shim_loaded() is False


def test_the_shim_is_recognised_without_importing_it(monkeypatch):
    # The deployed stub does not call itself `torch`, which is the first of the
    # three shapes the helper knows.
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("jittor_torch"))
    assert torch_runtime.torch_shim_loaded() is True

    marked = types.ModuleType("torch")
    marked._jittor_torch_shim_placeholder = True
    monkeypatch.setitem(sys.modules, "torch", marked)
    assert torch_runtime.torch_shim_loaded() is True

    installed = types.ModuleType("torch")
    installed._torch_compat_install_context = object()
    monkeypatch.setitem(sys.modules, "torch", installed)
    assert torch_runtime.torch_shim_loaded() is True


def test_no_torch_at_all_is_not_the_shim(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch")
    assert torch_runtime.torch_shim_loaded() is False


def test_the_import_guard_recognises_the_deployed_stub(tmp_path):
    """The policy refuses `import torch` in native mode for the stub, not for
    a real PyTorch -- that is what keeps an oracle comparison importable."""
    import importlib.machinery
    import importlib.util

    from _helpers import pytest_policy

    def spec_for(body):
        path = tmp_path / f"torch_{abs(hash(body))}.py"
        path.write_text(body, encoding="utf-8")
        return importlib.util.spec_from_file_location("torch", path)

    stub = spec_for("_sys.modules[__name__]._jittor_torch_shim_placeholder = True\n")
    genuine = spec_for("__version__ = '2.11.0'\n")
    assert torch_runtime._spec_is_deployed_torch_shim(stub) is True
    assert torch_runtime._spec_is_deployed_torch_shim(genuine) is False
