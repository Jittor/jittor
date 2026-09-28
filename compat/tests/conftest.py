"""Load the shared monorepo dev policy; compat wheels do not ship tests."""

import importlib
from pathlib import Path
import sys

_support = Path(__file__).resolve().parents[2] / "tests"
if not (_support / "_helpers/pytest_policy.py").is_file():
    raise RuntimeError("compat tests require the monorepo tests/_helpers dev support")
if str(_support) not in sys.path:
    sys.path.insert(0, str(_support))


def pytest_addoption(parser, pluginmanager):
    name = "_helpers.pytest_policy"
    if not pluginmanager.hasplugin(name):
        pluginmanager.register(importlib.import_module(name), name)


def _torch_mode_is_active():
    import os
    value = os.environ.get("JITTOR_TORCH_SHIM", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def pytest_configure(config):
    """Deploy the packaged shim, because this suite is what tests it.

    ``JITTOR_TORCH_SHIM=1`` installs the Torch namespace *inside* a process
    that imports jittor. By design it does not write a ``site-packages`` tree,
    so on its own it cannot make ``import torch`` work as a process's first
    import, and it leaves the bundled stubs (``flash_attn``, ``torchvision``,
    ...) unimportable. Both are what the deployed package exists for, and
    tests here exercise both -- through child processes that start with
    ``import torch`` and through ``import flash_attn``. Deploy one into the
    cache and point this process and its children at it.
    """
    import os
    import sys as _sys
    if not _torch_mode_is_active():
        return
    from jittor.compat.shim.deploy import deploy
    from jittor.compat.shim.preflight import resources_root
    import jittor_utils
    target = os.path.join(jittor_utils.home(), ".cache", "jittor",
                          "torch-shim", "pytest-site-packages")
    deploy(target)
    # In *this* process the stubs come from the packaged resources rather than
    # from the copy: they are what is under test, and a test that asks where
    # `flash_attn` came from should be shown the source it is maintained in.
    # The copy is for children, which reach it through PYTHONPATH below.
    stubs = os.fspath(resources_root() / "stubs")
    if stubs not in _sys.path:
        _sys.path.insert(0, stubs)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if target not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts + [target])
