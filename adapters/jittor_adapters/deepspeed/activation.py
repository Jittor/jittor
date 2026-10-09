"""Owned pre-import source loaders and first-accelerator selection.

Only DeepSpeed modules are intercepted. No files, framework attributes,
compiled extensions or builtins.__import__ are replaced.
"""

import importlib.abc
import importlib.machinery
from pathlib import Path
import sys

from jittor.compat.module_patcher import patch_method
from jittor.compat.transaction import (
    TransactionConflict,
    active_transaction,
    owned_runtime_hook,
    release_runtime_hooks,
    set_attr,
)

from .._common import UnsupportedAdapterVersion, require_version
from . import SUPPORTED_VERSIONS
from . import source

_OWNER = "deepspeed.explicit_adapter"
_IMPORT_OWNER = "deepspeed.import"
_finder = None
_imported = False


def _deepspeed_modules():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "deepspeed" or name.startswith("deepspeed.")
    }


def _undo_modules(owned):
    conflicts = []
    for name, module in sorted(owned.items(), reverse=True):
        if sys.modules.get(name) is module:
            del sys.modules[name]
        elif name in sys.modules:
            conflicts.append(name)
    if conflicts:
        raise TransactionConflict("DeepSpeed modules replaced externally: " + ", ".join(conflicts))


class _Loader(importlib.abc.Loader):
    def __init__(self, finder, fullname, relative, origin):
        self.finder, self.fullname = finder, fullname
        self.relative, self.origin = relative, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.fullname == "deepspeed":
            return self._execute_root(module)
        text = source.verified_source(self.finder.package_root, self.relative)
        if self.fullname == "deepspeed.runtime.engine":
            text = source.transform_engine(text)
        elif self.fullname == "deepspeed.runtime.zero.utils":
            text = source.transform_zero_utils(text)
        elif self.fullname == "deepspeed.runtime.zero.stage_1_and_2":
            text = source.transform_zero_stage_1_and_2(text)
            text = source.transform_zero_adagrad_guard(text, "stage_1_and_2")
            if self.finder.device == "npu":
                text = source.transform_zero_npu_norm(text)
                text = source.transform_zero_npu_stage1(text)
        elif self.fullname == "deepspeed.runtime.zero.stage3":
            text = source.transform_zero_adagrad_guard(text, "stage3")
            if self.finder.device == "npu":
                text = source.transform_zero_npu_stage3_norm(text)
        elif self.fullname == "deepspeed.elasticity":
            text = source.transform_elasticity(text)
        elif self.fullname == "deepspeed.comm.comm":
            text = source.transform_comm(text)
        elif self.fullname == "deepspeed.utils.torch":
            text = source.transform_utils_torch(text)
        exec(compile(text, self.origin, "exec"), module.__dict__)
        if self.fullname == "deepspeed.accelerator.real_accelerator":
            self._select_accelerator(module)
        elif self.fullname == "deepspeed.runtime.engine":
            from .scope import guard

            original = module.DeepSpeedEngine.__init__
            patch_method(
                module.DeepSpeedEngine,
                "__init__",
                guard(original, self.finder.device),
                expected=original,
            )

    @owned_runtime_hook(_IMPORT_OWNER)
    def _execute_root(self, module):
        global _imported
        # The parent module object exists now, before its original body imports
        # ops. Publish its OWN source-defined version early, never metadata from
        # an unrelated installed package, then use the shared version contract.
        version = source.inspect_package(self.finder.package_root)
        module.__version__ = version
        require_version("deepspeed", SUPPORTED_VERSIONS)
        before = _deepspeed_modules()
        before.pop("deepspeed", None)
        try:
            text = source.verified_source(self.finder.package_root, "__init__.py")
            exec(compile(text, self.origin, "exec"), module.__dict__)
            require_version("deepspeed", SUPPORTED_VERSIONS)
            if module.__version__ != version:
                raise UnsupportedAdapterVersion("DeepSpeed runtime/source versions disagree")
            from .scope import guard

            original = module.initialize
            patch_method(
                module, "initialize", guard(original, self.finder.device), expected=original
            )
            set_attr(sys.modules[__name__], "_imported", True)
        finally:
            owned = {
                name: value for name, value in _deepspeed_modules().items() if name not in before
            }
            active_transaction().record_undo(lambda: _undo_modules(owned))

    def _select_accelerator(self, module):
        require_version("deepspeed", SUPPORTED_VERSIONS)
        if module.ds_accelerator is not None:
            raise RuntimeError("DeepSpeed accelerator was selected before adapter activation")
        from .accelerator import create_accelerator

        accelerator = create_accelerator(self.finder.device)
        before = dict(vars(module))
        # This runs before accelerator/__init__.py publishes get_accelerator.
        module.set_accelerator(accelerator)
        active_transaction().record_object_diffs(module, before)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, device):
        self.device = device
        self.package_root = None

    def find_spec(self, fullname, path=None, target=None):
        relatives = {
            "deepspeed": "__init__.py",
            "deepspeed.accelerator.real_accelerator": "accelerator/real_accelerator.py",
            "deepspeed.comm.comm": "comm/comm.py",
            "deepspeed.elasticity": "elasticity/__init__.py",
            "deepspeed.runtime.engine": "runtime/engine.py",
            "deepspeed.runtime.zero.utils": "runtime/zero/utils.py",
            "deepspeed.runtime.zero.stage_1_and_2": "runtime/zero/stage_1_and_2.py",
            "deepspeed.runtime.zero.stage3": "runtime/zero/stage3.py",
            "deepspeed.utils.torch": "utils/torch.py",
        }
        if fullname not in relatives:
            return None
        if target is not None:
            raise UnsupportedAdapterVersion(
                "DeepSpeed adapter does not support module reload; use a fresh process"
            )
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None or spec.loader is None:
            raise UnsupportedAdapterVersion("DeepSpeed source package is unavailable")
        origin = Path(spec.origin).resolve()
        if fullname == "deepspeed":
            root = origin.parent
            source.inspect_package(root)
            self.package_root = root
        if self.package_root is None or origin != self.package_root / relatives[fullname]:
            raise UnsupportedAdapterVersion(
                "DeepSpeed modules must share one validated source tree"
            )
        source.verified_source(self.package_root, relatives[fullname])
        spec.loader = _Loader(self, fullname, relatives[fullname], str(origin))
        return spec


def activate(device="cpu"):
    if device not in ("cpu", "npu"):
        raise ValueError("DeepSpeed adapter device must be cpu or npu")
    if _finder is not None:
        if _finder.device != device:
            raise RuntimeError("DeepSpeed adapter already selected another device")
        return status()
    if _deepspeed_modules():
        raise RuntimeError("Activate the DeepSpeed adapter before importing any DeepSpeed module")
    return _activate(device)


@owned_runtime_hook(_OWNER)
def _activate(device):
    global _finder
    finder = _Finder(device)
    sys.meta_path.insert(0, finder)

    def undo_finder():
        if finder not in sys.meta_path:
            raise TransactionConflict("DeepSpeed adapter finder was removed externally")
        sys.meta_path.remove(finder)

    active_transaction().record_undo(undo_finder)
    set_attr(sys.modules[__name__], "_finder", finder)
    return status()


def deactivate():
    conflicts = []
    for owner in (_IMPORT_OWNER, _OWNER):
        try:
            release_runtime_hooks(owner)
        except TransactionConflict as error:
            conflicts.append(str(error))
    if conflicts:
        raise TransactionConflict("; ".join(conflicts))
    return status()


def status():
    return {
        "active": _finder is not None,
        "device": _finder.device if _finder is not None else None,
        "imported": _imported,
        "supported_versions": sorted(SUPPORTED_VERSIONS),
        "scope": "CPU import/config/model; NPU FP32 HCCL Stage 0 (one/two rank) and Stage 1/2/3 (two rank) with explicit AdamW",
        "source_sha256": {name: list(hashes) for name, hashes in source.SOURCE_SHA256.items()},
        "source_identity_scope": "Pinned source identity only; runtime acceptance is recorded separately",
    }
