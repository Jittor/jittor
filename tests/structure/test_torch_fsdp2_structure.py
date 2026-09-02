"""Architecture contracts for the canonical FSDP2 compatibility package."""

import ast
import base64
import enum
import hashlib
import importlib
import inspect
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

from _helpers.process_modes import SUBPROCESS_TIMEOUT

import jittor as jt
from jittor.compat import fsdp2 as fsdp
from jittor.compat.fsdp2 import api
from jittor.compat.fsdp2 import common
from jittor.compat.fsdp2 import compat_types
from jittor.compat.fsdp2 import config
from jittor.compat.fsdp2 import dtensor
from jittor.compat.fsdp2 import grad_sync
from jittor.compat.fsdp2 import installer
from jittor.compat.fsdp2 import optimizer
from jittor.compat.fsdp2 import shard
from jittor.compat.torch.context import ModuleRegistry


_PUBLIC_NAMES = {
    "BackwardPrefetch", "CPUOffload", "CPUOffloadPolicy", "CustomPolicy",
    "DTensor", "DTensorSpec", "DataParallelMeshDims", "DeviceMesh",
    "FSDPMeshInfo", "FSDPModule", "FSDPState", "FSDP_WRAPPED_MODULE",
    "FlatParameter", "FullOptimStateDictConfig", "FullStateDictConfig",
    "FullyShardedDataParallel", "LocalOptimStateDictConfig",
    "LocalStateDictConfig", "MixedPrecision", "MixedPrecisionPolicy",
    "ModuleWrapPolicy", "NoOffloadPolicy", "OffloadPolicy",
    "OptimStateDictConfig", "OptimStateKeyType", "ParallelStyle", "Partial",
    "Placement", "Replicate", "Shard", "ShardPlacementResult",
    "ShardedGradScaler", "ShardedOptimStateDictConfig",
    "ShardedStateDictConfig", "ShardingStrategy", "StateDictConfig",
    "StateDictSettings", "StateDictType", "TrainingState", "UnshardHandle",
    "clear_fsdp_optimizer_grads", "collect_fsdp_full_params_for_backward",
    "compute_local_shape_and_global_offset", "contextlib", "distribute_module",
    "distribute_tensor", "empty", "enum", "fill_fsdp_optimizer_grads_from_grad_map",
    "full", "fully_shard", "init_device_mesh", "install", "is_dtensor",
    "is_fsdp_managed_param", "jt", "linspace", "local_sharded_state_dict",
    "logspace", "loss_parallel", "nn", "np", "ones",
    "optimizer_has_fsdp_params", "optimizer_has_non_fsdp_params",
    "optimizer_step", "os", "parallelize_module", "rand", "randn",
    "refresh_optimizer_fsdp_params", "refresh_visible_full_grads",
    "register_fsdp_forward_method", "sharded_sgd_step", "share_comm_ctx",
    "sync_sharded_grads", "sys", "types", "zeros",
}

_OWNERSHIP = {
    common: {
        "_prod", "_world_size", "_rank", "_in_true_distributed", "_nccl_ops",
        "_flatten_var", "_ceil_div", "_pad_flat", "_slice_flat",
        "_all_gather_shards", "_reduce_scatter_padded", "_param_numel",
        "_fsdp2_flat_enabled",
    },
    dtensor: {
        "DeviceMesh", "init_device_mesh", "Placement", "Replicate", "Shard",
        "Partial", "_mark_dtensor", "_DTensorMeta", "DTensor",
        "distribute_tensor", "distribute_module", "is_dtensor",
        "_shape_from_args", "_np_dtype", "_dtensor_from_array", "empty",
        "ones", "zeros", "full", "rand", "randn", "linspace", "logspace",
    },
    config: {
        "StateDictType", "ShardingStrategy", "BackwardPrefetch", "CPUOffload",
        "_Config", "StateDictConfig", "OptimStateDictConfig",
        "FullStateDictConfig", "LocalStateDictConfig", "ShardedStateDictConfig",
        "FullOptimStateDictConfig", "LocalOptimStateDictConfig",
        "ShardedOptimStateDictConfig", "StateDictSettings", "OptimStateKeyType",
        "FlatParameter", "MixedPrecisionPolicy", "MixedPrecision",
        "OffloadPolicy", "CPUOffloadPolicy", "NoOffloadPolicy",
        "DataParallelMeshDims", "UnshardHandle",
    },
    shard: {
        "_flat_local_overlap", "_flat_entry_slices", "_refresh_flat_entry_shards",
        "_mark_fsdp_param_var", "_fsdp_param_entry", "is_fsdp_managed_param",
        "_fsdp_var_to_local", "_fsdp_var_full_tensor", "_fsdp_var_redistribute",
        "_named_parameters_with_owner", "_iter_modules", "_iter_fsdp_modules",
        "_apply_fsdp_attr", "_init_true_fsdp_state", "_unshard_module_params",
        "_reshard_module_params", "_execute_with_true_fsdp",
        "_install_true_fsdp_execute",
    },
    grad_sync: {
        "sync_sharded_grads", "_sync_sharded_grads_from_full_grads",
        "_globally_used_grads", "_visible_full_grads_from_shards",
        "_local_grad_from_visible_full", "_sync_visible_full_grads_to_optimizer",
        "refresh_visible_full_grads", "_fsdp_states_from_optimizers",
        "optimizer_has_fsdp_params", "optimizer_has_non_fsdp_params",
        "collect_fsdp_full_params_for_backward",
        "fill_fsdp_optimizer_grads_from_grad_map",
    },
    optimizer: {
        "clear_fsdp_optimizer_grads", "_optimizer_param_steps",
        "_assign_preserve_trainability", "refresh_optimizer_fsdp_params",
        "_refresh_all_optimizer_fsdp_params", "_sgd_hparams",
        "_sgd_update_for_param", "_adam_hparams", "_adam_update_for_param",
        "_optimizer_kind", "optimizer_step", "sharded_sgd_step",
        "local_sharded_state_dict",
    },
    api: {
        "_FSDPModuleMeta", "FSDPModule", "_FSDP_METHODS",
        "_inject_fsdp_methods", "fully_shard", "register_fsdp_forward_method",
        "share_comm_ctx", "FullyShardedDataParallel", "ShardedGradScaler",
    },
    compat_types: {
        "FSDPMeshInfo", "ShardPlacementResult", "_get_mesh_info", "FSDPState",
        "TrainingState", "FSDP_WRAPPED_MODULE", "DTensorSpec",
        "_get_module_fsdp_state", "_get_module_fsdp_state_if_fully_sharded_module",
        "_is_fsdp_managed_module", "_lazy_init", "_get_post_forward_mesh_info",
        "compute_local_shape_and_global_offset", "ParallelStyle",
        "parallelize_module", "loss_parallel", "_checkpoint_wrapper",
        "_apply_activation_checkpointing", "_checkpoint", "ModuleWrapPolicy",
        "CustomPolicy",
    },
    installer: {
        "_ensure_module", "_install_wrap_helpers", "install",
        "_install_fsdp2_distributed",
    },
}

_REGISTERED_MODULES = (
    "torch.distributed.tensor", "torch.distributed._tensor",
    "torch.distributed.tensor._api", "torch.distributed.tensor.placement_types",
    "torch.distributed.tensor._dtensor_spec", "torch.distributed.tensor._utils",
    "torch.distributed.tensor.parallel", "torch.distributed.tensor.parallel.api",
    "torch.distributed.tensor.parallel.style", "torch.distributed.tensor.parallel.loss",
    "torch.distributed.device_mesh", "torch.distributed._tensor.device_mesh",
    "torch.distributed.fsdp", "torch.distributed.fsdp.api",
    "torch.distributed.fsdp.fully_sharded_data_parallel",
    "torch.distributed.fsdp.wrap", "torch.distributed.fsdp._traversal_utils",
    "torch.distributed.fsdp._runtime_utils", "torch.distributed.fsdp._common_utils",
    "torch.distributed.fsdp._fsdp_state", "torch.distributed.fsdp.sharded_grad_scaler",
    "torch.distributed.fsdp._fully_shard",
    "torch.distributed.fsdp._fully_shard._fully_shard",
    "torch.distributed.fsdp._fully_shard._fsdp_api",
    "torch.distributed.fsdp._fully_shard._fsdp_common",
    "torch.distributed.fsdp._fully_shard._fsdp_init",
    "torch.distributed.fsdp._fully_shard._fsdp_state",
    "torch.distributed.fsdp._fully_shard._fsdp_param",
    "torch.distributed.fsdp._fully_shard._fsdp_collectives",
    "torch.distributed._composable", "torch.distributed._composable.fsdp",
    "torch.distributed._composable.fsdp.fully_shard",
    "torch.distributed._composable.fsdp._fsdp_api",
    "torch.distributed._functional_collectives", "torch.distributed.algorithms",
    "torch.distributed.algorithms._checkpoint",
    "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
)

_LEGACY_PICKLES = {
    "DeviceMesh": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApEZXZpY2VNZXNoCnEALg==",
    "FSDPModule": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApGU0RQTW9kdWxlCnEALg==",
    "FullStateDictConfig": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApGdWxsU3RhdGVEaWN0Q29uZmlnCnEALg==",
    "FullyShardedDataParallel": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApGdWxseVNoYXJkZWREYXRhUGFyYWxsZWwKcQAu",
    "Shard": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApTaGFyZApxAC4=",
    "StateDictType": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApTdGF0ZURpY3RUeXBlCnEALg==",
    "StateDictType_FULL_STATE_DICT": "gAJjaml0dG9yLnRvcmNoX2ZzZHAyX2NvbXBhdApTdGF0ZURpY3RUeXBlCnEAWAQAAABmdWxscQGFcQJScQMu",
}


class TestTorchFSDP2Structure(unittest.TestCase):
    def _isolated_graph(self, registry_type=ModuleRegistry):
        root = types.ModuleType("torch")
        root.__path__ = []
        dist = types.ModuleType("torch.distributed")
        modules = {"torch": root, "torch.distributed": dist}
        return root, dist, registry_type(root, modules), modules

    def test_registry_first_and_repeated_install_preserve_module_graph(self):
        root, dist, registry, modules = self._isolated_graph()
        first = installer.install_with_registry(dist, root, registry=registry)
        identities = {
            name: id(modules[name]) for name in _REGISTERED_MODULES
        }
        second = installer.install_with_registry(dist, root, registry=registry)
        self.assertIs(first, dist)
        self.assertIs(second, dist)
        self.assertIs(root.distributed, dist)
        self.assertEqual(
            {name: id(modules[name]) for name in _REGISTERED_MODULES},
            identities,
        )
        self.assertIs(
            modules["torch.distributed.fsdp"].api,
            modules["torch.distributed.fsdp.api"],
        )

    def test_public_install_reuses_root_install_context_registry(self):
        from jittor.compat.torch.context import InstallContext

        root = types.ModuleType("_stage7_fsdp_context_root")
        root.__path__ = []
        dist = types.ModuleType("torch.distributed")
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in tuple(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name, None)
            context = InstallContext.for_module(root)
            context.registry.publish("torch", root)
            context.registry.publish("torch.distributed", dist)
            installer.install(dist, root)
            identities = {
                name: context.registry._published[name]
                for name in _REGISTERED_MODULES
            }
            installer.install(dist, root)
            self.assertEqual(set(identities), set(_REGISTERED_MODULES))
            for name, module in identities.items():
                with self.subTest(name=name):
                    self.assertIs(context.registry._published[name], module)

    def test_registry_partial_failure_retries_with_existing_identity(self):
        class FailOnceRegistry(ModuleRegistry):
            def __init__(self, root_module, modules):
                super().__init__(root_module, modules)
                self.failed = False

            def ensure(self, name, factory=None, package=False):
                if name == "torch.distributed.tensor._api" and not self.failed:
                    self.failed = True
                    raise RuntimeError("injected registry failure")
                return super().ensure(name, factory=factory, package=package)

        root, dist, registry, modules = self._isolated_graph(FailOnceRegistry)
        with self.assertRaisesRegex(RuntimeError, "injected registry failure"):
            installer.install_with_registry(dist, root, registry=registry)
        tensor = modules["torch.distributed.tensor"]
        self.assertFalse(getattr(dist, installer._INSTALL_MARKER, False))

        installer.install_with_registry(dist, root, registry=registry)
        self.assertIs(modules["torch.distributed.tensor"], tensor)
        self.assertIs(dist.tensor, tensor)
        self.assertTrue(getattr(dist, installer._INSTALL_MARKER, False))

    def test_registry_rejects_conflicting_distributed_root(self):
        root = types.ModuleType("torch")
        foreign = types.ModuleType("torch.distributed")
        dist = types.ModuleType("torch.distributed")
        modules = {"torch": root, "torch.distributed": foreign}
        registry = ModuleRegistry(root, modules)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            installer.install_with_registry(dist, root, registry=registry)
        self.assertIs(modules["torch.distributed"], foreign)

    def test_canonical_package_and_legacy_alias_are_one_object(self):
        package_path = Path(fsdp.__file__).resolve()
        self.assertEqual(package_path.parent.name, "fsdp2")
        self.assertEqual(package_path.parent.parent.name, "compat")
        jittor_root = package_path.parents[2]
        self.assertFalse((jittor_root / "_torch_fsdp2").exists())
        self.assertFalse((jittor_root / "torch_fsdp2_compat").exists())
        legacy = importlib.import_module("jittor.torch_fsdp2_compat")
        self.assertIs(legacy, fsdp)
        self.assertIs(sys.modules["jittor.torch_fsdp2_compat"], fsdp)
        self.assertIs(jt.torch_fsdp2_compat, fsdp)
        self.assertEqual(set(fsdp.__all__), _PUBLIC_NAMES)
        self.assertEqual(len(fsdp.__all__), 79)
        self.assertEqual(
            {name for name in vars(fsdp) if not name.startswith("_")},
            _PUBLIC_NAMES,
        )
        namespace = {}
        exec("from jittor.compat.fsdp2 import *", {}, namespace)
        self.assertEqual(set(namespace), _PUBLIC_NAMES)

    def test_explicit_legacy_child_import_binds_parent_lazily(self):
        self.assertNotIn("api", {
            name for name in vars(fsdp) if not name.startswith("_")
        })
        try:
            legacy_api = importlib.import_module("jittor.torch_fsdp2_compat.api")
            self.assertIs(legacy_api, api)
            self.assertIs(fsdp.api, api)
        finally:
            vars(fsdp).pop("api", None)

    def test_canonical_and_legacy_first_import_orders(self):
        repo_root = Path(fsdp.__file__).resolve().parents[4]
        env = os.environ.copy()
        python_root = str(repo_root / "python")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            python_root, env.get("PYTHONPATH", ""))))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        template = r"""
import importlib
import sys
first = importlib.import_module(%r)
second = importlib.import_module(%r)
import jittor
assert first is second
assert second is sys.modules['jittor.compat.fsdp2']
assert second is sys.modules['jittor.torch_fsdp2_compat']
assert second is jittor.torch_fsdp2_compat
"""
        for first, second in (
            ("jittor.compat.fsdp2", "jittor.torch_fsdp2_compat"),
            ("jittor.torch_fsdp2_compat", "jittor.compat.fsdp2"),
        ):
            result = subprocess.run(
                [sys.executable, "-c", template % (first, second)],
                cwd=str(repo_root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=SUBPROCESS_TIMEOUT,
            )
            self.assertEqual(
                result.returncode, 0,
                "import order %s -> %s failed:\n%s\n%s"
                % (first, second, result.stdout, result.stderr),
            )

    def test_implementation_ownership_and_real_origins(self):
        names = set()
        for module, expected in _OWNERSHIP.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(set(module._EXPORTS), expected)
                self.assertTrue(names.isdisjoint(expected))
                names.update(expected)
                for name in expected:
                    value = getattr(module, name)
                    self.assertIs(getattr(fsdp, name), value)
                    if callable(value):
                        self.assertEqual(value.__module__, module.__name__)
        self.assertIs(fsdp._install_fsdp2_distributed, fsdp.install)

    def test_public_contract_signatures_reflection_and_pickle(self):
        public = sorted(name for name in vars(fsdp) if not name.startswith("_"))
        callables = {}
        for name in public:
            value = getattr(fsdp, name)
            if not callable(value):
                continue
            is_enum = inspect.isclass(value) and issubclass(value, enum.Enum)
            if is_enum:
                kind = "enum"
                signature = None
                members = [(member.name, member.value) for member in value]
            else:
                kind = "class" if inspect.isclass(value) else type(value).__name__
                try:
                    signature = str(inspect.signature(value))
                except (TypeError, ValueError):
                    signature = None
                members = None
            serialized = pickle.dumps(value)
            self.assertIs(pickle.loads(serialized), value)
            self.assertTrue(value.__module__.startswith("jittor.compat.fsdp2."))
            callables[name] = {
                "kind": kind,
                "members": members,
                "module": "jittor.torch_fsdp2_compat",
                "qualname": value.__qualname__,
                "signature": signature,
                "pickle": bool(serialized),
            }
        payload = {"public": public, "callables": callables}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(len(public), 79)
        self.assertEqual(len(callables), 70)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "ca8aea5689aa5280fcd65aa3157274e7f2cd0d2ee3e0d607ec14880d28e71e73",
        )

    def test_protocol_2_legacy_pickle_fixtures_load_canonical_objects(self):
        expected = {
            "DeviceMesh": fsdp.DeviceMesh,
            "FSDPModule": fsdp.FSDPModule,
            "FullStateDictConfig": fsdp.FullStateDictConfig,
            "FullyShardedDataParallel": fsdp.FullyShardedDataParallel,
            "Shard": fsdp.Shard,
            "StateDictType": fsdp.StateDictType,
            "StateDictType_FULL_STATE_DICT": fsdp.StateDictType.FULL_STATE_DICT,
        }
        for name, encoded in _LEGACY_PICKLES.items():
            payload = base64.b64decode(encoded)
            with self.subTest(name=name):
                self.assertEqual(payload[:2], b"\x80\x02")
                self.assertIs(pickle.loads(payload), expected[name])

        repo_root = Path(fsdp.__file__).resolve().parents[4]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(repo_root / "python"), env.get("PYTHONPATH", ""))))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        code = r"""
import base64
import pickle
payload = base64.b64decode(%r)
value = pickle.loads(payload)
from jittor.compat import fsdp2
assert value is fsdp2.DeviceMesh
""" % _LEGACY_PICKLES["DeviceMesh"]
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(repo_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        self.assertEqual(
            result.returncode, 0,
            "pickle-first legacy import failed:\n%s\n%s"
            % (result.stdout, result.stderr),
        )

    def test_registered_torch_modules_remain_idempotent(self):
        fsdp.install(jt.distributed, jt.__dict__)
        before = {name: sys.modules[name] for name in _REGISTERED_MODULES}
        symbol_paths = (
            ("torch.distributed.fsdp.wrap", "enable_wrap"),
            ("torch.distributed.fsdp.wrap", "wrap"),
            ("torch.distributed.tensor.parallel.style", "RowwiseParallel"),
            ("torch.distributed._functional_collectives", "AsyncCollectiveTensor"),
            (
                "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
                "CheckpointImpl",
            ),
            ("torch.distributed", "is_available"),
        )
        symbols_before = {
            path: getattr(importlib.import_module(path[0]), path[1])
            for path in symbol_paths
        }
        fsdp.install(jt.distributed, jt.__dict__)
        after = {name: sys.modules[name] for name in _REGISTERED_MODULES}
        symbols_after = {
            path: getattr(importlib.import_module(path[0]), path[1])
            for path in symbol_paths
        }
        self.assertEqual(len(before), 37)
        for name in _REGISTERED_MODULES:
            with self.subTest(module=name):
                self.assertIs(before[name], after[name])
        for path in symbol_paths:
            with self.subTest(symbol="%s.%s" % path):
                self.assertIs(symbols_before[path], symbols_after[path])

        fsdp_mod = importlib.import_module("torch.distributed.fsdp")
        self.assertIs(fsdp_mod.FSDP, fsdp.FullyShardedDataParallel)
        self.assertIs(fsdp_mod.FSDPModule, fsdp.FSDPModule)
        self.assertIs(
            importlib.import_module("torch.distributed.tensor").DTensor,
            fsdp.DTensor,
        )
        origin_values = (
            importlib.import_module("torch.distributed.fsdp.wrap").enable_wrap,
            importlib.import_module("torch.distributed.fsdp.wrap").wrap,
            importlib.import_module(
                "torch.distributed.tensor.parallel.style"
            ).RowwiseParallel,
            importlib.import_module(
                "torch.distributed._functional_collectives"
            ).AsyncCollectiveTensor,
            importlib.import_module(
                "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
            ).CheckpointImpl,
        )
        self.assertTrue(all(
            value.__module__ == installer.__name__ for value in origin_values
        ))

        original_device_mesh = dtensor.DeviceMesh
        created = object()
        try:
            dtensor.DeviceMesh = lambda *args, **kwargs: created
            self.assertIs(dtensor.init_device_mesh("cpu", (1,)), created)
        finally:
            dtensor.DeviceMesh = original_device_mesh

        original_fully_shard = api.fully_shard
        wrapped = object()
        marker = object()
        try:
            api.fully_shard = lambda module, **kwargs: wrapped
            wrap_mod = importlib.import_module("torch.distributed.fsdp.wrap")
            self.assertIs(wrap_mod.wrap(marker), wrapped)
        finally:
            api.fully_shard = original_fully_shard

        original_sync = grad_sync.sync_sharded_grads
        synced = object()
        try:
            grad_sync.sync_sharded_grads = lambda *args, **kwargs: synced
            self.assertIs(fsdp.FSDPModule().sync_sharded_grads(marker), synced)
        finally:
            grad_sync.sync_sharded_grads = original_sync

        original_distributed = common._in_true_distributed
        original_gather = common._all_gather_shards
        gathered = object()
        try:
            common._in_true_distributed = lambda: True
            common._all_gather_shards = lambda tensor: gathered
            collectives = importlib.import_module(
                "torch.distributed.fsdp._fully_shard._fsdp_collectives"
            )
            self.assertIs(collectives.all_gather(marker), gathered)
        finally:
            common._in_true_distributed = original_distributed
            common._all_gather_shards = original_gather

    def test_import_direction_and_package_discovery(self):
        # The file-size budgets that used to be asserted here are gone: they made
        # the gate red for growth rather than for a boundary violation, which is
        # what the rest of this test is about.
        package_path = Path(fsdp.__file__).resolve()
        for module in _OWNERSHIP:
            path = Path(module.__file__).resolve()
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            with self.subTest(module=module.__name__):
                self.assertNotIn("facade", source)
                self.assertNotIn("preserve_facade_origins", source)
                self.assertNotIn("_torch_fsdp2", source)
                self.assertNotIn("torch_fsdp2_compat", source)
                relative_imports = {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.level == 1 and node.module is None
                    for alias in node.names
                }
                if module is installer:
                    self.assertEqual(relative_imports, {
                        "api", "common", "compat_types", "config", "dtensor",
                        "grad_sync", "optimizer", "shard",
                    })

        repo_root = package_path.parents[4]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages
        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor.compat.fsdp2", packages)
        self.assertNotIn("jittor._torch_fsdp2", packages)
        self.assertNotIn("jittor.torch_fsdp2_compat", packages)


if __name__ == "__main__":
    unittest.main()
