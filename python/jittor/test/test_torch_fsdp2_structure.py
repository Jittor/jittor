"""Architecture contracts for the FSDP2 compatibility package."""

import ast
import enum
import hashlib
import importlib
import inspect
import json
import pickle
from pathlib import Path
import sys
import types
import unittest

import jittor as jt
from jittor import torch_fsdp2_compat as fsdp
from jittor._torch_fsdp2 import compat_types
from jittor._torch_fsdp2 import config
from jittor._torch_fsdp2 import dtensor
from jittor._torch_fsdp2 import fsdp_api
from jittor._torch_fsdp2 import grad_sync
from jittor._torch_fsdp2 import installer
from jittor._torch_fsdp2 import optimizer
from jittor._torch_fsdp2 import runtime
from jittor._torch_fsdp2 import shard_common
from jittor._torch_fsdp2 import shard_runtime


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
    shard_common: {
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
    shard_runtime: {
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
    fsdp_api: {
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


class _ModuleImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "jittor" or alias.name.startswith("jittor."):
                self.violations.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node):
        if node.level >= 2:
            self.violations.append((node.lineno, "." * node.level + (node.module or "")))
        elif node.level == 0 and node.module and (
            node.module == "jittor" or node.module.startswith("jittor.")
        ):
            self.violations.append((node.lineno, node.module))


class TestTorchFSDP2Structure(unittest.TestCase):
    def test_root_file_is_replaced_by_stable_package_facade(self):
        facade_path = Path(fsdp.__file__).resolve()
        self.assertEqual(facade_path.name, "__init__.py")
        self.assertEqual(facade_path.parent.name, "torch_fsdp2_compat")
        self.assertFalse((facade_path.parent.parent / "torch_fsdp2_compat.py").exists())
        self.assertIs(jt.torch_fsdp2_compat, fsdp)
        self.assertIsInstance(jt._torch_fsdp2, types.ModuleType)
        self.assertFalse(hasattr(fsdp, "__all__"))
        self.assertEqual(
            {name for name in vars(fsdp) if not name.startswith("_")},
            _PUBLIC_NAMES,
        )

    def test_private_ownership_and_facade_identity(self):
        names = set()
        for module, expected in _OWNERSHIP.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(set(module.FACADE_EXPORTS), expected)
                self.assertTrue(names.isdisjoint(expected))
                names.update(expected)
                for name in expected:
                    self.assertIs(getattr(fsdp, name), getattr(module, name))
        self.assertIs(fsdp._install_fsdp2_distributed, fsdp.install)
        self.assertIs(runtime.jt._module, jt)
        self.assertIs(runtime.nn._module, jt.nn)
        self.assertIs(runtime.fsdp._module, fsdp)

    def test_origin_restoration_handles_bound_methods(self):
        class DynamicModule:
            def execute(self):
                return None

        module = DynamicModule()
        runtime.preserve_facade_origins((module.execute,), source_module=__name__)
        self.assertEqual(
            DynamicModule.execute.__module__,
            "jittor.torch_fsdp2_compat",
        )

    def test_public_contract_signatures_reflection_and_pickle(self):
        public = sorted(name for name in vars(fsdp) if not name.startswith("_"))
        callables = {}
        for name in public:
            value = getattr(fsdp, name)
            if callable(value) and getattr(value, "__module__", None) == fsdp.__name__:
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
                callables[name] = {
                    "kind": kind,
                    "members": members,
                    "module": value.__module__,
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

    def test_registered_torch_modules_remain_idempotent(self):
        fsdp.install(jt.distributed, jt.__dict__)
        before = {name: sys.modules[name] for name in _REGISTERED_MODULES}
        fsdp.install(jt.distributed, jt.__dict__)
        after = {name: sys.modules[name] for name in _REGISTERED_MODULES}
        self.assertEqual(len(before), 37)
        for name in _REGISTERED_MODULES:
            with self.subTest(module=name):
                self.assertIs(before[name], after[name])

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
            value.__module__ == "jittor.torch_fsdp2_compat"
            for value in origin_values
        ))

        original_device_mesh = fsdp.DeviceMesh
        class PatchedDeviceMesh:
            pass
        try:
            fsdp.DeviceMesh = PatchedDeviceMesh
            fsdp.install(jt.distributed, jt.__dict__)
            self.assertIs(
                importlib.import_module("torch.distributed.tensor").DeviceMesh,
                PatchedDeviceMesh,
            )
            self.assertIs(jt.distributed.DeviceMesh, PatchedDeviceMesh)
        finally:
            fsdp.DeviceMesh = original_device_mesh
            fsdp.install(jt.distributed, jt.__dict__)

        original_fully_shard = fsdp.fully_shard
        wrapped = object()
        marker = object()
        try:
            fsdp.fully_shard = lambda module, **kwargs: wrapped
            wrap_mod = importlib.import_module("torch.distributed.fsdp.wrap")
            self.assertIs(wrap_mod.wrap(marker), wrapped)
        finally:
            fsdp.fully_shard = original_fully_shard

        original_sync = fsdp.sync_sharded_grads
        synced = object()
        try:
            fsdp.sync_sharded_grads = lambda *args, **kwargs: synced
            self.assertIs(fsdp.FSDPModule().sync_sharded_grads(marker), synced)
        finally:
            fsdp.sync_sharded_grads = original_sync

        original_distributed = fsdp._in_true_distributed
        original_gather = fsdp._all_gather_shards
        gathered = object()
        try:
            fsdp._in_true_distributed = lambda: True
            fsdp._all_gather_shards = lambda tensor: gathered
            collectives = importlib.import_module(
                "torch.distributed.fsdp._fully_shard._fsdp_collectives"
            )
            self.assertIs(collectives.all_gather(marker), gathered)
        finally:
            fsdp._in_true_distributed = original_distributed
            fsdp._all_gather_shards = original_gather

    def test_private_import_direction_and_file_budgets(self):
        facade_lines = Path(fsdp.__file__).read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(facade_lines), 80)
        for module in (runtime, *_OWNERSHIP):
            path = Path(module.__file__).resolve()
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=module.__name__):
                self.assertLessEqual(len(source.splitlines()), 400)
                visitor = _ModuleImportVisitor()
                visitor.visit(ast.parse(source, filename=str(path)))
                self.assertEqual(visitor.violations, [])
                if module is installer:
                    relative_imports = {
                        node.module
                        for node in ast.walk(ast.parse(source, filename=str(path)))
                        if isinstance(node, ast.ImportFrom) and node.level == 1
                    }
                    self.assertEqual(relative_imports, {"runtime"})

    def test_setup_declares_both_packages(self):
        setup_path = Path(fsdp.__file__).resolve().parents[3] / "setup.py"
        if not setup_path.is_file():
            self.skipTest("setup.py is only available in a source checkout")
        tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
        package_lists = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "setup"
            for keyword in node.keywords
            if keyword.arg == "packages" and isinstance(keyword.value, ast.List)
        ]
        self.assertEqual(len(package_lists), 1)
        packages = {
            item.value for item in package_lists[0].elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("jittor._torch_fsdp2", packages)
        self.assertIn("jittor.torch_fsdp2_compat", packages)


if __name__ == "__main__":
    unittest.main()
