"""FSDP2 compatibility tests for ``import jittor as torch``.

Run:
    python -m jittor.test.test_torch_compat_fsdp2
"""
import unittest
import numpy as np
import jittor as torch
import jittor as jt


class TestFSDP2Compat(unittest.TestCase):
    def test_single_rank_fully_shard_preserves_math_and_state(self):
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            DataParallelMeshDims,
            FSDPModule,
            FlatParameter,
            FullOptimStateDictConfig,
            FullStateDictConfig,
            MixedPrecisionPolicy,
            OptimStateKeyType,
            StateDictType,
            StateDictSettings,
            FullyShardedDataParallel,
            ShardedGradScaler,
            fully_shard,
            share_comm_ctx,
        )
        from torch.distributed._composable.fsdp import fully_shard as composable_fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint,
            checkpoint_wrapper,
        )

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertTrue(torch.distributed.is_available())
        self.assertIs(fully_shard, composable_fully_shard)
        self.assertEqual(mesh.size("dp"), 1)
        self.assertEqual(mesh.size(mesh_dim=0), 1)
        with mesh:
            pass
        self.assertEqual(DataParallelMeshDims(shard="dp").shard_names, ("dp",))

        module = torch.nn.Linear(3, 2)
        x = jt.array(np.random.RandomState(11).randn(4, 3).astype("float32"))
        ref = module(x).numpy()
        param_ids = [id(p) for p in module.parameters()]

        returned = fully_shard(
            module,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32),
            offload_policy=CPUOffloadPolicy(),
            dp_mesh_dims=DataParallelMeshDims(shard="dp"),
        )

        self.assertIs(returned, module)
        self.assertIsInstance(module, FSDPModule)
        self.assertEqual([id(p) for p in module.parameters()], param_ids)
        np.testing.assert_allclose(module(x).numpy(), ref, atol=1e-6)
        self.assertEqual(sorted(module.state_dict().keys()), ["bias", "weight"])

        grads = jt.grad((module(x) ** 2).sum(), list(module.parameters()))
        self.assertTrue(all(float(jt.abs(g).sum().item()) > 0 for g in grads))

        module.unshard()
        module.reshard()
        module.set_requires_gradient_sync(False)
        module.set_requires_all_reduce(False)
        module.set_all_reduce_hook(lambda *args, **kwargs: None, stream=None)
        module.set_allocate_memory_from_process_group_for_comm(True)
        module.set_custom_all_gather(lambda *args, **kwargs: None)
        module.set_custom_reduce_scatter(lambda *args, **kwargs: None)
        module.set_reduce_scatter_unused_params(True)
        module.set_reduce_scatter_max_input_buffers(2)
        module.set_separate_reduce_scatter_group(None)
        module.set_reshard_after_backward(False)
        module.set_unshard_in_backward(False)
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm("NCCL")
        module.set_post_optim_event(None)
        module.reset_iter_state()
        with share_comm_ctx([module]):
            pass
        state = module._get_fsdp_state()
        self.assertTrue(callable(state.all_reduce_hook))
        self.assertTrue(state.allocate_memory_from_process_group_for_comm)
        self.assertTrue(callable(state.custom_all_gather))
        self.assertTrue(callable(state.custom_reduce_scatter))
        self.assertTrue(state.reduce_scatter_unused_params)
        self.assertEqual(state.reduce_scatter_max_input_buffers, 2)
        self.assertIsNone(state.separate_reduce_scatter_group)
        self.assertTrue(state.iter_state_reset)
        self.assertTrue(state.share_comm_ctx)
        self.assertEqual(state.symm_mem_for_comm, "NCCL")
        self.assertIs(module._get_fsdp_state(), getattr(module, "_fsdp_state", None))

        wrapped = FullyShardedDataParallel(torch.nn.Linear(3, 2))
        self.assertEqual(tuple(wrapped(x).shape), (4, 2))
        self.assertEqual(list(wrapped.buffers()), [])
        self.assertEqual(list(wrapped.named_buffers()), [])
        self.assertEqual(len(list(StateDictType)), 3)
        settings = StateDictSettings(
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(),
            FullOptimStateDictConfig(),
        )
        self.assertIs(settings.state_dict_type, StateDictType.FULL_STATE_DICT)
        self.assertIs(OptimStateKeyType.PARAM_NAME, OptimStateKeyType.PARAM_NAME)
        self.assertIsNotNone(FlatParameter(torch.ones(1)))
        FullyShardedDataParallel.set_state_dict_type(
            wrapped,
            StateDictType.LOCAL_STATE_DICT,
            state_dict_config=FullStateDictConfig(),
        )
        self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.LOCAL_STATE_DICT)
        with FullyShardedDataParallel.state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT):
            self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.SHARDED_STATE_DICT)
        self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.LOCAL_STATE_DICT)
        scaler = ShardedGradScaler(enabled=False)
        self.assertEqual(scaler.state_dict(), {"enabled": False})
        scaler.load_state_dict({"enabled": True})
        self.assertEqual(scaler.state_dict(), {"enabled": True})
        self.assertIs(checkpoint_wrapper(module), module)
        self.assertEqual(tuple(checkpoint(module, jt.ones((1, 3))).shape), (1, 2))

    def test_dtensor_and_private_import_paths(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import (
            DTensor,
            Placement,
            Replicate,
            Shard,
            distribute_tensor,
            ones,
            zeros,
            full,
            randn,
        )
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
        from torch.distributed.tensor._api import empty as api_empty
        from torch.distributed._tensor import distribute_tensor as legacy_distribute_tensor
        from torch.distributed._tensor import linspace as legacy_linspace
        from torch.distributed._tensor.device_mesh import init_device_mesh as legacy_init_device_mesh
        from torch.distributed.tensor.placement_types import Partial
        from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
        from torch.distributed.tensor.parallel.api import parallelize_module as api_parallelize_module
        from torch.distributed.tensor.parallel.loss import loss_parallel
        from torch.distributed.tensor.parallel.style import RowwiseParallel
        from torch.distributed.fsdp.wrap import CustomPolicy, ModuleWrapPolicy, _or_policy
        from torch.distributed.fsdp._fully_shard import fully_shard as package_fully_shard
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state_if_fully_sharded_module,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
            all_gather,
            reduce_scatter,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param import (
            FlatParameter as PrivateFlatParameter,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            FSDPMeshInfo,
            ShardPlacementResult,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_init import (
            _get_mesh_info,
            _get_post_forward_mesh_info,
        )

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertEqual(legacy_init_device_mesh("cpu", (1,)).size(mesh_dim=0), 1)
        self.assertIs(fully_shard, package_fully_shard)

        dt = distribute_tensor(jt.array([1, 2, 3]), mesh, [Replicate()])
        dt2 = legacy_distribute_tensor(jt.array([1, 2, 3]), mesh, [Shard(0)])
        for tensor in (
            ones((2, 3), device_mesh=mesh, placements=[Replicate()]),
            zeros(2, 3, device_mesh=mesh, placements=[Replicate()]),
            full((2, 3), 7, device_mesh=mesh, placements=[Replicate()]),
            randn(2, 3, device_mesh=mesh, placements=[Replicate()]),
            api_empty(2, 3, device_mesh=mesh, placements=[Replicate()]),
            legacy_linspace(0, 1, 3, device_mesh=mesh, placements=[Replicate()]),
        ):
            self.assertIsInstance(tensor, DTensor)
            self.assertTrue(hasattr(tensor, "to_local"))
        self.assertIsInstance(dt, DTensor)
        self.assertIsInstance(dt2, DTensor)
        self.assertIs(dt.to_local(), dt)
        self.assertTrue(isinstance(Shard(0), Placement))
        self.assertTrue(Partial().is_partial())
        self.assertEqual(np.asarray(DTensor(jt.array([1, 2, 3]))).tolist(), [1, 2, 3])
        self.assertIsInstance(DTensorSpec(mesh=mesh, placements=[Replicate()]), DTensorSpec)

        self.assertIsInstance(FSDPMeshInfo(mesh=mesh), FSDPMeshInfo)
        self.assertIsInstance(ShardPlacementResult(shard_dim=0), ShardPlacementResult)
        self.assertIs(_get_mesh_info(mesh).mesh, mesh)
        self.assertIs(_get_post_forward_mesh_info(mesh).mesh, mesh)

        module = torch.nn.Linear(2, 2)
        self.assertIs(parallelize_module(module), module)
        self.assertIs(api_parallelize_module(module), module)
        self.assertIsInstance(ParallelStyle(), ParallelStyle)
        self.assertIsInstance(RowwiseParallel(), ParallelStyle)
        self.assertTrue(ModuleWrapPolicy([torch.nn.Linear])(module, False, 0))
        self.assertTrue(CustomPolicy(lambda *args: True)(module, False, 0))
        self.assertTrue(_or_policy(module, False, 0, policies=[lambda **kwargs: True]))
        with loss_parallel():
            pass
        fully_shard(module)
        self.assertIs(_lazy_init(module._get_fsdp_state()), module._get_fsdp_state())
        self.assertIs(_get_module_fsdp_state(module), module._get_fsdp_state())
        self.assertIs(
            _get_module_fsdp_state_if_fully_sharded_module(module),
            module._get_fsdp_state(),
        )
        self.assertIs(all_gather(dt), dt)
        self.assertIs(reduce_scatter(dt), dt)
        self.assertIsNotNone(PrivateFlatParameter(torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
