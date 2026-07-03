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
            MixedPrecisionPolicy,
            StateDictType,
            FullyShardedDataParallel,
            fully_shard,
        )
        from torch.distributed._composable.fsdp import fully_shard as composable_fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertTrue(torch.distributed.is_available())
        self.assertIs(fully_shard, composable_fully_shard)
        self.assertEqual(mesh.size("dp"), 1)

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
        module.set_reshard_after_backward(False)
        module.set_unshard_in_backward(False)
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm(False)
        module.set_post_optim_event(None)
        self.assertIs(module._get_fsdp_state(), getattr(module, "_fsdp_state", None))

        wrapped = FullyShardedDataParallel(torch.nn.Linear(3, 2))
        self.assertEqual(tuple(wrapped(x).shape), (4, 2))
        self.assertEqual(len(list(StateDictType)), 3)
        self.assertIs(checkpoint_wrapper(module), module)

    def test_dtensor_and_private_import_paths(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import (
            DTensor,
            Placement,
            Replicate,
            Shard,
            distribute_tensor,
        )
        from torch.distributed._tensor import distribute_tensor as legacy_distribute_tensor
        from torch.distributed.tensor.placement_types import Partial
        from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
        from torch.distributed.fsdp._fully_shard import fully_shard as package_fully_shard
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            FSDPMeshInfo,
            ShardPlacementResult,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertIs(fully_shard, package_fully_shard)

        dt = distribute_tensor(jt.array([1, 2, 3]), mesh, [Replicate()])
        dt2 = legacy_distribute_tensor(jt.array([1, 2, 3]), mesh, [Shard(0)])
        self.assertIsInstance(dt, DTensor)
        self.assertIsInstance(dt2, DTensor)
        self.assertIs(dt.to_local(), dt)
        self.assertTrue(isinstance(Shard(0), Placement))
        self.assertTrue(Partial().is_partial())

        self.assertIsInstance(FSDPMeshInfo(mesh=mesh), FSDPMeshInfo)
        self.assertIsInstance(ShardPlacementResult(shard_dim=0), ShardPlacementResult)
        self.assertIs(_get_mesh_info(mesh).mesh, mesh)

        module = torch.nn.Linear(2, 2)
        self.assertIs(parallelize_module(module), module)
        self.assertIsInstance(ParallelStyle(), ParallelStyle)


if __name__ == "__main__":
    unittest.main()
