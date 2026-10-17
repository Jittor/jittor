"""Mocked checks for the MPI-free HCCL bootstrap on hosts without an NPU."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jittor.compat.torch.installers import distributed
from jittor.distributed import process_group


class HCCLBootstrapTests(unittest.TestCase):
    def test_hccl_backend_capability_is_advertised_only_on_acl(self):
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=1), has_cuda=False,
            compile_extern=SimpleNamespace(has_mpi=False))
        with patch.object(distributed, 'jt', fake_jt):
            self.assertTrue(distributed._api_dist_is_backend_available('hccl'))
            self.assertFalse(distributed._api_dist_is_backend_available('nccl'))
            fake_jt.compiler.has_acl = 0
            self.assertFalse(distributed._api_dist_is_backend_available('hccl'))

    def test_hccl_world_is_recognized(self):
        fake_jt = SimpleNamespace(world_size=2, rank=1, in_mpi=False)
        with (patch.object(process_group, 'jt', fake_jt),
              patch.dict(os.environ, {'JT_HCCL_WORLD_SIZE': '2'}, clear=True)):
            self.assertTrue(process_group.is_initialized())
            self.assertEqual(process_group.get_rank(), 1)
            self.assertEqual(process_group.get_world_size(), 2)

    def test_dynamic_hccl_bootstrap_sets_backend_and_rank(self):
        core = SimpleNamespace(Var=type('FakeVar', (), {}))
        external = SimpleNamespace(
            setup_hccl=Mock(), hccl_ops=SimpleNamespace(),
            rank=0, world_size=1, in_mpi=False)
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=1),
            flags=SimpleNamespace(use_acl=0),
            compile_extern=external, core=core)
        external.hccl_mod = SimpleNamespace(hccl_init=Mock())
        published = {}
        with (patch.object(distributed, 'jt', fake_jt),
              patch.object(distributed, 'set_env',
                           side_effect=lambda key, value: published.__setitem__(key, value)),
              patch.object(distributed, 'set_flag',
                           side_effect=lambda flags, key, value: setattr(flags, key, value)),
              patch.object(distributed, '_clear_stale_rendezvous'),
              patch.dict(os.environ, {
                  'JITTOR_TORCH_DISTRIBUTED_AUTO_INIT': '1',
                  'JT_HCCL_ROOTINFO_FILE': '/tmp/jittor-test-hccl-rootinfo',
                  'ASCEND_RT_VISIBLE_DEVICES': '0,1',
                  'LOCAL_RANK': '1',
              }, clear=True)):
            self.assertTrue(distributed._bootstrap_native_distributed(
                1, 2, backend='hccl'))
        external.setup_hccl.assert_called_once_with(no_mpi=True)
        external.hccl_mod.hccl_init.assert_called_once_with()
        self.assertEqual(published['JT_HCCL_WORLD_SIZE'], 2)
        self.assertEqual(published['JT_HCCL_RANK'], 1)
        self.assertEqual(published['JT_HCCL_LOCAL_RANK'], 1)
        self.assertEqual(published['use_hccl'], '1')
        self.assertEqual(fake_jt.flags.use_acl, 1)
        self.assertIs(core.Var.mpi_all_reduce,
                      distributed._bootstrap_var_all_reduce)
        self.assertIs(core.Var.mpi_broadcast,
                      distributed._bootstrap_var_broadcast)

    def test_hccl_global_rank_one_uses_local_zero_with_one_visible_npu(self):
        """vLLM may isolate each rank to one logical NPU before bootstrap."""
        core = SimpleNamespace(Var=type('FakeIsolatedNpuVar', (), {}))
        external = SimpleNamespace(
            setup_hccl=Mock(), hccl_ops=SimpleNamespace(),
            rank=0, world_size=1, in_mpi=False)
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=1),
            flags=SimpleNamespace(use_acl=0),
            compile_extern=external, core=core)
        external.hccl_mod = SimpleNamespace(hccl_init=Mock())
        published = {}
        with (patch.object(distributed, 'jt', fake_jt),
              patch.object(distributed, 'set_env',
                           side_effect=lambda key, value:
                           published.__setitem__(key, value)),
              patch.object(distributed, 'set_flag',
                           side_effect=lambda flags, key, value:
                           setattr(flags, key, value)),
              patch.object(distributed, '_clear_stale_rendezvous'),
              patch.dict(os.environ, {
                  'JITTOR_TORCH_DISTRIBUTED_AUTO_INIT': '1',
                  'JT_HCCL_ROOTINFO_FILE': '/tmp/jittor-test-isolated-hccl',
                  'ASCEND_RT_VISIBLE_DEVICES': '1',
                  'LOCAL_RANK': '1',
              }, clear=True)):
            self.assertTrue(distributed._bootstrap_native_distributed(
                1, 2, backend='hccl'))
        self.assertEqual(published['JT_HCCL_RANK'], 1)
        self.assertEqual(published['JT_HCCL_LOCAL_RANK'], 0)
        external.setup_hccl.assert_called_once_with(no_mpi=True)
        external.hccl_mod.hccl_init.assert_called_once_with()

    def test_dynamic_nccl_bootstrap_is_unchanged(self):
        core = SimpleNamespace(Var=type('FakeCudaVar', (), {}))
        external = SimpleNamespace(
            setup_nccl=Mock(), nccl_ops=SimpleNamespace(),
            rank=0, world_size=1, in_mpi=False)
        fake_jt = SimpleNamespace(
            has_cuda=True, flags=SimpleNamespace(use_cuda=0),
            compile_extern=external, core=core)
        published = {}
        with (patch.object(distributed, 'jt', fake_jt),
              patch.object(distributed, 'set_env',
                           side_effect=lambda key, value: published.__setitem__(key, value)),
              patch.object(distributed, 'set_flag',
                           side_effect=lambda flags, key, value: setattr(flags, key, value)),
              patch.object(distributed, '_clear_stale_rendezvous'),
              patch.dict(os.environ, {
                  'JITTOR_TORCH_DISTRIBUTED_AUTO_INIT': '1',
                  'JT_NCCL_ROOTINFO_FILE': '/tmp/jittor-test-nccl-rootinfo',
                  'CUDA_VISIBLE_DEVICES': '0,1',
                  'LOCAL_RANK': '1',
              }, clear=True)):
            self.assertTrue(distributed._bootstrap_native_distributed(
                1, 2, backend='nccl'))
        external.setup_nccl.assert_called_once_with(store=None)
        self.assertEqual(published['JT_NCCL_WORLD_SIZE'], 2)
        self.assertEqual(published['JT_NCCL_LOCAL_RANK'], 1)
        self.assertEqual(published['use_nccl'], '1')
        self.assertEqual(fake_jt.flags.use_cuda, 1)

    def test_hccl_gather_uses_the_requested_subgroup(self):
        tensor = SimpleNamespace(reshape=Mock())
        tensor.reshape.return_value = tensor
        result = object()
        gather = Mock(return_value=result)
        fake_jt = SimpleNamespace(compile_extern=SimpleNamespace(
            hccl_ops=SimpleNamespace(hccl_all_gather=gather)))
        group = SimpleNamespace(_backend_handle=5)
        with (patch.object(distributed, 'jt', fake_jt),
              patch.dict(os.environ, {'JT_HCCL_WORLD_SIZE': '2'}),
              patch.object(distributed._collectives, '_all_gather_shards',
                           side_effect=AssertionError('NCCL path used'))):
            self.assertIs(distributed._native_all_gather_flat(tensor, group),
                          result)
            self.assertIs(distributed._native_all_gather_flat(tensor), result)
        self.assertEqual(gather.call_args_list[0].args, (tensor, 5))
        self.assertEqual(gather.call_args_list[1].args, (tensor, 0))


if __name__ == '__main__':
    unittest.main()
