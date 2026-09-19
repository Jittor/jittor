"""Focused public config defaults for the FSDP2 torch shim."""

import unittest

import torch


class TestFSDP2ConfigDefaults(unittest.TestCase):
    def test_state_dict_config_defaults_match_torch_26(self):
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            OptimStateDictConfig,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictConfig,
        )

        self.assertFalse(StateDictConfig().offload_to_cpu)
        self.assertTrue(OptimStateDictConfig().offload_to_cpu)
        self.assertFalse(FullStateDictConfig().offload_to_cpu)
        self.assertTrue(FullOptimStateDictConfig().offload_to_cpu)
        self.assertFalse(ShardedStateDictConfig().offload_to_cpu)
        self.assertFalse(ShardedStateDictConfig()._use_dtensor)
        self.assertTrue(ShardedOptimStateDictConfig().offload_to_cpu)
        self.assertFalse(ShardedOptimStateDictConfig()._use_dtensor)

    def test_sharded_dtensor_flag_is_preserved(self):
        from torch.distributed.fsdp import (
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
        )

        self.assertTrue(ShardedStateDictConfig(_use_dtensor=True)._use_dtensor)
        self.assertTrue(ShardedOptimStateDictConfig(_use_dtensor=True)._use_dtensor)


if __name__ == "__main__":
    unittest.main()
