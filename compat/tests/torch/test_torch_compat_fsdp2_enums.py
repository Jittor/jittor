"""FSDP public enums must carry torch's integer member numbers.

Accelerate normalises plugin arguments with ``ShardingStrategy(int(value))``
and ``BackwardPrefetch(int(value))``. The shim used to publish string values,
which made those conversions raise ValueError before the FSDP model was ever
built. The members themselves must be integers as well, so ``int(member)`` and
identity lookups behave like the native enums.
"""

import unittest

import torch


try:
    from accelerate.utils import FullyShardedDataParallelPlugin
except ImportError:  # pragma: no cover - optional downstream dependency
    FullyShardedDataParallelPlugin = None


class TestFSDP2EnumValues(unittest.TestCase):
    def test_sharding_strategy_members_match_native_numbers(self):
        from torch.distributed.fsdp import ShardingStrategy

        expected = {
            "FULL_SHARD": 1,
            "SHARD_GRAD_OP": 2,
            "NO_SHARD": 3,
            "HYBRID_SHARD": 4,
            "_HYBRID_SHARD_ZERO2": 5,
        }
        self.assertEqual({m.name: m.value for m in ShardingStrategy}, expected)
        for name, number in expected.items():
            member = ShardingStrategy[name]
            self.assertIs(ShardingStrategy(number), member)
            self.assertEqual(int(member), number)

    def test_state_dict_type_members_match_native_numbers(self):
        from torch.distributed.fsdp import StateDictType

        expected = {
            "FULL_STATE_DICT": 1,
            "LOCAL_STATE_DICT": 2,
            "SHARDED_STATE_DICT": 3,
        }
        self.assertEqual({m.name: m.value for m in StateDictType}, expected)
        for name, number in expected.items():
            member = StateDictType[name]
            self.assertIs(StateDictType(number), member)
            self.assertEqual(int(member), number)

    def test_backward_prefetch_members_match_native_numbers(self):
        from torch.distributed.fsdp import BackwardPrefetch

        expected = {"BACKWARD_PRE": 1, "BACKWARD_POST": 2}
        self.assertEqual({m.name: m.value for m in BackwardPrefetch}, expected)
        for name, number in expected.items():
            member = BackwardPrefetch[name]
            self.assertIs(BackwardPrefetch(number), member)
            self.assertEqual(int(member), number)

    def test_unknown_numbers_and_names_are_rejected(self):
        from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy

        with self.assertRaises(ValueError):
            ShardingStrategy(6)
        with self.assertRaises(KeyError):
            BackwardPrefetch["NO_PREFETCH"]


@unittest.skipIf(
    FullyShardedDataParallelPlugin is None, "Accelerate is not installed"
)
class TestAccelerateFSDPPluginEnumArguments(unittest.TestCase):
    def test_plugin_accepts_integer_sharding_strategy(self):
        # Accelerate leaves a non-string FSDP1 value untouched, so an integer
        # must at least construct the plugin without raising.
        plugin = FullyShardedDataParallelPlugin(sharding_strategy=1)
        self.assertEqual(plugin.sharding_strategy, 1)

    def test_plugin_resolves_named_strategy_through_integer_value(self):
        # This is the path that used to fail: Accelerate maps the name to the
        # native member number and then calls ``ShardingStrategy(1)``.
        from torch.distributed.fsdp import ShardingStrategy

        plugin = FullyShardedDataParallelPlugin(sharding_strategy="FULL_SHARD")
        self.assertIs(plugin.sharding_strategy, ShardingStrategy.FULL_SHARD)

    def test_plugin_accepts_integer_backward_prefetch(self):
        from torch.distributed.fsdp import BackwardPrefetch

        plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="NO_SHARD", backward_prefetch=2
        )
        self.assertIs(plugin.backward_prefetch, BackwardPrefetch.BACKWARD_POST)


if __name__ == "__main__":
    unittest.main()
