"""The DDP marker carries the field the native optimizer step guard reads.

`jittor/nn/parallel/distributed_data_parallel.py::_assert_ddp_step_ready`
refuses an update when a parameter's `_jittor_ddp_state` says gradients were
only accumulated locally. It reads `state.has_unsynced_grads` whenever
`state.world_size > 1`, but compat DDP used to build the marker as
`SimpleNamespace(sync_enabled, world_size)`: every two-rank
`loss.backward(); optimizer.step()` raised AttributeError before touching the
optimizer, while one-rank runs passed because `world_size > 1` short-circuits.

Run:
    python -m pytest compat/tests/torch/test_torch_ddp_step_guard.py
"""

import types
import unittest

import torch
import jittor as jt


class TestDDPStepGuardMarker(unittest.TestCase):
    def _ddp_parameter_and_state(self):
        module = torch.nn.Linear(4, 3)
        ddp = torch.nn.parallel.DistributedDataParallel(module)
        for _name, parameter in ddp.named_parameters():
            state = getattr(parameter, "_jittor_ddp_state", None)
            if state is not None:
                return parameter, state
        self.fail("DDP did not stamp a parameter with _jittor_ddp_state")

    def test_marker_has_the_field_the_native_guard_reads(self):
        _parameter, state = self._ddp_parameter_and_state()
        # These are the two members of `world_size > 1 and has_unsynced_grads`.
        self.assertIsInstance(state.world_size, int)
        self.assertFalse(state.has_unsynced_grads)
        self.assertTrue(state.sync_enabled)

    def test_native_step_guard_reads_the_field_on_more_than_one_rank(self):
        from jittor.nn.parallel.distributed_data_parallel import (
            _assert_ddp_step_ready,
        )

        parameter, state = self._ddp_parameter_and_state()
        optimizer = types.SimpleNamespace(
            param_groups=[{"params": [parameter]}])
        original_world_size = state.world_size
        try:
            # One rank short-circuits before the attribute is read; two ranks
            # take the branch that used to raise AttributeError.
            state.world_size = 2
            _assert_ddp_step_ready(optimizer)

            # And the field is not decorative: a local-only accumulation is
            # still refused by the native guard.
            state.has_unsynced_grads = True
            with self.assertRaises(jt.JittorError):
                _assert_ddp_step_ready(optimizer)
        finally:
            state.world_size = original_world_size
            state.has_unsynced_grads = False


if __name__ == "__main__":
    unittest.main()
