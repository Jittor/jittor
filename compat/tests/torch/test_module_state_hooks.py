"""Portable module state hooks must preserve nested CUDA model state."""

import unittest

import jittor as jt
import torch

from _helpers import capability
from _helpers.common import selected_device_types


class _Portable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = torch.nn.Linear(2, 2).cuda()
        self.loads = []
        self._register_state_dict_hook(self._export)
        self.register_load_state_dict_pre_hook(self._restore)

    @staticmethod
    def _export(module, state, prefix, metadata):
        source = prefix + "wrapped."
        for key in tuple(state):
            if key.startswith(source):
                state[prefix + key[len(source):]] = state.pop(key)
        return state

    @staticmethod
    def _restore(module, state, prefix, metadata, strict, missing, unexpected, errors):
        module.loads.append(prefix)
        for key in tuple(state):
            if key.startswith(prefix):
                state[prefix + "wrapped." + key[len(prefix):]] = state.pop(key)

    def forward(self, value):
        return self.wrapped(value)


class TestModuleStateHooksCUDA(unittest.TestCase):
    def setUp(self):
        selected = selected_device_types()
        if selected is not None and "cuda" not in selected:
            self.skipTest("requires selected CUDA execution")
        capability.require_accelerator("cuda")
        self.scope = jt.flag_scope(use_cuda=1, backend_fallback="error")
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.fallbacks = int(jt.core.backend_fallback_count())

    def tearDown(self):
        self.assertEqual(int(jt.core.backend_fallback_count()), self.fallbacks)

    def test_nested_portable_state_loads_once_without_mutating_the_caller(self):
        plain = torch.nn.Sequential(torch.nn.Linear(2, 2).cuda())
        target = torch.nn.Sequential(_Portable())
        state = {key: value.detach().clone() for key, value in plain.state_dict().items()}
        keys = tuple(state)
        result = target.load_state_dict(state)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        self.assertEqual(target[0].loads, ["0."])
        self.assertEqual(tuple(state), keys)
        self.assertEqual(set(target.state_dict()), set(state))
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        value = torch.tensor([[.25, .75]], device="cuda")
        torch.testing.assert_close(target(value), plain(value), rtol=0, atol=0)

    def test_export_order_prefix_destination_and_removable_handles(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2).cuda(),
                                    torch.nn.Linear(2, 2).cuda())
        calls = []
        handles = []
        for name, module in (("left", model[0]), ("right", model[1]), ("root", model)):
            def record(module, state, prefix, metadata, name=name):
                calls.append((name, prefix))
            handles.append(module.register_state_dict_post_hook(record))
        destination = {}
        self.assertIs(model.state_dict(destination=destination, prefix="outer."), destination)
        self.assertEqual(calls, [("left", "outer.0."), ("right", "outer.1."),
                                 ("root", "outer.")])
        self.assertTrue(all(key.startswith("outer.") for key in destination))
        for handle in handles:
            handle.remove()
        model.state_dict()
        self.assertEqual(len(calls), 3)

    def test_pre_hook_errors_and_public_post_hook_returns_are_not_hidden(self):
        model = torch.nn.Linear(2, 2).cuda()
        state = model.state_dict()
        def refuse(module, state, prefix, metadata, strict, missing, unexpected, errors):
            errors.append("state rejected by caller")
        handle = model.register_load_state_dict_pre_hook(refuse)
        with self.assertRaisesRegex(RuntimeError, "state rejected by caller"):
            model.load_state_dict(state)
        handle.remove()
        model.load_state_dict(state)
        handle = model.register_state_dict_post_hook(lambda *args: {})
        with self.assertRaisesRegex(RuntimeError, "must return None"):
            model.state_dict()
        handle.remove()
        self.assertEqual(set(model.state_dict()), {"weight", "bias"})
