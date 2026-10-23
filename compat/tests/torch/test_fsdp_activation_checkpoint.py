"""Public FSDP checkpoint routing and portable CUDA module state."""

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jittor as jt
import numpy as np
import torch

from _helpers import capability
from _helpers.common import selected_device_types
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import (
    ModuleWrapPolicy, transformer_auto_wrap_policy, lambda_auto_wrap_policy,
)
from jittor.compat.fsdp2._state_dict import _load_full_state_dict


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
        self.register_buffer("offset", torch.ones(2, device="cuda"))
        self.calls = 0

    def forward(self, value, scale=1.):
        self.calls += 1
        return (self.linear(value).tanh() + self.offset) * scale


class TestFSDPActivationCheckpoint(unittest.TestCase):
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
        jt.sync_all(True)
        self.assertEqual(int(jt.core.backend_fallback_count()), self.fallbacks)

    def test_custom_checkpoint_dispatch_preserves_options_and_module_aliases(self):
        calls = []
        def custom(function, *args, marker, **kwargs):
            calls.append((function, marker, kwargs))
            return function(*args, **kwargs)
        module = _Block()
        wrapped = checkpoint_wrapper(module, checkpoint_fn=custom, marker="sentinel")
        self.assertIsNot(wrapped, module)
        self.assertIs(wrapped.linear, module.linear)
        self.assertIs(wrapped.offset, module.offset)
        self.assertEqual([id(v) for v in wrapped.parameters()], [id(v) for v in module.parameters()])
        value = torch.ones((2, 2), device="cuda")
        result = wrapped(value, scale=2.)
        self.assertEqual(calls, [(module, "sentinel", {"scale": 2.})])
        torch.testing.assert_close(result, module(value, scale=2.), rtol=0, atol=0)
        self.assertEqual(list(dict(wrapped.named_parameters())), list(dict(module.named_parameters())))
        sequence = torch.nn.Sequential(module)
        self.assertIs(checkpoint_wrapper(sequence, checkpoint_fn=custom, marker=1)[0], module)

    def test_bottom_up_policy_precedence_and_repeated_application(self):
        model = torch.nn.Sequential(torch.nn.Sequential(_Block()), _Block())
        wrapped = []
        def wrap(module):
            wrapped.append(module)
            return checkpoint_wrapper(module, checkpoint_fn=lambda f, *a, **k: f(*a, **k))
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={_Block})
        self.assertTrue(lambda_auto_wrap_policy(model, True, 0, lambda_fn=lambda _: False))
        result = apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=wrap,
            check_fn=lambda _: self.fail("auto policy must take precedence"), auto_wrap_policy=policy)
        self.assertIsNone(result)
        self.assertEqual(len(wrapped), 2)
        self.assertIs(model[0][0]._checkpoint_wrapped_module, wrapped[0])
        self.assertIs(model[1]._checkpoint_wrapped_module, wrapped[1])
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrap,
                                      auto_wrap_policy=ModuleWrapPolicy({_Block}))
        self.assertEqual(len(wrapped), 2)
        value = torch.ones((2, 2), device="cuda")
        self.assertEqual(tuple(model(value).shape), (2, 2))

    def test_check_predicate_never_wraps_root_and_policy_can_prune(self):
        model = torch.nn.Sequential(torch.nn.Sequential(_Block()), _Block())
        selected = []
        def select(module):
            self.assertIsNot(module, model)
            return isinstance(module, _Block)
        def wrap(module):
            selected.append(module)
            return checkpoint_wrapper(module, checkpoint_fn=lambda f, *a, **k: f(*a, **k))
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrap, check_fn=select)
        self.assertEqual(len(selected), 2)
        other = torch.nn.Sequential(torch.nn.Sequential(_Block()), _Block())
        pruned = other[0]
        selected.clear()
        def policy(module, recurse, nonwrapped_numel):
            self.assertGreaterEqual(nonwrapped_numel, 0)
            return module is not pruned if recurse else isinstance(module, _Block)
        apply_activation_checkpointing(other, checkpoint_wrapper_fn=wrap, auto_wrap_policy=policy)
        self.assertEqual(len(selected), 1)
        self.assertIs(other[0], pruned)
        self.assertIsInstance(other[0][0], _Block)

    def test_nested_state_roundtrip_and_real_reentrant_parameter_updates(self):
        plain = torch.nn.Sequential(_Block(), _Block())
        saved = {key: value.detach().clone() for key, value in plain.state_dict().items()}
        target = torch.nn.Sequential(_Block(), _Block())
        apply_activation_checkpointing(
            target, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT),
            check_fn=lambda module: isinstance(module, _Block))
        keys = tuple(saved)
        result = target.load_state_dict(saved)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        self.assertEqual(tuple(saved), keys)
        self.assertEqual(set(target.state_dict()), set(saved))
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        value = torch.tensor([[.2, .4], [.6, .8]], device="cuda", requires_grad=True)
        torch.testing.assert_close(target(value), plain(value), rtol=0, atol=0)
        opt = torch.optim.SGD(target.parameters(), lr=.03)
        before = [v.detach().cpu().numpy().copy() for v in target.parameters()]
        opt.zero_grad()
        target(value,).square().mean().backward()
        for v in target.parameters():
            self.assertIsNotNone(v.grad)
            self.assertTrue(bool(v.grad.isfinite().all().item()))
        opt.step()
        self.assertTrue(any((old != new.detach().cpu().numpy()).any()
                            for old, new in zip(before, target.parameters())))
        # One output comparison forward, then the trained forward and backward replay.
        self.assertEqual([target[i].calls for i in range(2)], [3, 3])
        restored = torch.nn.Sequential(_Block(), _Block())
        restored.load_state_dict(target.state_dict())
        torch.testing.assert_close(restored(value), target(value), rtol=0, atol=0)

    def test_full_state_load_uses_exported_checkpoint_wrapper_key(self):
        module = _Block()
        wrapped = checkpoint_wrapper(
            module, checkpoint_fn=lambda function, *args, **kwargs: function(*args, **kwargs))
        parameter = module.linear.weight
        shard_value = jt.Var.copy(parameter.reshape(-1)).stop_grad()
        shard_value.start_grad()
        entry = SimpleNamespace(
            name="weight", owner=module.linear, attr="weight", shard=shard_value,
            shape=tuple(parameter.shape), numel=parameter.numel(), padded_numel=parameter.numel(),
            shard_numel=parameter.numel(), requires_grad=True, full_param=None)
        state = SimpleNamespace(
            true_fsdp_initialized=True, true_fsdp_params=[entry], true_fsdp_flat=False,
            true_fsdp_rank=0, true_fsdp_unsharded=False)
        object.__setattr__(module.linear, "_is_fsdp_module", True)
        object.__setattr__(module.linear, "_fsdp_state", state)
        object.__setattr__(module.linear, "weight", shard_value)
        object.__setattr__(shard_value, "_jittor_fsdp2_state", state)
        object.__setattr__(shard_value, "_jittor_fsdp2_entry", entry)

        exported = wrapped.state_dict()
        self.assertIn("linear.weight", exported)
        self.assertNotIn("_checkpoint_wrapped_module.linear.weight", exported)
        full_state = {key: value.detach().clone() for key, value in exported.items()}
        full_state["linear.weight"] = torch.ones_like(parameter) * 2
        invalid_state = dict(full_state, unexpected=torch.ones(1, device="cuda"))
        with self.assertRaisesRegex(RuntimeError, "state dict keys mismatch"):
            _load_full_state_dict(wrapped, invalid_state)

        _load_full_state_dict(wrapped, full_state)

        np.testing.assert_array_equal(
            shard_value.numpy().reshape(tuple(parameter.shape)),
            full_state["linear.weight"].numpy(),
        )
        self.assertEqual(set(wrapped.state_dict()), set(full_state))

    def test_default_mode_is_forwarded_without_remapping(self):
        module = _Block()
        calls = []
        def observe(function, *args, use_reentrant, **kwargs):
            calls.append(use_reentrant)
            return function(*args, **kwargs)
        with patch("torch.utils.checkpoint.checkpoint", observe):
            wrapped = checkpoint_wrapper(module)
        self.assertIs(wrapped.checkpoint_impl, CheckpointImpl.NO_REENTRANT)
        self.assertEqual(tuple(wrapped(torch.ones((2, 2), device="cuda")).shape), (2, 2))
        self.assertEqual(calls, [False])
