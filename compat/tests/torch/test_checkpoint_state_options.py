"""DCP options and FQN restoration, runnable against independent binary Torch.

CHECKPOINT_DEVICE selects the actual execution device. CHECKPOINT_OPTIONS_REPORT
optionally records the deterministic continuation for cross-runtime comparison.
"""

import copy
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    get_state_dict, set_model_state_dict, set_optimizer_state_dict, set_state_dict,
)

DEVICE = os.environ.get("CHECKPOINT_DEVICE", "cpu")


def tensor(value):
    native = sys.modules.get("jittor")
    return torch.is_tensor(value) or (native is not None and isinstance(value, native.Var))


def assert_resident(value):
    assert value.device.type == DEVICE
    native = sys.modules.get("jittor")
    if native is not None:
        value.sync()
        backend, _ = native.core.dispatch_context([value])
        assert backend == ("cuda" if DEVICE == "cuda" else "cpu")
        if DEVICE == "cuda":
            assert value.location() == "device"


def snapshot(value):
    if tensor(value):
        return torch.clone(value.detach())
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return copy.deepcopy(value)


def numbers(value):
    if tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {key: numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numbers(item) for item in value]
    return value


class Blocks(torch.nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.ab = torch.nn.Linear(2, 2)
        self.register_buffer("counter", torch.tensor(3.0))
        self.a.register_buffer("offset", torch.tensor([.01, -.02]))
        if tied:
            self.ab.weight = self.a.weight
        self.to(DEVICE)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = np.arange(parameter.numel(), dtype=np.float32).reshape(tuple(parameter.shape))
                parameter.copy_(torch.tensor(values / 20 + .03 * (index + 1), device=DEVICE))

    def forward(self, x):
        return self.ab(self.a(x) + self.a.offset)


def optimizer(model, reverse=False):
    parameters = list(model.parameters())
    if reverse:
        parameters.reverse()
    return torch.optim.AdamW(parameters, lr=.017, betas=(.7, .93), eps=1e-4, weight_decay=.02)


def update(model, optim):
    optim.zero_grad()
    x = torch.tensor([[.2, -.3], [.7, .4], [-.1, .8]], device=DEVICE, requires_grad=True)
    target = torch.tensor([[.1, -.2], [.3, .2], [-.4, .6]], device=DEVICE)
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    for value in (x, output, loss, x.grad):
        assert_resident(value)
    for parameter in model.parameters():
        assert_resident(parameter)
        if parameter.grad is not None:
            assert_resident(parameter.grad)
    for buffer in model.buffers():
        assert_resident(buffer)
    gradients = {name: snapshot(parameter.grad) for name, parameter in model.named_parameters()}
    gradients["input"] = snapshot(x.grad)
    optim.step()
    return float(loss.item()), gradients


class TestCheckpointStateOptions(unittest.TestCase):
    def setUp(self):
        self.jt = sys.modules.get("jittor")
        if self.jt is not None:
            self.scope = self.jt.runtime.scope(use_cuda=DEVICE == "cuda", backend_fallback="error")
            self.scope.__enter__()
            self.fallbacks = self.jt.core.backend_fallback_count()

    def tearDown(self):
        if self.jt is not None:
            try:
                self.jt.sync_all()
                self.assertEqual(self.jt.core.backend_fallback_count(), self.fallbacks)
            finally:
                self.scope.__exit__(None, None, None)

    def assert_tree(self, actual, expected):
        if tensor(expected):
            np.testing.assert_allclose(numbers(actual), numbers(expected), atol=2e-6, rtol=2e-5)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                self.assert_tree(actual[key], value)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_selected_model_prefix_boundary_and_buffer(self):
        model = Blocks()
        selected = get_model_state_dict(model, submodules={model.a})
        self.assertEqual(set(selected), {"a.weight", "a.bias", "a.offset"})
        self.assert_tree(selected["a.offset"], model.a.offset)

    def test_remove_prefix_and_restore_into_child(self):
        model = Blocks()
        selected = snapshot(get_model_state_dict(model, submodules={model.a},
            options=StateDictOptions(keep_submodule_prefixes=False)))
        self.assertEqual(set(selected), {"weight", "bias", "offset"})
        with torch.no_grad():
            model.a.weight.fill_(4.)
        set_model_state_dict(model.a, selected)
        self.assert_tree(model.a.weight, selected["weight"])

    def test_multiple_submodules_keep_unique_prefixes(self):
        model = Blocks()
        selected = get_model_state_dict(model, submodules={model.a, model.ab})
        self.assertEqual(set(selected), {"a.weight", "a.bias", "a.offset", "ab.weight", "ab.bias"})

    def test_combined_submodule_get_preserves_full_optimizer_fqns(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        for keep in (False, True):
            with self.subTest(keep=keep):
                options = StateDictOptions(keep_submodule_prefixes=keep, cpu_offload=True)
                state, optim_state = get_state_dict(model, optim, submodules={model.a}, options=options)
                self.assertEqual(set(state), {"a.weight", "a.bias", "a.offset"} if keep
                                 else {"weight", "bias", "offset"})
                self.assertEqual(set(optim_state["state"]), set(dict(model.named_parameters())))
                self.assert_tree(optim_state, get_optimizer_state_dict(model, optim,
                    submodules={model.a}, options=options))
                self.assertTrue(all(value.device.type == "cpu" for value in state.values()))

    def test_module_keyed_state_restores_selected_model(self):
        model = Blocks()
        selected = snapshot(get_model_state_dict(model.a))
        untouched = snapshot(get_model_state_dict(model.ab))
        with torch.no_grad():
            model.a.bias.fill_(7.)
        incompatible = set_model_state_dict(model, {model.a: selected}, options=StateDictOptions(strict=False))
        self.assert_tree(get_model_state_dict(model.a), selected)
        self.assert_tree(get_model_state_dict(model.ab), untouched)
        self.assertEqual(set(incompatible.missing_keys), {"counter", "ab.weight", "ab.bias"})

    def test_ignore_frozen_with_submodule_selection(self):
        model = Blocks()
        model.a.bias.requires_grad_(False)
        selected = get_model_state_dict(model, submodules={model.a},
            options=StateDictOptions(ignore_frozen_params=True))
        self.assertEqual(set(selected), {"a.weight", "a.offset"})

    def test_ignore_frozen_does_not_filter_optimizer_state(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        model.a.bias.requires_grad_(False)
        state = get_optimizer_state_dict(model, optim, options=StateDictOptions(ignore_frozen_params=True))
        self.assertIn("a.bias", state["state"])
        self.assertIn("a.bias", state["param_groups"][0]["params"])

    def test_strict_missing_optimizer_state_rejected(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        original = snapshot(get_optimizer_state_dict(model, optim))
        partial = snapshot(original)
        del partial["state"]["a.weight"]
        partial["param_groups"][0]["lr"] = .9
        with self.assertRaisesRegex(RuntimeError, "a.weight"):
            set_optimizer_state_dict(model, optim, partial, options=StateDictOptions(strict=True))
        self.assert_tree(get_optimizer_state_dict(model, optim), original)

    def test_nonstrict_missing_optimizer_state_resets_only_that_parameter(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        partial = snapshot(get_optimizer_state_dict(model, optim))
        del partial["state"]["a.weight"]
        set_optimizer_state_dict(model, optim, partial, options=StateDictOptions(strict=False))
        self.assert_tree(get_optimizer_state_dict(model, optim), partial)
        update(model, optim)
        state = get_optimizer_state_dict(model, optim)["state"]
        self.assertEqual(float(state["a.weight"]["step"].item()), 1.)
        self.assertEqual(float(state["ab.weight"]["step"].item()), 2.)

    def test_strict_missing_frozen_optimizer_state_allowed(self):
        model = Blocks()
        model.a.bias.requires_grad_(False)
        optim = optimizer(model)
        update(model, optim)
        state = snapshot(get_optimizer_state_dict(model, optim))
        self.assertNotIn("a.bias", state["state"])
        set_optimizer_state_dict(model, optim, state)
        self.assert_tree(get_optimizer_state_dict(model, optim), state)

    def test_unknown_optimizer_state_is_ignored(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        state = snapshot(get_optimizer_state_dict(model, optim))
        extra = snapshot(state)
        extra["state"]["not_a_parameter"] = snapshot(state["state"]["a.weight"])
        set_optimizer_state_dict(model, optim, extra)
        self.assert_tree(get_optimizer_state_dict(model, optim), state)

    def test_shared_parameter_alternate_fqn_load(self):
        model = Blocks(tied=True)
        optim = optimizer(model)
        update(model, optim)
        canonical = snapshot(get_optimizer_state_dict(model, optim))
        alternate = snapshot(canonical)
        alternate["state"]["ab.weight"] = alternate["state"].pop("a.weight")
        alternate["param_groups"][0]["params"] = [
            "ab.weight" if name == "a.weight" else name for name in alternate["param_groups"][0]["params"]]
        set_optimizer_state_dict(model, optim, alternate)
        self.assert_tree(get_optimizer_state_dict(model, optim), canonical)
        self.assertIs(model.a.weight, model.ab.weight)

    def test_shared_parameter_flattened_alternate_fqn_load(self):
        model = Blocks(tied=True)
        optim = optimizer(model)
        update(model, optim)
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        canonical = snapshot(get_optimizer_state_dict(model, optim, options=options))
        alternate = {key.replace(".a.weight.", ".ab.weight."): value for key, value in canonical.items()}
        set_optimizer_state_dict(model, optim, alternate, options=options)
        self.assert_tree(get_optimizer_state_dict(model, optim, options=options), canonical)

    def test_reordered_parameter_slots_consume_saved_options(self):
        model = Blocks()
        original = optimizer(model)
        update(model, original)
        state = snapshot(get_optimizer_state_dict(model, original))
        reordered = optimizer(model, reverse=True)
        reordered.param_groups[0]["lr"] = 4.
        reordered.param_groups[0]["betas"] = (.5, .6)
        params = state["param_groups"][0]["params"]
        state["param_groups"][0]["params"] = params[1:] + params[:1]
        saved_model = snapshot(get_model_state_dict(model))
        expected_loss, expected_gradients = update(model, original)
        expected_model = snapshot(get_model_state_dict(model))
        set_model_state_dict(model, saved_model)
        set_optimizer_state_dict(model, reordered, state)
        restored = get_optimizer_state_dict(model, reordered)
        self.assert_tree(restored["state"], state["state"])
        self.assertEqual(reordered.param_groups[0]["lr"], .017)
        self.assertEqual(reordered.param_groups[0]["betas"], (.7, .93))
        loss, gradients = update(model, reordered)
        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assert_tree(gradients, expected_gradients)
        self.assert_tree(get_model_state_dict(model), expected_model)

    def test_empty_optimizer_selection_joint_get(self):
        model = Blocks()
        for optimizers in ((), iter(())):
            state, optim_state = get_state_dict(model, optimizers)
            self.assert_tree(state, get_model_state_dict(model))
            self.assertEqual(optim_state, {})
        with self.assertRaises(RuntimeError):
            get_optimizer_state_dict(model, ())
        result = set_state_dict(model, iter(()), model_state_dict=state, optim_state_dict={})
        self.assertEqual(result.missing_keys, [])

    def test_partial_group_nonstrict_consumes_options(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        state = snapshot(get_optimizer_state_dict(model, optim))
        del state["state"]["a.bias"]
        state["param_groups"][0]["params"].remove("a.bias")
        state["param_groups"][0]["lr"] = .023
        set_optimizer_state_dict(model, optim, state, options=StateDictOptions(strict=False))
        restored = get_optimizer_state_dict(model, optim)
        self.assertNotIn("a.bias", restored["state"])
        self.assertEqual(optim.param_groups[0]["lr"], .023)
        self.assertEqual(set(restored["param_groups"][0]["params"]), set(dict(model.named_parameters())))

    def test_duplicate_shared_parameter_aliases_rejected(self):
        model = Blocks(tied=True)
        optim = optimizer(model)
        update(model, optim)
        state = snapshot(get_optimizer_state_dict(model, optim))
        state["state"]["ab.weight"] = snapshot(state["state"]["a.weight"])
        state["param_groups"][0]["params"].append("ab.weight")
        with self.assertRaises(ValueError):
            set_optimizer_state_dict(model, optim, state)

    def test_strict_failure_does_not_load_an_earlier_optimizer(self):
        model = Blocks()
        first = torch.optim.AdamW(model.a.parameters(), lr=.01)
        second = torch.optim.AdamW(model.ab.parameters(), lr=.02)
        update(model, first)
        update(model, second)
        state = snapshot(get_optimizer_state_dict(model, (first, second)))
        original = snapshot(get_optimizer_state_dict(model, first))
        del state["state"]["ab.bias"]
        state["param_groups"][0]["lr"] = 4.
        with self.assertRaises(RuntimeError):
            set_optimizer_state_dict(model, (first, second), state)
        if self.jt is not None:
            # DCP builds all native load plans before applying either one.
            self.assert_tree(get_optimizer_state_dict(model, first), original)

    def test_full_flatten_loading_is_explicitly_unsupported(self):
        if self.jt is None:
            self.skipTest("upstream currently fails with KeyError; test the explicit shim refusal")
        model = Blocks()
        optim = optimizer(model)
        options = StateDictOptions(full_state_dict=True, flatten_optimizer_state_dict=True)
        state = get_optimizer_state_dict(model, optim, options=options)
        with self.assertRaisesRegex(NotImplementedError, "full_state_dict"):
            set_optimizer_state_dict(model, optim, state, options=options)
        original = snapshot(get_model_state_dict(model))
        replacement = {key: torch.zeros_like(value) for key, value in original.items()}
        with self.assertRaisesRegex(NotImplementedError, "full_state_dict"):
            set_state_dict(model, optim, model_state_dict=replacement, optim_state_dict=state, options=options)
        self.assert_tree(get_model_state_dict(model), original)

    def test_joint_optimizer_only_loading_keeps_model_untouched(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        saved = snapshot(get_optimizer_state_dict(model, optim))
        update(model, optim)
        current_model = snapshot(get_model_state_dict(model))
        result = set_state_dict(model, optim, model_state_dict={}, optim_state_dict=saved)
        self.assertFalse(result.missing_keys)
        self.assert_tree(get_model_state_dict(model), current_model)
        self.assert_tree(get_optimizer_state_dict(model, optim), saved)

    def test_empty_optimizer_payload_rejected_before_model_loading(self):
        model = Blocks()
        optim = optimizer(model)
        update(model, optim)
        original = snapshot(get_model_state_dict(model))
        with self.assertRaises(RuntimeError):
            set_optimizer_state_dict(model, optim, {})
        replacement = {key: torch.zeros_like(value) for key, value in original.items()}
        with self.assertRaises(RuntimeError):
            set_state_dict(model, optim, model_state_dict=replacement, optim_state_dict={})
        self.assert_tree(get_model_state_dict(model), original)

    def test_stateless_sgd_checkpoint_load(self):
        model = Blocks()
        optim = torch.optim.SGD(model.parameters(), lr=.1)
        state = snapshot(get_optimizer_state_dict(model, optim))
        self.assertEqual(state["state"], {})
        set_optimizer_state_dict(model, optim, state)

    def test_resumed_cpu_offloaded_options_match_next_step(self):
        for flattened, full in ((False, False), (False, True), (True, False)):
            with self.subTest(flattened=flattened, full=full):
                model = Blocks()
                optim = optimizer(model)
                update(model, optim)
                options = StateDictOptions(cpu_offload=True, flatten_optimizer_state_dict=flattened,
                                           full_state_dict=full)
                state = snapshot(get_state_dict(model, optim, options=options))
                expected_loss, expected_gradients = update(model, optim)
                expected = snapshot(get_state_dict(model, optim, options=options))
                set_state_dict(model, optim, model_state_dict=state[0], optim_state_dict=state[1], options=options)
                loss, gradients = update(model, optim)
                self.assertAlmostEqual(loss, expected_loss, places=6)
                self.assert_tree(gradients, expected_gradients)
                self.assert_tree(get_state_dict(model, optim, options=options), expected)
                self.assertTrue(all(parameter.device.type == DEVICE for parameter in model.parameters()))
                resident = get_optimizer_state_dict(model, optim)["state"]
                for fields in resident.values():
                    for key, value in fields.items():
                        if tensor(value) and key != "step":
                            assert_resident(value)
                if not flattened and not full and os.environ.get("CHECKPOINT_OPTIONS_REPORT"):
                    report = {"device": DEVICE, "torch": torch.__version__, "torch_file": torch.__file__,
                              "binary_torch": self.jt is None and hasattr(torch._C, "__file__"),
                              "loss": loss, "gradients": numbers(gradients), "model": numbers(expected[0]),
                              "optimizer_state": numbers(expected[1]["state"]),
                              "fallback_count": self.jt.core.backend_fallback_count() - self.fallbacks if self.jt else 0}
                    Path(os.environ["CHECKPOINT_OPTIONS_REPORT"]).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
