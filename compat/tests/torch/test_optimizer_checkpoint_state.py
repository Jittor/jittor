"""Optimizer FQN checkpoints and real Accelerate numerical continuation.

The same file runs against an independent binary PyTorch (direct Python) and
against the shim. CHECKPOINT_DEVICE chooses a genuine CPU/CUDA process; a JSON
trajectory can be emitted via CHECKPOINT_REPORT for external oracle comparison.
"""

import copy
import importlib.util
import io
import pickle
import random
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from _accelerate_protocol import Capture, hyperparameters
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    get_state_dict, set_model_state_dict, set_optimizer_state_dict, set_state_dict,
)


DEVICE = os.environ.get("CHECKPOINT_DEVICE", "cpu")


def is_tensor(value):
    native = sys.modules.get("jittor")
    return torch.is_tensor(value) or (native is not None and isinstance(value, native.Var))


def snapshot(value):
    if is_tensor(value):
        # Native Var.clone is an alias; the public Torch clone owns a copy.
        return torch.clone(value.detach())
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return copy.deepcopy(value)


def numbers(value):
    if is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {key: numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [numbers(item) for item in value]
    return value


def model_optimizer():
    model = torch.nn.Linear(3, 2).to(DEVICE)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3) / 20, device=DEVICE))
        model.bias.copy_(torch.tensor([0.03, -0.04], device=DEVICE))
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr": .01},
        {"params": [model.bias], "lr": .015},
    ], betas=(.7, .93), eps=1e-4, weight_decay=.02)
    return model, optimizer


def update(model, optimizer, backward=None, capture=None):
    optimizer.zero_grad()
    input_values = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    input_seed = int(os.environ.get("JITTOR_ACCELERATE_INPUT_SEED", "0"))
    if input_seed:
        input_values += np.random.RandomState(input_seed).normal(0, .01, input_values.shape).astype(np.float32)
    x = torch.tensor(input_values, device=DEVICE, requires_grad=True)
    y = torch.tensor(np.arange(8, dtype=np.float32).reshape(4, 2) / 20, device=DEVICE)
    output = model(x)
    loss = ((output - y) ** 2).mean()
    (backward or (lambda value: value.backward()))(loss)
    if capture is not None:
        capture["input"] = snapshot(x)
        capture["target"] = snapshot(y)
        capture["output"] = snapshot(output)
        capture["loss"] = snapshot(loss)
        capture["input_gradient"] = snapshot(x.grad)
        capture["parameter_gradients"] = snapshot(dict((name, parameter.grad) for name, parameter in model.named_parameters()))
    optimizer.step()
    return float(loss.detach().cpu().item())


class CheckpointTag:
    def __init__(self, value):
        self.value = value


class TestOptimizerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.jt = sys.modules.get("jittor")
        self.fallbacks = None
        if self.jt is not None:
            self.scope = self.jt.runtime.scope(use_cuda=DEVICE != "cpu", backend_fallback="error")
            self.scope.__enter__()
            self.fallbacks = self.jt.core.backend_fallback_count()

    def tearDown(self):
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), self.fallbacks)
            self.scope.__exit__(None, None, None)

    def assert_tree(self, actual, expected, path="root"):
        if is_tensor(expected):
            np.testing.assert_allclose(actual.detach().cpu().numpy(), expected.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree(actual[key], expected[key], path + "." + str(key))
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree(left, right, path + "[]")
        else:
            self.assertEqual(type(actual), type(expected), path)
            self.assertEqual(actual, expected, path)

    def test_safe_global_registration_controls_checkpoint_loading(self):
        previous = torch.serialization.get_safe_globals()
        stream = io.BytesIO()
        torch.save(CheckpointTag(17), stream)
        try:
            torch.serialization.clear_safe_globals()
            stream.seek(0)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(stream, weights_only=True)
            with torch.serialization.safe_globals([CheckpointTag]):
                stream.seek(0)
                self.assertEqual(torch.load(stream, weights_only=True).value, 17)
                self.assertIn(CheckpointTag, torch.serialization.get_safe_globals())
            self.assertNotIn(CheckpointTag, torch.serialization.get_safe_globals())
            stream.seek(0)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(stream, weights_only=True)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)

    def test_optimizer_fqns_and_combined_resume(self):
        model, optimizer = model_optimizer()
        update(model, optimizer)
        saved_model, saved_optimizer = snapshot(get_state_dict(model, optimizer))
        self.assertEqual(set(saved_optimizer["state"]), {"weight", "bias"})
        self.assertEqual([group["params"] for group in saved_optimizer["param_groups"]], [["weight"], ["bias"]])
        expected_loss = update(model, optimizer)
        expected_model, expected_optimizer = snapshot(get_state_dict(model, optimizer))
        set_state_dict(model, optimizer, model_state_dict=saved_model, optim_state_dict=saved_optimizer)
        self.assert_tree(get_state_dict(model, optimizer), (saved_model, saved_optimizer))
        self.assertAlmostEqual(update(model, optimizer), expected_loss, places=6)
        self.assert_tree(get_state_dict(model, optimizer), (expected_model, expected_optimizer))
        self.assertTrue(all(parameter.device.type == DEVICE for parameter in model.parameters()))

    def test_load_matches_fqns_after_optimizer_reordering(self):
        model, optimizer = model_optimizer()
        update(model, optimizer)
        saved = snapshot(get_optimizer_state_dict(model, optimizer))
        reordered = torch.optim.AdamW([
            {"params": [model.bias], "lr": 1.0},
            {"params": [model.weight], "lr": 2.0},
        ])
        set_optimizer_state_dict(model, reordered, saved)
        restored = get_optimizer_state_dict(model, reordered)
        self.assert_tree(restored["state"], saved["state"])
        self.assertEqual([g["lr"] for g in reordered.param_groups], [.015, .01])

    def test_multiple_optimizers_and_flatten_cpu_offload(self):
        model, optimizer = model_optimizer()
        update(model, optimizer)
        opts = StateDictOptions(cpu_offload=True, flatten_optimizer_state_dict=True)
        saved = snapshot(get_optimizer_state_dict(model, (optimizer,), options=opts))
        self.assertIn("state.weight.exp_avg", saved)
        self.assertIn("param_groups.bias.lr", saved)
        self.assertEqual(saved["state.weight.exp_avg"].device.type, "cpu")
        update(model, optimizer)
        set_optimizer_state_dict(model, (optimizer,), saved, options=opts)
        self.assert_tree(get_optimizer_state_dict(model, optimizer, options=opts), saved)
        optimizer.zero_grad(set_to_none=True)
        first = torch.optim.SGD([model.weight], lr=.2, momentum=.8)
        second = torch.optim.SGD([model.bias], lr=.3, momentum=.9)
        combined = snapshot(get_optimizer_state_dict(model, [first, second]))
        set_optimizer_state_dict(model, [first, second], combined)
        self.assertEqual(len(combined["param_groups"]), 2)
        self.assertEqual(set(combined["state"]), {"weight", "bias"})

    def test_full_cpu_offload_roundtrip_and_model_strict_option(self):
        model, optimizer = model_optimizer()
        update(model, optimizer)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        saved = snapshot(get_state_dict(model, optimizer, options=options))
        self.assertTrue(all(value.device.type == "cpu" for value in saved[0].values()))
        for fields in saved[1]["state"].values():
            self.assertTrue(all(value.device.type == "cpu" for value in fields.values() if is_tensor(value)))
        update(model, optimizer)
        set_state_dict(model, optimizer, model_state_dict=saved[0],
                       optim_state_dict=saved[1], options=options)
        self.assert_tree(get_state_dict(model, optimizer, options=options), saved)
        partial = {"weight": saved[0]["weight"]}
        result = set_model_state_dict(model, partial, options=StateDictOptions(strict=False))
        self.assertEqual(result.missing_keys, ["bias"])
        with self.assertRaises(RuntimeError):
            set_model_state_dict(model, partial, options=StateDictOptions(strict=True))

    def test_options_filter_frozen_model_and_validate_broadcast(self):
        model, optimizer = model_optimizer()
        model.bias.requires_grad_(False)
        state = get_model_state_dict(model, options=StateDictOptions(ignore_frozen_params=True, cpu_offload=True))
        self.assertEqual(set(state), {"weight"})
        with self.assertRaises(ValueError):
            get_optimizer_state_dict(model, optimizer, options=StateDictOptions(broadcast_from_rank0=True))

    def test_accelerate_save_load_continues_model_optimizer_scheduler(self):
        if importlib.util.find_spec("accelerate") is None:
            if os.environ.get("JITTOR_REQUIRE_ACCELERATE") == "1":
                self.fail("required Accelerate dependency is not installed")
            self.skipTest("optional accelerate dependency is not installed")
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState, GradientState
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()
        accelerator = Accelerator(cpu=DEVICE == "cpu")
        try:
            model, optimizer = model_optimizer()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            self.assertEqual(accelerator.device.type, DEVICE)
            capture = Capture("single_checkpoint", torch, DEVICE)
            capture.add("initial/model", dict(accelerator.unwrap_model(model).named_parameters()))
            step_index = [0]

            def measured_update():
                observed = {}
                loss = update(model, optimizer, accelerator.backward, observed)
                scheduler.step()
                state = snapshot(get_state_dict(accelerator.unwrap_model(model), optimizer))
                capture.add("step/%d" % step_index[0], {**observed, "model": state[0],
                                                        "optimizer": state[1]["state"],
                                                        "hyperparameters": hyperparameters(optimizer),
                                                        "learning_rates": scheduler.get_last_lr()})
                step_index[0] += 1
                return {"loss": loss, "model": numbers(state[0]), "optimizer": numbers(state[1]),
                        "gradients": numbers(observed), "learning_rates": scheduler.get_last_lr()}

            def random_sample():
                return {"python": random.random(), "numpy": np.random.random(4).tolist(),
                        "torch_cpu": numbers(torch.rand(4, device="cpu")),
                        "torch_device": numbers(torch.rand(4, device=DEVICE))}

            from _accelerate_fresh_process import fresh_phase, run_fresh_phase
            if fresh_phase():
                def full_state():
                    state = get_state_dict(accelerator.unwrap_model(model), optimizer)
                    return {"model": numbers(state[0]), "optimizer": numbers(state[1]),
                            "scheduler": numbers(scheduler.state_dict())}

                step_index[0] = 5 if fresh_phase() == "consume" else 0
                run_fresh_phase(capture, accelerator, measured_update, full_state, random_sample)
                return

            before_save = [measured_update() for _ in range(3 if capture.enabled else 1)]
            with tempfile.TemporaryDirectory() as path:
                accelerator.save_state(path, safe_serialization=False)

                random_expected = random_sample()
                saved = snapshot(get_state_dict(accelerator.unwrap_model(model), optimizer))
                expected_resume = [measured_update() for _ in range(2 if capture.enabled else 1)]
                expected = snapshot(get_state_dict(accelerator.unwrap_model(model), optimizer))
                expected_lr = scheduler.get_last_lr()
                buffer_types = {
                    (gi, key): [type(buffer) for buffer in group.get(key, [])]
                    for gi, group in enumerate(optimizer.param_groups)
                    for key in ("m", "values", "v", "d", "pre_grad")}
                accelerator.load_state(path)
                resumed_random = random_sample()
                self.assertEqual(resumed_random, random_expected)
                self.assert_tree(get_state_dict(accelerator.unwrap_model(model), optimizer), saved)
                if self.jt is not None:
                    # Loading through the Torch facade must not swap buffer
                    # classes: the next optimizer step would then take a
                    # different operator path and drift in the last bits.
                    for gi, group in enumerate(optimizer.param_groups):
                        for key in ("m", "values", "v", "d", "pre_grad"):
                            for index, buffer in enumerate(group.get(key, [])):
                                if buffer is not None:
                                    self.assertIs(
                                        type(buffer),
                                        buffer_types[(gi, key)][index],
                                        (key, index))
                restored = numbers(get_state_dict(accelerator.unwrap_model(model), optimizer))
                actual_resume = [measured_update() for _ in range(2 if capture.enabled else 1)]
                self.assertEqual(actual_resume, expected_resume)
                self.assert_tree(get_state_dict(accelerator.unwrap_model(model), optimizer), expected)
                self.assertEqual(scheduler.get_last_lr(), expected_lr)
                expected_loss, actual_loss = expected_resume[-1]["loss"], actual_resume[-1]["loss"]
                actual_gradients = actual_resume[-1]["gradients"]
                capture.finish({"before_save": before_save, "expected_resume": expected_resume,
                                "actual_resume": actual_resume, "saved": numbers(saved),
                                "after_restore": restored, "expected_random": random_expected,
                                "resumed_random": resumed_random})
                report = {"device": DEVICE, "torch": torch.__version__,
                          "accelerate": __import__("accelerate").__version__,
                          "oracle": not hasattr(torch, "_torch_compat_install_context"),
                          "expected_loss": expected_loss, "resumed_loss": actual_loss,
                          "model": numbers(expected[0]), "optimizer": numbers(expected[1]),
                          "gradients": numbers(actual_gradients),
                          "torch_file": getattr(torch, "__file__", None),
                          "learning_rates": expected_lr,
                          "fallbacks": (self.jt.core.backend_fallback_count() - self.fallbacks) if self.jt else 0}
                if os.environ.get("CHECKPOINT_REPORT"):
                    Path(os.environ["CHECKPOINT_REPORT"]).write_text(json.dumps(report, indent=2))
        finally:
            accelerator.end_training()
            AcceleratorState._reset_state(reset_partial_state=True)
            GradientState._shared_state.clear()


if __name__ == "__main__":
    unittest.main()
