"""CPU RNG checkpoints and the strictly representable CUDA checkpoint subset."""

import os
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch


def values(tensor):
    return tensor.detach().clone().cpu().numpy().copy()


def assert_random_device(tensor, device):
    if str(tensor.device).split(":", 1)[0] != device:
        raise AssertionError("random draw executed on the wrong device")
    if device == "cuda" and tensor.device.index != torch.cuda.current_device():
        raise AssertionError("random draw executed on the wrong CUDA device index")
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        backend, index = jt.core.dispatch_context([tensor])
        if backend != device or (device == "cuda" and index != torch.cuda.current_device()):
            raise AssertionError("random draw has the wrong native backend/device placement")


def draws(device, sizes=(7, 9, 8, 5)):
    result = []
    for index, size in enumerate(sizes):
        factory = torch.rand if index % 2 == 0 else torch.randn
        dtype = torch.float32 if index < 2 else torch.float64
        tensor = factory(size, dtype=dtype, device=device)
        assert_random_device(tensor, device)
        result.append(values(tensor))
    return result


class RNGStateContract:
    target_device = "cpu"

    def checkpoint_draws(self, sizes=(7, 9, 8, 5)):
        if self.target_device == "cuda":
            result = []
            for size in sizes:
                tensor = torch.rand(size, device="cuda", dtype=torch.float32)
                assert_random_device(tensor, "cuda")
                result.append(values(tensor))
            return result
        return draws("cpu", sizes)

    def capture(self):
        return torch.get_rng_state() if self.target_device == "cpu" else torch.cuda.get_rng_state()

    def restore(self, state):
        if self.target_device == "cpu":
            torch.set_rng_state(state)
        else:
            torch.cuda.set_rng_state(state)

    def seed(self, seed):
        if self.target_device == "cpu":
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)

    def assert_draws_equal(self, expected, actual):
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_mixed_distributions_precisions_and_repeated_restore(self):
        for seed in (42, 1729):
            self.seed(seed)
            self.checkpoint_draws()
            for _ in range(3):
                state = self.capture()
                self.assertEqual(state.dtype, torch.uint8)
                self.assertEqual(str(state.device), "cpu")
                self.assertEqual(state.ndim, 1)
                self.assertGreater(state.numel(), 1)
                expected = draws(self.target_device)
                self.restore(state)
                self.assert_draws_equal(expected, draws(self.target_device))
                self.restore(state)
                self.checkpoint_draws()

    def test_large_batch_boundaries(self):
        self.seed(1729)
        for size in (4095, 4096, 4097, 65537):
            self.checkpoint_draws((size, size, size, size))
            state = self.capture()
            expected = draws(self.target_device)
            self.restore(state)
            self.assert_draws_equal(expected, draws(self.target_device))
            self.restore(state)

    def test_full_seed_range_and_illegal_seed_are_atomic(self):
        for seed in (-1, -(1 << 63), (1 << 63) + 17, (1 << 64) - 1):
            self.seed(seed)
            actual_seed = torch.initial_seed() if self.target_device == "cpu" else torch.cuda.initial_seed()
            self.assertEqual(actual_seed, seed % (1 << 64))
            self.checkpoint_draws()
            state = self.capture()
            expected = draws(self.target_device)
            self.seed(42)
            self.restore(state)
            self.assert_draws_equal(expected, draws(self.target_device))
            self.restore(state)
        current = self.capture()
        for invalid in (-(1 << 63) - 1, 1 << 64):
            with self.assertRaises(RuntimeError):
                self.seed(invalid)
            np.testing.assert_array_equal(values(self.capture()), values(current))

    def test_pending_random_graph_is_resolved_before_capture_and_restore(self):
        self.seed(41)
        factory = torch.rand if self.target_device == "cuda" else torch.randn
        old = factory(9, device=self.target_device)
        after_old = self.capture()
        expected = values(torch.rand(11, device=self.target_device))
        self.restore(after_old)
        pending = torch.rand(11, device=self.target_device)
        self.restore(after_old)
        np.testing.assert_array_equal(values(pending), expected)
        np.testing.assert_array_equal(values(torch.rand(11, device=self.target_device)), expected)
        self.assertEqual(tuple(old.shape), (9,))

    def test_invalid_state_does_not_rewind_or_change_seed(self):
        self.seed(1729)
        self.checkpoint_draws()
        state = self.capture()
        initial_seed = torch.initial_seed() if self.target_device == "cpu" else torch.cuda.initial_seed()
        for invalid in ([1], torch.tensor([1], dtype=torch.int64),
                        torch.tensor([255, 254], dtype=torch.uint8),
                        torch.zeros(1, dtype=torch.uint8)):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises((TypeError, RuntimeError, ValueError, UnicodeDecodeError,
                                        AttributeError)):
                    self.restore(invalid)
                np.testing.assert_array_equal(values(self.capture()), values(state))
                current_seed = torch.initial_seed() if self.target_device == "cpu" else torch.cuda.initial_seed()
                self.assertEqual(initial_seed, current_seed)

    def test_accelerate_fresh_process_training_resume(self):
        from _helpers.child_process import run_python_child
        try:
            import accelerate
        except ModuleNotFoundError as exc:
            if exc.name != "accelerate" or os.environ.get("JITTOR_REQUIRE_ACCELERATE") == "1":
                raise
            self.skipTest("Accelerate is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            for mode in ("reference", "resume"):
                command = [str(Path(__file__).resolve()), "--resume-worker",
                           self.target_device, mode, str(checkpoint), str(root / (mode + ".json"))]
                completed = run_python_child(
                    command, timeout=180, text=True,
                    repo_paths=hasattr(torch, "_torch_compat_install_context"))
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            expected = json.loads((root / "reference.json").read_text())
            actual = json.loads((root / "resume.json").read_text())
            self.assertEqual(actual, expected)


class TestCPURNGState(RNGStateContract, unittest.TestCase):
    def test_initial_seed_and_external_random_streams_are_preserved(self):
        torch.manual_seed(42)
        state42 = torch.get_rng_state()
        torch.manual_seed(1729)
        random.seed(71)
        np.random.seed(81)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch.set_rng_state(state42)
        self.assertEqual(torch.initial_seed(), 42)
        self.assertEqual(random.getstate(), python_state)
        for left, right in zip(np.random.get_state(), numpy_state):
            np.testing.assert_array_equal(left, right)

    def test_manual_seed_does_not_reseed_python_or_numpy(self):
        random.seed(71)
        np.random.seed(81)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch.manual_seed(-1)
        self.assertEqual(random.getstate(), python_state)
        for left, right in zip(np.random.get_state(), numpy_state):
            np.testing.assert_array_equal(left, right)


@unittest.skipUnless("cuda" in os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")
                     and torch.cuda.is_available(), "requires explicitly requested real CUDA")
class TestCUDARNGState(RNGStateContract, unittest.TestCase):
    target_device = "cuda"

    def test_cuda_seed_and_restore_do_not_touch_cpu_or_external_streams(self):
        torch.manual_seed(42)
        cpu_state = torch.get_rng_state()
        random.seed(71)
        np.random.seed(81)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch.cuda.manual_seed(1729)
        self.checkpoint_draws()
        state = torch.cuda.get_rng_state()
        self.checkpoint_draws()
        torch.cuda.set_rng_state(state)
        self.assertEqual(torch.cuda.initial_seed(), 1729)
        np.testing.assert_array_equal(values(torch.get_rng_state()), values(cpu_state))
        self.assertEqual(random.getstate(), python_state)
        for left, right in zip(np.random.get_state(), numpy_state):
            np.testing.assert_array_equal(left, right)

    def test_all_states_and_invalid_list_are_atomic(self):
        torch.cuda.manual_seed_all(1729)
        self.checkpoint_draws()
        states = torch.cuda.get_rng_state_all()
        self.assertEqual(len(states), torch.cuda.device_count())
        expected = draws("cuda")
        torch.cuda.set_rng_state_all(states)
        self.assert_draws_equal(expected, draws("cuda"))
        torch.cuda.set_rng_state_all(states)
        if not hasattr(torch, "_torch_compat_install_context"):
            return
        # Atomic rejection is Jittor's explicit policy. Torch applies states
        # sequentially and can modify earlier devices before a later error.
        torch.cuda.manual_seed_all(42)
        current = torch.cuda.get_rng_state_all()
        with self.assertRaises((ValueError, RuntimeError, IndexError)):
            torch.cuda.set_rng_state_all(states + [states[0]])
        bad = list(states)
        bad[-1] = torch.tensor([0], dtype=torch.uint8)
        with self.assertRaises((ValueError, RuntimeError)):
            torch.cuda.set_rng_state_all(bad)
        for left, right in zip(current, torch.cuda.get_rng_state_all()):
            np.testing.assert_array_equal(values(left), values(right))

    def test_dropout_continuation_and_gradients(self):
        torch.cuda.manual_seed(1729)
        layer = torch.nn.Dropout(0.25).to("cuda")
        x = torch.arange(1, 25, device="cuda", dtype=torch.float32).reshape(4, 6)
        layer(x).sum().item()
        state = torch.cuda.get_rng_state()
        first_input = x.clone().requires_grad_(True)
        first = layer(first_input)
        first.sum().backward()
        expected, expected_grad = values(first), values(first_input.grad)
        torch.cuda.set_rng_state(state)
        second_input = x.clone().requires_grad_(True)
        second = layer(second_input)
        second.sum().backward()
        np.testing.assert_array_equal(values(second), expected)
        np.testing.assert_array_equal(values(second_input.grad), expected_grad)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two visible CUDA devices")
    def test_device_generators_are_isolated_and_restore_preserves_current_device(self):
        previous = torch.cuda.current_device()
        try:
            for index, seed in ((0, 42), (1, 1729)):
                torch.cuda.set_device(index)
                torch.cuda.manual_seed(seed)
                self.checkpoint_draws()
            torch.cuda.set_device(1)
            states = torch.cuda.get_rng_state_all()
            self.assertEqual(torch.cuda.current_device(), 1)
            expected = {}
            for index in (0, 1):
                torch.cuda.set_device(index)
                expected[index] = draws("cuda")
            torch.cuda.set_device(1)
            torch.cuda.set_rng_state_all(states)
            self.assertEqual(torch.cuda.current_device(), 1)
            torch.cuda.set_rng_state(states[0], device=0)
            self.assertEqual(torch.cuda.current_device(), 1)
            np.testing.assert_array_equal(values(torch.cuda.get_rng_state(1)), values(states[1]))
            self.assertEqual(torch.cuda.current_device(), 1)
            for index in (0, 1):
                torch.cuda.set_device(index)
                self.assert_draws_equal(expected[index], draws("cuda"))
                self.assertEqual(torch.cuda.initial_seed(), (42, 1729)[index])
        finally:
            torch.cuda.set_device(previous)

    @unittest.skipUnless(hasattr(torch, "_torch_compat_install_context"),
                         "Jittor's opaque Host cuRAND checkpoint limitation")
    def test_unrepresentable_history_fails_and_prior_safe_state_restores(self):
        for factory, dtype in ((torch.randn, torch.float32), (torch.rand, torch.float64),
                               (torch.randn, torch.float64)):
            torch.cuda.manual_seed(1729)
            self.checkpoint_draws((4097, 65537))
            state = torch.cuda.get_rng_state()
            expected = draws("cuda", (4097, 65537, 4097, 65537))
            torch.cuda.set_rng_state(state)
            generated = factory(4097, dtype=dtype, device="cuda")
            assert_random_device(generated, "cuda")
            with self.assertRaisesRegex(RuntimeError, "complete CUDA RNG state is unsupported"):
                torch.cuda.get_rng_state()
            self.assertEqual(values(generated).shape, (4097,))
            torch.cuda.set_rng_state(state)
            self.assert_draws_equal(expected, draws("cuda", (4097, 65537, 4097, 65537)))
            torch.cuda.set_rng_state(state)
            legacy = values(state).tobytes().replace(b"XORWOW_U32_V1", b"XORWOW_V1")
            invalid = torch.tensor(np.frombuffer(legacy, dtype=np.uint8).copy(), dtype=torch.uint8)
            with self.assertRaises(RuntimeError):
                torch.cuda.set_rng_state(invalid)
            np.testing.assert_array_equal(values(torch.cuda.get_rng_state()), values(state))


def resume_worker(target_device, mode, checkpoint, output_file):
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=target_device == "cpu", mixed_precision="no")
    if accelerator.device.type != target_device:
        raise AssertionError("Accelerate selected the wrong device")
    torch.manual_seed(1729 if mode == "reference" else 991)
    random.seed(71)
    np.random.seed(81)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    model, optimizer = accelerator.prepare(model, optimizer)
    # Initializers may use unsupported cuRAND distributions. This fixture tests
    # continuation of uniform FP32 noise/dropout after model preparation.
    torch.manual_seed(1729 if mode == "reference" else 991)
    base = torch.arange(1, 9, dtype=torch.float32, device=target_device).reshape(2, 4) / 8
    records = []

    def step(index):
        augmentation = random.random() + float(np.random.random())
        noise = torch.rand(2, 4, device=target_device) * 0.1
        optimizer.zero_grad()
        prediction = model(base + noise + augmentation)
        if str(prediction.device).split(":", 1)[0] != target_device:
            raise AssertionError("forward executed on the wrong device")
        loss = prediction.square().mean()
        accelerator.backward(loss)
        optimizer.step()
        records.append({"step": index, "augmentation": augmentation,
                        "output": values(prediction).tolist(),
                        "parameters": [values(p).tolist() for p in model.parameters()],
                        "loss": float(loss.item())})

    if mode == "reference":
        step(1)
        accelerator.save_state(checkpoint, safe_serialization=False)
    else:
        accelerator.load_state(checkpoint)
    step(2)
    step(3)
    result = {"records": [record for record in records if record["step"] > 1],
              "cpu_state": values(torch.get_rng_state()).tolist(),
              "cuda_states": ([values(state).tolist() for state in torch.cuda.get_rng_state_all()]
                              if target_device == "cuda" else [])}
    Path(output_file).write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resume-worker":
        resume_worker(*sys.argv[2:])
    else:
        unittest.main()
