"""Opt-in execution pipelining for ``import jittor as torch``.

A lazy graph reaches the device only at the next sync, so the GPU is idle for the
whole of the Python-side construction. Pipelining launches the graph built so far
at a module boundary instead. It must stay off unless asked for, and must not
change what a model computes beyond floating-point regrouping.

Run:  python -m pytest tests/compat/torch/test_torch_execution_pipelining.py
"""
import unittest

import numpy as np
import jittor as torch
import jittor as jt


_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _model():
    jt.set_global_seed(4)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
    )


class TestExecutionPipelining(unittest.TestCase):
    def tearDown(self):
        torch.nn.Module.set_execution_pipelining(0)

    def test_off_by_default_and_setter_reports_previous(self):
        self.assertEqual(torch.nn.Module.get_execution_pipelining(), 0)
        self.assertEqual(torch.nn.Module.set_execution_pipelining(128), 0)
        self.assertEqual(torch.nn.Module.get_execution_pipelining(), 128)
        self.assertEqual(torch.nn.Module.set_execution_pipelining(0), 128)
        self.assertEqual(torch.nn.Module.get_execution_pipelining(), 0)

    def test_negative_and_float_thresholds_are_normalised(self):
        torch.nn.Module.set_execution_pipelining(-5)
        self.assertEqual(torch.nn.Module.get_execution_pipelining(), 0)
        torch.nn.Module.set_execution_pipelining(64.9)
        self.assertEqual(torch.nn.Module.get_execution_pipelining(), 64)

    def test_forward_matches_the_unpipelined_result(self):
        def body(dev):
            model = _model()
            x = jt.array(np.random.RandomState(5).randn(8, 16).astype("float32"))
            expected = model(x).numpy()
            torch.nn.Module.set_execution_pipelining(8)
            try:
                got = model(x).numpy()
            finally:
                torch.nn.Module.set_execution_pipelining(0)
            # Flushing moves fusion boundaries, which regroups float32
            # accumulation; the result may differ by a rounding step, no more.
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5,
                                       err_msg="pipelined forward %s" % dev)
        both_devices(body)

    def test_training_trajectory_matches(self):
        def body(dev):
            def run(threshold):
                model = _model()
                x = jt.array(np.random.RandomState(6).randn(8, 16).astype("float32"))
                y = jt.array(np.random.RandomState(7).randn(8, 8).astype("float32"))
                optimizer = torch.nn.SGD(model.parameters(), lr=1e-2)
                torch.nn.Module.set_execution_pipelining(threshold)
                try:
                    losses = []
                    for _ in range(3):
                        loss = ((model(x) - y) ** 2).mean()
                        losses.append(float(loss.numpy().reshape(-1)[0]))
                        optimizer.step(loss)
                    return losses
                finally:
                    torch.nn.Module.set_execution_pipelining(0)

            plain = run(0)
            pipelined = run(8)
            assert np.isfinite(pipelined).all(), pipelined
            np.testing.assert_allclose(pipelined, plain, atol=1e-5, rtol=1e-5,
                                       err_msg="pipelined trajectory %s" % dev)
        both_devices(body)

    def test_a_module_returning_a_tuple_is_left_alone(self):
        # The hook only ever syncs a Var. A module handing back a tuple whose
        # first element is not one must pass through untouched.
        class Pair(torch.nn.Module):
            def execute(self, value):
                return ("tag", value * 2)

        module = Pair()
        torch.nn.Module.set_execution_pipelining(1)
        try:
            tag, doubled = module(jt.array(np.array([1.0, 2.0], "float32")))
        finally:
            torch.nn.Module.set_execution_pipelining(0)
        self.assertEqual(tag, "tag")
        np.testing.assert_allclose(doubled.numpy(), [2.0, 4.0])


class TestEnvironmentOptIn(unittest.TestCase):
    """The env var is read once when the installer runs, so this checks the
    parser rather than re-importing jittor."""

    def _parse(self, raw):
        import os
        from jittor.compat.torch.installers import nn as installer
        previous = os.environ.get("JITTOR_EXECUTION_PIPELINING")
        if raw is None:
            os.environ.pop("JITTOR_EXECUTION_PIPELINING", None)
        else:
            os.environ["JITTOR_EXECUTION_PIPELINING"] = raw
        try:
            return installer._pipelining_from_environment()
        finally:
            if previous is None:
                os.environ.pop("JITTOR_EXECUTION_PIPELINING", None)
            else:
                os.environ["JITTOR_EXECUTION_PIPELINING"] = previous

    def test_unset_is_off(self):
        self.assertEqual(self._parse(None), 0)

    def test_value_is_read(self):
        self.assertEqual(self._parse("200"), 200)

    def test_negative_is_off(self):
        self.assertEqual(self._parse("-3"), 0)

    def test_garbage_is_off_rather_than_fatal(self):
        # A tuning knob must never break an import.
        self.assertEqual(self._parse("fast"), 0)
        self.assertEqual(self._parse(""), 0)


if __name__ == "__main__":
    unittest.main()
