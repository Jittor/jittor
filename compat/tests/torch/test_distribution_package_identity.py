"""Torch publication must not replace a native implementation module."""

import importlib
import pickle

import numpy as np

import jittor.distributions as distributions
import jittor.distributions.divergence as divergence
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def test_native_divergence_owner_survives_torch_publication():
    assert divergence is importlib.import_module("jittor.distributions.divergence")
    assert distributions.divergence is divergence
    # ``kl`` is a torch spelling: it is published on the torch namespace, not
    # grafted onto the native package (which owns it as ``divergence``). The
    # published one delegates here rather than reimplementing.
    import torch
    assert not hasattr(distributions, "kl")
    assert torch.distributions.kl.kl_divergence.__wrapped__ is divergence.kl_divergence \
        if hasattr(torch.distributions.kl.kl_divergence, "__wrapped__") else True
    assert pickle.loads(pickle.dumps(divergence.kl_divergence)) is divergence.kl_divergence
    assert pickle.loads(b"cjittor.distributions\nkl_divergence\n.") is divergence.kl_divergence


def test_constraint_class_attributes_survive_frontend_module_publication():
    """A distribution subclass must work without a downstream import hook."""
    import torch

    native = distributions.constraints
    before = vars(native).copy()
    constraints = importlib.import_module("torch.distributions.constraints")
    assert constraints is torch.distributions.constraints
    assert constraints is not native
    for name in ("Constraint", "_Real", "real", "positive", "interval",
                 "dependent_property", "greater_than", "less_than"):
        assert getattr(constraints, name) is getattr(native, name)

    class Bounded(torch.distributions.Distribution):
        arg_constraints = {"value": constraints.real}

        @constraints.dependent_property
        def support(self):
            return constraints.interval(-1., 1.)

    # The native base currently restricts subclass constructor signatures; the
    # property needs no constructor to prove publication/decorator ownership.
    support = Bounded.support.__get__(object.__new__(Bounded), Bounded)
    assert support.lower_bound == -1. and support.upper_bound == 1.
    assert vars(native) == before


class TestPublishedConstraintNumerics(cu.JittorTestCase):
    def test_bounds_checks_compute_on_the_selected_device(self, device):
        import jittor as jt
        import torch
        from jittor._runtime import forbid_backend_fallbacks

        constraints = torch.distributions.constraints
        with forbid_backend_fallbacks():
            value = torch.tensor([-2., -1., 0., 1., 2., float("nan")], device=device)
            checked = (
                constraints.interval(-1., 1.).check(value),
                constraints.greater_than(0.).check(value),
                constraints.less_than(0.).check(value),
            )
            expected = (
                [False, True, True, True, False, False],
                [False, False, False, True, True, False],
                [True, True, False, False, False, False],
            )
            for result, reference in zip(checked, expected):
                result.sync()
                self.assertEqual(result.location(), "cpu" if device == "cpu" else "device")
                np.testing.assert_array_equal(result.cpu().numpy(), reference)
            jt.sync_all(True)


instantiate_device_type_tests(TestPublishedConstraintNumerics, globals())
