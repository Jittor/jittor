"""Torch publication must not replace a native implementation module."""

import importlib
import pickle

import jittor.distributions as distributions
import jittor.distributions.divergence as divergence


def test_native_divergence_owner_survives_torch_publication():
    assert divergence is importlib.import_module("jittor.distributions.divergence")
    assert distributions.divergence is divergence
    assert distributions.kl.kl_divergence is divergence.kl_divergence
    assert pickle.loads(pickle.dumps(divergence.kl_divergence)) is divergence.kl_divergence
    assert pickle.loads(b"cjittor.distributions\nkl_divergence\n.") is divergence.kl_divergence
