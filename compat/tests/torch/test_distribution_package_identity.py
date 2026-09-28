"""Torch publication must not replace a native implementation module."""

import importlib
import pickle

import jittor.distributions as distributions
import jittor.distributions.divergence as divergence


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
