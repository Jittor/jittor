"""Canonical module dispatch argument contract and portable CPU behavior."""

import importlib

import jittor as jt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def cpu_device():
    with jt.runtime.scope(use_cuda=0, use_acl=0):
        yield


def test_module_passes_canonical_functional_arguments_to_selected_kernel(monkeypatch):
    owner = importlib.import_module("jittor.nn.modules.convolution_transpose")
    layer = jt.nn.ConvTranspose2d(4, 6, (2, 3), stride=(2, 3), padding=(0, 1),
                                  output_padding=(1, 2), groups=2, dilation=(1, 2))
    x = jt.ones((1, 4, 3, 4))
    result = jt.array([19.])
    selected, executed = [], []

    def execute(*args):
        executed.append(args)
        return result

    def select(operation, *args):
        selected.append((operation, args))
        return execute

    monkeypatch.setattr(owner, "select_kernel", select)
    assert layer(x) is result
    expected = (x, layer.weight, layer.bias, (2, 3), (0, 1), (1, 2), (1, 2), 2)
    assert selected == [("conv_transpose2d", expected)]
    assert executed == [expected]


@pytest.mark.parametrize("groups", [1, 2])
def test_declined_module_and_function_keep_cpu_composite_numerics(monkeypatch, groups):
    owner = importlib.import_module("jittor.nn.modules.convolution_transpose")
    declined = []

    def decline(operation, *args):
        declined.append(operation)
        return None

    monkeypatch.setattr(owner, "select_kernel", decline)
    source = np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3) / 32
    weight = np.full((4, 4 // groups, 3, 3), .125, dtype=np.float32)
    bias = np.arange(4, dtype=np.float32) / 16
    expected = np.zeros((1, 4, 4, 6), dtype=np.float32)
    channels = 4 // groups
    for ic, row, col, kh, kw in np.ndindex(4, 2, 3, 3, 3):
        oy, ox = row * 2 - 1 + kh, col * 2 - 1 + kw
        if 0 <= oy < 4 and 0 <= ox < 6:
            group = ic // channels
            expected[:, group * channels:(group + 1) * channels, oy, ox] += source[:, ic, row, col, None] * .125
    expected += bias[None, :, None, None]
    layer = jt.nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1, groups=groups)
    layer.weight, layer.bias = jt.array(weight), jt.array(bias)
    x = jt.array(source)
    with jt.no_grad():
        module = layer(x)
        functional = jt.nn.conv_transpose2d(x, layer.weight, layer.bias, stride=2,
                                            padding=1, output_padding=1, groups=groups)
        for actual in (module, functional):
            actual.sync(True)
            assert actual.location() == "cpu"
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert declined == ["conv_transpose2d"]
