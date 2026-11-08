"""Singleton broadcasts have real zero-stride storage in Torch forward/backward."""
import torch

def test_singleton_expand_is_zero_stride_and_writes_through():
    base = torch.tensor([1.0], dtype=torch.float32)
    expanded = base.expand(2)
    assert expanded.stride() == (0,)
    assert expanded.data_ptr() == base.data_ptr()
    expanded[1].copy_(torch.tensor(3.0))
    assert base.tolist() == [3.0]
    assert expanded.tolist() == [3.0, 3.0]

def test_sum_vjp_keeps_zero_stride_in_custom_backward():
    captured = {}
    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.expand(2)
        @staticmethod
        def backward(ctx, incoming):
            captured["incoming"] = incoming
            return incoming.sum().reshape(1)
    value = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
    output = Probe.apply(value)
    gradient, = torch.autograd.grad(output.sum(), value)
    incoming = captured["incoming"]
    assert output.stride() == incoming.stride() == (0,)
    assert incoming.tolist() == [1.0, 1.0]
    assert gradient.tolist() == [2.0]
    incoming[1].copy_(torch.tensor(3.0))
    assert incoming.tolist() == [3.0, 3.0]
    assert gradient.tolist() == [2.0]

def test_same_width_dtype_view_preserves_unit_axis_zero_stride():
    base = torch.tensor([1.0], dtype=torch.float32)
    singleton = base.expand(2)[:1]
    assert singleton.stride() == (0,)
    assert singleton.is_contiguous()
    words = singleton.view(torch.uint32)
    assert words.stride() == (0,)
    assert words.data_ptr() == base.data_ptr()
    words[0].copy_(torch.tensor(0x40000000, dtype=torch.uint32))
    assert base.tolist() == [2.0]
    assert singleton.tolist() == [2.0]
