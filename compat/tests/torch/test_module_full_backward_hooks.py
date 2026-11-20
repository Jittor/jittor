import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class _Layer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, device=device))

    def forward(self, x, tag, *, offset):
        y = x * self.weight + offset
        return y, tag, y * 5


class _Pair(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, a, b, offset):
        first = self.layer(a, "first", offset=offset)
        second = self.layer(b, "second", offset=offset)
        assert first[1] == "first" and second[1] == "second"
        return first[0] * 2 + second[2] * 3


def _values(tensors):
    return tuple(tensor.item() if tensor is not None else None for tensor in tensors)


class TestModuleFullBackwardHooks:
    def test_shared_child_call_edges_and_order(self, device):
        layer = _Layer(device)
        pair = _Pair(layer)
        trace = []

        def observe(module, inputs, outputs):
            assert module is layer
            trace.append(("observe", _values(inputs), _values(outputs)))

        def replace(module, inputs, outputs):
            trace.append(("replace", _values(inputs), _values(outputs)))
            return inputs[0] * 3, None

        observe_handle = layer.register_full_backward_hook(observe)
        replace_handle = layer.register_full_backward_hook(replace, prepend=True)
        a = torch.tensor(7., device=device, requires_grad=True)
        b = torch.tensor(11., device=device, requires_grad=True)
        offset = torch.tensor(0., device=device, requires_grad=True)
        with torch.no_grad():
            assert not pair(a, b, offset).requires_grad
        loss = pair(a, b, offset)
        loss.backward(retain_graph=True)
        loss.backward()
        assert sorted(trace) == sorted([
            ("replace", (4., None), (2., None, None)),
            ("observe", (12., None), (2., None, None)),
            ("replace", (30., None), (None, None, 3.)),
            ("observe", (90., None), (None, None, 3.)),
        ] * 2)
        assert _values((a.grad, b.grad, offset.grad, layer.weight.grad)) == (24., 180., 34., 358.)
        observe_handle.remove()
        replace_handle.remove()
        trace.clear()
        pair(a, b, offset).backward()
        assert not trace

    def test_frozen_input_fires_from_outputs(self, device):
        layer = _Layer(device)
        trace = []
        layer.register_full_backward_hook(
            lambda module, inputs, outputs: trace.append((_values(inputs), _values(outputs))))
        x = torch.tensor(5., device=device)
        offset = torch.tensor(0., device=device, requires_grad=True)
        with pytest.warns(UserWarning, match="Full backward hook"):
            layer(x, None, offset=offset)[0].backward()
        assert trace == [((None, None), (1., None, None))]
        assert _values((x.grad, offset.grad, layer.weight.grad)) == (None, 1., 5.)

    def test_wrong_number_of_replacements(self, device):
        layer = torch.nn.Identity()
        layer.register_full_backward_hook(lambda module, inputs, outputs: ())
        x = torch.tensor(5., device=device, requires_grad=True)
        with pytest.raises(RuntimeError, match="grad_input"):
            layer(x).backward()

    def test_inplace_forward_and_output_views_preserve_custom_backward(self, device):
        class Inplace(torch.nn.Module):
            def forward(self, value):
                return value.add_(1)

        layer = Inplace()
        layer.register_full_backward_hook(lambda *args: None)
        x = torch.tensor([-1., 2.], device=device, requires_grad=True)
        with pytest.raises(RuntimeError, match="view.*modified inplace"):
            layer(x * 2)

        layer = torch.nn.Identity()
        layer.register_full_backward_hook(lambda *args: None)
        y = layer(x * 2)
        for view in (y, y.view(2)):
            with pytest.raises(RuntimeError, match="view.*modified inplace"):
                view.add_(1)
        z = y.clone()
        z.add_(1)
        z.sum().backward()
        assert _values(x.grad) == (2., 2.)

    def test_mutation_through_alias_invalidates_saved_hook_view(self, device):
        layer = torch.nn.Identity()
        layer.register_full_backward_hook(lambda *args: None)
        for alias in ("output", "base", "detached"):
            x = torch.tensor([1., 2.], device=device, requires_grad=True)
            base = x * 2
            y = layer(base)
            if alias == "base":
                base.add_(1)
            elif alias == "detached":
                y.detach().add_(1)
            else:
                with torch.no_grad():
                    y.add_(1)
            with pytest.raises(RuntimeError, match="view.*modified inplace"):
                y.sum().backward()


instantiate_device_type_tests(TestModuleFullBackwardHooks, globals(), only_for=("cpu", "cuda"))
