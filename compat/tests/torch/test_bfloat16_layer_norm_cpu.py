"""CPU non-mixed BF16 LayerNorm statistics and affine gradient boundaries."""
import importlib
from unittest.mock import patch

import numpy as np
import torch
import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
from _helpers.device_types import instantiate_device_type_tests


_X = [[0., 1., 2., 3.], [1.25, 1.5, 1.75, 2.]]
_DY = [[.125, -.25, .5, -1.], [-.375, .75, -1.5, 3.]]
_OUTPUT = [[-.796875, -.197265625, .294921875, 3.1875]] * 2
_DX = [[-.318359375, -.1337890625, 1.2265625, -.76953125],
       [3.828125, 1.609375, -14.6875, 9.25]]


def _snapshot(value):
    value.sync()
    assert value.location() == "cpu"
    return value.detach().float().numpy()


class TestBFloat16LayerNormCpu:
    def test_saved_statistics_and_bfloat16_affine_partials(self, device):
        # Golden observations use independent binary Torch 2.11 CPU, four
        # threads, two rows. Unrounded FP32 statistics or wide affine partials
        # give different input/weight gradients on these exact BF16 inputs.
        with forbid_backend_fallbacks():
            for shape in ((2, 4), (2, 2, 2)):
                x = torch.tensor(_X, dtype=torch.bfloat16).reshape(shape).requires_grad_()
                weight = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.bfloat16).reshape(shape[1:]).requires_grad_()
                bias = torch.tensor([-.125, .25, -.375, .5], dtype=torch.bfloat16).reshape(shape[1:]).requires_grad_()
                dy = torch.tensor(_DY, dtype=torch.bfloat16).reshape(shape)
                output = torch.nn.functional.layer_norm(x, shape[1:], weight, bias)
                dx, dw, db = torch.autograd.grad((output.float() * dy.float()).sum(),
                                                (x, weight, bias), create_graph=True)
                for value in (output, dx, dw, db):
                    assert value.dtype == torch.bfloat16
                np.testing.assert_array_equal(_snapshot(output).reshape(2, 4), _OUTPUT)
                np.testing.assert_array_equal(_snapshot(dx).reshape(2, 4), _DX)
                np.testing.assert_array_equal(_snapshot(dw).reshape(4), [.3359375, -.224609375, -.44921875, 2.6875])
                np.testing.assert_array_equal(_snapshot(db).reshape(4), [-.25, .5, -1., 2.])
                seed = torch.tensor([[1., -2., .5, 3.], [-.5, 1.5, -2., .25]], dtype=torch.bfloat16).reshape(shape)
                second = torch.autograd.grad((dx.float() * seed.float()).sum(), weight)[0]
                assert second.dtype == torch.bfloat16
                # The CPU BF16 second derivative retains a one-ULP rounding
                # difference, within the existing ecosystem gradient budget.
                np.testing.assert_allclose(_snapshot(second).reshape(4),
                                           [.859375, 4.875, 9.125, 5.71875], rtol=.01, atol=0)

    def test_no_affine_and_mixed_parameters_keep_their_domains(self, device):
        owner = importlib.import_module("jittor.nn.functional.normalization")
        with forbid_backend_fallbacks():
            x = torch.tensor(_X, dtype=torch.bfloat16, requires_grad=True)
            dy = torch.tensor(_DY, dtype=torch.bfloat16)
            out = torch.nn.functional.layer_norm(x, (4,))
            dx = torch.autograd.grad((out.float() * dy.float()).sum(), x)[0]
            np.testing.assert_array_equal(_snapshot(out), [[-1.34375, -.447265625, .447265625, 1.34375]] * 2)
            np.testing.assert_array_equal(_snapshot(dx), [[-.1005859375, -.201171875, .703125, -.40234375],
                                                        [1.2109375, 2.421875, -8.4375, 4.84375]])
            weight = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.float32, requires_grad=True)
            bias = torch.tensor([-.125, .25, -.375, .5], dtype=torch.float32, requires_grad=True)
            with patch.object(owner, "_ln_function_cls", wraps=owner._ln_function_cls) as selected:
                out = torch.nn.functional.layer_norm(x, (4,), weight, bias)
                gradients = torch.autograd.grad((out.float() * dy.float()).sum(), (x, weight, bias))
            assert selected.call_count == 1 and len(selected.call_args.args) == 2
            assert out.dtype == gradients[0].dtype == torch.bfloat16
            assert gradients[1].dtype == gradients[2].dtype == torch.float32
            np.testing.assert_array_equal(_snapshot(out), _OUTPUT)
            np.testing.assert_array_equal(_snapshot(gradients[0]), _DX)

    def test_backward_keeps_values_when_callers_rebind_input_holders(self, device):
        with forbid_backend_fallbacks():
            x = torch.tensor(_X, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.bfloat16, requires_grad=True)
            bias = torch.tensor([-.125, .25, -.375, .5], dtype=torch.bfloat16, requires_grad=True)
            xholder, wholder = (jt.ops.reshape(v, v.shape) for v in (x, weight))
            output = torch.nn.functional.layer_norm(xholder, (4,), wholder, bias)
            output.sync()
            xholder.assign(torch.full(x.shape, 7., dtype=torch.bfloat16))
            wholder.assign(torch.full(weight.shape, 7., dtype=torch.bfloat16))
            dy = torch.tensor(_DY, dtype=torch.bfloat16)
            dx, dw, db = torch.autograd.grad((output.float() * dy.float()).sum(), (x, weight, bias))
            np.testing.assert_array_equal(_snapshot(dx), _DX)
            np.testing.assert_array_equal(_snapshot(dw), [.3359375, -.224609375, -.44921875, 2.6875])
            np.testing.assert_array_equal(_snapshot(db), [-.25, .5, -1., 2.])


instantiate_device_type_tests(TestBFloat16LayerNormCpu, globals(), only_for=("cpu",))
