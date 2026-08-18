"""Forward and backward parity for common networks against independent PyTorch.

``tests/models/test_models.py`` compares the torchvision-style zoo forward pass
only.  This module closes the two gaps the 2.0 goals name explicitly:

  * the **backward** pass is compared as well, for the input gradient and for
    every parameter gradient, so an operator whose derivative diverges cannot
    hide behind a matching forward;
  * the architecture set covers ResNet, ViT, GPT-2 and a diffusion UNet rather
    than convolutional classifiers alone.

Weights are copied from the PyTorch module into the Jittor module before the
comparison, so a failure is an operator-semantics difference and never an
initialization difference.  Run with an independent binary PyTorch::

    REAL_TORCH_SITE=/path/to/site-packages python -m pytest tests/models/test_network_parity.py
"""

import unittest

import numpy as np

import jittor as jt

from _helpers.torch_runtime import import_torch_modules, modules_available

import _parity_networks


skip_this_test = not modules_available("torch", "torchvision")

torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")


def _relative_error(actual, expected):
    """Scale-aware error that stays meaningful for near-zero references."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1e-6)
    return float(np.abs(actual - expected).max() / scale)


def _load_torch_weights(jittor_model, torch_model):
    """Copy PyTorch parameters and buffers into the matched Jittor module."""
    state = {
        key: value.detach().cpu().numpy()
        for key, value in torch_model.state_dict().items()
    }
    jittor_model.load_parameters(state)
    loaded = {name for name, _ in jittor_model.named_parameters()}
    missing = sorted(key for key in state if key not in loaded)
    return missing


def _eval_like_torch(jittor_model):
    """Switch to inference behaviour while keeping parameters differentiable.

    ``jittor.nn.Module.eval`` also calls ``stop_grad`` on every parameter, so a
    plain ``eval()`` here would silently reduce the parameter-gradient
    comparison below to nothing.  PyTorch's ``eval()`` only changes dropout and
    normalization behaviour, which is the semantics this comparison needs.
    """
    jittor_model.eval()
    for _name, parameter in jittor_model.named_parameters():
        parameter.start_grad()


class NetworkParity(unittest.TestCase):
    """Compare a Jittor network with a structurally identical PyTorch one."""

    forward_tolerance = 2e-4
    backward_tolerance = 2e-3

    def _run(self, name):
        jittor_model, torch_model, inputs = _parity_networks.build(name)
        torch_model.eval()
        _load_torch_weights(jittor_model, torch_model)
        _eval_like_torch(jittor_model)

        torch_inputs = []
        jittor_inputs = []
        for array in inputs:
            if np.issubdtype(array.dtype, np.integer):
                torch_inputs.append(torch.from_numpy(array.astype("int64")))
                jittor_inputs.append(jt.array(array.astype("int32")))
            else:
                tensor = torch.from_numpy(array.copy())
                tensor.requires_grad_(True)
                torch_inputs.append(tensor)
                variable = jt.array(array.copy())
                variable.requires_grad = True
                jittor_inputs.append(variable)

        torch_output = torch_model(*torch_inputs)
        jittor_output = jittor_model(*jittor_inputs)

        self.assertEqual(tuple(jittor_output.shape), tuple(torch_output.shape))
        forward_error = _relative_error(
            jittor_output.numpy(), torch_output.detach().cpu().numpy()
        )
        self.assertLess(
            forward_error,
            self.forward_tolerance,
            "{} forward diverged: {:.3e}".format(name, forward_error),
        )

        # A plain sum would let sign-cancelling per-element errors mask a real
        # gradient divergence, so weight the loss with a fixed random pattern.
        weights = np.random.RandomState(1234).randn(*torch_output.shape).astype("float32")
        torch_loss = (torch_output * torch.from_numpy(weights)).sum()
        jittor_loss = (jittor_output * jt.array(weights)).sum()

        torch_loss.backward()
        float_inputs = [
            (jittor_input, torch_input)
            for jittor_input, torch_input in zip(jittor_inputs, torch_inputs)
            if torch_input.dtype.is_floating_point
        ]
        jittor_parameters = [
            (name_, parameter)
            for name_, parameter in jittor_model.named_parameters()
            if parameter.requires_grad
        ]
        targets = [variable for variable, _ in float_inputs]
        targets += [parameter for _, parameter in jittor_parameters]
        jittor_grads = jt.grad(jittor_loss, targets)

        torch_named = dict(torch_model.named_parameters())
        checked = 0
        for index, (jittor_input, torch_input) in enumerate(float_inputs):
            self.assertIsNotNone(torch_input.grad, "torch input {} has no grad".format(index))
            error = _relative_error(
                jittor_grads[index].numpy(), torch_input.grad.detach().cpu().numpy()
            )
            self.assertLess(
                error,
                self.backward_tolerance,
                "{} input[{}] gradient diverged: {:.3e}".format(name, index, error),
            )
            checked += 1

        offset = len(float_inputs)
        for index, (parameter_name, _parameter) in enumerate(jittor_parameters):
            reference = torch_named.get(parameter_name)
            if reference is None or reference.grad is None:
                continue
            error = _relative_error(
                jittor_grads[offset + index].numpy(), reference.grad.detach().cpu().numpy()
            )
            self.assertLess(
                error,
                self.backward_tolerance,
                "{} grad[{}] diverged: {:.3e}".format(name, parameter_name, error),
            )
            checked += 1

        # Guard the comparison itself: a name mismatch, or a mode switch that
        # detaches parameters, would otherwise silently reduce this to a
        # forward-only test.
        expected = len(float_inputs) + sum(
            1
            for parameter_name, _parameter in jittor_parameters
            if parameter_name in torch_named and torch_named[parameter_name].grad is not None
        )
        self.assertEqual(
            checked,
            expected,
            "{} compared {} of {} gradients".format(name, checked, expected),
        )
        self.assertGreater(checked, 1, "{} compared only {} gradients".format(name, checked))


@unittest.skipIf(skip_this_test, "independent PyTorch is unavailable")
class TestNetworkParityCPU(NetworkParity):
    def test_resnet18(self):
        self._run("resnet18")

    def test_vit(self):
        self._run("vit")

    def test_gpt2(self):
        self._run("gpt2")

    def test_diffusion_unet(self):
        self._run("diffusion_unet")


@unittest.skipIf(skip_this_test, "independent PyTorch is unavailable")
@unittest.skipIf(not jt.has_cuda, "CUDA is unavailable")
class TestNetworkParityCUDA(NetworkParity):
    # CUDA convolution and reduction kernels select different accumulation
    # orders than the CPU reference, so the accepted band is wider here.
    forward_tolerance = 1e-3
    backward_tolerance = 6e-3

    def _run(self, name):
        with jt.flag_scope(use_cuda=1):
            super()._run(name)

    def test_resnet18(self):
        self._run("resnet18")

    def test_vit(self):
        self._run("vit")

    def test_gpt2(self):
        self._run("gpt2")

    def test_diffusion_unet(self):
        self._run("diffusion_unet")


if __name__ == "__main__":
    unittest.main()
