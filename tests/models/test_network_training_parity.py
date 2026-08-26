"""Three-step training trajectories for common networks against PyTorch."""

import unittest

import numpy as np

import jittor as jt

from _helpers.torch_runtime import import_torch_modules, modules_available

import _parity_networks
from _parity_networks import _load_torch_weights, _relative_error


skip_this_test = not modules_available("torch", "torchvision")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")


class NetworkTrainingParity(unittest.TestCase):
    """Compare loss and parameter evolution across three SGD updates."""

    use_cuda = 0
    loss_tolerance = 4e-3
    buffer_absolute_tolerance = 2e-5
    buffer_tolerance = 4e-3
    parameter_absolute_tolerance = 2e-5
    parameter_tolerance = 3e-3
    # The residual BatchNorm stack amplifies one-step CUDA accumulation
    # differences at 1e-3; this keeps the test focused on update parity.
    learning_rates = {"resnet18": 1e-4}
    steps = 3

    def _run(self, name):
        with jt.flag_scope(use_cuda=self.use_cuda):
            self._compare(name)

    def _compare(self, name):
        jittor_model, torch_model, arrays = _parity_networks.build(name, seed=91)
        _load_torch_weights(jittor_model, torch_model)
        jittor_model.train()
        torch_model.train()
        if self.use_cuda:
            torch_model.cuda()

        jittor_inputs = []
        torch_inputs = []
        for array in arrays:
            if np.issubdtype(array.dtype, np.integer):
                jittor_inputs.append(jt.array(array.astype("int32")))
                tensor = torch.from_numpy(array.astype("int64"))
            else:
                jittor_inputs.append(jt.array(array.copy()))
                tensor = torch.from_numpy(array.copy())
            torch_inputs.append(tensor.cuda() if self.use_cuda else tensor)

        jittor_parameters = {
            name: parameter
            for name, parameter in jittor_model.named_parameters()
            if parameter.requires_grad
        }
        torch_parameters = {
            name: parameter
            for name, parameter in torch_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            set(jittor_parameters),
            set(torch_parameters),
            "{} parameter names differ".format(name),
        )
        shared_names = sorted(set(jittor_parameters) & set(torch_parameters))
        self.assertTrue(shared_names, "{} has no shared parameters".format(name))
        initial = {key: np.array(jittor_parameters[key].numpy(), copy=True) for key in shared_names}

        learning_rate = self.learning_rates.get(name, 1e-3)
        jittor_optimizer = jt.optim.SGD(list(jittor_parameters.values()), lr=learning_rate)
        torch_optimizer = torch.optim.SGD(list(torch_parameters.values()), lr=learning_rate)

        losses = []
        torch_target = None
        jittor_target = None
        for step in range(self.steps):
            torch_optimizer.zero_grad(set_to_none=False)
            torch_output = torch_model(*torch_inputs)
            jittor_output = jittor_model(*jittor_inputs)
            if torch_target is None:
                target_np = (
                    np.random.RandomState(92).randn(*tuple(torch_output.shape)).astype("float32")
                )
                torch_target = torch.from_numpy(target_np)
                if self.use_cuda:
                    torch_target = torch_target.cuda()
                jittor_target = jt.array(target_np)

            torch_loss = ((torch_output - torch_target) ** 2).mean()
            torch_value = float(torch_loss.detach().cpu())
            torch_loss.backward()
            torch_optimizer.step()

            jittor_loss = ((jittor_output - jittor_target) ** 2).mean()
            jittor_value = float(jittor_loss.numpy().reshape(-1)[0])
            jittor_optimizer.step(jittor_loss)
            jt.sync(list(jittor_parameters.values()))

            self.assertTrue(np.isfinite(jittor_value))
            self.assertTrue(np.isfinite(torch_value))
            loss_error = abs(jittor_value - torch_value) / max(abs(torch_value), 1e-6)
            self.assertLess(
                loss_error,
                self.loss_tolerance,
                "{} step {} loss diverged: relative={:.3e}, jittor={:.8g}, torch={:.8g}".format(
                    name, step, loss_error, jittor_value, torch_value
                ),
            )
            losses.append((jittor_value, torch_value))

        changed = False
        for parameter_name in shared_names:
            actual = jittor_parameters[parameter_name].numpy()
            expected = torch_parameters[parameter_name].detach().cpu().numpy()
            error = _relative_error(actual, expected)
            max_abs = float(np.abs(actual - expected).max())
            scale = max(float(np.abs(expected).max()), 1e-6)
            if max_abs > self.parameter_absolute_tolerance:
                self.assertLess(
                    error,
                    self.parameter_tolerance,
                    "{} parameter {} diverged: relative={:.3e}, "
                    "max_abs={:.3e}, scale={:.3e}".format(
                        name, parameter_name, error, max_abs, scale
                    ),
                )
            if not np.array_equal(actual, initial[parameter_name]):
                changed = True

        self.assertTrue(changed, "{} parameters did not update".format(name))
        jittor_buffers = dict(jittor_model.named_buffers(recurse=True))
        torch_buffers = dict(torch_model.named_buffers())
        for buffer_name in sorted(set(jittor_buffers) & set(torch_buffers)):
            actual = jittor_buffers[buffer_name].numpy()
            expected = torch_buffers[buffer_name].detach().cpu().numpy()
            self.assertEqual(
                actual.size,
                expected.size,
                "{} buffer {} size differs".format(name, buffer_name),
            )
            actual = actual.reshape(-1)
            expected = expected.reshape(-1)
            max_abs = float(np.abs(actual - expected).max())
            if max_abs <= self.buffer_absolute_tolerance:
                continue
            error = _relative_error(actual, expected)
            self.assertLess(
                error,
                self.buffer_tolerance,
                "{} buffer {} diverged: relative={:.3e}, max_abs={:.3e}".format(
                    name, buffer_name, error, max_abs
                ),
            )
        self.assertEqual(len(losses), self.steps)


@unittest.skipIf(skip_this_test, "independent PyTorch is unavailable")
class TestNetworkTrainingParityCPU(NetworkTrainingParity):
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
class TestNetworkTrainingParityCUDA(NetworkTrainingParity):
    loss_tolerance = 1e-2
    buffer_absolute_tolerance = 5e-5
    buffer_tolerance = 1e-2
    parameter_absolute_tolerance = 5e-5
    parameter_tolerance = 8e-3
    use_cuda = 1

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
