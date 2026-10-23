"""Convolution inference and training latency, checked against a NumPy oracle.

No external model package or downloaded weights are needed. Each setup checks
the actual timed shape, including input, weight and bias gradients in training
mode. Strict float32 (TF32 disabled) keeps this a single numerical contract.
CUDA runs reject CPU fallback and check all operand and result residency.
"""

from __future__ import annotations

import numpy as np

from ._shared import (
    as_numpy,
    backend_tensor,
    cleanup_backend,
    load_backend,
    synchronize,
    working_set_bytes,
)


CASES = {
    "standard_3x3": ((4, 16, 32, 32), (32, 16, 3, 3), 1, 1, 1),
    "pointwise_stride2": ((4, 32, 32, 32), (64, 32, 1, 1), 2, 0, 1),
    "depthwise_3x3": ((4, 32, 32, 32), (32, 1, 3, 3), 1, 1, 32),
}


def _reference(x, weight, bias, cotangent, stride, padding, groups):
    """Float64 cross-correlation and its vector-Jacobian product, in NCHW."""
    x = np.asarray(x, dtype="float64")
    weight = np.asarray(weight, dtype="float64")
    n, channels, height, width = x.shape
    outputs, channels_per_group, kh, kw = weight.shape
    if channels != channels_per_group * groups or outputs % groups:
        raise ValueError("incompatible convolution groups")
    oh = (height + 2 * padding - kh) // stride + 1
    ow = (width + 2 * padding - kw) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.broadcast_to(
        np.asarray(bias, dtype="float64")[None, :, None, None], (n, outputs, oh, ow)
    ).copy()
    grad_padded = np.zeros_like(padded)
    grad_weight = np.zeros_like(weight)
    cotangent = np.asarray(cotangent, dtype="float64")
    if cotangent.shape != output.shape:
        raise ValueError("cotangent does not match convolution output")
    for group in range(groups):
        ci = slice(group * channels_per_group, (group + 1) * channels_per_group)
        co = slice(group * (outputs // groups), (group + 1) * (outputs // groups))
        dy = cotangent[:, co]
        for i in range(kh):
            for j in range(kw):
                hs = slice(i, i + oh * stride, stride)
                ws = slice(j, j + ow * stride, stride)
                patch = padded[:, ci, hs, ws]
                kernel = weight[co, :, i, j]
                output[:, co] += np.einsum("nchw,oc->nohw", patch, kernel)
                grad_weight[co, :, i, j] = np.einsum("nohw,nchw->oc", dy, patch)
                grad_padded[:, ci, hs, ws] += np.einsum("nohw,oc->nchw", dy, kernel)
    grad_input = grad_padded[:, :, padding : padding + height, padding : padding + width]
    grad_bias = cotangent.sum(axis=(0, 2, 3))
    return output, (grad_input, grad_weight, grad_bias)


def _assert_result(output, gradients, expected_output, expected_gradients):
    values = [("output", output, expected_output)]
    if len(gradients) != len(expected_gradients):
        raise RuntimeError("convolution did not return every requested gradient")
    values.extend(
        zip(("input gradient", "weight gradient", "bias gradient"), gradients, expected_gradients)
    )
    for name, actual, expected in values:
        actual = np.asarray(actual)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise RuntimeError("convolution %s is non-finite or has the wrong shape" % name)
        if not np.allclose(actual, expected, rtol=3e-4, atol=3e-5):
            error = float(np.max(np.abs(actual - expected)))
            raise RuntimeError("convolution %s differs from NumPy (max error %g)" % (name, error))


class Conv2dBenchmarks:
    params = (["jittor", "torch"], ["cpu", "cuda"], list(CASES), ["forward", "forward_backward"])
    param_names = ["backend", "device", "case", "mode"]
    number = 1
    repeat = (3, 7, 30.0)
    rounds = 1
    timeout = 300

    def setup(self, backend_name, device, case, mode):
        if device not in ("cpu", "cuda"):
            raise ValueError("unsupported convolution device: %s" % device)
        if mode not in ("forward", "forward_backward"):
            raise ValueError("unsupported convolution mode: %s" % mode)
        x_shape, w_shape, self.stride, self.padding, self.groups = CASES[case]
        self.backend_name, self.device, self.mode = backend_name, device, mode
        self.backend = load_backend(backend_name, device)
        self._saved_tf32 = None
        self._saved_flags = {}
        if device == "cuda":
            if backend_name == "jittor":
                if not self.backend.flags.use_cuda:
                    raise RuntimeError("CUDA convolution cannot run on a CPU backend")
                for name, value in (
                    ("float32_matmul_precision", "highest"),
                    ("use_tensorcore", 0),
                    ("cuda_allow_tf32", 0),
                    ("cuda_allow_cudnn_tf32", 0),
                    ("backend_fallback", "error"),
                ):
                    self._saved_flags[name] = getattr(self.backend.flags, name)
                    setattr(self.backend.flags, name, value)
                self._fallback_before = self.backend.core.backend_fallback_count()
            else:
                self._saved_tf32 = self.backend.backends.cudnn.allow_tf32
                self.backend.backends.cudnn.allow_tf32 = False

        rng = np.random.default_rng(20261014)
        host_x = (rng.standard_normal(x_shape) * 0.2).astype("float32")
        host_weight = (rng.standard_normal(w_shape) * 0.2).astype("float32")
        host_bias = (rng.standard_normal(w_shape[0]) * 0.1).astype("float32")
        oh = (x_shape[2] + 2 * self.padding - w_shape[2]) // self.stride + 1
        ow = (x_shape[3] + 2 * self.padding - w_shape[3]) // self.stride + 1
        host_cotangent = (rng.standard_normal((x_shape[0], w_shape[0], oh, ow)) * 0.01).astype(
            "float32"
        )
        self.x, self.weight, self.bias = [
            backend_tensor(backend_name, self.backend, host, device)
            for host in (host_x, host_weight, host_bias)
        ]
        if mode == "forward_backward":
            self.cotangent = backend_tensor(backend_name, self.backend, host_cotangent, device)
        for value in (self.x, self.weight, self.bias):
            if backend_name == "jittor":
                value.start_grad() if mode == "forward_backward" else value.stop_grad()
            else:
                value.requires_grad_(mode == "forward_backward")
                if value.device.type != device:
                    raise RuntimeError("convolution tensor is on the wrong device")
        if backend_name == "jittor" and mode == "forward_backward":
            self.cotangent.stop_grad()
        synchronize(backend_name, self.backend, device)
        expected, expected_grads = _reference(
            host_x, host_weight, host_bias, host_cotangent, self.stride, self.padding, self.groups
        )
        self._run()
        synchronize(backend_name, self.backend, device)
        if backend_name == "jittor" and mode == "forward_backward":
            # Backward may release the output's allocation after consuming it.
            # Materialize it during setup, outside the timed benchmark, before
            # checking the actual output placement and comparing the oracle.
            self.output.sync()
        self._assert_device()
        _assert_result(
            as_numpy(backend_name, self.output),
            [as_numpy(backend_name, grad) for grad in self.gradients],
            expected,
            expected_grads if mode == "forward_backward" else (),
        )
        # Warm a second iteration to exercise graph reuse before ASV times it.
        self._run()
        synchronize(backend_name, self.backend, device)
        if backend_name == "jittor" and mode == "forward_backward":
            self.output.sync()
        self._assert_device()

    def _assert_device(self):
        values = [self.x, self.weight, self.bias, self.output] + list(self.gradients)
        if self.mode == "forward_backward":
            values.append(self.cotangent)
        if self.backend_name == "jittor":
            expected = "device" if self.device == "cuda" else "cpu"
            if any(value.location() != expected for value in values):
                raise RuntimeError("convolution operands or results are on the wrong device")
            if self.device == "cuda":
                if self.backend.core.backend_fallback_count() != self._fallback_before:
                    raise RuntimeError("CUDA convolution attempted a CPU fallback")
        elif any(value.device.type != self.device for value in values):
            raise RuntimeError("convolution operands or results are on the wrong device")

    def _forward(self):
        functional = (
            self.backend.nn if self.backend_name == "jittor" else self.backend.nn.functional
        )
        return functional.conv2d(
            self.x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

    def _run(self):
        if self.mode == "forward":
            with self.backend.no_grad():
                self.output = self._forward()
            self.gradients = ()
        else:
            self.output = self._forward()
            loss = (self.output * self.cotangent).sum()
            inputs = (self.x, self.weight, self.bias)
            if self.backend_name == "jittor":
                self.gradients = self.backend.grad(loss, inputs)
            else:
                self.gradients = self.backend.autograd.grad(loss, inputs)

    def time_convolution(self, backend_name, device, case, mode):
        self._run()
        # Every lazy iteration must execute, including all three gradients.
        synchronize(backend_name, self.backend, device)

    def track_working_set_bytes(self, backend_name, device, case, mode):
        self._run()
        synchronize(backend_name, self.backend, device)
        return working_set_bytes(backend_name, self.backend, device)

    track_working_set_bytes.unit = "bytes"

    def teardown(self, backend_name, device, case, mode):
        backend = getattr(self, "backend", None)
        for name in ("x", "weight", "bias", "cotangent", "output", "gradients"):
            if hasattr(self, name):
                delattr(self, name)
        if backend is not None:
            for name, value in self._saved_flags.items():
                setattr(backend.flags, name, value)
            if self._saved_tf32 is not None:
                backend.backends.cudnn.allow_tf32 = self._saved_tf32
            cleanup_backend(backend_name, backend)
