"""Operator-level Jittor and optional real-PyTorch comparisons."""

from __future__ import annotations

import numpy as np

from ._shared import (
    as_numpy,
    backend_tensor,
    cleanup_backend,
    load_backend,
    reset_memory_stats,
    synchronize,
    working_set_bytes,
)


class OperatorBenchmarks:
    params = (["jittor", "torch"], ["cpu", "cuda"], ["matmul", "softmax", "layernorm", "gelu"])
    param_names = ["backend", "device", "operator"]
    number = 1
    repeat = (3, 7, 30.0)
    rounds = 1
    timeout = 180

    def setup(self, backend_name, device, operator):
        self.backend_name = backend_name
        self.device = device
        self.backend = load_backend(backend_name, device)
        rng = np.random.default_rng(20260811)

        if operator == "matmul":
            self.a = backend_tensor(
                backend_name,
                self.backend,
                rng.standard_normal((8, 256, 256)).astype("float32"),
                device,
            )
            self.b = backend_tensor(
                backend_name,
                self.backend,
                rng.standard_normal((8, 256, 256)).astype("float32"),
                device,
            )
        elif operator == "softmax":
            self.x = backend_tensor(
                backend_name,
                self.backend,
                rng.standard_normal((16, 128, 1024)).astype("float32"),
                device,
            )
        elif operator == "layernorm":
            self.x = backend_tensor(
                backend_name,
                self.backend,
                rng.standard_normal((16, 128, 768)).astype("float32"),
                device,
            )
            self.weight = backend_tensor(
                backend_name, self.backend, np.ones((768,), dtype="float32"), device
            )
            self.bias = backend_tensor(
                backend_name, self.backend, np.zeros((768,), dtype="float32"), device
            )
        elif operator == "gelu":
            self.x = backend_tensor(
                backend_name,
                self.backend,
                rng.standard_normal((16, 128, 1024)).astype("float32"),
                device,
            )
        else:
            raise ValueError(operator)

        self.output = self._run(operator)
        synchronize(backend_name, self.backend, device)
        output = as_numpy(backend_name, self.output)
        if output.size == 0 or not np.isfinite(output).all():
            raise RuntimeError("%s produced an empty or non-finite output" % operator)

    def _run(self, operator):
        backend = self.backend
        with backend.no_grad():
            if operator == "matmul":
                return backend.matmul(self.a, self.b)
            if operator == "softmax":
                return backend.nn.functional.softmax(self.x, dim=-1)
            if operator == "layernorm":
                return backend.nn.functional.layer_norm(
                    self.x, (768,), self.weight, self.bias, 1e-5
                )
            if operator == "gelu":
                return backend.nn.functional.gelu(self.x)
        raise ValueError(operator)

    def time_operator(self, backend_name, device, operator):
        self.output = self._run(operator)
        synchronize(backend_name, self.backend, device)

    def track_working_set_bytes(self, backend_name, device, operator):
        reset_memory_stats(backend_name, self.backend, device)
        self.output = self._run(operator)
        synchronize(backend_name, self.backend, device)
        return working_set_bytes(backend_name, self.backend, device)

    track_working_set_bytes.unit = "bytes"

    def teardown(self, backend_name, device, operator):
        backend = self.backend
        for name in ("a", "b", "x", "weight", "bias", "output"):
            if hasattr(self, name):
                delattr(self, name)
        cleanup_backend(backend_name, backend)
