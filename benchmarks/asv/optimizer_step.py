"""Optimizer-step scaling as parameter tensor count increases."""

from __future__ import annotations

import numpy as np

from ._shared import cleanup_backend, load_backend, synchronize, working_set_bytes


class OptimizerStepBenchmarks:
    params = (["sgd", "adamw"], [32, 128, 512], ["cpu", "cuda"])
    param_names = ["optimizer", "tensor_count", "device"]
    number = 1
    repeat = (3, 7, 60.0)
    rounds = 1
    timeout = 300
    total_elements = 262144

    def setup(self, optimizer_name, tensor_count, device):
        self.jt = load_backend("jittor", device)
        size = max(1, self.total_elements // tensor_count)
        self.params_for_step = [
            self.jt.array(np.full((size,), 0.1 + index * 1e-7, dtype="float32"))
            for index in range(tensor_count)
        ]
        for parameter in self.params_for_step:
            parameter.start_grad()

        if optimizer_name == "sgd":
            self.optimizer = self.jt.optim.SGD(
                self.params_for_step,
                lr=1e-3,
                momentum=0.9,
                weight_decay=0.01,
            )
        elif optimizer_name == "adamw":
            self.optimizer = self.jt.optim.AdamW(
                self.params_for_step,
                lr=1e-3,
                weight_decay=0.01,
            )
        else:
            raise ValueError(optimizer_name)

        self.gradient_sources = [
            self.jt.ones(parameter.shape, dtype="float32").stop_grad()
            for parameter in self.params_for_step
        ]
        self.gradients = [
            self.jt.ones(parameter.shape, dtype="float32").stop_grad()
            for parameter in self.params_for_step
        ]
        self._attach_gradients()

        # Compile and validate the update graph outside the timed region.
        self.optimizer.step()
        synchronize("jittor", self.jt, device)
        sample = float(self.params_for_step[0].sum().item())
        if not np.isfinite(sample):
            raise RuntimeError("optimizer warmup produced a non-finite parameter")
        self._reset_gradients()

    def _attach_gradients(self):
        self.optimizer.param_groups[0]["grads"] = self.gradients
        self.optimizer._build_grad_map()

    def _reset_gradients(self):
        for gradient, source in zip(self.gradients, self.gradient_sources):
            gradient.update(source)
        self._attach_gradients()
        synchronize("jittor", self.jt, self.device)

    @property
    def device(self):
        return "cuda" if self.jt.flags.use_cuda else "cpu"

    def time_step(self, optimizer_name, tensor_count, device):
        self.optimizer.step()
        synchronize("jittor", self.jt, device)

    def track_working_set_bytes(self, optimizer_name, tensor_count, device):
        self.optimizer.step()
        synchronize("jittor", self.jt, device)
        return working_set_bytes("jittor", self.jt, device)

    track_working_set_bytes.unit = "bytes"

    def teardown(self, optimizer_name, tensor_count, device):
        backend = getattr(self, "jt", None)
        self.params_for_step = []
        self.gradient_sources = []
        self.gradients = []
        if hasattr(self, "optimizer"):
            del self.optimizer
        if backend is not None:
            cleanup_backend("jittor", backend)
