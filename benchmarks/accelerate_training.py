"""Bounded FP32 Jittor Torch training, with and without Accelerate.

Run only through the dedicated benchmark_accelerate session: this benchmark
claims the Torch namespace and must not share an interpreter with real Torch.
Each measurement trains eight resident batches and waits for actual device
completion. Setup warms every shape, checks gradients, then discards all warm
training objects and reconstructs the fixed initial state outside timing.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from ._shared import cleanup_backend, require_isolated_cache, synchronize


class AccelerateTrainingBenchmarks:
    params = (["direct", "full"], ["sgd", "adamw"], ["cpu", "cuda"])
    param_names = ["wrapping", "optimizer", "device"]
    number = 1
    repeat = (3, 3, 30.0)
    rounds = 1
    timeout = 300
    steps = 8

    def setup(self, wrapping, optimizer_name, device):
        if os.environ.get("cache_name") != "asv-accelerate":
            raise NotImplementedError("Accelerate training requires the dedicated benchmark_accelerate session")
        require_isolated_cache()
        existing = sys.modules.get("torch")
        if existing is not None and not hasattr(existing, "_torch_compat_install_context"):
            raise RuntimeError("Accelerate ASV requires an isolated Jittor Torch process")
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        self.jt = importlib.import_module("jittor")
        self.torch = importlib.import_module("torch")
        if self.torch is self.jt or not hasattr(self.torch, "_torch_compat_install_context"):
            raise RuntimeError("Accelerate ASV did not resolve the independent Jittor frontend")
        if device == "cuda" and not self.jt.compiler.has_cuda:
            raise NotImplementedError("Accelerate ASV CUDA requires a CUDA build")
        self.jt.flags.use_cuda = int(device == "cuda")
        self.jt.flags.cuda_allow_tf32 = 0
        self.jt.flags.cuda_allow_cudnn_tf32 = 0
        self.jt.flags.use_tensorcore = 0
        self.jt.flags.float32_matmul_precision = "highest"
        self.device = device
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState, GradientState
        from jittor._runtime.fallback import forbid_backend_fallbacks

        self.Accelerator = Accelerator
        self.AcceleratorState = AcceleratorState
        self.GradientState = GradientState
        self.fallback = forbid_backend_fallbacks()
        self.fallback.__enter__()
        rng = np.random.default_rng(20260916)
        self.initial = [rng.normal(0, .02, shape).astype("float32") for shape in
                        ((2048, 1024), (2048,), (1024, 2048), (1024,))]
        # Four slots prevent the workload from degenerating into one input's
        # cache reuse. Data stays resident; loader transfer and collation are
        # included equally in direct and full modes.
        x = rng.normal(size=(64, 1024)).astype("float32")
        y = rng.normal(size=(64, 1024)).astype("float32")
        self.dataset = self.torch.utils.data.TensorDataset(
            self.torch.tensor(x, device=device), self.torch.tensor(y, device=device))
        if any(tensor.device.type != device for tensor in self.dataset.tensors):
            raise RuntimeError("Accelerate ASV data is not on the requested device")
        self._build(wrapping, optimizer_name)
        self._train(4, validate=True)
        self._discard_training_objects()
        self._build(wrapping, optimizer_name)
        synchronize("jittor", self.jt, device)
        audit_directory = os.environ.get("JITTOR_ASV_AUDIT_DIR")
        if audit_directory:
            repo = Path(__file__).resolve().parents[1]
            sources = ("benchmarks/accelerate_training.py", "compat/torch/grad.py",
                       "compat/torch/installers/tensor/method_api.py",
                       "compat/torch/installers/data.py", "compat/torch/optimizer_api.py",
                       "python/jittor/optim/base.py", "src/runtime/init.cc",
                       "src/core/var.cc")
            report = {
                "pid": os.getpid(), "wrapping": wrapping, "optimizer": optimizer_name,
                "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "benchmark_module": str(Path(__file__).resolve()),
                "jittor_module": str(Path(self.jt.__file__).resolve()),
                "torch_namespace": str(type(self.torch)),
                "torch_compat_active": hasattr(self.torch, "_torch_compat_install_context"),
                "cache": self.jt.compiler.cache_path, "warmup_steps": 4,
                "warmup_gradient_and_update_sanity": True, "timed_steps": self.steps,
                "timed_host_tensor_reads": False, "device_sync": "block boundary, true device wait",
                "fallback_count": self.jt.core.backend_fallback_count(),
                "initial_sha256": hashlib.sha256(b"".join(value.tobytes() for value in self.initial)).hexdigest(),
                "inputs_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                "source_hashes": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources},
            }
            output = Path(audit_directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / ("%s_%s_%s_%d.json" % (wrapping, optimizer_name, device, os.getpid()))).write_text(json.dumps(report, indent=2))

    def _build(self, wrapping, optimizer_name):
        torch = self.torch
        self.AcceleratorState._reset_state(reset_partial_state=True)
        self.GradientState._shared_state.clear()
        self.model = torch.nn.Sequential(torch.nn.Linear(1024, 2048),
                                         torch.nn.ReLU(), torch.nn.Linear(2048, 1024)).to(self.device)
        with torch.no_grad():
            for parameter, value in zip(self.model.parameters(), self.initial):
                parameter.copy_(torch.tensor(value, device=self.device))
        if any(parameter.device.type != self.device for parameter in self.model.parameters()):
            raise RuntimeError("Accelerate ASV model is not on the requested device")
        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01,
                                             momentum=.9, weight_decay=.0005)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001,
                                               betas=(.9, .999), weight_decay=.01)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=16,
                                                 shuffle=False, num_workers=0)
        self.accelerator = None
        if wrapping == "full":
            self.accelerator = self.Accelerator(cpu=self.device == "cpu", mixed_precision="no")
            self.model, self.optimizer, self.loader = self.accelerator.prepare(
                self.model, self.optimizer, self.loader)

    def _train(self, steps, validate=False):
        iterator = iter(self.loader)
        for index in range(steps):
            if index and index % 4 == 0:
                iterator = iter(self.loader)
            x, target = next(iterator)
            self.optimizer.zero_grad(set_to_none=True)
            loss = (self.model(x) - target).square().mean()
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            if validate and index == 0:
                if not np.isfinite(float(loss.item())):
                    raise RuntimeError("Accelerate ASV warmup loss is non-finite")
                for parameter in self.model.parameters():
                    grad = parameter.grad
                    if grad is None or grad.device.type != self.device:
                        raise RuntimeError("Accelerate ASV warmup gradient/device mismatch")
                    value = grad.detach().clone().cpu().numpy()
                    if not np.isfinite(value).all() or not np.any(value):
                        raise RuntimeError("Accelerate ASV warmup gradient is zero/non-finite")
            self.optimizer.step()
        # Parameter updates remain live on the model and are consumed before
        # ending timing; no per-step host metrics or phase synchronizations.
        synchronize("jittor", self.jt, self.device)
        if validate:
            for parameter, initial in zip(self.model.parameters(), self.initial):
                value = parameter.detach().clone().cpu().numpy()
                if not np.isfinite(value).all() or np.array_equal(value, initial):
                    raise RuntimeError("Accelerate ASV warmup did not update every parameter")

    def time_eight_training_steps(self, wrapping, optimizer_name, device):
        self._train(self.steps)

    def _discard_training_objects(self):
        if self.accelerator is not None:
            self.accelerator.free_memory()
        self.model = self.optimizer = self.loader = self.accelerator = None
        gc.collect()

    def teardown(self, wrapping, optimizer_name, device):
        self._discard_training_objects()
        self.dataset = None
        self.AcceleratorState._reset_state(reset_partial_state=True)
        self.GradientState._shared_state.clear()
        self.fallback.__exit__(None, None, None)
        cleanup_backend("jittor", self.jt)
