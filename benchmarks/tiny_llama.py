"""Tiny Llama forward and backward benchmarks using Transformers 4.56.2."""

from __future__ import annotations

import contextlib

import numpy as np

from ._shared import (
    as_numpy,
    backend_tensor,
    cleanup_backend,
    import_transformers,
    load_backend,
    reset_memory_stats,
    synchronize,
    working_set_bytes,
)


class TinyLlamaBenchmarks:
    params = (["jittor", "torch"], ["forward", "forward_backward"])
    param_names = ["backend", "phase"]
    number = 1
    repeat = (3, 5, 60.0)
    rounds = 1
    timeout = 300

    def setup(self, backend_name, phase):
        self.backend_name = backend_name
        self.device = "cuda"
        backend = load_backend(backend_name, self.device)
        self.backend, transformers = import_transformers(backend_name, backend)

        config = transformers.LlamaConfig(
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=768,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=256,
            attention_dropout=0.0,
            hidden_act="silu",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        self.model = transformers.LlamaModel(config)
        self.model.eval()
        if backend_name == "torch":
            self.model = self.model.to(self.backend.device("cuda"))

        self.params_for_grad = tuple(self.model.parameters())
        if not self.params_for_grad:
            raise RuntimeError("Tiny Llama exposed no trainable parameters")
        if phase == "forward_backward":
            for parameter in self.params_for_grad:
                parameter.requires_grad_(True)

        rng = np.random.default_rng(20260811)
        self.slots = []
        for _ in range(4):
            ids = rng.integers(1, 2048, size=(2, 128), dtype=np.int64)
            mask = np.ones((2, 128), dtype=np.int64)
            slot = {
                "input_ids": backend_tensor(backend_name, self.backend, ids, "cuda"),
                "attention_mask": backend_tensor(backend_name, self.backend, mask, "cuda"),
            }
            if phase == "forward_backward":
                upstream = rng.standard_normal((2, 128, 256)).astype("float32")
                slot["upstream"] = backend_tensor(
                    backend_name, self.backend, upstream, "cuda"
                )
            self.slots.append(slot)

        self.kept = self._run(phase, self.slots[0])
        synchronize(backend_name, self.backend, self.device)
        self._verify(phase, self.kept)

    def _run(self, phase, slot):
        context = self.backend.no_grad() if phase == "forward" else contextlib.nullcontext()
        with context:
            output = self.model(
                input_ids=slot["input_ids"],
                attention_mask=slot["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            if phase == "forward":
                return (output,)
            loss = (output * slot["upstream"]).sum()
            gradients = self.backend.autograd.grad(loss, self.params_for_grad, retain_graph=False)
            return (output,) + tuple(gradients)

    def _verify(self, phase, values):
        output = as_numpy(self.backend_name, values[0])
        if output.shape != (2, 128, 256) or not np.isfinite(output).all():
            raise RuntimeError("Tiny Llama output is malformed or non-finite")
        if phase == "forward_backward":
            for index, gradient in enumerate(values[1:]):
                array = as_numpy(self.backend_name, gradient)
                if not np.isfinite(array).all() or not np.any(array):
                    raise RuntimeError("Tiny Llama gradient %d is zero or non-finite" % index)

    def time_model(self, backend_name, phase):
        self.kept = self._run(phase, self.slots[0])
        synchronize(backend_name, self.backend, self.device)

    def track_working_set_bytes(self, backend_name, phase):
        reset_memory_stats(backend_name, self.backend, self.device)
        self.kept = self._run(phase, self.slots[0])
        synchronize(backend_name, self.backend, self.device)
        return working_set_bytes(backend_name, self.backend, self.device)

    track_working_set_bytes.unit = "bytes"

    def teardown(self, backend_name, phase):
        backend = self.backend
        self.kept = ()
        self.slots = []
        self.params_for_grad = ()
        del self.model
        cleanup_backend(backend_name, backend)
