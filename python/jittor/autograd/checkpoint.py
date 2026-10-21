"""First-order activation rematerialization shared by native and Torch APIs.

The function must reproduce its tensor computation from its inputs and captured
module parameters. Container structure is fixed between forward and replay.
Registered buffers are replayed on copies; other Python side effects run twice.
Higher-order differentiation and non-reentrant early-stop replay are unsupported.
"""

from contextlib import nullcontext

import jittor as jt
from jittor._core.function import Function
from ._checkpoint_capture import Capture, flatten, unflatten, structure


def _rng_snapshot():
    jt.sync_all(True)
    cpu = jt.core.rng_state("cpu")
    if jt.flags.use_cuda and jt.compiler.has_acl:
        device = int(jt.current_device())
        return cpu, "acl", device, jt.core.rng_state("acl", device)
    if jt.flags.use_cuda:
        device = int(jt.current_device())
        backend = jt.compile_extern.get_library("curand", load=True)
        return cpu, "cuda", device, (int(backend.curand_generator_seed(device)),
                                     int(backend.curand_generator_offset(device)))
    return cpu, "cpu", 0, None


def _rng_restore(state):
    jt.sync_all(True)
    cpu, backend, device, payload = state
    jt.core.set_rng_state("cpu", cpu)
    if backend == "acl":
        jt.core.set_rng_state("acl", payload, device)
    elif backend == "cuda":
        jt.compile_extern.get_library("curand", load=True).curand_restore_state(
            device, payload[0] & ((1 << 64) - 1), payload[1])


class _Rematerialize(Function):
    def execute(self, *values):
        args, kwargs = unflatten(self.capture.spec, values)
        # This is a training forward, despite no_grad. Automatic inference
        # replay would retain a second graph and can replay mutable buffers.
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0):
            outputs = self.capture.function(*args, **kwargs)
            flat = []
            self.output_spec = flatten(outputs, flat, {})
            # Function.tape_together stops execute's outputs. Never lend it an
            # original input/parameter returned by an identity function.
            boundary = [output.detach() for output in flat]
            # Submit the unrecorded forward now: no Python reference to an
            # internal activation or its graph survives this boundary.
            for output in boundary:
                output.sync()
        self.output_meta = [(tuple(v.shape), str(v.dtype)) for v in flat]
        for output in boundary:
            if output.dtype.is_float():
                output.start_grad()
        return tuple(boundary)

    def grad(self, *grad_outputs):
        if ([(value.id, value._autograd_version()) for value in self.capture.values]
                != self.input_versions):
            raise RuntimeError("checkpoint input or captured parameter was modified before backward")
        leaves = []
        for value, enabled in zip(self.capture.values, self.needs_grad):
            # Parameter operations normally return plain frontend Tensors.
            # A functional parameter binding must retain its actual subtype,
            # without invoking a constructor or copying hooks/runtime caches.
            token = jt.core._set_tensor_frontend_type(type(value))
            try:
                leaf = value.detach()
            finally:
                jt.core._reset_tensor_frontend_type(token)
            leaf.start_grad() if enabled else leaf.stop_grad()
            leaves.append(leaf)
        current_rng = _rng_snapshot() if self.preserve_rng_state else None
        try:
            if self.preserve_rng_state:
                _rng_restore(self.rng_before)
            with jt.enable_grad(), jt.flag_scope(auto_graph_replay=0, **self.amp_flags), self.context_fn(), self.capture.replay(leaves) as fn:
                args, kwargs = unflatten(self.capture.spec, leaves)
                outputs = fn(*args, **kwargs)
                flat = []
                spec = flatten(outputs, flat, {})
                if structure(spec) != structure(self.output_spec):
                    raise RuntimeError("checkpoint replay changed the output container structure")
                if [(tuple(v.shape), str(v.dtype)) for v in flat] != self.output_meta:
                    raise RuntimeError("checkpoint replay changed output shapes or dtypes")
                terms = [(out * grad.detach()).sum() for out, grad in zip(flat, grad_outputs)
                         if grad is not None and out.dtype.is_float() and not out.is_stop_grad()]
                if not terms:
                    raise RuntimeError("checkpoint replay has no differentiable output")
                loss = terms[0]
                for term in terms[1:]:
                    loss = loss + term
                indices = [i for i, enabled in enumerate(self.needs_grad) if enabled]
                # Missing (unused) captures are ordinary zero derivatives, not
                # parameters to accumulate into as a side effect of this VJP.
                computed = jt.core.grad_optional(loss, [leaves[i] for i in indices], True)
                result = [None] * len(leaves)
                for index, gradient in zip(indices, computed):
                    if gradient is not None:
                        gradient.sync()
                        gradient._set_first_order_only()
                    result[index] = gradient
                return tuple(result)
        finally:
            if self.preserve_rng_state:
                _rng_restore(current_rng)


def checkpoint(function, *args, preserve_rng_state=True, context_fn=None, **kwargs):
    """Run an unrecorded forward and replay it during first-order backward.

    Differentiable tensors in nested positional/keyword arguments, bound or
    closure-captured modules and Python function tensor captures are taped once
    per unique tensor. ``context_fn`` optionally creates a fresh context manager
    for each backward replay (e.g. a frontend's autocast context).
    """
    if jt.flags.no_grad:
        return function(*args, **kwargs)
    capture = Capture(function, args, kwargs)
    needs_grad = [value.dtype.is_float() and bool(value.requires_grad) for value in capture.values]
    if not any(needs_grad):
        return function(*args, **kwargs)
    capture.prepare_buffers()
    op = _Rematerialize()
    op.capture = capture
    op.needs_grad = needs_grad
    op.input_versions = [(value.id, value._autograd_version()) for value in capture.values]
    op.context_fn = context_fn or nullcontext
    op.preserve_rng_state = bool(preserve_rng_state)
    op.rng_before = _rng_snapshot() if preserve_rng_state else None
    op.amp_flags = {key: int(getattr(jt.flags, key))
                    for key in ("amp_reg", "auto_mixed_precision_level")}
    ctx = op._new_call_context()
    flat = ctx._run_call(*capture.values)
    return unflatten(ctx.output_spec, flat)


__all__ = ["checkpoint"]
