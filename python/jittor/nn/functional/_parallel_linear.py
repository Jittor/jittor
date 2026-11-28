"""Group independent dense projections during no-grad step capture."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import dispatch_context


_PLANNER = ContextVar("jittor_parallel_linear_planner", default=None)
_MAX_PACK_BYTES = 192 << 20
_MAX_PROJECTION_BYTES = 2 << 20


@contextmanager
def parallel_linear_scope(planner):
    if planner is None:
        yield
        return
    import jittor as jt

    token = _PLANNER.set(planner)
    native = jt.core._set_parallel_linear_scope(True)
    try:
        yield
    finally:
        jt.core._set_parallel_linear_scope(native)
        _PLANNER.reset(token)


def try_parallel_linear(value, weight, bias, unpacked):
    planner = _PLANNER.get()
    if planner is None:
        return None
    import jittor as jt

    if not jt.flags.no_grad or jt.flags.amp_reg:
        return None
    if not all(isinstance(t, jt.Var) for t in (value, weight)):
        return None
    if bias is not None and not isinstance(bias, jt.Var):
        return None
    return planner(value, weight, bias, unpacked)


def observe_gated_projections(gate, value):
    planner = _PLANNER.get()
    if isinstance(planner, LinearDiscovery):
        planner.observe_gate(gate, value)


def _version(tensor):
    return tensor.var_ptr, tensor._saved_storage_version()


def _signature(weight, bias):
    return weight.var_ptr, None if bias is None else bias.var_ptr


class LinearDiscovery:
    """Record shapes and parameter holders during the existing warmup call."""

    def __init__(self, inputs):
        self.inputs = {value.var_ptr for value in inputs}
        self.projections = defaultdict(list)
        self.uses = Counter()
        self.large_outputs = {}
        self.gated_pairs = {}

    def __call__(self, value, weight, bias, unpacked=None):
        key = _signature(weight, bias)
        self.uses[key] += 1
        if value.ndim < 2 or weight.ndim != 2 or not value.numel():
            return
        shape = tuple(value.shape)
        cout, cin = weight.shape
        dtype = dtype_name(value.dtype)
        if dtype not in ("float32", "float16", "bfloat16") or weight.dtype != value.dtype:
            return
        # FP32 biased products also benefit from packing narrow projections.
        min_width = 64 if dtype == "float32" and bias is not None else 256
        if cin != shape[-1] or min(cin, cout) < min_width or value.numel() // cin < 32:
            return
        element_bytes = 4 if dtype == "float32" else 2
        large = value.numel() // cin * cout * element_bytes > _MAX_PROJECTION_BYTES
        if large and (unpacked is None or dtype == "float32"):
            return
        operands = (weight,) if bias is None else (weight, bias)
        if any(t.var_ptr in self.inputs or not t._storage_is_contiguous() for t in operands):
            return
        if bias is not None and (tuple(bias.shape) != (cout,) or bias.dtype != weight.dtype):
            return
        if not value._storage_is_contiguous():
            return
        context = dispatch_context(value, *operands)
        if context.backend != "cuda":
            return
        mark = (int(value.id), shape, dtype, bias is not None, context.device_id)
        call = (key, operands, tuple(_version(t) for t in operands))
        if large:
            output = unpacked(value, weight, bias)
            self.large_outputs[int(output.id)] = (mark, call, output._saved_storage_version())
            return output
        self.projections[mark].append(call)

    def observe_gate(self, gate, value):
        first = self.large_outputs.get(int(gate.id))
        second = self.large_outputs.get(int(value.id))
        if first is None or second is None:
            return
        mark, left, left_stamp = first
        other_mark, right, right_stamp = second
        if (mark == other_mark and left[0] != right[0]
                and gate._saved_storage_version() == left_stamp
                and value._saved_storage_version() == right_stamp):
            self.gated_pairs.setdefault(left[0], (mark, (left, right)))

    def groups(self):
        groups = []
        used = set()
        for mark, calls in (*self.gated_pairs.values(), *self.projections.items()):
            keys = {key for key, _, _ in calls}
            if (2 <= len(calls) <= 4 and not keys & used
                    and all(self.uses[key] == 1 for key in keys)):
                groups.append((mark, calls))
                used.update(keys)
        return groups


class PackedLinears:
    def __init__(self, groups, max_retained_bytes):
        import jittor as jt

        self.by_parameter = {}
        self.memo = {}
        self.used = {}
        budget = _MAX_PACK_BYTES if max_retained_bytes is None else min(
            _MAX_PACK_BYTES, max_retained_bytes)
        for (_, shape, dtype, has_bias, device), calls in groups:
            live = tuple(t for _, operands, _ in calls for t in operands)
            checks = tuple(v for _, _, versions in calls for v in versions)
            size = sum(t.numel() for t in live) * (4 if dtype == "float32" else 2)
            if size > budget or tuple(_version(t) for t in live) != checks:
                continue
            weights = jt.concat(tuple(operands[0] for _, operands, _ in calls), 0)
            bias = jt.concat(tuple(operands[1] for _, operands, _ in calls), 0) if has_bias else None
            tensors = [weights] if bias is None else [weights, bias]
            jt.sync(tensors)
            pack = (weights, bias, shape, dtype, device, live, checks)
            start = 0
            for key, operands, _ in calls:
                stop = start + operands[0].shape[0]
                self.by_parameter[key] = pack, start, stop
                start = stop
            budget -= size

    def __call__(self, value, weight, bias, unpacked=None):
        match = self.by_parameter.get(_signature(weight, bias))
        if match is None:
            return None
        pack, start, stop = match
        weights, bias, shape, dtype, device, live, checks = pack
        if (tuple(value.shape) != shape or dtype_name(value.dtype) != dtype
                or not value._storage_is_contiguous()):
            return None
        context = dispatch_context(value, weights)
        if context.backend != "cuda" or context.device_id != device:
            return None
        if tuple(_version(t) for t in live) != checks:
            return None
        key = (weights.var_ptr, value.var_ptr, tuple(value._saved_storage_version()))
        result = self.memo.get(key)
        if result is None:
            import jittor as jt

            result = self.memo[key] = jt.nn.linear(value, weights, bias)
            self.used[weights.var_ptr] = (live, checks)
        return result[..., start:stop].contiguous()

    def guarded_leaves(self):
        return [(tensor, pointer, version)
                for tensors, stamps in self.used.values()
                for tensor, (pointer, version) in zip(tensors, stamps)]
