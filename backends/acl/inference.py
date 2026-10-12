"""Explicit FP16 linear snapshots for ACL inference.

Load a normal checkpoint, move it to ACL, and select evaluation mode first.
Then, inside ``jt.no_grad()``, call ``freeze_linear_layers_(model)``. For the
independent Torch frontend pass ``linear_type=torch.nn.Linear`` explicitly.
Always use the returned module: a root Linear is replaced as well.

FrozenLinear stores packed buffers, with no mutable weight Parameter or
implicit cache. Later changes to the source Linear do not change its snapshot.
Its state_dict contains the packed representation and optional bias; an
ordinary Linear checkpoint must be loaded before conversion.
"""

import jittor as jt

_NATIVE_DTYPE = jt.core.Var.__dict__["dtype"]


def _require_acl_inference(*values):
    if not jt.flags.no_grad:
        raise RuntimeError("FrozenLinear requires no_grad; training is unsupported")
    tensors = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, jt.Var):
            raise TypeError("FrozenLinear expects Jittor tensors")
        # The frontend dtype property would build a Torch dtype only for us
        # to turn it back into a native name on every Linear invocation.
        if str(_NATIVE_DTYPE.__get__(value)) != "float16":
            raise TypeError("FrozenLinear supports float16 tensors only")
        tensors.append(value)
    # The inputs are already flat and type-checked. Ask the canonical native
    # placement owner directly instead of walking them and collecting dtypes
    # a second time through the general Python dispatcher.
    if (not jt.compiler.has_acl or not jt.flags.use_cuda
            or jt.core.dispatch_context(tensors)[0] not in ("acl", "acl_legacy")):
        raise RuntimeError("FrozenLinear requires the ACL device backend")


def _positive_features(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def pack_linear_weight(weight):
    """Snapshot an ACL FP16 ``[out_features, in_features]`` weight on device.

    The returned contiguous buffer has shape ``[ceil(K/16), ceil(N/16),16,16]``.
    K and N remain the logical input/output sizes; padding is zero. The copy
    and device packing finish before this function returns, so subsequent
    writes to ``weight`` cannot change this snapshot, even before first use.
    """
    _require_acl_inference(weight)
    if weight.ndim != 2:
        raise ValueError("linear weight must have two dimensions")
    out_features, in_features = map(int, weight.shape)
    _positive_features(in_features, "in_features")
    _positive_features(out_features, "out_features")
    n_blocks = (out_features + 15) // 16
    k_blocks = (in_features + 15) // 16
    snapshot = jt.Var.copy(weight).stop_grad()
    pad_n = n_blocks * 16 - out_features
    pad_k = k_blocks * 16 - in_features
    if pad_n or pad_k:
        snapshot = jt.nn.pad(snapshot, (0, pad_k, 0, pad_n), value=0)
    blocked = snapshot.reshape((n_blocks, 16, k_blocks, 16))
    # Use the native permutation spelling: the Torch Tensor.transpose method
    # accepts two axes, while this conversion permutes all four block axes.
    packed = jt.core.ops.transpose(blocked, (2, 0, 1, 3)).contiguous().stop_grad()
    packed.sync(True)
    return packed


def _validate_packed_linear(x, packed_weight, in_features, out_features, bias):
    """Preserve the public errors on the uncommon validation/bias path."""
    _require_acl_inference(x, packed_weight, bias)
    in_features = _positive_features(in_features, "in_features")
    out_features = _positive_features(out_features, "out_features")
    if x.ndim < 1 or int(x.shape[-1]) != in_features:
        raise ValueError("input last dimension must equal in_features")
    expected = ((in_features + 15) // 16, (out_features + 15) // 16, 16, 16)
    if tuple(packed_weight.shape) != expected or not packed_weight._storage_is_contiguous():
        raise ValueError("packed weight shape or storage does not match the feature dimensions")
    if bias is not None and tuple(bias.shape) != (out_features,):
        raise ValueError("bias shape must be (out_features,)")


def packed_linear(x, packed_weight, in_features, out_features, bias=None):
    """Run an explicit packed snapshot with its logical feature dimensions."""
    # The generated Var/int conversions accept more than this public API
    # (arrays and bools, for example). Keep its strict argument types, and
    # reject strided packed storage before the dense-input boundary copies it.
    if (not isinstance(x, jt.Var) or not isinstance(packed_weight, jt.Var)
            or type(in_features) is not int or type(out_features) is not int):
        _validate_packed_linear(x, packed_weight, in_features, out_features, bias)
    if bias is not None or not packed_weight._storage_is_contiguous():
        _validate_packed_linear(x, packed_weight, in_features, out_features, bias)
    try:
        output = jt.core.ops.mapped_packed_matmul(
            x, packed_weight, in_features, out_features)
    except RuntimeError:
        # Native construction checks dtype, placement and dimensions once.
        # Translate invalid arguments to the existing TypeError/ValueError;
        # a backend or runtime error with valid inputs must escape unchanged.
        _validate_packed_linear(x, packed_weight, in_features, out_features, bias)
        raise
    if bias is not None and output.numel():
        # Keep the separately rounded FP16 add, including the empty-output
        # shortcut, rather than changing arithmetic through a fused bias.
        output = output + bias
    return output


class FrozenLinear(jt.Module):
    """An explicit, independent packed snapshot of a Linear's FP16 weights.

    Evaluation and ``no_grad`` are required. Re-create the snapshot after
    loading new ordinary weights; there is no mutable ``weight`` Parameter.
    Registered packed buffers are the sole source of forward computations.
    """

    def __init__(self, weight, bias=None):
        super().__init__()
        _require_acl_inference(weight, bias)
        if weight.ndim != 2:
            raise ValueError("linear weight must have two dimensions")
        self.out_features, self.in_features = map(int, weight.shape)
        if bias is not None and tuple(bias.shape) != (self.out_features,):
            raise ValueError("bias shape must be (out_features,)")
        self.register_buffer("packed_weight", pack_linear_weight(weight))
        copied_bias = None if bias is None else jt.Var.copy(bias).stop_grad()
        if copied_bias is not None:
            copied_bias.sync(True)
        self.register_buffer("bias", copied_bias)
        self.is_train = False

    @classmethod
    def from_linear(cls, linear):
        """Snapshot an evaluated Linear after its checkpoint and device moves."""
        # Replacing the module would discard these per-instance behaviors.
        # Reject them before preparing the snapshot rather than silently
        # changing the model's forward or transplanting hook closures/handles.
        if any(linear.__dict__.get(name) for name in ("_forward_pre_hooks", "_forward_hooks")):
            raise RuntimeError("cannot freeze a Linear with forward hooks")
        if any(name in linear.__dict__ for name in ("forward", "execute")):
            raise RuntimeError("cannot freeze a Linear with instance forward/execute overrides")
        if linear.training:
            raise RuntimeError("call eval() before creating a FrozenLinear snapshot")
        return cls(linear.weight, linear.bias)

    def execute(self, x):
        if self.training:
            raise RuntimeError("FrozenLinear does not support training")
        return packed_linear(x, self.packed_weight, self.in_features, self.out_features, self.bias)

    def forward(self, x):
        return self.execute(x)

    def train(self, mode=True):
        if bool(mode):
            raise RuntimeError("FrozenLinear does not support training")
        self.is_train = False
        return self


def freeze_linear_layers_(module, *, linear_type=None):
    """Replace exact Linear modules with snapshots; return the converted root.

    ``linear_type`` defaults to native ``jt.nn.Linear``. Pass the independent
    frontend's ``torch.nn.Linear`` to convert a Torch model. Subclasses are
    preserved. Linear instances with forward hooks or per-instance forward/
    execute overrides are rejected. Registered container children and shared module aliases retain
    their names and identity relationships. All snapshots are prepared before
    any module is replaced, so a rejected layer leaves the original tree intact.
    """
    if not isinstance(module, jt.Module):
        raise TypeError("freeze_linear_layers_ expects a Jittor module")
    if linear_type is None:
        linear_type = jt.nn.Linear
    if not isinstance(linear_type, type) or not issubclass(linear_type, jt.Module):
        raise TypeError("linear_type must be a Jittor Linear module type")
    if not jt.flags.no_grad:
        raise RuntimeError("freeze_linear_layers_ requires no_grad")

    # This local identity map preserves shared modules during this one explicit
    # conversion. It never caches weights or survives the conversion call.
    visited, replacements, edges = set(), {}, []

    def prepare(current):
        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)
        if type(current) is linear_type:
            replacements[identity] = FrozenLinear.from_linear(current)
            return
        # named_children may remove duplicates in frontend implementations.
        # The registration mapping preserves every alias, including containers.
        for name, child in tuple(current._modules.items()):
            if not isinstance(child, jt.Module):
                continue
            edges.append((current, name, child))
            prepare(child)

    prepare(module)
    for parent, name, child in edges:
        replacement = replacements.get(id(child))
        if replacement is not None:
            parent.add_module(name, replacement)
    return replacements.get(id(module), module)


__all__ = ["FrozenLinear", "freeze_linear_layers_", "pack_linear_weight", "packed_linear"]
