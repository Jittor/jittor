"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
from jittor import nn

from ..context import registry_for
from ..grad import (
    _clip_grads_with_norm_device,
    _clip_grad_norm_device,
    _get_total_norm_device,
)
from ..nested import (
    _torch_make_parameter, _torch_register_leaf,
)
from ..types import (
    _device_is_cpu, _device_is_cuda, _dtype_to_str,
    _make_cpu_resident, _make_cuda_resident, device, dtype,
)

def _install_nn_extras(nn, registry=None):
    # Activation modules torch has that jittor.nn may lack.
    _modules = registry_for(jt, registry).module_map
    import jittor as _jt
    _install_init_aliases(registry)
    import types as _types_nn_private

    if not getattr(getattr(nn, "Parameter", None), "_torch_compat_type", False):
        _native_parameter = getattr(nn, "Parameter", None)
        class _ParameterMeta(type):
            def __instancecheck__(cls, obj):
                return (
                    isinstance(obj, _jt.Var)
                    and bool(getattr(obj, "_is_torch_parameter", False))
                )
            def __call__(cls, data=None, requires_grad=True):
                return _torch_make_parameter(data, requires_grad=requires_grad)
        class Parameter(metaclass=_ParameterMeta):
            pass
        Parameter._torch_compat_type = True
        class UninitializedTensorMixin:
            pass
        class UninitializedParameter:
            pass
        class UninitializedBuffer:
            pass
        nn.Parameter = Parameter
        param_mod = _types_nn_private.ModuleType("torch.nn.parameter")
        param_mod.Parameter = Parameter
        param_mod.UninitializedTensorMixin = UninitializedTensorMixin
        param_mod.UninitializedParameter = UninitializedParameter
        param_mod.UninitializedBuffer = UninitializedBuffer
        _modules["torch.nn.parameter"] = param_mod
        nn.parameter = param_mod

    param_mod = getattr(nn, "parameter", None)
    if param_mod is not None:
        _modules["torch.nn.parameter"] = param_mod

    modules_pkg = getattr(nn, "modules", None)
    if modules_pkg is None:
        try:
            from jittor.nn import modules as modules_pkg
        except Exception:
            modules_pkg = None
    if modules_pkg is None:
        modules_pkg = _types_nn_private.ModuleType("torch.nn.modules")
    _modules["torch.nn.modules"] = modules_pkg
    modules_pkg.__path__ = getattr(modules_pkg, "__path__", [])
    module_mod = _modules.get("torch.nn.modules.module")
    if module_mod is None:
        module_mod = _types_nn_private.ModuleType("torch.nn.modules.module")
        _modules["torch.nn.modules.module"] = module_mod
    module_mod.Module = nn.Module
    module_mod._EXTRA_STATE_KEY_SUFFIX = "_extra_state"
    module_mod._global_backward_hooks = getattr(module_mod, "_global_backward_hooks", {})
    module_mod._global_forward_hooks = getattr(module_mod, "_global_forward_hooks", {})
    module_mod._global_forward_pre_hooks = getattr(module_mod, "_global_forward_pre_hooks", {})
    module_mod._IncompatibleKeys = getattr(module_mod, "_IncompatibleKeys", type(
        "_IncompatibleKeys", (tuple,), {
            "__new__": lambda cls, missing_keys, unexpected_keys: tuple.__new__(cls, (missing_keys, unexpected_keys)),
            "missing_keys": property(lambda self: self[0]),
            "unexpected_keys": property(lambda self: self[1]),
        }))
    modules_pkg.Module = nn.Module
    modules_pkg.module = module_mod
    for _cn in dir(nn):
        if _cn and _cn[0].isupper() and not hasattr(modules_pkg, _cn):
            try:
                setattr(modules_pkg, _cn, getattr(nn, _cn))
            except Exception:
                pass
    container_mod = _modules.get("torch.nn.modules.container")
    if container_mod is None:
        container_mod = _types_nn_private.ModuleType("torch.nn.modules.container")
        _modules["torch.nn.modules.container"] = container_mod
    for _cn in ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict"):
        if hasattr(nn, _cn):
            setattr(container_mod, _cn, getattr(nn, _cn))
    modules_pkg.container = container_mod
    try:
        from jittor.misc import _single, _pair, _triple, _ntuple
    except Exception:
        _single = lambda x: x if isinstance(x, tuple) else (x,)
        _pair = lambda x: x if isinstance(x, tuple) else (x, x)
        _triple = lambda x: x if isinstance(x, tuple) else (x, x, x)
        def _ntuple(n):
            return lambda x: x if isinstance(x, tuple) else tuple([x] * n)

    def _mk_nn_submod(_name, **_attrs):
        _full = "torch.nn.modules." + _name
        _mod = _modules.get(_full)
        if _mod is None:
            _mod = _types_nn_private.ModuleType(_full)
            _modules[_full] = _mod
        for _ak, _av in _attrs.items():
            if _av is not None:
                setattr(_mod, _ak, _av)
        setattr(modules_pkg, _name, _mod)
        return _mod

    _mk_nn_submod("utils", _single=_single, _pair=_pair, _triple=_triple,
                  _ntuple=_ntuple, _quadruple=_ntuple(4))
    _mk_nn_submod("batchnorm",
                  _BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm1d=getattr(nn, "BatchNorm1d", getattr(nn, "BatchNorm", None)),
                  BatchNorm2d=getattr(nn, "BatchNorm2d", getattr(nn, "BatchNorm", None)),
                  BatchNorm3d=getattr(nn, "BatchNorm3d", getattr(nn, "BatchNorm", None)),
                  SyncBatchNorm=getattr(nn, "SyncBatchNorm", getattr(nn, "BatchNorm", None)))
    _mk_nn_submod("normalization",
                  GroupNorm=getattr(nn, "GroupNorm", None),
                  LayerNorm=getattr(nn, "LayerNorm", None),
                  LocalResponseNorm=getattr(nn, "LocalResponseNorm", None))
    _mk_nn_submod("activation",
                  ReLU=getattr(nn, "ReLU", None), SiLU=getattr(nn, "SiLU", None),
                  Sigmoid=getattr(nn, "Sigmoid", None), Tanh=getattr(nn, "Tanh", None),
                  GELU=getattr(nn, "GELU", None), LeakyReLU=getattr(nn, "LeakyReLU", None))
    parallel_mod = _modules.get("torch.nn.parallel")
    if parallel_mod is None:
        parallel_mod = _types_nn_private.ModuleType("torch.nn.parallel")
        _modules["torch.nn.parallel"] = parallel_mod

    class _DataParallel(nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

        def execute(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class _DistributedDataParallel(_DataParallel):
        require_backward_grad_sync = True

        def no_sync(self):
            import contextlib as _ctxlib
            return _ctxlib.nullcontext()

    parallel_mod.DataParallel = getattr(parallel_mod, "DataParallel", _DataParallel)
    parallel_mod.DistributedDataParallel = getattr(
        parallel_mod, "DistributedDataParallel", _DistributedDataParallel)
    parallel_distributed_mod = _modules.get("torch.nn.parallel.distributed")
    if parallel_distributed_mod is None:
        parallel_distributed_mod = _types_nn_private.ModuleType("torch.nn.parallel.distributed")
        _modules["torch.nn.parallel.distributed"] = parallel_distributed_mod
    parallel_distributed_mod.DistributedDataParallel = parallel_mod.DistributedDataParallel
    parallel_mod.distributed = parallel_distributed_mod
    nn.DataParallel = parallel_mod.DataParallel
    nn.parallel = parallel_mod

    # transformers 4.56.x imports torch.nn.attention.flex_attention from
    # masking_utils when torch is reported available. TRELLIS does not execute
    # PyTorch flex attention through this API, but the namespace must exist for
    # lazy model imports such as DINOv3ViTModel.
    # torch.nn is the physical jittor.nn package, so keep its real attention
    # capability module intact and register the torch path as an alias.
    from jittor.nn import attention as attn_mod
    _modules["torch.nn.attention"] = attn_mod
    flex_mod = _modules.get("torch.nn.attention.flex_attention")
    if flex_mod is None:
        flex_mod = _types_nn_private.ModuleType("torch.nn.attention.flex_attention")
        def _flex_attention(*args, **kwargs):
            raise NotImplementedError("flex_attention is not supported on jittor backend")
        flex_mod.flex_attention = _flex_attention
        flex_mod.create_block_mask = lambda *args, **kwargs: None
        flex_mod.BlockMask = type("BlockMask", (), {})
        flex_mod._DEFAULT_SPARSE_BLOCK_SIZE = 128
        flex_mod.and_masks = lambda *args, **kwargs: None
        flex_mod.or_masks = lambda *args, **kwargs: None
        flex_mod.AuxRequest = type("AuxRequest", (), {})
        flex_mod.AuxOutput = type("AuxOutput", (), {})
        flex_mod.flex_attention_hop = None
        flex_mod.noop_mask = lambda *args, **kwargs: None
        _modules["torch.nn.attention.flex_attention"] = flex_mod
    attn_mod.flex_attention = flex_mod
    nn.attention = attn_mod

    # nn.utils.clip_grad_norm_/clip_grad_value_ (also provided by torch_shim,
    # but needed for the bare `import jittor as torch` path too).
    if not hasattr(nn, "utils") or not hasattr(getattr(nn, "utils", None), "clip_grad_norm_"):
        import types as _t
        _u = getattr(nn, "utils", None) or _t.ModuleType("torch.nn.utils")
        def _grads_of(params):
            params = list(params)
            opt = getattr(_jt, "_current_optimizer", None)
            out = []
            for p in params:
                gg = None
                if opt is not None:
                    try: gg = opt.find_grad(p)
                    except Exception: gg = None
                if gg is None:
                    gg = getattr(p, "grad", None)
                if gg is not None:
                    out.append(gg)
            return out
        def clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                            error_if_nonfinite=False, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            grads = _grads_of(parameters)
            return _clip_grad_norm_device(
                grads, max_norm, norm_type, error_if_nonfinite)
        def clip_grad_value_(parameters, clip_value, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            for g in _grads_of(parameters):
                g.update(g.clamp(-clip_value, clip_value))
        _u.clip_grad_norm_ = clip_grad_norm_
        _u.clip_grad_value_ = clip_grad_value_

        # --- weight_norm / spectral_norm (reparametrizations) ---
        # torch reparametrizes a module's `weight` param into other params/buffers and
        # recomputes `weight` before each forward via a pre-forward hook. jittor has a
        # single-slot pre-forward hook, so route every reparametrization through one
        # dispatcher that calls each registered recompute fn (supports weight_norm +
        # spectral_norm on the same module, and preserves any pre-existing hook).
        from jittor.nn.utils.weight_norm import (
            _ensure_reparam_hook,
            _norm_except_dim,
            remove_weight_norm as _native_remove_weight_norm,
            weight_norm as _native_weight_norm,
        )

        def weight_norm(module, name="weight", dim=0):
            return _native_weight_norm(module, name, dim)

        def remove_weight_norm(module, name="weight"):
            return _native_remove_weight_norm(module, name)

        def _l2_normalize(x, eps):
            return x / (_jt.sqrt((x * x).sum()) + eps)

        def spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12, dim=None):
            w = getattr(module, name)
            sdim = 0 if dim is None else dim
            def _to_mat(W):
                if sdim == 0:
                    return W.reshape(W.shape[0], -1)
                perm = [sdim] + [d for d in range(W.ndim) if d != sdim]
                return W.permute(*perm).reshape(W.shape[sdim], -1)
            wmat = _to_mat(w)
            h, wd = int(wmat.shape[0]), int(wmat.shape[1])
            try: delattr(module, name)
            except Exception: pass
            setattr(module, name + "_orig", w.clone())
            module.register_buffer(name + "_u", _l2_normalize(_jt.randn(h), eps))
            module.register_buffer(name + "_v", _l2_normalize(_jt.randn(wd), eps))
            def _recompute(mod):
                W = getattr(mod, name + "_orig"); Wm = _to_mat(W)
                uu = getattr(mod, name + "_u"); vv = getattr(mod, name + "_v")
                for _ in range(max(1, n_power_iterations)):
                    vv = _l2_normalize(_jt.matmul(Wm.transpose(0, 1), uu), eps)
                    uu = _l2_normalize(_jt.matmul(Wm, vv), eps)
                getattr(mod, name + "_u").update(uu)     # warm-start next forward
                getattr(mod, name + "_v").update(vv)
                sigma = _jt.matmul(uu.reshape(1, -1), _jt.matmul(Wm, vv.reshape(-1, 1)))
                neww = W / sigma                          # sigma is 1-element -> scalar divide
                neww.persistent = False
                setattr(mod, name, neww)
            _ensure_reparam_hook(module).append(_recompute)
            _recompute(module)
            return module

        _u.weight_norm = weight_norm
        _u.remove_weight_norm = remove_weight_norm
        _u.spectral_norm = spectral_norm

        # --- nn.utils.rnn.pad_sequence ---
        import types as _trnn
        _rnn = _trnn.ModuleType("torch.nn.utils.rnn")
        def pad_sequence(sequences, batch_first=False, padding_value=0.0):
            seqs = list(sequences)
            max_len = max(int(s.shape[0]) for s in seqs)
            trailing = tuple(seqs[0].shape[1:])
            out = []
            for s in seqs:
                pl = max_len - int(s.shape[0])
                if pl > 0:
                    pad = _jt.ones((pl,) + trailing, dtype=s.dtype) * padding_value
                    s = _jt.concat([s, pad], dim=0)
                out.append(s)
            stacked = _jt.stack(out, dim=0)               # (B, T, *)
            return stacked if batch_first else stacked.transpose(0, 1)
        _rnn.pad_sequence = pad_sequence
        _u.rnn = _rnn
        _modules.setdefault("torch.nn.utils.rnn", _rnn)

        nn.utils = _u

    # Newer PyTorch exposes torch.nn.utils.parametrize and
    # torch.nn.utils.parametrizations. transformers 4.56 probes
    # nn.utils.parametrizations.weight_norm while remapping checkpoint keys.
    import types as _types_nn_utils
    _u = getattr(nn, "utils", None) or _types_nn_utils.ModuleType("torch.nn.utils")
    _u.__path__ = getattr(_u, "__path__", [])
    _modules.setdefault("torch.nn.utils", _u)
    nn.utils = _u
    _clip_grad = _types_nn_utils.ModuleType("torch.nn.utils.clip_grad")

    def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False,
                        foreach=None):
        del foreach
        if isinstance(tensors, _jt.Var):
            tensors = [tensors]
        return _get_total_norm_device(
            list(tensors), norm_type, error_if_nonfinite)

    def _clip_grads_with_norm_(parameters, max_norm, total_norm,
                               foreach=None):
        del foreach
        if isinstance(parameters, _jt.Var):
            parameters = [parameters]
        params = list(parameters)
        opt = getattr(_jt, "_current_optimizer", None)
        grads = []
        for parameter in params:
            grad = None
            if opt is not None:
                try:
                    grad = opt.find_grad(parameter)
                except Exception:
                    grad = None
            if grad is None:
                grad = getattr(parameter, "grad", None)
            if grad is not None:
                grads.append(grad)
        _clip_grads_with_norm_device(grads, max_norm, total_norm)

    _clip_grad._get_total_norm = _get_total_norm
    _clip_grad._clip_grads_with_norm_ = _clip_grads_with_norm_
    _clip_grad.clip_grad_norm_ = getattr(_u, "clip_grad_norm_", None)
    _clip_grad.clip_grad_value_ = getattr(_u, "clip_grad_value_", None)
    _modules["torch.nn.utils.clip_grad"] = _clip_grad
    _u.clip_grad = _clip_grad
    if not hasattr(_u, "parametrize"):
        _parametrize = _types_nn_utils.ModuleType("torch.nn.utils.parametrize")
        _parametrize.register_parametrization = lambda module, *a, **k: module
        _parametrize.remove_parametrizations = lambda module, *a, **k: module
        _parametrize.is_parametrized = lambda module, *a, **k: False
        _parametrize.type_before_parametrizations = lambda module: type(module)
        _u.parametrize = _parametrize
        _modules["torch.nn.utils.parametrize"] = _parametrize
    else:
        _modules.setdefault("torch.nn.utils.parametrize", _u.parametrize)
    if not hasattr(_u, "parametrizations"):
        _parametrizations = _types_nn_utils.ModuleType("torch.nn.utils.parametrizations")
        _parametrizations.weight_norm = getattr(_u, "weight_norm", lambda module, name="weight", dim=0: module)
        _parametrizations.spectral_norm = getattr(_u, "spectral_norm", lambda module, *a, **k: module)
        _parametrizations.orthogonal = lambda module, *a, **k: module
        _u.parametrizations = _parametrizations
        _modules["torch.nn.utils.parametrizations"] = _parametrizations
    else:
        _modules.setdefault("torch.nn.utils.parametrizations", _u.parametrizations)

    # torchmetrics imports torch.nn.utils.rnn at module import time. Install the
    # module unconditionally because some bootstrap paths create nn.utils before
    # the clip/weight-norm block above runs.
    import builtins as _builtins_rnn
    import collections as _collections_rnn
    _rnn = getattr(_u, "rnn", None)
    if _rnn is None:
        _rnn = _types_nn_utils.ModuleType("torch.nn.utils.rnn")

    def _rnn_lengths_to_list(lengths):
        if isinstance(lengths, _jt.Var):
            lengths = lengths.numpy()
        if hasattr(lengths, "tolist"):
            lengths = lengths.tolist()
        if isinstance(lengths, (_builtins_rnn.int, _builtins_rnn.float)):
            lengths = [lengths]
        return [_builtins_rnn.int(x) for x in list(lengths)]

    def _rnn_index_tensor(x, order, batch_first):
        order = _rnn_lengths_to_list(order)
        if not order:
            return x
        if batch_first:
            return _jt.stack([x[i] for i in order], dim=0)
        return _jt.stack([x[:, i] for i in order], dim=1)

    def _rnn_pad_sequence(sequences, batch_first=False, padding_value=0.0):
        seqs = list(sequences)
        if not seqs:
            raise ValueError("pad_sequence expects a non-empty sequence list")
        max_len = _builtins_rnn.max(_builtins_rnn.int(s.shape[0]) for s in seqs)
        trailing = tuple(seqs[0].shape[1:])
        padded = []
        for s in seqs:
            pad_len = max_len - _builtins_rnn.int(s.shape[0])
            if pad_len > 0:
                pad = _jt.ones((pad_len,) + trailing, dtype=s.dtype) * padding_value
                s = _jt.concat([s, pad], dim=0)
            padded.append(s)
        out = _jt.stack(padded, dim=0)
        return out if batch_first else out.transpose(0, 1)

    _PackedSequenceBase = _collections_rnn.namedtuple(
        "PackedSequence", ("data", "batch_sizes", "sorted_indices", "unsorted_indices"))

    class PackedSequence(_PackedSequenceBase):
        __slots__ = ()

        def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
            return _PackedSequenceBase.__new__(cls, data, batch_sizes, sorted_indices, unsorted_indices)

        def to(self, *args, **kwargs):
            data = self.data.to(*args, **kwargs) if hasattr(self.data, "to") else self.data
            return type(self)(data, self.batch_sizes, self.sorted_indices, self.unsorted_indices)

        cuda = to
        cpu = to

    def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
        lengths_list = _rnn_lengths_to_list(lengths)
        if not enforce_sorted:
            order = sorted(range(len(lengths_list)), key=lambda i: lengths_list[i], reverse=True)
            unsorted = [0] * len(order)
            for sorted_pos, original_pos in enumerate(order):
                unsorted[original_pos] = sorted_pos
            input = _rnn_index_tensor(input, order, batch_first)
            lengths_list = [lengths_list[i] for i in order]
            sorted_indices = _jt.array(order).int64()
            unsorted_indices = _jt.array(unsorted).int64()
        else:
            sorted_indices = None
            unsorted_indices = None

        max_len = lengths_list[0] if lengths_list else 0
        pieces = []
        batch_sizes = []
        for t in range(max_len):
            active = _builtins_rnn.sum(1 for n in lengths_list if n > t)
            if active <= 0:
                break
            batch_sizes.append(active)
            if batch_first:
                pieces.append(input[:active, t])
            else:
                pieces.append(input[t, :active])
        if pieces:
            data = _jt.concat(pieces, dim=0)
        else:
            trailing = tuple(input.shape[2:])
            data = _jt.ones((0,) + trailing, dtype=input.dtype)
        return PackedSequence(data, _jt.array(batch_sizes).int64(), sorted_indices, unsorted_indices)

    def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
        if not isinstance(sequence, PackedSequence):
            return sequence, None
        batch_sizes = _rnn_lengths_to_list(sequence.batch_sizes)
        max_len = len(batch_sizes)
        batch_size = _builtins_rnn.max(batch_sizes) if batch_sizes else 0
        data = sequence.data
        trailing = tuple(data.shape[1:])
        steps = []
        offset = 0
        for active in batch_sizes:
            step = data[offset:offset + active]
            offset += active
            if active < batch_size:
                pad = _jt.ones((batch_size - active,) + trailing, dtype=data.dtype) * padding_value
                step = _jt.concat([step, pad], dim=0)
            steps.append(step)
        if steps:
            out = _jt.stack(steps, dim=0)
        else:
            out = _jt.ones((0, batch_size) + trailing, dtype=data.dtype) * padding_value
        if total_length is not None:
            total_length = _builtins_rnn.int(total_length)
            if total_length < max_len:
                raise ValueError("total_length must be at least the packed sequence length")
            if total_length > max_len:
                pad = _jt.ones((total_length - max_len, batch_size) + trailing, dtype=data.dtype) * padding_value
                out = _jt.concat([out, pad], dim=0)
        lengths_list = [_builtins_rnn.sum(1 for n in batch_sizes if n > i) for i in range(batch_size)]
        if sequence.unsorted_indices is not None:
            out = _rnn_index_tensor(out, sequence.unsorted_indices, batch_first=False)
            order = _rnn_lengths_to_list(sequence.unsorted_indices)
            lengths_list = [lengths_list[i] for i in order]
        if batch_first:
            out = out.transpose(0, 1)
        return out, _jt.array(lengths_list).int64()

    _rnn.pad_sequence = _rnn_pad_sequence
    _rnn.pack_padded_sequence = pack_padded_sequence
    _rnn.pad_packed_sequence = pad_packed_sequence
    _rnn.PackedSequence = PackedSequence
    _u.rnn = _rnn
    _modules["torch.nn.utils.rnn"] = _rnn

    if "torch.nn.utils.prune" not in _modules:
        _prune = _types_nn_utils.ModuleType("torch.nn.utils.prune")

        def _unsupported_prune(*args, **kwargs):
            raise NotImplementedError("torch.nn.utils.prune is not supported on jittor backend")

        class BasePruningMethod:
            PRUNING_TYPE = "unstructured"

            def __call__(self, module, inputs):
                return inputs

            @classmethod
            def apply(cls, module, name, *args, **kwargs):
                return _unsupported_prune(module, name, *args, **kwargs)

            def remove(self, module):
                return module

        class L1Unstructured(BasePruningMethod):
            PRUNING_TYPE = "unstructured"

        class RandomUnstructured(BasePruningMethod):
            PRUNING_TYPE = "unstructured"

        class LnStructured(BasePruningMethod):
            PRUNING_TYPE = "structured"

        class RandomStructured(BasePruningMethod):
            PRUNING_TYPE = "structured"

        _prune.BasePruningMethod = BasePruningMethod
        _prune.L1Unstructured = L1Unstructured
        _prune.RandomUnstructured = RandomUnstructured
        _prune.LnStructured = LnStructured
        _prune.RandomStructured = RandomStructured
        _prune.l1_unstructured = _unsupported_prune
        _prune.random_unstructured = _unsupported_prune
        _prune.ln_structured = _unsupported_prune
        _prune.random_structured = _unsupported_prune
        _prune.global_unstructured = _unsupported_prune
        _prune.remove = _unsupported_prune
        _prune.is_pruned = lambda module: False
        _modules["torch.nn.utils.prune"] = _prune
    _u.prune = _modules["torch.nn.utils.prune"]
    if "torch.nn.utils._named_member_accessor" not in _modules:
        _named_accessor = _types_nn_utils.ModuleType("torch.nn.utils._named_member_accessor")
        def _resolve_parent(module, name):
            parts = str(name).split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            return parent, parts[-1]
        def swap_tensor(module, name, tensor):
            parent, leaf = _resolve_parent(module, name)
            old = getattr(parent, leaf, None)
            setattr(parent, leaf, tensor)
            return old
        _named_accessor.swap_tensor = swap_tensor
        _modules["torch.nn.utils._named_member_accessor"] = _named_accessor
    _u._named_member_accessor = _modules["torch.nn.utils._named_member_accessor"]

    if not hasattr(nn, "Hardswish"):
        class Hardswish(nn.Module):
            def execute(self, x):
                return x * _jt.clamp(x + 3, 0, 6) / 6
        nn.Hardswish = Hardswish
    if not hasattr(nn, "CELU"):           # timm uses nn.CELU
        class CELU(nn.Module):
            def __init__(self, alpha=1.0, inplace=False):
                super().__init__(); self.alpha = alpha
            def execute(self, x):
                a = self.alpha
                return _jt.maximum(x, 0.0) + _jt.minimum(0.0, a * (_jt.exp(x / a) - 1))
        nn.CELU = CELU
    # A batch of standard torch activations jittor.nn may lack (timm's act-layer
    # registry references all of them at import). All are pure elementwise.
    if not hasattr(nn, "SELU"):
        _SELU_S, _SELU_A = 1.0507009873554805, 1.6732632423543772
        class SELU(nn.Module):
            def __init__(self, inplace=False): super().__init__()
            def execute(self, x):
                return _SELU_S * (_jt.maximum(x, 0.0) + _jt.minimum(0.0, _SELU_A * (_jt.exp(x) - 1)))
        nn.SELU = SELU
    if not hasattr(nn, "Softsign"):
        class Softsign(nn.Module):
            def execute(self, x): return x / (1 + _jt.abs(x))
        nn.Softsign = Softsign
    if not hasattr(nn, "Tanhshrink"):
        class Tanhshrink(nn.Module):
            def execute(self, x): return x - _jt.tanh(x)
        nn.Tanhshrink = Tanhshrink
    if not hasattr(nn, "Softplus"):
        class Softplus(nn.Module):
            def __init__(self, beta=1, threshold=20): super().__init__(); self.beta=beta; self.threshold=threshold
            def execute(self, x):
                bx = self.beta * x
                return _jt.ternary(bx > self.threshold, x, _jt.log1p(_jt.exp(bx)) / self.beta)
        nn.Softplus = Softplus
    if not hasattr(nn, "Hardshrink"):
        class Hardshrink(nn.Module):
            def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
            def execute(self, x): return x * ((x > self.lambd) | (x < -self.lambd)).float()
        nn.Hardshrink = Hardshrink
    if not hasattr(nn, "Softshrink"):
        class Softshrink(nn.Module):
            def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
            def execute(self, x):
                l = self.lambd
                return _jt.maximum(x - l, 0.0) - _jt.maximum(-x - l, 0.0)
        nn.Softshrink = Softshrink
    if not hasattr(nn, "Hardsigmoid"):
        class Hardsigmoid(nn.Module):
            def execute(self, x):
                return _jt.clamp(x + 3, 0, 6) / 6
        nn.Hardsigmoid = Hardsigmoid
    if not hasattr(nn, "Identity"):
        class Identity(nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def execute(self, x): return x
        nn.Identity = Identity
    # ModuleList/Sequential/ModuleDict usually exist; alias ParameterList if not
    if not hasattr(nn, "ParameterList"):
        nn.ParameterList = nn.ModuleList if hasattr(nn, "ModuleList") else list
    # ModuleDict (peft LoRA layers need it); jittor lacks it.
    if not hasattr(nn, "ModuleDict"):
        class ModuleDict(nn.Module):
            def __init__(self, modules=None):
                super().__init__()
                self._keys = []
                if modules:
                    self.update(modules)
            def update(self, modules):
                items = modules.items() if hasattr(modules, "items") else modules
                for k, v in items:
                    self[k] = v
            def __setitem__(self, key, module):
                setattr(self, key, module)
                if key not in self._keys:
                    self._keys.append(key)
            def __getitem__(self, key):
                return getattr(self, key)
            def __delitem__(self, key):
                delattr(self, key)
                if key in self._keys:
                    self._keys.remove(key)
            def __contains__(self, key):
                return key in self._keys
            def __len__(self):
                return len(self._keys)
            def __iter__(self):
                return iter(self._keys)
            def keys(self):
                return list(self._keys)
            def values(self):
                return [getattr(self, k) for k in self._keys]
            def items(self):
                return [(k, getattr(self, k)) for k in self._keys]
            def pop(self, key):
                v = getattr(self, key); self.__delitem__(key); return v
        nn.ModuleDict = ModuleDict

    # Layer classes torch has that jittor.nn may lack -- needed at least for
    # isinstance() checks in model init. Provide a distinct empty subclass so
    # isinstance discrimination still works.
    if not hasattr(nn, "ConvTranspose1d"):
        class ConvTranspose1d(nn.Module):
            # Real 1D transpose-conv (SABL's side_aware_feature_extractor uses it),
            # implemented via conv_transpose2d with a unit height dim so it also
            # rides the cuDNN memory-efficient path.
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, output_padding=0, groups=1, bias=True,
                         dilation=1, **k):
                super().__init__()
                import jittor as _jt2, math as _math
                g1 = lambda v: v[0] if isinstance(v, (tuple, list)) else v
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = g1(kernel_size)
                self.stride = g1(stride)
                self.padding = g1(padding)
                self.output_padding = g1(output_padding)
                self.dilation = g1(dilation)
                self.groups = groups
                self.weight = _jt2.init.invariant_uniform(
                    [in_channels, out_channels // groups, self.kernel_size], dtype="float")
                if bias:
                    fan = (in_channels // groups) * self.kernel_size
                    bound = 1.0 / _math.sqrt(fan) if fan > 0 else 0.0
                    self.bias = _jt2.init.uniform([out_channels], "float", -bound, bound)
                else:
                    self.bias = None
            def execute(self, x):
                import jittor as _jt2
                x2 = x.unsqueeze(2)                       # (N,Cin,1,L)
                w2 = self.weight.unsqueeze(2)             # (Cin,Cout/g,1,K)
                y = _jt2.nn.conv_transpose2d(
                    x2, w2, None, (1, self.stride), (0, self.padding),
                    (0, self.output_padding), self.groups, (1, self.dilation))
                y = y.squeeze(2)                          # (N,Cout,Lout)
                if self.bias is not None:
                    y = y + self.bias.broadcast(y.shape, [0, 2])
                return y
        nn.ConvTranspose1d = ConvTranspose1d
    if not hasattr(nn, "RMSNorm"):
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **k):
                super().__init__()
                import jittor as _jt2
                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.weight = _jt2.ones(normalized_shape) if elementwise_affine else None
            def execute(self, x):
                import jittor as _jt2
                v = (x.float32() ** 2).mean(-1, keepdims=True)
                x = x * _jt2.rsqrt(v + self.eps)
                return x * self.weight if self.weight is not None else x
        nn.RMSNorm = RMSNorm
    # Transformer modules build on the canonical jittor.nn.MultiheadAttention.
    import jittor as _jtm

    def _act_fn(activation):
        if callable(activation):
            return activation
        return {"relu": nn.relu, "gelu": nn.gelu}.get(activation, nn.relu)

    if not hasattr(nn, "TransformerEncoderLayer"):
        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", layer_norm_eps=1e-5, batch_first=False,
                         norm_first=False, bias=True, device=None, dtype=None):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=batch_first, bias=bias)
                self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
                self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
                self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm_first = norm_first
                self.activation = _act_fn(activation)

            def _sa(self, x, attn_mask, kpm, is_causal):
                return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=kpm,
                                      need_weights=False, is_causal=is_causal)[0]

            def _ff(self, x):
                return self.linear2(self.activation(self.linear1(x)))

            def execute(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                x = src
                if self.norm_first:
                    x = x + self._sa(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
                    x = x + self._ff(self.norm2(x))
                else:
                    x = self.norm1(x + self._sa(x, src_mask, src_key_padding_mask, is_causal))
                    x = self.norm2(x + self._ff(x))
                return x
        nn.TransformerEncoderLayer = TransformerEncoderLayer

    if not hasattr(nn, "TransformerEncoder"):
        import copy as _copy
        class TransformerEncoder(nn.Module):
            def __init__(self, encoder_layer, num_layers, norm=None, **kw):
                super().__init__()
                self.layers = nn.ModuleList([_copy.deepcopy(encoder_layer) for _ in range(num_layers)])
                self.num_layers = num_layers
                self.norm = norm

            def execute(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
                out = src
                for layer in self.layers:
                    out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                if self.norm is not None:
                    out = self.norm(out)
                return out
        nn.TransformerEncoder = TransformerEncoder

    if not hasattr(nn, "TransformerDecoderLayer"):
        class TransformerDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", layer_norm_eps=1e-5, batch_first=False,
                         norm_first=False, bias=True, device=None, dtype=None):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=batch_first, bias=bias)
                self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                            batch_first=batch_first, bias=bias)
                self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
                self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
                self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm_first = norm_first
                self.activation = _act_fn(activation)

            def _sa(self, x, m, kpm, ic):
                return self.self_attn(x, x, x, attn_mask=m, key_padding_mask=kpm,
                                      need_weights=False, is_causal=ic)[0]

            def _ca(self, x, mem, m, kpm, ic):
                return self.multihead_attn(x, mem, mem, attn_mask=m, key_padding_mask=kpm,
                                           need_weights=False, is_causal=ic)[0]

            def _ff(self, x):
                return self.linear2(self.activation(self.linear1(x)))

            def execute(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        tgt_is_causal=False, memory_is_causal=False):
                x = tgt
                if self.norm_first:
                    x = x + self._sa(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                    x = x + self._ca(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                    x = x + self._ff(self.norm3(x))
                else:
                    x = self.norm1(x + self._sa(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
                    x = self.norm2(x + self._ca(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
                    x = self.norm3(x + self._ff(x))
                return x
        nn.TransformerDecoderLayer = TransformerDecoderLayer

    if not hasattr(nn, "TransformerDecoder"):
        import copy as _copy2
        class TransformerDecoder(nn.Module):
            def __init__(self, decoder_layer, num_layers, norm=None, **kw):
                super().__init__()
                self.layers = nn.ModuleList([_copy2.deepcopy(decoder_layer) for _ in range(num_layers)])
                self.num_layers = num_layers
                self.norm = norm

            def execute(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        tgt_is_causal=None, memory_is_causal=False):
                out = tgt
                for layer in self.layers:
                    out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                memory_is_causal=memory_is_causal)
                if self.norm is not None:
                    out = self.norm(out)
                return out
        nn.TransformerDecoder = TransformerDecoder

    if not hasattr(nn, "Transformer"):
        class Transformer(nn.Module):
            def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                         num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                         activation="relu", custom_encoder=None, custom_decoder=None,
                         layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                         bias=True, device=None, dtype=None):
                super().__init__()
                self.batch_first = batch_first
                self.d_model = d_model
                self.nhead = nhead
                if custom_encoder is not None:
                    self.encoder = custom_encoder
                else:
                    el = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first, bias)
                    self.encoder = nn.TransformerEncoder(el, num_encoder_layers,
                                                         nn.LayerNorm(d_model, eps=layer_norm_eps))
                if custom_decoder is not None:
                    self.decoder = custom_decoder
                else:
                    dl = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first, bias)
                    self.decoder = nn.TransformerDecoder(dl, num_decoder_layers,
                                                         nn.LayerNorm(d_model, eps=layer_norm_eps))

            def execute(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                        src_key_padding_mask=None, tgt_key_padding_mask=None,
                        memory_key_padding_mask=None, src_is_causal=None,
                        tgt_is_causal=None, memory_is_causal=False):
                memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    memory_is_causal=memory_is_causal)

            @staticmethod
            def generate_square_subsequent_mask(sz, device=None, dtype=None):
                # upper-triangular -inf mask (additive), like torch
                m = _jtm.triu(_jtm.ones((sz, sz)), 1) * (-1e30)
                return m
        nn.Transformer = Transformer

    # ---- nn.SyncBatchNorm (single-device: behaves exactly like BatchNorm) ----
    # mmdetection's rtmdet calls `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`;
    # with no real process group there is nothing to synchronise, so the convert
    # entry returns the model unchanged (BN already pools over the whole batch here).
    if not hasattr(nn, "SyncBatchNorm"):
        class SyncBatchNorm(nn.BatchNorm):
            def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                         track_running_stats=True, process_group=None, **kw):
                super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
            @classmethod
            def convert_sync_batchnorm(cls, module, process_group=None):
                return module
        nn.SyncBatchNorm = SyncBatchNorm

    # ---- nn.functional extras used by mmdetection (not auto-copied from jittor.nn) ----
    F = nn.functional
    if not hasattr(F, "_Reduction"):
        # torch's private reduction-string -> enum helper; mmdet's loss utils do
        # `F._Reduction.get_enum(reduction)` then branch 0/1/2 = none/mean/sum.
        class _Reduction:
            @staticmethod
            def get_enum(reduction):
                return {"none": 0, "mean": 1, "elementwise_mean": 1,
                        "sum": 2}.get(reduction, 1)
            @staticmethod
            def legacy_get_string(size_average, reduce, emit_warning=True):
                sa = True if size_average is None else size_average
                rd = True if reduce is None else reduce
                if not rd:
                    return "none"
                return "mean" if sa else "sum"
        F._Reduction = _Reduction
    if not hasattr(F, "adaptive_max_pool2d"):
        def _adaptive_max_pool2d(input, output_size, return_indices=False):
            out = nn.AdaptiveMaxPool2d(output_size)(input)
            return (out, None) if return_indices else out
        F.adaptive_max_pool2d = _adaptive_max_pool2d
    if not getattr(F, "_torch_linear_wrapped", False):
        # torch's F.linear accepts a 1-D weight (matrix-vector product), e.g. GFL's
        # Integral does F.linear(x[N,K], project[K]) -> [N]; jittor's linear asserts 2-D.
        _jt_linear = F.linear
        def _linear(input, weight, bias=None):
            if hasattr(weight, "ndim") and weight.ndim == 1:
                out = (input * weight).sum(-1)
                return out if bias is None else out + bias
            return _jt_linear(input, weight, bias)
        F.linear = _linear
        F._torch_linear_wrapped = True
    if not hasattr(F, "relu_"):
        F.relu_ = lambda input: nn.relu(input)   # in-place relu (graph-equivalent)
    if not hasattr(F, "upsample_bilinear"):
        # deprecated torch alias == interpolate(mode='bilinear', align_corners=True)
        def _upsample_bilinear(input, size=None, scale_factor=None):
            return F.interpolate(input, size=size, scale_factor=scale_factor,
                                 mode="bilinear", align_corners=True)
        F.upsample_bilinear = _upsample_bilinear
    if not hasattr(F, "upsample"):
        def _upsample(input, size=None, scale_factor=None, mode="nearest",
                      align_corners=None):
            return F.interpolate(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners)
        F.upsample = _upsample

    # torch's nn.Conv2d exposes .transposed / .output_padding (torchvision &
    # mmcv's ConvModule read them to introspect the layer); jittor's Conv lacks
    # them. Add torch-compatible class attributes.
    for _cn in ("Conv", "Conv1d", "Conv3d"):
        _c = getattr(nn, _cn, None)
        if _c is not None:
            if not hasattr(_c, "transposed"):
                _c.transposed = False
            if not hasattr(_c, "output_padding"):
                _c.output_padding = (0, 0)
    for _cn in ("ConvTranspose", "ConvTranspose1d", "ConvTranspose3d"):
        _c = getattr(nn, _cn, None)
        if _c is not None:
            _c.transposed = True
            if not hasattr(_c, "output_padding"):
                _c.output_padding = (0, 0)

    # torch's nn.Dropout/Dropout2d/Dropout3d take an `inplace` kwarg that jittor's
    # don't (DETR-family configs pass dropout=dict(..., inplace=...)). Make the
    # constructors tolerate (and ignore) it.
    for _dn in ("Dropout", "Dropout2d", "Dropout3d"):
        _dc = getattr(nn, _dn, None)
        if _dc is not None and not getattr(_dc, "_torch_inplace_patched", False):
            def _mk_drop_init(orig):
                def _init(self, p=0.5, inplace=False, *a, **k):
                    orig(self, p, *a, **k)
                return _init
            _dc.__init__ = _mk_drop_init(_dc.__init__)
            _dc._torch_inplace_patched = True

    # jittor names several activation/layer classes lowercase or snake_case
    # (nn.ReLU.__name__ == 'relu'); torch code and mmcv's registry key layers by
    # type(layer).__name__, so normalize them to the torch class names.
    _TORCH_CLASS_NAMES = [
        "ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU", "ELU", "CELU", "SELU",
        "GELU", "SiLU", "Mish", "Sigmoid", "Tanh", "Softmax", "Softplus",
        "Hardswish", "Hardsigmoid", "Hardtanh", "GLU", "Identity",
    ]
    for _nm in _TORCH_CLASS_NAMES:
        _cls = getattr(nn, _nm, None)
        if isinstance(_cls, type) and getattr(_cls, "__name__", None) != _nm:
            try:
                _cls.__name__ = _nm
                _cls.__qualname__ = _nm
            except Exception:
                pass

    _install_module_methods(nn, registry)


def _install_module_methods(nn, registry=None):
    """Add torch-compatible methods to jittor's nn.Module."""
    _modules = registry_for(jt, registry).module_map
    M = nn.Module

    # torch models define forward(); jittor calls execute(). Make the base
    # execute() delegate to a subclass-defined forward() so torch models run.
    _orig_execute = M.execute
    def _execute(self, *args, **kwargs):
        fwd = getattr(type(self), "forward", None)
        if fwd is not None and fwd is not _forward_alias:
            return fwd(self, *args, **kwargs)
        return _orig_execute(self, *args, **kwargs)
    def _forward_alias(self, *args, **kwargs):
        # if a subclass only defines execute(), forward() routes to it
        return self.execute(*args, **kwargs)
    M.execute = _execute
    if not hasattr(M, "forward"):
        M.forward = _forward_alias

    # Central dispatch fix: an HF module may SUBCLASS a jittor builtin (e.g.
    # transformers OPTLearnedPositionalEmbedding(nn.Embedding)) and override
    # forward() with a different signature. The builtin (Embedding) defines its
    # own execute(), which MRO-shadows the patched base Module.execute above, so
    # `module(...)` -> __call__ -> self.execute(...) lands on the builtin's
    # execute() and never sees the subclass forward() -> TypeError.
    #
    # Decide per class whether the OWN forward() override should take precedence
    # over the inherited builtin execute(): it should iff a real (non-alias)
    # forward() is defined at an MRO position at least as derived as the nearest
    # execute(). Conservative: classes that only define execute() (every native
    # jittor module + jittor-native subclasses of builtins) keep calling
    # execute() exactly as before; only a genuine, more-derived forward()
    # override flips dispatch.
    _dispatch_cache = {}
    def _prefer_forward(cls):
        cached = _dispatch_cache.get(cls)
        if cached is not None:
            return cached
        fwd_idx = exec_idx = None
        for i, c in enumerate(cls.__mro__):
            d = c.__dict__
            if fwd_idx is None and "forward" in d and d["forward"] is not _forward_alias:
                fwd_idx = i
            if exec_idx is None and "execute" in d and d["execute"] is not _execute:
                exec_idx = i
        # forward() wins only if it exists and is no less derived than execute()
        result = fwd_idx is not None and (exec_idx is None or fwd_idx <= exec_idx)
        _dispatch_cache[cls] = result
        return result

    _orig_call = M.__call__
    def _standard_rms_norm(self, args, kwargs):
        cls_name = type(self).__name__
        if (
            not cls_name.endswith("RMSNorm")
            or cls_name.endswith("RMSNormGated")
            or len(args) != 1
            or kwargs
            or "variance_epsilon" not in self.__dict__
        ):
            return None
        value = args[0]
        weight = getattr(self, "weight", None)
        if not isinstance(value, jt.Var) or not isinstance(weight, jt.Var):
            return None
        epsilon = self.__dict__["variance_epsilon"]
        fast = jt.nn._rms_norm_training_cuda(value, weight, epsilon)
        if fast is None:
            fast = jt.nn._rms_norm_cuda(value, weight, epsilon)
        return fast

    _pipeline_state = {"threshold": 0, "mark": 0}

    def set_execution_pipelining(pending_ops):
        ''' Launch the pending graph at module boundaries once it holds this many
        ops, instead of waiting for the next sync. 0 (the default) disables it.

        Returns the previous setting. See ``_maybe_pipeline`` for the trade.
        '''
        previous = _pipeline_state["threshold"]
        _pipeline_state["threshold"] = max(0, int(pending_ops))
        _pipeline_state["mark"] = jt.core.number_of_lived_ops()
        return previous

    def get_execution_pipelining():
        ''' The current pending-op threshold; 0 when pipelining is off. '''
        return _pipeline_state["threshold"]

    def _maybe_pipeline(result):
        # A lazy graph reaches the device only at the next sync, so the GPU sits
        # idle for the whole of the Python-side construction: measured on
        # ViT-base, one contiguous ~6ms stall per step, immediately before the
        # first kernel of the forward. Launching the graph built so far at a
        # module boundary -- ``jt.sync`` does not wait for the device -- lets the
        # GPU start while Python keeps building.
        #
        # The cost is fusion: ops either side of a flush cannot fuse, which also
        # regroups floating-point accumulation, so results move by a rounding
        # step. Off unless asked for, and the threshold counts pending ops rather
        # than module calls, so leaf modules do not each trigger one.
        threshold = _pipeline_state["threshold"]
        if threshold <= 0:
            return result
        # Count ops added since the last flush, not ops alive: the live count
        # includes everything the graph still holds, so once it crossed the
        # threshold every later module call would flush and no two ops would ever
        # fuse.
        lived = jt.core.number_of_lived_ops()
        if lived - _pipeline_state["mark"] < threshold:
            if lived < _pipeline_state["mark"]:
                _pipeline_state["mark"] = lived
            return result
        target = result[0] if isinstance(result, (tuple, list)) and result else result
        if isinstance(target, jt.Var):
            jt.sync([target])
            _pipeline_state["mark"] = jt.core.number_of_lived_ops()
        return result

    def _call(self, *args, **kwargs):
        def dispatch(*call_args, **call_kwargs):
            # torch lets a module override forward per-INSTANCE (`self.forward =
            # fn`, used by vLLM's samplers / CustomOp dispatch). Honor it before
            # class-level dispatch.
            inst_fwd = self.__dict__.get("forward", None)
            if inst_fwd is not None and callable(inst_fwd):
                return inst_fwd(*call_args, **call_kwargs)
            rms_norm = _standard_rms_norm(self, call_args, call_kwargs)
            if rms_norm is not None:
                return rms_norm
            if _prefer_forward(type(self)):
                return type(self).forward(self, *call_args, **call_kwargs)
            return _orig_call(self, *call_args, **call_kwargs)

        state = getattr(self, "_fsdp_state", None)
        if state is not None and getattr(state, "true_fsdp_initialized", False):
            from jittor.compat.fsdp2 import shard as _fsdp2_shard
            return _maybe_pipeline(_fsdp2_shard._execute_with_true_fsdp(
                self, dispatch, *args, **kwargs))
        return _maybe_pipeline(dispatch(*args, **kwargs))
    M.__call__ = _call
    M.set_execution_pipelining = staticmethod(set_execution_pipelining)
    M.get_execution_pipelining = staticmethod(get_execution_pipelining)

    # torch's named_parameters/named_buffers/named_modules accept extra kwargs
    # (remove_duplicate, prefix, recurse) and return iterators; jittor's take
    # only `recurse` and return lists, with named_buffers defaulting recurse=
    # False (torch defaults True). Wrap to be torch-compatible.
    _orig_named_parameters = M.named_parameters
    _orig_named_buffers = M.named_buffers
    _orig_named_modules = M.named_modules

    def _named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        reg = getattr(jt, "_torch_leaf_params", None)
        if reg is None:
            reg = jt._torch_leaf_params = {}
        seen = set()
        for name, v in _orig_named_parameters(self, recurse=recurse):
            if remove_duplicate and id(v) in seen:
                continue
            seen.add(id(v))
            # register trainable params as autograd leaves so the no-optimizer
            # loss.backward() path can populate their .grad (see parameters()).
            try:
                if isinstance(v, jt.Var) and v.requires_grad:
                    reg[id(v)] = v
            except Exception:
                pass
            yield (prefix + ("." if prefix else "") + name, v)
    M.named_parameters = _named_parameters

    def _named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for name, v in _orig_named_buffers(self, recurse=recurse):
            if remove_duplicate and id(v) in seen:
                continue
            seen.add(id(v))
            yield (prefix + ("." if prefix else "") + name, v)
    M.named_buffers = _named_buffers

    def _named_modules(self, memo=None, prefix="", remove_duplicate=True):
        for item in _orig_named_modules(self):
            # jittor yields (name, module) pairs
            if isinstance(item, tuple) and len(item) == 2:
                name, mod = item
            else:
                name, mod = "", item
            yield (prefix + ("." if prefix and name else "") + name, mod)
    M.named_modules = _named_modules

    # torch's Module.load_state_dict(state, strict=True, assign=False) accepts a
    # `strict` kwarg and returns a namedtuple(missing_keys, unexpected_keys);
    # jittor's takes only `params` and returns None. Wrap for torch callers
    # (peft's set_peft_model_state_dict passes strict=False).
    _orig_load_state_dict = M.load_state_dict
    import collections as _collections2
    _IncompatibleKeys = _collections2.namedtuple("IncompatibleKeys",
                                                  ["missing_keys", "unexpected_keys"])
    def _find_state_target(root, key):
        obj = root
        for part in str(key).split("."):
            if isinstance(obj, nn.Sequential):
                if part in obj.layers:
                    obj = obj.layers[part]
                elif str(part).isdigit() and int(part) in obj.layers:
                    obj = obj.layers[int(part)]
                else:
                    return None
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def _state_source_to_var(value):
        if isinstance(value, jt.Var):
            return value
        try:
            return jt.array(value.cpu().detach().numpy())
        except Exception:
            return jt.array(value)

    def _preserve_target_dtypes_for_load(root, state_dict):
        # torch.load_state_dict(assign=False), the default used by TRELLIS.2,
        # copies checkpoint values into existing parameters/buffers and keeps
        # the destination dtype.  Jittor's native load replaces through update(),
        # so a bf16 target can be widened to fp32 when the loader had to widen a
        # BF16 safetensor through numpy. Cast the source to the live target dtype
        # before delegating to native load_state_dict.
        if not isinstance(state_dict, dict):
            return state_dict
        converted = None
        for key, value in state_dict.items():
            target = _find_state_target(root, key)
            if not isinstance(target, jt.Var):
                continue
            src = _state_source_to_var(value)
            if not isinstance(src, jt.Var):
                continue
            if src.shape != target.shape:
                continue
            target_dtype = str(target.dtype)
            if str(src.dtype) == target_dtype:
                continue
            if converted is None:
                converted = dict(state_dict)
            converted[key] = src.cast(target_dtype)
        return state_dict if converted is None else converted

    def _load_state_dict(self, state_dict, strict=True, assign=False):
        # preserve trainable flags: jittor assign can flip stop_grad
        trainable = set()
        try:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    trainable.add(n)
        except Exception:
            pass
        load_state = state_dict if assign else _preserve_target_dtypes_for_load(self, state_dict)
        _orig_load_state_dict(self, load_state)
        try:
            for n, p in self.named_parameters():
                if n in trainable and p.is_stop_grad():
                    p.start_grad()
        except Exception:
            pass
        return _IncompatibleKeys([], [])
    M.load_state_dict = _load_state_dict

    # torch's Module.parameters() returns an *iterator*; peft does
    # `next(model.parameters())`. jittor returns a list (needed for len()/
    # indexing by optimizers). Return a list subclass that is also an iterator
    # so both `next(...)` and `len(...)`/indexing work.
    class _ParamList(list):
        def __iter__(self):
            return list.__iter__(self)
        def __next__(self):
            it = getattr(self, "_it", None)
            if it is None:
                it = self._it = list.__iter__(self)
            return next(it)
    # Register every trainable parameter as an autograd "leaf" the first time a
    # module's params are enumerated. torch code reads param.grad only after
    # enumerating params (optimizer construction, gradient clipping, gradcheck,
    # manual inspection all call parameters()/named_parameters() first), so this
    # is the reliable hook that lets the optimizer-free loss.backward() path
    # (below) populate param.grad. jittor params are trainable-by-default and
    # almost never pass through the requires_grad setter, which is why the prior
    # registry stayed empty (bert: 0/39 grads exposed). Enumeration is also the
    # *leak-safe* hook: only declared parameters are captured -- never transient
    # forward activations, which a Module.__setattr__ hook would wrongly retain
    # and leak one Var per step. Idempotent (id-keyed); skips frozen params so
    # their .grad stays None like torch.
    def _register_leaf_params(params):
        try:
            reg = getattr(jt, "_torch_leaf_params", None)
            if reg is None:
                reg = jt._torch_leaf_params = {}
            for p in params:
                if isinstance(p, jt.Var) and p.requires_grad:
                    reg[id(p)] = p
        except Exception:
            pass
    _orig_parameters = M.parameters
    def _parameters(self, recurse=True):
        pl = _orig_parameters(self, recurse=recurse)
        _register_leaf_params(pl)
        return _ParamList(pl)
    M.parameters = _parameters

    # torch's Module.train(mode=True)/eval() take a mode arg; jittor's train()
    # takes none. Wrap to accept it and toggle jittor's real training flag.
    #
    # The flag that controls layers like Dropout/BatchNorm is `is_train` -- an
    # instance attribute read by jittor.nn.Dropout.execute. `is_training` is a
    # *method* and `training` a *property*, so they must NEVER be assigned a
    # bool (the old code did `m.is_training = False`, which both shadowed the
    # method and failed to flip the flag the layers actually read). We set
    # `is_train` recursively on every submodule. We deliberately do NOT touch
    # parameter stop-grad state (torch's .eval() leaves requires_grad alone),
    # so this is purely a mode flip with no gradient side effects.
    def _set_is_train(self, mode):
        mode = bool(mode)
        try:
            mods = self.modules() if hasattr(self, "modules") else [self]
        except Exception:
            mods = [self]
        for m in mods:
            try:
                m.is_train = mode
            except Exception:
                pass
    def _train(self, mode=True):
        # torch semantics: set this module's flag, then recurse into DIRECT
        # children calling each child's .train(mode) so overridden train()
        # methods run (e.g. e2cnn's R2Conv.train() rebuilds/discards its cached
        # filter; a flat is_train sweep silently bypasses it, leaving stale or
        # empty filters and zero output). For ordinary modules this is
        # behaviourally identical to the old flat sweep.
        mode = bool(mode)
        try:
            self.is_train = mode
        except Exception:
            pass
        kids = None
        try:
            kids = list(self.children())
        except Exception:
            kids = None
        if kids is None:
            _set_is_train(self, mode)          # fallback: flat sweep
            return self
        for child in kids:
            tr = getattr(child, "train", None)
            if callable(tr):
                try:
                    tr(mode)
                    continue
                except Exception:
                    pass
            _set_is_train(child, mode)
        return self
    M.train = _train
    def _eval(self):
        return _train(self, False)
    M.eval = _eval

    _MODULE_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")

    def _module_cast_var_if_needed(v, ds, copy=False):
        if copy or str(v.dtype) != ds:
            return v.cast(ds)
        return v

    def _module_cast_float_dtype(self, ds):
        if ds is not None and ds in _MODULE_FLOAT_DTYPES:
            for p in self.parameters():
                if p.dtype.is_float() if hasattr(p.dtype, "is_float") else ("float" in str(p.dtype)):
                    new_p = _module_cast_var_if_needed(p, ds)
                    if new_p is not p:
                        p.assign(new_p)
        return self

    def _module_replace_vars(self, convert):
        converted = {}
        try:
            modules = list(self.modules()) if hasattr(self, "modules") else [self]
        except Exception:
            modules = [self]
        if not modules or modules[0] is not self:
            modules.insert(0, self)
        seen = set()
        for module in modules:
            mid = id(module)
            if mid in seen:
                continue
            seen.add(mid)
            attrs = []
            if hasattr(module, "params"):
                attrs.append(("params", getattr(module, "params")))
            attrs.append(("__dict__", getattr(module, "__dict__", {})))
            for _container_name, container in attrs:
                if not isinstance(container, dict):
                    continue
                buffer_names = getattr(module, "_buffer_names", set())
                for name, value in list(container.items()):
                    if isinstance(value, jt.Var):
                        if _container_name == "__dict__":
                            is_public_param = not (isinstance(name, str) and name.startswith("_"))
                            is_buffer = getattr(value, "is_buffer", False) or name in buffer_names
                            if not (is_public_param or is_buffer):
                                continue
                        vid = id(value)
                        if vid in converted:
                            new_value = converted[vid]
                        else:
                            new_value = convert(value)
                            converted[vid] = new_value
                            if new_value is not value:
                                try:
                                    new_value.persistent = getattr(value, "persistent")
                                except Exception:
                                    pass
                                try:
                                    new_value.is_buffer = getattr(value, "is_buffer")
                                except Exception:
                                    pass
                                try:
                                    new_value._torch_grad = getattr(value, "_torch_grad")
                                except Exception:
                                    pass
                                try:
                                    if value.is_stop_grad() and not new_value.is_stop_grad():
                                        new_value.stop_grad()
                                    elif (not value.is_stop_grad()) and new_value.is_stop_grad():
                                        new_value.start_grad()
                                        _torch_register_leaf(new_value)
                                except Exception:
                                    pass
                                try:
                                    reg = getattr(jt, "_torch_leaf_params", None)
                                    if isinstance(reg, dict) and vid in reg:
                                        reg.pop(vid, None)
                                        if not new_value.is_stop_grad():
                                            reg[id(new_value)] = new_value
                                except Exception:
                                    pass
                        if new_value is value:
                            continue
                        container[name] = new_value
        return self

    def _module_to(self, *args, **kwargs):
        # torch Module.to(device/dtype/...) casts floating tensors and migrates
        # tensor residency when an explicit cpu/cuda device is requested.
        ds = None
        dev = kwargs.get("device")
        copy = bool(kwargs.get("copy", False))
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dtype):
                ds = a.name
            elif isinstance(a, device):
                dev = a
            elif isinstance(a, jt.Var):
                ds = str(a.dtype)
                dev = a.device
            elif isinstance(a, str):
                bare = a.replace("torch.", "")
                if bare in dtype._registry:
                    ds = bare
                elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                    dev = bare
        if _device_is_cuda(dev):
            jt.flags.use_cuda = 1

        def convert(v):
            out = v
            if ds is not None and ds in _MODULE_FLOAT_DTYPES:
                is_float = v.dtype.is_float() if hasattr(v.dtype, "is_float") else ("float" in str(v.dtype))
                if is_float:
                    out = _module_cast_var_if_needed(out, ds, copy=copy)
            if _device_is_cpu(dev):
                out = _make_cpu_resident(out, inplace=(out is v))
            elif _device_is_cuda(dev):
                out = _make_cuda_resident(out, force=True, inplace=(out is v))
            return out

        if dev is not None or ds is not None:
            return _module_replace_vars(self, convert)
        return self
    M.to = _module_to

    def _module_to_empty(self, *, device, recurse=True):
        # Jittor does not expose meta storage. Models are already materialized,
        # so preserve their values while honoring the requested residency.
        return _module_to(self, device=device)
    M.to_empty = _module_to_empty

    def _module_cuda(self, dev=None):
        return _module_to(self, device("cuda", dev) if isinstance(dev, int) else "cuda")
    def _module_npu(self, dev=None):
        return _module_to(self, device("npu", dev) if isinstance(dev, int) else "npu")
    M.cuda = _module_cuda
    M.npu = _module_npu
    M.cpu = lambda self: _module_to(self, "cpu")
    if not hasattr(M, "float"):
        M.float = lambda self: _module_cast_float_dtype(self, "float32")
    if not hasattr(M, "double"):
        M.double = lambda self: _module_cast_float_dtype(self, "float64")
    if not hasattr(M, "half"):
        M.half = lambda self: _module_cast_float_dtype(self, "float16")
    # torch's zero_grad() clears each param's .grad so the next backward starts
    # fresh; the optimizer-free backward path below accumulates with += (matching
    # torch), so a real reset is required. The prior no-op left grads silently
    # accumulating across steps. Clear the torch-exposed grad and, when an
    # optimizer is bridged, delegate to its zero_grad as well.
    def _zero_grad(self, set_to_none=True):
        try:
            for p in self.parameters():
                if getattr(p, "_torch_grad", None) is not None:
                    object.__setattr__(p, "_torch_grad", None)
        except Exception:
            pass
        opt = getattr(jt, "_current_optimizer", None)
        if opt is not None:
            try:
                opt.zero_grad()
            except Exception:
                pass
        return None
    M.zero_grad = _zero_grad
    if not hasattr(M, "buffers"):
        M.buffers = lambda self, recurse=True: [v for _, v in self.named_buffers()]
    if not hasattr(M, "get_submodule"):
        def _get_submodule(self, target):
            mod = self
            for part in target.split("."):
                if part:
                    mod = getattr(mod, part)
            return mod
        M.get_submodule = _get_submodule
    if not hasattr(M, "get_parameter"):
        def _get_parameter(self, target):
            mod = self
            parts = target.split(".")
            for part in parts[:-1]:
                if part:
                    mod = getattr(mod, part)
            leaf = parts[-1]
            if not hasattr(mod, leaf):
                raise AttributeError(f"`{target}` is not a parameter")
            v = getattr(mod, leaf)
            import jittor as _jtp
            # a parameter is a trainable Var directly attached to the module
            if isinstance(v, _jtp.Var) and not v.is_stop_grad():
                return v
            if isinstance(v, _jtp.Var):
                # could still be a (frozen) parameter; distinguish from buffers
                names = {n for n, _ in self.named_parameters()}
                if target in names:
                    return v
            raise AttributeError(f"`{target}` is not a parameter")
        M.get_parameter = _get_parameter
    if not hasattr(M, "get_buffer"):
        def _get_buffer(self, target):
            mod = self
            parts = target.split(".")
            for part in parts[:-1]:
                if part:
                    mod = getattr(mod, part)
            leaf = parts[-1]
            if not hasattr(mod, leaf):
                raise AttributeError(f"`{target}` is not a buffer")
            v = getattr(mod, leaf)
            import jittor as _jtp
            names = {n for n, _ in self.named_buffers()}
            if isinstance(v, _jtp.Var) and target in names:
                return v
            raise AttributeError(f"`{target}` is not a buffer")
        M.get_buffer = _get_buffer
    if not hasattr(M, "register_parameter"):
        def _register_parameter(self, name, param):
            setattr(self, name, param)
        M.register_parameter = _register_parameter
    if not hasattr(M, "type"):
        M.type = lambda self, dst_type=None: self

    # torch's nn.Module keeps `_non_persistent_buffers_set`, a set of the
    # *immediate* (non-recursive) buffer attribute names that were registered
    # with persistent=False. transformers' from_pretrained reads it via
    # `named_non_persistent_buffers()` (parent._non_persistent_buffers_set).
    # jittor instead tags each buffer Var with `.persistent`; derive the set
    # from that. It's a property so it stays correct as buffers are (de)added.
    if not isinstance(M.__dict__.get("_non_persistent_buffers_set"), property):
        import jittor as _jtb
        def _nonpersist_set(self):
            out = set()
            for k, v in self.__dict__.items():
                if (isinstance(k, str) and not k.startswith("_")
                        and isinstance(v, _jtb.Var)
                        and getattr(v, "is_buffer", False)
                        and not getattr(v, "persistent", True)):
                    out.add(k)
            return out
        M._non_persistent_buffers_set = property(_nonpersist_set)


def _install_init_aliases(registry=None):
    _modules = registry_for(jt, registry).module_map
    import jittor.init as _init
    import jittor as _jt2
    # torch-style in-place initializers, tolerant of torch kwargs (e.g.
    # `generator=`, which jittor ignores). Each writes into `tensor` in place.
    def _assign(tensor, value):
        # Preserve the tensor's grad-tracking: jittor's .assign() adopts the
        # source var's stop_grad flag, and our `value` (jt.normal/zeros/...) is
        # stop_grad, which would silently freeze the parameter. Re-enable grad
        # unless the param was explicitly stop-grad before.
        was_trainable = not tensor.is_stop_grad()
        parent = getattr(tensor, "_torch_index_parent", None)
        parent_slices = getattr(tensor, "_torch_index_slices", None)
        tensor.assign(value)
        # Basic indexing materializes a Var in Jittor, while torch initializers
        # mutate a view's underlying storage. Write the initialized value back
        # through the recorded parent chain (TorchQuantum initializes U3 columns
        # via init.constant_(parameter[:, k], value)).
        if isinstance(parent, _jt2.Var):
            parent[parent_slices] = value
        if was_trainable:
            tensor.start_grad()
        return tensor

    # in-place inits are sometimes called on a NON-Var constant: jittor represents a
    # disabled affine term (e.g. LayerNorm(bias=False) -> self.bias = 0.0) as a Python
    # scalar, and a model's _init_weights may still call init.zeros_(module.bias) on it.
    # Such a constant isn't a learnable parameter, so initializing it is a no-op.
    def _not_var(t):
        return not isinstance(t, _jt2.Var)
    def normal_(tensor, mean=0.0, std=1.0, generator=None):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.normal(float(mean), float(std), tensor.shape).cast(str(tensor.dtype)))
    def uniform_(tensor, a=0.0, b=1.0, generator=None, *, low=None, high=None):
        if low is not None:
            if a != 0.0 and a != low:
                raise TypeError("uniform_ received conflicting values for a and low")
            a = low
        if high is not None:
            if b != 1.0 and b != high:
                raise TypeError("uniform_ received conflicting values for b and high")
            b = high
        if _not_var(tensor): return tensor
        return _assign(tensor, (_jt2.rand(tensor.shape) * (b - a) + a).cast(str(tensor.dtype)))
    def zeros_(tensor):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.zeros(tensor.shape, tensor.dtype))
    def ones_(tensor):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype))
    def constant_(tensor, val=0.0, *, value=None):
        if value is not None:
            if val != 0.0 and val != value:
                raise TypeError("constant_ received conflicting values for val and value")
            val = value
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype) * val)
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
        if _not_var(tensor): return tensor
        import numpy as _np
        # simple clamp of a normal sample (no scipy dependency)
        x = _np.random.normal(mean, std, tensor.shape).astype("float32")
        x = _np.clip(x, mean + a * std, mean + b * std)
        return _assign(tensor, _jt2.array(x).cast(str(tensor.dtype)))
    # override with the tolerant versions (also covers jittor's own names)
    for name, fn in [("normal_", normal_), ("uniform_", uniform_),
                     ("zeros_", zeros_), ("ones_", ones_), ("constant_", constant_),
                     ("trunc_normal_", trunc_normal_)]:
        setattr(_init, name, fn)
    # jittor's native kaiming/xavier/gauss initializers do `var.assign(src)` without
    # re-enabling grad. Under transformers' @torch.no_grad() weight init, `src` is
    # stop_grad, so .assign() silently FREEZES the parameter -- Conv2d/Linear inited
    # with kaiming (resnet/regnet/...) end up stop_grad and get zero weight grads
    # (forward stays exact, so it's invisible until you train/check gradients). Wrap
    # them with the same grad-preserving guard used by _assign() above: a no-op for
    # already-frozen params, so it can't regress anything.
    def _grad_preserving(fn):
        def wrapped(tensor, *a, **k):
            was_trainable = hasattr(tensor, "is_stop_grad") and not tensor.is_stop_grad()
            r = fn(tensor, *a, **k)
            if was_trainable and hasattr(tensor, "start_grad"):
                tensor.start_grad()
            return r
        return wrapped
    for _nm in ("kaiming_normal_", "kaiming_uniform_", "gauss_",
                "xavier_uniform_", "xavier_gauss_", "xavier_normal_",
                "relu_invariant_gauss_", "invariant_uniform_"):
        if hasattr(_init, _nm):
            setattr(_init, _nm, _grad_preserving(getattr(_init, _nm)))
            if hasattr(_jt2.Var, _nm):   # keep the Var-bound method spelling in sync
                setattr(_jt2.Var, _nm, getattr(_init, _nm))
    # keep jittor's good xavier/kaiming; add torch-name aliases for the rest
    aliases = {"xavier_normal_": "xavier_gauss_"}
    for tname, jname in aliases.items():
        if not hasattr(_init, tname) and hasattr(_init, jname):
            setattr(_init, tname, getattr(_init, jname))
    # initializers torch has that jittor lacks -- best-effort implementations
    if not hasattr(_init, "_calculate_fan_in_and_fan_out"):
        def _fan(t):
            sh = t.shape
            if len(sh) < 2:
                return sh[0], sh[0]
            num_input_fmaps, num_output_fmaps = sh[1], sh[0]
            rf = 1
            for s in sh[2:]:
                rf *= s
            return num_input_fmaps * rf, num_output_fmaps * rf
        _init._calculate_fan_in_and_fan_out = _fan
    if not hasattr(_init, "_calculate_correct_fan"):
        def _calculate_correct_fan(tensor, mode):
            mode = str(mode).lower()
            if mode not in ("fan_in", "fan_out"):
                raise ValueError("Mode %s not supported, please use fan_in or fan_out" % mode)
            fan_in, fan_out = _init._calculate_fan_in_and_fan_out(tensor)
            return fan_in if mode == "fan_in" else fan_out
        _init._calculate_correct_fan = _calculate_correct_fan
    if not hasattr(_init, "dirac_"):
        _init.dirac_ = lambda t, *a, **k: t   # best-effort no-op
    if not hasattr(_init, "orthogonal_"):
        def _orth(t, gain=1.0):
            import numpy as _np
            sh = t.shape
            flat = (sh[0], int(t.numel() // sh[0])) if len(sh) > 1 else (sh[0], 1)
            a = _np.random.randn(*flat)
            q, r = _np.linalg.qr(a)
            q = q * _np.sign(_np.diag(r))
            if flat[0] < flat[1]:
                q = q.T
            t.assign(jt.array((gain * q).reshape(sh).astype("float32")))
            return t
        _init.orthogonal_ = _orth
    if not hasattr(_init, "sparse_"):
        _init.sparse_ = lambda t, *a, **k: t  # best-effort no-op

    # torch.nn.init also exposes deprecated non-underscore spellings of the
    # in-place initializers (normal/xavier_normal/kaiming_uniform/kaiming_normal),
    # which forward to the `_` versions. Some older model code calls them. Add
    # each alias only when its `_` target exists and the alias is still missing.
    for tname in ("normal", "xavier_normal", "kaiming_uniform", "kaiming_normal"):
        target = tname + "_"
        if not hasattr(_init, tname) and hasattr(_init, target):
            setattr(_init, tname, getattr(_init, target))

    # Keep transformers/diffusers no_init_weights() from replacing jittor's
    # construction-time init functions with no-op stubs. torch.nn is jittor.nn
    # on the bare `import jittor as torch` path, and jittor.nn.Conv/Linear call
    # the same module-global init functions to allocate weights.
    import types as _types_init
    class _GuardedInit(_types_init.ModuleType):
        _protected = set()
        def __setattr__(self, key, value):
            if key in self._protected:
                name = getattr(value, "__name__", "")
                if (not callable(value)) or name in ("_skip_init", "skip_init", "<lambda>"):
                    return
            object.__setattr__(self, key, value)
    guarded = _GuardedInit("torch.nn.init")
    protected = set()
    for key in dir(_init):
        if not key.startswith("__"):
            try:
                value = getattr(_init, key)
                object.__setattr__(guarded, key, value)
                if callable(value):
                    protected.add(key)
            except Exception:
                pass
    object.__setattr__(guarded, "_protected", protected)
    nn.init = guarded
    _modules["torch.nn.init"] = guarded


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    if not hasattr(nn, "functional"):
        import types as _types
        F = _types.ModuleType("jittor.nn.functional")
        for fname in dir(nn):
            fobj = getattr(nn, fname)
            if callable(fobj) and not isinstance(fobj, type):
                setattr(F, fname, fobj)
    else:
        F = nn.functional
    if hasattr(nn, "relu"): F.relu = nn.relu
    if hasattr(nn, "gelu"): F.gelu = nn.gelu
    if hasattr(nn, "softmax"):
        # torch: F.softmax(input, dim=None, _stacklevel=3, dtype=None).
        # When dtype is given, input is cast to it before softmax (used by
        # transformers' eager attention: F.softmax(scores, dim=-1, dtype=fp32)).
        _jt_softmax = nn.softmax
        def _softmax(input, dim=-1, _stacklevel=3, dtype=None):
            if dtype is not None:
                input = input.cast(_dtype_to_str(dtype))
            return _jt_softmax(input, dim=dim)
        F.softmax = _softmax
    if hasattr(nn, "linear"): F.linear = nn.linear
    if hasattr(nn, "interpolate"):
        # torch.nn.functional.interpolate defaults to mode='nearest', but
        # jittor.nn.interpolate defaults to 'bilinear'. Code that omits the
        # mode (e.g. YOLOV3Neck: F.interpolate(x, scale_factor=2)) silently
        # gets the wrong upsampling. Wrap so the torch-shim functional matches
        # torch's default and accepts torch's arg name / extra kwargs. Only
        # this shim copy is affected, not jittor's native nn.interpolate.
        _jt_interpolate = nn.interpolate
        def _interpolate(input=None, size=None, scale_factor=None,
                         mode="nearest", align_corners=None,
                         recompute_scale_factor=None, antialias=False,
                         **_kw):
            if input is None:
                input = _kw.pop("X")
            ac = False if align_corners is None else align_corners
            return _jt_interpolate(input, size=size,
                                   scale_factor=scale_factor, mode=mode,
                                   align_corners=ac)
        F.interpolate = _interpolate
    if hasattr(nn, "cross_entropy_loss"):
        _jt_ce = nn.cross_entropy_loss
        # torch.nn.functional.cross_entropy(..., label_smoothing=): jittor's
        # cross_entropy_loss has no label_smoothing (used by many training recipes:
        # ImageNet, translation, some SFT). Delegate to jittor for ls=0 (verified
        # correct incl. weight/ignore_index); implement smoothing to match torch:
        #   loss_i = (1-ls)*nll_i + (ls/C)*smooth_i,  nll_i = -w[t]*logp[i,t],
        #   smooth_i = -sum_c(w_c*logp[i,c]);  mean divides by sum(w[t]) (or count).
        def _cross_entropy(input, target, weight=None, size_average=None,
                           ignore_index=-100, reduce=None, reduction="mean",
                           label_smoothing=0.0):
            # torch: a floating-point target with the SAME shape as input is a
            # class-probability ("soft label") target (mixup / distillation / soft
            # label-smoothing). jittor's cross_entropy_loss only understands integer
            # class-index targets, so handle the soft case here.
            if (isinstance(target, jt.Var) and target.ndim == input.ndim
                    and "int" not in str(target.dtype)):
                Cc = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
                cdim = 1 if input.ndim >= 2 else -1
                logp = nn.log_softmax(input, dim=cdim)
                tgt = target
                if label_smoothing:
                    tgt = (1.0 - label_smoothing) * tgt + label_smoothing / Cc
                if weight is not None:
                    wsh = [1] * input.ndim; wsh[cdim] = Cc
                    wloss = -(tgt * logp * weight.reshape(wsh)).sum(dim=cdim)
                else:
                    wloss = -(tgt * logp).sum(dim=cdim)
                if reduction == "sum":
                    return wloss.sum()
                if reduction == "none":
                    return wloss
                return wloss.mean()        # torch divides the soft-target loss by N
            if not label_smoothing:
                ii = -100 if ignore_index is None else ignore_index
                return _jt_ce(input, target, weight=weight, ignore_index=ii,
                              reduction=reduction)
            C = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
            if input.ndim > 2:                  # (N,C,d...) -> (M,C)
                perm = [0] + list(range(2, input.ndim)) + [1]
                x = input.transpose(perm).reshape((-1, C))
            else:
                x = input
            t = target.reshape((-1,))
            logp = nn.log_softmax(x, dim=-1)
            ig = None if ignore_index is None else ignore_index
            t_safe = t if ig is None else jt.ternary(t == ig, jt.zeros_like(t), t)
            nll = -logp.gather(1, t_safe.reshape((-1, 1))).reshape((-1,))
            if weight is not None:
                wt = weight[t_safe]
                nll = nll * wt
                smooth = -(logp * weight.reshape((1, -1))).sum(dim=-1)
            else:
                wt = None
                smooth = -logp.sum(dim=-1)
            loss = (1.0 - label_smoothing) * nll + (label_smoothing / C) * smooth
            if ig is not None:
                keep = (t != ig).float32()
                loss = loss * keep
                norm = (wt * keep).sum() if wt is not None else keep.sum()
            else:
                norm = wt.sum() if wt is not None else jt.array(float(t.shape[0]))
            if reduction == "sum":
                return loss.sum()
            if reduction == "none":
                return loss.reshape(target.shape) if input.ndim > 2 else loss
            return loss.sum() / norm
        F.cross_entropy = _cross_entropy
    # These losses are native functional implementations.  Torch mode only
    # publishes the canonical objects; keeping a second fallback body here
    # would make signatures and fixes diverge between the two entry points.
    from jittor.nn.functional.loss import (
        binary_cross_entropy as _native_bce,
        cosine_embedding_loss as _native_cosine_embedding,
        gaussian_nll_loss as _native_gaussian_nll,
        huber_loss as _native_huber,
        kl_div as _native_kl_div,
        margin_ranking_loss as _native_margin_ranking,
    )
    for _name, _fn in (
        ("binary_cross_entropy", _native_bce),
        ("cosine_embedding_loss", _native_cosine_embedding),
        ("gaussian_nll_loss", _native_gaussian_nll),
        ("huber_loss", _native_huber),
        ("kl_div", _native_kl_div),
        ("margin_ranking_loss", _native_margin_ranking),
    ):
        setattr(F, _name, _fn)
    # nn.*Loss class versions (criterion = nn.HuberLoss()): thin wrappers over the
    # functional. KLDivLoss/BCELoss/BCEWithLogitsLoss/CrossEntropyLoss/MSELoss/L1Loss
    # already exist on jittor.nn (verified correct); add the rest.
    _Mod = nn.Module
    def _add_loss_class(cname, fn, defaults, arg_order):
        if hasattr(nn, cname):
            return
        class _L(_Mod):
            def __init__(self, *a, **k):
                super().__init__()
                self._kw = dict(defaults); self._kw.update(k)
                for nm, val in zip(arg_order, a):
                    self._kw[nm] = val
            def execute(self, *inputs):
                return fn(*inputs, **self._kw)
        _L.__name__ = cname
        setattr(nn, cname, _L)
    _add_loss_class("HuberLoss", F.huber_loss, dict(reduction="mean", delta=1.0), ("reduction", "delta"))
    _add_loss_class("SmoothL1Loss", F.smooth_l1_loss, dict(reduction="mean"), ("reduction",))
    _add_loss_class("MarginRankingLoss", F.margin_ranking_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
    _add_loss_class("CosineEmbeddingLoss", F.cosine_embedding_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
    _add_loss_class("GaussianNLLLoss", F.gaussian_nll_loss, dict(full=False, eps=1e-6, reduction="mean"), ("full", "eps", "reduction"))
    _add_loss_class("NLLLoss", F.nll_loss, dict(reduction="mean"), ("weight", "size_average", "ignore_index"))
    # pixel_shuffle / pixel_unshuffle (super-resolution, some VAE decoders): jittor's
    # functional lacks them. (N, C*r^2, H, W) <-> (N, C, H*r, W*r). Verified vs torch.
    if not hasattr(F, "pixel_shuffle"):
        def _pixel_shuffle(input, upscale_factor):
            r = upscale_factor
            N, Cr2, H, W = input.shape
            C = Cr2 // (r * r)
            return input.reshape((N, C, r, r, H, W)).permute(0, 1, 4, 2, 5, 3).reshape((N, C, H * r, W * r))
        F.pixel_shuffle = _pixel_shuffle
        g.pixel_shuffle = _pixel_shuffle
    if not hasattr(F, "pixel_unshuffle"):
        def _pixel_unshuffle(input, downscale_factor):
            r = downscale_factor
            N, C, H, W = input.shape
            return input.reshape((N, C, H // r, r, W // r, r)).permute(0, 1, 3, 5, 2, 4).reshape((N, C * r * r, H // r, W // r))
        F.pixel_unshuffle = _pixel_unshuffle
        g.pixel_unshuffle = _pixel_unshuffle
    for _pscn, _psfn in (("PixelShuffle", "pixel_shuffle"), ("PixelUnshuffle", "pixel_unshuffle")):
        if not hasattr(nn, _pscn):
            def _mk(fn):
                class _PS(nn.Module):
                    def __init__(self, factor): super().__init__(); self._f = factor
                    def execute(self, x): return getattr(F, fn)(x, self._f)
                return _PS
            _cls = _mk(_psfn); _cls.__name__ = _pscn; setattr(nn, _pscn, _cls)
    # F.logsigmoid (DPO/preference losses), F.gumbel_softmax (discrete/MoE sampling).
    if not hasattr(F, "logsigmoid"):
        # stable: log(sigmoid(x)) = min(x,0) - log(1+exp(-|x|))
        F.logsigmoid = lambda input: jt.minimum(input, 0.0) - jt.log(1.0 + jt.exp(-jt.abs(input)))
    if not hasattr(F, "gumbel_softmax"):
        def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
            u = jt.rand(logits.shape)
            g = -jt.log(-jt.log(u + eps) + eps)             # Gumbel(0,1) noise
            y = nn.softmax((logits + g) / tau, dim=dim)
            if hard:
                m = y.max(dim, keepdims=True)
                y_hard = (y >= m).float32()
                y = (y_hard - y).stop_grad() + y            # straight-through estimator
            return y
        F.gumbel_softmax = _gumbel_softmax
    if not hasattr(F, "rms_norm"):
        # F.rms_norm (torch 2.4+): x / sqrt(mean(x^2, over last len(normalized_shape)
        # dims) + eps) * weight. The norm modern LLMs (Llama/Qwen/Gemma) use.
        def _rms_norm(input, normalized_shape, weight=None, eps=None):
            if eps is None:
                eps = 1.1920929e-07                          # finfo(float32).eps, torch default
            ndn = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
            dims = list(range(input.ndim - ndn, input.ndim))
            out = input * (1.0 / jt.sqrt((input * input).mean(dims, keepdims=True) + eps))
            return out * weight if weight is not None else out
        F.rms_norm = _rms_norm
    # Activations / losses jittor's functional lacked (verified vs real torch 2.12).
    if not hasattr(F, "softmin"):
        F.softmin = lambda input, dim=-1, _stacklevel=3, dtype=None: nn.softmax(-input, dim=dim)
    if not hasattr(F, "tanhshrink"):
        F.tanhshrink = lambda input: input - jt.tanh(input)
    if not hasattr(F, "celu"):
        F.celu = lambda input, alpha=1.0, inplace=False: \
            jt.maximum(input, 0.0) + jt.minimum(0.0, alpha * (jt.exp(input / alpha) - 1))
    if not hasattr(F, "selu"):
        def _selu(input, inplace=False):
            a = 1.6732632423543772848170429916717
            s = 1.0507009873554804934193349852946
            return s * (jt.maximum(input, 0.0) + jt.minimum(0.0, a * (jt.exp(input) - 1)))
        F.selu = _selu
    if not hasattr(F, "threshold"):
        def _threshold(input, threshold, value, inplace=False):
            m = (input > threshold).float32()
            return m * input + (1 - m) * value
        F.threshold = _threshold
    if not hasattr(F, "triplet_margin_loss"):
        def _triplet(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                     swap=False, size_average=None, reduce=None, reduction="mean"):
            def _d(a, b):
                return ((jt.abs(a - b) ** p).sum(-1) + eps) ** (1.0 / p)
            dp, dn = _d(anchor, positive), _d(anchor, negative)
            if swap:
                dn = jt.minimum(dn, _d(positive, negative))
            loss = jt.maximum(dp - dn + margin, 0.0)
            return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
        F.triplet_margin_loss = _triplet
    if not hasattr(F, "poisson_nll_loss"):
        def _poisson_nll(input, target, log_input=True, full=False, size_average=None,
                         eps=1e-8, reduce=None, reduction="mean"):
            loss = (jt.exp(input) - target * input) if log_input else (input - target * jt.log(input + eps))
            if full:
                import math as _mp
                stir = target * jt.log(jt.maximum(target, eps)) - target + 0.5 * jt.log(2 * _mp.pi * jt.maximum(target, eps))
                loss = loss + jt.ternary(target > 1, stir, jt.zeros_like(target))
            return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
        F.poisson_nll_loss = _poisson_nll
    if not hasattr(F, "ctc_loss"):
        # F.ctc_loss (wav2vec2 / speech ASR): the CTC forward (alpha) DP in log space.
        # log_probs (T,N,C) log-softmax; targets (N,S) padded or 1-D concatenated.
        # Differentiable (grad flows to log_probs). Verified bit-equal to real torch.
        import numpy as _np_ctc
        _CNEG = -1e30
        def _ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
                      reduction="mean", zero_infinity=False):
            def _ints(v):
                return [int(x) for x in (v.numpy().reshape(-1) if isinstance(v, jt.Var) else _np_ctc.asarray(v).reshape(-1))]
            in_lens, tgt_lens = _ints(input_lengths), _ints(target_lengths)
            tnp = targets.numpy() if isinstance(targets, jt.Var) else _np_ctc.asarray(targets)
            flat = (tnp.ndim == 1)
            def _shift(v, k):
                return jt.concat([jt.full((k,), _CNEG), v[:int(v.shape[0]) - k]]) if k > 0 else v
            def _lse(mats):
                m = mats[0]
                for x in mats[1:]:
                    m = jt.maximum(m, x)
                return m + jt.safe_log(sum(jt.exp(x - m) for x in mats))
            N = log_probs.shape[1]
            losses, offset = [], 0
            for n in range(N):
                Tn, Sn = in_lens[n], tgt_lens[n]
                if flat:
                    seq = [int(x) for x in tnp[offset:offset + Sn]]; offset += Sn
                else:
                    seq = [int(x) for x in tnp[n, :Sn]]
                ext = [blank]
                for lab in seq:
                    ext += [lab, blank]
                L = len(ext)
                ext_idx = jt.array(_np_ctc.array(ext, dtype="int64"))
                skip = _np_ctc.zeros(L, dtype="float32")
                for s in range(2, L):
                    if ext[s] != blank and ext[s] != ext[s - 2]:
                        skip[s] = 1.0
                skip_v = jt.array(skip)
                start = _np_ctc.full(L, _CNEG, dtype="float32"); start[0] = 0.0
                if L > 1:
                    start[1] = 0.0
                lp_n = log_probs[:Tn, n, :]
                alpha = lp_n[0][ext_idx] + jt.array(start)
                for t in range(1, Tn):
                    a2 = _shift(alpha, 2) * skip_v + (1 - skip_v) * _CNEG
                    alpha = lp_n[t][ext_idx] + _lse([alpha, _shift(alpha, 1), a2])
                losses.append(-(_lse([alpha[L - 1], alpha[L - 2]]) if L > 1 else alpha[L - 1]))
            out = jt.stack(losses).reshape((N,))   # (N,1)->(N,): jittor has no 0-d scalar
            if zero_infinity:
                out = jt.ternary(jt.isfinite(out), out, jt.zeros_like(out))
            if reduction == "none":
                return out
            if reduction == "sum":
                return out.sum()
            tl = jt.array(_np_ctc.array([max(s, 1) for s in tgt_lens], dtype="float32"))
            return (out / tl).mean()
        F.ctc_loss = _ctc_loss
    if hasattr(nn, "layer_norm"): F.layer_norm = nn.layer_norm
    if hasattr(nn, "embedding"): F.embedding = nn.embedding
    nn.functional = F
    g.nn.functional = nn.functional
    if not hasattr(nn.functional, "cosine_similarity") and hasattr(nn, "cosine_similarity"):
        nn.functional.cosine_similarity = nn.cosine_similarity
    if not hasattr(nn.functional, "pairwise_distance") and hasattr(nn, "pairwise_distance"):
        nn.functional.pairwise_distance = nn.pairwise_distance

    import os as _os

    _sdpa_flash_backend_cache = {}

    def _sdpa_static_backend_cache_enabled():
        return (_os.environ.get("JITTOR_TORCH_INFERENCE") or "").strip().lower() \
            in ("1", "true", "yes", "on")

    def _sdpa_flash_stats():
        stats = getattr(jt, "_torch_sdpa_flash_stats", None)
        if stats is None:
            stats = {"hits": 0, "misses": {}, "casts": {}, "backend": None}
            jt._torch_sdpa_flash_stats = stats
        return stats

    def _sdpa_flash_miss(reason):
        misses = _sdpa_flash_stats()["misses"]
        misses[reason] = misses.get(reason, 0) + 1

    def _sdpa_flash_cast(reason):
        casts = _sdpa_flash_stats()["casts"]
        casts[reason] = casts.get(reason, 0) + 1

    def _sdpa_flash_hit(backend_name):
        stats = _sdpa_flash_stats()
        stats["hits"] += 1
        stats["backend"] = backend_name

    def _sdpa_flash_template_dim(dim):
        dim = int(dim)
        if dim <= 0 or dim > 256 or dim % 8 != 0:
            return None
        if dim <= 32:
            return 32
        if dim <= 64:
            return 64
        if dim <= 96:
            return 96
        if dim <= 128:
            return 128
        if dim <= 192:
            return 192
        return 256

    def _sdpa_flash_float32_cast_target():
        raw = (_os.environ.get("JITTOR_FLASH_ATTN_CAST_FLOAT32") or "").strip().lower()
        if raw in ("1", "true", "yes", "on", "fp16", "float16", "half"):
            return "float16"
        if raw in ("bf16", "bfloat16"):
            return "bfloat16"
        return None

    def _try_flash_scaled_dot_product_attention(query, key, value, attn_mask,
                                                dropout_p, is_causal, sf,
                                                enable_gqa=False):
        if attn_mask is not None:
            _sdpa_flash_miss("mask")
            return None
        dropout = float(dropout_p or 0.0)
        if dropout < 0.0 or dropout >= 1.0:
            _sdpa_flash_miss("dropout_range")
            return None
        if not jt.flags.use_cuda:
            _sdpa_flash_miss("not_cuda")
            return None
        training_requested = not getattr(jt.flags, "no_grad", 0) or dropout != 0.0
        q_shape, k_shape, v_shape = tuple(query.shape), tuple(key.shape), tuple(value.shape)
        if len(q_shape) < 3 or len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape):
            _sdpa_flash_miss("rank")
            return None
        if q_shape[:-3] != k_shape[:-3] or q_shape[:-3] != v_shape[:-3]:
            _sdpa_flash_miss("batch")
            return None
        query_heads = int(q_shape[-3])
        key_heads = int(k_shape[-3])
        value_heads = int(v_shape[-3])
        gqa_heads_ok = (key_heads > 0 and enable_gqa
                        and query_heads % key_heads == 0)
        if key_heads != value_heads or not (
                query_heads == key_heads or gqa_heads_ok):
            _sdpa_flash_miss("heads")
            return None
        if q_shape[-1] != k_shape[-1] or q_shape[-1] != v_shape[-1]:
            _sdpa_flash_miss("head_dim_mismatch")
            return None
        # For CLIP-style short self-attention, the two cuBLAS matmuls plus the
        # fused softmax are faster than materializing the three layout copies
        # required by the separate-QKV FlashAttention wrapper. Keep this
        # inference-only and narrowly shaped so decoding, GQA and training keep
        # their existing backend choice.
        short_square_math = (
            _sdpa_static_backend_cache_enabled()
            and not enable_gqa and not is_causal
            and len(q_shape) == 4 and 0 < int(q_shape[0]) <= 8
            and query_heads == key_heads == value_heads == 12
            and int(q_shape[-1]) == 64
            and int(q_shape[-2]) == int(k_shape[-2]) == int(v_shape[-2])
            and int(q_shape[-2]) <= 64
            and str(query.dtype) == str(key.dtype) == str(value.dtype)
            and str(query.dtype) == "float16")
        if short_square_math:
            _sdpa_flash_miss("short_square_math")
            return None
        template_dim = _sdpa_flash_template_dim(q_shape[-1])
        if template_dim is None:
            _sdpa_flash_miss("head_dim")
            return None
        q_dtype, k_dtype, v_dtype = str(query.dtype), str(key.dtype), str(value.dtype)
        original_dtype = q_dtype
        cast_back = False
        if not (q_dtype == k_dtype == v_dtype and q_dtype in ("float16", "bfloat16")):
            cast_target = _sdpa_flash_float32_cast_target()
            if cast_target is None or not (q_dtype == k_dtype == v_dtype == "float32"):
                _sdpa_flash_miss("dtype")
                return None
            query = query.to(cast_target)
            key = key.to(cast_target)
            value = value.to(cast_target)
            q_dtype = k_dtype = v_dtype = cast_target
            cast_back = True
            _sdpa_flash_cast("float32_to_%s" % cast_target)
        try:
            from jittor.compat.shim.backends import flash_attention as _fa_jittor
        except Exception:
            _sdpa_flash_miss("no_loader")
            return None
        if training_requested and dropout == 0.0 and not _fa_jittor.required():
            raw_min_scores = _os.environ.get(
                "JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES", str(1 << 24))
            try:
                min_scores = max(0, int(raw_min_scores))
            except ValueError:
                min_scores = 1 << 24
            score_elements = query_heads * int(q_shape[-2]) * int(k_shape[-2])
            for size in q_shape[:-3]:
                score_elements *= int(size)
            if min_scores and score_elements < min_scores:
                _sdpa_flash_miss("short_training_math")
                return None
        cache_key = (template_dim, q_dtype)
        static_cache = _sdpa_static_backend_cache_enabled() or training_requested
        token_fn = getattr(_fa_jittor, "backend_cache_token", None)
        backend_token = (token_fn() if static_cache and callable(token_fn)
                         else None)
        cached = (_sdpa_flash_backend_cache.get(cache_key)
                  if static_cache and backend_token is not None else None)
        if cached is not None and cached[0] == backend_token:
            backend, capability_miss = cached[1], None
        else:
            backend, capability_miss = _fa_jittor.load_backend_for(
                template_dim, q_dtype)
            publication_fn = getattr(
                _fa_jittor, "backend_publication_token", None)
            publication_token = (
                publication_fn(backend) if callable(publication_fn) else None)
            backend_token = (token_fn() if static_cache and callable(token_fn)
                             else None)
            if (static_cache and backend_token is not None
                    and publication_token == backend_token
                    and backend is not None and capability_miss is None):
                _sdpa_flash_backend_cache[cache_key] = (
                    backend_token,
                    backend,
                )
        if backend is None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but native "
                    "flash-attn backend is unavailable: %s"
                    % (_fa_jittor.last_error() or "unknown error")
                )
            _sdpa_flash_miss("no_backend")
            return None
        if capability_miss is not None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "native flash-attn backend could not expand for %s: %s"
                    % (capability_miss, _fa_jittor.last_error() or "unsupported capability")
                )
            _sdpa_flash_miss(capability_miss)
            return None
        if (training_requested
                and not getattr(backend, "_flashattn_jittor_training", False)):
            if _fa_jittor.required():
                raise RuntimeError(
                    "native flash-attn backend does not advertise backward/dropout support")
            _sdpa_flash_miss("no_training_backend")
            return None
        # load_backend_for() already returned the capability-checked backend.
        # Calling through the public flash_attn stub would invoke the loader a
        # second time for every layer and rescan all backend environment keys.
        fn = getattr(backend, "flash_attn_func", None)
        if not callable(fn):
            if _fa_jittor.required():
                raise RuntimeError("flash_attn shim has no flash_attn_func")
            _sdpa_flash_miss("no_func")
            return None
        prefix = q_shape[:-3]
        p = len(prefix)
        batch = 1
        for size in prefix:
            batch *= int(size)
        heads, lq, head_dim = query_heads, int(q_shape[-2]), int(q_shape[-1])
        lk = int(k_shape[-2])
        q_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
        # Native flash-attn is an external C++/CUDA extension. Crossing that
        # boundary with a lazy permute/reshape expression can leave the bridge
        # holding transient metadata; clone materializes a stable row-major
        # tensor while keeping the kernel path fused.
        q_dense = query.permute(*q_axes).reshape((batch, lq, heads, head_dim)).clone()
        k_dense = key.permute(*q_axes).reshape((batch, lk, key_heads, head_dim)).clone()
        v_dense = value.permute(*q_axes).reshape((batch, lk, value_heads, head_dim)).clone()
        try:
            out = fn(
                q_dense, k_dense, v_dense, dropout, float(sf), bool(is_causal))
        except Exception:
            if _fa_jittor.required():
                raise
            _sdpa_flash_miss("call_failed")
            return None
        if out is None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "native flash-attn backend returned no output while "
                    "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set"
                )
            _sdpa_flash_miss("returned_none")
            return None
        out = out.reshape(tuple(prefix) + (lq, heads, head_dim))
        out_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
        _sdpa_flash_hit(_fa_jittor.backend_name())
        out = out.permute(*out_axes)
        if cast_back and str(out.dtype) != original_dtype:
            out = out.to(original_dtype)
        return out

    # The Torch wrapper owns backend selection and GQA expansion. The math
    # fallback remains the canonical native functional implementation.
    import math as _math
    from jittor.nn.functional.attention import (
        scaled_dot_product_attention as _native_scaled_dot_product_attention,
    )

    def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                     dropout_p=0.0, is_causal=False,
                                     scale=None, enable_gqa=False, **kw):
        del kw
        dimension = int(query.shape[-1])
        scale_factor = (
            1.0 / _math.sqrt(dimension) if scale is None else scale
        )
        flash = _try_flash_scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal,
            scale_factor, enable_gqa=enable_gqa)
        if flash is not None:
            return flash
        if enable_gqa:
            query_heads = int(query.shape[-3])
            key_heads = int(key.shape[-3])
            value_heads = int(value.shape[-3])
            if key_heads != query_heads:
                if key_heads <= 0 or query_heads % key_heads != 0:
                    raise RuntimeError("key heads must divide query heads for GQA")
                key = key.repeat_interleave(query_heads // key_heads, dim=-3)
            if value_heads != query_heads:
                if value_heads <= 0 or query_heads % value_heads != 0:
                    raise RuntimeError("value heads must divide query heads for GQA")
                value = value.repeat_interleave(
                    query_heads // value_heads, dim=-3
                )
        return _native_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    g.scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
    g._torch_sdpa_flash_backend_cache = _sdpa_flash_backend_cache

    _install_nn_extras(nn, ctx.registry)
    _modules["torch.nn"] = nn
    if hasattr(nn, "functional"):
        _modules["torch.nn.functional"] = nn.functional


def install_parity(ctx):
    import abc
    import importlib
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    nn = g.nn
    modules = registry.get("torch.nn.modules")

    def abc_base(name, *concrete):
        base = abc.ABCMeta(name, (object,), {})
        for item in concrete:
            if isinstance(item, type):
                base.register(item)
        return base

    conv = module("torch.nn.modules.conv")
    conv._ConvNd = getattr(
        conv,
        "_ConvNd",
        abc_base(
            "_ConvNd",
            getattr(nn, "Conv", None),
            getattr(nn, "Conv1d", None),
            getattr(nn, "Conv2d", None),
            getattr(nn, "Conv3d", None),
        ),
    )
    conv._ConvTransposeNd = getattr(
        conv,
        "_ConvTransposeNd",
        abc_base(
            "_ConvTransposeNd",
            getattr(nn, "ConvTranspose", None),
            getattr(nn, "ConvTranspose1d", None),
            getattr(nn, "ConvTranspose2d", None),
            getattr(nn, "ConvTranspose3d", None),
        ),
    )
    conv._ConvTransposeMixin = conv._ConvTransposeNd
    for name in ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
        value = getattr(nn, name, None)
        if value is not None:
            setattr(conv, name, value)
    modules.conv = conv

    pooling = importlib.import_module("jittor.nn.modules.pooling")
    registry.publish("torch.nn.modules.pooling", pooling)
    pooling._MaxPoolNd = getattr(
        pooling,
        "_MaxPoolNd",
        abc_base(
            "_MaxPoolNd",
            getattr(nn, "Pool", None),
            getattr(nn, "MaxPool1d", None),
            getattr(nn, "MaxPool2d", None),
            getattr(nn, "MaxPool3d", None),
        ),
    )
    pooling._AvgPoolNd = getattr(
        pooling,
        "_AvgPoolNd",
        abc_base(
            "_AvgPoolNd",
            getattr(nn, "AvgPool1d", None),
            getattr(nn, "AvgPool2d", None),
            getattr(nn, "AvgPool3d", None),
        ),
    )
    pooling._AdaptiveAvgPoolNd = getattr(
        pooling,
        "_AdaptiveAvgPoolNd",
        abc_base(
            "_AdaptiveAvgPoolNd",
            getattr(nn, "AdaptiveAvgPool1d", None),
            getattr(nn, "AdaptiveAvgPool2d", None),
            getattr(nn, "AdaptiveAvgPool3d", None),
        ),
    )
    pooling._AdaptiveMaxPoolNd = getattr(
        pooling,
        "_AdaptiveMaxPoolNd",
        abc_base(
            "_AdaptiveMaxPoolNd",
            getattr(nn, "AdaptiveMaxPool1d", None),
            getattr(nn, "AdaptiveMaxPool2d", None),
            getattr(nn, "AdaptiveMaxPool3d", None),
        ),
    )
    modules.pooling = pooling

    instancenorm = module("torch.nn.modules.instancenorm")
    instancenorm._InstanceNorm = getattr(
        instancenorm,
        "_InstanceNorm",
        abc_base(
            "_InstanceNorm",
            getattr(nn, "InstanceNorm", None),
            getattr(nn, "InstanceNorm1d", None),
            getattr(nn, "InstanceNorm2d", None),
            getattr(nn, "InstanceNorm3d", None),
        ),
    )
    modules.instancenorm = instancenorm

    stateless = module("torch.nn.utils.stateless")
    stateless.functional_call = getattr(
        getattr(g, "func", None),
        "functional_call",
        lambda target, parameters, args=(), kwargs=None: target(
            *args, **(kwargs or {})
        ),
    )
    nn.utils.stateless = stateless
