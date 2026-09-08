from ...fidelity import Fidelity, register_api_bindings
import jittor as jt
from ...context import registry_for
from ..nn_init import _install_init_aliases
from ...grad import _clip_grad_norm_device
from ...nn_modules import install_module_namespace
from ...types import _dtype_to_str
from ...tensor_state import latest_optimizer
from ....diagnostics import EXPECTED, swallowed
from .... import fsdp_hooks as _fsdp_hooks
from .... import collectives as _collectives

from .module_methods import _install_module_methods
from .rnn import _rnn_pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence
from .norm_utils import _get_total_norm, _clip_grads_with_norm_

def _ddp_world_size():
    """This rank's view of the world. See collectives._world_size."""
    return _collectives._world_size()

from . import extra_api as _extra_api
from .extra_api import (
    _api_flex_mod_create_block_mask,
    _api_flex_mod_and_masks,
    _api_flex_mod_or_masks,
    _api_flex_mod_noop_mask,
    _api_parametrize_register_parametrization,
    _api_parametrize_remove_parametrizations,
    _api_parametrize_is_parametrized,
    _api_parametrize_type_before_parametrizations,
    _api_parametrizations_orthogonal,
    _api_prune_is_pruned,
    _api_F_relu_,
    _api_parametrizations_weight_norm,
    _api_parametrizations_spectral_norm,

    _adapt_extra,
    _act_fn,
    _adaptive_max_pool2d,
    _flex_attention,
    _grads_of,
    _l2_normalize,
    _lazy_batch_norm_init,
    _linear,
    _parameters_are_sharded,
    _resolve_parent,
    _unsupported_prune,
    _upsample,
    _upsample_bilinear,
    clip_grad_norm_,
    clip_grad_value_,
    embedding_init,
    linear_init,
    pad_sequence,
    remove_weight_norm,
    spectral_norm,
    swap_tensor,
    weight_norm,
)
from ...context import get_install_context
from types import MappingProxyType

def _mk_nn_submod(_modules, modules_pkg, _name, **_attrs):
    import types as _types_nn_private
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


def _install_nn_extras(nn, registry=None):
    # Activation modules torch has that jittor.nn may lack.
    active_registry = registry_for(jt, registry)
    _modules = active_registry.module_map
    _torch_target = active_registry.target_namespace
    _dropout_initializers = {}
    import jittor as _jt
    _install_init_aliases(registry)
    import types as _types_nn_private

    if not getattr(getattr(nn, "Parameter", None), "_torch_compat_type", False):
        # One implementation, not two: jittor.nn's Parameter already runs torch's
        # construction protocol, so a subclass gets its own __new__/__init__, its
        # extra keyword arguments, and an isinstance that answers for that
        # subclass alone. A second copy here would drift from it.
        from jittor.nn.modules.parameter import Parameter
        Parameter._torch_compat_type = True
        UninitializedTensorMixin = _adapt_extra(_extra_api.UninitializedTensorMixin, active_registry)
        UninitializedParameter = _adapt_extra(_extra_api.UninitializedParameter, active_registry)
        UninitializedBuffer = _adapt_extra(_extra_api.UninitializedBuffer, active_registry)
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

    modules_pkg = install_module_namespace(nn, registry)
    from jittor.ops.tuples import _single, _pair, _triple, _ntuple


    _mk_nn_submod(_modules, modules_pkg, "utils", _single=_single, _pair=_pair, _triple=_triple,
                  _ntuple=_ntuple, _quadruple=_ntuple(4))
    _mk_nn_submod(_modules, modules_pkg, "batchnorm",
                  _BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm1d=getattr(nn, "BatchNorm1d", getattr(nn, "BatchNorm", None)),
                  BatchNorm2d=getattr(nn, "BatchNorm2d", getattr(nn, "BatchNorm", None)),
                  BatchNorm3d=getattr(nn, "BatchNorm3d", getattr(nn, "BatchNorm", None)),
                  SyncBatchNorm=getattr(nn, "SyncBatchNorm", getattr(nn, "BatchNorm", None)))
    _mk_nn_submod(_modules, modules_pkg, "normalization",
                  GroupNorm=getattr(nn, "GroupNorm", None),
                  LayerNorm=getattr(nn, "LayerNorm", None),
                  LocalResponseNorm=getattr(nn, "LocalResponseNorm", None))
    _mk_nn_submod(_modules, modules_pkg, "activation",
                  ReLU=getattr(nn, "ReLU", None), SiLU=getattr(nn, "SiLU", None),
                  Sigmoid=getattr(nn, "Sigmoid", None), Tanh=getattr(nn, "Tanh", None),
                  GELU=getattr(nn, "GELU", None), LeakyReLU=getattr(nn, "LeakyReLU", None))
    parallel_mod = _modules.get("torch.nn.parallel")
    if parallel_mod is None:
        parallel_mod = _types_nn_private.ModuleType("torch.nn.parallel")
        _modules["torch.nn.parallel"] = parallel_mod

    _DataParallel = _adapt_extra(_extra_api._DataParallel, active_registry)


    _DistributedDataParallel = _adapt_extra(_extra_api._DistributedDataParallel, active_registry)

    _DDPNoSync = _adapt_extra(_extra_api._DDPNoSync, active_registry)

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
        flex_mod.flex_attention = _flex_attention
        flex_mod.create_block_mask = _api_flex_mod_create_block_mask
        flex_mod.BlockMask = type("BlockMask", (), {})
        flex_mod._DEFAULT_SPARSE_BLOCK_SIZE = 128
        flex_mod.and_masks = _api_flex_mod_and_masks
        flex_mod.or_masks = _api_flex_mod_or_masks
        flex_mod.AuxRequest = type("AuxRequest", (), {})
        flex_mod.AuxOutput = type("AuxOutput", (), {})
        flex_mod.flex_attention_hop = None
        flex_mod.noop_mask = _api_flex_mod_noop_mask
        _modules["torch.nn.attention.flex_attention"] = flex_mod
    attn_mod.flex_attention = flex_mod
    nn.attention = attn_mod

    # nn.utils.clip_grad_norm_/clip_grad_value_ (also provided by torch_shim,
    # but needed for the bare `import jittor as torch` path too).
    if not hasattr(nn, "utils") or not hasattr(getattr(nn, "utils", None), "clip_grad_norm_"):
        import types as _t
        _u = getattr(nn, "utils", None) or _t.ModuleType("torch.nn.utils")

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

        _u.weight_norm = weight_norm
        _u.remove_weight_norm = remove_weight_norm
        _u.spectral_norm = spectral_norm

        # --- nn.utils.rnn.pad_sequence ---
        import types as _trnn
        _rnn = _trnn.ModuleType("torch.nn.utils.rnn")
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

    _clip_grad._get_total_norm = _get_total_norm
    _clip_grad._clip_grads_with_norm_ = _clip_grads_with_norm_
    _clip_grad.clip_grad_norm_ = getattr(_u, "clip_grad_norm_", None)
    _clip_grad.clip_grad_value_ = getattr(_u, "clip_grad_value_", None)
    _modules["torch.nn.utils.clip_grad"] = _clip_grad
    _u.clip_grad = _clip_grad
    if not hasattr(_u, "parametrize"):
        _parametrize = _types_nn_utils.ModuleType("torch.nn.utils.parametrize")
        _parametrize.register_parametrization = _api_parametrize_register_parametrization
        _parametrize.remove_parametrizations = _api_parametrize_remove_parametrizations
        _parametrize.is_parametrized = _api_parametrize_is_parametrized
        _parametrize.type_before_parametrizations = _api_parametrize_type_before_parametrizations
        _u.parametrize = _parametrize
        _modules["torch.nn.utils.parametrize"] = _parametrize
    else:
        _modules.setdefault("torch.nn.utils.parametrize", _u.parametrize)
    if not hasattr(_u, "parametrizations"):
        _parametrizations = _types_nn_utils.ModuleType("torch.nn.utils.parametrizations")
        _parametrizations.weight_norm = getattr(_u, "weight_norm", _api_parametrizations_weight_norm)
        _parametrizations.spectral_norm = getattr(_u, "spectral_norm", _api_parametrizations_spectral_norm)
        _parametrizations.orthogonal = _api_parametrizations_orthogonal
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

    _rnn.pad_sequence = _rnn_pad_sequence
    _rnn.pack_padded_sequence = pack_padded_sequence
    _rnn.pad_packed_sequence = pad_packed_sequence
    _rnn.PackedSequence = PackedSequence
    _u.rnn = _rnn
    _modules["torch.nn.utils.rnn"] = _rnn

    if "torch.nn.utils.prune" not in _modules:
        _prune = _types_nn_utils.ModuleType("torch.nn.utils.prune")


        BasePruningMethod = _adapt_extra(_extra_api.BasePruningMethod, active_registry)

        L1Unstructured = _adapt_extra(_extra_api.L1Unstructured, active_registry)

        RandomUnstructured = _adapt_extra(_extra_api.RandomUnstructured, active_registry)

        LnStructured = _adapt_extra(_extra_api.LnStructured, active_registry)

        RandomStructured = _adapt_extra(_extra_api.RandomStructured, active_registry)

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
        _prune.is_pruned = _api_prune_is_pruned
        _modules["torch.nn.utils.prune"] = _prune
    _u.prune = _modules["torch.nn.utils.prune"]
    if "torch.nn.utils._named_member_accessor" not in _modules:
        _named_accessor = _types_nn_utils.ModuleType("torch.nn.utils._named_member_accessor")
        _named_accessor.swap_tensor = swap_tensor
        _modules["torch.nn.utils._named_member_accessor"] = _named_accessor
    _u._named_member_accessor = _modules["torch.nn.utils._named_member_accessor"]

    if not hasattr(nn, "Hardswish"):
        Hardswish = _adapt_extra(_extra_api.Hardswish, active_registry)
        nn.Hardswish = Hardswish
    if not hasattr(nn, "CELU"):           # timm uses nn.CELU
        CELU = _adapt_extra(_extra_api.CELU, active_registry)
        nn.CELU = CELU
    # A batch of standard torch activations jittor.nn may lack (timm's act-layer
    # registry references all of them at import). All are pure elementwise.
    if not hasattr(nn, "SELU"):
        _SELU_S, _SELU_A = 1.0507009873554805, 1.6732632423543772
        SELU = _adapt_extra(_extra_api.SELU, active_registry)
        nn.SELU = SELU
    if not hasattr(nn, "Softsign"):
        Softsign = _adapt_extra(_extra_api.Softsign, active_registry)
        nn.Softsign = Softsign
    if not hasattr(nn, "Tanhshrink"):
        Tanhshrink = _adapt_extra(_extra_api.Tanhshrink, active_registry)
        nn.Tanhshrink = Tanhshrink
    if not hasattr(nn, "Softplus"):
        Softplus = _adapt_extra(_extra_api.Softplus, active_registry)
        nn.Softplus = Softplus
    if not hasattr(nn, "Hardshrink"):
        Hardshrink = _adapt_extra(_extra_api.Hardshrink, active_registry)
        nn.Hardshrink = Hardshrink
    if not hasattr(nn, "Softshrink"):
        Softshrink = _adapt_extra(_extra_api.Softshrink, active_registry)
        nn.Softshrink = Softshrink
    if not hasattr(nn, "Hardsigmoid"):
        Hardsigmoid = _adapt_extra(_extra_api.Hardsigmoid, active_registry)
        nn.Hardsigmoid = Hardsigmoid
    if not hasattr(nn, "Identity"):
        Identity = _adapt_extra(_extra_api.Identity, active_registry)
        nn.Identity = Identity
    # ModuleList/Sequential/ModuleDict usually exist; alias ParameterList if not
    if not hasattr(nn, "ParameterList"):
        nn.ParameterList = nn.ModuleList if hasattr(nn, "ModuleList") else list
    # ModuleDict (peft LoRA layers need it); jittor lacks it.
    if not hasattr(nn, "ModuleDict"):
        ModuleDict = _adapt_extra(_extra_api.ModuleDict, active_registry)
        nn.ModuleDict = ModuleDict

    # Layer classes torch has that jittor.nn may lack -- needed at least for
    # isinstance() checks in model init. Provide a distinct empty subclass so
    # isinstance discrimination still works.
    if not hasattr(nn, "ConvTranspose1d"):
        ConvTranspose1d = _adapt_extra(_extra_api.ConvTranspose1d, active_registry)
        nn.ConvTranspose1d = ConvTranspose1d
    if not hasattr(nn, "RMSNorm"):
        RMSNorm = _adapt_extra(_extra_api.RMSNorm, active_registry)
        nn.RMSNorm = RMSNorm
    # Transformer modules build on the canonical jittor.nn.MultiheadAttention.
    import jittor as _jtm


    if not hasattr(nn, "TransformerEncoderLayer"):
        TransformerEncoderLayer = _adapt_extra(_extra_api.TransformerEncoderLayer, active_registry)
        nn.TransformerEncoderLayer = TransformerEncoderLayer

    if not hasattr(nn, "TransformerEncoder"):
        import copy as _copy
        TransformerEncoder = _adapt_extra(_extra_api.TransformerEncoder, active_registry)
        nn.TransformerEncoder = TransformerEncoder

    if not hasattr(nn, "TransformerDecoderLayer"):
        TransformerDecoderLayer = _adapt_extra(_extra_api.TransformerDecoderLayer, active_registry)
        nn.TransformerDecoderLayer = TransformerDecoderLayer

    if not hasattr(nn, "TransformerDecoder"):
        import copy as _copy2
        TransformerDecoder = _adapt_extra(_extra_api.TransformerDecoder, active_registry)
        nn.TransformerDecoder = TransformerDecoder

    if not hasattr(nn, "Transformer"):
        Transformer = _adapt_extra(_extra_api.Transformer, active_registry)
        nn.Transformer = Transformer

    # ---- nn.SyncBatchNorm (single-device: behaves exactly like BatchNorm) ----
    # mmdetection's rtmdet calls `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`;
    # with no real process group there is nothing to synchronise, so the convert
    # entry returns the model unchanged (BN already pools over the whole batch here).
    if not hasattr(nn, "SyncBatchNorm"):
        SyncBatchNorm = _adapt_extra(_extra_api.SyncBatchNorm, active_registry)
        nn.SyncBatchNorm = SyncBatchNorm
    for lazy_batch_norm_name in (
        "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d"
    ):
        if hasattr(nn, lazy_batch_norm_name):
            continue


        lazy_batch_norm = type(
            lazy_batch_norm_name,
            (nn.Module,),
            {
                "__init__": _lazy_batch_norm_init,
                "__module__": "torch.nn.modules.batchnorm",
            },
        )
        setattr(nn, lazy_batch_norm_name, lazy_batch_norm)

    # ---- nn.functional extras used by mmdetection (not auto-copied from jittor.nn) ----
    F = nn.functional
    if not hasattr(F, "_Reduction"):
        # torch's private reduction-string -> enum helper; mmdet's loss utils do
        # `F._Reduction.get_enum(reduction)` then branch 0/1/2 = none/mean/sum.
        _Reduction = _adapt_extra(_extra_api._Reduction, active_registry)
        F._Reduction = _Reduction
    if not hasattr(F, "adaptive_max_pool2d"):
        F.adaptive_max_pool2d = _adaptive_max_pool2d
    if not getattr(F, "_torch_linear_wrapped", False):
        # torch's F.linear accepts a 1-D weight (matrix-vector product), e.g. GFL's
        # Integral does F.linear(x[N,K], project[K]) -> [N]; jittor's linear asserts 2-D.
        _jt_linear = F.linear
        F.linear = _linear
        F._torch_linear_wrapped = True
    if not hasattr(F, "relu_"):
        F.relu_ = _api_F_relu_   # in-place relu (graph-equivalent)
    if not hasattr(F, "upsample_bilinear"):
        # deprecated torch alias == interpolate(mode='bilinear', align_corners=True)
        F.upsample_bilinear = _upsample_bilinear
    if not hasattr(F, "upsample"):
        F.upsample = _upsample

    # torch's nn.Conv2d exposes .transposed / .output_padding (torchvision &
    # mmcv's ConvModule read them to introspect the layer); jittor's Conv lacks
    # them. Add torch-compatible class attributes.
    for _cn in ("Conv", "Conv1d", "Conv2d", "Conv3d"):
        _c = getattr(nn, _cn, None)
        if _c is not None:
            if not hasattr(_c, "transposed"):
                _c.transposed = False
            if not hasattr(_c, "output_padding"):
                _c.output_padding = (0, 0)
    for _cn in (
        "ConvTranspose", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"
    ):
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
            _dropout_initializers[_dc] = _dc.__init__
            _dc.__init__ = _extra_api._dropout_init
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
            except (AttributeError, TypeError) as exc:
                swallowed("torch/installers/nn.py _install_nn_extras: _cls.__name__ = _nm", exc)

    _install_module_methods(nn, registry)

    linear_cls = getattr(nn, "Linear", None)
    if linear_cls is not None and not getattr(linear_cls, "_torch_dtype_patched", False):
        native_linear_init = linear_cls.__init__


        linear_cls.__init__ = linear_init
        linear_cls._torch_dtype_patched = True

    embedding_cls = getattr(nn, "Embedding", None)
    if (embedding_cls is not None and
            not getattr(embedding_cls, "_torch_dtype_patched", False)):
        native_embedding_init = embedding_cls.__init__


        embedding_cls.__init__ = embedding_init
        embedding_cls._torch_dtype_patched = True

    get_install_context(_torch_target).state["nn_extra_native"] = MappingProxyType({
        "dropout_initializers": MappingProxyType(_dropout_initializers),
        "_jt_linear": locals().get("_jt_linear"),
        "native_linear_init": locals().get("native_linear_init"),
        "native_embedding_init": locals().get("native_embedding_init"),
    })

    register_api_bindings(nn, 'torch.nn',
        ('CELU', 'ConvTranspose1d', 'DataParallel', 'Hardshrink', 'Hardsigmoid', 'Hardswish', 'Identity', 'ModuleDict', 'Parameter', 'ParameterList', 'RMSNorm', 'SELU', 'Softplus', 'Softshrink', 'Softsign', 'SyncBatchNorm', 'Tanhshrink', 'Transformer', 'TransformerDecoder', 'TransformerDecoderLayer', 'TransformerEncoder', 'TransformerEncoderLayer', 'attention', 'parallel', 'parameter', 'utils') + tuple(()),
        Fidelity.APPROXIMATE, 'Native neural-network implementations and installation-owned layer adapters; unsupported Torch modes and parameter subsets remain restricted')
    register_api_bindings(F, "torch.nn.functional", ("linear",),
        Fidelity.APPROXIMATE, "Native linear mathematics with frontend dtype promotion and installation-owned delegates")
    register_api_bindings(
        nn.utils, "torch.nn.utils",
        ("clip_grad_norm_", "clip_grad_value_", "weight_norm", "remove_weight_norm",
         "spectral_norm", "get_total_norm", "clip_grads_with_norm_"),
        Fidelity.APPROXIMATE, "Native gradient and reparametrization operations; "
        "distributed, hook, dtype and foreach restrictions apply")
    register_api_bindings(
        parallel_mod, "torch.nn.parallel", ("DataParallel", "DistributedDataParallel"),
        Fidelity.APPROXIMATE, "Native module wrapper and registered collectives; "
        "the complete Torch process-group and bucket interface is not supported")
    for _path, _names in (
        ("torch.nn.attention.flex_attention", ("flex_attention", "create_block_mask",
         "and_masks", "or_masks", "noop_mask")),
        ("torch.nn.utils.parametrize", ("register_parametrization", "remove_parametrizations",
         "is_parametrized", "type_before_parametrizations")),
        ("torch.nn.utils.prune", ("is_pruned",)),
    ):
        _namespace = _modules.get(_path)
        if _namespace is not None:
            register_api_bindings(_namespace, _path, _names, Fidelity.UNIMPLEMENTED,
                "Import compatibility only: raises or returns a placeholder without "
                "implementing the requested transformation")
