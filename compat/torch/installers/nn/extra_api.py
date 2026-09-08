"""Module-owned Torch neural-network implementations and layer templates."""
import jittor as jt
from jittor import nn
import jittor as _jt
import jittor as _jtm
import copy as _copy
import copy as _copy2
import types as _types_nn_private
from ...context import get_install_context
from ...types import _dtype_to_str
from ...tensor_state import latest_optimizer
from ...grad import _clip_grad_norm_device
from ....diagnostics import EXPECTED, swallowed
from .... import fsdp_hooks as _fsdp_hooks
from .... import collectives as _collectives
from .rnn import _rnn_pad_sequence
from jittor.nn.utils.weight_norm import (_ensure_reparam_hook, weight_norm as _native_weight_norm, remove_weight_norm as _native_remove_weight_norm)
_SELU_S, _SELU_A = 1.0507009873554805, 1.6732632423543772

def _ddp_world_size():
    return _collectives._world_size()

def _adapt_extra(template, registry):
    if not issubclass(template, jt.Module):
        return template
    return get_install_context(registry.target_namespace).state["nn_class_adapter"](template)

class UninitializedTensorMixin:
    pass


class UninitializedParameter:
    pass


class UninitializedBuffer:
    pass


class _DataParallel(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _DistributedDataParallel(_DataParallel):
    """DDP that really synchronises: broadcast at construction, all-reduce
    at the backward completion point.

    Before 7.02 this was a plain forwarding wrapper -- no bucket, no hook,
    no initial broadcast, ``no_sync()`` a nullcontext. Jittor only ever
    all-reduced inside ``opt.step(loss)``, while the torch-idiomatic
    ``loss.backward(); opt.step()`` fills gradients through ``Var.backward``
    and never touched MPI. On N ranks that trained N different models from
    N different random initialisations and reported nothing, so 7.01 made
    it refuse instead. It no longer has to.

    Two things have to be true for the ranks to stay identical:

    * **they must start identical** -- every parameter and buffer is
      broadcast from rank 0 here, because each rank ran its own random
      init;
    * **they must apply the same update** -- gradients are averaged across
      ranks at the point ``backward()`` finishes, before anything reads
      ``.grad``. torch does this in autograd hooks for the same reason:
      gradient clipping and logging between ``backward()`` and ``step()``
      have to see the synchronised gradient, not this rank's own.

    The all-reduce itself lives in ``installers/tensor.py`` (the backward
    is there, and it sits *below* this file), reached through the
    ``_jittor_ddp_state`` marker each parameter carries -- the same
    inversion FSDP2 uses. ``_jittor_ddp_order`` is assigned here, in
    ``module.parameters()`` order, so every rank issues its collectives in
    the same sequence: that order is identical across ranks, whereas the
    backward's own leaf collection is keyed by ``id()`` and is not.
    """

    require_backward_grad_sync = True

    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        state = _types_nn_private.SimpleNamespace(
            sync_enabled=True, world_size=_ddp_world_size())
        object.__setattr__(self, "_jittor_ddp_state", state)
        self._jittor_ddp_broadcast_parameters()
        self._jittor_ddp_mark_parameters()

    def _jittor_ddp_named_parameters(self):
        named = getattr(self.module, "named_parameters", None)
        if callable(named):
            for item in named():
                yield item[1] if isinstance(item, tuple) else item
            return
        for p in self.module.parameters():
            yield p

    def _jittor_ddp_broadcast_parameters(self):
        """Make every rank start from rank 0's weights."""
        if _ddp_world_size() <= 1:
            return
        for p in self._jittor_ddp_named_parameters():
            if isinstance(p, _jt.Var):
                _collectives._broadcast_from_rank0(p)
        buffers = getattr(self.module, "buffers", None)
        if callable(buffers):
            for b in buffers():
                b = b[1] if isinstance(b, tuple) else b
                if isinstance(b, _jt.Var):
                    _collectives._broadcast_from_rank0(b)
        _jt.sync_all()

    def _jittor_ddp_mark_parameters(self):
        """Tag the parameters the backward has to all-reduce, in rank-stable
        order."""
        state = self._jittor_ddp_state
        for index, p in enumerate(self._jittor_ddp_named_parameters()):
            if not isinstance(p, _jt.Var):
                continue
            try:
                object.__setattr__(p, "_jittor_ddp_state", state)
                object.__setattr__(p, "_jittor_ddp_order", index)
            except EXPECTED as exc:
                swallowed(
                    "torch/installers/nn.py DDP: mark parameter %d for "
                    "gradient all-reduce" % index, exc,
                    "that parameter's gradient will NOT be synchronised, "
                    "so this rank's copy of it will diverge")

    def no_sync(self):
        """Skip the gradient all-reduce inside the block.

        Used for gradient accumulation: the local gradients add up over
        several micro-batches and only the final backward pays for one
        collective. It used to be ``nullcontext()``, which was accidentally
        correct only because nothing was ever synchronised.
        """
        return _DDPNoSync(self)


class _DDPNoSync:
    def __init__(self, ddp):
        self._ddp = ddp
        self._previous = None

    def __enter__(self):
        state = self._ddp._jittor_ddp_state
        self._previous = state.sync_enabled
        state.sync_enabled = False
        self._ddp.require_backward_grad_sync = False
        return self._ddp

    def __exit__(self, *exc_info):
        state = self._ddp._jittor_ddp_state
        state.sync_enabled = self._previous
        self._ddp.require_backward_grad_sync = bool(self._previous)
        return False


def _flex_attention(*args, **kwargs):
    raise NotImplementedError("flex_attention is not supported on jittor backend")


def _grads_of(params):
    _context = get_install_context(jt)
    _torch_target = _context.target_namespace
    params = list(params)
    opt = latest_optimizer(_torch_target)
    out = []
    for p in params:
        gg = None
        if opt is not None:
            try: gg = opt.find_grad(p)
            except EXPECTED as exc:
                swallowed("torch/installers/nn.py _grads_of: gg = opt.find_grad(p)", exc)
                gg = None
        if gg is None:
            gg = getattr(p, "grad", None)
        if gg is not None:
            out.append(gg)
    return out


def _parameters_are_sharded(parameters):
    """True when these gradients are FSDP *shards*.

    Each rank then holds a different slice of the same logical
    gradient, so the norm has to be combined across ranks before the
    root is taken -- otherwise every rank clips by its own slice's
    norm, which is always too small, and each scales by a different
    coefficient. For DDP the answer is False and must stay False: its
    ranks already hold the same averaged gradient, so reducing would
    count one norm N times.

    Asked through the seam rather than by importing fsdp2, which sits
    above this file (jittor/compat/fsdp_hooks.py).
    """
    fsdp = _fsdp_hooks.provider()
    if fsdp is None:
        return False
    for p in parameters:
        if isinstance(p, _jt.Var) and fsdp.is_fsdp_managed_param(p):
            return True
    return False


def clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                    error_if_nonfinite=False, **k):
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    parameters = list(parameters)
    grads = _grads_of(parameters)
    return _clip_grad_norm_device(
        grads, max_norm, norm_type, error_if_nonfinite,
        shard_reduce=_parameters_are_sharded(parameters))


def clip_grad_value_(parameters, clip_value, **k):
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    for g in _grads_of(parameters):
        g.update(g.clamp(-clip_value, clip_value))


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
    except (AttributeError, TypeError) as exc: swallowed("torch/installers/nn.py spectral_norm: delattr(module, name)", exc)
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


class Hardswish(nn.Module):
    def execute(self, x):
        return x * _jt.clamp(x + 3, 0, 6) / 6


class CELU(nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(); self.alpha = alpha
    def execute(self, x):
        a = self.alpha
        return _jt.maximum(x, 0.0) + _jt.minimum(0.0, a * (_jt.exp(x / a) - 1))


class SELU(nn.Module):
    def __init__(self, inplace=False): super().__init__()
    def execute(self, x):
        return _SELU_S * (_jt.maximum(x, 0.0) + _jt.minimum(0.0, _SELU_A * (_jt.exp(x) - 1)))


class Softsign(nn.Module):
    def execute(self, x): return x / (1 + _jt.abs(x))


class Tanhshrink(nn.Module):
    def execute(self, x): return x - _jt.tanh(x)


class Softplus(nn.Module):
    def __init__(self, beta=1, threshold=20): super().__init__(); self.beta=beta; self.threshold=threshold
    def execute(self, x):
        bx = self.beta * x
        return _jt.ternary(bx > self.threshold, x, _jt.log1p(_jt.exp(bx)) / self.beta)


class Hardshrink(nn.Module):
    def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
    def execute(self, x): return x * ((x > self.lambd) | (x < -self.lambd)).float()


class Softshrink(nn.Module):
    def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
    def execute(self, x):
        l = self.lambd
        return _jt.maximum(x - l, 0.0) - _jt.maximum(-x - l, 0.0)


class Hardsigmoid(nn.Module):
    def execute(self, x):
        return _jt.clamp(x + 3, 0, 6) / 6


class Identity(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def execute(self, x): return x


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


def _act_fn(activation):
    _context = get_install_context(jt)
    nn = _context.target_namespace.nn
    if callable(activation):
        return activation
    return {"relu": nn.relu, "gelu": nn.gelu}.get(activation, nn.relu)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        _context = get_install_context(jt)
        nn = _context.target_namespace.nn
        from jittor._core.dtypes import dtype_for_compute
        if dtype is not None:
            dtype_for_compute(dtype)
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


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, **kw):
        _context = get_install_context(jt)
        nn = _context.target_namespace.nn
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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        _context = get_install_context(jt)
        nn = _context.target_namespace.nn
        from jittor._core.dtypes import dtype_for_compute
        if dtype is not None:
            dtype_for_compute(dtype)
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


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, **kw):
        _context = get_install_context(jt)
        nn = _context.target_namespace.nn
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


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 bias=True, device=None, dtype=None):
        _context = get_install_context(jt)
        nn = _context.target_namespace.nn
        from jittor._core.dtypes import dtype_for_compute
        if dtype is not None:
            dtype_for_compute(dtype)
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
        from jittor._core.dtypes import dtype_for_compute
        if dtype is not None:
            dtype_for_compute(dtype)
        m = _jtm.triu(_jtm.ones((sz, sz)), 1) * (-1e30)
        return m


class SyncBatchNorm(nn.BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None, **kw):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        return module


def _lazy_batch_norm_init(self, *args, _name=None, **kwargs):
    raise NotImplementedError(
        "%s is not implemented by Jittor Torch compatibility" % (_name or type(self).__name__)
    )


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


def _adaptive_max_pool2d(input, output_size, return_indices=False):
    _context = get_install_context(jt)
    nn = _context.target_namespace.nn
    out = nn.AdaptiveMaxPool2d(output_size)(input)
    return (out, None) if return_indices else out


def _linear(input, weight, bias=None):
    _context = get_install_context(jt)
    _jt_linear = _context.state["nn_extra_native"]['_jt_linear']
    if hasattr(weight, "ndim") and weight.ndim == 1:
        out = (input * weight).sum(-1)
        return out if bias is None else out + bias
    return _jt_linear(input, weight, bias)


def _upsample_bilinear(input, size=None, scale_factor=None):
    _context = get_install_context(jt)
    F = _context.target_namespace.nn.functional
    return F.interpolate(input, size=size, scale_factor=scale_factor,
                         mode="bilinear", align_corners=True)


def _upsample(input, size=None, scale_factor=None, mode="nearest",
              align_corners=None):
    _context = get_install_context(jt)
    F = _context.target_namespace.nn.functional
    return F.interpolate(input, size=size, scale_factor=scale_factor,
                         mode=mode, align_corners=align_corners)


def linear_init(self, in_features, out_features, bias=True,
                device=None, dtype=None):
    _context = get_install_context(jt)
    _torch_target = _context.target_namespace
    native_linear_init = _context.state["nn_extra_native"]['native_linear_init']
    native_linear_init(self, in_features, out_features, bias)
    target_dtype = dtype if dtype is not None else _torch_target.get_default_dtype()
    self.to(device=device, dtype=target_dtype)


def embedding_init(self, num_embeddings, embedding_dim, padding_idx=None,
                   max_norm=None, norm_type=2.0,
                   scale_grad_by_freq=False, sparse=False,
                   _weight=None, _freeze=False, device=None, dtype=None):
    _context = get_install_context(jt)
    _torch_target = _context.target_namespace
    native_embedding_init = _context.state["nn_extra_native"]['native_embedding_init']
    target_dtype = dtype if dtype is not None else _torch_target.get_default_dtype()
    native_embedding_init(
        self, num_embeddings, embedding_dim, padding_idx,
        _dtype_to_str(target_dtype), max_norm, norm_type,
        scale_grad_by_freq, sparse, _weight, _freeze, device,
    )
    self.to(device=device, dtype=target_dtype)



def _dropout_init(self, p=0.5, inplace=False, *a, **k):
    initializers = get_install_context(jt).state["nn_extra_native"]["dropout_initializers"]
    for base in type(self).__mro__:
        if base in initializers:
            return initializers[base](self, p, *a, **k)
    raise RuntimeError("Dropout initializer has no native owner")


def _api_flex_mod_create_block_mask(*args, **kwargs):
    return None


def _api_flex_mod_and_masks(*args, **kwargs):
    return None


def _api_flex_mod_or_masks(*args, **kwargs):
    return None


def _api_flex_mod_noop_mask(*args, **kwargs):
    return None


def _api_parametrize_register_parametrization(module, *a, **k):
    return module


def _api_parametrize_remove_parametrizations(module, *a, **k):
    return module


def _api_parametrize_is_parametrized(module, *a, **k):
    return False


def _api_parametrize_type_before_parametrizations(module):
    return type(module)


def _api_parametrizations_orthogonal(module, *a, **k):
    return module


def _api_prune_is_pruned(module):
    return False


def _api_F_relu_(input):
    nn = get_install_context(jt).target_namespace.nn
    return nn.relu(input)


def _api_parametrizations_weight_norm(module, name='weight', dim=0):
    return module


def _api_parametrizations_spectral_norm(module, *a, **k):
    return module
