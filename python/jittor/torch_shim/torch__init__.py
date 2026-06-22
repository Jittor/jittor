"""Shim package: `import torch` -> jittor (with torch_compat layer).

Lets torch-targeted libraries (transformers, LlamaFactory, ...) run on jittor
unmodified. jittor's torch_compat layer supplies the torch-style API on the
jittor module; this package re-exports it as `torch` and wires up the common
`torch.<submodule>` paths.
"""
import sys, types, contextlib
import builtins as _builtins
import jittor as _jt
import jittor as _jittor

from jittor import *          # noqa: F401,F403
for _k in dir(_jittor):
    if not _k.startswith("__"):
        globals().setdefault(_k, getattr(_jittor, _k))

__version__ = "2.11.0"

# ---- nn / functional ----
import jittor.nn as nn
sys.modules["torch.nn"] = nn
if hasattr(nn, "functional"):
    sys.modules["torch.nn.functional"] = nn.functional

# nn.init -> jittor.init
try:
    import jittor.init as _init
    # torch.nn.init helpers peft/transformers import that jittor lacks
    import math as _math_init
    def _calculate_fan_in_and_fan_out(tensor):
        shape = tensor.shape
        nd = len(shape)
        if nd < 2:
            return shape[0], shape[0]
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive = 1
        if nd > 2:
            for s in shape[2:]:
                receptive *= s
        return num_input_fmaps * receptive, num_output_fmaps * receptive
    def _calculate_correct_fan(tensor, mode):
        mode = mode.lower()
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == "fan_in" else fan_out
    def _calculate_gain(nonlinearity, param=None):
        linear_fns = ("linear", "conv1d", "conv2d", "conv3d", "sigmoid")
        if nonlinearity in linear_fns or nonlinearity == "sigmoid":
            return 1.0
        if nonlinearity == "tanh":
            return 5.0 / 3
        if nonlinearity == "relu":
            return _math_init.sqrt(2.0)
        if nonlinearity == "leaky_relu":
            neg = 0.01 if param is None else param
            return _math_init.sqrt(2.0 / (1 + neg ** 2))
        if nonlinearity == "selu":
            return 3.0 / 4
        return 1.0
    for _fn_name, _fn in [("_calculate_fan_in_and_fan_out", _calculate_fan_in_and_fan_out),
                          ("_calculate_correct_fan", _calculate_correct_fan),
                          ("calculate_gain", _calculate_gain)]:
        if not hasattr(_init, _fn_name):
            setattr(_init, _fn_name, _fn)
    nn.init = _init
    sys.modules["torch.nn.init"] = _init
except Exception:
    pass

# nn.parameter.Parameter -> jittor Var (with requires_grad)
# Must be usable both as a constructor `Parameter(data)` AND in isinstance()
# checks. jittor has no distinct Parameter type (params are just trainable
# Vars), so we use a metaclass whose __instancecheck__ treats any Var as a
# Parameter, and __call__ returns a (cloned, grad-tracking) Var.
class _ParameterMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _jt.Var)
    def __call__(cls, data=None, requires_grad=True):
        v = data if isinstance(data, _jt.Var) else _jt.array(data)
        v = v.clone()
        if requires_grad:
            try: v.requires_grad = True
            except Exception: v.start_grad()
        else:
            v.stop_grad()
        return v
class Parameter(metaclass=_ParameterMeta):
    pass
nn.Parameter = Parameter
_param_mod = types.ModuleType("torch.nn.parameter")
_param_mod.Parameter = Parameter
_param_mod.UninitializedParameter = Parameter
_param_mod.UninitializedBuffer = Parameter
sys.modules["torch.nn.parameter"] = _param_mod
# torch.nn IS jittor.nn here; bind submodule as attribute so `nn.parameter` works
nn.parameter = _param_mod

# ---- torch.nn.utils (clip_grad_*, weight_norm, parametrize, rnn helpers) ----
_nn_utils = types.ModuleType("torch.nn.utils")
def _collect_grads(parameters):
    # parameters may be a generator (torch passes model.parameters()); realize it
    params = list(parameters)
    out = []
    for p in params:
        g = None
        # prefer the optimizer-held grad Var so in-place clip reaches step()
        if hasattr(p, "opt_grad"):
            try:
                opt = getattr(_jt, "_current_optimizer", None)
                if opt is not None:
                    g = opt.find_grad(p)
            except Exception:
                g = None
        if g is None:
            g = getattr(p, "grad", None)
        if g is not None:
            out.append(g)
    return out
def _clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    import builtins as _b
    _inf = _b.float("inf")
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    grads = _collect_grads(parameters)
    if not grads:
        return _jt.array(0.0)
    if norm_type == _inf:
        total = _jt.concat([g.abs().reshape(-1) for g in grads]).max()
    else:
        sq = _jt.concat([g.cast("float32").sqr().reshape(-1) for g in grads])
        total = _jt.sqrt(sq.sum())
    # max_norm may be inf (transformers calls clip with inf just to read norm)
    try:
        mn = _b.float(max_norm)
    except Exception:
        mn = _inf
    if mn != _inf:
        clip_coef = mn / (_b.float(total.item()) + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.update(g * clip_coef)   # .update() -> reflected in optimizer.step()
    return total
def _clip_grad_value_(parameters, clip_value, foreach=None):
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    for g in _collect_grads(parameters):
        g.update(g.clamp(-clip_value, clip_value))
_nn_utils.clip_grad_norm_ = _clip_grad_norm_
_nn_utils.clip_grad_value_ = _clip_grad_value_
_nn_utils.weight_norm = lambda module, name="weight", dim=0: module
_nn_utils.remove_weight_norm = lambda module, name="weight": module
_nn_utils.spectral_norm = lambda module, *a, **k: module
def _parameters_to_vector(parameters):
    flats = [p.reshape(-1) for p in parameters]
    return _jt.concat(flats) if flats else _jt.array([])
_nn_utils.parameters_to_vector = _parameters_to_vector
def _vector_to_parameters(vec, parameters):
    off = 0
    for p in parameters:
        n = _builtins.int(p.numel())
        p.assign(vec[off:off+n].reshape(p.shape))
        off += n
_nn_utils.vector_to_parameters = _vector_to_parameters
sys.modules["torch.nn.utils"] = _nn_utils
nn.utils = _nn_utils
# torch.nn.utils.parametrize (peft / some models probe it)
_parametrize = types.ModuleType("torch.nn.utils.parametrize")
_parametrize.register_parametrization = lambda module, *a, **k: module
_parametrize.remove_parametrizations = lambda module, *a, **k: module
_parametrize.is_parametrized = lambda module, *a, **k: False
_parametrize.type_before_parametrizations = lambda module: type(module)
_nn_utils.parametrize = _parametrize
sys.modules["torch.nn.utils.parametrize"] = _parametrize
# torch.nn.utils.rnn (pad/pack helpers)
_rnn = types.ModuleType("torch.nn.utils.rnn")
def _pad_sequence(sequences, batch_first=False, padding_value=0.0):
    max_len = _builtins.max(s.shape[0] for s in sequences)
    padded = []
    for s in sequences:
        if s.shape[0] < max_len:
            pad_shape = list(s.shape); pad_shape[0] = max_len - s.shape[0]
            s = _jt.concat([s, _jt.full(pad_shape, padding_value).cast(str(s.dtype))], dim=0)
        padded.append(s)
    out = _jt.stack(padded, dim=0)
    return out if batch_first else out.transpose(0, 1)
_rnn.pad_sequence = _pad_sequence
_rnn.pack_padded_sequence = lambda input, lengths, *a, **k: input
_rnn.pad_packed_sequence = lambda sequence, *a, **k: (sequence, None)
_rnn.PackedSequence = type("PackedSequence", (), {})
_nn_utils.rnn = _rnn
sys.modules["torch.nn.utils.rnn"] = _rnn
# torch.nn.utils.stateless (functional_call)
_stateless = types.ModuleType("torch.nn.utils.stateless")
_stateless.functional_call = lambda module, params, args=(), kwargs=None: module(*args, **(kwargs or {}))
_nn_utils.stateless = _stateless
sys.modules["torch.nn.utils.stateless"] = _stateless

# nn.modules.module.Module -> jittor Module
_mod_module = types.ModuleType("torch.nn.modules.module")
_mod_module.Module = nn.Module
_mod_module._EXTRA_STATE_KEY_SUFFIX = "_extra_state"
_mod_module._global_backward_hooks = {}
_mod_module._global_forward_hooks = {}
_mod_module._global_forward_pre_hooks = {}
_modules_pkg = types.ModuleType("torch.nn.modules")
_modules_pkg.Module = nn.Module
_modules_pkg.module = _mod_module
sys.modules["torch.nn.modules"] = _modules_pkg
sys.modules["torch.nn.modules.module"] = _mod_module
# torch.nn IS jittor.nn here; attribute access torch.nn.modules reads from it,
# so bind the submodules onto the jittor.nn module object directly.
nn.modules = _modules_pkg
if not hasattr(nn, "Parameter"):
    nn.Parameter = Parameter

# nn.attention.flex_attention (stub: transformers guards usage at runtime)
_attn = types.ModuleType("torch.nn.attention")
_flex = types.ModuleType("torch.nn.attention.flex_attention")
def _flex_attention(*a, **k):
    raise NotImplementedError("flex_attention not supported on jittor backend")
_flex.flex_attention = _flex_attention
_flex.create_block_mask = lambda *a, **k: None
_flex.BlockMask = type("BlockMask", (), {})
_flex._DEFAULT_SPARSE_BLOCK_SIZE = 128
_flex.and_masks = lambda *a, **k: None
_flex.or_masks = lambda *a, **k: None
_flex.AuxRequest = type("AuxRequest", (), {})
_flex.AuxOutput = type("AuxOutput", (), {})
_flex.flex_attention_hop = None
_flex.noop_mask = lambda *a, **k: None
_attn.flex_attention = _flex
sys.modules["torch.nn.attention"] = _attn
sys.modules["torch.nn.attention.flex_attention"] = _flex
nn.attention = _attn

# ---- nn.parallel (DDP/DataParallel are no-op passthroughs on single device) ----
_parallel = types.ModuleType("torch.nn.parallel")
class _DataParallel(nn.Module):
    def __init__(self, module, *a, **k):
        super().__init__()
        self.module = module
    def execute(self, *a, **k):
        return self.module(*a, **k)
class _DistributedDataParallel(_DataParallel):
    pass
_parallel.DataParallel = _DataParallel
_parallel.DistributedDataParallel = _DistributedDataParallel
nn.DataParallel = _DataParallel
nn.parallel = _parallel
sys.modules["torch.nn.parallel"] = _parallel
_parallel_distrib = types.ModuleType("torch.nn.parallel.distributed")
_parallel_distrib.DistributedDataParallel = _DistributedDataParallel
sys.modules["torch.nn.parallel.distributed"] = _parallel_distrib
_parallel.distributed = _parallel_distrib

# ---- cuda ----
if hasattr(_jittor, "cuda"):
    sys.modules["torch.cuda"] = _jittor.cuda

# ---- backends (accelerate/transformers probe torch.backends.*) ----
_backends = types.ModuleType("torch.backends")
_b_mps = types.ModuleType("torch.backends.mps")
_b_mps.is_available = lambda: False
_b_mps.is_built = lambda: False
_b_cudnn = types.ModuleType("torch.backends.cudnn")
_b_cudnn.is_available = lambda: _builtins.bool(getattr(_jittor.flags, "use_cuda", 0))
_b_cudnn.enabled = True
_b_cudnn.benchmark = False
_b_cudnn.deterministic = False
_b_cudnn.version = lambda: None
_b_cuda = types.ModuleType("torch.backends.cuda")
class _SDPKernel:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
_b_cuda.sdp_kernel = lambda *a, **k: _SDPKernel()
_b_cuda.enable_flash_sdp = lambda *a, **k: None
_b_cuda.enable_mem_efficient_sdp = lambda *a, **k: None
_b_cuda.enable_math_sdp = lambda *a, **k: None
_b_cuda.matmul = type("_m", (), {"allow_tf32": False})()
_b_cpu = types.ModuleType("torch.backends.cpu")
_b_cpu.get_cpu_capability = lambda: "DEFAULT"
_b_mkldnn = types.ModuleType("torch.backends.mkldnn")
_b_mkldnn.is_available = lambda: False
_b_mkldnn.enabled = False
_backends.mps = _b_mps
_backends.cudnn = _b_cudnn
_backends.cuda = _b_cuda
_backends.cpu = _b_cpu
_backends.mkldnn = _b_mkldnn
sys.modules["torch.backends"] = _backends
sys.modules["torch.backends.mps"] = _b_mps
sys.modules["torch.backends.cudnn"] = _b_cudnn
sys.modules["torch.backends.cuda"] = _b_cuda
sys.modules["torch.backends.cpu"] = _b_cpu
sys.modules["torch.backends.mkldnn"] = _b_mkldnn
globals()["backends"] = _backends

# ---- distributed (single-process stubs; HCCL multi-card handled separately) ----
dist = types.ModuleType("torch.distributed")
def _is_available(): return False
def _is_initialized(): return False
dist.is_available = _is_available
dist.is_initialized = _is_initialized
dist.get_rank = lambda *a, **k: 0
dist.get_world_size = lambda *a, **k: 1
dist.init_process_group = lambda *a, **k: None
dist.destroy_process_group = lambda *a, **k: None
dist.barrier = lambda *a, **k: None
dist.all_reduce = lambda *a, **k: None
dist.all_gather = lambda *a, **k: None
dist.broadcast = lambda *a, **k: None
dist.ReduceOp = type("ReduceOp", (), {"SUM": 0, "MEAN": 1, "MAX": 2, "MIN": 3})
dist.is_torchelastic_launched = lambda: False
dist.GroupMember = type("GroupMember", (), {"WORLD": None})
dist.group = type("group", (), {"WORLD": None})
sys.modules["torch.distributed"] = dist
for _sub in ("tensor", "fsdp", "device_mesh", "_composable", "checkpoint", "algorithms"):
    _m = types.ModuleType(f"torch.distributed.{_sub}")
    sys.modules[f"torch.distributed.{_sub}"] = _m
    setattr(dist, _sub, _m)
dist.tensor.DTensor = type("DTensor", (), {})
dist.tensor.Replicate = type("Replicate", (), {})
dist.tensor.Shard = type("Shard", (), {})

# ---- optim ----
import jittor.optim as _optim
sys.modules["torch.optim"] = _optim
# torch exposes optim.Optimizer base; map to jittor's
if not hasattr(_optim, "Optimizer"):
    _optim.Optimizer = getattr(_optim, "Optimizer", object)

# torch optimizers accept kwargs jittor's don't (fused, foreach, capturable,
# amsgrad, differentiable, maximize). Wrap the common ones to drop unknown
# kwargs and normalize betas/eps so transformers' Trainer can build them.
def _make_torch_optim(jit_cls, accepted):
    import builtins as _b
    class _Wrapped(jit_cls):
        def __init__(self, params, lr=1e-3, **kw):
            # normalize torch param-group dicts: keep only keys jittor knows
            clean = {k: v for k, v in kw.items() if k in accepted}
            # jittor wants a plain list of Vars or list of {'params':...} dicts;
            # torch passes list of dicts possibly with extra keys (lr, wd, etc.)
            try:
                super().__init__(params, lr, **clean)
            except Exception:
                # last resort: strip param-group extras to bare params
                if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
                    flat = []
                    for g in params:
                        flat.extend(list(g.get("params", [])))
                    super().__init__(flat, lr, **clean)
                else:
                    raise
            # register as the active optimizer so a bare `loss.backward()`
            # (torch/accelerate style) can route grads through jittor.
            _jt._current_optimizer = self
            # torch stores lr per param_group; jittor keeps it only in self.lr.
            # accelerate's AcceleratedOptimizer delegates param_groups but NOT
            # .lr, so a scheduler reading pg["lr"] would see nothing and default
            # to 0. Mirror lr into every param_group so it survives wrapping.
            try:
                for pg in self.param_groups:
                    pg.setdefault("lr", self.lr)
            except Exception:
                pass
        def zero_grad(self, set_to_none=True):
            return super().zero_grad()
        def load_state_dict(self, state):
            # jittor's load_state_dict runs a dfs that calls .stop_grad() on
            # every Var it meets -- including the model params nested under
            # param_groups -- which silently FREEZES all trainable params.
            # accelerate round-trips state_dict()/load_state_dict() at wrap time,
            # so guard it: snapshot which params were trainable, restore after.
            trainable = []
            try:
                for pg in self.param_groups:
                    for p in pg.get("params", []):
                        if not p.is_stop_grad():
                            trainable.append(p)
            except Exception:
                pass
            super().load_state_dict(state)
            for p in trainable:
                try:
                    p.start_grad()
                except Exception:
                    pass
    _Wrapped.__name__ = jit_cls.__name__
    return _Wrapped

_ADAM_KW = ("eps", "betas", "weight_decay")
_SGD_KW = ("momentum", "weight_decay", "dampening", "nesterov")
if hasattr(_optim, "AdamW"):
    _optim.AdamW = _make_torch_optim(_optim.AdamW, _ADAM_KW)
if hasattr(_optim, "Adam"):
    _optim.Adam = _make_torch_optim(_optim.Adam, _ADAM_KW)
if hasattr(_optim, "SGD"):
    _optim.SGD = _make_torch_optim(_optim.SGD, _SGD_KW)
try:
    import jittor.lr_scheduler as _lrs
    # torch-compatible LR schedulers driving jittor optimizers. jittor reads lr
    # from pg.get("lr", self.lr), so we must update BOTH optimizer.lr and each
    # param_group["lr"] on every step for the new lr to take effect.
    def _set_opt_lr(optimizer, lrs):
        for pg, lr in zip(optimizer.param_groups, lrs):
            pg["lr"] = lr
        try:
            optimizer.lr = lrs[0]
        except Exception:
            pass
    def _base_lrs(optimizer):
        base = []
        for pg in optimizer.param_groups:
            base.append(pg.get("lr", getattr(optimizer, "lr", 0.0)))
        return base or [getattr(optimizer, "lr", 0.0)]

    class _LRScheduler:
        """torch-compatible base scheduler over a jittor optimizer."""
        def __init__(self, optimizer, last_epoch=-1, verbose=False):
            self.optimizer = optimizer
            self.base_lrs = _base_lrs(optimizer)
            self.last_epoch = last_epoch
            self._step_count = 0
            self._last_lr = list(self.base_lrs)
            self.step()   # initialize lr at epoch 0 (torch convention)
        def get_lr(self):
            return list(self.base_lrs)
        def get_last_lr(self):
            return list(self._last_lr)
        def state_dict(self):
            return {k: v for k, v in self.__dict__.items() if k not in ("optimizer",)}
        def load_state_dict(self, sd):
            self.__dict__.update(sd)
        def step(self, epoch=None):
            self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
            self._step_count += 1
            lrs = self.get_lr()
            self._last_lr = list(lrs)
            _set_opt_lr(self.optimizer, lrs)
    _lrs._LRScheduler = _LRScheduler
    _lrs.LRScheduler = _LRScheduler

    class LambdaLR(_LRScheduler):
        def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
            self.base_lrs = _base_lrs(optimizer)
            n = len(self.base_lrs)
            self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda]*n
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            e = _builtins.max(self.last_epoch, 0)
            return [base * fn(e) for base, fn in zip(self.base_lrs, self.lr_lambdas)]
    _lrs.LambdaLR = LambdaLR

    class MultiplicativeLR(_LRScheduler):
        def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
            self.base_lrs = _base_lrs(optimizer)
            n = len(self.base_lrs)
            self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda]*n
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            if self.last_epoch <= 0:
                return list(self.base_lrs)
            return [lr * fn(self.last_epoch) for lr, fn in zip(self._last_lr, self.lr_lambdas)]
    _lrs.MultiplicativeLR = MultiplicativeLR

    class _ConstantLR(_LRScheduler):
        def get_lr(self):
            return list(self.base_lrs)
    _lrs.ConstantLR = _ConstantLR

    class StepLR(_LRScheduler):
        def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
            self.step_size = step_size; self.gamma = gamma
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            f = self.gamma ** (_builtins.max(self.last_epoch, 0) // self.step_size)
            return [base * f for base in self.base_lrs]
    _lrs.StepLR = StepLR

    if not hasattr(_lrs, "ReduceLROnPlateau"):
        class ReduceLROnPlateau:
            def __init__(self, optimizer, *a, **k): self.optimizer = optimizer
            def step(self, *a, **k): pass
            def get_last_lr(self): return [getattr(self.optimizer, "lr", 0.0)]
            def state_dict(self): return {}
            def load_state_dict(self, sd): pass
        _lrs.ReduceLROnPlateau = ReduceLROnPlateau
    _optim.lr_scheduler = _lrs
    sys.modules["torch.optim.lr_scheduler"] = _lrs
except Exception as _e:
    print("[torch-shim] lr_scheduler setup failed:", _e)

# ---- utils.data ----
_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = _utils
try:
    import jittor.dataset as _ds
    _data = types.ModuleType("torch.utils.data")
    _data.Dataset = getattr(_ds, "Dataset", object)
    _data.DataLoader = getattr(_ds, "DataLoader", getattr(_ds, "Dataset", object))

    # ---- pure-Python sampler/dataset implementations (mirror torch.utils.data) ----
    class _TorchDataset:
        def __getitem__(self, i): raise NotImplementedError
        def __add__(self, other): return _ConcatDataset([self, other])
    class _IterableDataset(_TorchDataset):
        def __iter__(self): raise NotImplementedError
    class _TensorDataset(_TorchDataset):
        def __init__(self, *tensors): self.tensors = tensors
        def __getitem__(self, i): return tuple(t[i] for t in self.tensors)
        def __len__(self): return len(self.tensors[0]) if self.tensors else 0
    class _ConcatDataset(_TorchDataset):
        def __init__(self, datasets):
            self.datasets = list(datasets)
            self.cum = []
            s = 0
            for d in self.datasets:
                s += len(d); self.cum.append(s)
        def __len__(self): return self.cum[-1] if self.cum else 0
        def __getitem__(self, idx):
            import bisect
            di = bisect.bisect_right(self.cum, idx)
            prev = self.cum[di-1] if di else 0
            return self.datasets[di][idx-prev]
    class _Subset(_TorchDataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset; self.indices = list(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return self.dataset[self.indices[i]]

    class _Sampler:
        def __init__(self, data_source=None): self.data_source = data_source
        def __iter__(self): raise NotImplementedError
    class _SequentialSampler(_Sampler):
        def __iter__(self): return iter(range(len(self.data_source)))
        def __len__(self): return len(self.data_source)
    class _RandomSampler(_Sampler):
        def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
            self.data_source = data_source; self.replacement = replacement
            self._num_samples = num_samples; self.generator = generator
        @property
        def num_samples(self):
            return len(self.data_source) if self._num_samples is None else self._num_samples
        def __iter__(self):
            import random as _r
            n = len(self.data_source)
            idx = list(range(n)); _r.shuffle(idx)
            return iter(idx[:self.num_samples])
        def __len__(self): return self.num_samples
    class _SubsetRandomSampler(_Sampler):
        def __init__(self, indices, generator=None): self.indices = list(indices)
        def __iter__(self):
            import random as _r
            idx = list(self.indices); _r.shuffle(idx); return iter(idx)
        def __len__(self): return len(self.indices)
    class _BatchSampler(_Sampler):
        def __init__(self, sampler, batch_size, drop_last):
            self.sampler = sampler; self.batch_size = batch_size; self.drop_last = drop_last
        def __iter__(self):
            batch = []
            for x in self.sampler:
                batch.append(x)
                if len(batch) == self.batch_size:
                    yield batch; batch = []
            if batch and not self.drop_last:
                yield batch
        def __len__(self):
            n = len(self.sampler)
            if self.drop_last: return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size

    class _DataLoader:
        """Pure-Python DataLoader (mirrors torch.utils.data.DataLoader semantics)."""
        def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0, collate_fn=None,
                     pin_memory=False, drop_last=False, timeout=0,
                     worker_init_fn=None, generator=None, prefetch_factor=None,
                     persistent_workers=False, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.collate_fn = collate_fn if collate_fn is not None else (lambda b: b)
            self.worker_init_fn = worker_init_fn
            self.generator = generator
            if batch_sampler is not None:
                self.batch_sampler = batch_sampler
                self.sampler = None
            else:
                if sampler is None:
                    sampler = _RandomSampler(dataset) if shuffle else _SequentialSampler(dataset)
                self.sampler = sampler
                self.batch_sampler = _BatchSampler(sampler, batch_size, drop_last)
        def __iter__(self):
            for batch_idx in self.batch_sampler:
                yield self.collate_fn([self.dataset[i] for i in batch_idx])
        def __len__(self):
            return len(self.batch_sampler)
    _data.DataLoader = _DataLoader

    _data.Dataset = _TorchDataset
    _data.IterableDataset = _IterableDataset
    _data.TensorDataset = _TensorDataset
    _data.ConcatDataset = _ConcatDataset
    _data.Subset = _Subset
    _data.Sampler = _Sampler
    _data.SequentialSampler = _SequentialSampler
    _data.RandomSampler = _RandomSampler
    _data.SubsetRandomSampler = _SubsetRandomSampler
    _data.BatchSampler = _BatchSampler
    def _get_worker_info(): return None
    _data.get_worker_info = _get_worker_info
    def _default_collate(batch): return batch
    def _default_convert(x): return x
    _data.default_collate = _default_collate
    _data.default_convert = _default_convert
    # _utils.collate submodule (accelerate/transformers probe it)
    _du = types.ModuleType("torch.utils.data._utils")
    _duc = types.ModuleType("torch.utils.data._utils.collate")
    _duc.default_collate = _default_collate
    _du.collate = _duc
    sys.modules["torch.utils.data._utils"] = _du
    sys.modules["torch.utils.data._utils.collate"] = _duc
    _data._utils = _du
    # DataLoaderDispatcher placeholder used by accelerate
    sys.modules["torch.utils.data"] = _data
    _utils.data = _data
    _distm = types.ModuleType("torch.utils.data.distributed")
    _distm.DistributedSampler = _Sampler
    sys.modules["torch.utils.data.distributed"] = _distm
    _data.distributed = _distm
except Exception:
    pass

# checkpoint (gradient checkpointing) -> just call the function
_ckpt = types.ModuleType("torch.utils.checkpoint")
def checkpoint(fn, *args, use_reentrant=None, **kwargs):
    return fn(*args, **kwargs)
_ckpt.checkpoint = checkpoint
sys.modules["torch.utils.checkpoint"] = _ckpt
_utils.checkpoint = _ckpt

# _pytree (used widely) -> minimal impl
_pytree = types.ModuleType("torch.utils._pytree")
def _tree_flatten(x):
    leaves = []
    def rec(o):
        if isinstance(o, (list, tuple)):
            for e in o: rec(e)
        elif isinstance(o, dict):
            for e in o.values(): rec(e)
        else:
            leaves.append(o)
    rec(x)
    return leaves, None
_pytree.tree_flatten = _tree_flatten
_pytree.tree_unflatten = lambda leaves, spec: list(leaves)
_pytree.tree_map = lambda f, x: f(x)
_pytree.register_pytree_node = lambda *a, **k: None
_pytree._register_pytree_node = lambda *a, **k: None
sys.modules["torch.utils._pytree"] = _pytree
_utils._pytree = _pytree

# ---- distributions ----
try:
    import jittor.distributions as _distrib
    if not hasattr(_distrib, "constraints"):
        _con = types.ModuleType("torch.distributions.constraints")
        class _Constraint:
            def __init__(self, *a, **k): pass
            def check(self, x): return True
        for _cn in ("Constraint","positive","real","nonnegative","nonnegative_integer",
                    "positive_integer","unit_interval","simplex","lower_cholesky",
                    "greater_than","greater_than_eq","less_than","interval",
                    "half_open_interval","integer_interval","boolean","real_vector",
                    "positive_definite","cat","stack","dependent","independent"):
            setattr(_con, _cn, _Constraint())
        _con.Constraint = _Constraint
        _distrib.constraints = _con
        sys.modules["torch.distributions.constraints"] = _con
    sys.modules["torch.distributions"] = _distrib
    globals()["distributions"] = _distrib

    # torch.distributions is a *package* with importable submodules; jittor's is
    # a flat module. Register the submodules peft/transformers import. We back
    # them with light pure-Python distributions (reparameterized where needed).
    if not hasattr(_distrib, "Distribution"):
        class _Distribution:
            has_rsample = False
            def __init__(self, *a, **k): pass
            def sample(self, sample_shape=()): raise NotImplementedError
            def rsample(self, sample_shape=()): return self.sample(sample_shape)
            def log_prob(self, value): raise NotImplementedError
        _distrib.Distribution = _Distribution
    else:
        _Distribution = _distrib.Distribution

    def _sigmoid(x): return 1.0 / (1.0 + (-x).exp())

    class _RelaxedBernoulli(_Distribution):
        has_rsample = True
        def __init__(self, temperature=1.0, probs=None, logits=None, **k):
            self.temperature = temperature
            if logits is None and probs is not None:
                logits = (probs / (1 - probs)).log()
            self.logits = logits
            self.probs = probs
        def rsample(self, sample_shape=()):
            shape = self.logits.shape
            u = _jt.rand(shape).clamp(1e-6, 1 - 1e-6)
            noise = (u.log() - (1 - u).log())
            return _sigmoid((self.logits + noise) / self.temperature)
        def sample(self, sample_shape=()):
            return self.rsample(sample_shape)
    _relaxed = types.ModuleType("torch.distributions.relaxed_bernoulli")
    _relaxed.RelaxedBernoulli = _RelaxedBernoulli
    _relaxed.LogitRelaxedBernoulli = _RelaxedBernoulli
    sys.modules["torch.distributions.relaxed_bernoulli"] = _relaxed
    _distrib.relaxed_bernoulli = _relaxed
    _distrib.RelaxedBernoulli = _RelaxedBernoulli

    # bind existing jittor distributions and common aliases at expected paths
    for _dn, _alias in [("Normal", "normal"), ("Categorical", "categorical"),
                        ("Uniform", "uniform"), ("Geometric", "geometric")]:
        if hasattr(_distrib, _dn):
            _sm = types.ModuleType(f"torch.distributions.{_alias}")
            setattr(_sm, _dn, getattr(_distrib, _dn))
            sys.modules[f"torch.distributions.{_alias}"] = _sm
    _distmod = types.ModuleType("torch.distributions.distribution")
    _distmod.Distribution = _distrib.Distribution
    sys.modules["torch.distributions.distribution"] = _distmod
    _distrib.kl = types.ModuleType("torch.distributions.kl")
    _distrib.kl.kl_divergence = getattr(_distrib, "kl_divergence", lambda *a, **k: 0)
    sys.modules["torch.distributions.kl"] = _distrib.kl
    if not hasattr(_distrib, "register_kl"):
        _distrib.register_kl = lambda *a, **k: (lambda f: f)
except Exception as _e:
    pass

# ---- compiler / fx / autograd / export (stubs) ----
_compiler = types.ModuleType("torch.compiler")
_cid = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda g: g))
_compiler.is_compiling = lambda: False
_compiler.is_dynamo_compiling = lambda: False
_compiler.is_exporting = lambda: False
_compiler.disable = _cid
_compiler.allow_in_graph = _cid
_compiler.assume_constant_result = _cid
_compiler.wrap_numpy = _cid
_compiler.reset = lambda *a, **k: None
_compiler.cudagraph_mark_step_begin = lambda *a, **k: None
sys.modules["torch.compiler"] = _compiler
globals()["compiler"] = _compiler

def compile(model=None, *a, **k):
    return model if model is not None else (lambda m: m)
globals()["compile"] = compile

_autograd = types.ModuleType("torch.autograd")
_autograd.Function = getattr(_jittor, "Function", object)
_autograd.no_grad = _jittor.no_grad
_autograd.enable_grad = _jittor.enable_grad
sys.modules["torch.autograd"] = _autograd
globals()["autograd"] = _autograd

_fx = types.ModuleType("torch.fx")
_fx.Graph = type("Graph", (), {})
_fx.GraphModule = type("GraphModule", (), {})
_fx.wrap = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))
sys.modules["torch.fx"] = _fx
globals()["fx"] = _fx

# torch.overrides (used by transformers utils)
_ovr = types.ModuleType("torch.overrides")
_ovr.is_tensor_like = lambda x: isinstance(x, _jt.Var)
sys.modules["torch.overrides"] = _ovr


# Bind submodules as attributes too (some libs do torch.distributed.x not import)
import sys as _sys2
for _name, _m in list(_sys2.modules.items()):
    if _name.startswith("torch.") and _name.count(".") == 1:
        globals().setdefault(_name.split(".", 1)[1], _m)
globals()["nn"] = nn
globals()["distributed"] = _sys2.modules["torch.distributed"]
globals()["optim"] = _sys2.modules["torch.optim"]
globals()["utils"] = _sys2.modules["torch.utils"]
globals()["cuda"] = _sys2.modules.get("torch.cuda", globals().get("cuda"))

# torch.random module (accelerate calls torch.random.initial_seed())
import builtins as _builtins
_random_mod = types.ModuleType("torch.random")
_random_mod._seed = 0
def _initial_seed():
    return _random_mod._seed
def _manual_seed(seed):
    _random_mod._seed = _builtins.int(seed)
    try: _jt.set_global_seed(_builtins.int(seed))
    except Exception: pass
    return _random_mod
_random_mod.initial_seed = _initial_seed
_random_mod.manual_seed = _manual_seed
_random_mod.seed = lambda: _random_mod._seed
_random_mod.get_rng_state = lambda: _jt.array([0])
_random_mod.set_rng_state = lambda state: None
_random_mod.fork_rng = lambda *a, **k: __import__("contextlib").nullcontext()
sys.modules["torch.random"] = _random_mod
globals()["random"] = _random_mod
# top-level torch.initial_seed / manual_seed / seed
globals()["initial_seed"] = _initial_seed
globals()["manual_seed"] = _manual_seed
globals()["seed"] = lambda: _random_mod._seed
globals()["get_rng_state"] = _random_mod.get_rng_state
globals()["set_rng_state"] = _random_mod.set_rng_state

# torch.version submodule (libs probe .cuda / .hip)
_version = types.ModuleType("torch.version")
_version.__version__ = __version__
_version.cuda = None
_version.hip = None
_version.git_version = "jittor"
sys.modules["torch.version"] = _version
globals()["version"] = _version

# torch._dynamo internals (stubs; transformers guards real dynamo usage)
_dynamo = types.ModuleType("torch._dynamo")
_identity_deco = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda g: g))
_dynamo.disable = _identity_deco
_dynamo.allow_in_graph = _identity_deco
_dynamo.disallow_in_graph = _identity_deco
_dynamo.assume_constant_result = _identity_deco
_dynamo.is_compiling = lambda: False
_dynamo.mark_static_address = lambda *a, **k: None
_dynamo.mark_dynamic = lambda *a, **k: None
_dynamo.graph_break = lambda *a, **k: None
_dynamo.reset = lambda *a, **k: None
class _DynamoConfig: pass
_dynamo.config = _DynamoConfig()
sys.modules["torch._dynamo"] = _dynamo
globals()["_dynamo"] = _dynamo
# torch._dynamo.eval_frame.OptimizedModule (accelerate's is_compiled_module check)
_eval_frame = types.ModuleType("torch._dynamo.eval_frame")
class OptimizedModule:  # nothing is ever an instance -> compiled checks return False
    pass
_eval_frame.OptimizedModule = OptimizedModule
_dynamo.eval_frame = _eval_frame
sys.modules["torch._dynamo.eval_frame"] = _eval_frame
_twh = types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
class TransformGetItemToIndex:
    def __enter__(self): return self
    def __exit__(self, *a): return False
_twh.TransformGetItemToIndex = TransformGetItemToIndex
sys.modules["torch._dynamo._trace_wrapped_higher_order_op"] = _twh

# torch._C internal namespace (some libs probe it) -- minimal stub
_C = types.ModuleType("torch._C")
_C._get_tracing_state = lambda: None
sys.modules["torch._C"] = _C
globals()["_C"] = _C

# torch.utils._pytree already set; add torch.utils.hooks stub
_hooks = types.ModuleType("torch.utils.hooks")
class RemovableHandle:
    def __init__(self, *a, **k): pass
    def remove(self): pass
_hooks.RemovableHandle = RemovableHandle
sys.modules["torch.utils.hooks"] = _hooks

# torch.library (custom op registration) -- pass-through decorators
_library = types.ModuleType("torch.library")
def _custom_op(name=None, *a, **k):
    def deco(fn): return fn
    return deco
_library.custom_op = _custom_op
_library.register_fake = lambda *a, **k: (lambda f: f)
_library.register_kernel = lambda *a, **k: (lambda f: f)
_library.impl = lambda *a, **k: (lambda f: f)
_library.Library = type("Library", (), {"__init__": lambda self, *a, **k: None,
                                          "define": lambda self, *a, **k: None,
                                          "impl": lambda self, *a, **k: None})
_library.get_ctx = lambda: None
_library.register_autograd = lambda *a, **k: (lambda f: f)
_library.register_torch_dispatch = lambda *a, **k: (lambda f: f)
_library.register_vmap = lambda *a, **k: (lambda f: f)
_library.opcheck = lambda *a, **k: None
sys.modules["torch.library"] = _library
globals()["library"] = _library

# torch.amp top-level (autocast / GradScaler)
# autocast must be a context-manager AND a decorator (accelerate does
# `autocast(model_forward)`); reuse torch_compat's _AutocastContext via _jt.autocast.
_amp_mod = types.ModuleType("torch.amp")
_amp_mod.autocast = getattr(_jt, "autocast", lambda *a, **k: contextlib.nullcontext())
# Functional fp16 dynamic loss scaler lives in torch_compat (_GradScaler); reuse it.
_amp_mod.GradScaler = getattr(_jt, "GradScaler", None) or (lambda *a, **k: None)
sys.modules["torch.amp"] = _amp_mod
globals()["amp"] = _amp_mod
globals()["autocast"] = _amp_mod.autocast
if "cuda" in globals():
    globals()["cuda"].amp.GradScaler = _amp_mod.GradScaler

# ---- readable errors: surface jittor's buried [Reason] instead of the cryptic
# "Wrong inputs arguments, help(jt.sync)" / "rerun with JT_SYNC=1" noise ----
def _install_error_clarifier():
    import re as _re
    _prev = sys.excepthook
    def _hook(etype, evalue, tb):
        try:
            msg = str(evalue)
            if any(s in msg for s in ("executor.cc", "Async error", "Wrong inputs arguments", "[Reason]")):
                reason = _re.search(r"\[Reason\]:\s*([^\n]*)", msg)
                optype = _re.search(r"\[OP TYPE\]:\s*([^\n]*)", msg)
                inp = _re.search(r"\[Input\]:\s*([^\n]*)", msg)
                low = msg.lower()
                lines = ["", "=== jittor op error (summary) ==="]
                if optype: lines.append("  op:     " + optype.group(1).strip())
                if inp:    lines.append("  inputs: " + inp.group(1).strip())
                if reason: lines.append("  cause:  " + reason.group(1).strip())
                if "not supported dtype" in low:
                    lines.append("  hint:   this dtype has no NPU kernel; cast to float32/bfloat16.")
                elif "unable to alloc" in low or "out of memory" in low or " alloc " in low:
                    lines.append("  hint:   NPU out of memory; reduce batch size / seqlen / model size.")
                elif "wrong inputs arguments" in low and "[reason]" not in low:
                    lines.append("  hint:   an async op failed earlier; rerun with env JT_SYNC=1 to pinpoint it.")
                lines.append("=== full traceback below ===")
                print("\n".join(lines), file=sys.stderr)
        except Exception:
            pass
        return _prev(etype, evalue, tb)
    sys.excepthook = _hook
try:
    _install_error_clarifier()
except Exception:
    pass

# torch.testing (used in some asserts)
_testing = types.ModuleType("torch.testing")
sys.modules["torch.testing"] = _testing

# torch.fft (peft c3a imports fft/ifft at module load) -- numpy-backed
_fft = types.ModuleType("torch.fft")
def _np_fft_wrap(np_fn):
    def _f(input, n=None, dim=-1, norm=None, **k):
        import numpy as _np
        arr = input.numpy() if hasattr(input, "numpy") else _np.asarray(input)
        res = np_fn(arr, n=n, axis=dim, norm=norm)
        return _jt.array(res.real.astype("float32")) if not _np.iscomplexobj(res) else _jt.array(res.real.astype("float32"))
    return _f
import numpy as _np_for_fft
_fft.fft = _np_fft_wrap(_np_for_fft.fft.fft)
_fft.ifft = _np_fft_wrap(_np_for_fft.fft.ifft)
_fft.rfft = _np_fft_wrap(_np_for_fft.fft.rfft)
_fft.irfft = _np_fft_wrap(_np_for_fft.fft.irfft)
_fft.fft2 = lambda input, *a, **k: input
_fft.ifft2 = lambda input, *a, **k: input
sys.modules["torch.fft"] = _fft
globals()["fft"] = _fft

# torch.profiler (accelerate references ProfilerActivity at class-def time) -- stubs
_profiler = types.ModuleType("torch.profiler")
class ProfilerActivity:
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
class _ProfilerAction:
    def __init__(self, *a, **k): pass
class profile:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def step(self): pass
    def export_chrome_trace(self, *a, **k): pass
    def key_averages(self, *a, **k): return []
class schedule:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return 0
_profiler.ProfilerActivity = ProfilerActivity
_profiler.profile = profile
_profiler.schedule = schedule
_profiler.ProfilerAction = _ProfilerAction
_profiler.tensorboard_trace_handler = lambda *a, **k: (lambda *aa, **kk: None)
_profiler.record_function = lambda *a, **k: contextlib.nullcontext()
sys.modules["torch.profiler"] = _profiler
globals()["profiler"] = _profiler

# torch.jit (scripting/tracing) -- stubs; jittor has no torchscript
_jit = types.ModuleType("torch.jit")
_jit.is_tracing = lambda: False
_jit.is_scripting = lambda: False
_jit.script = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda g: g))
_jit.trace = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda g: g))
_jit.export = lambda f: f
_jit.ignore = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda g: g))
_jit.unused = lambda f: f
_jit._overload_method = lambda f: f
_jit.interface = lambda c: c
_jit.ScriptModule = type("ScriptModule", (), {})
sys.modules["torch.jit"] = _jit
globals()["jit"] = _jit

# ---------------------------------------------------------------------------
# safetensors "pt" backend shim
# ---------------------------------------------------------------------------
# safetensors' framework="pt" path goes through the Rust binding which builds
# real torch C-level storage (torch.UntypedStorage) -- impossible on jittor.
# numpy framework can't represent bf16 either. So we provide a pure-Python
# reader/writer that yields jittor Vars and handles bf16/fp8 by widening to
# float32 on load (downstream casts to the model's param dtype anyway).
def _install_safetensors_shim():
    import json, struct, numpy as _np
    import jittor as _j

    # safetensors dtype string -> (numpy dtype or None for special, itemsize)
    _ST = {
        "F64": (_np.float64, 8), "F32": (_np.float32, 4), "F16": (_np.float16, 2),
        "BF16": (None, 2), "I64": (_np.int64, 8), "I32": (_np.int32, 4),
        "I16": (_np.int16, 2), "I8": (_np.int8, 1), "U8": (_np.uint8, 1),
        "U16": (_np.uint16, 2), "U32": (_np.uint32, 4), "U64": (_np.uint64, 8),
        "BOOL": (_np.bool_, 1), "F8_E4M3": (None, 1), "F8_E5M2": (None, 1),
    }

    def _bytes_to_np(raw, st_dtype, shape):
        npd, _isz = _ST[st_dtype]
        if st_dtype == "BF16":
            u16 = _np.frombuffer(raw, dtype=_np.uint16).astype(_np.uint32)
            f32 = (u16 << 16).view(_np.float32)
            return f32.reshape(shape)
        if st_dtype in ("F8_E4M3", "F8_E5M2"):
            # rare; widen bytes to float32 best-effort (zeros if unsupported)
            return _np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32).reshape(shape)
        return _np.frombuffer(raw, dtype=npd).reshape(shape) if shape else \
            _np.frombuffer(raw, dtype=npd)

    class _PySafeSlice:
        def __init__(self, raw, st_dtype, shape):
            self._raw, self._dtype, self._shape = raw, st_dtype, shape
        def get_shape(self): return list(self._shape)
        def get_dtype(self): return self._dtype
        def __getitem__(self, idx):
            arr = _bytes_to_np(self._raw, self._dtype, self._shape)
            if idx is not Ellipsis and idx != slice(None):
                arr = arr[idx]
            return _j.array(_np.ascontiguousarray(arr))

    class _PySafeOpen:
        def __init__(self, filename, framework="pt", device="cpu", backend="mmap"):
            self._device = device
            with open(filename, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._header = json.loads(fh.read(n).decode("utf-8"))
                self._data = fh.read()
            self._meta = self._header.pop("__metadata__", {})
        def keys(self):
            return list(self._header.keys())
        def metadata(self): return self._meta
        def _entry(self, k):
            e = self._header[k]
            s, t = e["data_offsets"]
            return e["dtype"], e["shape"], self._data[s:t]
        def get_slice(self, k):
            dt, shp, raw = self._entry(k)
            return _PySafeSlice(raw, dt, shp)
        def get_tensor(self, k):
            dt, shp, raw = self._entry(k)
            return _j.array(_np.ascontiguousarray(_bytes_to_np(raw, dt, shp)))
        def get_dtype(self, k): return self._header[k]["dtype"]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _load_bytes(data):
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8+n].decode("utf-8"))
        header.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for k, e in header.items():
            s, t = e["data_offsets"]
            arr = _bytes_to_np(data[base+s:base+t], e["dtype"], e["shape"])
            out[k] = _j.array(_np.ascontiguousarray(arr))
        return out

    def _load_file(filename, device="cpu"):
        with _PySafeOpen(filename) as f:
            return {k: f.get_tensor(k) for k in f.keys()}

    _NP_TO_ST = {"float64": "F64", "float32": "F32", "float16": "F16",
                 "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
                 "uint8": "U8", "bool": "BOOL", "bfloat16": "BF16"}

    def _save_dict(tensors, metadata=None):
        header = {}
        blobs = []
        off = 0
        for k, v in tensors.items():
            arr = v.numpy() if hasattr(v, "numpy") else _np.asarray(v)
            arr = _np.ascontiguousarray(arr)
            st = _NP_TO_ST.get(str(arr.dtype), "F32")
            if st not in _ST or _ST[st][0] is None:
                arr = arr.astype(_np.float32); st = "F32"
            b = arr.tobytes()
            header[k] = {"dtype": st, "shape": list(arr.shape),
                         "data_offsets": [off, off+len(b)]}
            blobs.append(b); off += len(b)
        if metadata:
            header["__metadata__"] = {str(a): str(b) for a, b in metadata.items()}
        hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
        return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)

    def _save_file(tensors, filename, metadata=None):
        with open(filename, "wb") as fh:
            fh.write(_save_dict(tensors, metadata))

    import safetensors as _st
    _st.safe_open = _PySafeOpen
    sys.modules["safetensors"].safe_open = _PySafeOpen
    try:
        import safetensors.torch as _stt
        _stt.safe_open = _PySafeOpen
        _stt.load = _load_bytes
        _stt.load_file = _load_file
        _stt.save = lambda tensors, metadata=None: _save_dict(tensors, metadata)
        _stt.save_file = _save_file
    except Exception:
        pass
    try:
        import safetensors.numpy as _stn
        _stn.load_file = _load_file
        _stn.save_file = _save_file
    except Exception:
        pass

try:
    _install_safetensors_shim()
except Exception as _e:
    print("[torch-shim] safetensors shim not installed:", _e)
