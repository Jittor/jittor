"""Shim package: `import torch` -> jittor (with torch_compat layer).

Lets torch-targeted libraries (transformers, LlamaFactory, ...) run on jittor
unmodified. jittor's torch_compat layer supplies the torch-style API on the
jittor module; this package re-exports it as `torch` and wires up the common
`torch.<submodule>` paths.
"""
import sys, types, contextlib
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
    nn.init = _init
    sys.modules["torch.nn.init"] = _init
except Exception:
    pass

# nn.parameter.Parameter -> jittor Var (with requires_grad)
def Parameter(data, requires_grad=True):
    v = data if isinstance(data, _jt.Var) else _jt.array(data)
    v = v.clone()
    if requires_grad:
        v.requires_grad = True
    return v
nn.Parameter = Parameter
_param_mod = types.ModuleType("torch.nn.parameter")
_param_mod.Parameter = Parameter
sys.modules["torch.nn.parameter"] = _param_mod

# nn.modules.module.Module -> jittor Module
_mod_module = types.ModuleType("torch.nn.modules.module")
_mod_module.Module = nn.Module
sys.modules["torch.nn.modules"] = types.ModuleType("torch.nn.modules")
sys.modules["torch.nn.modules"].Module = nn.Module
sys.modules["torch.nn.modules.module"] = _mod_module

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

# ---- cuda ----
if hasattr(_jittor, "cuda"):
    sys.modules["torch.cuda"] = _jittor.cuda

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
try:
    import jittor.lr_scheduler as _lrs
    _optim.lr_scheduler = _lrs
    sys.modules["torch.optim.lr_scheduler"] = _lrs
except Exception:
    pass

# ---- utils.data ----
_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = _utils
try:
    import jittor.dataset as _ds
    _data = types.ModuleType("torch.utils.data")
    _data.Dataset = getattr(_ds, "Dataset", object)
    _data.DataLoader = getattr(_ds, "DataLoader", getattr(_ds, "Dataset", object))
    class _IterableDataset: pass
    _data.IterableDataset = _IterableDataset
    _data.TensorDataset = getattr(_ds, "TensorDataset", object)
    _data.RandomSampler = getattr(_ds, "RandomSampler", object)
    _data.Sampler = getattr(_ds, "Sampler", object)
    _data.SequentialSampler = getattr(_ds, "SequentialSampler", object)
    sys.modules["torch.utils.data"] = _data
    _utils.data = _data
    _distm = types.ModuleType("torch.utils.data.distributed")
    _distm.DistributedSampler = getattr(_ds, "Sampler", object)
    sys.modules["torch.utils.data.distributed"] = _distm
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
_amp_mod = types.ModuleType("torch.amp")
_amp_mod.autocast = lambda *a, **k: contextlib.nullcontext()
class _GradScaler:
    def __init__(self, *a, **k): pass
    def scale(self, loss): return loss
    def step(self, opt): return opt.step() if hasattr(opt, "step") else None
    def update(self, *a, **k): pass
    def unscale_(self, *a, **k): pass
    def get_scale(self): return 1.0
_amp_mod.GradScaler = _GradScaler
sys.modules["torch.amp"] = _amp_mod
globals()["amp"] = _amp_mod
globals()["autocast"] = _amp_mod.autocast
if "cuda" in globals():
    globals()["cuda"].amp.GradScaler = _GradScaler

# torch.testing (used in some asserts)
_testing = types.ModuleType("torch.testing")
sys.modules["torch.testing"] = _testing

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
