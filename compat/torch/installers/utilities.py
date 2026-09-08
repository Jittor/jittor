"""Module-level utility APIs; installation only publishes their namespace."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
import numpy as np

from ..context import get_install_context, registry_for
from ..fidelity import Fidelity, register_fidelity
from .._placeholder_context import _PlaceholderContext

from ..grad import (
    _amp_passthrough_decorator, _AutocastContext,
    _GradScaler,
)
from ..library import Tag
from ..nested import (
    _torch_make_parameter,
)
from ...diagnostics import EXPECTED, swallowed


def _install_external_adapter(ctx, entry_point):
    from ...module_patcher import install_module_patches
    from ...transaction import active_transaction
    return install_module_patches(
        transaction=active_transaction(ctx),
        expected_entry_points=(entry_point,),
    )


def _install_flash_attn_shim(registry=None):
    """Register the Jittor-backed flash_attn stub for the bare jittor path."""
    _modules = registry_for(jt, registry).module_map
    import importlib.util as _ilu
    import os as _os
    torch_mod = _modules.get("torch")
    if torch_mod is not None and torch_mod is not jt:
        # When the deployed `import torch` shim imports jittor, the `torch`
        # package body is still half-built. The optional flash_attn stub imports
        # torch.nn.functional, so importing it at that point can abort the core
        # torch_compat install. deploy.py installs a normal flash_attn package
        # for that path; only register this direct stub for `import jittor as
        # torch` flows.
        return
    mod = _modules.get("flash_attn")
    if mod is not None and getattr(mod, "_jittor_flash_attn_stub", False):
        return
    from jittor.compat.shim import deploy as _shim_deploy

    src = _os.path.join(
        _shim_deploy.resources_root(),
        "stubs",
        "flash_attn",
        "__init__.py",
    )
    if not _os.path.isfile(src):
        return
    spec = _ilu.spec_from_file_location("flash_attn", src)
    shim = _ilu.module_from_spec(spec)
    old_flash = _modules.get("flash_attn")
    _modules["flash_attn"] = shim
    _modules["torch"] = jt
    try:
        spec.loader.exec_module(shim)
    except EXPECTED as exc:
        swallowed("torch/installers/utilities.py _install_flash_attn_shim: spec.loader.exec_module(shim)", exc)
        if old_flash is None:
            _modules.pop("flash_attn", None)
        else:
            _modules["flash_attn"] = old_flash
        raise
    shim._jittor_flash_attn_stub = True


import contextlib as _ctxlib_utils
import os
import os as _os_hub
import types as _types2
import urllib.request as _urlreq_hub
from urllib.parse import urlparse as _urlparse_hub


def _utilities_context():
    return get_install_context(jt)


class _NodeDef:
    pass


class FakeTensor:
    """Annotation placeholder; no fake execution semantics."""


class FunctionalTensor:
    """Annotation placeholder; no functionalization semantics."""


class TorchDispatchMode:
    """Annotation placeholder; native dispatch does not consult this mode."""


class FakeTensorMode(_PlaceholderContext):
    """Import-compatible mode placeholder; no fake-tensor dispatch is installed."""


def _profile_schedule_action(step):
    return _ProfilerAction.NONE


def _profile_trace_handler(*args, **kwargs):
    return None

def _flatten_dense_tensors(tensors):
    tensors = list(tensors)
    if len(tensors) == 1:
        return tensors[0].reshape(-1).clone()
    return jt.concat([t.reshape(-1) for t in tensors]) if tensors else jt.array([])


def _unflatten_dense_tensors(flat, tensors):
    outputs, offset = [], 0
    for t in tensors:
        n = 1
        for s in t.shape:
            n *= int(s)
        outputs.append(flat[offset:offset + n].reshape(t.shape))
        offset += n
    return outputs


def _take_tensors(tensors, size_limit):
    buckets = {}
    for t in tensors:
        key = _jittor_dtype_name(getattr(t, "dtype", "object"))
        b = buckets.setdefault(key, [[], 0])
        n = int(t.numel()) if hasattr(t, "numel") else 1
        b[0].append(t)
        b[1] += n * 4
        if b[1] >= size_limit:
            yield b[0]
            buckets[key] = [[], 0]
    for b in buckets.values():
        if b[0]:
            yield b[0]


def _get_available_device_type():
    if hasattr(_utilities_context().target_namespace, "cuda") and _utilities_context().target_namespace.cuda.is_available():
        return "cuda"
    if hasattr(_utilities_context().target_namespace, "npu") and _utilities_context().target_namespace.npu.is_available():
        return "npu"
    if hasattr(_utilities_context().target_namespace, "mps") and _utilities_context().target_namespace.mps.is_available():
        return "mps"
    return None


def _get_device_module(device_type):
    if device_type is None:
        return None
    return getattr(_utilities_context().target_namespace, str(device_type), None)


def _hub_dir():
    return _os_hub.path.expanduser(_os_hub.environ.get("TORCH_HOME", "~/.cache/torch"))


def _hub_checkpoints_dir():
    path = _os_hub.path.join(_hub_dir(), "hub", "checkpoints")
    _os_hub.makedirs(path, exist_ok=True)
    return path


def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    _os_hub.makedirs(_os_hub.path.dirname(_os_hub.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".partial"
    if _os_hub.path.exists(tmp):
        try:
            _os_hub.remove(tmp)
        except OSError as exc:
            swallowed("torch/installers/utilities.py _download_url_to_file: _os_hub.remove(tmp)", exc)
    _urlreq_hub.urlretrieve(url, tmp)
    if (not _os_hub.path.isfile(tmp)) or _os_hub.path.getsize(tmp) == 0:
        try:
            _os_hub.remove(tmp)
        except OSError as exc:
            swallowed("torch/installers/utilities.py _download_url_to_file: _os_hub.remove(tmp)", exc)
        raise RuntimeError(f"downloaded empty checkpoint from {url}")
    _os_hub.replace(tmp, dst)


def _load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True,
                              check_hash=False, file_name=None, weights_only=False):
    if model_dir is None:
        model_dir = _hub_checkpoints_dir()
    _os_hub.makedirs(model_dir, exist_ok=True)
    filename = file_name or _os_hub.path.basename(_urlparse_hub(url).path)
    cached_file = _os_hub.path.join(model_dir, filename)
    if (not _os_hub.path.isfile(cached_file)) or _os_hub.path.getsize(cached_file) == 0:
        if _os_hub.path.exists(cached_file):
            try:
                _os_hub.remove(cached_file)
            except OSError as exc:
                swallowed("torch/installers/utilities.py _load_state_dict_from_url: _os_hub.remove(cached_file)", exc)
        _download_url_to_file(url, cached_file, progress=progress)
    return _utilities_context().target_namespace.load(cached_file, map_location=map_location, weights_only=weights_only)


class ProfilerActivity:
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
    HPU = "hpu"
    MTIA = "mtia"


class _ProfilerAction:
    NONE = "none"
    WARMUP = "warmup"
    RECORD = "record"
    RECORD_AND_SAVE = "record_and_save"


class _ProfileContext:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def step(self):
        pass
    def export_chrome_trace(self, *args, **kwargs):
        pass


def _real_summary_writer():
    """The genuine tensorboard writer, if one is installed.

    Only tensorboardX is probed: `torch.utils.tensorboard` IS this
    module under the shim, so importing it back would be circular.
    """
    try:
        from tensorboardX import SummaryWriter as _RealWriter
        return _RealWriter
    except EXPECTED as exc:
        swallowed("torch/installers/utilities.py _real_summary_writer: from tensorboardX import SummaryWriter as _RealWriter", exc)
        return None


class SummaryWriter:
    """torch.utils.tensorboard.SummaryWriter.

    Every add_* method used to return None and write nothing, and this
    branch was always taken -- so a training run's entire logging
    output silently vanished while the script reported success.

    Now: delegate to a real writer when one is installed, otherwise
    refuse at construction. Nothing here pretends to log.
    """

    def __new__(cls, *args, **kwargs):
        real = _real_summary_writer()
        if real is not None:
            return real(*args, **kwargs)
        return object.__new__(cls)

    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix="", *args, **kwargs):
        from ...stub_policy import unimplemented
        unimplemented(
            "torch.utils.tensorboard.SummaryWriter",
            "accept every add_scalar/add_image/add_graph call and write "
            "nothing, silently discarding the whole training log",
            "Install tensorboardX (or tensorboard) to get a real "
            "writer.")
        self.log_dir = log_dir
        self.comment = comment
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix
        self.args = args
        self.kwargs = kwargs
    def add_scalar(self, *a, **k): return None
    def add_scalars(self, *a, **k): return None
    def add_image(self, *a, **k): return None
    def add_images(self, *a, **k): return None
    def add_graph(self, *a, **k): return None
    def add_histogram(self, *a, **k): return None
    def add_text(self, *a, **k): return None
    def flush(self): return None
    def close(self): return None
    def __enter__(self): return self
    def __exit__(self, *exc): self.close(); return False


class _UtilsNS(_types2.ModuleType):
    def __getattr__(self, name):
        full = "torch.utils." + name
        if full in _utilities_context().registry.module_map:
            return _utilities_context().registry.module_map[full]
        raise AttributeError(name)


class LeafSpec:
    pass


class TreeSpec:
    def __init__(self, type, context, children_specs):
        self.type = type
        self.context = context
        self.children_specs = list(children_specs)


class MappingKey:
    def __init__(self, key):
        self.key = key
    def __hash__(self):
        return hash(self.key)
    def __eq__(self, other):
        return isinstance(other, MappingKey) and self.key == other.key
    def __repr__(self):
        return f"[{self.key!r}]"


class SequenceKey:
    def __init__(self, idx):
        self.idx = idx
    def __hash__(self):
        return hash(self.idx)
    def __eq__(self, other):
        return isinstance(other, SequenceKey) and self.idx == other.idx
    def __repr__(self):
        return f"[{self.idx}]"


class GetAttrKey:
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return isinstance(other, GetAttrKey) and self.name == other.name
    def __repr__(self):
        return "." + str(self.name)


def _list_flatten(x):
    return list(x), None


def _list_unflatten(values, context):
    return list(values)


def _list_flatten_with_keys(x):
    return [(i, v) for i, v in enumerate(x)], None


def _tuple_flatten(x):
    return list(x), None


def _dict_flatten(x):
    keys = list(x.keys())
    return [x[k] for k in keys], keys


def _dict_unflatten(values, context):
    return {k: v for k, v in zip(context, values)}


def _get_node_type(x):
    return dict if isinstance(x, dict) else type(x)


def _tree_flatten(x):
    leaves = []
    def rec(o):
        node_type = _get_node_type(o)
        if node_type not in _utilities_context().state["pytree_nodes"]:
            leaves.append(o)
            return LeafSpec()
        child_pytrees, context = _utilities_context().state["pytree_nodes"][node_type].flatten_fn(o)
        child_specs = [rec(c) for c in child_pytrees]
        return TreeSpec(node_type, context, child_specs)
    return leaves, rec(x)


def _tree_unflatten(leaves, spec):
    it = iter(leaves)
    def rec(s):
        if isinstance(s, LeafSpec):
            return next(it)
        children = [rec(c) for c in s.children_specs]
        if s.type is tuple:
            return tuple(children)
        if s.type is dict:
            return {k: v for k, v in zip(s.context, children)}
        return list(children)
    return rec(spec)


def _tree_map(f, x, *rests):
    # This used to be `lambda f, x: f(x)` -- no recursion at all, right
    # next to a real recursive _tree_flatten. So the standard
    # `tree_map_only(Tensor, lambda t: t.to(dev), batch)` returned the
    # batch unchanged: nothing moved to the device, and nothing failed.
    leaves, spec = _tree_flatten(x)
    if not rests:
        return _tree_unflatten([f(leaf) for leaf in leaves], spec)
    rest_leaves = [_tree_flatten(r)[0] for r in rests]
    mapped = [f(leaf, *[rl[i] for rl in rest_leaves])
              for i, leaf in enumerate(leaves)]
    return _tree_unflatten(mapped, spec)


def _tree_map_only(typ, f, x, *rests):
    def _apply(leaf, *others):
        return f(leaf, *others) if isinstance(leaf, typ) else leaf
    return _tree_map(_apply, x, *rests)


class _DecoratorContextManager(_ctxlib_utils.ContextDecorator):
    def clone(self):
        return type(self)()
    def __call__(self, orig_func):
        return super().__call__(orig_func)


class RemovableHandle:
    def __init__(self, hooks_dict=None, *args, **kwargs):
        self.hooks_dict = hooks_dict
        try:
            self.id = max(hooks_dict.keys(), default=0) + 1 if hooks_dict is not None else 0
        except EXPECTED as exc:
            swallowed("torch/installers/utilities.py __init__: self.id = max(hooks_dict.keys(), default=0) + 1 if hook...", exc)
            self.id = 0
    def remove(self):
        try:
            if self.hooks_dict is not None:
                self.hooks_dict.pop(self.id, None)
        except EXPECTED as exc:
            swallowed("torch/installers/utilities.py remove: if self.hooks_dict is not None:", exc)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.remove()
        return False


def _dlpack_not_implemented(*args, **kwargs):
    raise NotImplementedError("torch.utils.dlpack is not implemented by jittor torch_compat")


class FlopCounterMode:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_total_flops(self):
        return 0

    def get_flop_counts(self):
        return {}


def assert_close(actual, expected, rtol=1e-5, atol=1e-8, **kwargs):
    import numpy as np

    actual = actual.numpy() if hasattr(actual, "numpy") else actual
    expected = expected.numpy() if hasattr(expected, "numpy") else expected
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def load_url(*args, **kwargs):
    raise NotImplementedError(
        "torch.utils.model_zoo.load_url is not supported on the Jittor shim"
    )


def _api_tutils__rebuild_tensor(data, *a, **k):
    return data


def _api_tutils__rebuild_tensor_v2(data, *a, **k):
    return data


def _api_tutils__rebuild_parameter(data, requires_grad=True, *a, **k):
    return _torch_make_parameter(data, requires_grad)


def _api_tutils__rebuild_parameter_with_state(data, requires_grad=True, backward_hooks=None, state=None):
    return _torch_make_parameter(data, requires_grad)


def _api_hub_get_dir():
    return _os_hub.path.join(_hub_dir(), 'hub')


def _api_profiler_profile(*args, **kwargs):
    return _ProfileContext()


def _api_profiler_schedule(*args, **kwargs):
    return _profile_schedule_action


def _api_profiler_tensorboard_trace_handler(*args, **kwargs):
    return _profile_trace_handler


def _api_profiler_record_function(*args, **kwargs):
    return _ProfileContext()


def _api_profiler_kineto_available():
    return False


def _api_g_compiled_with_cxx11_abi():
    from jittor.compat.shim import cpp_extension as _cpp_extension
    return bool(_cpp_extension.CXX11_ABI)


def _api_pytree_tree_all(pred, x):
    return all((pred(l) for l in _tree_flatten(x)[0]))


def _api_pytree_tree_any(pred, x):
    return any((pred(l) for l in _tree_flatten(x)[0]))


def _api_pytree_tree_leaves(x):
    return _tree_flatten(x)[0]


def _api_pytree_register_pytree_node(*a, **k):
    return None


def _api_pytree__register_pytree_node(*a, **k):
    return None


def _api_fake_tensor_unset_fake_temporarily():
    return _ctxlib_utils.nullcontext()


def _api_python_dispatch__get_current_dispatch_mode(*args, **kwargs):
    return None


def _api_g_get_num_threads():
    return os.cpu_count() or 1


def _api_g_set_num_threads(*args, **kwargs):
    return None


def _api_g_get_num_interop_threads():
    return os.cpu_count() or 1


def _api_g_set_num_interop_threads(*args, **kwargs):
    return None


_UTILITY_PLACEHOLDERS = frozenset((
    FakeTensor, FakeTensorMode, FunctionalTensor, TorchDispatchMode,
    _ProfileContext, _api_profiler_profile, _api_profiler_schedule,
    _api_profiler_tensorboard_trace_handler, _api_profiler_record_function,
    FlopCounterMode, _api_pytree_register_pytree_node, _api_pytree__register_pytree_node,
    _api_fake_tensor_unset_fake_temporarily, _api_python_dispatch__get_current_dispatch_mode,
    _api_g_set_num_threads, _api_g_set_num_interop_threads,
    _dlpack_not_implemented, load_url,
))


def _register_utility_fidelity(ctx):
    namespaces = [("torch", ctx.target_namespace)]
    namespaces.extend((name, module) for name, module in ctx.registry.module_map.items()
                      if name in ("torch._utils", "torch.hub", "torch.profiler", "torch.testing")
                      or name.startswith(("torch.utils", "torch._subclasses")))
    for namespace, module in namespaces:
        for name, implementation in tuple(vars(module).items()):
            if not callable(implementation) or getattr(implementation, "__module__", None) != __name__:
                continue
            placeholder = implementation in _UTILITY_PLACEHOLDERS
            detail = ("Unsupported placeholder; existing no-op or explicit refusal is retained."
                      if placeholder else
                      "Existing utility behavior retained; restricted arguments and external dependency behavior may differ from PyTorch.")
            if implementation in (_download_url_to_file, _load_state_dict_from_url):
                detail = "Download/cache facade; hash-prefix/check_hash verification is not implemented."
            register_fidelity(namespace + "." + name, implementation,
                              Fidelity.UNIMPLEMENTED if placeholder else Fidelity.APPROXIMATE, detail)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    # ---- torch._utils ----
    _tutils = _types2.ModuleType("torch._utils")
    _tutils._flatten_dense_tensors = _flatten_dense_tensors
    _tutils._unflatten_dense_tensors = _unflatten_dense_tensors
    _tutils._take_tensors = _take_tensors
    _tutils._get_available_device_type = _get_available_device_type
    _tutils._get_device_module = _get_device_module
    _tutils._rebuild_tensor = _api_tutils__rebuild_tensor
    _tutils._rebuild_tensor_v2 = _api_tutils__rebuild_tensor_v2
    _tutils._rebuild_parameter = _api_tutils__rebuild_parameter
    _tutils._rebuild_parameter_with_state = _api_tutils__rebuild_parameter_with_state
    _modules["torch._utils"] = _tutils
    g._utils = _tutils

    # ---- torch.hub ----
    import types as _types_hub, os as _os_hub, urllib.request as _urlreq_hub
    from urllib.parse import urlparse as _urlparse_hub
    hub = _types_hub.ModuleType("torch.hub")
    hub.download_url_to_file = _download_url_to_file
    hub.load_state_dict_from_url = _load_state_dict_from_url
    hub.get_dir = _api_hub_get_dir
    import re as _re_hub
    hub.HASH_REGEX = _re_hub.compile(r"-([a-f0-9]*)\\.")
    hub.tqdm = None
    hub.urlparse = _urlparse_hub
    hub.urlopen = _urlreq_hub.urlopen
    hub.Request = _urlreq_hub.Request
    hub._get_torch_home = hub.get_dir
    g.hub = hub
    _modules.setdefault("torch.hub", hub)

    # torch.profiler: accelerate/transformers reference this namespace at
    # import-time for type annotations and optional profiling config. Do not
    # expose jittor_core.profiler here; it lacks PyTorch's ProfilerActivity API.
    _profiler = _types2.ModuleType("torch.profiler")
    _profiler.ProfilerActivity = ProfilerActivity
    _profiler.ProfilerAction = _ProfilerAction
    _profiler.profile = _api_profiler_profile
    _profiler.schedule = _api_profiler_schedule
    _profiler.tensorboard_trace_handler = _api_profiler_tensorboard_trace_handler
    _profiler.record_function = _api_profiler_record_function
    _profiler.kineto_available = _api_profiler_kineto_available
    _modules["torch.profiler"] = _profiler
    g.profiler = _profiler

    if "torch.utils.tensorboard" not in _modules:
        _tb = _types2.ModuleType("torch.utils.tensorboard")


        _tb.SummaryWriter = SummaryWriter
        _modules["torch.utils.tensorboard"] = _tb

    _amp = _types2.ModuleType("torch.amp")
    _amp.autocast = _AutocastContext
    _amp.GradScaler = _GradScaler
    _amp.custom_fwd = _amp_passthrough_decorator
    _amp.custom_bwd = _amp_passthrough_decorator
    _modules["torch.amp"] = _amp
    g.amp = _amp
    try:
        if hasattr(g, "cuda"):
            if not hasattr(g.cuda, "amp"):
                g.cuda.amp = _types2.ModuleType("torch.cuda.amp")
            g.cuda.amp.autocast = _amp.autocast
            g.cuda.amp.GradScaler = _GradScaler
            g.cuda.amp.custom_fwd = _amp_passthrough_decorator
            g.cuda.amp.custom_bwd = _amp_passthrough_decorator
            _modules["torch.cuda.amp"] = g.cuda.amp
    except EXPECTED as exc:
        swallowed("torch/installers/utilities.py install: if hasattr(g, 'cuda'):", exc)

    # `import jittor as torch; torch.utils.data.Dataset` (attribute access, used by some
    # HF/training code as a base class) needs a `utils` namespace on the jittor module --
    # the `from torch.utils.data import X` form already resolves via _modules. Lazily
    # resolve torch.utils.<sub> (data/checkpoint/rnn/...) on access.
    if not hasattr(g, "utils") or not isinstance(getattr(g, "utils"), _types2.ModuleType):
        g.utils = _UtilsNS("torch.utils")
    g.utils.__path__ = []
    g.utils.__package__ = "torch"
    _modules["torch.utils"] = g.utils
    if "torch.utils.tensorboard" in _modules:
        g.utils.tensorboard = _modules["torch.utils.tensorboard"]
    if "torch.utils._pytree" not in _modules:
        _pytree = _types2.ModuleType("torch.utils._pytree")
        SUPPORTED_NODES = {
            list: _NodeDef(),
            tuple: _NodeDef(),
            dict: _NodeDef(),
        }
        SUPPORTED_NODES[list].flatten_fn = _list_flatten
        SUPPORTED_NODES[tuple].flatten_fn = _tuple_flatten
        SUPPORTED_NODES[dict].flatten_fn = _dict_flatten
        ctx.state["pytree_nodes"] = SUPPORTED_NODES
        _pytree.SUPPORTED_NODES = SUPPORTED_NODES
        _pytree.LeafSpec = LeafSpec
        _pytree.TreeSpec = TreeSpec
        _pytree.PyTree = object
        _pytree.Context = object
        _pytree.MappingKey = MappingKey
        _pytree.SequenceKey = SequenceKey
        _pytree.GetAttrKey = GetAttrKey
        _pytree.KeyEntry = (MappingKey, SequenceKey, GetAttrKey)
        _pytree.FlattenFunc = object
        _pytree.UnflattenFunc = object
        _pytree._get_node_type = _get_node_type
        _pytree._list_flatten = _list_flatten
        _pytree._list_unflatten = _list_unflatten
        _pytree._list_flatten_with_keys = _list_flatten_with_keys
        _pytree._dict_flatten = _dict_flatten
        _pytree._dict_unflatten = _dict_unflatten
        _pytree.tree_flatten = _tree_flatten
        _pytree.tree_unflatten = _tree_unflatten


        _pytree.tree_map = _tree_map
        _pytree.tree_map_only = _tree_map_only
        _pytree.tree_map_ = _tree_map
        _pytree.tree_all = _api_pytree_tree_all
        _pytree.tree_any = _api_pytree_tree_any
        _pytree.tree_leaves = _api_pytree_tree_leaves
        _pytree.register_pytree_node = _api_pytree_register_pytree_node
        _pytree._register_pytree_node = _api_pytree__register_pytree_node
        _modules["torch.utils._pytree"] = _pytree
    g.utils._pytree = _modules["torch.utils._pytree"]
    if "torch.utils._contextlib" not in _modules:
        _contextlib_mod = _types2.ModuleType("torch.utils._contextlib")
        import contextlib as _ctxlib_utils
        _contextlib_mod._DecoratorContextManager = _DecoratorContextManager
        _modules["torch.utils._contextlib"] = _contextlib_mod
    g.utils._contextlib = _modules["torch.utils._contextlib"]
    if "torch.utils.hooks" not in _modules:
        _hooks = _types2.ModuleType("torch.utils.hooks")
        _hooks.RemovableHandle = RemovableHandle
        _modules["torch.utils.hooks"] = _hooks
    g.utils.hooks = _modules["torch.utils.hooks"]
    if "torch.utils.dlpack" not in _modules:
        _dlpack = _types2.ModuleType("torch.utils.dlpack")
        _dlpack.from_dlpack = _dlpack_not_implemented
        _dlpack.to_dlpack = _dlpack_not_implemented
        _modules["torch.utils.dlpack"] = _dlpack
    g.utils.dlpack = _modules["torch.utils.dlpack"]
    if "torch._subclasses.fake_tensor" not in _modules:
        _subclasses = _types2.ModuleType("torch._subclasses")
        _fake_tensor = _types2.ModuleType("torch._subclasses.fake_tensor")
        _functional_tensor = _types2.ModuleType("torch._subclasses.functional_tensor")
        _fake_tensor.FakeTensor = FakeTensor
        _fake_tensor.FakeTensorMode = FakeTensorMode
        _fake_tensor.unset_fake_temporarily = (
            _api_fake_tensor_unset_fake_temporarily
        )
        _functional_tensor.FunctionalTensor = FunctionalTensor
        _subclasses.fake_tensor = _fake_tensor
        _subclasses.functional_tensor = _functional_tensor
        # torch re-exports these from the package itself, and code imports them
        # from whichever of the two spellings it was written against.
        _subclasses.FakeTensor = _fake_tensor.FakeTensor
        _subclasses.FakeTensorMode = _fake_tensor.FakeTensorMode
        _subclasses.FunctionalTensor = _functional_tensor.FunctionalTensor
        _modules["torch._subclasses"] = _subclasses
        _modules["torch._subclasses.fake_tensor"] = _fake_tensor
        _modules["torch._subclasses.functional_tensor"] = _functional_tensor
        setattr(g, "_subclasses", _subclasses)
    if "torch.utils.flop_counter" not in _modules:
        _flop_counter = _types2.ModuleType("torch.utils.flop_counter")
        _flop_counter.FlopCounterMode = FlopCounterMode
        _modules["torch.utils.flop_counter"] = _flop_counter
    g.utils.flop_counter = _modules["torch.utils.flop_counter"]
    from jittor.compat.shim import cpp_extension as _cpp_extension
    from jittor.compat.shim.cpp_extension.torch_utils import install_cpp_extension

    g.compiled_with_cxx11_abi = _api_g_compiled_with_cxx11_abi
    g._C._GLIBCXX_USE_CXX11_ABI = bool(_cpp_extension.CXX11_ABI)
    install_cpp_extension(g.utils, registry=ctx.registry)
    _register_utility_fidelity(ctx)


def install_torchmetrics(ctx):
    return _install_external_adapter(ctx, "jittor_torchmetrics")


def install_transformers(ctx):
    return _install_external_adapter(ctx, "jittor_transformers")


def install_flash(ctx):
    _install_flash_attn_shim(ctx.registry)


def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    module = registry.ensure
    testing = module("torch.testing")
    if not hasattr(testing, "assert_close"):

        testing.assert_close = assert_close
    g.testing = testing

    model_zoo = module("torch.utils.model_zoo")
    if not hasattr(model_zoo, "load_url"):

        model_zoo.load_url = load_url
    g.utils.model_zoo = model_zoo

    python_dispatch = module("torch.utils._python_dispatch")
    python_dispatch.TorchDispatchMode = getattr(
        python_dispatch, "TorchDispatchMode", TorchDispatchMode
    )
    python_dispatch._get_current_dispatch_mode = _api_python_dispatch__get_current_dispatch_mode
    g.utils._python_dispatch = python_dispatch
    _register_utility_fidelity(ctx)


def install_runtime_knobs(ctx):
    """Thread-count knobs and the canonical torch.Tag enum.

    Jittor sizes its own thread pools, so the setters are accepted and ignored
    while the getters report the machine's CPU count -- code that reads one back
    to size a work queue then gets a sane number rather than an AttributeError.
    torch.Tag is published here before the compiler-family installer reaches
    torch.library; both paths use the same canonical enum object.
    """
    import os

    g = ctx.jittor_module

    if not hasattr(g, "get_num_threads"):
        g.get_num_threads = _api_g_get_num_threads
    if not hasattr(g, "set_num_threads"):
        g.set_num_threads = _api_g_set_num_threads
    if not hasattr(g, "get_num_interop_threads"):
        g.get_num_interop_threads = _api_g_get_num_interop_threads
    if not hasattr(g, "set_num_interop_threads"):
        g.set_num_interop_threads = _api_g_set_num_interop_threads

    g.Tag = Tag
    _register_utility_fidelity(ctx)
