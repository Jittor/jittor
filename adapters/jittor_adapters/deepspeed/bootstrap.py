"""Decline optional Torch-private facilities, keeping DeepSpeed's real Engine.

Pinned ZeRO utility methods are replaced only for verified Jittor CUDA paths.
DeepCompile needs private Inductor/SymPy modules; elastic launch needs Torch's
elastic agent; the optional CPU shared-memory extension requires binary Torch.
The upstream public collective path remains responsible for communication.
"""
import ast
from functools import wraps
import hashlib
import importlib.abc
import importlib.machinery
import inspect
from pathlib import Path
import sys

import jittor as jt
from jittor.compat.transaction import (
    TransactionConflict, active_transaction, owned_runtime_hook, runtime_hook,
)
from .._common import (
    UnsupportedAdapterVersion, replace, replace_bound_aliases, require_version,
)
from ._finite import make_has_inf_or_nan
from ._norm import make_global_norm
from ._zero import make_zero_optimizer_methods

SUPPORTED_VERSIONS = frozenset(("0.19.7",))
UPSTREAM_COMMIT = "f5af15c2b4da10411e340850c00ba0cb5b14fd87"
SOURCE_SHA256 = {
    "deepspeed.compile": "f838aeb1b3edfc152d1c80e3c55b4c88b12e14351be80366ebbd6a7cbb945d56",
    "deepspeed.runtime.engine": "2ec177075887663b31be6e14cf1d519e26fcb98fff2524b8160197bcc45ab216",
    "deepspeed.comm.torch": "5a926f19355689ae145b976f831e2e8c066d64bd8949a2ad3b80a6876a021451",
    "deepspeed.elasticity.utils": "fa66cffd55fe492b2d64aac4a748641707b3aa5f6990012fc71dc1c4c3b6fa27",
    "deepspeed.utils.torch": "3482342c7dbbb1d2aade2a937ea0e93d51dbf12371c014ba534beee34f66dc46",
    "deepspeed.runtime.utils": "7abe78ded0be93f1ac41bc6cfde6035ac594cf8857784d96027e910236db9c67",
    "deepspeed.runtime.zero.stage_1_and_2": "f878e5677ab1da63e905e221e0a4d6f7e8f06851adbc03f016b80bb4dd315a7f",
}
_OWNER = "deepspeed.activation"


class _Unavailable:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(self.name + " requires unsupported DeepCompile")

    def __getattr__(self, name):
        raise NotImplementedError(self.name + "." + name + " requires unsupported DeepCompile")

    def __getitem__(self, name):
        raise NotImplementedError(self.name + " requires unsupported DeepCompile")


def _unavailable_capability():
    return False


def _unavailable_shm():
    # None is the upstream capability protocol, selecting its actual
    # torch.distributed path instead of a PyTorch-ABI shared-memory extension.
    return None


def _register_grad_hook(parameter, hook):
    # Upstream branches on torch.__version__, selecting a private
    # AccumulateGrad path on older releases. Select the actual public API
    # capability instead; the framework version is never rewritten.
    return parameter.register_post_accumulate_grad_hook(hook)


def _count_used_parameters(parameters):
    from jittor.compat.torch.gradient_task import participating_tensors
    return len(participating_tensors(parameters))


def _has_participation_query():
    from jittor.compat.torch.gradient_task import participating_tensors
    return callable(participating_tensors)


def _cuda_norm_allowed(capability):
    cuda = capability.accelerator("cuda")
    return bool(cuda.enabled and cuda.evidence.get("build_backend") == "cuda")


def _install_finite_check(module):
    original = module.has_inf_or_nan
    replacement = make_has_inf_or_nan(
        module, original, cuda_allowed=_cuda_norm_allowed(jt.capability))
    replace(module, "has_inf_or_nan", replacement, original)
    replace_bound_aliases(
        "deepspeed", "has_inf_or_nan", original, replacement)


def _install_zero_optimizer(module):
    methods = make_zero_optimizer_methods(
        module, native=jt, cuda_allowed=_cuda_norm_allowed(jt.capability))
    for name, (original, replacement) in methods.items():
        replace(module.DeepSpeedZeroOptimizer, name, replacement, original)


def _install_global_norm(module):
    original = module.get_global_norm_of_tensors
    replacement = make_global_norm(
        module, original, cuda_allowed=_cuda_norm_allowed(jt.capability))
    replace(module, "get_global_norm_of_tensors", replacement, original)
    # A circular import may already have bound an alias while utils executed.
    # Later from-imports naturally see the replacement; do not eagerly import
    # bf16_optimizer here. All existing aliases share transaction ownership.
    replace_bound_aliases("deepspeed", "get_global_norm_of_tensors",
                          original, replacement)


def _source_version(root):
    metadata = root / "git_version_info_installed.py"
    if not metadata.is_file():
        raise UnsupportedAdapterVersion("DeepSpeed must be installed from its pinned source")
    for node in ast.parse(metadata.read_text(encoding="utf8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "version" for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise UnsupportedAdapterVersion("DeepSpeed source version is missing")


def _check_source(name, source):
    if hashlib.sha256(source.encode("utf8")).hexdigest() != SOURCE_SHA256[name]:
        raise UnsupportedAdapterVersion("DeepSpeed source contract changed: " + name)


def _transform(name, source, filename):
    _check_source(name, source)
    tree = ast.parse(source, filename=filename)
    unavailable = {}
    for index, node in enumerate(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue
        target = ((name == "deepspeed.compile" and node.module == "util") or
                  (name == "deepspeed.runtime.engine" and
                   (node.module or "").startswith("deepspeed.compile") and
                   node.module != "deepspeed.compile.z3_eager_fallback"))
        if target:
            for alias in node.names:
                binding = alias.asname or alias.name
                unavailable[binding] = (_unavailable_capability if binding == "is_deepcompile_supported"
                                        else _Unavailable(node.module + "." + alias.name))
            tree.body[index] = ast.copy_location(ast.Pass(), node)
    return compile(tree, filename, "exec"), unavailable


def _guard_engine(module):
    original = module.DeepSpeedEngine.__init__
    signature = inspect.signature(original)

    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        config = bound.arguments.get("config_class")
        if config is not None and config.compile_config.deepcompile:
            raise NotImplementedError("DeepSpeed DeepCompile is unsupported on Jittor")
        if config is not None and config.elasticity_enabled:
            raise NotImplementedError("DeepSpeed elastic launch is unsupported on Jittor")
        return original(self, *args, **kwargs)

    replace(module.DeepSpeedEngine, "__init__", initialize, original)


def _record_imports(module, before):
    # A source-transformed package must be removed as a whole on rollback:
    # unmodified sibling modules may hold aliases of its Engine class.
    loaded = {name: value for name, value in tuple(sys.modules.items())
              if (name == "deepspeed" or name.startswith("deepspeed."))
              and (name not in before or name == module.__name__)}
    def undo():
        conflicts = []
        for name, value in sorted(loaded.items(), key=lambda item: item[0].count("."), reverse=True):
            current = sys.modules.get(name)
            if current is not value:
                conflicts.append(name)
                continue
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and vars(parent).get(child) is value:
                delattr(parent, child)
            elif parent is not None and child in vars(parent):
                conflicts.append(name + " parent alias")
            sys.modules.pop(name)
        if conflicts:
            raise TransactionConflict("DeepSpeed modules replaced externally: " + ", ".join(conflicts))
    active_transaction().record_undo(undo)


class _Loader(importlib.abc.Loader):
    def __init__(self, original, name, version):
        self.original, self.name, self.version = original, name, version

    def create_module(self, spec):
        return self.original.create_module(spec)

    @owned_runtime_hook(_OWNER)
    def exec_module(self, module):
        before = set(sys.modules)
        try:
            if self.name == "deepspeed":
                require_version("deepspeed", SUPPORTED_VERSIONS, source_version=self.version)
                self.original.exec_module(module)
                require_version("deepspeed", SUPPORTED_VERSIONS)
            else:
                source = self.original.get_source(self.name)
                _check_source(self.name, source)
                if self.name in ("deepspeed.compile", "deepspeed.runtime.engine"):
                    code, bindings = _transform(self.name, source, module.__file__)
                    vars(module).update(bindings)
                    exec(code, vars(module))
                    if self.name == "deepspeed.runtime.engine":
                        _guard_engine(module)
                else:
                    self.original.exec_module(module)
                    if self.name == "deepspeed.runtime.utils":
                        replace(module, "count_used_parameters_in_backward", _count_used_parameters,
                                module.count_used_parameters_in_backward)
                        replace(module, "check_internal_apis_for_count_used_parameters", _has_participation_query,
                                module.check_internal_apis_for_count_used_parameters)
                        _install_finite_check(module)
                        _install_global_norm(module)
                        return
                    if self.name == "deepspeed.runtime.zero.stage_1_and_2":
                        _install_zero_optimizer(module)
                        return
                    name, replacement = {
                        "deepspeed.comm.torch": ("build_shm_op", _unavailable_shm),
                        "deepspeed.elasticity.utils": ("is_torch_elastic_compatible", _unavailable_capability),
                        "deepspeed.utils.torch": ("register_grad_hook", _register_grad_hook),
                    }[self.name]
                    replace(module, name, replacement, getattr(module, name))
        finally:
            if self.name == "deepspeed":
                _record_imports(module, before)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, root, version):
        self.root, self.version = root, version

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "deepspeed" and fullname not in SOURCE_SHA256:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not callable(getattr(spec.loader, "get_source", None)):
            raise UnsupportedAdapterVersion("DeepSpeed needs its pinned Python source: " + fullname)
        if self.root not in Path(spec.origin).resolve().parents:
            raise UnsupportedAdapterVersion("DeepSpeed import source changed after adapter activation")
        spec.loader = _Loader(spec.loader, fullname, self.version)
        return spec


def arm(transaction=None):
    """Validate installed source before arming; activate only in shim processes."""
    with runtime_hook(_OWNER, parent_transaction=transaction):
        return _arm()


def _arm():
    import jittor
    import torch
    if not issubclass(torch.Tensor, jittor.Var):
        raise RuntimeError("DeepSpeed adapter requires the active Jittor Torch frontend")
    existing = [finder for finder in sys.meta_path if isinstance(finder, _Finder)]
    if existing:
        return True
    if any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules):
        raise UnsupportedAdapterVersion("Activate the DeepSpeed adapter before importing DeepSpeed")
    spec = importlib.machinery.PathFinder.find_spec("deepspeed")
    if spec is None:
        raise ModuleNotFoundError("DeepSpeed is not installed", name="deepspeed")
    root = Path(spec.origin).resolve().parent
    version = _source_version(root)
    require_version("deepspeed", SUPPORTED_VERSIONS, source_version=version)
    for name in SOURCE_SHA256:
        relative = name.split(".")[1:]
        path = root.joinpath(*relative).with_suffix(".py")
        if name == "deepspeed.compile":
            path = root / "compile" / "__init__.py"
        _check_source(name, path.read_text(encoding="utf8"))
    finder = _Finder(root, version)
    sys.meta_path.insert(0, finder)
    def undo():
        if not sys.meta_path or sys.meta_path[0] is not finder:
            raise TransactionConflict("DeepSpeed finder moved or replaced externally")
        sys.meta_path.pop(0)
    active_transaction().record_undo(undo)
    return True
