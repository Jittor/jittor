"""Lazily register the project-owned HF async rollout for torch_npu workers."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import json
from pathlib import Path
import sys


_TARGET = "verl.workers.rollout.replica"
_WORKER_TARGET = "verl.single_controller.base.worker"
_VALUEHEAD_TARGET = "verl.utils.model"
_FSDP_ENGINE_TARGET = "verl.workers.engine.fsdp.transformer_impl"
_JAGGED_PADDING_TARGET = "verl.workers.utils.padding"
_TD_TRANSPORT_TARGET = "tensordict._reductions"
_oracle_td_transport_bindings = []


def _jagged_positions_enabled():
    value = os.environ.get("VERL_CORRECT_JAGGED_POSITIONS", "0")
    if value not in ("0", "1"):
        raise SystemExit("VERL_CORRECT_JAGGED_POSITIONS must be absent, 0 or 1")
    return value == "1"

# This is a text-only oracle contract. Never discover the host's optional
# Jittor-backed torchvision facade, nor import it through an unguarded caller.
_find_spec = importlib.util.find_spec

# Opt-in admits only the native paired CUDA wheels in THIS clean oracle venv.
# Admission is provenance, not vision-operator/CUDA correctness evidence.
_NATIVE_VISION_ENV = "VERL_ORACLE_NATIVE_TORCHVISION"
_NATIVE_VISION_PAIR = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128"}
_native_vision_roots = None


def _native_vision_enabled():
    value = os.environ.get(_NATIVE_VISION_ENV, "0")
    if value not in ("0", "1"):
        raise SystemExit(_NATIVE_VISION_ENV + " must be absent, 0 or 1")
    return value == "1"


def _native_vision_path(value, root, *, directory=False):
    try:
        path = Path(value)
        resolved = path.resolve(strict=True)
        valid_kind = path.is_dir() if directory else path.is_file()
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit("native oracle source path unavailable: " + str(value)) from error
    if (not path.is_absolute() or path != resolved or not valid_kind
            or (path != root and root not in path.parents)):
        raise SystemExit("native oracle source outside canonical package/venv: " + str(path))
    return path


def _native_vision_check_spec(fullname, spec, root):
    if spec is None or spec.name != fullname or spec.loader is None or not spec.origin:
        raise SystemExit("native oracle requires a concrete source spec: " + fullname)
    _native_vision_path(spec.origin, root)
    if spec.submodule_search_locations is not None:
        for location in spec.submodule_search_locations:
            _native_vision_path(location, root, directory=True)
    return spec


def _native_vision_check_loaded_torch():
    module = sys.modules.get("torch")
    if module is None:
        raise SystemExit("native torchvision did not load official Torch")
    require_official_torch(module)
    if str(getattr(module, "__version__", "")) != _NATIVE_VISION_PAIR["torch"]:
        raise SystemExit("loaded native oracle Torch version differs from paired distribution")
    root = _native_vision_roots["torch"]
    _native_vision_path(getattr(module, "__file__", None), root)
    _native_vision_check_spec("torch", getattr(module, "__spec__", None), root)


def _prepare_native_torchvision():
    global _native_vision_roots
    loaded = [(name, module) for name, module in tuple(sys.modules.items())
              if name == "torchvision" or name.startswith("torchvision.")]
    if not _native_vision_enabled():
        if loaded:
            raise SystemExit("text-only oracle rejects preloaded torchvision: " + loaded[0][0])
        return
    if loaded and _native_vision_roots is None:
        raise SystemExit("native oracle rejects torchvision preloaded before admission: " + loaded[0][0])
    require_clean_venv()
    import importlib.metadata
    import sysconfig

    prefix = Path(sys.prefix)
    _native_vision_path(prefix, prefix, directory=True)
    if Path(sys.executable).parent.parent != prefix:
        raise SystemExit("native oracle interpreter/venv prefix mismatch")
    purelib = _native_vision_path(sysconfig.get_path("purelib"), prefix, directory=True)
    roots = {}
    for name, expected in _NATIVE_VISION_PAIR.items():
        try:
            distribution = importlib.metadata.distribution(name)
            version = distribution.version
            location = Path(distribution.locate_file(""))
            files = list(distribution.files or ())
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError, TypeError) as error:
            raise SystemExit("native oracle distribution unavailable: " + name) from error
        if version != expected or location != purelib:
            raise SystemExit("native oracle paired version/location mismatch: " + name)
        _native_vision_path(location, purelib, directory=True)
        metadata_files = [item for item in files if len(Path(item).parts) == 2
                          and Path(item).parts[0].startswith(name + "-")
                          and Path(item).parts[0].endswith(".dist-info")
                          and Path(item).name == "METADATA"]
        if len(metadata_files) != 1:
            raise SystemExit("native oracle requires one owned distribution METADATA: " + name)
        _native_vision_path(distribution.locate_file(metadata_files[0]), purelib)
        root = _native_vision_path(purelib / name, purelib, directory=True)
        if name == "torchvision":
            package_files = [item for item in files if Path(item).parts
                             and Path(item).parts[0] == name]
            extensions = [item for item in package_files if Path(item).parent == Path(name)
                          and any(Path(item).name == "_C" + suffix
                                  for suffix in importlib.machinery.EXTENSION_SUFFIXES)]
            if Path(name + "/__init__.py") not in package_files or len(extensions) != 1:
                raise SystemExit("native torchvision wheel requires owned package/extension RECORD entries")
            # load_library(_C) may bypass MetaPathFinder; check RECORD paths too.
            for item in package_files:
                _native_vision_path(distribution.locate_file(item), root)
        spec = importlib.machinery.PathFinder.find_spec(name)
        _native_vision_check_spec(name, spec, root)
        roots[name] = root
    if _native_vision_roots is not None and _native_vision_roots != roots:
        raise SystemExit("native oracle package ownership changed after admission")
    _native_vision_roots = roots
    if "torch" in sys.modules:
        _native_vision_check_loaded_torch()
    for name, module in loaded:
        if module is None:
            raise SystemExit("native oracle rejects incomplete preloaded torchvision: " + name)
        root = roots["torchvision"]
        _native_vision_path(getattr(module, "__file__", None), root)
        _native_vision_check_spec(name, getattr(module, "__spec__", None), root)
        if name == "torchvision" and getattr(module, "__version__", None) != _NATIVE_VISION_PAIR[name]:
            raise SystemExit("preloaded native torchvision version mismatch")


class _NativeVisionLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def create_module(self, spec):
        try:
            creator = getattr(self.wrapped, "create_module", None)
            return creator(spec) if creator is not None else None
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as error:
            raise SystemExit("native torchvision module creation failed: " + str(error)) from error

    def exec_module(self, module):
        try:
            self.wrapped.exec_module(module)
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as error:
            raise SystemExit("native torchvision import failed: " + str(error)) from error
        _native_vision_path(getattr(module, "__file__", None), _native_vision_roots["torchvision"])
        _native_vision_check_loaded_torch()
        if (module.__name__ == "torchvision"
                and getattr(module, "__version__", None) != _NATIVE_VISION_PAIR["torchvision"]):
            raise SystemExit("loaded native torchvision version mismatch")


def _native_torchvision_spec(fullname, path=None, target=None):
    if not _native_vision_enabled() or _native_vision_roots is None:
        raise SystemExit("native torchvision requested before oracle admission")
    root = _native_vision_roots["torchvision"]
    if path is not None:
        for location in path:
            _native_vision_path(location, root, directory=True)
    spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
    _native_vision_check_spec(fullname, spec, root)
    spec.loader = _NativeVisionLoader(spec.loader)
    return spec


def _oracle_find_spec(name, package=None):
    if name == "torchvision" or name.startswith("torchvision."):
        if not _native_vision_enabled():
            return None
        if _native_vision_roots is None:
            raise SystemExit("native torchvision discovery before oracle admission")
        try:
            spec = _find_spec(name, package)
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as error:
            raise SystemExit("native torchvision discovery failed: " + name) from error
        return _native_vision_check_spec(name, spec, _native_vision_roots["torchvision"])
    return _find_spec(name, package)


class _BlockJittorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jittor" or fullname.startswith("jittor."):
            raise ModuleNotFoundError(
                "official torch_npu oracle refuses to import the Jittor runtime",
                name=fullname,
            )
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            if _native_vision_enabled():
                return _native_torchvision_spec(fullname, path, target)
            raise ModuleNotFoundError("text-only oracle excludes torchvision", name=fullname)
        return None


def _install_runtime_guards():
    if (os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1"
            and os.environ.get("JITTOR_VERL_RAY_CUDA_CONTEXT") == "1"):
        raise SystemExit("oracle rejects simultaneous NPU and CUDA context markers")
    for name in ("JITTOR_TORCH_SHIM", "JITTOR_TRANSFORMERS_MATERIALIZED_INIT",
                 "JITTOR_VERL_NATIVE_NPU_KERNELS", "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT"):
        if os.environ.get(name) not in (None, "", "0"):
            raise SystemExit("official torch_npu oracle rejects candidate variable " + name)
    contaminated = [name for name in sys.modules
                    if name == "jittor" or name.startswith("jittor.")]
    if contaminated:
        raise SystemExit("Jittor modules already loaded in official oracle: " + contaminated[0])
    _prepare_native_torchvision()
    if not any(isinstance(finder, _BlockJittorFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockJittorFinder())
    importlib.util.find_spec = _oracle_find_spec


def require_official_torch(torch_api):
    if hasattr(torch_api, "_torch_compat_install_context") or not hasattr(torch_api, "_C"):
        raise SystemExit("official torch_npu oracle resolved a shim or non-binary torch")
    contaminated = [name for name in sys.modules
                    if name == "jittor" or name.startswith("jittor.")]
    if contaminated:
        raise SystemExit("Jittor modules already loaded in official oracle: " + contaminated[0])


def _configure_cuda_strict_fp32(torch_api):
    profile = os.environ.get("VERL_CUDA_STRICT_FP32", "0")
    if profile not in ("0", "1"):
        raise SystemExit("VERL_CUDA_STRICT_FP32 must be absent, 0 or 1")
    if profile == "0":
        return
    if (os.environ.get("VERL_DEVICE") != "cuda"
            or os.environ.get("VERL_TRAIN_DTYPE", "fp32") != "fp32"
            or os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1"):
        raise SystemExit("VERL_CUDA_STRICT_FP32 requires CUDA fp32 training")
    torch_api.backends.cuda.matmul.allow_tf32 = False
    torch_api.backends.cudnn.allow_tf32 = False
    torch_api.backends.cudnn.benchmark = False
    torch_api.set_float32_matmul_precision("highest")
    conditions = {
        "cuda_matmul_allow_tf32": torch_api.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch_api.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch_api.backends.cudnn.benchmark,
        "float32_matmul_precision": torch_api.get_float32_matmul_precision(),
    }
    if (any(conditions[key] is not False for key in
            ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark"))
            or conditions["float32_matmul_precision"] != "highest"):
        raise RuntimeError("strict FP32 precision readback mismatch: " + repr(conditions))
    print("VERL_CUDA_STRICT_FP32_READY " + json.dumps({
        "framework": "torch", "pid": os.getpid(),
        "runtime_conditions": conditions, "scope": "public_precision_readback_only",
    }, sort_keys=True), flush=True)


def require_clean_venv():
    config = Path(sys.executable).parent.parent / "pyvenv.cfg"
    if not config.is_file():
        raise SystemExit("oracle must run in its own venv (missing pyvenv.cfg)")
    settings = {key.strip().lower(): value.strip().lower()
                for key, value in (line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)}
    if settings.get("include-system-site-packages") != "false":
        raise SystemExit("oracle venv must disable include-system-site-packages")


def _register_hf_async(replica_module) -> None:
    from jittor_adapters.verl import hf_async_rollout as adapter
    from verl.workers.rollout import base as rollout_base

    # Unlike candidate patches, the oracle registration must not import the
    # Jittor patcher. It still has to enforce the SAME downstream source pins.
    adapter.require_version("verl", adapter.SUPPORTED_VERSIONS)
    adapter._check_source(rollout_base, adapter.BASE_SOURCE_SHA256)
    adapter._check_source(replica_module, adapter.REPLICA_SOURCE_SHA256)
    adapter._ensure_classes()

    key = (adapter.NAME, "async")
    fqdn = adapter.__name__ + ".HFAsyncServerAdapter"
    existing = rollout_base._ROLLOUT_REGISTRY.get(key)
    if existing not in (None, fqdn):
        raise RuntimeError(f"rollout registry key {key!r} is already owned by {existing!r}")
    rollout_base._ROLLOUT_REGISTRY[key] = fqdn

    registry = replica_module.RolloutReplicaRegistry
    loader = registry._registry.get(adapter.NAME)
    if loader is None:
        def load_hf_async():
            return adapter.HFAsyncReplica

        load_hf_async._jittor_hf_async_loader = True
        registry.register(adapter.NAME, load_hf_async)
    elif not getattr(loader, "_jittor_hf_async_loader", False):
        raise RuntimeError(f"rollout replica key {adapter.NAME!r} is already owned")


def _register_valuehead_attention(module) -> None:
    from jittor_adapters.verl import valuehead_attention as adapter

    if not adapter.enabled():
        return
    original, replacement = adapter.prepare(module)
    if original is replacement:
        return
    # Oracle owns only downstream attributes. Do not call the candidate's
    # module patcher or import the Jittor runtime into this independent process.
    module.load_valuehead_model = replacement
    for name, loaded in tuple(sys.modules.items()):
        if ((name == "verl" or name.startswith("verl.")) and loaded is not None
                and vars(loaded).get(adapter.FUNCTION) is original):
            setattr(loaded, adapter.FUNCTION, replacement)


def _register_resident_reference(module) -> None:
    from jittor_adapters.verl import fsdp2_resident_reference as adapter

    if not adapter.cuda_enabled():
        return
    engine, original, replacement = adapter.prepare(module)
    if original is not replacement:
        # The policy changes only this downstream-owned builder, never Torch
        # FSDP policy classes or the independent runtime's framework state.
        engine._build_fsdp_module = replacement


def _register_jagged_positions(module) -> None:
    if not _jagged_positions_enabled():
        return
    from jittor_adapters.verl import jagged_positions as adapter
    target = module
    if module.__name__ == "verl":
        target = sys.modules.get(adapter.TU_MODULE)
        if target is None:
            raise SystemExit("verl root did not load its jagged TensorDict helper")
    try:
        plans = adapter.prepare(target)
    except adapter.UnsupportedAdapterVersion as error:
        # Python site swallows Exception; a partial policy must abort startup.
        raise SystemExit("oracle jagged-position registration failed: " + str(error)) from error
    for name, original, replacement in plans:
        if vars(target).get(name) is not original:
            raise SystemExit("oracle jagged-position helper ownership changed")
        # Process-local downstream attributes only. Do not call the candidate
        # patcher, import Jittor, or mutate either framework's native namespace.
        setattr(target, name, replacement)
        for path, loaded in tuple(sys.modules.items()):
            if ((path == "verl" or path.startswith("verl.")) and loaded is not None
                    and vars(loaded).get(name) is original):
                setattr(loaded, name, replacement)


def _restore_td_transport_binding(owner, name, original, replacement):
    current = vars(owner).get(name)
    if current is original:
        return  # A failed setter left the pre-install binding untouched.
    if current is not replacement:
        raise SystemExit("oracle TensorDict transport binding changed externally: " + name)
    setattr(owner, name, original)


def _restore_td_transport():
    # Validate all ownership first; never overwrite an external participant.
    for owner, name, original, replacement in _oracle_td_transport_bindings:
        if vars(owner).get(name) is not replacement:
            raise SystemExit("oracle TensorDict transport ownership changed before restore")
    for binding in reversed(_oracle_td_transport_bindings):
        _restore_td_transport_binding(*binding)
    _oracle_td_transport_bindings.clear()


def _register_td_transport(module) -> None:
    if not _jagged_positions_enabled():
        return
    root = sys.modules.get("tensordict")
    if root is None or "__version__" not in vars(root):
        return  # Its root completion will validate; never import downstream here.
    if module.__name__ not in ("tensordict", _TD_TRANSPORT_TARGET) or sys.modules.get(module.__name__) is not module:
        raise SystemExit("oracle TensorDict transport callback ownership changed")
    target = sys.modules.get(_TD_TRANSPORT_TARGET)
    if target is None:
        return  # The registered reductions callback handles a later module load.
    try:
        from contextlib import ExitStack
        from jittor_adapters.tensordict import consolidated_transport as adapter
        plans = adapter.prepare(target)
        if not plans:
            current = vars(target).get(adapter.FUNCTION)
            owned = [b for b in _oracle_td_transport_bindings if b[0] is target and b[1] == adapter.FUNCTION]
            if len(owned) != 1 or owned[0][3] is not current:
                raise SystemExit("oracle TensorDict transport replacement is not locally owned")
            for owner, name, original, replacement in _oracle_td_transport_bindings:
                if vars(owner).get(name) is not replacement:
                    raise SystemExit("oracle TensorDict transport alias ownership changed")
            return
        bindings = []
        for name, original, replacement in plans:
            bindings.append((target, name, original, replacement))
            for path, loaded in tuple(sys.modules.items()):
                if (loaded is not None and loaded is not target
                        and (path == "tensordict" or path.startswith("tensordict."))
                        and vars(loaded).get(name) is original
                        and not any(b[0] is loaded and b[1] == name for b in bindings)):
                    bindings.append((loaded, name, original, replacement))
        with ExitStack() as rollback:
            for owner, name, original, replacement in bindings:
                if vars(owner).get(name) is not original:
                    raise SystemExit("oracle TensorDict transport binding changed before install")
                # Register undo before mutation so a setter failure cannot leak
                # already installed bindings. Modules normally have plain attrs.
                rollback.callback(_restore_td_transport_binding, owner, name, original, replacement)
                setattr(owner, name, replacement)
            rollback.pop_all()
        _oracle_td_transport_bindings.extend(bindings)
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as error:
        raise SystemExit("oracle TensorDict transport registration failed: " + str(error)) from error


def _bind_ray_npu_context(worker_module) -> None:
    worker = worker_module.Worker
    original = worker._setup_env_cuda_visible_devices
    if getattr(original, "_jittor_oracle_npu_context", False):
        return

    def setup_env_and_npu_context(self):
        original(self)
        from verl.utils.device import get_torch_device, get_visible_devices_keyword
        from verl.utils.ray_utils import ray_noset_visible_devices

        if (
            not ray_noset_visible_devices()
            and get_visible_devices_keyword().upper() == "ASCEND_RT_VISIBLE_DEVICES"
        ):
            # Ray exposes exactly one physical NPU to each worker, hence its
            # process-local device index is always zero.
            get_torch_device().set_device(0)

    setup_env_and_npu_context._jittor_oracle_npu_context = True
    worker._setup_env_cuda_visible_devices = setup_env_and_npu_context


class _AfterLoader(importlib.abc.Loader):
    def __init__(self, wrapped, callback):
        self.wrapped = wrapped
        self.callback = callback

    def create_module(self, spec):
        creator = getattr(self.wrapped, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        self.callback(module)


class _OracleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        callbacks = {}
        if _jagged_positions_enabled():
            callbacks.update({"verl": _register_jagged_positions,
                              _JAGGED_PADDING_TARGET: _register_jagged_positions,
                              "tensordict": _register_td_transport,
                              _TD_TRANSPORT_TARGET: _register_td_transport})
        if os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1":
            callbacks.update({_TARGET: _register_hf_async,
                              _WORKER_TARGET: _bind_ray_npu_context})
        if os.environ.get("VERL_DEVICE") == "cuda":
            if os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1":
                callbacks[_VALUEHEAD_TARGET] = _register_valuehead_attention
            if os.environ.get("JITTOR_VERL_FSDP2_RESIDENT_REFERENCE") == "1":
                callbacks[_FSDP_ENGINE_TARGET] = _register_resident_reference
        callback = callbacks.get(fullname)
        if callback is None:
            return None
        # Ask the standard path finder directly so this finder cannot recurse.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _AfterLoader(spec.loader, callback)
        return spec


# Validate the opt-in even when every optional rollout/precision feature is off.
_native_vision_enabled()
_jagged_positions_enabled()

if (_native_vision_enabled()
        or _jagged_positions_enabled()
        or os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1"
        or os.environ.get("VERL_CUDA_STRICT_FP32") == "1"
        or (os.environ.get("VERL_DEVICE") == "cuda"
            and os.environ.get("JITTOR_VERL_FSDP2_RESIDENT_REFERENCE") == "1")):
    # Guards precede any Torch import, including rollout context initialization.
    _install_runtime_guards()
    if os.environ.get("VERL_CUDA_STRICT_FP32", "0") not in ("0", "1"):
        raise SystemExit("VERL_CUDA_STRICT_FP32 must be absent, 0 or 1")
    if os.environ.get("VERL_CUDA_STRICT_FP32") == "1":
        try:
            import torch

            require_official_torch(torch)
            _configure_cuda_strict_fp32(torch)
        except (ImportError, RuntimeError, AttributeError, ValueError, TypeError) as error:
            # Ordinary sitecustomize exceptions are swallowed by Python.
            raise SystemExit("oracle strict FP32 initialization failed: " + str(error)) from error
    # The rollout server is a new Ray process. Its serialized constructor can
    # import verl before __init__, so establish the ACL context only for that
    # explicitly marked process. Generic workers retain normal lazy imports.
    if os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1":
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if not visible.isdecimal():
            raise SystemExit("rollout oracle requires exactly one pre-bound visible NPU")
        try:
            import torch

            require_official_torch(torch)
            torch.npu.set_device(0)
        except (ImportError, RuntimeError, AttributeError) as error:
            # Python's site module swallows Exception from sitecustomize. A
            # failed context must abort startup, never leave a partial worker.
            raise SystemExit("oracle NPU context initialization failed: " + str(error)) from error
    if os.environ.get("JITTOR_VERL_RAY_CUDA_CONTEXT") == "1":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdecimal():
            raise SystemExit("rollout oracle requires exactly one pre-bound CUDA device")
        try:
            import torch

            require_official_torch(torch)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(0)
        except (ImportError, RuntimeError, AttributeError) as error:
            raise SystemExit("oracle CUDA context initialization failed: " + str(error)) from error
    if os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1" and _TARGET in sys.modules:
        _register_hf_async(sys.modules[_TARGET])
    if os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1" and _WORKER_TARGET in sys.modules:
        _bind_ray_npu_context(sys.modules[_WORKER_TARGET])
    if (os.environ.get("VERL_DEVICE") == "cuda"
            and os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1"
            and _VALUEHEAD_TARGET in sys.modules):
        _register_valuehead_attention(sys.modules[_VALUEHEAD_TARGET])
    if (os.environ.get("VERL_DEVICE") == "cuda"
            and os.environ.get("JITTOR_VERL_FSDP2_RESIDENT_REFERENCE") == "1"
            and _FSDP_ENGINE_TARGET in sys.modules):
        _register_resident_reference(sys.modules[_FSDP_ENGINE_TARGET])
    if _jagged_positions_enabled():
        if "tensordict" in sys.modules:
            _register_td_transport(sys.modules["tensordict"])
        if _TD_TRANSPORT_TARGET in sys.modules:
            _register_td_transport(sys.modules[_TD_TRANSPORT_TARGET])
        if "verl" in sys.modules:
            _register_jagged_positions(sys.modules["verl"])
        if _JAGGED_PADDING_TARGET in sys.modules:
            _register_jagged_positions(sys.modules[_JAGGED_PADDING_TARGET])
    if not any(isinstance(finder, _OracleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _OracleFinder())
