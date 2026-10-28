"""Keep the pinned FSDP2 reference model resident on its device.

verl 0.9.1 unconditionally selects ``CPUOffloadPolicy`` for every forward-only
engine, even when the user disabled parameter offload. Jittor FSDP2 does not
implement that policy and must not silently treat it as a no-op. For the
explicit resident profile, call the exact pinned builder with a temporary
configuration whose forward-only bit is false. This changes only placement:
the original configuration is restored before optimizer construction, so the
reference model remains forward-only and no optimizer is created. The existing
native-NPU policy is unchanged. CUDA requires VERL_DEVICE=cuda and the explicit
JITTOR_VERL_FSDP2_RESIDENT_REFERENCE=1 opt-in on BOTH candidate and oracle.
The original initialized world-one CUDA mesh scope is retained. World two is
limited to an initialized NCCL group and a full-world one-dimensional CUDA
mesh with matching ranks and shard group; other topologies are rejected.
Admission alone does not establish actual residency, accelerator execution,
zero fallback, numerical correctness or complete PPO support.

Upstream: verl-project/verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
Revalidate when pinned builder/config or compat public runtime/patcher changes.
"""
import copy
from functools import wraps
import hashlib
import inspect
import os
from pathlib import Path
import sys
import types

from .._common import require_version, required_patch, replace, UnsupportedAdapterVersion


MODULE = "verl.workers.engine.fsdp.transformer_impl"
SOURCE_SHA256 = "79d8779fb2a0b8a1ac7b3ec85fba2521ac8cae079925a497c4366fd77f8abfbe"
METHOD_SHA256 = "35b7b539858443e45894da2d9675fb5a1d7b2bbb7c7c8b35605b0746d3179213"
POLICY = "single_rank_fsdp2_reference_resident_on_npu_v1"
CUDA_POLICY = "world1_or2_fsdp2_reference_resident_on_cuda_v2"
OPT_IN = "JITTOR_VERL_FSDP2_RESIDENT_REFERENCE"
SUPPORTED_VERSIONS = frozenset(("0.9.1",))


def cuda_enabled():
    value = os.environ.get(OPT_IN)
    if value not in (None, "", "0", "1"):
        raise UnsupportedAdapterVersion(OPT_IN + " must be absent, 0 or 1")
    return os.environ.get("VERL_DEVICE") == "cuda" and value == "1"


def _check_source(module, method):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        method_source = inspect.getsource(method).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl FSDPEngine source is required") from error
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl FSDPEngine module source changed")
    if hashlib.sha256(method_source).hexdigest() != METHOD_SHA256:
        raise UnsupportedAdapterVersion("verl FSDPEngine._build_fsdp_module source changed")


def _require_cuda_runtime():
    require_version("verl", SUPPORTED_VERSIONS)
    if os.environ.get("JITTOR_VERL_NATIVE_NPU_KERNELS") == "1":
        raise UnsupportedAdapterVersion("CUDA resident policy rejects native-NPU opt-in")
    frontend = sys.modules.get("torch")
    if frontend is None:
        raise UnsupportedAdapterVersion("CUDA resident policy requires the loaded Torch runtime")
    if os.environ.get("JITTOR_TORCH_SHIM") == "1":
        # Only the candidate enters this branch. Use the existing public
        # namespace-owner and live policy APIs, never framework private state.
        from jittor.compat.torch import namespace_owner

        owner = namespace_owner(frontend)
        if owner is None:
            raise UnsupportedAdapterVersion("CUDA resident policy requires the actual Jittor Torch namespace")
        policy = owner.introspection.policy.runtime
        # use_acl is a legacy alias of use_cuda, not an independent backend
        # flag. Capability's frozen build evidence plus live policy is the
        # public way to reject a CUDA facade backed by ACL or CPU here.
        capability = owner.capability.accelerator("cuda")
        if (not capability.enabled or capability.evidence.get("build_backend") != "cuda"
                or int(policy.use_cuda) != 1):
            raise UnsupportedAdapterVersion("CUDA resident policy requires active native CUDA, not ACL/CPU")
    else:
        # The independent oracle must not import Jittor merely to identify it.
        version = getattr(frontend, "version", None)
        if (not hasattr(frontend, "_C") or hasattr(frontend, "_torch_compat_install_context")
                or not getattr(version, "cuda", None) or getattr(version, "hip", None) is not None):
            raise UnsupportedAdapterVersion("CUDA resident policy requires official binary CUDA Torch")
    available = getattr(getattr(frontend, "cuda", None), "is_available", None)
    if not callable(available) or not available():
        raise UnsupportedAdapterVersion("CUDA resident policy requires an available CUDA backend")


def _require_runtime(policy=None):
    if cuda_enabled():
        if policy not in (None, CUDA_POLICY):
            raise UnsupportedAdapterVersion("resident reference policy backend changed")
        _require_cuda_runtime()
        return CUDA_POLICY
    if policy == CUDA_POLICY:
        raise UnsupportedAdapterVersion("CUDA resident policy requires explicit CUDA resident opt-in")
    from .native_npu import _require_runtime as require_npu_runtime

    require_npu_runtime()
    return POLICY


def _require_cuda_mesh_rank(engine):
    frontend = sys.modules.get("torch")
    distributed = getattr(frontend, "distributed", None)
    initialized = getattr(distributed, "is_initialized", None)
    if not callable(initialized) or not initialized():
        raise UnsupportedAdapterVersion("CUDA resident reference requires an initialized process group")
    world_size = distributed.get_world_size()
    mesh = getattr(engine, "device_mesh", None)
    if world_size == 1:
        # Preserve the existing single-rank admission contract. The stronger
        # checks below scope only the newly admitted two-rank configuration.
        if mesh is None or getattr(mesh, "device_type", None) != "cuda" or mesh.size() != 1:
            raise UnsupportedAdapterVersion("CUDA resident reference requires a single-rank CUDA device mesh")
        return
    if type(world_size) is not int or world_size != 2:
        raise UnsupportedAdapterVersion("CUDA resident reference is scoped to actual world size one or two")
    rank = distributed.get_rank()
    if type(rank) is not int or rank not in (0, 1):
        raise UnsupportedAdapterVersion("CUDA resident reference requires an actual rank in world two")
    if str(distributed.get_backend()).lower() != "nccl":
        raise UnsupportedAdapterVersion("CUDA resident reference requires the actual NCCL backend")
    if (mesh is None or getattr(mesh, "device_type", None) != "cuda"
            or getattr(mesh, "ndim", None) != 1 or mesh.size() != world_size
            or mesh.get_rank() != rank
            or mesh.mesh.reshape(-1).tolist() != [0, 1]):
        raise UnsupportedAdapterVersion("CUDA resident reference requires a full-world one-dimensional CUDA shard mesh")
    group = mesh.get_group()
    if (group is None or group.size() != world_size or group.rank() != rank
            or distributed.get_world_size(group) != world_size
            or distributed.get_rank(group) != rank
            or distributed.get_process_group_ranks(group) != [0, 1]
            or str(distributed.get_backend(group)).lower() != "nccl"):
        raise UnsupportedAdapterVersion("CUDA resident reference requires the actual full-world NCCL shard group")


def _make_builder(original, policy=POLICY):
    @wraps(original)
    def build_with_resident_reference(self, module):
        config = self.engine_config
        if getattr(config, "strategy", None) != "fsdp2" or not getattr(config, "forward_only", False):
            return original(self, module)
        _require_runtime(policy)
        if policy == CUDA_POLICY:
            _require_cuda_mesh_rank(self)
        if any(bool(getattr(config, name, False)) for name in (
                "offload_policy", "param_offload", "optimizer_offload")):
            raise UnsupportedAdapterVersion(
                "resident reference policy requires all explicit FSDP2 offload options disabled"
            )
        resident = copy.copy(config)
        object.__setattr__(resident, "forward_only", False)
        self.engine_config = resident
        try:
            result = original(self, module)
        finally:
            self.engine_config = config
        if getattr(self, "_uses_fsdp2_cpu_offload_policy", False):
            raise RuntimeError("resident reference policy unexpectedly selected CPUOffloadPolicy")
        self._jittor_verl_resident_reference = policy
        self._jittor_verl_resident_reference_backend = "cuda" if policy == CUDA_POLICY else "npu"
        print("JITTOR_VERL_REFERENCE_RESIDENT_OK "
              "backend=" + self._jittor_verl_resident_reference_backend +
              " policy=" + policy + " forward_only=true offload_policy=none", flush=True)
        return result

    build_with_resident_reference._jittor_verl_resident_reference_policy = policy
    build_with_resident_reference._jittor_verl_original = original
    return build_with_resident_reference


def prepare(module):
    """Return the exact pinned builder for either runtime without importing Jittor in the oracle."""
    policy = _require_runtime()
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected verl FSDP engine module")
    engine = vars(module).get("FSDPEngine")
    if not isinstance(engine, type) or engine.__module__ != MODULE:
        raise UnsupportedAdapterVersion("verl FSDPEngine ownership changed")
    current = vars(engine).get("_build_fsdp_module")
    if (getattr(current, "_jittor_verl_resident_reference_policy", None) == policy
            and isinstance(current, types.FunctionType) and current.__globals__ is globals()):
        _check_source(module, current._jittor_verl_original)
        return engine, current, current
    if (not isinstance(current, types.FunctionType) or current.__module__ != MODULE
            or current.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("verl FSDPEngine._build_fsdp_module ownership changed")
    _check_source(module, current)
    return engine, current, _make_builder(current, policy)


@required_patch
def patch_fsdp_engine(module):
    engine, current, replacement = prepare(module)
    if current is replacement:
        return False
    replace(engine, "_build_fsdp_module", replacement, current)
    return True


def register(register_module_patch):
    from .native_npu import enabled

    if cuda_enabled() or enabled():
        register_module_patch(MODULE, patch_fsdp_engine)
