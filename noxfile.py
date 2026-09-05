"""Canonical local and CI sessions for the staged repository modernization."""

from __future__ import print_function

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nox

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parent

# The gate's scope lives next to the tests it selects, so `nox -s cpu` and
# `tools/run_test_suite.py` cannot drift apart -- and so adding a test file is
# enough to have it gated. See tests/_helpers/gate_scope.py.
sys.path.insert(0, str(REPO_ROOT / "tests"))
try:
    from _helpers.gate_scope import (  # noqa: E402
        native_arguments as gate_native_arguments,
        torch_arguments as gate_torch_arguments,
    )
    from _helpers.tiers import worker_thread_budget  # noqa: E402
    from _helpers.process_modes import TORCH_MODE_PATHS  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "tests"))

# Nox imports this file through Python's loader before session isolation starts.
_loader_cache = globals().get("__cached__")
if isinstance(_loader_cache, str):
    try:
        os.unlink(_loader_cache)
    except FileNotFoundError:
        pass
    try:
        os.rmdir(os.path.dirname(_loader_cache))
    except OSError:
        pass

LAB_ROOT = (
    Path(os.environ.get("JITTOR_LAB_ROOT", str(REPO_ROOT.parent / "jittor-lab")))
    .expanduser()
    .resolve()
)
NOX_STATE_ROOT = LAB_ROOT / "_state" / "nox"
# Two caches that deliberately outlive a session.
#
# Every session used to get a fresh, empty JITTOR_HOME under its own scratch
# directory, so every session paid a full cold build -- the C++ core, then
# every operator -- and had to reach the network for MKL, cub, cutt and NCCL
# first. That is most of what the 40-minute and 2-hour gates were spending
# their time on, and none of it was testing anything: the build being
# exercised is identical between sessions whose build configuration is
# identical. It is safe to share because the cache path already partitions
# below this root by everything that makes two builds different -- Jittor and
# compiler and Python version, platform, CPU, and the build-configuration
# fingerprint (cc_flags, nvcc_flags, kernel_flags, cuda_archs, enable_lto,
# nvcc_path) -- and because a single lock now serialises the writers.
#
# Set JITTOR_NOX_SHARED_CACHE=0 to go back to one empty cache per session,
# which is what to do when a *build* is what is under suspicion.
NOX_JITTOR_CACHE = NOX_STATE_ROOT / "cache" / "jittor"
# Third-party archives, fetched once by `nox -s prefetch` and copied from
# afterwards. Sharing the cache above already removes the repeated downloads
# on one machine; this is what makes a *fresh* machine, or a cleared cache,
# not need the network at all.
NOX_JITTOR_ASSETS = NOX_STATE_ROOT / "cache" / "jittor-assets"

RUFF = "ruff==0.15.22"
MYPY = "mypy==1.8.0"
ASV = "asv==0.6.6"
BUILD = "build==1.3.0"
PYTEST = "pytest==7.4.4"
PYTEST_TIMEOUT = "pytest-timeout==2.3.1"
PYTEST_XDIST = "pytest-xdist==3.6.1"
SCIPY = "scipy==1.13.1"
SETUPTOOLS = "setuptools==83.0.0"
WHEEL = "wheel==0.45.1"
JUPYTEXT = "jupytext==1.17.3"
NBCLIENT = "nbclient==0.10.2"
NBFORMAT = "nbformat==5.10.4"
IPYKERNEL = "ipykernel==6.29.5"
DOCS_REQUIREMENTS = REPO_ROOT / "requirements" / "docs.txt"

NN_MIGRATION_FILES = (
    "python/jittor/nn/__init__.py",
    "python/jittor/nn/_bindings.py",
    "python/jittor/nn/backends/batch_norm_training_cuda.py",
    "python/jittor/nn/backends/channel_bias_cuda.py",
    "python/jittor/nn/backends/group_norm_cuda.py",
    "python/jittor/nn/backends/layer_norm_training_cuda.py",
    "python/jittor/nn/backends/rms_norm_training_cuda.py",
    "python/jittor/nn/functional/__init__.py",
    "python/jittor/nn/functional/attention.py",
    "python/jittor/nn/functional/autograd.py",
    "python/jittor/nn/functional/complex.py",
    "python/jittor/nn/functional/dropout.py",
    "python/jittor/nn/functional/embedding.py",
    "python/jittor/nn/functional/fold.py",
    "python/jittor/nn/functional/grid.py",
    "python/jittor/nn/functional/interpolation.py",
    "python/jittor/nn/functional/linear.py",
    "python/jittor/nn/functional/matrix.py",
    "python/jittor/nn/functional/multihead_attention.py",
    "python/jittor/nn/functional/padding.py",
    "python/jittor/nn/functional/pooling.py",
    "python/jittor/nn/functional/shape.py",
    "python/jittor/nn/functional/tensor.py",
    "python/jittor/nn/legacy_complex.py",
    "python/jittor/nn/modules/__init__.py",
    "python/jittor/nn/modules/activation.py",
    "python/jittor/nn/modules/attention.py",
    "python/jittor/nn/modules/bilinear.py",
    "python/jittor/nn/modules/container.py",
    "python/jittor/nn/modules/dropout.py",
    "python/jittor/nn/modules/embedding.py",
    "python/jittor/nn/modules/fold.py",
    "python/jittor/nn/modules/loss.py",
    "python/jittor/nn/modules/normalization.py",
    "python/jittor/nn/modules/parameter.py",
    "python/jittor/nn/modules/shape.py",
    "python/jittor/nn/modules/upsampling.py",
    "python/jittor/nn/utils/__init__.py",
    "python/jittor/nn/utils/weight_norm.py",
    "tests/nn/test_attention_oracle.py",
    "tests/nn/test_nn_capabilities.py",
)

RATCHET_FILES = (
    *NN_MIGRATION_FILES,
    "noxfile.py",
    "agent/scripts/check_sdist_contents.py",
    "agent/scripts/check_wheel_contents.py",
    "docs/_myst_autodoc.py",
    "docs/conf.py",
    "python/jittor/selftest.py",
    "python/jittor_utils/cuda_wheel.py",
    "python/jittor/compat/shim/deploy.py",
    "tests/_helpers/torch_runtime.py",
    "tests/compat/torch/test_torchmetrics_compat.py",
    "tests/conftest.py",
    "tools/release/pack_offline.py",
    "tools/docs/check_build.py",
    "tools/docs/check_catalogs.py",
    "tools/docs/check_links.py",
    "tests/integration/test_notebooks.py",
    "tests/models/test_network_training_parity.py",
    "tests/structure/test_cleanup_structure.py",
    "tests/structure/test_docs_structure.py",
    "tests/structure/test_pytest_contract.py",
    "tests/structure/test_selftest_structure.py",
    "tests/structure/test_stage2_delivery.py",
)
FORMAT_FILES = (
    *NN_MIGRATION_FILES,
    "noxfile.py",
    "agent/scripts/check_sdist_contents.py",
    "agent/scripts/test_check_sdist_contents.py",
    "docs/_myst_autodoc.py",
    "docs/conf.py",
    "python/jittor/selftest.py",
    "tests/_helpers/torch_runtime.py",
    "tests/conftest.py",
    "tools/release/pack_offline.py",
    "tools/docs/check_build.py",
    "tools/docs/check_catalogs.py",
    "tools/docs/check_links.py",
    "tests/integration/test_notebooks.py",
    "tests/models/test_network_training_parity.py",
    "tests/structure/test_cleanup_structure.py",
    "tests/structure/test_docs_structure.py",
    "tests/structure/test_packaging_structure.py",
    "tests/structure/test_torch_shim_structure.py",
    "tests/structure/test_pytest_contract.py",
    "tests/structure/test_selftest_structure.py",
    "tests/structure/test_stage2_delivery.py",
)
STRUCTURE_TESTS = (
    "agent/scripts/test_check_sdist_contents.py",
    "agent/scripts/test_check_wheel_contents.py",
    "tests/structure",
)
# No CPU_TESTS list any more. The CPU gate runs the whole tree in two
# processes -- native semantics and Torch-compatibility semantics cannot share
# an interpreter -- and the only paths it skips are the ones that say why in
# tests/_helpers/gate_scope.py. The list this replaced reached 22 of 332 test
# files; everything else was written, merged and then never run by CI again.
CPU_TORCH_ORACLE_TESTS = (
    "tests/ops/test_cumprod_op.py",
    "tests/optim/test_adamw.py",
    "tests/optim/test_lr_scheduler.py",
    "tests/nn/test_affine_grid.py",
    "tests/nn/test_attention_oracle.py",
    "tests/nn/test_batchnorm.py",
    "tests/nn/test_loss.py",
    "tests/nn/test_relu.py",
    "tests/models/test_network_training_parity.py",
)
CUDA_TESTS = (
    "tests/backends/cuda",
    "tests/backends/parity/test_dtype_coverage.py",
    # The longest single entry in this gate by a wide margin: ~227 generated
    # cases at about a minute each. It stays in one process; the `cuda` session
    # below carries the measurement that says why.
    "tests/backends/parity/test_device_parity.py",
    "tests/compat/torch/test_torch_compat_cuda_tf32.py",
    "tests/ops/test_ops.py",
    "tests/models/test_network_training_parity.py",
)
OPTIONAL_COMPAT_PACKAGES = (
    "torchmetrics",
    "mmcv",
    "mmengine",
    "peft",
    "safetensors",
    "tensordict",
    "flash_attn",
)
OPTIONAL_COMPAT_TESTS = (
    "tests/compat/torch/test_torchmetrics_compat.py",
    "tests/compat/torch/test_mmcv_compat.py",
    "tests/compat/torch/test_peft.py",
    "tests/compat/torch/test_tensordict_compat.py",
    "tests/compat/torch/test_flash_attn_compat.py",
)
OPTIONAL_NATIVE_FLASH_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_dropout_replays_seed_and_backward",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_varlen_backward_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_qkvpacked_backward_matches_dense",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_mask_fallback_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_short_training_prefers_math",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_training_reuses_capability_checked_backend",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_higher_order_rejected_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_gqa_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_float32_opt_in_cast_cuda",
)
OPTIONAL_NATIVE_FLASH_BF16_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_dropout_replays_seed_and_backward_bf16",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_varlen_backward_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_qkvpacked_backward_matches_dense_bf16",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_mask_fallback_bf16_cuda",
)
OPTIONAL_NATIVE_FLASH_BF16_HDIM64_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim64_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_dropout_hdim64_bf16",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_varlen_backward_hdim64_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_qkvpacked_backward_hdim64_bf16",
)
OPTIONAL_NATIVE_FLASH_BF16_HDIM96_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim96_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim96_bf16",
)
OPTIONAL_NATIVE_FLASH_BF16_HDIM128_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim128_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim128_bf16",
)
OPTIONAL_NATIVE_FLASH_BF16_HDIM192_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim192_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim192_bf16",
)
OPTIONAL_NATIVE_FLASH_BF16_HDIM256_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim256_bf16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim256_bf16",
)
OPTIONAL_NATIVE_FLASH_HDIM64_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim64_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_dropout_hdim64_fp16",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_varlen_backward_hdim64_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_qkvpacked_backward_hdim64_fp16",
)
OPTIONAL_NATIVE_FLASH_HDIM96_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim96_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim96_fp16",
)
OPTIONAL_NATIVE_FLASH_HDIM128_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim128_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim128_fp16",
)
OPTIONAL_NATIVE_FLASH_HDIM192_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim192_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim192_fp16",
)
OPTIONAL_NATIVE_FLASH_HDIM256_TESTS = (
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_sdpa_native_flash_attn_backward_hdim256_fp16_cuda",
    "tests/compat/torch/test_torch_compat_attention.py::TestSDPA::test_native_flash_attn_training_variants_hdim256_fp16",
)
NPU_TESTS = (
    "tests/backends/npu/test_acl.py",
    "tests/backends/npu/test_acl_torch_compat.py",
    "tests/backends/npu/test_aclop.py",
    "tests/backends/npu/test_acl_indexing.py",
    "tests/ops/test_ops.py",
    "tests/core/test_floor_divide.py::TestFloorDivideNPU",
    "tests/compiler/test_kernel_traps.py::TestKernelTraps::test_nan_handling_isfinite_isnan_isinf",
    "tests/ops/test_fusion_correctness.py::TestFusionCorrectness::test_float_comparisons_with_nan",
)
ROCM_TESTS = ("tests/backends/rocm/test_rocm.py",)
MPI_TESTS = (
    "tests/distributed/test_mpi.py",
    "tests/distributed/test_mpi_batchnorm.py",
    "tests/distributed/test_mpi_op.py",
    "tests/distributed/test_single_process_scope.py",
)
NCCL_TESTS = ("tests/distributed/test_fsdp2_nccl.py",)

NOX_STATE_ROOT.mkdir(parents=True, exist_ok=True)
nox.options.envdir = str(NOX_STATE_ROOT / "envs")
nox.options.error_on_missing_interpreters = True
# Not stop_on_first_error: a gate exists to report what is broken, and stopping
# at the first failing session reports one failure per run. Finding the second
# one then costs another full run -- for the CPU gate, most of an hour (0.15).
nox.options.stop_on_first_error = False
nox.options.sessions = [
    "lint",
    "format",
    "typing",
    "structure",
    "cpu",
    "packaging",
    "py37",
    "py312",
    "py313",
]

for name, path in {
    "PIP_CACHE_DIR": NOX_STATE_ROOT / "cache" / "pip",
    "PRE_COMMIT_HOME": NOX_STATE_ROOT / "cache" / "pre-commit",
}.items():
    os.environ.setdefault(name, str(path))
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


_SESSION_ENV_PASSTHROUGH = (
    # Tool discovery and dynamic-library roots are properties of the runner.
    # Test behavior controls are deliberately absent and set below instead.
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "PKG_CONFIG_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_PATH",
    "ROCM_HOME",
    "ROCM_PATH",
    # Keep package downloads usable without inheriting unrelated test knobs.
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "PIP_EXTRA_INDEX_URL",
    "PIP_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    # Required by process creation on Windows.
    "COMSPEC",
    "PATHEXT",
    "SYSTEMROOT",
)

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _available_cpu_ids():
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return sorted(get_affinity(0))
        except OSError:
            pass
    return list(range(os.cpu_count() or 1))


def _physical_cores_in_affinity(cpu_ids):
    """Count physical cores in the inherited mask without importing Jittor."""
    cores = set()
    for cpu_id in cpu_ids:
        topology = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu_id)
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return len(cpu_ids)
        cores.add((package, core))
    return len(cores) or len(cpu_ids)


def _isolated_outer_environment():
    """Block Nox's implicit outer-env merge, then admit named runner inputs."""
    env = {name: None for name in os.environ}
    for name in _SESSION_ENV_PASSTHROUGH:
        if name in os.environ:
            env[name] = os.environ[name]
    return env


_PYTHON_CONFIG_PROBE = (
    "import os, sys; "
    "expected=set(map(int, os.environ['JITTOR_GATE_CPU_AFFINITY'].split(','))); "
    "actual=set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else expected; "
    "assert actual == expected, 'CPU affinity changed: %r != %r' % (actual, expected); "
    "assert int(os.environ['OMP_NUM_THREADS']) > 0; "
    "assert os.environ['OMP_PROC_BIND'] == 'false'; "
    "roots = [os.path.dirname(sys.executable), os.path.join(sys.base_prefix, 'bin')]; "
    "names = ['python3.%d-config' % sys.version_info[1], "
    "sys.executable + '-config', 'python3-config']; "
    "paths = [name if os.path.isabs(name) else os.path.join(root, name) "
    "for root in roots for name in names]; "
    "print(next((path for path in paths if os.path.isfile(path)), ''))"
)


def _set_python_config(session, python, env, external=False, required=False):
    """Select the config helper belonging to the interpreter that will run Jittor."""
    if os.name == "nt":
        env["python_config_path"] = None
        return
    python_config = session.run(
        python,
        "-c",
        _PYTHON_CONFIG_PROBE,
        env=env,
        external=external,
        silent=True,
    ).strip()
    if not python_config and required:
        session.error("Python config helper not found for %s" % python)
    if python_config:
        env["python_config_path"] = python_config
    else:
        env["python_config_path"] = None


def _shared_jittor_cache():
    """The Jittor cache root a session should use, honouring the opt-out."""
    if os.environ.get("JITTOR_NOX_SHARED_CACHE", "1") == "0":
        return None
    NOX_JITTOR_CACHE.mkdir(parents=True, exist_ok=True)
    return NOX_JITTOR_CACHE


def _session_env(session, backend):
    root = Path(session.create_tmp()).resolve()
    if root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True)
    paths = {
        "HOME": root / "home",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "JITTOR_TEST_STATE_ROOT": root / "test-state",
        "TMPDIR": root / "tmp",
        "CUDA_CACHE_PATH": root / "cuda-cache",
    }
    # Everything above is scratch and is wiped with the session. The Jittor
    # cache is not: it holds compiled products that are a pure function of the
    # build configuration, and rebuilding them per session is the single
    # largest cost in the gates.
    paths["JITTOR_HOME"] = _shared_jittor_cache() or (root / "jittor-home")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    cpu_ids = _available_cpu_ids()
    thread_count = str(_physical_cores_in_affinity(cpu_ids))
    env = _isolated_outer_environment()
    env.update({name: str(path) for name, path in paths.items()})
    env.update(
        {
            "BLIS_NUM_THREADS": thread_count,
            "JITTOR_GATE_CPU_AFFINITY": ",".join(str(cpu) for cpu in cpu_ids),
            "LC_ALL": "C",
            "MKL_DYNAMIC": "false",
            "MKL_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "OMP_DYNAMIC": "false",
            "OMP_NUM_THREADS": thread_count,
            "OMP_PROC_BIND": "false",
            "OPENBLAS_NUM_THREADS": thread_count,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf8",
            "PYTHONPATH": str(REPO_ROOT / "python"),
            "VECLIB_MAXIMUM_THREADS": thread_count,
            "cache_name": "nox_%s" % backend,
        }
    )
    if NOX_JITTOR_ASSETS.is_dir():
        env["JITTOR_OFFLINE_PATH"] = str(NOX_JITTOR_ASSETS)
    if session.python is False:
        env["python_config_path"] = None
    else:
        _set_python_config(session, "python", env)
    return root, env


def _source_copy(destination):
    ignored = shutil.ignore_patterns(
        ".git",
        ".claude",
        ".codex",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "*.egg-info",
        "*.py[co]",
        "dist",
    )

    def ignore_generated(path, names):
        excluded = set(ignored(path, names))
        # tools/build is source-owned; every other build directory is generated.
        if Path(path).resolve() != REPO_ROOT / "tools" and "build" in names:
            excluded.add("build")
        return excluded

    shutil.copytree(str(REPO_ROOT), str(destination), symlinks=True, ignore=ignore_generated)


def _mode_env(env, args):
    """The environment for one pytest invocation, with its process mode stated.

    Torch compatibility mode is process-global and is now chosen by
    ``JITTOR_TORCH_SHIM`` alone (0.13). It used to be inferred from the command
    line, which meant a session that listed both kinds of path silently ran the
    native ones under the shim -- and a session that listed only Torch-mode
    paths depended on that inference to work at all.
    """
    paths = [str(item).split("::", 1)[0] for item in args
             if not str(item).startswith("-")]
    mode = "1" if any(path.startswith(TORCH_MODE_PATHS) for path in paths) else "0"
    if env.get("JITTOR_TORCH_SHIM") == mode:
        return env
    scoped = env.copy()
    scoped["JITTOR_TORCH_SHIM"] = mode
    return scoped


def _by_process_mode(targets):
    """Group targets into the two process modes, order preserved.

    Torch compatibility mode is process-global, so the two groups cannot share
    an interpreter. Within a group they can -- and should: one pytest process
    per path pays the interpreter start and the jittor import once per path,
    and a per-path loop stops at the first failing path, so a gate reports one
    failure per run instead of all of them.
    """
    native, torch = [], []
    for target in targets:
        path = str(target).split("::", 1)[0]
        (torch if path.startswith(TORCH_MODE_PATHS) else native).append(target)
    return native, torch


#: What both CPU tiers install. One list, because a test that passes in the
#: fast tier and fails in the full one because a dependency was only in the
#: latter is a difference nobody can read off the failure.
CPU_GATE_REQUIREMENTS = (
    "astunparse==1.6.3",
    IPYKERNEL,
    JUPYTEXT,
    NBCLIENT,
    NBFORMAT,
    "numpy==1.26.4",
    "pillow==11.0.0",
    PYTEST,
    PYTEST_TIMEOUT,
    PYTEST_XDIST,
    SCIPY,
    SETUPTOOLS,
    "tqdm==4.67.1",
)

#: Workers a CPU gate runs with. Sized for the CI runner, not for the largest
#: machine anyone has: ``tests/_helpers/tiers.SMOKE_WORKERS`` is the same number
#: and the fast tier's budget is checked against it, so the two have to agree.
GATE_WORKERS = int(os.environ.get("JITTOR_GATE_WORKERS", "4"))


def _xdist(workers, distribution="loadfile"):
    """pytest-xdist arguments, or nothing when the gate stays in one process.

    ``loadfile`` by default, and that is the load-bearing part. Cross-file state
    leakage is a known and surveyed property of this tree (0.12/0.15): the
    survey in ``tests/conftest.py`` measures what each *file* leaves behind, and
    several files' tests depend on running in their written order. Distributing
    per test (``--dist load``) would break both at once and turn a catalogued
    problem into an irreproducible one. Per file, a worker sees a subset of the
    files in a fixed order -- the same kind of run the survey describes.

    ``load`` is correct for a *generated* battery where every case is
    independent by construction; ``test_device_parity.py`` is the one such
    caller (0.16) and passes it explicitly.
    """
    if not workers or workers <= 1:
        return ()
    return ("-n", str(workers), "--dist", distribution)


def _split_threads(env, workers):
    """Give each worker its share of the cores, once, at the top.

    Without this every worker starts one OpenMP thread per core it can see --
    all of them -- so N workers oversubscribe the machine N-fold. See
    ``tests/_helpers/tiers.worker_thread_budget``.
    """
    budget = worker_thread_budget(workers)
    if budget is None:
        return env
    env = env.copy()
    for name in _THREAD_ENV_NAMES:
        env[name] = str(budget)
    return env


def _run_pytest(session, defaults, env, runner=None):
    if session.posargs:
        _run_pytest_once(session, tuple(session.posargs),
                         _mode_env(env, session.posargs), runner, timeout=600)
        return
    for group in _by_process_mode(defaults):
        if group:
            _run_pytest_once(session, tuple(group), _mode_env(env, group),
                             runner, timeout=600)


def _run_pytest_once(session, args, env, runner=None, timeout=900):
    """One pytest process for one whole set of paths.

    A tree-wide gate runs as a single invocation on purpose: one process per
    path would pay the interpreter and import cost hundreds of times, and a
    per-path loop stops at the first failing path, which is how a gate ends up
    reporting one failure per run instead of all of them.
    """
    python = runner or "python"
    session.run(
        python,
        "-m",
        "pytest",
        "-v",
        "--timeout=%d" % timeout,
        *args,
        env=env,
        external=runner is not None,
    )


def _hardware_python():
    return os.environ.get("JITTOR_CI_PYTHON", sys.executable)


def _set_hardware_python_config(session, python, env):
    """Use the config helper belonging to an external hardware interpreter."""
    _set_python_config(session, python, env, external=True, required=True)


def _install_docs_wheel(session, root, env):
    """Build and install this checkout's wheel for a full autodoc run."""
    session.install(
        "-r",
        str(DOCS_REQUIREMENTS),
        BUILD,
        SETUPTOOLS,
        WHEEL,
        "astunparse==1.6.3",
        "numpy==1.26.4",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    source = root / "source"
    dist = root / "dist"
    _source_copy(source)
    with session.chdir(source):
        session.run(
            "python",
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(dist),
            env=env,
        )
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        session.error("expected exactly one documentation wheel, found %d" % len(wheels))
    session.install("--no-deps", "--force-reinstall", str(wheels[0]))

    docs_env = env.copy()
    docs_env["PYTHONPATH"] = None
    docs_env["PYTHONNOUSERSITE"] = "1"
    docs_env["nvcc_path"] = ""
    docs_env["cache_name"] = "nox_docs_wheel"
    docs_env["CUDA_VISIBLE_DEVICES"] = ""
    docs_env["use_cuda"] = "0"
    docs_env["use_mkl"] = "0"
    docs_env["use_mpi"] = "0"
    docs_env["use_nccl"] = "0"
    docs_env["use_cutt"] = "0"
    docs_env["use_cutlass"] = "0"
    probe = r"""
from pathlib import Path
import sys
import jittor

module_path = Path(jittor.__file__).resolve()
repo_root = Path(sys.argv[1]).resolve()
prefix = Path(sys.prefix).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise SystemExit("autodoc imported the source tree: %s" % module_path)
if prefix != module_path and prefix not in module_path.parents:
    raise SystemExit("autodoc did not import from the nox environment: %s" % module_path)
print("autodoc wheel:", module_path)
"""
    with session.chdir(root):
        session.run("python", "-c", probe, str(REPO_ROOT), env=docs_env)
    return docs_env


def _sphinx_html(session, root, env, language, source_root=None):
    output = root / "html" / language
    source_root = source_root or REPO_ROOT / "docs"
    session.run(
        "python",
        "-m",
        "sphinx",
        "-E",
        "-a",
        "-W",
        "--keep-going",
        "-n",
        "-b",
        "html",
        "-D",
        "language=%s" % language,
        str(source_root),
        str(output),
        env=env,
    )
    return output


def _run_with_cann(session, python, args, env):
    session.run(
        "bash",
        "-eu",
        "-o",
        "pipefail",
        "-c",
        ': "${LD_LIBRARY_PATH:=}"; : "${CMAKE_PREFIX_PATH:=}"; '
        'source "$CANN_SET_ENV"; exec "$JITTOR_CI_PYTHON" "$@"',
        "jittor-npu",
        *args,
        env=env,
        external=True,
    )


def _asv_state_path(variable, fallback):
    raw_path = os.environ.get(variable)
    path = Path(raw_path).expanduser() if raw_path else fallback
    path = path.resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise RuntimeError("%s must point outside the source checkout: %s" % (variable, path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_asv_config(root, results_dir, html_dir):
    config = json.loads((REPO_ROOT / "asv.conf.json").read_text(encoding="utf-8"))
    config.update(
        {
            "repo": str(REPO_ROOT),
            "benchmark_dir": str(REPO_ROOT / "benchmarks"),
            "env_dir": str(root / "asv-env"),
            "results_dir": str(results_dir),
            "html_dir": str(html_dir),
        }
    )
    config_path = root / "asv-ci.conf.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def _asv_result_commits(results_dir):
    commits = set()
    for path in results_dir.glob("*/*.json"):
        if path.name in ("benchmarks.json", "machine.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        commit = data.get("commit_hash")
        if isinstance(commit, str) and commit:
            commits.add(commit)
    return commits


def _asv_has_measurement(results_dir, commit_hash):
    def has_finite_number(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, list):
            return any(has_finite_number(item) for item in value)
        return False

    for path in results_dir.glob("*/*.json"):
        if path.name in ("benchmarks.json", "machine.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("commit_hash") != commit_hash:
            continue
        for result in data.get("results", {}).values():
            measurements = result[0] if isinstance(result, list) and result else result
            if has_finite_number(measurements):
                return True
    return False


def _git_output(*arguments):
    result = subprocess.run(
        ("git",) + arguments,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _asv_compare_base(results_dir, current_commit):
    commits = _asv_result_commits(results_dir)
    commits.discard(current_commit)
    requested = os.environ.get("ASV_COMPARE_BASE", "").strip()
    if requested:
        resolved = _git_output("rev-parse", "--verify", "%s^{commit}" % requested)
        if resolved in commits:
            is_ancestor = subprocess.run(
                ("git", "merge-base", "--is-ancestor", resolved, current_commit),
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if is_ancestor.returncode == 0:
                return resolved

    ancestors = []
    for commit in commits:
        is_ancestor = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, current_commit),
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if is_ancestor.returncode != 0:
            continue
        timestamp = _git_output("show", "-s", "--format=%ct", commit)
        if timestamp and timestamp.isdigit():
            ancestors.append((int(timestamp), commit))
    return max(ancestors)[1] if ancestors else None


_PREFETCH_SCRIPT = """
import os, platform, sys
from jittor_utils import manifest
from jittor_utils.misc import download_url_to_local

destination = sys.argv[1]
os.makedirs(destination, exist_ok=True)
system = platform.system().lower().replace("darwin", "darwin")
machine = platform.machine()
wanted = "linux-x86_64" if (system, machine) == ("linux", "x86_64") else None
fetched, skipped = [], []
for asset in manifest.offline_assets(include_cuda=False):
    # Only what this platform's build will actually ask for. The CUDA
    # toolkits are excluded above: they are gigabytes, and a machine that
    # needs one is not one this mirror is for.
    if asset.platform not in ("any", wanted):
        skipped.append(asset.filename)
        continue
    digest = manifest.digest_of(asset)[1]
    if not digest:
        # Nothing to verify it against, so it is not something to mirror.
        skipped.append(asset.filename)
        continue
    try:
        download_url_to_local(asset.url, asset.filename, destination, digest)
        fetched.append(asset.filename)
    except Exception as error:
        print("could not prefetch %s: %s" % (asset.filename, error))
print("mirror at", destination)
print("present:", len(fetched), "skipped:", len(skipped))
"""


@nox.session(python="3.11", venv_backend="venv")
def prefetch(session):
    """Fill the shared third-party mirror so later sessions need no network.

    Every gate session used to download MKL, cub, cutt and NCCL from one host
    in Beijing, because every session started from an empty cache. Sharing the
    cache removes the repeat downloads on a machine that has run once; this
    session is what makes the *first* run on a fresh machine, or a CI job whose
    only restorable state is a directory, not need the network either. Point
    JITTOR_OFFLINE_PATH at the directory it fills, or let `_session_env` do it.
    """
    session.install("tqdm")
    NOX_JITTOR_ASSETS.mkdir(parents=True, exist_ok=True)
    session.run(
        "python", "-c", _PREFETCH_SCRIPT, str(NOX_JITTOR_ASSETS),
        env={"PYTHONPATH": str(REPO_ROOT / "python")},
    )


@nox.session(python="3.11")
def lint(session):
    """Run the ratcheted, Python 3.7-aware Ruff lint baseline."""
    session.install(RUFF)
    session.run("ruff", "check", "--no-cache", *RATCHET_FILES)


@nox.session(python="3.11")
def format(session):
    """Check Ruff formatting for files admitted to the format ratchet."""
    session.install(RUFF)
    session.run("ruff", "format", "--check", "--no-cache", *FORMAT_FILES)


@nox.session(python="3.11")
def typing(session):
    """Type-check the explicit mypy ratchet without writing a repository cache."""
    cache_dir = str(Path(session.create_tmp()) / "mypy-cache")
    session.install(MYPY)
    session.run("mypy", "--cache-dir", cache_dir)


@nox.session(python="3.11")
def structure(session):
    """Run the fast layout, checker, and complete structure-test gate."""
    _root, env = _session_env(session, "structure")
    env.update(
        {
            # tests/structure is a Torch-mode path (process_modes.TORCH_MODE_PATHS):
            # several of its modules import the shim at module scope. The mode is
            # now stated rather than guessed from the command line (0.13).
            "JITTOR_TORCH_SHIM": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "nvcc_path": "",
            "use_cuda": "0",
            "use_cutt": "0",
            "use_cutlass": "0",
            "use_mkl": "0",
            "use_mpi": "0",
            "use_nccl": "0",
            "use_parallel_op_compiler": "0",
        }
    )
    session.install(
        PYTEST,
        PYTEST_TIMEOUT,
        SETUPTOOLS,
        "astunparse==1.6.3",
        JUPYTEXT,
        NBFORMAT,
        "numpy==1.26.4",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    session.run("bash", "agent/scripts/check_repo_layout.sh", external=True, env=env)
    test_paths = tuple(session.posargs) or STRUCTURE_TESTS
    session.run(
        "python",
        "-m",
        "pytest",
        "-v",
        "--timeout=600",
        *test_paths,
        env=env,
    )


@nox.session(python="3.11")
def packaging(session):
    """Build, audit, install, and self-test direct and sdist-derived artifacts."""
    root, env = _session_env(session, "packaging")
    session.install(
        BUILD,
        PYTEST,
        PYTEST_TIMEOUT,
        SETUPTOOLS,
        WHEEL,
        "astunparse==1.6.3",
        "numpy==1.26.4",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )

    source = root / "source"
    dist = root / "dist"
    for path in (source, dist):
        if path.exists():
            shutil.rmtree(str(path))
    _source_copy(source)
    with session.chdir(source):
        session.run(
            "python",
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist),
            env=env,
        )
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1:
        session.error("expected exactly one wheel, found %d" % len(wheels))
    if len(sdists) != 1:
        session.error("expected exactly one sdist, found %d" % len(sdists))
    session.run(
        "python",
        "agent/scripts/check_sdist_contents.py",
        str(sdists[0]),
        env=env,
    )
    sdist_wheel_dist = root / "sdist-wheel-dist"
    session.run(
        "python",
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(sdist_wheel_dist),
        str(sdists[0]),
        env=env,
    )
    sdist_wheels = sorted(sdist_wheel_dist.glob("*.whl"))
    if len(sdist_wheels) != 1:
        session.error("expected exactly one sdist-derived wheel, found %d" % len(sdist_wheels))
    wheel_args = tuple(session.posargs)
    for wheel in (wheels[0], sdist_wheels[0]):
        session.run(
            "python",
            "agent/scripts/check_wheel_contents.py",
            "compare",
            str(wheel),
            *wheel_args,
            env=env,
        )

    wheel_install = root / "wheel-install"
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(wheel_install),
        str(wheels[0]),
        env=env,
    )
    selftest_env = env.copy()
    selftest_env["PYTHONPATH"] = str(wheel_install)
    selftest_env["PYTHONNOUSERSITE"] = "1"
    selftest_env["nvcc_path"] = ""
    selftest_env["cache_name"] = "nox_wheel_selftest"
    with session.chdir(root):
        session.run("python", "-m", "jittor.selftest", env=selftest_env)


@nox.session(python="3.11", venv_backend="venv")
def docs(session):
    """Build strict English HTML from an installed wheel and audit API anchors."""
    root, env = _session_env(session, "docs-en")
    docs_env = _install_docs_wheel(session, root, env)
    html = _sphinx_html(session, root, docs_env, "en")
    session.run(
        "python",
        str(REPO_ROOT / "tools" / "docs" / "check_build.py"),
        "--en",
        str(html),
        env=docs_env,
    )


@nox.session(python="3.11", venv_backend="venv")
def docs_zh(session):
    """Check gettext freshness and build strict English and Simplified Chinese HTML."""
    root, env = _session_env(session, "docs-zh")
    docs_env = _install_docs_wheel(session, root, env)
    gettext_root = root / "gettext"
    session.run(
        "python",
        "-m",
        "sphinx",
        "-E",
        "-a",
        "-W",
        "--keep-going",
        "-n",
        "-b",
        "gettext",
        str(REPO_ROOT / "docs"),
        str(gettext_root),
        env=docs_env,
    )
    localized_source = root / "docs-source"
    shutil.copytree(str(REPO_ROOT / "docs"), str(localized_source))
    catalog_copy = localized_source / "locales"
    session.run(
        "sphinx-intl",
        "update",
        "-p",
        str(gettext_root),
        "-d",
        str(catalog_copy),
        "-l",
        "zh_CN",
        env=docs_env,
    )
    session.run(
        "python",
        str(REPO_ROOT / "tools" / "docs" / "check_catalogs.py"),
        str(REPO_ROOT / "docs" / "locales"),
        str(catalog_copy),
        env=docs_env,
    )
    english = _sphinx_html(session, root, docs_env, "en", localized_source)
    chinese = _sphinx_html(session, root, docs_env, "zh_CN", localized_source)
    session.run(
        "python",
        str(REPO_ROOT / "tools" / "docs" / "check_build.py"),
        "--en",
        str(english),
        "--zh-cn",
        str(chinese),
        env=docs_env,
    )


@nox.session(python="3.11", venv_backend="venv")
def docs_links(session):
    """Check deterministic internal Markdown, image, MyST role, and toctree targets."""
    _root, env = _session_env(session, "docs-links")
    session.run(
        "python",
        str(REPO_ROOT / "tools" / "docs" / "check_links.py"),
        env=env,
    )


@nox.session(python="3.11", venv_backend="venv")
def tutorials(session):
    """Materialize MyST sources and execute five offline CPU tutorial smokes."""
    _root, env = _session_env(session, "tutorials")
    env["nvcc_path"] = ""
    env["JITTOR_TEST_DEVICES"] = "cpu"
    session.install(
        "-r",
        str(DOCS_REQUIREMENTS),
        "astunparse==1.6.3",
        "numpy==1.26.4",
        "pillow==11.0.0",
        PYTEST,
        PYTEST_TIMEOUT,
        SETUPTOOLS,
        "tqdm==4.67.1",
    )
    _run_pytest(session, ("tests/integration/test_notebooks.py",), env)


def _record_asv(session, root, env, asv_command, default_machine, external=False):
    asv_home = root / "jittor-asv-home"
    asv_home.mkdir(parents=True, exist_ok=True)
    env["JITTOR_HOME"] = str(asv_home)
    env["JITTOR_ASV_HOME"] = str(asv_home)
    env["ASV_CONF_DIR"] = str(REPO_ROOT)
    # ASV deliberately removes PYTHONPATH before launching an existing
    # environment. ASV_PYTHONPATH is its supported source-tree escape hatch.
    env["ASV_PYTHONPATH"] = str(REPO_ROOT / "python")
    machine = os.environ.get("ASV_MACHINE", default_machine)
    factor = os.environ.get("ASV_COMPARE_FACTOR", "1.10")
    try:
        if float(factor) <= 1.0:
            raise ValueError
    except ValueError:
        session.error("ASV_COMPARE_FACTOR must be a number greater than 1.0")

    results_dir = _asv_state_path("ASV_RESULTS_DIR", root / "asv-results")
    html_dir = _asv_state_path("ASV_HTML_DIR", root / "asv-html")
    config_path = _write_asv_config(root, results_dir, html_dir)
    current_commit = _git_output("rev-parse", "HEAD")
    if not current_commit:
        session.error("cannot resolve the current commit for ASV")
    dirty = _git_output("status", "--porcelain", "--untracked-files=all")
    if dirty and os.environ.get("ASV_ALLOW_DIRTY") != "1":
        session.error(
            "ASV refuses to label a dirty checkout as %s; commit or stash changes first"
            % current_commit
        )
    compare_base = _asv_compare_base(results_dir, current_commit)

    with session.chdir(REPO_ROOT):
        session.run(
            *(tuple(asv_command) + ("check", "--config", str(config_path), "--python=same")),
            env=env,
            external=external,
        )
        session.run(
            *(
                tuple(asv_command)
                + (
                    "machine",
                    "--config",
                    str(config_path),
                    "--machine",
                    machine,
                    "--yes",
                )
            ),
            env=env,
            external=external,
        )
        session.run(
            *(
                tuple(asv_command)
                + (
                    "run",
                    "--config",
                    str(config_path),
                    "--python=same",
                    "--set-commit-hash",
                    current_commit,
                    "--machine",
                    machine,
                    "--record-samples",
                    "--show-stderr",
                    "--no-pull",
                )
                + tuple(session.posargs)
            ),
            env=env,
            external=external,
        )
        if not _asv_has_measurement(results_dir, current_commit):
            session.error("ASV produced no finite measurements for %s" % current_commit)
        if compare_base is None:
            compare_base = current_commit
            session.log("no cached ancestor result; bootstrapping ASV comparison")
        session.run(
            *(
                tuple(asv_command)
                + (
                    "compare",
                    "--config",
                    str(config_path),
                    "--python=same",
                    "--machine",
                    machine,
                    "--factor",
                    factor,
                    "--split",
                    compare_base,
                    current_commit,
                )
            ),
            env=env,
            external=external,
        )
        session.run(
            *(
                tuple(asv_command)
                + (
                    "publish",
                    "--config",
                    str(config_path),
                    "--no-pull",
                    "--html-dir",
                    str(html_dir),
                )
            ),
            env=env,
            external=external,
        )

    if not any(results_dir.glob("*/*.json")):
        session.error("ASV produced no result files in %s" % results_dir)
    if not (html_dir / "index.html").is_file():
        session.error("ASV publish did not create %s" % (html_dir / "index.html"))
    session.log("ASV results: %s" % results_dir)
    session.log("ASV HTML: %s" % html_dir)


@nox.session(python="3.11")
def benchmark(session):
    """Record selected CPU benchmarks for this commit and publish ASV HTML."""
    root, env = _session_env(session, "asv-cpu")
    env["cache_name"] = "asv-nox-cpu"
    env["nvcc_path"] = ""
    session.install(
        ASV,
        "astunparse==1.6.3",
        "numpy==1.26.4",
        "pillow==11.0.0",
        SETUPTOOLS,
        "tqdm==4.67.1",
    )
    _record_asv(session, root, env, ("asv",), "jittor-ci-cpu")


@nox.session(python=False)
def benchmark_cuda(session):
    """Record scheduled Tiny Llama and optimizer ASV results on real CUDA."""
    root, env = _session_env(session, "asv-cuda")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    if not nvcc:
        session.error("CUDA benchmark requires nvcc_path or nvcc on PATH")
    env["nvcc_path"] = nvcc
    env["cache_name"] = "asv-nox-cuda"
    session.run("nvidia-smi", external=True, env=env)
    session.run(nvcc, "--version", external=True, env=env)
    dependency_probe = (
        "import asv, numpy, transformers; "
        "assert transformers.__version__ == '4.56.2'; "
        "print('CUDA ASV dependencies OK')"
    )
    session.run(python, "-c", dependency_probe, external=True, env=env)
    _record_asv(
        session,
        root,
        env,
        (python, "-m", "asv"),
        "jittor-ci-rtx4090-cuda12-2",
        external=True,
    )


@nox.session(python="3.7", venv_backend="venv")
def py37(session):
    """Compile every repository Python file with a real Python 3.7 interpreter."""
    _root, env = _session_env(session, "py37")
    script = r"""
import pathlib
import subprocess
import sys

if sys.version_info[:2] != (3, 7):
    raise SystemExit("py37 requires Python 3.7, found %s" % (sys.version.split()[0],))

root = pathlib.Path(sys.argv[1]).resolve()
listed = subprocess.check_output([
    "git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
    "--exclude-standard", "--", "*.py",
]).decode("utf-8").split("\0")
failed = []
checked = 0
for relative in sorted(item for item in listed if item):
    path = root / relative
    checked += 1
    try:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    except (SyntaxError, UnicodeError) as error:
        failed.append("%s: %s" % (path.relative_to(root), error))
if failed:
    print("Python 3.7 compile failures:")
    print("\n".join(failed))
    raise SystemExit(1)
print("Python 3.7 compile OK: %d files" % checked)
"""
    session.run("python", "-c", script, str(REPO_ROOT), env=env)


def _upper_python_compatibility(session, version, session_name, numpy_requirement):
    """Build and exercise an installed wheel on a maintained upper Python version."""
    root, env = _session_env(session, session_name)
    _set_python_config(session, "python", env, required=True)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DISABLE_MULTIPROCESSING": "1",
            "JITTOR_TEST_DEVICES": "cpu",
            "nvcc_path": "",
            "use_cuda": "0",
            "use_cutt": "0",
            "use_cutlass": "0",
            "use_mkl": "0",
            "use_mpi": "0",
            "use_nccl": "0",
            "use_parallel_op_compiler": "0",
        }
    )
    session.install(
        BUILD,
        SETUPTOOLS,
        WHEEL,
        "astunparse==1.6.3",
        numpy_requirement,
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    compile_script = r"""
import pathlib
import subprocess
import sys
import warnings

version = tuple(int(part) for part in sys.argv[2].split("."))
session_name = sys.argv[3]
if sys.version_info[:2] != version:
    raise SystemExit("%s requires Python %s, found %s" % (
        session_name, sys.argv[2], sys.version.split()[0]))

warnings.simplefilter("error", SyntaxWarning)
root = pathlib.Path(sys.argv[1]).resolve()
listed = subprocess.check_output([
    "git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
    "--exclude-standard", "--", "*.py",
]).decode("utf-8").split("\0")
failed = []
checked = 0
for relative in sorted(item for item in listed if item):
    path = root / relative
    checked += 1
    try:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    except (SyntaxError, UnicodeError) as error:
        failed.append("%s: %s" % (path.relative_to(root), error))
if failed:
    print("Python %s compile failures:" % sys.argv[2])
    print("\n".join(failed))
    raise SystemExit(1)
print("Python %s compile OK without SyntaxWarning: %d files" % (sys.argv[2], checked))
"""
    session.run(
        "python",
        "-W",
        "error::SyntaxWarning",
        "-c",
        compile_script,
        str(REPO_ROOT),
        version,
        session_name,
        env=env,
    )

    source = root / "source"
    dist = root / "dist"
    _source_copy(source)
    with session.chdir(source):
        session.run(
            "python",
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(dist),
            env=env,
        )
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        session.error("expected exactly one Python %s wheel, found %d" % (version, len(wheels)))

    wheel_install = root / "wheel-install"
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(wheel_install),
        str(wheels[0]),
        env=env,
    )
    selftest_env = env.copy()
    selftest_env["PYTHONPATH"] = str(wheel_install)
    selftest_env["PYTHONNOUSERSITE"] = "1"
    selftest_env["cache_name"] = "nox_%s_wheel_selftest" % session_name
    with session.chdir(root):
        session.run(
            "python",
            "-W",
            "error::SyntaxWarning",
            "-m",
            "jittor.selftest",
            env=selftest_env,
        )
        compatibility_probe = r"""
import numpy as np
import jittor as jt

expected_numpy_major = int(__import__("sys").argv[1])
if int(np.__version__.split(".")[0]) != expected_numpy_major:
    raise SystemExit("expected NumPy major %d, found %s" % (
        expected_numpy_major, np.__version__))

class Scale(jt.Module):
    def execute(self, value):
        return value * 2

source = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
with jt.flag_scope(trace_py_var=2):
    result = Scale()(jt.array(source))
    np.testing.assert_array_equal(result.numpy(), source * 2)
    trace = jt.dump_trace_data()
    jt.clear_trace_data()
if not trace["node_data"]:
    raise SystemExit("trace_py_var produced no node data")
print("Python compatibility probe passed with NumPy", np.__version__)
"""
        expected_numpy_major = "2" if version == "3.13" else "1"
        session.run("python", "-c", compatibility_probe, expected_numpy_major, env=selftest_env)


@nox.session(python="3.12", venv_backend="venv")
def py312(session):
    """Build, install, and execute Jittor with a real Python 3.12 interpreter."""
    _upper_python_compatibility(session, "3.12", "py312", "numpy==1.26.4")


@nox.session(python="3.13", venv_backend="venv")
def py313(session):
    """Build, install, and execute Jittor with a real Python 3.13 interpreter."""
    _upper_python_compatibility(session, "3.13", "py313", "numpy>=2.1,<3.0")


#: One CPU probe for both tiers: the gate has to prove it is running the tree it
#: thinks it is before it reports anything about the tree.
_CPU_PROBE = (
    "import jittor as jt; "
    "assert not jt.compiler.has_cuda; "
    "assert not getattr(jt.compiler, 'has_acl', 0); "
    "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
    "assert float(x.item()) == 6.0"
)


def _cpu_gate_env(session):
    """The environment both CPU tiers run in, and what it refuses to inherit."""
    _root, env = _session_env(session, "cpu")
    # Nox overlays this mapping on the parent environment, so removing the key
    # would still leak a caller-provided real Torch into ordinary Jittor tests.
    env["REAL_TORCH_SITE"] = ""
    env["nvcc_path"] = ""
    env["JITTOR_TEST_DEVICES"] = "cpu"
    # Explicit, not inherited: a developer with the shim exported would
    # otherwise run the native half of the gate in Torch mode and never know.
    env["JITTOR_TORCH_SHIM"] = "0"
    # Stated rather than defaulted. The gate compiles kernels in parallel --
    # that is where a whole-tree run's time goes -- and an "asynchronous compile
    # error lands on the wrong test" report used to be answered by switching it
    # off somewhere (see 0.16). It is a lifetime bug, not a concurrency one, and
    # it is fixed where it lives; the gate does not pay for the workaround.
    env["use_parallel_op_compiler"] = os.environ.get(
        "use_parallel_op_compiler", "16")
    return env


def _require_execution(env):
    """0.18, enforced rather than only reported.

    ``tests/conftest.py`` says "the gates do" set this; until now only the
    nightly ecosystem session did, so the CPU gate printed the per-file
    accounting and then passed regardless. Measured before switching it on: a
    whole-tree native run reports 81 files that executed nothing and **all 81
    are explained** by a skip reason naming something this machine lacks (no
    CUDA, no ACL, no independent PyTorch). Zero unexplained, so the flag costs
    nothing today and catches the next entry that quietly stops testing.
    """
    env = env.copy()
    env["JITTOR_TEST_REQUIRE_EXECUTION"] = "1"
    return env


@nox.session(python="3.11", venv_backend="venv")
def smoke(session):
    """The pull-request tier: the whole tree minus the recorded slow files.

    Fast is the whole point, so say what is traded for it. This tier runs every
    file ``gate_scope`` reaches except those listed in
    ``tests/_helpers/tiers.SLOW_FILES``, each with its measured cost and the
    reason it is worth an entry; ``nox -s cpu`` runs all of them and nothing is
    dropped from the tree. The oracle comparison against an independent PyTorch
    and the manual probes stay in the full tier too: both need a second
    interpreter or a Jupyter kernel, and neither is what a pull request is
    waiting to hear.
    """
    env = _require_execution(_cpu_gate_env(session))
    session.install(*CPU_GATE_REQUIREMENTS)
    session.run("python", "-c", _CPU_PROBE, env=env)
    if session.posargs:
        _run_pytest(session, (), env)
        return
    fast = ("-m", "not slow") + _xdist(GATE_WORKERS)
    env = _split_threads(env, GATE_WORKERS)
    _run_pytest_once(session, gate_native_arguments() + fast, env)
    torch_env = env.copy()
    torch_env["JITTOR_TORCH_SHIM"] = "1"
    _run_pytest_once(session, gate_torch_arguments() + fast, torch_env)


@nox.session(python="3.11", venv_backend="venv")
def cpu(session):
    """The full tier: the whole test tree on CPU, in both process modes.

    Everything ``nox -s smoke`` runs plus everything it defers: the files listed
    in ``tests/_helpers/tiers.SLOW_FILES``, the manual probes, and the oracle
    comparison against an independent PyTorch. Nothing here is optional -- the
    fast tier is an ordering, not a subset of what is verified.
    """
    real_torch_site = os.environ.get("REAL_TORCH_SITE", "").strip()
    require_real_torch = os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").strip() == "1"
    if require_real_torch and not real_torch_site:
        session.error("JITTOR_REQUIRE_REAL_TORCH=1 requires REAL_TORCH_SITE")
    env = _cpu_gate_env(session)
    session.install(*CPU_GATE_REQUIREMENTS)
    session.run("python", "-c", _CPU_PROBE, env=env)
    if session.posargs:
        _run_pytest(session, (), env)
        return
    # Two processes, one tree. Torch compatibility mode is process-global -- it
    # changes lazy execution, reduction defaults and gradient semantics -- so a
    # single `pytest tests` run cannot assert both. Each session still selects
    # by exclusion, so a new test file is gated the moment it is written.
    parallel = _xdist(GATE_WORKERS)
    full_env = _require_execution(_split_threads(env, GATE_WORKERS))
    _run_pytest_once(session, gate_native_arguments() + parallel, full_env)
    torch_env = full_env.copy()
    torch_env["JITTOR_TORCH_SHIM"] = "1"
    _run_pytest_once(session, gate_torch_arguments() + parallel, torch_env)
    # The manual probes get their own process, which is the whole reason they
    # are marked: they drive Jupyter kernels and spawn their own children, and
    # they do not survive being run after a thousand other tests have already
    # used the runtime. Running them here rather than leaving them opt-in keeps
    # them gated -- before 0.13 they ran inside the native session by accident
    # (the marker was attached after the decision that reads it) and cost 537 s
    # there; on their own they take about the same and mean something.
    manual_env = env.copy()
    manual_env["JITTOR_TEST_MANUAL"] = "1"
    _run_pytest_once(
        session, gate_native_arguments() + ("-m", "manual"),
        manual_env, timeout=1800)
    oracle_env = env.copy()
    if real_torch_site:
        oracle_env["REAL_TORCH_SITE"] = real_torch_site
    elif require_real_torch:
        session.error("independent PyTorch oracle is required but unavailable")
    _run_pytest(session, CPU_TORCH_ORACLE_TESTS, oracle_env)


@nox.session(python="3.11", venv_backend="venv")
def full(session):
    """Stable name for the complete CPU/nightly gate (10.01)."""
    cpu(session)


#: The ecosystem comparison: numbers first, then wall clock.
ECOSYSTEM_TESTS = (
    "tests/compat/torch/test_ecosystem_parity.py",
    "tests/compat/torch/test_ecosystem_speed.py",
)

#: Speed ceiling for the nightly ecosystem gate: Jittor may take at most this
#: multiple of PyTorch's wall clock.
#:
#: Asserted in nightly only, never in the PR gate. A wall-clock upper bound is
#: the one kind of assertion a loaded machine can fail on its own (see the
#: `load_sensitive` marker), and a false red on the speed gate costs more than
#: a late true red: it teaches people to ignore the gate that also checks the
#: numbers. The ratio is always *reported*, whether or not it is asserted.
ECOSYSTEM_SPEED_RATIO = "1.07"


@nox.session(python="3.11", venv_backend="venv")
def ecosystem(session):
    """Compare seven downstream frameworks against an independent PyTorch.

    This is the project's headline claim -- "import torch and your code runs,
    with the same numbers and comparable speed" -- and until now it had no
    automatic gate at all. The comparison needs two interpreters, because real
    PyTorch and the Jittor shim both own the name ``torch``: ``REAL_TORCH_PYTHON``
    produces the weights and the reference values, this one recomputes from the
    same weights.

    Fail-closed on purpose. These tests skip themselves when
    ``REAL_TORCH_PYTHON`` is unset -- correctly, since comparing the shim
    against itself would prove nothing -- so a nightly job that lost its oracle
    would otherwise report success for the one thing it exists to check.
    ``JITTOR_REQUIRE_REAL_TORCH=1`` turns "skipped for want of torch" into a
    failure that names every case that did not happen.
    """
    _root, env = _session_env(session, "ecosystem")
    oracle = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not oracle:
        session.error(
            "REAL_TORCH_PYTHON must name an interpreter whose `import torch` is "
            "an independent binary PyTorch. Comparing Jittor's shim against "
            "itself is self-referential, which is why the cases skip without it "
            "-- and why this session refuses to run rather than report a green "
            "that means nothing."
        )
    if not Path(oracle).is_file():
        session.error("REAL_TORCH_PYTHON is not an interpreter: %s" % oracle)
    session.install(
        "numpy==1.26.4",
        "pillow==11.0.0",
        PYTEST,
        PYTEST_TIMEOUT,
        SCIPY,
        SETUPTOOLS,
        "tqdm==4.67.1",
    )
    env.update(
        {
            "REAL_TORCH_PYTHON": oracle,
            "JITTOR_REQUIRE_REAL_TORCH": "1",
            "JITTOR_TEST_REQUIRE_EXECUTION": "1",
            "JITTOR_TORCH_SHIM": "1",
            "JITTOR_ECOSYSTEM_SPEED_RATIO": os.environ.get(
                "JITTOR_ECOSYSTEM_SPEED_RATIO", ECOSYSTEM_SPEED_RATIO),
            # Without this the whole speed half skips itself -- "set
            # JITTOR_ECOSYSTEM_LARGE=1 to run the realistic-size measurement"
            # -- and a nightly speed gate that measures nothing is the same
            # fail-open this session exists to close, one layer down. The
            # realistic sizes *are* the measurement; toy sizes would compare
            # framework overhead rather than the work.
            "JITTOR_ECOSYSTEM_LARGE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    reference_site = os.environ.get(
        "JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE", "").strip()
    if reference_site:
        env["JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE"] = reference_site
    # Both interpreters report which device they ran on, and the harness fails
    # if they disagree -- a CPU-versus-GPU comparison would be meaningless.
    env.setdefault("JITTOR_TEST_DEVICES", os.environ.get("JITTOR_TEST_DEVICES", "cpu"))
    session.run(
        oracle, "-c",
        "import torch, sys; "
        "assert not hasattr(torch, '_torch_compat_install_context'), "
        "'REAL_TORCH_PYTHON resolves to the Jittor shim, not an independent build'; "
        "print('oracle torch', torch.__version__)",
        external=True, env=env,
    )
    _run_pytest_once(session, ECOSYSTEM_TESTS, env, timeout=3600)


@nox.session(python=False)
def optional(session):
    """Run fail-closed optional compatibility gates on real CUDA."""
    _root, env = _session_env(session, "optional")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    if not nvcc:
        session.error("optional session requires nvcc_path or nvcc on PATH")
    flash_source = os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC", "").strip()
    if flash_source:
        flash_root = Path(flash_source).expanduser().resolve()
        flash_api = flash_root / "csrc" / "flash_attn" / "flash_api.cpp"
        if not flash_api.is_file():
            session.error(
                "JITTOR_FLASH_ATTN_JITTOR_SRC is not an official flash-attn checkout"
            )
        flash_source = os.fspath(flash_root)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "JITTOR_REQUIRE_OPTIONAL_DEPS": "1",
            "JITTOR_TEST_DEVICES": "cuda",
            "JITTOR_TORCH_SHIM": "1",
            "JITTOR_FLASH_ATTN_JITTOR_REQUIRED": "0",
            "JITTOR_FLASH_ATTN_JITTOR_SRC": "",
            "TRANSFORMERS_OFFLINE": "1",
            "nvcc_path": nvcc,
            "use_cuda": "1",
        }
    )
    native_env = env.copy()
    if flash_source:
        native_env["JITTOR_FLASH_ATTN_JITTOR_REQUIRED"] = "1"
        native_env["JITTOR_FLASH_ATTN_JITTOR_SRC"] = flash_source
        requested_head_dims = (
            os.environ.get("JITTOR_FLASH_ATTN_HEAD_DIMS")
            or os.environ.get("FLASH_ATTN_HEAD_DIMS")
            or ""
        )
        requested_dtypes = (
            os.environ.get("JITTOR_FLASH_ATTN_DTYPES")
            or os.environ.get("FLASH_ATTN_DTYPES")
            or ""
        )
        if requested_head_dims.strip().lower() in {"all", "full", "*"}:
            native_env["JITTOR_FLASH_ATTN_HEAD_DIMS"] = "all"
        else:
            head_dims = ["32"] + [
                item.strip() for item in requested_head_dims.replace(";", ",").split(",")
                if item.strip()
            ]
            native_env["JITTOR_FLASH_ATTN_HEAD_DIMS"] = ",".join(
                dict.fromkeys(head_dims))
        if requested_dtypes.strip().lower() in {"all", "full", "*"}:
            native_env["JITTOR_FLASH_ATTN_DTYPES"] = "all"
        else:
            dtypes = ["fp16"] + [
                item.strip().lower()
                for item in requested_dtypes.replace(";", ",").split(",")
                if item.strip()
            ]
            native_env["JITTOR_FLASH_ATTN_DTYPES"] = ",".join(
                dict.fromkeys(dtypes))
    packages = repr(OPTIONAL_COMPAT_PACKAGES)
    dependency_probe = (
        "import importlib.util; "
        "required=" + packages + "; "
        "missing=[name for name in required if importlib.util.find_spec(name) is None]; "
        "assert not missing, 'missing optional dependencies: ' + ', '.join(missing); "
        "print('optional compatibility dependencies OK')"
    )
    session.run("nvidia-smi", external=True, env=env)
    session.run(nvcc, "--version", external=True, env=env)
    session.run(python, "-c", dependency_probe, external=True, env=env)
    if session.posargs:
        native_requested = flash_source and any(
            "native_flash_attn" in arg for arg in session.posargs)
        _run_pytest(
            session, (), native_env if native_requested else env, runner=python)
        return
    _run_pytest(session, OPTIONAL_COMPAT_TESTS, env, runner=python)
    if flash_source:
        native_tests = OPTIONAL_NATIVE_FLASH_TESTS
        dtype_spec = native_env["JITTOR_FLASH_ATTN_DTYPES"].lower()
        head_dim_spec = native_env["JITTOR_FLASH_ATTN_HEAD_DIMS"].lower()
        configured_dtypes = {
            item.strip() for item in dtype_spec.replace(";", ",").split(",")
        }
        configured_head_dims = {
            item.strip() for item in head_dim_spec.replace(";", ",").split(",")
        }
        bf16_enabled = bool(
            configured_dtypes & {"bf16", "bfloat16", "all", "full", "*"})
        if bf16_enabled:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_TESTS
        if bf16_enabled and configured_head_dims & {"64", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_HDIM64_TESTS
        if bf16_enabled and configured_head_dims & {"96", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_HDIM96_TESTS
        if bf16_enabled and configured_head_dims & {"128", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_HDIM128_TESTS
        if bf16_enabled and configured_head_dims & {"192", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_HDIM192_TESTS
        if bf16_enabled and configured_head_dims & {"256", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_BF16_HDIM256_TESTS
        if configured_head_dims & {"64", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_HDIM64_TESTS
        if configured_head_dims & {"96", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_HDIM96_TESTS
        if configured_head_dims & {"128", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_HDIM128_TESTS
        if configured_head_dims & {"192", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_HDIM192_TESTS
        if configured_head_dims & {"256", "all", "full", "*"}:
            native_tests += OPTIONAL_NATIVE_FLASH_HDIM256_TESTS
        _run_pytest(session, native_tests, native_env, runner=python)


@nox.session(python=False)
def cuda(session):
    """Run CUDA gates in a pre-provisioned CUDA 12.2 environment."""
    _root, env = _session_env(session, "cuda")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    if not nvcc:
        session.error("CUDA session requires nvcc_path or nvcc on PATH")
    env["nvcc_path"] = nvcc
    env["JITTOR_TEST_DEVICES"] = "cuda"
    env["JITTOR_TEST_REQUIRE_CUDA"] = "1"
    env["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"
    session.run("nvidia-smi", external=True, env=env)
    session.run(nvcc, "--version", external=True, env=env)
    session.run(python, "-m", "pytest", "--version", external=True, env=env)
    probe = (
        "import jittor as jt; "
        "assert jt.compiler.has_cuda; "
        "assert not getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_cuda = 1; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    session.run(python, "-c", probe, env=env, external=True)
    # No sharding here, and 0.16 is the reason rather than an oversight. The
    # device-parity battery was the obvious candidate -- ~227 generated cases,
    # one per operator, independent by construction -- and it was measured
    # before being changed. On the same 26-operator subset, a cold cache each,
    # one GPU and eight cores:
    #
    #   one process, use_parallel_op_compiler=0 (what this used to force) 1444 s
    #   one process, parallel op compiler on                              1420 s
    #   four xdist workers, --dist load                                   1353 s
    #   one process, cache already warm                                   1405 s
    #
    # Sharding buys 6%, and in that run two workers died ("node down: Not
    # properly terminated") -- one operator reported FAILED for a reason that
    # was not its own, the replacement worker died too, and xdist ended the
    # session with an INTERNALERROR: 23 verdicts out of 26. A verifier that
    # sometimes returns no verdict is worse than a slow one.
    #
    # The warm row is what explains the other three: this battery is not
    # compile-bound, so neither the compiler's thread pool nor a shared cache
    # moves it. Every case runs its operator's samples forward and backward on
    # *both* devices, and the CPU half already uses every core through OpenMP --
    # so four processes on one machine divide the cores they were already using.
    # More workers cannot help; fewer comparisons or more machines could.
    _run_pytest(session, CUDA_TESTS, env, runner=python)


@nox.session(python=False)
def npu(session):
    """Run ACL gates in a pre-provisioned Ascend CANN environment."""
    _root, env = _session_env(session, "npu")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    cann_set_env = os.environ.get("CANN_SET_ENV")
    if not cann_set_env or not os.path.isfile(cann_set_env):
        session.error("NPU session requires CANN_SET_ENV pointing to set_env.sh")
    env["CANN_SET_ENV"] = cann_set_env
    env["JITTOR_CI_PYTHON"] = python
    env["JITTOR_TEST_DEVICES"] = "npu"
    session.run("npu-smi", "info", external=True, env=env)
    _run_with_cann(session, python, ("-m", "pytest", "--version"), env)
    probe = (
        "import jittor as jt; "
        "assert getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_acl = 1; "
        "jt.flags.use_cuda = 1; "
        "a = jt.float32([[1, 2], [3, 4]]); "
        "b = jt.float32([[5, 6], [7, 8]]); "
        "x = jt.matmul(a, b); x.sync(); "
        "assert x.numpy().tolist() == [[19.0, 22.0], [43.0, 50.0]]"
    )
    _run_with_cann(session, python, ("-c", probe), env)
    groups = ((tuple(session.posargs),) if session.posargs
              else tuple(group for group in _by_process_mode(NPU_TESTS) if group))
    for group in groups:
        _run_with_cann(
            session,
            python,
            ("-m", "pytest", "-v", "--timeout=600", *group),
            _mode_env(env, group),
        )


@nox.session(python=False)
def rocm(session):
    """Run ROCm gates in a pre-provisioned AMD GPU environment."""
    _root, env = _session_env(session, "rocm")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    env["JITTOR_TEST_DEVICES"] = "rocm"
    session.run("rocminfo", external=True, env=env)
    session.run(python, "-m", "pytest", "--version", external=True, env=env)
    probe = (
        "import jittor as jt; "
        "assert jt.compiler.has_rocm; "
        "jt.flags.use_rocm = 1; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    session.run(python, "-c", probe, env=env, external=True)
    _run_pytest(session, ROCM_TESTS, env, runner=python)


@nox.session(python=False)
def mpi(session):
    """Run MPI gates with a pre-provisioned launcher and Python environment."""
    _root, env = _session_env(session, "mpi")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    # "mpi" is a transport, not a device label: the harness knows cpu/cuda/rocm/npu
    # and used to filter every device away when it saw this, so every
    # device-parametrized test in this session generated zero cases and passed.
    # These tests gate themselves on jt.has_cuda and exercise both, so leave the
    # selection unset and let the build decide.
    env["JITTOR_TEST_DEVICES"] = None
    session.run("mpirun", "--version", external=True, env=env)
    session.run(python, "-m", "pytest", "--version", external=True, env=env)
    _run_pytest(session, MPI_TESTS, env, runner=python)


@nox.session(python=False)
def nccl(session):
    """Run multi-rank NCCL/FSDP2 gates with one isolated cache per rank."""
    root, env = _session_env(session, "nccl")
    python = _hardware_python()
    _set_hardware_python_config(session, python, env)
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    if not nvcc:
        session.error("NCCL session requires nvcc_path or nvcc on PATH")
    env["nvcc_path"] = nvcc
    env["JITTOR_TEST_DEVICES"] = "cuda"
    env["use_cuda"] = "1"
    env["use_nccl"] = "1"
    env["use_mpi"] = "0"
    env["use_mkl"] = "0"
    env["use_cutt"] = "0"
    env["use_cutlass"] = "0"
    env["use_parallel_op_compiler"] = "0"

    raw_devices = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in raw_devices.split(",") if item.strip()]
    raw_world_size = os.environ.get("JITTOR_NCCL_WORLD_SIZE", "2").strip()
    try:
        world_size = int(raw_world_size)
    except ValueError:
        session.error("JITTOR_NCCL_WORLD_SIZE must be an integer")
    if world_size < 2:
        session.error("NCCL session requires JITTOR_NCCL_WORLD_SIZE >= 2")
    if raw_devices and len(devices) < world_size:
        session.error(
            "NCCL session requires at least %d CUDA_VISIBLE_DEVICES" % world_size
        )
    if not devices:
        devices = [str(index) for index in range(world_size)]
    selected_devices = devices[:world_size]

    session.run("nvidia-smi", external=True, env=env)
    session.run(nvcc, "--version", external=True, env=env)
    session.run(python, "-m", "pytest", "--version", external=True, env=env)
    probe = (
        "import jittor as jt; "
        "assert jt.compiler.has_cuda; "
        "assert jt.compile_extern.nccl_ops is not None; "
        "jt.flags.use_cuda = 1; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    for rank, device in enumerate(selected_devices):
        warm_env = env.copy()
        warm_env["CUDA_VISIBLE_DEVICES"] = device
        warm_env["cache_name"] = "nccl%d" % rank
        warm_rootinfo = root / ("warm-rank%d-rootinfo.bin" % rank)
        warm_env["JT_NCCL_WORLD_SIZE"] = "1"
        warm_env["JT_NCCL_RANK"] = "0"
        warm_env["JT_NCCL_LOCAL_RANK"] = "0"
        warm_env["JT_NCCL_ROOTINFO_FILE"] = str(warm_rootinfo)
        session.run(python, "-c", probe, env=warm_env, external=True)
        try:
            warm_rootinfo.unlink()
        except FileNotFoundError:
            pass

    launch_env = env.copy()
    launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
    launch_env["JITTOR_TORCH_SHIM"] = "1"
    targets = tuple(session.posargs) if session.posargs else NCCL_TESTS
    session.run(
        python,
        str(REPO_ROOT / "python" / "jittor" / "distributed" / "launch.py"),
        "-n",
        str(world_size),
        "--backend",
        "nccl",
        "--logdir",
        str(root / "logs"),
        "--",
        python,
        "-m",
        "pytest",
        "-v",
        "--timeout=600",
        *targets,
        env=launch_env,
        external=True,
    )
