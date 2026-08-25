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

RUFF = "ruff==0.15.22"
MYPY = "mypy==1.8.0"
ASV = "asv==0.6.6"
BUILD = "build==1.3.0"
PYTEST = "pytest==7.4.4"
PYTEST_TIMEOUT = "pytest-timeout==2.3.1"
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
    "python/jittor/nn/backends/group_norm_cuda.py",
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
    "tests/conftest.py",
    "tools/release/pack_offline.py",
    "tools/docs/check_build.py",
    "tools/docs/check_catalogs.py",
    "tools/docs/check_links.py",
    "tests/integration/test_notebooks.py",
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
CPU_TESTS = (
    "tests/compiler/test_custom_op.py",
    "tests/compiler/test_utils.py",
    "tests/core/test_autograd_engine.py",
    "tests/core/test_misc_shape.py",
    "tests/core/test_regression.py",
    "tests/core/test_rootcause_semantics.py",
    "tests/nn/test_attention.py",
    "tests/nn/test_depthwise_conv.py",
    "tests/nn/test_nn_capabilities.py",
    "tests/ops/test_reduce_op.py",
    "tests/core/test_array.py::TestArray::test_array_dtype",
    "tests/optim/test_opt_state_dict.py",
    "tests/optim/test_optim_core.py",
    "tests/optim/test_optimizer.py",
    "tests/optim/test_optimizer_save_load.py",
    "tests/compat/torch/test_torch_compat_grad_management.py",
    "tests/compat/torch/test_torch_bootstrap.py::TestTorchBootstrap::test_preflight_nvcc_flags_keep_command_separators",
    "tests/compat/torch/test_torch_cpp_extension.py::TestTorchCppExtensionArchFlags::test_reports_the_builder_cxx11_abi",
    "tests/integration/test_notebooks.py",
)
CPU_TORCH_ORACLE_TESTS = (
    "tests/ops/test_cumprod_op.py",
    "tests/optim/test_adamw.py",
    "tests/optim/test_lr_scheduler.py",
    "tests/nn/test_affine_grid.py",
    "tests/nn/test_attention_oracle.py",
    "tests/nn/test_batchnorm.py",
    "tests/nn/test_loss.py",
    "tests/nn/test_relu.py",
)
CUDA_TESTS = (
    "tests/backends/cuda",
    "tests/backends/parity/test_dtype_coverage.py",
    "tests/backends/parity/test_device_parity.py",
    "tests/compat/torch/test_torch_compat_cuda_tf32.py",
    "tests/ops/test_ops.py",
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
    "tests/backends/npu/test_aclop.py",
    "tests/backends/npu/test_acl_indexing.py",
    "tests/ops/test_ops.py",
)
ROCM_TESTS = ("tests/backends/rocm/test_rocm.py",)
MPI_TESTS = (
    "tests/distributed/test_mpi.py",
    "tests/distributed/test_mpi_batchnorm.py",
    "tests/distributed/test_mpi_op.py",
    "tests/distributed/test_single_process_scope.py",
)

NOX_STATE_ROOT.mkdir(parents=True, exist_ok=True)
nox.options.envdir = str(NOX_STATE_ROOT / "envs")
nox.options.error_on_missing_interpreters = True
nox.options.stop_on_first_error = True
nox.options.sessions = [
    "lint",
    "format",
    "typing",
    "structure",
    "packaging",
    "py37",
    "py312",
]

for name, path in {
    "PIP_CACHE_DIR": NOX_STATE_ROOT / "cache" / "pip",
    "PRE_COMMIT_HOME": NOX_STATE_ROOT / "cache" / "pre-commit",
}.items():
    os.environ.setdefault(name, str(path))
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


def _session_env(session, backend):
    root = Path(session.create_tmp()).resolve()
    if root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True)
    paths = {
        "HOME": root / "home",
        "JITTOR_HOME": root / "jittor-home",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "JITTOR_TEST_STATE_ROOT": root / "test-state",
        "TMPDIR": root / "tmp",
        "CUDA_CACHE_PATH": root / "cuda-cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({name: str(path) for name, path in paths.items()})
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf8",
            "PYTHONPATH": str(REPO_ROOT / "python"),
            "cache_name": "nox_%s" % backend,
        }
    )
    python_config = os.environ.get("python_config_path") or shutil.which(
        "python3.%d-config" % sys.version_info[1]
    )
    if python_config:
        env["python_config_path"] = python_config
    return root, env


def _source_copy(destination):
    ignored = shutil.ignore_patterns(
        ".git",
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


def _pytest_invocations(session, defaults):
    if session.posargs:
        return (tuple(session.posargs),)
    return tuple((target,) for target in defaults)


def _run_pytest(session, defaults, env, runner=None):
    python = runner or "python"
    for args in _pytest_invocations(session, defaults):
        session.run(
            python,
            "-m",
            "pytest",
            "-v",
            "--timeout=600",
            *args,
            env=env,
            external=runner is not None,
        )


def _hardware_python():
    return os.environ.get("JITTOR_CI_PYTHON", sys.executable)


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
    docs_env.pop("PYTHONPATH", None)
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
    python_config = shutil.which("python3.%d-config" % sys.version_info[1])
    if python_config:
        docs_env["python_config_path"] = python_config
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
import sys

if sys.version_info[:2] != (3, 7):
    raise SystemExit("py37 requires Python 3.7, found %s" % (sys.version.split()[0],))

root = pathlib.Path(sys.argv[1]).resolve()
excluded = {".git", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", "__pycache__"}
failed = []
checked = 0
for path in sorted(root.rglob("*.py")):
    if any(part in excluded for part in path.parts):
        continue
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


@nox.session(python="3.12", venv_backend="venv")
def py312(session):
    """Build, install, and execute Jittor with a real Python 3.12 interpreter."""
    root, env = _session_env(session, "py312")
    python_config = session.run(
        "python",
        "-c",
        ("import os, sys; print(os.path.join(sys.base_prefix, 'bin', 'python3.12-config'))"),
        silent=True,
    ).strip()
    if os.name != "nt" and not Path(python_config).is_file():
        session.error("Python 3.12 config helper not found: %s" % python_config)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JITTOR_TEST_DEVICES": "cpu",
            "nvcc_path": "",
            "python_config_path": python_config,
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
        "numpy==1.26.4",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    compile_script = r"""
import pathlib
import sys
import warnings

if sys.version_info[:2] != (3, 12):
    raise SystemExit("py312 requires Python 3.12, found %s" % (sys.version.split()[0],))

warnings.simplefilter("error", SyntaxWarning)
root = pathlib.Path(sys.argv[1]).resolve()
excluded = {".git", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", "__pycache__"}
failed = []
checked = 0
for path in sorted(root.rglob("*.py")):
    if any(part in excluded for part in path.parts):
        continue
    checked += 1
    try:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    except (SyntaxError, UnicodeError) as error:
        failed.append("%s: %s" % (path.relative_to(root), error))
if failed:
    print("Python 3.12 compile failures:")
    print("\n".join(failed))
    raise SystemExit(1)
print("Python 3.12 compile OK without SyntaxWarning: %d files" % checked)
"""
    session.run(
        "python",
        "-W",
        "error::SyntaxWarning",
        "-c",
        compile_script,
        str(REPO_ROOT),
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
        session.error("expected exactly one Python 3.12 wheel, found %d" % len(wheels))

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
    selftest_env["cache_name"] = "nox_py312_wheel_selftest"
    with session.chdir(root):
        session.run(
            "python",
            "-W",
            "error::SyntaxWarning",
            "-m",
            "jittor.selftest",
            env=selftest_env,
        )


@nox.session(python="3.11", venv_backend="venv")
def cpu(session):
    """Run the maintained CPU smoke gate on a clean Jittor cache."""
    _root, env = _session_env(session, "cpu")
    real_torch_site = os.environ.get("REAL_TORCH_SITE", "").strip()
    require_real_torch = os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").strip() == "1"
    if require_real_torch and not real_torch_site:
        session.error("JITTOR_REQUIRE_REAL_TORCH=1 requires REAL_TORCH_SITE")
    # Nox overlays this mapping on the parent environment, so removing the key
    # would still leak a caller-provided real Torch into ordinary Jittor tests.
    env["REAL_TORCH_SITE"] = ""
    env["nvcc_path"] = ""
    env["JITTOR_TEST_DEVICES"] = "cpu"
    session.install(
        "astunparse==1.6.3",
        IPYKERNEL,
        JUPYTEXT,
        NBCLIENT,
        NBFORMAT,
        "numpy==1.26.4",
        "pillow==11.0.0",
        PYTEST,
        PYTEST_TIMEOUT,
        SCIPY,
        SETUPTOOLS,
        "tqdm==4.67.1",
    )
    probe = (
        "import jittor as jt; "
        "assert not jt.compiler.has_cuda; "
        "assert not getattr(jt.compiler, 'has_acl', 0); "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    session.run("python", "-c", probe, env=env)
    session.run(
        "python",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "tests",
        env=env,
    )
    _run_pytest(session, CPU_TESTS, env)
    oracle_env = env.copy()
    if real_torch_site:
        oracle_env["REAL_TORCH_SITE"] = real_torch_site
    elif require_real_torch:
        session.error("independent PyTorch oracle is required but unavailable")
    _run_pytest(session, CPU_TORCH_ORACLE_TESTS, oracle_env)


@nox.session(python=False)
def optional(session):
    """Run fail-closed optional compatibility gates on real CUDA."""
    _root, env = _session_env(session, "optional")
    python = _hardware_python()
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
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    if not nvcc:
        session.error("CUDA session requires nvcc_path or nvcc on PATH")
    env["nvcc_path"] = nvcc
    env["JITTOR_TEST_DEVICES"] = "cuda"
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
    _run_pytest(session, CUDA_TESTS, env, runner=python)


@nox.session(python=False)
def npu(session):
    """Run ACL gates in a pre-provisioned Ascend CANN environment."""
    _root, env = _session_env(session, "npu")
    python = _hardware_python()
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
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    _run_with_cann(session, python, ("-c", probe), env)
    for args in _pytest_invocations(session, NPU_TESTS):
        _run_with_cann(
            session,
            python,
            ("-m", "pytest", "-v", "--timeout=600", *args),
            env,
        )


@nox.session(python=False)
def rocm(session):
    """Run ROCm gates in a pre-provisioned AMD GPU environment."""
    _root, env = _session_env(session, "rocm")
    python = _hardware_python()
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
    env["JITTOR_TEST_DEVICES"] = "mpi"
    session.run("mpirun", "--version", external=True, env=env)
    session.run(python, "-m", "pytest", "--version", external=True, env=env)
    _run_pytest(session, MPI_TESTS, env, runner=python)
