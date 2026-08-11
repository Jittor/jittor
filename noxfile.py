"""Canonical local and CI sessions for the staged repository modernization."""

from __future__ import print_function

import os
from pathlib import Path
import shutil
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
SETUPTOOLS = "setuptools==83.0.0"
WHEEL = "wheel==0.45.1"
JUPYTEXT = "jupytext==1.17.3"
NBCLIENT = "nbclient==0.10.2"
NBFORMAT = "nbformat==5.10.4"
IPYKERNEL = "ipykernel==6.29.5"

RATCHET_FILES = (
    "noxfile.py",
    "agent/scripts/check_sdist_contents.py",
    "agent/scripts/check_wheel_contents.py",
    "python/jittor/selftest.py",
    "python/jittor_utils/cuda_wheel.py",
    "python/jittor/torch_shim/deploy.py",
    "tools/release/pack_offline.py",
    "tests/integration/test_notebooks.py",
    "tests/structure/test_cleanup_structure.py",
    "tests/structure/test_pytest_contract.py",
    "tests/structure/test_selftest_structure.py",
)
FORMAT_FILES = (
    "noxfile.py",
    "agent/scripts/check_sdist_contents.py",
    "agent/scripts/test_check_sdist_contents.py",
    "python/jittor/selftest.py",
    "tools/release/pack_offline.py",
    "tests/integration/test_notebooks.py",
    "tests/structure/test_cleanup_structure.py",
    "tests/structure/test_packaging_structure.py",
    "tests/structure/test_pytest_contract.py",
    "tests/structure/test_selftest_structure.py",
)
FILESYSTEM_TESTS = (
    "agent/scripts/test_check_sdist_contents.py",
    "agent/scripts/test_check_wheel_contents.py",
    "tests/structure/test_cleanup_structure.py",
    "tests/structure/test_packaging_structure.py",
    "tests/structure/test_pytest_contract.py",
    "tests/structure/test_selftest_structure.py",
    "tests/structure/test_torch_shim_deploy.py",
    "tests/structure/test_cuda_wheel.py",
)
CPU_TESTS = (
    "tests/compiler/test_custom_op.py",
    "tests/compiler/test_utils.py",
    "tests/core/test_autograd_engine.py",
    "tests/core/test_regression.py",
    "tests/integration/test_notebooks.py",
)
CUDA_TESTS = (
    "tests/backends/cuda/test_cuda.py",
    "tests/ops/test_ops.py",
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
nox.options.sessions = ["lint", "format", "typing", "structure", "py37"]

for name, path in {
    "PIP_CACHE_DIR": NOX_STATE_ROOT / "cache" / "pip",
    "PRE_COMMIT_HOME": NOX_STATE_ROOT / "cache" / "pre-commit",
}.items():
    os.environ.setdefault(name, str(path))
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


def _session_env(session, backend):
    root = Path(session.create_tmp()).resolve()
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
    """Run filesystem tests, then build, audit, and self-test an installed wheel."""
    root, env = _session_env(session, "structure")
    session.install(
        BUILD,
        PYTEST,
        PYTEST_TIMEOUT,
        SETUPTOOLS,
        WHEEL,
        "astunparse==1.6.3",
        JUPYTEXT,
        NBFORMAT,
        "numpy==1.26.4",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    session.run("bash", "agent/scripts/check_repo_layout.sh", external=True, env=env)
    for test_path in FILESYSTEM_TESTS:
        session.run(
            "python",
            "-m",
            "pytest",
            "-v",
            "--timeout=600",
            test_path,
            env=env,
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
    wheel_args = tuple(session.posargs) or (
        "--removal-allowlist",
        "agent/baselines/wheel-removals-stage6.txt",
    )
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


@nox.session(python="3.11")
def benchmark(session):
    """Validate ASV and execute one mandatory Jittor CPU benchmark case."""
    root, env = _session_env(session, "asv-cpu")
    asv_home = root / "jittor-asv-home"
    asv_home.mkdir(parents=True, exist_ok=True)
    env["JITTOR_HOME"] = str(asv_home)
    env["ASV_CONF_DIR"] = str(REPO_ROOT)
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
    with session.chdir(REPO_ROOT):
        session.run("asv", "check", "--python=same", env=env)
    smoke = """
from benchmarks.operators import OperatorBenchmarks

case = OperatorBenchmarks()
case.setup("jittor", "cpu", "gelu")
try:
    case.time_operator("jittor", "cpu", "gelu")
    used = case.track_working_set_bytes("jittor", "cpu", "gelu")
    if not isinstance(used, int) or used <= 0:
        raise RuntimeError("ASV memory benchmark returned %r" % (used,))
finally:
    case.teardown("jittor", "cpu", "gelu")
print("mandatory ASV smoke OK: operators/jittor/cpu/gelu (%d bytes)" % used)
"""
    with session.chdir(REPO_ROOT):
        session.run("python", "-c", smoke, env=env)


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


@nox.session(python="3.11", venv_backend="venv")
def cpu(session):
    """Run the maintained CPU smoke gate on a clean Jittor cache."""
    _root, env = _session_env(session, "cpu")
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
