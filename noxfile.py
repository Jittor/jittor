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
SETUPTOOLS = "setuptools==83.0.0"
WHEEL = "wheel==0.45.1"

RATCHET_FILES = (
    "noxfile.py",
    "agent/scripts/check_wheel_contents.py",
    "python/jittor_utils/cuda_wheel.py",
    "python/jittor/torch_shim/deploy.py",
    "python/jittor/test/_runner.py",
)
FORMAT_FILES = ("noxfile.py",)
FILESYSTEM_TESTS = (
    "agent/scripts/test_check_wheel_contents.py",
    "python/jittor/test/test_packaging_structure.py",
    "python/jittor/test/test_torch_shim_deploy.py",
    "python/jittor/test/test_cuda_wheel.py",
    "python/jittor/test/test_test_runner.py",
)
CPU_TESTS = (
    "jittor.test.test_autograd_engine",
    "jittor.test.test_regression",
)
CUDA_TESTS = (
    "jittor.test.test_cuda",
    "jittor.test.test_ops",
)
NPU_TESTS = (
    "jittor.test.test_acl",
    "jittor.test.test_aclop",
    "jittor.test.test_acl_indexing",
    "jittor.test.test_ops",
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
        "build",
        "dist",
    )
    shutil.copytree(str(REPO_ROOT), str(destination), symlinks=True, ignore=ignored)


def _test_modules(session, defaults, env, runner=None):
    modules = tuple(session.posargs) or defaults
    python = runner or "python"
    for module in modules:
        if "." not in module:
            module = "jittor.test." + module
        session.run(python, "-m", module, "-v", env=env, external=runner is not None)


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
    """Run pure filesystem tests, then build and audit a wheel outside the tree."""
    root, env = _session_env(session, "structure")
    session.install(BUILD, SETUPTOOLS, WHEEL)
    session.run("bash", "agent/scripts/check_repo_layout.sh", external=True, env=env)
    for test_path in FILESYSTEM_TESTS:
        session.run("python", test_path, "-v", env=env)

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
            "--outdir",
            str(dist),
            env=env,
        )
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        session.error("expected exactly one wheel, found %d" % len(wheels))
    wheel_args = tuple(session.posargs) or (
        "--removal-allowlist",
        "agent/baselines/wheel-removals-stage3.txt",
    )
    session.run(
        "python",
        "agent/scripts/check_wheel_contents.py",
        "compare",
        str(wheels[0]),
        *wheel_args,
        env=env,
    )


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
        "numpy==1.26.4",
        "pillow==11.0.0",
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
    _test_modules(session, CPU_TESTS, env)


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
    probe = (
        "import jittor as jt; "
        "assert jt.compiler.has_cuda; "
        "assert not getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_cuda = 1; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    session.run(python, "-c", probe, env=env, external=True)
    _test_modules(session, CUDA_TESTS, env, runner=python)


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
    probe = (
        "import jittor as jt; "
        "assert getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_acl = 1; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0"
    )
    _run_with_cann(session, python, ("-c", probe), env)
    modules = tuple(session.posargs) or NPU_TESTS
    for module in modules:
        if "." not in module:
            module = "jittor.test." + module
        _run_with_cann(session, python, ("-m", module, "-v"), env)
