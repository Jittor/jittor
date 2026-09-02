"""Import-time environment preparation for the Torch compatibility runtime.

This module is deliberately stdlib-only: :mod:`jittor.__init__` imports it
before the compiler and native core exist.
"""

from __future__ import absolute_import

import glob
import hashlib
import os
import pathlib
import sys
import importlib.util
from ..diagnostics import EXPECTED, swallowed


_TRUTHY = frozenset(("1", "true", "yes", "on"))
_ENTRY_MARKERS = (
    "jittor.compat.shim",
    "jittor.torch_shim",
    "torch_shim",
    "import jittor as torch",
)
_DRIVER_LIBRARY_CANDIDATES = (
    "/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
)


class PreflightResult:
    __slots__ = (
        "active", "project_root", "runtime_root", "real_home", "jtcuda", "trigger"
    )

    def __init__(
        self,
        active,
        project_root="",
        runtime_root="",
        real_home="",
        jtcuda="",
        trigger="",
    ):
        self.active = bool(active)
        self.project_root = project_root
        self.runtime_root = runtime_root
        self.real_home = real_home
        self.jtcuda = jtcuda
        self.trigger = trigger


def is_truthy(value):
    return str(value or "").strip().lower() in _TRUTHY


def package_root():
    return pathlib.Path(__file__).resolve().parent


def resources_root():
    return package_root() / "resources"


def jittor_python_root():
    spec = importlib.util.find_spec("jittor")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if not locations:
        raise RuntimeError("cannot locate the installed jittor package")
    return pathlib.Path(locations[0]).resolve().parent


def project_dir(path=None):
    value = pathlib.Path.cwd() if path is None else pathlib.Path(os.fspath(path)).expanduser()
    if value.exists() and value.is_file():
        return value.parent.resolve()
    if str(value).endswith(".py"):
        return value.parent.resolve()
    return value.resolve()


def project_runtime_root(project_root, environ=None):
    env = os.environ if environ is None else environ
    root = project_dir(project_root)
    cache_root = env.get("JITTOR_TORCH_CACHE_ROOT")
    if cache_root:
        base = pathlib.Path(cache_root).expanduser()
    else:
        xdg_cache = env.get("XDG_CACHE_HOME")
        if not xdg_cache:
            home = env.get("REAL_HOME") or env.get("HOME") or os.path.expanduser("~")
            xdg_cache = os.path.join(home, ".cache")
        base = pathlib.Path(xdg_cache).expanduser() / "jittor" / "torch-shim"
    safe_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in (root.name or "project")
    )[:64] or "project"
    digest = hashlib.sha256(os.fsencode(os.fspath(root))).hexdigest()[:16]
    return (base / (safe_name + "-" + digest)).resolve()


def _entry_project_root(argv):
    entry = argv[0] if argv else ""
    if not entry or entry in ("-c", "-m"):
        return None
    entry_path = pathlib.Path(entry).expanduser().resolve()
    if not entry_path.is_file():
        return None
    try:
        with open(
            os.fspath(entry_path), "r", encoding="utf-8", errors="ignore"
        ) as entry_file:
            head = entry_file.read(65536)
    except OSError:
        return None
    if any(marker in head for marker in _ENTRY_MARKERS):
        return entry_path.parent
    return None


def _prepend_env_path(environ, name, path):
    value = os.fspath(path)
    current = [item for item in environ.get(name, "").split(os.pathsep) if item]
    environ[name] = os.pathsep.join([value] + [item for item in current if item != value])


def prepend_sys_path(path, after=None):
    """Put ``path`` ahead of everything already on ``sys.path``.

    Reserved for directories this layer owns: anything placed here precedes the
    standard library, so a stray ``types.py`` or ``copy.py`` inside it breaks
    the interpreter. Use :func:`append_sys_path` for directories that belong to
    the user's project.
    """
    value = os.fspath(path)
    if not value:
        return
    if value in sys.path:
        sys.path.remove(value)
    if after is not None:
        marker = os.fspath(after)
        if marker in sys.path:
            sys.path.insert(sys.path.index(marker) + 1, value)
            return
    sys.path.insert(0, value)


def append_sys_path(path):
    """Make ``path`` importable without letting it shadow anything.

    The shim used to insert the *project* directory at ``sys.path[0]``, ahead of
    the standard library and of Jittor's own package root. A project holding a
    file named after a stdlib module -- ``types.py`` and ``copy.py`` are the
    ones seen in practice, and both are imported by Jittor itself -- then broke
    the interpreter from the moment ``enable()`` ran, with a traceback pointing
    anywhere but at the shim. A project directory only has to be *reachable*;
    it has no claim to precedence, so it goes at the end.
    """
    value = os.fspath(path)
    if not value or value in sys.path:
        return
    sys.path.append(value)


_UNWRITABLE_HINT = (
    "This path is derived from HOME/XDG_CACHE_HOME. On a read-only or "
    "unwritable HOME, point the shim somewhere writable with "
    "JITTOR_TORCH_CACHE_ROOT=<dir> (moves every project's runtime) or "
    "JITTOR_TORCH_RUNTIME_ROOT=<dir> (this project only). Set "
    "JITTOR_TORCH_SHIM=0 to start Jittor without the torch runtime at all."
)


def _ensure_dir(path, purpose=None):
    """Create a directory the shim needs, or say what it was for and how to move it.

    This runs during ``import jittor``, before the compiler and the native core
    exist. A bare ``PermissionError: '/home/x/.cache/jittor'`` from here reads
    as a Jittor installation problem, names no cause and offers no fix -- while
    the actual cause is that the torch shim decided to put its runtime under an
    unwritable HOME.
    """
    value = pathlib.Path(path)
    what = (" (%s)" % purpose) if purpose else ""
    try:
        value.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise type(exc)(
            "Jittor's torch compatibility layer could not create its runtime "
            "directory %s%s: %s. %s" % (value, what, exc, _UNWRITABLE_HINT)
        ) from exc
    if not os.access(os.fspath(value), os.W_OK):
        raise PermissionError(
            "Jittor's torch compatibility layer needs to write to %s%s, but "
            "that directory is not writable. %s" % (value, what, _UNWRITABLE_HINT)
        )
    return value


def _set_env_dir(environ, name, path, override=False):
    if override or not environ.get(name):
        environ[name] = os.fspath(path)
    _ensure_dir(environ[name], "the directory $%s points at" % name)


def _strict_math_nvcc_flags(value, remove_fast_math=False):
    tokens = str(value or "").split()
    if remove_fast_math:
        tokens = [token for token in tokens if token != "--use_fast_math"]
    for token in ("--fmad=false", "--prec-div=true", "--prec-sqrt=true"):
        if token not in tokens:
            tokens.append(token)
    # jit_compiler.cc concatenates nvcc_flags directly after quoted source
    # paths. Keep explicit separators instead of relying on every caller.
    return " " + " ".join(tokens) + " "


def _remove_strict_math_nvcc_flags(value):
    strict_tokens = {"--fmad=false", "--prec-div=true", "--prec-sqrt=true"}
    tokens = [token for token in str(value or "").split()
              if token not in strict_tokens]
    return " " + " ".join(tokens) + " " if tokens else ""


def _acl_environment(environ):
    return bool(
        environ.get("ASCEND_TOOLKIT_HOME")
        or environ.get("ASCEND_HOME_PATH")
        or environ.get("tikcc_path")
    )


def _add_nvcc_flags(environ):
    if _acl_environment(environ):
        environ["nvcc_flags"] = _remove_strict_math_nvcc_flags(
            environ.get("nvcc_flags", "")
        )
        return
    if is_truthy(environ.get("JITTOR_TORCH_KEEP_FAST_MATH")):
        return
    environ["nvcc_flags"] = _strict_math_nvcc_flags(
        environ.get("nvcc_flags", "")
    )


def configure_torch_math_flags(jittor_module):
    compiler = getattr(jittor_module, "compiler", None)
    flags = getattr(compiler, "flags", None)
    if getattr(compiler, "has_acl", False):
        os.environ["nvcc_flags"] = _remove_strict_math_nvcc_flags(
            os.environ.get("nvcc_flags", "")
        )
        current = getattr(flags, "nvcc_flags", None)
        if isinstance(current, str):
            flags.nvcc_flags = _remove_strict_math_nvcc_flags(current)
        return
    _add_nvcc_flags(os.environ)
    if is_truthy(os.environ.get("JITTOR_TORCH_KEEP_FAST_MATH")):
        return
    try:
        current = getattr(flags, "nvcc_flags", None)
        if isinstance(current, str):
            flags.nvcc_flags = _strict_math_nvcc_flags(
                current, remove_fast_math=True
            )
    except EXPECTED as exc:
        swallowed("shim/preflight.py configure_torch_math_flags: current = getattr(flags, 'nvcc_flags', None)", exc)


def _best_jtcuda(environ, real_home):
    candidates = []
    if environ.get("JTCUDA"):
        candidates.append(environ["JTCUDA"])
    for home in (real_home, environ.get("HOME")):
        if home:
            candidates.extend(
                glob.glob(os.path.join(home, ".cache", "jittor", "jtcuda", "cuda*_linux"))
            )
    valid = []
    for candidate in dict.fromkeys(candidates):
        path = pathlib.Path(candidate)
        if (path / "bin" / "nvcc").is_file():
            valid.append(path)
    if not valid:
        return None
    valid.sort(
        key=lambda path: (
            not (path / "include" / "cudnn.h").is_file(),
            "cuda12.2" not in path.name,
            os.fspath(path),
        )
    )
    return valid[0]


def _configure_cuda(environ, real_home):
    if is_truthy(environ.get("JITTOR_TORCH_KEEP_CUDA")):
        return None
    jtcuda = _best_jtcuda(environ, real_home)
    if jtcuda is None:
        return None
    environ.setdefault("JTCUDA", os.fspath(jtcuda))
    environ.setdefault("nvcc_path", os.fspath(jtcuda / "bin" / "nvcc"))
    environ.setdefault("CUDA_HOME", os.fspath(jtcuda))
    _prepend_env_path(environ, "PATH", jtcuda / "bin")
    _prepend_env_path(environ, "LD_LIBRARY_PATH", jtcuda / "lib64")
    _prepend_env_path(environ, "LIBRARY_PATH", jtcuda / "lib64")
    return jtcuda


def configure_runtime_driver_lib(runtime, environ=None):
    env = os.environ if environ is None else environ
    candidates = tuple(pathlib.Path(path) for path in _DRIVER_LIBRARY_CANDIDATES)
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return
    lib_dir = _ensure_dir(pathlib.Path(runtime) / "lib")
    target = lib_dir / "libcuda.so"
    try:
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                target.unlink()
        if not target.exists():
            target.symlink_to(source)
    except OSError:
        return
    _prepend_env_path(env, "LD_LIBRARY_PATH", lib_dir)
    _prepend_env_path(env, "LIBRARY_PATH", lib_dir)
    env.setdefault("TRITON_LIBCUDA_PATH", os.fspath(lib_dir))


def prepare_import_environment(
    argv=None,
    environ=None,
    project_root=None,
    runtime_root=None,
    force=False,
    local_home=True,
    configure_cuda=True,
):
    """Prepare the shim environment before native compiler/core import."""

    env = os.environ if environ is None else environ
    args = sys.argv if argv is None else argv
    entry_root = _entry_project_root(args)
    explicit_project = (
        project_root
        if project_root is not None
        else env.get("JITTOR_TORCH_PROJECT_ROOT")
    )
    explicit_runtime = (
        runtime_root
        if runtime_root is not None
        else env.get("JITTOR_TORCH_RUNTIME_ROOT")
    )
    active = bool(
        force
        or explicit_project
        or explicit_runtime
        or is_truthy(env.get("JITTOR_TORCH_SHIM"))
        or entry_root
    )
    if not active:
        return PreflightResult(False)

    trigger = (
        "forced"
        if force
        else "environment"
        if (
            explicit_project
            or explicit_runtime
            or is_truthy(env.get("JITTOR_TORCH_SHIM"))
        )
        else "entry"
    )
    project = project_dir(explicit_project or entry_root or pathlib.Path.cwd())
    runtime = pathlib.Path(
        os.fspath(explicit_runtime or project_runtime_root(project, env))
    ).expanduser().resolve()
    _ensure_dir(runtime, "the torch shim's per-project runtime root")
    real_home = env.get("REAL_HOME") or env.get("HOME") or os.path.expanduser("~")
    env.setdefault("REAL_HOME", real_home)
    env["JITTOR_TORCH_PROJECT_ROOT"] = os.fspath(project)
    env["JITTOR_TORCH_RUNTIME_ROOT"] = os.fspath(runtime)
    # The optional accelerator externs below stay off: measurements on cuDNN and
    # cuBLAS show no benefit from cuTT or CUTLASS for the shim's workloads, and
    # NCCL belongs to an explicitly configured distributed run.
    #
    # oneDNN is deliberately NOT in that list. Turning it off removes the
    # `mkl_conv` and `mkl_matmul` relays, so every CPU convolution falls back to
    # the generic reindex kernel: a 4x64x32x32 conv went from 1.4ms to 156ms and
    # a 512x512 matmul from 0.6ms to 11ms. That made ordinary CPU inference under
    # `import torch` unusable, which is the opposite of what the shim is for.
    for name, value in (
        ("JITTOR_TORCH_SHIM", "1"),
        ("FIX_TORCH_ERROR", "0"),
        ("DISABLE_MULTIPROCESSING", "1"),
        ("use_cutt", "0"),
        ("use_cutlass", "0"),
        ("use_nccl", "0"),
    ):
        env.setdefault(name, value)
    for name, subdir in (
        ("JITTOR_HOME", "jittor_cache"),
        ("TORCH_HOME", "torch_home"),
        ("JITTOR_TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("XDG_CACHE_HOME", "xdg_cache"),
        ("CUDA_CACHE_PATH", "cuda_cache"),
        ("TRITON_HOME", "triton_home"),
        ("TRITON_CACHE_DIR", "triton_home/cache"),
        ("TRITON_OVERRIDE_DIR", "triton_home/override"),
        ("TRITON_DUMP_DIR", "triton_home/dump"),
        ("PIP_CACHE_DIR", "pip_cache"),
    ):
        _set_env_dir(env, name, runtime / subdir)
    if local_home and not is_truthy(env.get("JITTOR_TORCH_KEEP_HOME")):
        env["HOME"] = env.get("JITTOR_TORCH_HOME", os.fspath(runtime / "home"))
        _ensure_dir(env["HOME"], "HOME for this process; "
                                "JITTOR_TORCH_KEEP_HOME=1 keeps the real one")
    _set_env_dir(
        env,
        "TMPDIR",
        runtime / "tmp",
        override=not is_truthy(env.get("JITTOR_TORCH_KEEP_TMPDIR")),
    )
    _add_nvcc_flags(env)
    jtcuda = _configure_cuda(env, real_home) if configure_cuda else None
    configure_runtime_driver_lib(runtime, env)
    return PreflightResult(
        True,
        os.fspath(project),
        os.fspath(runtime),
        real_home,
        os.fspath(jtcuda) if jtcuda else "",
        trigger,
    )
