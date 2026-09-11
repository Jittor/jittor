# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import subprocess as sp
import json
import os
import re
import sys
import glob
import inspect
import datetime
import threading
import platform
import ctypes
from ctypes import cdll
from ctypes.util import find_library

import jittor_utils as jit_utils
from jittor_utils import (
    LOG,
    run_cmd,
    find_exe,
    try_find_exe,
    # There used to be a byte-identical copy of this in this module, shadowing
    # the one here; fixing one meant missing the other.
    env_or_try_find,
    cc_path,
    cc_type,
    cache_path,
)
from jittor_utils import env_config
from jittor_utils.env_config import build_env, build_flag
from jittor_utils.compiler_flags import remove_flags, shsplit
from . import pyjt_compiler
from .._runtime.flag_policy import flag_category
from jittor_utils import lock
from jittor_utils import install_cuda
__version__ = jit_utils.get_jittor_version()
import hashlib
from functools import partial
from jittor_utils import backend_discovery as _backend_discovery
from jittor_utils import build_config as _build_config_api
from jittor_utils import backend_resources as _backend_resources_api
from jittor_utils.backend_resources import backend_root, core_root
from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource, ModuleBuildServices


def _module_build_services(config):
    return ModuleBuildServices(
        pyjt_compiler.compile_single, fix_cl_flags, config.cc_path,
        jit_utils.cache_path, config.jittor_path,
    )


def make_backend_context(config=None, *, publish_library=None, mpi_compile_flags=""):
    config = build_config if config is None else config
    return BuildContext(
        config=config,
        compile_module=partial(jit_utils.compile_module, services=_module_build_services(config)),
        compile=compile, compile_custom_ops=compile_custom_ops,
        publish_library=publish_library, make_cache_dir=make_cache_dir,
        load_library=ctypes.CDLL, mpi_compile_flags=mpi_compile_flags, so=so,
        native_core=globals().get("core"),
    )


def compile_backend_sources(config, common_flags):
    objects, commands = [], []
    directory = os.path.join(config.cache_path, "obj_files", "backends")
    os.makedirs(directory, exist_ok=True)
    for source in config.backend_sources:
        path = os.path.abspath(source.path)
        identity = json.dumps((path, source.language, source.flags, source.compiler),
                              separators=(",", ":"))
        tag = hashlib.sha256(identity.encode("utf8")).hexdigest()[:12]
        output = os.path.join(directory, os.path.basename(path) + "." + tag + ".o")
        flags = remove_flags(common_flags + " " + source.flags,
                             ['-l', '-L', '-Wl,', '.lib', '-shared'])
        driver = source.compiler or config.cc_path
        if source.language in ("cuda", "hip"):
            driver = source.compiler or config.kernel_compiler or config.nvcc_path
            if not driver:
                raise RuntimeError("backend source requires a compiler: " + path)
            if config.convert_nvcc_flags is not None:
                flags = config.convert_nvcc_flags(flags)
        if "nan_checker" in path:
            # A NaN check compiled under a promise that NaN does not occur
            # cannot work. `--use_fast_math` is still on every CUDA compile
            # line by default (see `nvcc_flags` below), so this exemption is
            # live for `nan_checker.cu`. `-Ofast` is no longer added to kernel
            # flags -- KI-BACKEND-005 replaced it with `-O3` -- and is stripped
            # here only because a user-supplied `cc_flags`/`nvcc_flags` can
            # still carry it in.
            flags = remove_flags(flags, ["--use_fast_math", "-Ofast"]) + " -O2 "
        command = '"%s" "%s" %s -c -o "%s"' % (driver, path, flags, output)
        commands.append(fix_cl_flags(command))
        objects.append(output)
    if commands:
        jit_utils.run_cmds(commands, config.cache_path, config.jittor_path,
                           "Compiling " + config.backend + " backend")
    return objects

def find_jittor_path():
    return os.path.dirname(os.path.dirname(__file__))

def make_cache_dir(cache_path):
    if not os.path.isdir(cache_path):
        LOG.i(f"Create cache dir: {cache_path}")
        os.mkdir(cache_path)

def moveback_flags(flags, rm_flags):
    flags = shsplit(flags)
    output = []
    output2 = []
    for s in flags:
        ss = s.replace("\"", "")
        for rm in rm_flags:
            if ss.startswith(rm) or ss.endswith(rm):
                output2.append(s)
                break
        else:
            output.append(s)
    return " ".join(output+output2)

def map_flags(flags, func):
    flags = shsplit(flags)
    output = []
    for s in flags:
        output.append(func(s))
    return " ".join(output)

from .codegen import (
    gen_jit_tests,
    strip_cxx_comments,
    gen_jit_flags,
    parse_var_members,
    gen_jit_op_maker,
    _VAR_MEMBER_DECL,
    _VAR_MEMBER_LOOSE,
)

from .compilation import (
    _source_path,
    compile,
    compile_custom_op,
    compile_custom_ops,
    _import_custom_op_lib,
    product_build_stamp_path,
    _stat_signature,
    _is_within,
    _include_tree_signature,
    _custom_op_include_dirs,
    custom_op_build_ingredients,
    product_build_is_current,
    _write_product_build_stamp,
    compile_if_stale,
)


def get_full_path_of_executable(name):
    full_path = os.path.abspath(name)
    while os.path.islink(full_path):
        full_path = os.path.realpath(full_path)
    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
        return full_path
    return get_full_path_of_executable(find_exe(name))

cuda_wheel_stack = None
cuda_include_dirs = []
cuda_lib_dirs = []
cuda_runtime_lib = ""
_loaded_cuda_libraries = {}


def _cuda_library_sort_key(path):
    name = os.path.basename(path)
    return (name.count("."), len(name), name)


def find_cuda_library(name, component=None):
    """Find a CUDA library, including wheel-only versioned SONAME files."""

    if cuda_wheel_stack:
        path = cuda_wheel_stack.find_library(name, component)
        if path:
            return path
    dirs = cuda_lib_dirs or [globals().get("cuda_lib", ""), globals().get("cuda_bin", "")]
    if os.name == "nt":
        patterns = (name + "64*.dll", name + "*.dll")
    elif platform.system() == "Darwin":
        patterns = ("lib" + name + ".dylib", "lib" + name + ".*.dylib")
    else:
        patterns = ("lib" + name + ".so", "lib" + name + ".so.*")
    matches = []
    for directory in dirs:
        if not directory:
            continue
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(directory, pattern)))
    matches = [os.path.abspath(path) for path in matches if os.path.isfile(path)]
    if not matches:
        return None
    matches.sort(key=_cuda_library_sort_key)
    return matches[0]


def cuda_library_link_flags(name, path=None):
    path = path or find_cuda_library(name)
    if not path:
        raise RuntimeError("CUDA library lib%s was not found" % name)
    directory = os.path.dirname(path)
    if os.name == "nt":
        return '-L"%s" -l%s' % (directory, name)
    return '-L"%s" -l:%s' % (directory, os.path.basename(path))


def preload_cuda_library(name, required=False):
    """Load a CUDA library and its wheel dependencies into the global scope."""

    if cuda_wheel_stack:
        paths = cuda_wheel_stack.preload_paths(name)
    else:
        path = find_cuda_library(name)
        paths = [path] if path else []
    if required and not paths:
        raise RuntimeError("CUDA library lib%s was not found" % name)
    for path in paths:
        if path not in _loaded_cuda_libraries:
            _loaded_cuda_libraries[path] = ctypes.CDLL(path, dlopen_flags)
    return _loaded_cuda_libraries.get(paths[-1]) if paths else None

def check_cuda():
    if not nvcc_path:
        return
    global cc_flags, has_cuda, is_cuda, core_link_flags, cuda_dir, cuda_lib, cuda_include, cuda_home, cuda_bin
    global cuda_include_dirs, cuda_lib_dirs, cuda_runtime_lib
    global cuda_sdk_flags, cuda_link_flags
    cuda_dir = os.path.dirname(get_full_path_of_executable(nvcc_path))
    cuda_bin = cuda_dir
    cuda_home = os.path.abspath(os.path.join(cuda_dir, ".."))
    # try default nvidia-cuda-toolkit in Ubuntu 20.04
    # assert cuda_dir.endswith("bin") and "cuda" in cuda_dir.lower(), f"Wrong cuda_dir: {cuda_dir}"
    cuda_include = os.path.abspath(os.path.join(cuda_dir, "..", "include"))
    cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib64"))
    if nvcc_path == "/usr/bin/nvcc":
        # this nvcc is install by package manager
        cuda_lib = "/usr/lib/x86_64-linux-gnu"
    cuda_include_dirs = [cuda_include]
    cuda_lib_dirs = [cuda_lib, cuda_bin]
    if cuda_wheel_stack:
        cuda_include_dirs = cuda_wheel_stack.include_dirs() + cuda_include_dirs
        cuda_lib_dirs = cuda_wheel_stack.lib_dirs() + cuda_lib_dirs
    cuda_include_dirs = list(dict.fromkeys(
        path for path in cuda_include_dirs if os.path.isdir(path)
    ))
    cuda_lib_dirs = list(dict.fromkeys(
        path for path in cuda_lib_dirs if os.path.isdir(path)
    ))
    cuda_include2 = os.path.join(backend_root(jittor_path, "cuda"), "include")
    cc_flags += " -DHAS_ACCELERATOR -DHAS_CUDA -DIS_CUDA "
    cuda_sdk_flags = "".join(f' -I"{path}"' for path in cuda_include_dirs)
    cuda_sdk_flags += f" -I\"{cuda_include2}\" "
    if os.name == 'nt':
        cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib", "x64"))
        # cc_flags += f" \"{cuda_lib}\\cudart.lib\" "
        cuda_link_flags = f" -lcudart -L\"{cuda_lib}\" -L\"{cuda_bin}\" "
    else:
        cuda_runtime_lib = find_cuda_library("cudart")
        if not cuda_runtime_lib:
            raise RuntimeError(
                "CUDA compiler was found, but libcudart was not found in %s"
                % cuda_lib_dirs
            )
        cuda_link_flags = " " + cuda_library_link_flags("cudart", cuda_runtime_lib) + " "
    is_cuda = has_cuda = 1

def _write_jit_utils_cache_key(files, output):
    """Write ``<output>.key`` from a child process.

    ``cache_compile`` records the key as a side effect of building, so the only
    way to get one is to run a build. Running it here would replace a library
    this process already has mapped; running it in a child leaves this process
    with a single, settled file to import.
    """
    cmd = compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True, return_cmd=True)
    # Import from the directory that actually holds the freshly built library,
    # which is not necessarily this module's ``cache_path`` once a CUDA suffix
    # has been appended to it.
    script = (
        "import sys\n"
        "sys.path.insert(0, {module_dir!r})\n"
        "import jit_utils_core\n"
        "jit_utils_core.cache_compile({cmd!r}, {cache!r}, {jittor!r})\n"
    ).format(module_dir=os.path.dirname(os.path.abspath(output)),
             cache=cache_path, cmd=cmd, jittor=jittor_path)
    try:
        result = sp.run([sys.executable, "-c", script],
                        stdout=sp.PIPE, stderr=sp.STDOUT)
    except OSError as error:
        LOG.v("cache key child could not start: %s" % error)
        return
    if result.returncode != 0:
        LOG.v("cache key child failed: " + result.stdout.decode("utf8", "replace"))


#: Exit code used when jit_utils was rebuilt and the process must be restarted.
#: Distinct from 1 so a caller can tell "rerun me" apart from a real failure.
JIT_UTILS_UPDATED_EXIT_CODE = 3


def check_cache_compile():
    files = [
        "src/utils/cache_compile.cc",
        "src/utils/log.cc",
        "src/utils/tracer.cc",
        "src/utils/jit_utils.cc",
        "src/utils/str_utils.cc",
    ]
    if os.name == 'nt':
        files = [ x.replace('/', '\\') for x in files ]
    global jit_utils_core_files
    jit_utils_core_files = files
    output = jit_utils.cache_path+'/jit_utils_core'+extension_suffix
    recompile = compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True)
    if recompile and jit_utils.cc:
        # A non-zero code, because nothing the caller asked for was done. With
        # exit 0 a CI job or a training script that touched src/utils/*.cc
        # "succeeded" without running a single line of the user's program, and
        # the only trace was one line of log among hundreds.
        LOG.e("jit_utils was rebuilt and cannot be reloaded in this process. "
              "Nothing else has run: rerun the same command.")
        sys.exit(JIT_UTILS_UPDATED_EXIT_CODE)
    if not jit_utils.cc:
        # The cache key is written by cache_compile, which needs this very
        # library to be importable first. Doing that after importing it would
        # relink the file while this process has it mapped: the linker unlinks
        # the output before writing, so the inode this process holds no longer
        # matches the one on disk, and the loader later resolves jittor_core's
        # dependency to a *second* copy of the same library. Two copies mean two
        # sets of the flags and log-capture state defined in these files, which
        # is why log_capture_scope silently returned nothing after a cold build.
        # Generate the key in a short-lived child, then import the settled file.
        _write_jit_utils_cache_key(files, output)
        with jit_utils.import_scope(import_flags):
            jit_utils.try_import_jit_utils_core()
        if not jit_utils.cc:
            raise RuntimeError(
                f"jit_utils_core was built at {output} but could not be "
                f"imported. The build directory is on sys.path; a stale or "
                f"partially written file there is the usual cause -- remove "
                f"it and run again.")
        if not os.path.isfile(output + ".key"):
            # The child could not run; fall back to the original behaviour so a
            # missing key never blocks startup. The duplicate mapping above is
            # the cost, and the next run starts from a cached key.
            compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True)

def query_cuda_archs():
    """Compute capabilities of the GPUs on this machine, or None if unknown.

    Runs a short child because the CUDA driver cannot be loaded and unloaded
    inside this process. Only bare integers are accepted from its output: the
    child inherits stdout and stderr from a full Jittor import, and *any* line
    it happens to log used to be spliced into the cache directory name -- one
    stray "Create lock file: ..." produced a cache directory named after that
    message, and a second process that did not log it built into a different
    one.

    The answer is cached against the driver version, so the usual import does
    not start a second interpreter at all. That child is also the one that has
    deadlocked: it was started while this process held the build lock, and
    anything it did that needed the lock waited for a holder that was waiting
    for it.
    """
    driver = ""
    try:
        with open("/proc/driver/nvidia/version") as f:
            driver = f.readline().strip()
    except OSError:
        pass
    return jit_utils.probe.cached(
        "cuda_archs", [], lambda: _read_cuda_archs(), extra=driver)


def _read_cuda_archs():
    try:
        child = sp.run(
            [sys.executable, "-m", "jittor_utils.query_cuda_cc"],
            stdout=sp.PIPE, stderr=sp.PIPE,
            env=env_config.child_env(log_v=0, log_silent=1))
    except OSError as error:
        LOG.v(f"could not query cuda archs: {error}")
        return None
    if child.returncode != 0:
        LOG.v("could not query cuda archs: "
              + child.stderr.decode("utf8", "replace"))
        return None
    archs = set()
    for token in child.stdout.decode("utf8", "replace").split():
        if token.isdigit():
            archs.add(token)
        else:
            LOG.v(f"ignoring non-numeric output from query_cuda_cc: {token!r}")
    return sorted(archs)

def parse_nvcc_arch_list(text):
    """The integers in an ``nvcc --list-gpu-arch`` listing."""
    archs = set()
    for token in text.split():
        token = token.strip()
        if token.startswith("compute_") and token[len("compute_"):].isdigit():
            archs.add(int(token[len("compute_"):]))
    return sorted(archs)


def _read_nvcc_archs(nvcc):
    try:
        child = sp.run([nvcc, "--list-gpu-arch"], stdout=sp.PIPE, stderr=sp.PIPE)
    except OSError as error:
        LOG.v(f"could not list nvcc gpu archs: {error}")
        return []
    if child.returncode != 0:
        # --list-gpu-arch arrived in CUDA 11.1. Older toolkits fall back to
        # the version table in the caller.
        LOG.v("could not list nvcc gpu archs: "
              + child.stderr.decode("utf8", "replace"))
        return []
    return parse_nvcc_arch_list(child.stdout.decode("utf8", "replace"))


def query_nvcc_archs(nvcc):
    """Virtual architectures this nvcc can generate code for, newest last.

    This used to be a literal: ``max_arch = 90``, with a version ladder below
    it for older toolkits. A literal ceiling ages into a wrong answer -- every
    CUDA release after the one it was written for reads as 90 -- and the
    consequence is not a missing optimisation but a build that cannot run.
    Empty if this nvcc is too old to be asked (before CUDA 11.1).
    """
    return jit_utils.probe.cached("nvcc_archs:" + nvcc, [nvcc],
                                  lambda: _read_nvcc_archs(nvcc))


def select_cuda_archs(requested, max_arch, min_arch=30):
    """The architectures to actually compile for, sorted and deduplicated.

    Anything below ``min_arch`` is dropped: nvcc cannot target it at all.
    Anything above ``max_arch`` is replaced by ``max_arch``, which is only
    survivable because ``cuda_arch_flags`` keeps that architecture's PTX for
    the driver to JIT -- see the warning below.
    """
    archs = []
    for arch in requested:
        arch = int(arch)
        if arch < min_arch:
            LOG.w(f"CUDA arch({arch})<{min_arch} is not supported")
            continue
        if arch > max_arch:
            LOG.w(f"CUDA arch({arch}) is newer than anything this nvcc can "
                  f"generate code for (highest is {max_arch}). Building for "
                  f"compute_{max_arch} and embedding its PTX, which the driver "
                  f"JIT-compiles for sm_{arch} the first time each kernel runs. "
                  f"A CUDA toolkit that knows sm_{arch} removes that cost.")
            arch = max_arch
        archs.append(arch)
    return sorted(set(archs))


def cuda_arch_flags(archs):
    """``-gencode`` for each architecture, keeping PTX for the newest one.

    The old form was ``-arch=compute_<min>`` followed by one ``-code=sm_X`` per
    architecture. That has two faults. Every cubin is compiled from the
    *lowest* virtual architecture, so an sm_90 cubin is built from compute_70
    sources and none of the newer instructions are available to it. And the
    product contains cubins only: no PTX. A GPU newer than every ``sm_`` in
    the list therefore has neither a loadable cubin nor anything to JIT, and
    fails at launch with "no kernel image is available for execution on the
    device". PTX for the highest architecture is the only thing that makes a
    build forward compatible at all.
    """
    archs = sorted(set(int(arch) for arch in archs))
    if not archs:
        return ""
    # Two things about the spelling, both learned the hard way.
    #
    # `--generate-code=...` rather than `-gencode ...`: cache_compile parses
    # this command line to find the source files, and takes any token that is
    # not itself an option to be one. Split across two tokens, the value is
    # read as a file name and every CUDA operator fails to compile with
    # "Source read failed: arch=compute_89,...".
    #
    # Two clauses rather than the `code=[sm_X,compute_X]` list: the command
    # goes to a shell, where square brackets are a glob pattern. Unmatched
    # patterns survive today, but nothing here needs to depend on that.
    parts = [f" --generate-code=arch=compute_{arch},code=sm_{arch} "
             for arch in archs]
    top = archs[-1]
    parts.append(f" --generate-code=arch=compute_{top},code=compute_{top} ")
    return "".join(parts)


def check_pybt(gdb_path, python_path):
    if gdb_path=='' or python_path=='':
        return False
    return True
    # return False

def check_debug_flags():
    global is_debug
    is_debug = 0
    if build_flag("debug"):
        is_debug = 1
        global cc_flags
        cc_flags += " -g -DNODE_MEMCHECK "

def check_save_mem_flags():
    """Warn when the unfinished swapping build is explicitly enabled.

    ``jt_config_macro_flags`` adds the normalized define with the other
    source-controlled JT_* macros later in command-line construction.
    """
    flags = jit_utils.save_mem_build_flags()
    if not flags:
        return
    LOG.w("JT_SAVE_MEM=1: memory swapping is built in. It is unfinished -- "
          "share_with, migrate, the dual allocator and foreign allocators are "
          "all still on its TODO list (src/mem/swap.h) -- and it needs its own "
          "build, so this configuration compiles into a cache directory of "
          "its own.")

cc_flags = " "
# os.RTLD_NOW | os.RTLD_GLOBAL cause segfault when import torch first
import_flags = os.RTLD_NOW | os.RTLD_GLOBAL
if platform.system() == 'Linux':
    import_flags |= os.RTLD_DEEPBIND
# if cc_type=="icc":
#     # weird link problem, icc omp library may conflict and cause segfault
#     import_flags = os.RTLD_NOW | os.RTLD_GLOBAL
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL
if platform.system() == 'Linux':
    import_flags |= os.RTLD_DEEPBIND

with jit_utils.import_scope(import_flags):
    jit_utils.try_import_jit_utils_core()

jittor_path = find_jittor_path()
if os.name == 'nt':
    # prevent windows recompile
    jittor_path = jittor_path.lower()
check_debug_flags()
check_save_mem_flags()

sys.path.append(cache_path)
LOG.i(f"Jittor({__version__}) src: {jittor_path}")
LOG.i(f"{jit_utils.cc_type} at {jit_utils.cc_path}{jit_utils.get_version(jit_utils.cc_path)}")
LOG.i(f"cache_path: {cache_path}")

with jit_utils.import_scope(import_flags):
    jit_utils.try_import_jit_utils_core()

python_path = sys.executable
# sometime python do not return the correct sys executable
# this will happend when multiple python version installed
ex_python_path = python_path + '.' + str(sys.version_info.minor)
if os.path.isfile(ex_python_path):
    python_path = ex_python_path

def _discover_cuda_compiler(requested_backend):
    if requested_backend not in (None, "cuda"):
        return ""
    nvcc = None
    if install_cuda.has_installation() or os.name == 'nt':
        nvcc = install_cuda.install_cuda()
        if nvcc:
            nvcc = try_find_exe(nvcc)
    if not nvcc:
        nvcc = env_or_try_find('nvcc_path', 'nvcc') or \
            try_find_exe('/usr/local/cuda/bin/nvcc') or \
            try_find_exe('/usr/bin/nvcc') or \
            try_find_exe('/opt/cuda/bin/nvcc')
    if not nvcc:
        nvcc = install_cuda.install_cuda()
        if nvcc:
            nvcc = try_find_exe(nvcc)
    return build_env("nvcc_path", nvcc or "")


_requested_backend = _backend_discovery.requested_backend()
nvcc_path = _discover_cuda_compiler(_requested_backend)
gdb_path = env_or_try_find('gdb_path', 'gdb')
addr2line_path = try_find_exe('addr2line')
has_pybt = check_pybt(gdb_path, python_path)

if nvcc_path:
    # gen cuda key for cache_path
    cu = "cu"
    v = jit_utils.get_version(nvcc_path)[1:-1]
    nvcc_version = list(map(int,v.split('.')))
    cu += v
    cuda_wheel_stack = install_cuda.get_cuda_wheel_stack(v)
    if cuda_wheel_stack:
        cu += "_" + cuda_wheel_stack.fingerprint
    archs = query_cuda_archs()
    if archs is not None:
        if len(archs) == 0:
            LOG.e("No GPU Device Found!")
        cu += "_sm_" + "_".join(archs)
    LOG.i("cuda key:", cu)
    cache_path = os.path.join(cache_path, cu)
    # Ahead of the plain cache directory, which is already on the path. Any run
    # that imports Jittor without nvcc builds a CPU-only jittor_core into that
    # parent directory, and appending here would leave it earlier on the path --
    # so it wins the import and every CUDA op afterwards fails with
    # "Op ... doesn't have cuda version", for this run and every later one
    # sharing the cache.
    parent_cache = os.path.dirname(cache_path)
    if cache_path in sys.path:
        sys.path.remove(cache_path)
    sys.path.insert(
        sys.path.index(parent_cache) if parent_cache in sys.path else len(sys.path),
        cache_path,
    )


def check_clang_latest_supported_cpu():
    output = run_cmd('clang --print-supported-cpus')
    def find_latest_chip_version(pattern_prefix):
        apple_cpus = [cpu.strip() for cpu in output.split('\n') if pattern_prefix in cpu]
        apple_cpu_id = max([int(cpu[7:]) for cpu in apple_cpus])
        return pattern_prefix + str(apple_cpu_id)
    if 'apple-m' in output:
        return find_latest_chip_version('apple-m')
    else:
        return find_latest_chip_version('apple-a')

# cc_flags += " -Wall -Werror -Wno-unknown-pragmas -std=c++14 -fPIC "
cc_flags += " -Wall -Wno-unknown-pragmas -std=c++14 -fPIC "
# 1. Arch/CPU specific optimization
if platform.machine() in ["x86_64", "AMD64"]:
    cc_flags += " -march=native "
elif platform.machine() == 'arm64' and platform.system() == "Darwin":
    cc_flags += f" -mcpu={check_clang_latest_supported_cpu()} "
cc_flags += " -fdiagnostics-color=always "

#: Build-configuration macros the C++ sources test with `#ifdef`, and which the
#: environment can turn on. There are only a few, they change rarely, and
#: `tests/compiler/test_jt_config_macros.py` greps the sources and fails if
#: this tuple stops matching -- so the list cannot drift without anyone
#: noticing.
#:
#: This used to be discovered by `cache_compile.cc`, which scanned every source
#: it was already reading for dependency tracking and rewrote the compiler
#: command line in place when it found one. That coupled two unrelated jobs:
#: deciding the command line, which must happen *before* a compile, and
#: collecting dependencies, which the compiler can only report *after* one.
#: While they shared a scanner, the compiler's own `-MD -MF` could not be used
#: for dependencies at all -- the first, cold compile would have gone out
#: without its `-D`. Deciding the flags here, from a declared list, is what
#: separates them (task 9.21).
JT_CONFIG_MACROS = (
    "JT_CHECK_NAN",
    "JT_GRAPH_BUILD_PROFILE",
    "JT_HAS_HALF_SIMD",
    "JT_HCCL_NO_MPI",
    "JT_NCCL_NO_MPI",
    "JT_SAVE_MEM",
    "JT_SYNC",
    "JT_bfs_executor",
)

def jt_config_macro_flags(environ=None):
    """`-D` flags for the JT_* macros this environment turns on.

    A variable that is unset, empty or "0" is off, which is the same rule the
    scanner used. Everything downstream derives from cc_flags, so setting them
    here reaches the nvcc command line too.
    """
    environ = os.environ if environ is None else environ
    flags = []
    for name in JT_CONFIG_MACROS:
        if name == "JT_SAVE_MEM":
            value = jit_utils.save_mem_build_flags(environ)
            if value:
                flags.append(value)
            continue
        value = environ.get(name)
        if value in (None, "", "0"):
            continue
        flags.append(f" -D{name}={value} ")
    return "".join(flags)

cc_flags += jt_config_macro_flags()
# 2. Non standard include path
if platform.system() == 'Darwin':
    # TODO: if not using apple clang, there is no need to add -lomp
    cc_flags += " -undefined dynamic_lookup -lomp "
    if os.environ.get("CONDA_PREFIX", None):
        cc_flags += f" -L{os.path.join(os.environ['CONDA_PREFIX'], 'lib')} "
    # if platform.machine() == "arm64":
    #     cc_flags += " -I/opt/homebrew/include -L/opt/homebrew/lib  "
    # Homebrew does not symlink the openmp library (libomp >= 15.0.6) into /opt/homebrew/lib
    homebrew_openmp_paths = [
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp"
    ]
    for openmp_path in homebrew_openmp_paths:
        if os.path.exists(openmp_path):
            cc_flags += f" -I{openmp_path}/include -L{openmp_path}/lib"

# 3. User specified flags
#
# These are *appended*. That is now the only meaning the setting has: the native
# `cc_flags` flag used to be replaced wholesale by the same variable in
# `log.h`'s static initializer, and then overwritten again by the
# `flags.cc_flags = ...` assignment at the end of this file, so one name had two
# documented behaviours of which only this one ever survived. The core no longer
# reads any compiler-owned build flag from the environment (see
# `compiler_owned_flag_names` in `src/utils/log.cc`).
_user_cc_flags = build_env("cc_flags")
if _user_cc_flags is not None:
    cc_flags += _user_cc_flags + ' '

cc_flags += " -lstdc++ -ldl -shared "

opt_flags = ""

py_include = jit_utils.get_py3_include_path()
LOG.v(f"py_include: {py_include}")
extension_suffix = jit_utils.get_py3_extension_suffix()
lib_suffix = extension_suffix.rsplit(".", 1)[0]
LOG.v(f"extension_suffix: {extension_suffix}")
so = ".so" if os.name != 'nt' else ".dll"


kernel_opt_flags = build_env("kernel_flags", "") + opt_flags
if platform.system() == 'Darwin':
    # TODO: if not using apple clang, cannot add -Xpreprocessor
    kernel_opt_flags += " -Xpreprocessor -fopenmp "
elif cc_type != 'cl':
    kernel_opt_flags += " -fopenmp "
def fix_cl_flags(cmd):
    output = shsplit(cmd)
    output2 = []
    libpaths = []
    for s in output:
        if s.startswith("-l:"):
            output2.append(s)
        elif s.startswith("-l") and ("cpython" in s or "lib" in s):
            if platform.system() == 'Darwin':
                fname = s[2:] + ".so"
                for path in reversed(libpaths):
                    full = os.path.join(path, fname).replace("\"", "")
                    if os.path.isfile(full):
                        output2.append(full)
                        break
                else:
                    output2.append(s)
            else:
                output2.append(f"-l:{s[2:]}.so")
        elif s.startswith("-L"):
            libpaths.append(s[2:])
            output2.append(f"{s} -Wl,-rpath,{s[2:]}")
        else:
            output2.append(s)
    return " ".join(output2)

if os.name == 'nt':
    if cc_type == 'g++':
        pass
    elif cc_type == 'cl':
        py3_link_path = jit_utils.get_py3_link_path()
        cc_flags = remove_flags(cc_flags, ["-f", "-m"])
        cc_flags = cc_flags.replace("-std=c++14", "-std=c++17")
        cc_flags = cc_flags.replace("-lstdc++", "")
        cc_flags = cc_flags.replace("-ldl", "")
        cc_flags += f" -L\"{py3_link_path}\" -lpython3{sys.version_info.minor} "
        cc_flags += " -EHa -MD -utf-8 "
        import jittor_utils
        if jittor_utils.msvc_path:
            mp = jittor_utils.msvc_path
            cc_flags += f' -nologo -I"{mp}\\VC\\include" -I"{mp}\\win10_kits\\include\\ucrt" -I"{mp}\\win10_kits\\include\\shared" -I"{mp}\\win10_kits\\include\\um" -DNOMINMAX '
            cc_flags += f' -L"{mp}\\VC\\lib" -L"{mp}\\win10_kits\\lib\\um\\x64" -L"{mp}\\win10_kits\\lib\\ucrt\\x64" '
        win_libpaths = {}
        def fix_cl_flags(cmd):
            cmd = cmd.replace(".o ", ".obj ")
            cmd = cmd.replace(".o\"", ".obj\"")
            if cmd.endswith(".o"): cmd += "bj"
            if " -o " in cmd:
                if " -shared " in cmd:
                    cmd = cmd.replace(" -o ", " -Fe: ")
                    output = shsplit(cmd.split("-Fe:")[1].strip())[0]
                    cmd += f" -DEF:{output}.def -IGNORE:4102 -IGNORE:4197 -IGNORE:4217 "

                elif " -c -o " in cmd:
                    cmd = cmd.replace(" -c -o ", " -c -Fo: ")
            flags = shsplit(cmd)
            output = []
            output2 = []
            for f in flags:
                if f.startswith("-link"):
                    pass
                elif f.startswith("-l"):
                    output2.append(f[2:]+".lib")
                elif f.startswith("-LIB"):
                    output2.append(f)
                elif f.startswith("-LD"):
                    output.append(f)
                elif f.startswith("-L"):
                    path = f[2:].replace("\"", "")
                    if path not in win_libpaths:
                        win_libpaths[path] = 1
                        os.add_dll_directory(path)
                        os.environ["PATH"] = f";{path};" + os.environ["PATH"]
                    output2.append("-LIBPATH:"+f[2:])
                elif ".lib" in f:
                    output2.append(f)
                elif f.startswith("-DEF:"):
                    output2.append(f)
                elif f.startswith("-W") or f.startswith("-f"):
                    pass
                elif f.startswith("-std="):
                    output.append(f.replace("=", ":"))
                else:
                    output.append(f)
            cmd = " ".join(output)
            if len(output2):
                cmd += " -link " + " ".join(output2)
            cmd = cmd.replace("-include", "-FI")
            cmd = cmd.replace("-shared", "-LD")
            return cmd

if ' -O' not in cc_flags:
    if build_flag("debug"):
        opt_flags += " -O0 "
    else:
        opt_flags += " -O2 "
    # -O3, not -Ofast. `-Ofast` implies `-ffast-math`, which implies
    # `-ffinite-math-only`: a promise that no operand is ever infinite or NaN.
    # The compiler optimises on that promise, and operands that *are* infinite
    # take whatever path the transformed code happens to produce -- `1/0` came
    # back as `nan` instead of `inf`, and `-inf/0` likewise (KI-BACKEND-005).
    # The wrong answers are plausible rather than obviously broken, which is
    # what makes them expensive: a fully masked attention row subtracts its own
    # `-inf` maximum, and a finite result there produces a well-formed but
    # wrong softmax instead of an obvious `nan`.
    #
    # The project already knew: `nan_checker` had `-Ofast` stripped and `-O2`
    # substituted, because a NaN check compiled under a promise that NaN does
    # not occur cannot work. That exemption was applied where the problem was
    # noticed rather than where it applies.
    #
    # The reassociation `-ffast-math` also grants is not what was making
    # reductions fast: g++ 12.3 does not vectorise the real reduction kernels,
    # because the runtime `storage_stride(0)` blocks it. Accuracy at scale is
    # now `BlockedReductionPass`'s job, stated in the code rather than left to
    # a flag that also breaks arithmetic.
    kernel_opt_flags += " -O3 "
lto_flags = ""
if build_flag("enable_lto"):
    if cc_type == "icc":
        lto_flags = " -flto -ipo -ipo-c "
    elif cc_type == "g++":
        lto_flags = " -flto -fuse-linker-plugin "
    else:
        lto_flags = " -flto "

make_cache_dir(cache_path)
make_cache_dir(os.path.join(cache_path, "jit"))
make_cache_dir(os.path.join(cache_path, "obj_files"))
make_cache_dir(os.path.join(cache_path, "gen"))
make_cache_dir(os.path.join(cache_path, "tmp"))
ck_path = os.path.join(cache_path, "checkpoints")
make_cache_dir(ck_path)

# build cache_compile
cc_flags += f" -I\"{core_root(jittor_path)}\" "
cc_flags += f" -I\"{backend_root(jittor_path, 'cuda')}\" "

cc_flags += py_include

check_cache_compile()
LOG.v(f"Get cache_compile: {jit_utils.cc}")

# check cuda
is_cuda = has_cuda = has_acl = has_rocm = has_corex = 0
cuda_sdk_flags = cuda_link_flags = ""
check_cuda()
if _requested_backend == "cuda" and not has_cuda:
    raise RuntimeError("JT_BACKEND=cuda requires a usable CUDA compiler; "
                       "set JT_BUILD_NVCC_PATH")
nvcc_flags = build_env("nvcc_flags", "")
def convert_nvcc_flags(value):
    return value
if has_cuda:
    nvcc_flags += cc_flags + cuda_sdk_flags
    def convert_nvcc_flags(nvcc_flags):
        # nvcc don't support -Wall option
        if os.name == 'nt':
            nvcc_flags = nvcc_flags.replace("-fp:", "-Xcompiler -fp:")
            nvcc_flags = nvcc_flags.replace("-EH", "-Xcompiler -EH")
            nvcc_flags = nvcc_flags.replace("-M", "-Xcompiler -M")
            nvcc_flags = nvcc_flags.replace("-utf", "-Xcompiler -utf")
            nvcc_flags = nvcc_flags.replace("-nologo", "")
            nvcc_flags = nvcc_flags.replace("-std:", "-std=")
            nvcc_flags = nvcc_flags.replace("-Fo:", "-o")
            nvcc_flags = nvcc_flags.replace("-LD", "-shared")
            nvcc_flags = nvcc_flags.replace("-LIBPATH:", "-L")
            nvcc_flags = nvcc_flags.replace("-link", "")
            def func(x):
                if ".lib" not in x: return x
                x = x.replace("\"", "")
                a = os.path.dirname(x)
                b = os.path.basename(x)
                if not b.endswith(".lib"):
                    return x
                return f"-L\"{a}\" -l{b[:-4]}"
            nvcc_flags = map_flags(nvcc_flags, func)
        if nvcc_version >= [11,4]:
            nvcc_flags = nvcc_flags.replace("-std=c++17", "-std=c++14 -Xcompiler -std:c++14")
        else:
            nvcc_flags = nvcc_flags.replace("-std=c++17", "")
        nvcc_flags = nvcc_flags.replace("-Wall", "")
        nvcc_flags = nvcc_flags.replace("-Wno-unknown-pragmas", "")
        nvcc_flags = nvcc_flags.replace("-fopenmp", "")
        nvcc_flags = nvcc_flags.replace("-march", "-Xcompiler -march")
        nvcc_flags = nvcc_flags.replace("-Werror", "")
        nvcc_flags = nvcc_flags.replace("-fPIC", "-Xcompiler -fPIC")
        nvcc_flags = nvcc_flags.replace("-fdiagnostics", "-Xcompiler -fdiagnostics")
        nvcc_flags += f" -x cu --cudart=shared -ccbin=\"{cc_path}\" --use_fast_math "
        # nvcc warning is noise
        nvcc_flags += " -w "
        nvcc_flags += f" -I\"{os.path.join(backend_root(jittor_path, 'cuda'), 'include')}\" "
        if build_flag("cuda_debug"):
            nvcc_flags += " -G "
        return nvcc_flags
    nvcc_flags = convert_nvcc_flags(nvcc_flags)

build_config = BuildConfig(
    backend="cuda" if has_cuda else "cpu", cc_path=cc_path, cc_type=cc_type,
    cc_flags=cc_flags, nvcc_path=nvcc_path, nvcc_flags=nvcc_flags,
    kernel_flags=kernel_opt_flags, cache_path=cache_path, jittor_path=jittor_path,
    has_cuda=bool(has_cuda), is_cuda=bool(is_cuda), has_accelerator=bool(has_cuda),
    convert_nvcc_flags=convert_nvcc_flags,
    backend_sources=(
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "runtime/driver.cc"), flags=cuda_sdk_flags),
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "runtime/nan_checker.cc"), flags=cuda_sdk_flags),
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "kernels/debug/nan_checker.cu"),
                    language="cuda", flags=cuda_sdk_flags),
    ) if has_cuda else (),
    backend_link_flags=cuda_link_flags,
    extension_compile_flags=cuda_sdk_flags,
    kernel_compiler=nvcc_path if has_cuda else cc_path,
    kernel_language="cuda" if has_cuda else "cxx",
    kernel_compile_flags=cuda_sdk_flags,
    kernel_device_link=bool(has_cuda),
    kernel_source_roots=(os.path.join(backend_root(jittor_path, "cuda"), "kernels/core"),) if has_cuda else (),
)
_backend_provider = _backend_discovery.load_backend_provider(_requested_backend)
backend_modules = () if _backend_provider is None else (_backend_provider,)
if _backend_provider is not None:
    configured = _backend_provider.configure(make_backend_context(build_config))
    if not isinstance(configured, BuildConfig):
        raise TypeError("backend configure(context) must return BuildConfig")
    build_config = configured
for _env_name, _env_value in build_config.environment.items():
    os.environ[_env_name] = _env_value

# Only the bootstrap publishes compatibility names. Providers return values;
# they never mutate this module or append into its core source inventory.
cc_path, cc_type, cc_flags = build_config.cc_path, build_config.cc_type, build_config.cc_flags
nvcc_path, nvcc_flags = build_config.nvcc_path, build_config.nvcc_flags
kernel_opt_flags = build_config.kernel_flags
jittor_path, cache_path = build_config.jittor_path, build_config.cache_path
has_accelerator = build_config.has_accelerator
has_cuda, has_acl = build_config.has_cuda, build_config.has_acl
has_rocm, has_corex = build_config.has_rocm, build_config.has_corex
hipcc_path, tikcc_path = build_config.hipcc_path, build_config.tikcc_path
setup_fake_cuda_lib = build_config.setup_fake_cuda_lib
extra_core_files = build_config.extra_core_files
convert_nvcc_flags = build_config.convert_nvcc_flags or convert_nvcc_flags

is_cuda = build_config.is_cuda

# build core
core_output_name = 'jittor_core' + extension_suffix
core_output_path = os.path.join(cache_path, core_output_name)
cc_flags += f' -I\"{cache_path}\" -L\"{cache_path}\" -L\"{jit_utils.cache_path}\" '
cc_flags += f" -l\"jit_utils_core{lib_suffix}\" "

#: The flags the core itself is compiled with, frozen here.
#:
#: ``cc_flags`` keeps growing after this point -- ``-ljittor_core`` goes on as
#: soon as the core is linked, and it is what JIT ops are compiled with. So it
#: cannot be what the stamp records: read again after the import it no longer
#: matches what the build used, every check reports "stale", and the whole
#: point of the stamp is lost. That is not a hypothetical; it is what the
#: first version of this did.
core_cc_flags = cc_flags

#: Source files the core was built from, in compile order. Filled in by
#: :func:`build_core` -- from the build itself, or from the stamp it left
#: behind when there was nothing to build.
files = []

#: Stamp version. Bump it when the meaning of a recorded field changes, so an
#: older stamp is treated as "no stamp" instead of being misread as current.
CORE_BUILD_STAMP_VERSION = 2
CORE_GENERATOR_SIGNATURE_VERSION = 1


def core_build_stamp_path():
    return core_output_path + ".build_stamp.json"


def core_source_signature(root=None):
    """``{relative path: [mtime_ns, size]}`` for every core source file.

    ``root`` overrides the tree that is walked. It exists for the tests: the
    startup configuration on this module is frozen after bootstrap (see
    ``_runtime/state.py``), so ``jittor_path`` can no longer be patched, and a
    walk that can only ever look at the real checkout cannot be shown to
    notice a new or same-size-edited file.

    A walk of ``src/`` and ``backends/`` rather than the individual globs the
    generators use, so that a source or header that did not exist at the last
    build shows up too -- a dependency list recorded from that build can only
    name files that were already there.

    External headers (the toolkit's, Python's, the standard library's) are
    deliberately absent. ``cache_path`` is already partitioned by compiler
    version, Python version and cuda key, so a toolchain change lands in a
    different cache directory and gets a cold build rather than a stale one.
    Costs about 3 ms for ~600 files, which is what makes it affordable on
    every import.
    """
    tree = jittor_path if root is None else root
    signature = {}
    roots = [("src", core_root(tree) if root is None else os.path.join(tree, "src"))]
    for backend in ("cpu", "cuda", "acl", "rocm", "corex", "comm"):
        try:
            backend_directory = backend_root(tree, backend)
        except FileNotFoundError:
            continue
        roots.append(("backends/" + backend, backend_directory))
    for top, root_dir in roots:
        for directory, _, names in os.walk(root_dir):
            for name in names:
                path = os.path.join(directory, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                signature[os.path.join(top, os.path.relpath(path, root_dir))] = \
                    [info.st_mtime_ns, info.st_size]
    return signature


def core_generator_signature():
    """Describe Python code that generates the core translation units.

    These files live outside ``src/`` and ``backends/``, so the source walk used
    by the core stamp cannot see edits to them.  Keep both a cheap stat record
    and a content digest: an editor or checkout normally changes mtime, while
    the digest also catches an in-place same-size/same-mtime replacement.
    Missing files are represented explicitly so a partial installation cannot
    accidentally reuse a stamp made by a complete one.
    """
    paths = [
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "codegen.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "compilation.py"),
        os.path.abspath(pyjt_compiler.__file__),
        os.path.abspath(_build_config_api.__file__),
        os.path.abspath(_backend_discovery.__file__),
        os.path.abspath(_backend_resources_api.__file__),
        os.path.join(jittor_path, "_runtime", "flag_policy.py"),
    ]
    records = {}
    for path in paths:
        relative = os.path.relpath(path, jittor_path)
        try:
            info = os.stat(path)
        except OSError:
            records[relative] = None
            continue
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            records[relative] = None
            continue
        records[relative] = {
            "mtime_ns": info.st_mtime_ns,
            "size": info.st_size,
            "sha256": digest.hexdigest(),
        }
    return {
        "version": CORE_GENERATOR_SIGNATURE_VERSION,
        "files": records,
    }


def core_build_ingredients():
    """Everything besides the sources that the core's compile commands use.

    Kept as the ingredients rather than the assembled command lines because
    assembling those needs the generated pyjt sources, which is one of the
    steps a current build gets to skip.
    """
    return {
        "version": __version__,
        "cc_path": cc_path,
        "cc_type": cc_type,
        "cc_flags": core_cc_flags,
        "opt_flags": opt_flags,
        "lto_flags": lto_flags,
        "nvcc_path": nvcc_path,
        "nvcc_flags": build_env("nvcc_flags", ""),
        "extension_suffix": extension_suffix,
        "lib_suffix": lib_suffix,
        "has_cuda": int(bool(has_cuda)),
        "has_accelerator": int(bool(has_accelerator)),
        "has_rocm": int(bool(has_rocm)),
        "is_cuda": int(bool(is_cuda)),
        "extra_core_files": list(extra_core_files),
        "backend_sources": [vars(source) for source in build_config.backend_sources],
        "backend_link_flags": build_config.backend_link_flags,
        "kernel_compiler": build_config.kernel_compiler,
        "kernel_language": build_config.kernel_language,
        "kernel_compile_flags": build_config.kernel_compile_flags,
        "jit_utils_core_files": list(jit_utils_core_files),
        "os_name": os.name,
        "generators": core_generator_signature(),
    }


def _core_output_signature():
    try:
        info = os.stat(core_output_path)
    except OSError:
        return None
    return [info.st_mtime_ns, info.st_size]


def core_build_is_current(signature=None, ingredients=None):
    """True when ``jittor_core`` is already built from exactly these inputs.

    The point of asking is that finding out the expensive way costs about
    0.9 s of every warm import: regenerating headers that come out
    byte-identical, then handing all ~180 compile commands to a 16-process
    pool so each worker can hash a dependency closure and report that there
    is nothing to do.

    Deliberately conservative in one direction only. A mismatch means "build",
    and building re-runs the full per-file check that was always there, so a
    stamp that is wrong about being stale costs time and nothing else. The one
    way to be wrong in the other direction is to edit a file without changing
    either its size or its nanosecond mtime, which no editor, compiler or
    ``git checkout`` does.
    """
    stamp = None
    try:
        with open(core_build_stamp_path(), "r", encoding="utf8") as handle:
            stamp = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    if stamp.get("stamp_version") != CORE_BUILD_STAMP_VERSION:
        return False
    if stamp.get("output") != _core_output_signature():
        return False
    if stamp.get("ingredients") != (ingredients or core_build_ingredients()):
        return False
    if stamp.get("sources") != (signature or core_source_signature()):
        return False
    if not isinstance(stamp.get("files"), list):
        return False
    return stamp


def _write_core_build_stamp(signature, ingredients, compile_order):
    """Record what the core was just built from.

    Written to a temporary name and renamed into place: a reader that arrives
    mid-write must see either the old stamp or the new one, never a truncated
    file that json would reject and that would silently cost the next import
    a full rebuild.
    """
    stamp = {
        "stamp_version": CORE_BUILD_STAMP_VERSION,
        "output": _core_output_signature(),
        "ingredients": ingredients,
        "sources": signature,
        "files": list(compile_order),
    }
    if stamp["output"] is None:
        return
    temporary = core_build_stamp_path() + ".tmp." + str(os.getpid())
    try:
        with open(temporary, "w", encoding="utf8") as handle:
            json.dump(stamp, handle)
        os.replace(temporary, core_build_stamp_path())
    except OSError as error:
        # A read-only or full cache directory must not stop an import that
        # otherwise succeeded; the cost is that the next import checks the
        # expensive way again.
        LOG.v("could not write core build stamp: %s" % error)
        try:
            os.remove(temporary)
        except OSError:
            pass


class BuildNotAllowed(Exception):
    """Raised instead of compiling when builds are not allowed.

    ``import jittor`` is supposed to load a build, not produce one, but for
    most of Jittor's history the difference was invisible: a cache that did
    not match compiled for forty seconds to several minutes and said so only
    through progress lines. There was no way for a deployment to state "this
    import must not build" and be told when it was about to.

    ``JITTOR_NO_BUILD=1`` is that statement. Under it, anything that would
    have to compile raises this instead, naming the explicit entry point that
    is allowed to build. Offline and read-only installations are the reason it
    exists: there the compile does not merely cost a minute, it fails a minute
    later for a reason that has nothing to do with the actual problem.
    """


def build_is_allowed():
    """False when ``JITTOR_NO_BUILD`` forbids compiling during this process."""
    return os.environ.get("JITTOR_NO_BUILD", "0") in ("0", "")


def _refuse_build(what):
    raise BuildNotAllowed(
        "JITTOR_NO_BUILD=1 is set, and %s is not built for this "
        "configuration -- importing jittor would have to compile it.\n"
        "Build it once with the explicit entry point:\n"
        "    python -m jittor_utils.bootstrap\n"
        "then import with the same JITTOR_HOME and the same build flags "
        "(nvcc_path, cc_flags, nvcc_flags), because each combination gets "
        "its own cache directory.\n"
        "cache_path: %s" % (what, cache_path))


#: Stamp version for build products other than the core. Independent of the
#: core's, so either can change meaning without invalidating the other's.
PRODUCT_BUILD_STAMP_VERSION = 1

_SOURCE_EXTENSIONS = (".h", ".hpp", ".cuh", ".inc", ".cc", ".cpp", ".cu", ".c")


def build_core(force=False):
    """Generate the core's sources and compile ``jittor_core``.

    The single entry point for turning ``src/**`` into the core extension:
    the flag and test scanners, the op-maker header, the pyjt bindings, the
    compile order, and the compile itself. It used to be nine hundred lines
    of module body with no name, which is why "does importing jittor build
    anything" had no answer short of reading all of it.

    Returns True if it built, False if the stamp said the build was already
    current -- in which case nothing was generated, no compiler ran, and the
    only cost was stat()ing the source tree.
    """
    global files
    signature = core_source_signature()
    ingredients = core_build_ingredients()

    if not force:
        stamp = core_build_is_current(signature, ingredients)
        if stamp:
            files = stamp["files"]
            LOG.v("core build is current, skipping generation and compile")
            return False
        if not build_is_allowed():
            _refuse_build("jittor_core")

    gen_jit_flags()
    gen_jit_tests()
    op_headers = glob.glob(core_root(jittor_path)+"/ops/**/*op.h", recursive=True)
    jit_src = gen_jit_op_maker(op_headers)
    LOG.vvvv(jit_src)
    with open(os.path.join(cache_path, "gen", "jit_op_maker.h"), 'w', encoding='utf8') as f:
        f.write(jit_src)
    # gen pyjt
    pyjt_gen_src = pyjt_compiler.compile(cache_path, jittor_path)

    # initialize order:
    # 1. registers
    # 2. generate source
    # 3. op_utils
    # 4. other
    files2 = pyjt_gen_src
    files4 = glob.glob(core_root(jittor_path)+"/**/*.cc", recursive=True)
    # Keep the historical "src/..." spelling: at_beginning/at_last and
    # jit_utils_core_files below are matched with list.remove, which compares
    # strings. 4.15 moved the core out of jittor_path, so the old
    # f[len(jittor_path)+1:] slice would produce garbage; relpath against
    # core_root reproduces exactly the same names, and _source_path maps them
    # back to wherever the core now lives.
    files4 = [os.path.join("src", os.path.relpath(f, core_root(jittor_path)))
              for f in files4]
    indexing_schedule_source = os.path.join(
        backend_root(jittor_path, "cuda"), "kernels", "core", "indexing_schedule_codegen.cc")
    files4.append(indexing_schedule_source)
    if has_accelerator:
        files4 += [path for path in sorted(glob.glob(os.path.join(
            backend_root(jittor_path, "cuda"), "kernels", "core", "*_codegen.cc")))
            if path != indexing_schedule_source]
    at_beginning = [
        "src/ops/op_utils.cc",
        "src/ops/op_register.cc",
        "src/runtime/init.cc",
        "src/core/event_queue.cc",
        "src/mem/allocator/sfrl_allocator.cc",
        "src/mem/allocator.cc",
        "src/type/nano_string.cc",
    ]
    at_last = [
        "src/runtime/profiler/profiler.cc",
        "src/core/executor.cc",
    ]
    if os.name == 'nt':
        at_beginning = [ x.replace('/','\\') for x in at_beginning ]
        at_last = [ x.replace('/','\\') for x in at_last ]
    for i in range(len(at_beginning)):
        files4.remove(at_beginning[i])
        files4.insert(i, at_beginning[i])
    for v in at_last:
        files4.remove(v)
        files4.append(v)
    registers = [ name for name in files4 if "register" in name ]
    for name in registers: files4.remove(name)
    files = registers + files2 + files4
    files += extra_core_files
    for file in jit_utils_core_files:
        files.remove(file)
    LOG.vv("compile order:", files)

    # Everything a build needs, checked once and reported together -- but only
    # when there is actually a build to do. These preconditions used to be
    # checked in whatever order the module-level code ran in, so a user
    # missing three of them found out one `pip install` at a time, paying a
    # cold build between each.
    if not os.path.isfile(core_output_path):
        from jittor_utils import preflight as _preflight
        _preflight.assert_ready(config=build_config)

    try:
        provider_objects = compile_backend_sources(build_config, core_cc_flags+opt_flags+lto_flags)
        compile(cc_path, core_cc_flags+opt_flags+build_config.backend_link_flags,
                files + provider_objects, core_output_name)
    except RuntimeError as error:
        # A build that fails here usually fails for a reason the preconditions
        # above can name. Say all of them, rather than leaving the user with a
        # compiler diagnostic and nothing to act on.
        from jittor_utils import preflight as _preflight
        try:
            _report = _preflight.format_report(_preflight.run_all(config=build_config),
                                               only_problems=True)
        except Exception:
            raise error
        if _report and _report != "all build preconditions satisfied":
            raise RuntimeError("%s\n\nBuild preconditions on this machine:\n%s"
                               % (error, _report))
        raise

    # After the compile, so the stamp records the product that exists now. The
    # signature is the pre-build one on purpose: it is the state the sources
    # were in when this build was decided on, and a source edited *during* the
    # build must leave the stamp stale rather than claim to cover the edit.
    files += [source.path for source in build_config.backend_sources]
    _write_core_build_stamp(signature, ingredients, files)
    return True


if platform.system() == 'Linux':
    libname = {"clang":"omp", "icc":"iomp5", "g++":"gomp"}[cc_type]
    openmp_name = libname
    libname = ctypes.util.find_library(libname)
    if libname is None:
        raise RuntimeError(
            f"the OpenMP runtime lib{openmp_name} was not found, and every "
            f"CPU kernel Jittor compiles links against it. Install it "
            f"(libgomp for g++, libomp for clang) or set cc_path to a "
            f"compiler whose runtime is installed.")
    ctypes.CDLL(libname, os.RTLD_NOW | os.RTLD_GLOBAL)

# NOTE: sw_64 used to disable TLS certificate verification for the whole
# process here. See the note in jittor_utils/__init__.py: use SSL_CERT_FILE or
# certifi instead; nothing jittor does justifies turning it off for the
# application it is imported into.

build_core()
cc_flags += f" -l\"jittor_core{lib_suffix}\" "

with jit_utils.import_scope(import_flags):
    import jittor_core as core

flags = core.Flags()

if has_cuda and is_cuda and not hasattr(flags, "cuda_archs"):
    # The core just imported has no CUDA support while this process found nvcc
    # and is about to configure CUDA. The usual cause is a stale CPU-only
    # jittor_core one directory above the CUDA one -- any run that imports
    # Jittor without nvcc builds it there, and it then shadows the CUDA build
    # for every later run that shares the cache. Left unreported, the process
    # keeps going on CPU and every CUDA op fails with "Op ... doesn't have cuda
    # version"; this line would otherwise raise a bare AttributeError. Name the
    # file that actually got imported, which is the one to remove.
    raise RuntimeError(
        "jittor_core was imported from {}, which is built without CUDA "
        "support, but this process found nvcc at {} and expected the CUDA "
        "build in {}. Remove the file named first to stop it shadowing the "
        "CUDA build, or unset nvcc_path to keep running on CPU.".format(
            getattr(core, "__file__", "<unknown>"), nvcc_path, cache_path)
    )

if has_cuda and is_cuda:
    nvcc_flags = " " + build_env("nvcc_flags", "") + " "
    nvcc_flags += convert_nvcc_flags(cc_flags)
    nvcc_version = list(jit_utils.get_int_version(nvcc_path))
    supported_archs = query_nvcc_archs(nvcc_path)
    if supported_archs:
        max_arch = max(supported_archs)
    else:
        # nvcc too old for --list-gpu-arch (before CUDA 11.1), so the ceiling
        # has to come from its version. This table only has to stay correct
        # for toolkits that cannot answer for themselves, which is why it is
        # no longer the source of the answer for the ones that can.
        max_arch = 86
        if nvcc_version < [11,]:
            max_arch = 75
        elif nvcc_version < [11,1]:
            max_arch = 80
    if len(flags.cuda_archs):
        archs = select_cuda_archs(flags.cuda_archs, max_arch)
        flags.cuda_archs = archs
        nvcc_flags += cuda_arch_flags(archs)

flags.cc_path = cc_path
flags.cc_type = cc_type
flags.cc_flags = cc_flags + kernel_opt_flags
flags.nvcc_path = nvcc_path
flags.nvcc_flags = nvcc_flags
flags.python_path = python_path
flags.cache_path = cache_path
flags.jittor_path = jittor_path
flags.gdb_path = gdb_path
flags.addr2line_path = addr2line_path
flags.has_pybt = has_pybt

build_config = build_config.evolve(
    cc_flags=cc_flags, nvcc_flags=nvcc_flags, is_cuda=bool(is_cuda),
    kernel_flags=kernel_opt_flags,
)

# The accelerator JIT compiler is configured, not inferred from flags: the
# core's `configure_accelerator_compiler` is the only way it learns which
# compiler builds a device kernel. Without this call every accelerator JIT op
# fails with "Accelerator compiler is not configured", which is what an
# unconfigured accelerator looks like from the inside.
if build_config.has_accelerator and build_config.kernel_compiler:
    core.configure_accelerator_compiler(
        build_config.kernel_compiler,
        build_config.kernel_compile_flags + nvcc_flags
        if build_config.kernel_language == "cuda" else build_config.kernel_compile_flags,
        build_config.kernel_language,
        build_config.kernel_source_suffix,
        list(build_config.kernel_flag_filter),
        build_config.kernel_device_link,
    )
jit_utils.configure_module_build(_module_build_services(build_config))

# Hand the one lock descriptor over to C++. Both sides now take flock() on
# this single open file description; before this line the C++ side took a
# POSIX record lock on a descriptor of its own, which excluded neither the
# Python side nor anything else, and whose release was tied to whichever
# descriptor happened to be closed first.
lock.jittor_lock.bind_core(core)
