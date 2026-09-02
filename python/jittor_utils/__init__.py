# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from multiprocessing import Pool
import multiprocessing as mp
import subprocess as sp
import os
import re
import sys
import inspect
import datetime
import contextlib
import platform
import threading
import time
from ctypes import cdll
import shutil
import urllib.request
import ctypes

if platform.system() == 'Darwin':
    mp.set_start_method('fork')

from pathlib import Path
import json

from . import probe


def _user_config_file():
    src_path = os.path.join(str(Path.home()), ".cache", "jittor")
    os.makedirs(src_path, exist_ok=True)
    return os.path.join(src_path, "config.json")


def _read_user_config():
    """Read the persistent user configuration, tolerating a damaged file.

    Several Jittor processes can start at once, so this file may be observed
    while it is being replaced. A configuration that cannot be parsed is not a
    reason to refuse to start: the defaults below are always usable.
    """
    path = _user_config_file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_user_config(data):
    """Replace the configuration atomically so readers never see a partial file."""
    path = _user_config_file()
    temporary = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(temporary, "w") as f:
            json.dump(data, f)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def set_home(path):
    """Persist ``path`` as this user's default Jittor home directory."""
    global _jittor_home
    resolved = os.path.abspath(os.path.expanduser(path))
    os.makedirs(resolved, exist_ok=True)
    data = _read_user_config()
    data["JITTOR_HOME"] = resolved
    _write_user_config(data)
    _jittor_home = resolved
    return resolved


_jittor_home = None
def home():
    global _jittor_home
    if _jittor_home is not None:
        return _jittor_home

    default_path = _read_user_config().get("JITTOR_HOME", str(Path.home()))

    # A ``JITTOR_HOME`` in the environment is a per-process override -- test
    # runs, CI jobs and multi-device jobs rely on it to keep caches apart. It is
    # deliberately not written back to the shared configuration: doing so made
    # one isolated run silently become every later run's default. Use
    # ``set_home`` to change the persistent default on purpose.
    _home_path = os.environ.get("JITTOR_HOME", default_path)

    if not os.path.exists(_home_path):
        os.makedirs(_home_path, exist_ok=True)
    _home_path = os.path.abspath(_home_path)

    _jittor_home = _home_path
    return _home_path


def _available_cpu_ids():
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return sorted(get_affinity(0))
        except OSError:
            pass
    logical = os.cpu_count()
    return list(range(logical)) if logical else []


def _physical_core_count_from_sysfs(cpu_ids,
                                    root="/sys/devices/system/cpu"):
    """Count the distinct SMT sibling groups visible to this process."""
    sibling_groups = set()
    for cpu_id in cpu_ids:
        path = os.path.join(root, "cpu%d" % cpu_id, "topology",
                            "thread_siblings_list")
        try:
            with open(path) as handle:
                siblings = handle.read().strip()
        except OSError:
            return None
        if not siblings:
            return None
        sibling_groups.add(siblings)
    return len(sibling_groups) or None


def _physical_core_count_from_cpuinfo(cpu_ids, path="/proc/cpuinfo"):
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError:
        return None
    allowed = set(cpu_ids) if cpu_ids else None
    cores = set()
    for block in text.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
        try:
            cpu_id = int(fields["processor"])
            package = fields["physical id"]
            core = fields["core id"]
        except (KeyError, ValueError):
            continue
        if allowed is None or cpu_id in allowed:
            cores.add((package, core))
    return len(cores) or None


def physical_core_count():
    """Cores that can retire instructions in parallel, not SMT siblings.

    Returns ``None`` when the topology is not readable, which is the signal to
    leave the OpenMP default alone rather than guess.
    """
    cpu_ids = _available_cpu_ids()
    physical = _physical_core_count_from_sysfs(cpu_ids)
    if physical is not None:
        return physical
    return _physical_core_count_from_cpuinfo(cpu_ids)


def limit_openmp_to_physical_cores(environ):
    """Default OpenMP to one thread per physical core.

    OpenMP's own default is one thread per *logical* CPU. On an SMT machine
    that oversubscribes every core, and the cost is not a gentle slowdown: on a
    dual 32-core host, one batched oneDNN call took 437us with 64 threads and
    5955us with 128, because the barrier across twice as many threads dominates
    everything else. PyTorch defaults to the physical count for the same
    reason.

    Returns the value set, or ``None`` when the environment already chose one
    or the topology could not be read. An explicit ``OMP_NUM_THREADS`` always
    wins -- this only fills in a default.
    """
    if environ.get("OMP_NUM_THREADS", "").strip():
        return None
    physical = physical_core_count()
    if not physical:
        return None
    logical = os.cpu_count() or physical
    if physical >= logical:
        return None
    environ["OMP_NUM_THREADS"] = str(physical)
    return physical


class Logwrapper:
    def __init__(self):
        self.log_silent = int(os.environ.get("log_silent", "0"))
        self.log_v = int(os.environ.get("log_v", "0"))

    def log_capture_start(self):
        cc.log_capture_start()

    def log_capture_stop(self):
        cc.log_capture_stop()

    def log_capture_read(self):
        return cc.log_capture_read()

    def _log(self, level, verbose, *msg):
        if self.log_silent or verbose > self.log_v:
            return
        ss = ""
        for m in msg:
            if callable(m):
                m = m()
            ss += str(m)
        msg = ss
        f = inspect.currentframe()
        fileline = inspect.getframeinfo(f.f_back.f_back)
        fileline = f"{os.path.basename(fileline.filename)}:{fileline.lineno}"
        if cc and hasattr(cc, "log"):
            cc.log(fileline, level, verbose, msg)
        else:
            time = datetime.datetime.now().strftime("%m%d %H:%M:%S.%f")
            tid = threading.get_ident()%100
            v = f" v{verbose}" if verbose else ""
            print(f"[{level} {time} {tid:02}{v} {fileline}] {msg}")
    
    def V(self, verbose, *msg): self._log('i', verbose, *msg)
    def v(self, *msg): self._log('i', 1, *msg)
    def vv(self, *msg): self._log('i', 10, *msg)
    def vvv(self, *msg): self._log('i', 100, *msg)
    def vvvv(self, *msg): self._log('i', 1000, *msg)
    def i(self, *msg): self._log('i', 0, *msg)
    def w(self, *msg): self._log('w', 0, *msg)
    def e(self, *msg): self._log('e', 0, *msg)
    def f(self, *msg): self._log('f', 0, *msg)

class DelayProgress:
    def __init__(self, msg, n):
        self.msg = msg
        self.n = n
        self.time = time.time()

    def update(self, i):
        if LOG.log_silent:
            return
        used = time.time() - self.time
        if used > 2:
            eta = used / (i+1) * (self.n-i-1)
            print(f"{self.msg}({i+1}/{self.n}) used: {used:.3f}s eta: {eta:.3f}s", end='\r')
            if i==self.n-1: print()

# check is in jupyter notebook
def in_ipynb():
    try:
        cfg = get_ipython().config 
        if 'IPKernelApp' in cfg:
            return True
        else:
            return False
    except:
        return False

@contextlib.contextmanager
def simple_timer(name):
    print("Timer start", name)
    now = time.time()
    yield
    print("Time stop", name, time.time()-now)

@contextlib.contextmanager
def import_scope(flags):
    if os.name != 'nt':
        prev = sys.getdlopenflags()
        sys.setdlopenflags(flags)
    yield
    if os.name != 'nt':
        sys.setdlopenflags(prev)

def try_import_jit_utils_core(silent=None):
    global cc
    if cc: return
    if not (silent is None):
        prev = os.environ.get("log_silent", "0")
        os.environ["log_silent"] = str(int(silent))
    try:
        # if is in notebook, must log sync, and we redirect the log
        if is_in_ipynb: os.environ["log_sync"] = "1"
        import jit_utils_core as cc
        if is_in_ipynb:
            if os.name != 'nt':
                # windows jupyter has import error
                # disable ostream redirect
                # TODO: find a better way
                cc.ostream_redirect(True, True)
    except Exception as _:
        if int(os.environ.get("log_v", "0")) > 0:
            print(_)
        pass
    if not (silent is None):
        os.environ["log_silent"] = prev

def run_cmd(cmd, cwd=None, err_msg=None, print_error=True):
    LOG.v(f"Run cmd: {cmd}")
    if cwd:
        r = sp.run(cmd, cwd=cwd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    else:
        r = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    try:
        s = r.stdout.decode('utf8')
    except:
        s = r.stdout.decode('gbk')
    if r.returncode != 0:
        if print_error:
            sys.stderr.write(s)
        if err_msg is None:
            err_msg = f"Run cmd failed: {cmd}"
        if not print_error:
            err_msg += "\n"+s
        raise Exception(err_msg)
    if len(s) and s[-1] == '\n': s = s[:-1]
    return s


def do_compile(args):
    cmd, cache_path, jittor_path = args
    try_import_jit_utils_core(True)
    if cc:
        return cc.cache_compile(cmd, cache_path, jittor_path)
    else:
        run_cmd(cmd)
        return True

pool_size = 0

def pool_cleanup():
    global p
    p.__exit__(None, None, None)
    del p

def pool_initializer():
    if os.name == 'nt':
        os.environ['log_silent'] = '1'
        os.environ['gdb_path'] = ""
    if cc is None:
        try_import_jit_utils_core()
    if cc:
        cc.init_subprocess()

def run_cmds(cmds, cache_path, jittor_path, msg="run_cmds"):
    global pool_size, p
    # Under MPI (mpirun), the OpenMPI runtime installs atfork handlers and a
    # registration cache that make fork() after MPI_Init unsafe -- forking the
    # compile Pool there triggers SIGBUS. Compile serially in-process instead.
    # (Compilation under MPI should be rare anyway: warm the cache first.)
    under_mpi = ("OMPI_COMM_WORLD_SIZE" in os.environ) or ("PMI_SIZE" in os.environ)
    if under_mpi:
        n = len(cmds)
        dp = DelayProgress(msg, n)
        for i, cmd in enumerate(cmds):
            do_compile([cmd, cache_path, jittor_path])
            dp.update(i)
        return
    bk = mp.current_process()._config.get('daemon')
    mp.current_process()._config['daemon'] = False
    if pool_size == 0:
        try:
            mem_bytes = get_total_mem()
            mem_gib = mem_bytes/(1024.**3)
            pool_size = min(16,max(int(mem_gib // 3), 1))
            LOG.i(f"Total mem: {mem_gib:.2f}GB, using {pool_size} procs for compiling.")
        except ValueError:
            # On macOS, python with version lower than 3.9 do not support SC_PHYS_PAGES.
            # Use hard coded pool size instead.
            pool_size = 4
            LOG.i(f"using {pool_size} procs for compiling.")
        if os.name == 'nt':
            # a hack way to by pass windows
            # multiprocess spawn init_main_from_path.
            # check spawn.py:get_preparation_data
            spec_bk = sys.modules['__main__'].__spec__
            tmp = lambda x:x
            tmp.name = '__main__'
            sys.modules['__main__'].__spec__ = tmp
        p = Pool(pool_size, initializer=pool_initializer)
        p.__enter__()
        if os.name == 'nt':
            sys.modules['__main__'].__spec__ = spec_bk
        import atexit
        atexit.register(pool_cleanup)
    cmds = [ [cmd, cache_path, jittor_path] for cmd in cmds ]
    try:
        n = len(cmds)
        dp = DelayProgress(msg, n)
        for i,_ in enumerate(p.imap_unordered(do_compile, cmds)):
            dp.update(i)
    finally:
        mp.current_process()._config['daemon'] = bk

if os.name=='nt' and getattr(mp.current_process(), '_inheriting', False):
    # when windows spawn multiprocess, disable sub-subprocess
    os.environ["DISABLE_MULTIPROCESSING"] = '1'
    os.environ["log_silent"] = '1'
        
if os.environ.get("DISABLE_MULTIPROCESSING", '0') == '1':
    os.environ["use_parallel_op_compiler"] = '0'
    def run_cmds(cmds, cache_path, jittor_path, msg="run_cmds"):
        cmds = [ [cmd, cache_path, jittor_path] for cmd in cmds ]
        n = len(cmds)
        dp = DelayProgress(msg, n)
        for i,cmd in enumerate(cmds):
            dp.update(i)
            do_compile(cmd)


def download(url, filename):
    if os.path.isfile(filename):
        if os.path.getsize(filename) > 100:
            return
    LOG.v("Downloading", url)
    urllib.request.urlretrieve(url, filename)
    LOG.v("Download finished")

def get_jittor_version():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "../jittor/__init__.py"), "r", encoding='utf8') as fh:
        for line in fh:
            if line.startswith('__version__'):
                version = line.split("'")[1]
                break
        else:
            raise RuntimeError("Unable to find version string.")
    return version

def get_str_hash(s):
    import hashlib
    md5 = hashlib.md5()
    md5.update(s.encode())
    return md5.hexdigest()

def get_cpu_version():
    v = platform.processor()
    try:
        if os.name == 'nt':
            import winreg
            key_name = r"Hardware\Description\System\CentralProcessor\0"
            field_name = "ProcessorNameString"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name)
            value = winreg.QueryValueEx(key, field_name)[0]
            winreg.CloseKey(key)
            v = value
        elif platform.system() == "Darwin":
            r, s = sp.getstatusoutput("sysctl -a sysctl machdep.cpu.brand_string")
            if r==0:
                v = s.split(":")[-1].strip()
        else:
            with open("/proc/cpuinfo", 'r') as f:
                for l in f:
                    if l.startswith("model name"):
                        v = l.split(':')[-1].strip()
                        break
    except:
        pass
    return v
    
def short(s):
    ss = ""
    for c in s:
        if str.isidentifier(c) or str.isnumeric(c) \
            or str.isalpha(c) or c in '.-+':
            ss += c
    if len(ss)>14:
        return ss[:14]+'x'+get_str_hash(ss)[:2]
    return ss

# Environment variables that change what the compiler is asked to produce
# without changing any component of the cache directory below. Two processes
# that disagree on any of these produce *different* object code, and used to
# write it into the same directory: the torch shim, for example, appends
# ``--fmad=false --prec-div=true --prec-sqrt=true`` to ``nvcc_flags`` and drops
# ``--use_fast_math``, so turning it on or off recompiled every CUDA kernel
# into the same ``jit/`` directory -- and when the two ran at once, the second
# writer replaced a shared library the first had already dlopen'd.
BUILD_CONFIG_VARS = (
    "cc_flags",
    "nvcc_flags",
    "kernel_flags",
    "cuda_archs",
    "enable_lto",
)


def get_build_config():
    """The build knobs whose values decide what the compiled products are."""
    return {name: os.environ.get(name, "") for name in BUILD_CONFIG_VARS}


def build_config_fingerprint(config=None):
    """A short, stable directory name for one build configuration."""
    import hashlib
    if config is None:
        config = get_build_config()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf8")).hexdigest()
    return "cfg" + digest[:8]


def _write_build_config(path, config):
    """Leave the knobs next to the products, so ``cfgXXXXXXXX`` can be read back.

    Written once and never rewritten: every process that lands in this
    directory computed the same fingerprint from the same values, so the file
    is either absent or already correct.
    """
    record = os.path.join(path, "build_config.json")
    if os.path.exists(record):
        return
    temporary = "%s.%d.tmp" % (record, os.getpid())
    try:
        with open(temporary, "w") as f:
            json.dump(config, f, indent=1, sort_keys=True)
        os.replace(temporary, record)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


# Set by find_cache_path(). Deliberately *above* the build-configuration
# directory: the lock also guards the third-party downloads (mkl, cutt, cub)
# that every configuration on this toolchain shares.
lock_path = None


def _git_head_file(path):
    """The file whose change can change the branch name, or None.

    In a linked worktree ``.git`` is a *file* naming the real git directory,
    and HEAD lives there, not next to the sources.
    """
    path = os.path.abspath(path)
    while True:
        candidate = os.path.join(path, ".git")
        if os.path.isdir(candidate):
            return os.path.join(candidate, "HEAD")
        if os.path.isfile(candidate):
            try:
                with open(candidate) as f:
                    line = f.readline().strip()
            except OSError:
                return None
            if not line.startswith("gitdir:"):
                return None
            gitdir = line[len("gitdir:"):].strip()
            if not os.path.isabs(gitdir):
                gitdir = os.path.join(path, gitdir)
            return os.path.join(gitdir, "HEAD")
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _read_git_branch(cwd):
    r = sp.run(["git", "branch"], cwd=cwd, stdout=sp.PIPE, stderr=sp.PIPE)
    assert r.returncode == 0
    bs = r.stdout.decode().splitlines()
    for b in bs:
        if b.startswith("* "): break

    return b[2:]


def get_git_branch(cwd):
    """Branch of the checkout holding ``cwd``, remembered against its HEAD.

    Outside a checkout there is nothing to invalidate against, so the answer is
    computed every time rather than cached wrongly.
    """
    head = _git_head_file(cwd)
    if head is None:
        return _read_git_branch(cwd)
    return probe.cached("git_branch:" + os.path.abspath(cwd), [head],
                        lambda: _read_git_branch(cwd))


def find_cache_path():
    global lock_path
    path = home()
    # jittor version key
    jtv = "jt"+get_jittor_version().rsplit('.', 1)[0]
    # cc version key
    ccv = cc_type+get_version(cc_path)[1:-1] \
        if cc_type != "cl" else cc_type
    # os version key
    osv = platform.platform() + platform.node()
    if len(osv)>14:
        osv = osv[:14] + 'x'+get_str_hash(osv)[:2]
    # py version
    pyv = "py"+platform.python_version()
    # cpu version
    cpuv = get_cpu_version()
    jittor_path_key = get_str_hash(__file__)[:4]
    dirs = [".cache", "jittor", jtv, ccv, pyv, osv, cpuv, jittor_path_key]
    dirs = list(map(short, dirs))
    cache_name = "default"
    try:
        if "cache_name" in os.environ:
            cache_name = os.environ["cache_name"]
        else:
            cache_name = get_git_branch(os.path.dirname(__file__))
        for c in " (){}": cache_name = cache_name.replace(c, "_")
    except:
        pass
    if os.environ.get("debug")=="1":
        dirs[-1] += "_debug"
    for name in os.path.normpath(cache_name).split(os.path.sep):
        dirs.append(name)
    os.environ["cache_name"] = cache_name
    LOG.v("cache_name: ", cache_name)
    path = os.path.join(path, *dirs)
    lock_path = os.path.abspath(os.path.join(path, os.pardir, "jittor.lock"))
    config = get_build_config()
    path = os.path.join(path, build_config_fingerprint(config))
    os.makedirs(path, exist_ok=True)
    _write_build_config(path, config)
    if path not in sys.path:
        sys.path.append(path)
    return path

def resolve_exe(name):
    """Absolute path of the tool a probe is about, so it can be stamped."""
    if os.path.sep in name or (os.altsep and os.altsep in name):
        return name
    return shutil.which(name) or name


def get_version(output):
    """``--version`` of a tool, remembered until that tool's file changes.

    Six of these were ``nvcc --version``, one per CUDA library, every import.
    """
    tool = resolve_exe(output)
    return probe.cached("version:" + tool, [tool],
                        lambda: _read_version(output))


def _read_version(output):
    if output.endswith("mpicc"):
        version = run_cmd(f"\"{output}\" --showme:version")
    elif os.name == 'nt' and (
        output.endswith("cl") or output.endswith("cl.exe")):
        version = run_cmd(output)
    else:
        version = run_cmd(f"\"{output}\" --version")
    v = re.findall("[0-9]+\\.[0-9]+\\.[0-9]+", version)
    if len(v) == 0:
        v = re.findall("[0-9]+\\.[0-9]+", version)
    assert len(v) != 0, f"Can not find version number from: {version}"
    if 'clang' in version and platform.system() == 'Darwin':
        version = "("+v[-3]+")"
    else:
        version = "("+v[-1]+")"
    return version

def get_int_version(output):
    ver = get_version(output)
    ver = ver[1:-1].split('.')
    ver = tuple(( int(v) for v in ver ))
    return ver

def find_exe(name, check_version=True, silent=False):
    output = shutil.which(name)
    if not output:
        raise RuntimeError(f"{name} not found")
    if check_version:
        version = get_version(name)
    else:
        version = ""
    if not silent:
        LOG.i(f"Found {name}{version} at {output}.")
    return output

def env_or_find(name, bname, silent=False):
    if name in os.environ:
        path = os.environ[name]
        if path != "":
            version = get_version(path)
            if not silent:
                LOG.i(f"Found {bname}{version} at {path}")
        return path
    return find_exe(bname, silent=silent)

def env_or_try_find(name, bname):
    if name in os.environ:
        path = os.environ[name]
        if path != "":
            version = get_version(path)
            LOG.i(f"Found {bname}{version} at {path}")
        return path
    return try_find_exe(bname)

def try_find_exe(*args):
    try:
        return find_exe(*args)
    except:
        LOG.v(f"{args[0]} not found.")
        return ""

def get_cc_type(cc_path):
    bname = os.path.basename(cc_path)
    if "clang" in bname: return "clang"
    if "icc" in bname or "icpc" in bname: return "icc"
    if "g++" in bname: return "g++"
    if "cl" in bname: return "cl"
    LOG.f(f"Unknown cc type: {bname}")

def get_py3_link_path():
    py3_link_path = os.path.join(
            os.path.dirname(sys.executable),
            "libs",
    )
    if not os.path.exists(py3_link_path):
        candidate = [os.path.dirname(sys.executable)] + sys.path
        for p in candidate:
            p = os.path.join(p, "libs")
            if os.path.exists(p):
                py3_link_path = p
                break
    return py3_link_path

def get_py3_config_path():
    global _py3_config_path
    if _py3_config_path: 
        return _py3_config_path

    if os.name == 'nt':
        return None
    else:
        # Search python3.x-config
        # Note:
        #   This may be called via c++ console. In that case, sys.executable will
        #   be a path to the executable file, rather than python. So, we cannot infer 
        #   python-config path only from sys.executable.
        #   To address this issue, we add predefined paths to search,
        #       - Linux: /usr/bin/python3.x-config
        #       - macOS:
        #           - shiped with macOS 13: /Library/Developer/CommandLineTools/Library/Frameworks/
        #                                   Python3.framework/Versions/3.x/lib/python3.x/config-3.x-darwin/python-config.py
        #           - installed via homebrew: /usr/local/bin/python3.x-config
        #   There may be issues under other cases, e.g., installed via conda.
        py3_config_paths = [
            os.path.dirname(sys.executable) + f"/python3.{sys.version_info.minor}-config",
            sys.executable + "-config",
            f"/usr/bin/python3.{sys.version_info.minor}-config",
            f"/usr/local/bin/python3.{sys.version_info.minor}-config",
            os.path.dirname(sys.executable) + "/python3-config",
        ]
        if platform.system() == "Darwin":
            if "homebrew" in sys.executable:
                py3_config_paths.append(f'/opt/homebrew/bin/python3.{sys.version_info.minor}-config')
            else:
                py3_config_paths.append(f'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/'\
                                        f'Versions/3.{sys.version_info.minor}/lib/python3.{sys.version_info.minor}/'\
                                        f'config-3.{sys.version_info.minor}-darwin/python-config.py')

        if "python_config_path" in os.environ:
            py3_config_paths.insert(0, os.environ["python_config_path"])

        for py3_config_path in py3_config_paths:
            if os.path.isfile(py3_config_path):
                break
        else:
            raise RuntimeError(f"python3.{sys.version_info.minor}-config "
                f"not found in {py3_config_paths}, please specify "
                f"enviroment variable 'python_config_path',"
                f" or install python3.{sys.version_info.minor}-dev")
        _py3_config_path = py3_config_path
        return py3_config_path

def get_py3_include_path():
    global _py3_include_path
    if _py3_include_path: 
        return _py3_include_path
    
    if os.name == 'nt':
        # Windows
        sys.executable = sys.executable.lower()
        candidate = [os.path.dirname(sys.executable)] + sys.path
        for p in candidate:
            include_path = os.path.join(p, "include")
            if os.path.exists(include_path):
                break
        else:
            raise RuntimeError("Python include path not found. please report this bug to us.")
        _py3_include_path = '-I"' + include_path + '"'
    else:
        config_path = get_py3_config_path()
        _py3_include_path = probe.cached(
            "py3_includes:" + config_path, [config_path, sys.executable],
            lambda: run_cmd(config_path+" --includes"))
        
        # macOS (>=13) is shiped with a fake python3-config which outputs wrong include paths
        # check the include paths and fix them
        if platform.system() == "Darwin":
            is_real_path = False
            for include_path in _py3_include_path.strip().split():
                if os.path.exists(include_path[2:]):
                    is_real_path = True
            if not is_real_path:
                _py3_include_path = f"-I/Library/Developer/CommandLineTools/Library/Frameworks/"\
                                    f"Python3.framework/Versions/3.{sys.version_info.minor}/Headers"
    return _py3_include_path


def get_py3_extension_suffix():
    global _py3_extension_suffix
    if _py3_extension_suffix: 
        return _py3_extension_suffix
    
    if os.name == 'nt':
        # Windows
        _py3_extension_suffix = f".cp3{sys.version_info.minor}-win_amd64.pyd"
    else:
        config_path = get_py3_config_path()
        _py3_extension_suffix = probe.cached(
            "py3_extension_suffix:" + config_path,
            [config_path, sys.executable],
            lambda: run_cmd(config_path+" --extension-suffix"))
    return _py3_extension_suffix

def get_total_mem():
    if os.name == 'nt':
        from ctypes import Structure, c_int32, c_uint64, sizeof, byref, windll
        class MemoryStatusEx(Structure):
            _fields_ = [
                ('length', c_int32),
                ('memoryLoad', c_int32),
                ('totalPhys', c_uint64),
                ('availPhys', c_uint64),
                ('totalPageFile', c_uint64),
                ('availPageFile', c_uint64),
                ('totalVirtual', c_uint64),
                ('availVirtual', c_uint64),
                ('availExtendedVirtual', c_uint64)]
            def __init__(self):
                self.length = sizeof(self)
        m = MemoryStatusEx()
        assert windll.kernel32.GlobalMemoryStatusEx(byref(m))
        return m.totalPhys
    else:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

def dirty_fix_pytorch_runtime_error():
    ''' This funtion should be called before pytorch.
    
    Example::

        import jittor as jt
        jt.dirty_fix_pytorch_runtime_error()
        import torch
    '''
    import os, platform

    if platform.system() == 'Linux':
        os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND
        import jittor_utils
        with jittor_utils.import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
            import torch

is_in_ipynb = in_ipynb()
cc = None
LOG = Logwrapper()

check_msvc_install = False
msvc_path = ""
if os.name == 'nt' and os.environ.get("cc_path", "")=="":
    msvc_path = os.path.join(home(), ".cache", "jittor", "msvc")
    cc_path = os.path.join(msvc_path, "VC", r"_\_\_\_\_\bin", "cl.exe")
    check_msvc_install = True
elif platform.system() == "Darwin":
    # macOS has a fake "g++" which is actually clang++, so we search clang.
    cc_path = env_or_find('cc_path', 'clang++', silent=True)
else:
    cc_path = env_or_find('cc_path', 'g++', silent=True)
os.environ["cc_path"] = cc_path
cc_type = get_cc_type(cc_path)
cache_path = find_cache_path()

_py3_config_path = None
_py3_include_path = None
_py3_extension_suffix = None
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

try:
    import sys
    sys.setrecursionlimit(10**6)
    if os.name != 'nt':
        import resource
        resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
except:
    pass

if os.name == 'nt':
    if check_msvc_install:
        if not os.path.isfile(cc_path):
            from jittor_utils import install_msvc
            install_msvc.install(msvc_path)
    mpath = os.path.join(home(), ".cache", "jittor", "msvc")
    if cc_path.startswith(mpath):
        msvc_path = mpath
    os.RTLD_NOW = os.RTLD_GLOBAL = os.RTLD_DEEPBIND = 0
    path = os.path.dirname(cc_path).replace('/', '\\')
    if path:
        sys.path.insert(0, path)
        os.environ["PATH"] = path+';'+os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)

backends = []
def add_backend(mod):
    backends.append(mod)

from . import lock
@lock.lock_scope()
def compile_module(source, flags):
    """
    quick c extension:
    Example:

        import jittor as jt

        import jittor_utils
        import jittor.compiler as compiler


        mod = jittor_utils.compile_module('''
        #include "common.h"
        namespace jittor {
        // @pyjt(hello)
        string hello(const string& src) {
            LOGir << "hello" << src;
        }
        }''', compiler.cc_flags)

        mod.hello("aaa")

    """
    tmp_path = os.path.join(cache_path, "tmp")
    os.makedirs(tmp_path, exist_ok=True)
    hash = "hash_" + get_str_hash(source)
    so = get_py3_extension_suffix()
    header_name = os.path.join(tmp_path, hash+".h")
    source_name = os.path.join(tmp_path, hash+".cc")
    lib_name = hash+so
    with open(header_name, "w", encoding="utf8") as f:
        f.write(source)
    from jittor.pyjt_compiler import compile_single
    ok = compile_single(header_name, source_name)
    assert ok, "no pyjt interface found"
    
    entry_src = f'''
static void init_module(PyModuleDef* mdef, PyObject* m) {{
    mdef->m_doc = "generated py jittor_utils.compile_module";
    jittor::pyjt_def_{hash}(m);
}}
PYJT_MODULE_INIT({hash});
    '''
    with open(source_name, "r", encoding="utf8") as f:
        src = f.read()
    with open(source_name, "w", encoding="utf8") as f:
        f.write(src + entry_src)
    jittor_path = os.path.join(os.path.dirname(__file__), "..", "jittor")
    jittor_path = os.path.abspath(jittor_path)
    from jittor.compiler import fix_cl_flags
    do_compile([fix_cl_flags(f"\"{cc_path}\" \"{source_name}\" \"{jittor_path}/src/pyjt/py_arg_printer.cc\" {flags} -o \"{cache_path+'/'+lib_name}\" "),
        cache_path, jittor_path])
    # use __import__ (returns the module object) rather than
    # `exec("import X"); locals()["X"]`: since Python 3.13 (PEP 667) exec() no
    # longer leaks names into an optimized function's locals(), so the old
    # pattern raised KeyError on 3.13.
    with lock.unlock_scope():
        try:
            with import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
                mod = __import__(hash)
        except Exception as e:
            with import_scope(os.RTLD_GLOBAL | os.RTLD_LAZY):
                mod = __import__(hash)

    return mod

def process_jittor_source(device_type, callback):
    import jittor.compiler as compiler
    import shutil
    djittor = device_type + "_jittor"
    djittor_path = os.path.join(compiler.cache_path, djittor)
    os.makedirs(djittor_path, exist_ok=True)

    for root, dir, files in os.walk(compiler.jittor_path):
        root2 = root.replace(compiler.jittor_path, djittor_path)
        os.makedirs(root2, exist_ok=True)
        for name in files:
            fname = os.path.join(root, name)
            fname2 = os.path.join(root2, name)
            if fname.endswith(".h") or fname.endswith(".cc") or fname.endswith(".cu"):
                with open(fname, 'r', encoding="utf8") as f:
                    src = f.read()
                src = callback(src, name, {"fname":fname, "fname2":fname2})
                with open(fname2, 'w', encoding="utf8") as f:
                    f.write(src)
            else:
                shutil.copy(fname, fname2)
    compiler.cc_flags = compiler.cc_flags.replace(compiler.jittor_path, djittor_path) + f" -I\"{djittor_path}/extern/cuda/inc\" "
    compiler.jittor_path = djittor_path

import time
class time_scope:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, *exc):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"exec[{self.name}] time: {self.execution_time}s")
    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
            return ret
        return inner
