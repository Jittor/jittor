# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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

if platform.system() == 'Darwin':
    mp.set_start_method('fork')

class LogWarper:
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
    prev = sys.getdlopenflags()
    sys.setdlopenflags(flags)
    yield
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
    s = r.stdout.decode('utf8')
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
    if cc is None:
        try_import_jit_utils_core()
    cc.init_subprocess()

def run_cmds(cmds, cache_path, jittor_path, msg="run_cmds"):
    global pool_size, p
    bk = mp.current_process()._config.get('daemon')
    mp.current_process()._config['daemon'] = False
    if pool_size == 0:
        try:
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            mem_gib = mem_bytes/(1024.**3)
            pool_size = min(16,max(int(mem_gib // 3), 1))
            LOG.i(f"Total mem: {mem_gib:.2f}GB, using {pool_size} procs for compiling.")
        except ValueError:
            # On macOS, python with version lower than 3.9 do not support SC_PHYS_PAGES.
            # Use hard coded pool size instead.
            pool_size = 4
            LOG.i(f"using {pool_size} procs for compiling.")

        p = Pool(pool_size, initializer=pool_initializer)
        p.__enter__()
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

if os.environ.get("DISABLE_MULTIPROCESSING", '0') == '1':
    os.environ["use_parallel_op_compiler"] = '1'
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

def find_cache_path():
    from pathlib import Path
    path = str(Path.home())
    dirs = [".cache", "jittor", os.path.basename(cc_path)]
    if os.environ.get("debug")=="1":
        dirs[-1] += "_debug"
    cache_name = "default"
    try:
        if "cache_name" in os.environ:
            cache_name = os.environ["cache_name"]
        else:
            # try to get branch name from git
            r = sp.run(["git","branch"], cwd=os.path.dirname(__file__), stdout=sp.PIPE,
                   stderr=sp.PIPE)
            assert r.returncode == 0
            bs = r.stdout.decode().splitlines()
            for b in bs:
                if b.startswith("* "): break
            
            cache_name = b[2:]
        for c in " (){}": cache_name = cache_name.replace(c, "_")
    except:
        pass
    for name in cache_name.split("/"):
        dirs.insert(-1, name)
    os.environ["cache_name"] = cache_name
    LOG.v("cache_name: ", cache_name)
    for d in dirs:
        path = os.path.join(path, d)
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except:
                pass
        assert os.path.isdir(path)
    if path not in sys.path:
        sys.path.append(path)
    return path

def get_version(output):
    if output.endswith("mpicc"):
        version = run_cmd(output+" --showme:version")
    else:
        version = run_cmd(output+" --version")
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

def get_cc_type(cc_path):
    bname = os.path.basename(cc_path)
    if "clang" in bname: return "clang"
    if "icc" in bname or "icpc" in bname: return "icc"
    if "g++" in bname: return "g++"
    LOG.f(f"Unknown cc type: {bname}")


is_in_ipynb = in_ipynb()
cc = None
LOG = LogWarper()

cc_path = env_or_find('cc_path', 'g++', silent=True)
os.environ["cc_path"] = cc_path
cc_type = get_cc_type(cc_path)
cache_path = find_cache_path()


# Search python3.x-config
# Note:
#   This may be called via c++ console. In that case, sys.executable will
#   be a path to the executable file, rather than python. So, we cannot infer 
#   python-config path only from sys.executable.
#   To address this issue, we add predefined paths to search,
#       - Linux: /usr/bin/python3.x-config
#       - macOS (installed via homebrew): /usr/local/bin/python3.x-config
#   There may be issues under other cases, e.g., installed via conda.
py3_config_paths = [
    os.path.dirname(sys.executable) + f"/python3.{sys.version_info.minor}-config",
    sys.executable + "-config",
    f"/usr/bin/python3.{sys.version_info.minor}-config",
    f"/usr/local/bin/python3.{sys.version_info.minor}-config",
    f'/opt/homebrew/bin/python3.{sys.version_info.minor}-config',
    os.path.dirname(sys.executable) + "/python3-config",
]
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
