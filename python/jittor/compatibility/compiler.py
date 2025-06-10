import jittor as jt
import jittor_utils
import glob
import os
from jittor import pyjt_compiler
import sys
from jittor_utils import lock


jtorch_path = os.path.dirname(__file__)
cache_path = os.path.join(jt.compiler.cache_path, "jtorch")
# os.makedirs(cache_path, exist_ok=True)
os.makedirs(os.path.join(cache_path, "gen"), exist_ok=True)

with lock.lock_scope():
    pyjt_gen_src = pyjt_compiler.compile(cache_path, jtorch_path)

ext_args = 'c[cu]' if jt.has_cuda else 'cc'
files = glob.glob(jtorch_path+"/src/**/*."+ext_args, recursive=True)
files += pyjt_gen_src
cc_flags = " -I\""+os.path.join(jtorch_path, "src")+"\" "
if os.environ.get("use_data_o", "1") == "1":
    files += glob.glob(jtorch_path+"/src/**/*.o", recursive=True)
    files = [f for f in files if "__data__" not in f]


with lock.lock_scope():
    jt.compiler.compile(
        jt.compiler.cc_path,
        jt.compiler.cc_flags+jt.compiler.opt_flags+ cc_flags,
        files,
        "jtorch_core"+jt.compiler.extension_suffix,
        obj_dirname="jtorch_objs")

    
with jittor_utils.import_scope(jt.compiler.import_flags):
    import jtorch_core as core

jt.flags.th_mode = 1
