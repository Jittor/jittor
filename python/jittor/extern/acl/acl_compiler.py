# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob

has_acl = 0
cc_flags = ""
tikcc_path = env_or_try_find('tikcc_path', 'tikcc')
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL

def install():
    import jittor.compiler as compiler
    global has_acl, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home+"/**/*.cc", recursive=True))
    cc_flags += f" -DHAS_CUDA -DIS_ACL  -I/usr/local/Ascend/latest/x86_64-linux/include/ -I/usr/local/Ascend/latest/x86_64-linux/include/acl -L/usr/local/Ascend/latest/x86_64-linux/lib64  -I/usr/local/Ascend/runtime/include -I/usr/local/Ascend/driver/include -L/usr/local/Ascend/compiler/lib64 -L/usr/local/Ascend/runtime/lib64 -I{acl_compiler_home}  -ltikc_runtime -lascendcl  "
    ctypes.CDLL("libascendcl.so", dlopen_flags)
    jittor_utils.LOG.i("ACL detected")

    mod = jittor_utils.compile_module('''
#include "common.h"
namespace jittor {
// @pyjt(process)
string process_acl(const string& src, const string& name, const map<string,string>& kargs);
}''', compiler.cc_flags + " " + " ".join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source("acl", mod.process)

    has_acl = 1


def check():
    import jittor.compiler as compiler
    global has_acl, cc_flags
    if tikcc_path:
        try:
            install()
        except Exception as e:
            jittor_utils.LOG.w(f"load ACL failed, exception: {e}")
            has_acl = 0
    compiler.has_acl = has_acl
    compiler.tikcc_path = tikcc_path
    if not has_acl: return False
    compiler.cc_flags += cc_flags
    compiler.nvcc_path = tikcc_path
    compiler.nvcc_flags = compiler.cc_flags.replace("-std=c++14","")
    return True

