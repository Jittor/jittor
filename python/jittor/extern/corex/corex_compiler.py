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
import jittor.compiler as compiler

has_corex = 0
cc_flags = ""
compiler.has_corex = has_corex

def install():
    import jittor.compiler as compiler
    global has_corex, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home+"/**/*.cc", recursive=True))
    jittor_utils.LOG.i("COREX detected")

    mod = jittor_utils.compile_module('''
#include "common.h"
#include "utils/str_utils.h"

namespace jittor {
// @pyjt(process)
string process_acl(const string& src, const string& name, const map<string,string>& kargs) {
    auto new_src = src;
    new_src = replace(new_src, "helper_cuda.h", "../inc/helper_cuda.h");
    if (name == "string_view_map.h")
        new_src = replace(new_src, "using std::string_view;", "using string_view = string;");
    if (name == "nan_checker.cu")
        new_src = replace(new_src, "__trap()", "assert(0)");
    if (name == "jit_compiler.cc") {
        // remove asm tuner
        new_src = token_replace_all(new_src, "cmd = python_path$1;", "");
        new_src = token_replace_all(new_src, "JPU(op_compiler($1));", 
        R"(JPU(op_compiler($1));
            *extra_flags2 = replace(*extra_flags2, "--extended-lambda", "");
            *extra_flags2 = replace(*extra_flags2, "--expt-relaxed-constexpr", "");
        )");
        new_src = token_replace_all(new_src, 
            "if (is_cuda_op && $1 != string::npos)",
            "if (is_cuda_op)");
    }
    if (name == "where_op.cc") {
        // default where kernel cannot handle 64 warp size, use cub_where instead
        new_src = token_replace_all(new_src, "if (cub_where$1) {", "if (cub_where) {");
    }
    if (name == "loop_var_analyze_pass.cc") {
        new_src = token_replace_all(new_src, "DEFINE_FLAG($1, para_opt_level,$2,$3);", 
                                             "DEFINE_FLAG($1, para_opt_level, 4,$3);");
    }
    return new_src;
}
}''', compiler.cc_flags + " " + " ".join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source("corex", mod.process)
    # def nvcc_flags_to_corex(nvcc_flags):
    #     nvcc_flags = nvcc_flags.replace("--cudart=shared", "")
    #     nvcc_flags = nvcc_flags.replace("--cudart=shared", "")

    has_corex = 1
    compiler.has_corex = has_corex
    corex_home = "/usr/local/corex"
    compiler.nvcc_path = corex_home + "/bin/clang++"
    compiler.cc_path = compiler.nvcc_path
    compiler.cc_flags = compiler.cc_flags.replace("-fopenmp", "")
    # compiler.nvcc_flags = cc_flags_to_corex(compiler.cc_flags)
    compiler.nvcc_flags = compiler.cc_flags + " -x cu -Ofast -DNO_ATOMIC64 -Wno-c++11-narrowing "
    compiler.convert_nvcc_flags = lambda x:x
    compiler.is_cuda = 0
    os.environ["use_cutt"] = "0"
    compiler.cc_type = "clang"


def install_extern():
    return False


def check():
    global has_corex, cc_flags
    if os.path.isdir("/usr/local/corex"):
        try:
            install()
        except Exception as e:
            jittor_utils.LOG.w(f"load COREX failed, exception: {e}")
            has_corex = 0
    if not has_corex: return False
    return True

def post_process():
    if not has_corex: return
    import jittor.compiler as compiler
    compiler.flags.cc_flags = compiler.flags.cc_flags.replace("-fopenmp", "")