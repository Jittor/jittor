# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
from collections import namedtuple
import jittor_utils
from jittor_utils.compiler_flags import remove_flags
import glob


CorexDiscovery = namedtuple(
    "CorexDiscovery", "home compiler_path available reason")


def discover(corex_home=None):
    """Inspect a Corex installation without importing or changing Jittor state."""
    home = corex_home or os.environ.get("COREX_HOME") or "/usr/local/corex"
    home = os.path.abspath(os.path.expanduser(home))
    compiler_path = os.path.join(home, "bin", "clang++")
    if not os.path.isdir(home):
        return CorexDiscovery(home, compiler_path, False, "COREX_HOME is absent")
    if not os.path.isfile(compiler_path):
        return CorexDiscovery(
            home, compiler_path, False, "Corex compiler is missing: %s" % compiler_path)
    return CorexDiscovery(home, compiler_path, True, "ready")

def configure(context, corex_home=None):
    discovery = discover(corex_home)
    if not discovery.available:
        raise RuntimeError(discovery.reason)
    corex_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(corex_compiler_home+"/**/*.cc", recursive=True))
    jittor_utils.LOG.i("COREX detected")

    mod = context.compile_module('''
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
}''', context.config.cc_flags + " " + " ".join(cc_files))
    config = context.transform_sources(context.config, "corex", mod.process)
    cc_flags = remove_flags(config.cc_flags, ["-fopenmp", "-DIS_CUDA", "-DHAS_CUDA"])
    cc_flags += " -DHAS_CUDA "
    return config.evolve(
        backend="corex", has_corex=True, has_cuda=True, is_cuda=False,
        cc_path=discovery.compiler_path, nvcc_path=discovery.compiler_path,
        cc_type="clang", cc_flags=cc_flags,
        kernel_flags=config.kernel_flags.replace("-fopenmp", ""),
        nvcc_flags=cc_flags + " -x cu -Ofast -DNO_ATOMIC64 -Wno-c++11-narrowing ",
        convert_nvcc_flags=convert_nvcc_flags,
        environment=dict(config.environment, use_cutt="0"),
        resources=dict(config.resources, corex_home=discovery.home,
                       corex_converter=mod),
    )


def convert_nvcc_flags(flags):
    return flags


def install_extern(context):
    return False


def post_process(context):
    return context.config
