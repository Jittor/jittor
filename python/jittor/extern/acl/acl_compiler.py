# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
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
import jittor as jt
import math

from collections.abc import Sequence, Iterable


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple([x] * n)

    return parse


_pair = _ntuple(2)

has_acl = 0
cc_flags = ""
tikcc_path = env_or_try_find('tikcc_path', 'ccec')
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL
compiler.has_acl = has_acl

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aoe/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub:/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/Ascend910A:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/simulator/Ascend910A/lib:/opt/AXESMI/lib64:/usr/local/Ascend/driver/lib64/driver/
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python
# export tikcc_path=g++

# conda activate cann
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH
# export TASK_QUEUE_ENABLE=0
# python3 -m jittor.test.test_acl -k array
# jittor: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 -m jittor.test.test_acl -k test_sum
# export ASCEND_SLOG_PRINT_TO_STDOUT=0
# ASCEND_GLOBAL_LOG_LEVEL
# export DUMP_GE_GRAPH=1
# export DUMP_GRAPH_LEVEL=1

# build pytorch-npu
# bash ./ci/build.sh
# python3 -m pip install ./dist/torch_npu-1.11.0.post1-cp37-cp37m-linux_x86_64.whl  --force-reinstall
# pytorch: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export TASK_QUEUE_ENABLE=0  && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 ./mm_bench_pt_npu.py


def install():
    import jittor.compiler as compiler
    global has_acl, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home + "/**/*.cc",
                                recursive=True))
    cc_files2 = []
    for name in cc_files:
        if "acl_op_exec" in name:
            compiler.extra_core_files.append(name)
        else:
            cc_files2.append(name)
    cc_files = cc_files2
    ascend_toolkit_home = os.getenv('ASCEND_TOOLKIT_HOME')

    #print(ascend_toolkit_home)
    #print(acl_compiler_home)
    cc_flags += f" -DHAS_CUDA -DIS_ACL  \
    -I{ascend_toolkit_home}/include/ \
    -I{ascend_toolkit_home}/include/acl/ \
    -I{ascend_toolkit_home}/include/aclnn/ \
    -I{ascend_toolkit_home}/include/aclnnop/ \
    -I{acl_compiler_home} -lascendcl -lacl_op_compiler \
    -I{acl_compiler_home}/aclnn \
    -L{ascend_toolkit_home}/lib64/"

    cc_flags += " -llibascendcl "
    cc_flags += " -llibnnopbase "
    cc_flags += " -llibopapi "

    #pdb.set_trace()
    ctypes.CDLL("libascendcl.so", dlopen_flags)
    f'''
    -ltikc_runtime
    -I/usr/local/Ascend/driver/include/ \
    -L{ascend_toolkit_home}/compiler/lib64/ \
    -L{ascend_toolkit_home}/runtime/lib64/ \
    '''
    jittor_utils.LOG.i("ACL detected")

    global mod
    mod = jittor_utils.compile_module(
        '''
#include "common.h"
namespace jittor {
// @pyjt(process)
string process_acl(const string& src, const string& name, const map<string,string>& kargs);
// @pyjt(init_acl_ops)
void init_acl_ops();
}''', compiler.cc_flags + " " + " ".join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source("acl", mod.process)

    has_acl = 1
    os.environ["use_mkl"] = "0"
    compiler.setup_fake_cuda_lib = True


def install_extern():
    return False


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
    compiler.nvcc_flags = compiler.cc_flags.replace("-std=c++14", "")
    return True


def post_process():
    if has_acl:
        from jittor import pool
        pool.pool_use_code_op = False
        import jittor as jt
        jt.flags.use_cuda_host_allocator = 1
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.amp_reg |= 32 + 4  # 32 keep float16, 4 keep reduce type
        mod.init_acl_ops()


def acl_cmd(name: str,
            inputs: list,
            output_dtypes: list,
            output_shapes: list,
            attr_code: str = ""):
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(output_dtypes)):
        output_code += f"op.add(out{i}, false);\n"

    # read the tmp_file.cpp to the cuda_header
    with open(
            "/home/ma-user/work/zy/jittor/python/jittor/extern/acl/tmp_file.cpp",
            "r") as f:
        cuda_header = f.read()
    import jittor as jt
    return jt.code(output_shapes,
                   output_dtypes,
                   inputs,
                   cuda_header=cuda_header,
                   cuda_src=f"""
    // aclop
    AclOpRunner op("{name}");
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")


def change_function():
    import jittor as jt
    from jittor import Function

    class TriuACL(Function):

        def __init__(self):
            super(TriuACL, self).__init__()

        def execute(self, input, k):
            self.input = input

            attr_code = f"""
            op.jt_name = "triu";
            TriuAttr *attr = new TriuAttr();
            attr->diagonal = {k};
            op.op_attr.reset(attr);
            """

            result = acl_cmd("Triu", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            return grad_output

    class ConvACL(Function):

        def execute(self,
                    x,
                    weight,
                    bias=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1):
            self.input = x
            self.weight = weight
            self.bias = bias
            padding = _pair(padding)
            stride = _pair(stride)
            dilation = _pair(dilation)
            out_channels = weight.shape[0]
            if groups <= 0:
                raise ValueError("groups must be a positive integer")
            self.padding = padding
            self.stride = stride
            self.dilation = dilation
            self.groups = groups
            attr_code = f"""
            op.jt_name = "conv2d";
            ConvAttr *attr = new ConvAttr();
            attr->convStrides = {{ {stride[0]}, {stride[1]} }};
            attr->convPads = {{ {padding[0]}, {padding[1]} }};
            attr->convDilations = {{ {dilation[0]}, {dilation[1]} }};
            attr->group = {groups};
            attr->convOutPads = {{ 1,1}};
            op.op_attr.reset(attr);
            """
            input_height, input_width = x.shape[-2:]
            kernel_height, kernel_width = weight.shape[-2:]

            output_height = (input_height + 2 * padding[0] - dilation[0] *
                             (kernel_height - 1) - 1) // stride[0] + 1
            output_width = (input_width + 2 * padding[1] - dilation[1] *
                            (kernel_width - 1) - 1) // stride[1] + 1

            output_shape = (x.shape[0], out_channels, output_height,
                            output_width)
            
            inputs = [x, weight]
            if bias is not None:
                inputs.append(bias)
            result = acl_cmd(
                "Conv2d",
                inputs,
                output_dtypes=[x.dtype],
                output_shapes=[output_shape],
                attr_code=attr_code,
            )[0]
            return result

        def grad(self, grad_output):
            x = self.input
            weight = self.weight
            bias = self.bias
            inputs = [grad_output, x, weight]

            if bias is not None:
                inputs.append(bias)
            output_shapes = [x.shape, weight.shape]
            output_dtypes = [x.dtype, weight.dtype]
            if bias is not None:
                output_shapes.append(bias.shape)
                output_dtypes.append(bias.dtype)
            padding = self.padding
            stride = self.stride
            dilation = self.dilation
            groups = self.groups
            attr_code = f"""
            op.jt_name = "conv2dbackward";
            ConvAttr *attr = new ConvAttr();
            attr->convStrides = {{ {stride[0]}, {stride[1]} }};
            attr->convPads = {{ {padding[0]}, {padding[1]} }};
            attr->convDilations = {{ {dilation[0]}, {dilation[1]} }};
            attr->group = {groups};
            attr->convOutPads = {{ 1,1}};
            op.op_attr.reset(attr);
            """
            results = acl_cmd("Conv2dBackward",
                              inputs,
                              output_dtypes=output_dtypes,
                              output_shapes=output_shapes,
                              attr_code=attr_code)

            return results

    class Conv2D(jt.nn.Module):

        def __init__(self,
                     in_channels,
                     out_channels,
                     kernel_size,
                     stride=1,
                     padding=0,
                     dilation=1,
                     groups=1,
                     bias=True):
            if in_channels <= 0:
                raise ValueError(
                    f"in_channels must be greater than zero, got {in_channels}"
                )
            if out_channels <= 0:
                raise ValueError(
                    f"out_channels must be greater than zero, got {out_channels}"
                )
            if groups <= 0:
                raise ValueError(
                    f"groups must must be greater than zero, got {groups}")
            assert in_channels % groups == 0, 'in_channels must be divisible by groups'
            assert out_channels % groups == 0, 'out_channels must be divisible by groups'
            if isinstance(kernel_size, tuple):
                for size in kernel_size:
                    if size <= 0:
                        raise ValueError(
                            f"kernel_size must be greater than zero, got {kernel_size}"
                        )
            else:
                if kernel_size <= 0:
                    raise ValueError(
                        f"kernel_size must be greater than zero, got {kernel_size}"
                    )
            if isinstance(stride, tuple):
                for size in stride:
                    if size <= 0:
                        raise ValueError(
                            f"stride must be greater than zero, got {stride}")
            else:
                if stride <= 0:
                    raise ValueError(
                        f"stride must be greater than zero, got {stride}")
            if isinstance(padding, tuple):
                for size in padding:
                    if size < 0:
                        raise ValueError(
                            f"padding must be nonnegative, got {padding}")
            else:
                if padding < 0:
                    raise ValueError(
                        f"padding must be nonnegative, got {padding}")
            if isinstance(dilation, tuple):
                for size in dilation:
                    if size <= 0:
                        raise ValueError(
                            f"dilation must be greater than zero, got {dilation}"
                        )
            else:
                if dilation <= 0:
                    raise ValueError(
                        f"dilation must be greater than zero, got {dilation}")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(
                kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride,
                                                                    stride)
            self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                       padding)
            self.dilation = dilation if isinstance(
                dilation, tuple) else (dilation, dilation)
            self.groups = groups
            self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
            if self.is_depthwise_conv and jt.flags.use_cuda and jt.compiler.is_cuda:
                self.depthwise_conv = jt.nn.DepthwiseConv(
                    stride, padding, dilation)
            Kh, Kw = self.kernel_size

            # self.weight = init.relu_invariant_gauss([out_channels, in_channels//groups, Kh, Kw], dtype="float", mode="fan_out")
            self.weight = jt.init.invariant_uniform(
                [out_channels, in_channels // groups, Kh, Kw], dtype="float")
            if bias:
                fan = 1
                for i in self.weight.shape[1:]:
                    fan *= i
                bound = 1 / math.sqrt(fan)
                self.bias = jt.init.uniform([out_channels],
                                            dtype="float",
                                            low=-bound,
                                            high=bound)
            else:
                self.bias = None

        def execute(self, x):
            ret = jt.nn.conv2d(x, self.weight, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)
            return ret

    def warp(origin_func, new_func):

        def warpper(*args, **kwargs):
            if jt.flags.use_acl:
                return new_func(*args, **kwargs)
            return origin_func(*args, **kwargs)

        return warpper

    jt.triu = warp(jt.triu, TriuACL())
    jt.triu_ = warp(jt.triu, TriuACL())
    jt.Var.triu = lambda x: warp(jt.Var.triu, TriuACL())(x)
    jt.Var.triu_ = lambda x: warp(jt.Var.triu_, TriuACL())(x)
    jt.nn.conv2d = warp(jt.nn.conv2d, ConvACL())
    jt.nn.Conv2d = warp(jt.nn.Conv2d, Conv2D)
    jt.nn.Conv = warp(jt.nn.Conv, Conv2D)
