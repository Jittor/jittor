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
            output_dtypes: list = None,
            output_shapes: list = None,
            attr_code: str = "",
            attr_header: str = "",
            outputs: list = None):
    #         inputs: list,
    #         output_dtypes: list,
    #         output_shapes: list,
    #         attr_code: str = ""):
    # input_code = ''
    # for i in range(len(inputs)):
    #     input_code += f"op.add(in{i}, true);\n"

    # output_code = ''
    # for i in range(len(output_dtypes)):
    #     output_code += f"op.add(out{i}, false);\n"

    # # read the tmp_file.cpp to the cuda_header
    # with open(
    #         "/home/ma-user/work/zy/JittorHW/python/jittor/extern/acl/tmp_file.cpp",
    #         "r") as f:
    #     cuda_header = f.read()
    # import jittor as jt
    # return jt.code(output_shapes,
    #                output_dtypes,
    #                inputs,
    #                cuda_header=cuda_header,
    #                cuda_src=f"""
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"
    # print(attr_header)

    # read the tmp_file.cpp to the cuda_header
    with open(
            "/home/ma-user/work/zy/JittorHW/python/jittor/extern/acl/tmp_file.cpp",
            "r") as f:
        cuda_header = f.read()
    import jittor as jt
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        # print(f'{name } output_dtypes', output_dtypes)
        # print(f'{name } output_shapes', output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    # print(f'{name } outputs_', outputs_)
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
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
            else:
                output_shapes.append([1])
                output_dtypes.append(x.dtype)
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
            if self.bias is None:
                return results[0], results[1]

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

    class PoolACL(Function):

        def __init__(self,
                     kernel_size,
                     stride=None,
                     padding=0,
                     dilation=None,
                     return_indices=None,
                     ceil_mode=False,
                     count_include_pad=True,
                     op='maximum'):
            self.kernel_size = kernel_size if isinstance(
                kernel_size, tuple) else (kernel_size, kernel_size)
            stride = stride if stride else kernel_size
            self.stride = stride if isinstance(stride, tuple) else (stride,
                                                                    stride)
            self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                       padding)
            dilation = dilation if dilation else 1
            assert dilation == 1
            self.dilation = dilation if isinstance(
                dilation, tuple) else (dilation, dilation)
            for item in self.kernel_size:
                if item <= 0:
                    raise RuntimeError(
                        f"kernel_size must be greater than zero, but got {item}"
                    )
            for item in self.stride:
                if item <= 0:
                    raise RuntimeError(
                        f"stride must be greater than zero, but got {item}")
            for item in self.padding:
                if item < 0:
                    raise RuntimeError(
                        f"padding must be non-negative, but got {item}")
            self.op = op
            self.return_indices = return_indices
            self.ceil_mode = ceil_mode
            self.count_include_pad = count_include_pad

        def execute(self, input):
            self.input = input
            attr_code = f"""
            op.jt_name = "maxpool";
            PoolAttr *attr = new PoolAttr();
            attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
            attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
            attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
            attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
            attr->poolCeil = {"true" if self.ceil_mode else "false"};
            op.op_attr.reset(attr);
            """
            input_height, input_width = input.shape[-2:]
            kernel_height, kernel_width = self.kernel_size[-2:]

            output_height = (input_height + 2 * self.padding[0] -
                             (kernel_height - 1) - 1) // self.stride[0] + 1
            output_width = (input_width + 2 * self.padding[1] -
                            (kernel_width - 1) - 1) // self.stride[1] + 1

            output_shape = (input.shape[0], input.shape[1], output_height,
                            output_width)

            inputs = [input]

            if self.op == 'maximum':
                result = acl_cmd(
                    "Maxpool",
                    inputs,
                    output_dtypes=[input.dtype, 'int32'],
                    output_shapes=[output_shape, output_shape],
                    attr_code=attr_code,
                )
            elif self.op == 'mean':
                result = acl_cmd(
                    "Avgpool",
                    inputs,
                    output_dtypes=[input.dtype],
                    output_shapes=[output_shape],
                    attr_code=attr_code,
                )
            else:
                raise ValueError('no this type pool')

            if self.op == 'maximum':
                self.index = result[1]

            if self.return_indices:
                return result[0], result[1]
            else:
                return result[0]

        def grad(self, grad_output, indices=None):
            input = self.input
            inputs = [grad_output, input, indices]
            attr_code = f"""
            op.jt_name = "maxpoolbackward";
            PoolAttr *attr = new PoolAttr();
            attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
            attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
            attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
            attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
            attr->poolCeil = {"true" if self.ceil_mode else "false"};
            op.op_attr.reset(attr);
            """
            output_shapes = [input.shape]
            output_dtypes = [input.dtype]
            result = acl_cmd("MaxpoolBackward",
                             inputs,
                             output_dtypes=output_dtypes,
                             output_shapes=output_shapes,
                             attr_code=attr_code)[0]
            return result

    class FlipACL(Function):

        def __init__(self):
            super(FlipACL, self).__init__()

        def execute(self, input, dim):
            self.input = input
            attr_code = f"""
            op.jt_name = "flip";
            ReduceAttr *attr = new ReduceAttr();
            attr->axes = {{{', '.join(map(str, (list(dim))))}}};
            attr->prod_dim = {len(dim)};
            op.op_attr.reset(attr);
            """
            self.attr_code = attr_code
            result = acl_cmd("Flip", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code=self.attr_code)[0]
            return result

        def grad(self, grad_output):
            grad_input = acl_cmd("Flip", [grad_output],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code=self.attr_code)[0]
            return grad_input

    class ConcatACL(Function):

        def __init__(self):
            super(ConcatACL, self).__init__()

        def execute(self, input_tensors, dim=0):
            self.input = input_tensors
            self.dim = dim
            for i in range(len(input_tensors)):
                if input_tensors[i].dtype != input_tensors[0].dtype:
                    raise ValueError(
                        "All input tensors must have the same dtype")
                if input_tensors[i].shape[:dim] != input_tensors[
                        0].shape[:dim] or input_tensors[i].shape[
                            dim + 1:] != input_tensors[0].shape[dim + 1:]:
                    raise ValueError(
                        "All input tensors must have the same shape")
            attr_code = f"""
            op.jt_name = "concat";
            ConcatAttr *attr = new ConcatAttr();
            attr->tensorNum = {len(input_tensors)};
            attr->dim = {dim};
            op.op_attr.reset(attr);
            """
            result = acl_cmd(
                "Concat",
                input_tensors,
                output_dtypes=[input_tensors[0].dtype],
                output_shapes=[
                    jt.empty(self.calculate_output_shape(input_tensors,
                                                         dim)).shape
                ],
                attr_code=attr_code)[0]
            return result

        """def grad(self, grad_output):
            grad_inputs = self.split_grad(grad_output, self.input, self.axis)
            return grad_inputs"""

        def calculate_output_shape(self, input_tensors, axis):
            shape = list(input_tensors[0].shape)
            for tensor in input_tensors[1:]:
                shape[axis] += tensor.shape[axis]
            return tuple(shape)

        """def split_grad(self, grad_output, input_tensors, axis):
            offset = 0
            grad_inputs = []
            for tensor in input_tensors:
                grad_input = acl_cmd("Slice", [
                    grad_output, [0] * axis + [offset] + [0] *
                    (len(tensor.shape) - axis - 1), tensor.shape
                ])
                grad_inputs.append(grad_input)
                offset += tensor.shape[axis]
            return grad_inputs"""

    class GatherACL(Function):

        def __init__(self):
            super(GatherACL, self).__init__()

        def execute(self, input, dim, index):
            self.input = input
            self.dim = dim
            self.index = index
            attr_code = f"""
            op.jt_name = "gather";
            GatherAttr *attr = new GatherAttr();
            attr->dim = {dim};
            op.op_attr.reset(attr);
            """
            result = acl_cmd("Gather", [input, index],
                             output_dtypes=[input.dtype],
                             output_shapes=[index.shape],
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            tmp = jt.zeros(self.index.shape, dtype=grad_output.dtype)
            attr_code = f"""
            op.jt_name = "scatter";
            ScatterAttr *attr = new ScatterAttr();
            attr->axis = {self.dim};
            attr->reduction = {1};
            op.op_attr.reset(attr);
            """
            grad_input = acl_cmd("Scatter", [tmp, self.index, grad_output],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[tmp.shape],
                                 attr_code=attr_code)[0]
            return grad_input

    class CumsumACL(Function):

        def __init__(self):
            super(CumsumACL, self).__init__()

        def execute(self, input, dim=-1):
            self.input = input
            self.dim = dim
            attr_code = f"""
            op.jt_name = "cumsum";
            GatherAttr *attr = new GatherAttr();
            attr->dim = {dim};
            op.op_attr.reset(attr);
            """
            result = acl_cmd("Cumsum", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            cumsum_attr_code = f"""
            op.jt_name = "cumsum";
            GatherAttr *attr = new GatherAttr();
            attr->dim = {self.dim};
            op.op_attr.reset(attr);
            """
            flip_attr_code = f"""
            op.jt_name = "flip";
            ReduceAttr *attr = new ReduceAttr();
            attr->axes = {{{self.dim}}};
            attr->prod_dim = {{{1}}};
            op.op_attr.reset(attr);
            """
            flipped_grad_output = acl_cmd("Flip", [grad_output],
                                          output_dtypes=[grad_output.dtype],
                                          output_shapes=[grad_output.shape],
                                          attr_code=flip_attr_code)[0]
            cumulative_grad = acl_cmd("Cumsum", [flipped_grad_output],
                                      output_dtypes=[grad_output.dtype],
                                      output_shapes=[grad_output.shape],
                                      attr_code=cumsum_attr_code)[0]
            grad_input = acl_cmd("Flip", [cumulative_grad],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code=flip_attr_code)[0]
            return grad_input

    class IndexACL(Function):

        def __init__(self):
            super(IndexACL, self).__init__()

        def execute(self, inshape: list, dim=None, dtype="int32"):
            # zeros a tensor, shape is inshape, dtype is dtype
            if dim == None:
                dim = [i for i in range(len(inshape))]
            elif type(dim) == int:
                dim = [dim]
            results = []
            for d in dim:
                max_len = inshape[d]
                tmp = jt.zeros(max_len, dtype=dtype)
                result = acl_cmd(
                    "Index", [jt.Var(0), jt.Var(max_len),
                              jt.Var(1)],
                    output_dtypes=[tmp.dtype],
                    output_shapes=[tmp.shape],
                    attr_code="op.jt_name=\"index\";")[0]
                broadcast_dim = []
                for i in range(len(inshape)):
                    if i != d:
                        broadcast_dim.append(i)
                result = jt.broadcast(result,
                                      shape=inshape,
                                      dims=broadcast_dim)
                results.append(result)
            return tuple(results)

        def grad(self, grad_output):
            return grad_output

    class ScatterACL(Function):

        def __init__(self):
            super(ScatterACL, self).__init__()

        def __call__(self, input, dim, index, src, reduce='void'):
            return self.execute(input, dim, index, src, reduce)

        def execute(self, input, dim, index, src, reduce='void'):
            self.input = input
            self.dim = dim
            self.index = index
            self.reduce = reduce
            attr_code = f"""
            op.jt_name = "scatter";
            ScatterAttr *attr = new ScatterAttr();
            attr->axis = {dim};
            attr->reduction = {1 if reduce == 'add' else 2 if reduce == 'mul' else 0};
            op.op_attr.reset(attr);
            """
            result = acl_cmd("Scatter", [input, self.index, src],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            attr_code = f"""
            op.jt_name = "gather";
            GatherAttr *attr = new GatherAttr();
            attr->dim = {self.dim};
            op.op_attr.reset(attr);
            """
            grad_input = acl_cmd("Gather", [grad_output, self.index],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[self.index.shape],
                                 attr_code=attr_code)[0]
            return grad_output, None, None, grad_input

    class WhereACL(Function):

        def __init__(self):
            super(WhereACL, self).__init__()

        def execute(self, condition, x, y):
            self.condition = condition

            if x.dtype != y.dtype:
                if x.dtype == jt.float32:
                    y = y.float32()
                elif y.dtype == jt.float32:
                    x = x.float32()
                else:
                    x = x.to(y.dtype)

            self.x = x
            self.y = y

            result = acl_cmd("Where", [condition, x, y],
                             output_dtypes=[x.dtype],
                             output_shapes=[x.shape],
                             attr_code="op.jt_name=\"where\";")[0]
            return result

        def grad(self, grad_output):
            tmp = jt.zeros(grad_output.shape, dtype=grad_output.dtype)
            grad_x = acl_cmd("Where", [self.condition, grad_output, tmp],
                             output_dtypes=[self.x.dtype],
                             output_shapes=[self.x.shape],
                             attr_code="op.jt_name=\"where\";")[0]

            grad_y = acl_cmd("Where", [self.condition, tmp, grad_output],
                             output_dtypes=[self.y.dtype],
                             output_shapes=[self.y.shape],
                             attr_code="op.jt_name=\"where\";")[0]
            return grad_output, grad_x, grad_y

    class FloorIntACL(Function):

        def __init__(self):
            super(FloorIntACL, self).__init__()

        def execute(self, input):
            self.input = input
            self.shape = input.shape
            result = acl_cmd("Floor", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code="op.jt_name=\"floor\";")[0]
            return result

        def grad(self, grad_output):
            return jt.zeros(self.shape, dtype=grad_output.dtype)

    def caculate_shape(tensors):
        if isinstance(tensors, jt.Var):
            # tensors = tensors[0]
            return tensors.shape
        elif isinstance(tensors, (int, float)):
            return []
        elif isinstance(tensors, (list, tuple)):
            # return [caculate_shape(tensor) for tensor in tensors]
            sub_shape = caculate_shape(tensors[0])
            return [len(tensors)] + sub_shape
        else:
            assert False, f"not implemented for {type(tensors)}"

    def can_broadcast_and_shape(shape1, shape2):
        """
        检查两个张量是否可以广播，并返回广播后的形状。
        
        参数:
        - shape1: 第一个张量的形状（tuple 或 list）
        - shape2: 第二个张量的形状（tuple 或 list）
        
        返回:
        - can_broadcast: 布尔值，表示是否可以广播
        - broadcast_shape: 如果可以广播，返回广播后的形状；否则返回 None
        """
        # 将形状转换为元组，以防输入是列表
        shape1 = tuple(shape1)
        shape2 = tuple(shape2)

        # 使两个形状的长度一致，通过在前面补1
        len1, len2 = len(shape1), len(shape2)
        if len1 < len2:
            shape1 = (1, ) * (len2 - len1) + shape1
        elif len2 < len1:
            shape2 = (1, ) * (len1 - len2) + shape2

        broadcast_shape = []

        # 从最后一维开始检查每一维度
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2:
                broadcast_shape.append(dim1)
            elif dim1 == 1:
                broadcast_shape.append(dim2)
            elif dim2 == 1:
                broadcast_shape.append(dim1)
            else:
                # 如果在某一维度上不兼容，则不能广播
                return False, None

        return True, tuple(broadcast_shape)

    class GetItemACL(Function):

        def __init__(self):
            self.type_ = 'notype'

        def stride(self, x, dim):
            stride = 1
            for i in range(dim + 1, len(x.shape)):
                stride *= x.shape[i]
            return stride

        def execute(self, x, slices, return_x=None):
            self.x_shape = x.shape
            if not isinstance(slices, tuple):
                slices = (slices, )
            slices_list = list(slices)
            # if not isinstance(slices[0], slice):
            #check slices contains slice type
            contains_slice = False
            for s in slices:
                if isinstance(s, slice):
                    contains_slice = True
                    break
            if not contains_slice:
                indices = []
                output_shape = []
                slices_len = len(slices)
                boardcast_shape = caculate_shape(slices_list[0])
                for ii in range(1, len(slices)):
                    dd, boardcast_shape = can_broadcast_and_shape(
                        boardcast_shape, caculate_shape(slices_list[ii]))
                    assert dd is True, "can not broadcast"
                output_shape = boardcast_shape
                output_shape += x.shape[slices_len:]
                for ii in slices:
                    indices.append(jt.Var(ii))
                if isinstance(slices[0], jt.Var) or isinstance(
                        slices[0], int) or isinstance(
                            slices[0], list) or isinstance(slices[0], tuple):
                    self.indices = indices
                    inputs = [x] + indices
                    attr_code = f"""
                    op.jt_name = "index";
                    """
                    self.type_ = 'index'
                    result = acl_cmd("Index",
                                     inputs=inputs,
                                     output_dtypes=[x.dtype],
                                     output_shapes=[output_shape],
                                     attr_code=attr_code)[0]
                    return result

            x_dim = len(x.shape)
            if len(slices) < x_dim:
                slices += (slice(None, None, None), ) * (x_dim - len(slices))
            inputs = [x]
            sizes = []
            begins = []
            ends = []
            steps = []
            dims = []
            squeeze_dims = []
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                    squeeze_dims.append(dim)
                if isinstance(s, jt.Var):
                    assert False, "jt.Var not supported"
                start, stop, step = s.indices(x.size(dim))
                size = (stop - start - 1) // step + 1
                stride = self.stride(x, dim) * step
                sizes.append(size)
                steps.append(step)
                begins.append(start)
                ends.append(stop)
                dims.append(dim)
            if not sizes:
                sizes = [1]
                steps = [1]
            self.type_ = 'slicev2'
            self.begins = begins
            self.ends = ends
            self.steps = steps
            self.dims = dims
            attr_code = f"""
            op.jt_name = "slicev2";
            StrideAttr *attr = new StrideAttr();
            attr->begins = {{ {", ".join(map(str, begins))} }};
            attr->ends = {{ {", ".join(map(str, ends))} }};
            attr->steps = {{ {", ".join(map(str, steps))} }};
            attr->axes = {{ {", ".join(map(str, dims))} }};
            op.op_attr.reset(attr);
            """

            result = acl_cmd("SliceV2",
                             inputs,
                             output_dtypes=[x.dtype],
                             output_shapes=[jt.empty(sizes).shape],
                             attr_code=attr_code)[0]
            for dim in squeeze_dims[::-1]:
                result = jt.squeeze(result, dim)
            return result

        def grad(self, grad_output):
            if self.type_ == 'index':
                indices = self.indices
                inputs = [grad_output] + indices
                attr_code = f"""
                op.jt_name = "indexputimpl";
                """
                outputs = [jt.zeros(self.x_shape)]
                result = acl_cmd("IndexPutImpl",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
                return result
            elif self.type_ == 'slicev2':
                #TODO: wait for cann update
                assert False, f"wait for cann update"
                begins = self.begins
                ends = self.ends
                steps = self.steps
                dims = self.dims
                begins = jt.Var(begins).int64()
                ends = jt.Var(ends).int64()
                steps = jt.Var(steps).int64()
                dims = jt.Var(dims).int64()
                inputs = [grad_output, begins, ends, steps, dims]
                attr_code = f"""
                op.jt_name = "stridedsliceassignv2";
                """
                result = acl_cmd("StridedSliceAssignV2",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
                return result
            else:
                assert False, f"grad not implemented for {self.type_}"

    class BmmACL(Function):

        def __init__(self, trans_x2=False):
            super(BmmACL, self).__init__()
            self.trans_x2 = trans_x2

        def execute(self, x1, x2):
            if self.trans_x2:
                x2 = x2.transpose(-2, -1)
            self.input = [x1, x2]
            result = acl_cmd("BatchMatMul", [x1, x2],
                             output_dtypes=[x1.dtype],
                             output_shapes=[x1.shape[:-1] + x2.shape[-1:]],
                             attr_code="op.jt_name=\"bmm\";")[0]

            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            grad_x1 = acl_cmd(
                "BatchMatMul", [grad_output, x2.transpose(-2, -1)],
                output_dtypes=[x1.dtype],
                output_shapes=[grad_output.shape[:-1] + x1.shape[-1:]],
                attr_code="op.jt_name=\"bmm\";")[0]
            x2 = x2.transpose(-2, -1)
            grad_x2 = acl_cmd(
                "BatchMatMul", [x1.transpose(-2, -1), grad_output],
                output_dtypes=[x2.dtype],
                output_shapes=[x2.shape[:-1] + grad_output.shape[-1:]],
                attr_code="op.jt_name=\"bmm\";")[0]
            x1 = x1.transpose(-2, -1)
            return grad_x1, grad_x2

    class MatmulACL(Function):

        def __init__(self, trans_x2=False):
            super(MatmulACL, self).__init__()
            self.trans_x2 = trans_x2

        def execute(self, x1, x2):
            if self.trans_x2:
                x2 = x2.transpose(-2, -1)
            self.input = [x1, x2]
            result = acl_cmd("MatMul", [x1, x2],
                             output_dtypes=[x1.dtype],
                             output_shapes=[x1.shape[:-1] + x2.shape[-1:]],
                             attr_code="op.jt_name=\"matmul\";")[0]
            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            grad_x1 = acl_cmd(
                "MatMul", [grad_output, x2.transpose(-2, -1)],
                output_dtypes=[x1.dtype],
                output_shapes=[grad_output.shape[:-1] + x1.shape[-1:]],
                attr_code="op.jt_name=\"matmul\";")[0]
            grad_x2 = acl_cmd(
                "MatMul", [x1.transpose(-2, -1), grad_output],
                output_dtypes=[x2.dtype],
                output_shapes=[x2.shape[:-1] + grad_output.shape[-1:]],
                attr_code="op.jt_name=\"matmul\";")[0]
            return grad_x1, grad_x2

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
    jt.nn.Pool = warp(jt.nn.Pool, PoolACL)

    jt.flip = warp(jt.flip, FlipACL())
    jt.Var.flip = lambda x, dim_vector: warp(jt.Var.flip, FlipACL())(
        x, dim_vector)
    jt.concat = warp(jt.concat, ConcatACL())

    jt.gather = warp(jt.gather, GatherACL())

    jt.cumsum = warp(jt.cumsum, CumsumACL())
    # jt.index = warp(jt.index, IndexACL())
    # jt.Var.index = lambda x, dim=None: warp(jt.index, IndexACL())(x.shape, dim)

    jt.scatter = warp(jt.scatter, ScatterACL())
    jt.Var.scatter = lambda x, dim, index, src, reduce="void": warp(
        jt.scatter, ScatterACL())(x, dim, index, src, reduce)

    jt.floor_int = warp(jt.floor_int, FloorIntACL())
    jt.Var.floor_int = lambda x: warp(jt.floor_int, FloorIntACL())(x)

    jt.getitem = warp(jt.getitem, GetItemACL())
    jt.Var.getitem = lambda x, slices, return_x=None: warp(
        jt.getitem, GetItemACL())(x, slices)

    jt.nn.bmm = warp(jt.nn.bmm, BmmACL())
    jt.bmm = warp(jt.bmm, BmmACL())
    jt.nn.matmul = warp(jt.matmul, MatmulACL())
    jt.matmul = warp(jt.matmul, MatmulACL())
    jt.nn.matmul_transpose = warp(jt.nn.matmul_transpose, MatmulACL(True))
    jt.nn.bmm_transpose = warp(jt.nn.bmm_transpose, BmmACL(True))
    jt.bmm_transpose = warp(jt.bmm_transpose, BmmACL(True))
