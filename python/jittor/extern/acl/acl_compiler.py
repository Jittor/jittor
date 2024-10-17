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

    cuda_header = '#include"acl/acl_op.h"'
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

    def triu_acl(x, k=0):
        return TriuACL()(x, k)

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
            op.jt_name  = "{"avgpool" if self.op == 'mean' else "maxpool"}";
            PoolAttr *attr = new PoolAttr();
            attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
            attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
            attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
            attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
            attr->poolCeil = {"true" if self.ceil_mode else "false"};
            attr->countIncludePad = {"true" if self.count_include_pad else "false"};
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

        def grad(self, grad_output):
            input = self.input
            attr_code = f"""
            op.jt_name = "{"avgpoolbackward" if self.op == 'mean' else "maxpoolbackward"}";
            PoolAttr *attr = new PoolAttr();
            attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
            attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
            attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
            attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
            attr->poolCeil = {"true" if self.ceil_mode else "false"};
            attr->countIncludePad = {"true" if self.count_include_pad else "false"};
            op.op_attr.reset(attr);
            """
            output_shapes = [input.shape]
            output_dtypes = [input.dtype]
            if self.op == 'maximum':
                result = acl_cmd("MaxpoolBackward",
                                 inputs=[grad_output, input, self.index],
                                 output_dtypes=output_dtypes,
                                 output_shapes=output_shapes,
                                 attr_code=attr_code)[0]
            elif self.op == 'mean':
                result = acl_cmd("AvgpoolBackward",
                                 inputs=[grad_output, input],
                                 output_dtypes=output_dtypes,
                                 output_shapes=output_shapes,
                                 attr_code=attr_code)[0]
            else:
                raise ValueError('no this type pool')
            return result

    class FlipACL(Function):

        def __init__(self):
            super(FlipACL, self).__init__()

        def execute(self, input, dim):
            if type(dim) is not list:
                dim = [dim]
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

    def flip_acl(x, dim):
        return FlipACL()(x, dim)

    class ConcatACL(Function):

        def __init__(self):
            super(ConcatACL, self).__init__()

        def __call__(self, *args):
            assert isinstance(args[0], list)
            assert isinstance(args[1], int)
            if jt.flags.no_grad:
                return self.execute(*args)
            backup = args
            args = list(args)
            taped_inputs = []
            taped_outputs = []
            input_mask = [-1] * (len(args[0]) + 1)
            newargs = [list(), args[1]]
            for i, v in enumerate(args[0]):
                if isinstance(v, jt.Var):
                    if v.is_stop_grad():
                        # -2 in input_mask represents it is stop_grad
                        input_mask[i] = -2
                        newargs[0].append(v)
                        continue
                    v = v.tape()
                    newargs[0].append(v)
                    input_mask[i] = len(taped_inputs)
                    taped_inputs.append(v)

            ori_res = self.execute(*newargs)
            if not isinstance(ori_res, Sequence):
                res = [ori_res]
            else:
                res = list(ori_res)
            output_mask = [-1] * len(res)
            for i, v in enumerate(res):
                if isinstance(v, jt.Var):
                    v = v.tape()
                    output_mask[i] = len(taped_outputs)
                    res[i] = v
                    taped_outputs.append(v)
            self.input_mask = input_mask
            self.output_mask = output_mask
            # tape output and input together so
            # backward treat them as one operator
            jt.tape_together(taped_inputs, taped_outputs, self._grad)
            if isinstance(ori_res, Sequence):
                return res
            else:
                return res[0]

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

        def _grad(self, *args):
            new_args = ((args[i] if i >= 0 else None)
                        for i in self.output_mask)
            ret = self.grad(*new_args)
            new_ret = []
            for i, r in enumerate(ret):
                j = self.input_mask[i]
                if j < 0:
                    # -2 in input_mask represents it is stop_grad
                    assert r is None or j==-2, f"{type(self)}'s {i}-th returned grad should be None, "\
                        "because the input value is not jittor variable."
                else:
                    new_ret.append(r)
            return new_ret

        def grad(self, grad_output):
            grad_inputs = self.split_grad(grad_output, self.input, self.dim)
            return grad_inputs

        def calculate_output_shape(self, input_tensors, axis):
            shape = list(input_tensors[0].shape)
            for tensor in input_tensors[1:]:
                shape[axis] += tensor.shape[axis]
            return tuple(shape)

        def split_grad(self, grad_output, input_tensors, axis):
            offset = []
            shapeVec = []
            dtypeVec = []
            for tensor in input_tensors:
                offset.append(tensor.shape[axis])
                dtypeVec.append(tensor.dtype)
                shapeVec.append(tensor.shape)

            attr_code = f"""
            op.jt_name = "splitwithsize";
            auto *attr = new SplitWithSizeAttr();
            attr->splitSize = {{ {", ".join(map(str, offset))} }};
            attr->dim = {axis};
            op.op_attr.reset(attr);
            """

            result = acl_cmd("SplitWithSize", [grad_output],
                             output_dtypes=dtypeVec,
                             output_shapes=shapeVec,
                             attr_code=attr_code)
            return result

    def concat(x, dim=0):
        return ConcatACL()(x, dim)

    class GatherACL(Function):

        def __init__(self):
            super(GatherACL, self).__init__()

        def execute(self, input, dim, index):
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

    def gather_acl(input, dim, index):
        return GatherACL()(input, dim, index)

    class CumsumACL(Function):

        def __init__(self):
            super(CumsumACL, self).__init__()

        def execute(self, input, dim=-1):
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

    def cumsum_acl(input, dim=-1):
        return CumsumACL()(input, dim)

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
                range_attr_code = f"""
                op.jt_name = "range";
                RangeAttr *attr = new RangeAttr();
                attr->start = 0;
                attr->end = {max_len};
                attr->step = 1;
                op.op_attr.reset(attr);
                """
                result = acl_cmd("Range", [],
                                 output_dtypes=[tmp.dtype],
                                 output_shapes=[tmp.shape],
                                 attr_code=range_attr_code)[0]
                broadcast_dim = []
                for i in range(len(inshape)):
                    if i != d:
                        broadcast_dim.append(i)
                result = jt.broadcast(result,
                                      shape=inshape,
                                      dims=broadcast_dim)
                results.append(result)
            if len(results) != 1 or dim == None:
                return tuple(results)
            else:
                return results[0]

        def grad(self, grad_output):
            return grad_output

    def index_acl(inshape: list, dim=None, dtype="int32"):
        return IndexACL()(inshape, dim, dtype)

    class ScatterACL(Function):

        def __init__(self):
            super(ScatterACL, self).__init__()

        def execute(self, input, dim, index, src, reduce='void'):
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

    def scatter_acl(input, dim, index, src, reduce='void'):
        return ScatterACL()(input, dim, index, src, reduce)

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
            self.shape = input.shape
            result = acl_cmd("Floor", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr_code="op.jt_name=\"floor\";")[0]
            return result

        def grad(self, grad_output):
            return jt.zeros(self.shape, dtype=grad_output.dtype)

    def floor_int_acl(x):
        return FloorIntACL()(x)

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
            if isinstance(slices, jt.Var) and slices.dtype == 'bool':
                # assert False, "not support bool type now"
                #TODO:优化
                assert x.shape == slices.shape, "shape not match"
                output_len = slices.sum().item()
                # output = jt.empty((output_len,),dtype=x.dtype)
                x_len = x.numel()
                output = jt.empty((x_len), dtype=x.dtype)
                outputs = [output]
                inputs = [x, slices]
                # print(inputs,outputs)
                # print(output.shape)
                self.mask = slices
                self.type_ = 'mask'
                attr_code = f"""
                op.jt_name = "maskedselect";
                """
                result = acl_cmd("MaskedSelect",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
                result = result[:output_len]
                result.sync()
                return result
            self.x_shape = x.shape

            if not isinstance(slices, tuple):
                slices = (slices, )
            slices = list(slices)
            for i, s in enumerate(slices):
                if isinstance(s, int) and s < 0:
                    slices[i] = s + x.shape[i]
            slices = tuple(slices)
            slices_list = list(slices)
            # if not isinstance(slices[0], slice):
            #check slices contains slice type
            contains_slice = False
            for s in slices:
                if not isinstance(s, jt.Var) and (isinstance(s, slice)
                                                  or s == Ellipsis):
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
                if output_shape == []:
                    output_shape = [1]
                for ii in slices:
                    indices.append(jt.Var(ii).int32())
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
                    result.sync()
                    return result
            assert contains_slice, "slice type error"
            x_dim = len(x.shape)
            slices = list(slices)
            for s in slices:
                if not isinstance(s, jt.Var) and s == Ellipsis:
                    slices = slices[:slices.index(s)] + [
                        slice(None, None, None)
                    ] * (x_dim - len(slices) + 1) + slices[slices.index(s) +
                                                           1:]
                    break
            slices = tuple(slices)

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
                # stride = self.stride(x, dim) * step
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
            self.slices = slices
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
            self.squeeze_dims = squeeze_dims
            for dim in squeeze_dims[::-1]:
                result = jt.squeeze(result, dim)
            result.sync()
            return result

        def grad(self, grad_output):
            if self.type_ == 'index':
                indices = self.indices
                inputs = [grad_output] + indices
                attr_code = f"""
                op.jt_name = "indexputimplaccumulate";
                """
                outputs = [jt.zeros(self.x_shape, dtype=grad_output.dtype)]
                # breakpoint()
                result = acl_cmd("IndexPutImplAccumulate",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
                result.sync()
                return result, None
            elif self.type_ == 'slicev2':
                begins = self.begins
                ends = self.ends
                steps = self.steps
                dims = self.dims
                slices = self.slices
                #注意前向的维数可能会被压缩，所以这里要还原
                for dim in self.squeeze_dims:
                    grad_output = jt.unsqueeze(grad_output, dim)
                #适配华为奇怪的要求，最后一个维度的step必须是1
                expand_dim = False
                if isinstance(slices[-1], slice):
                    if slices[-1].step is not None and slices[-1].step != 1:
                        slices = slices + (slice(None, None, None), )
                        expand_dim = True
                elif isinstance(slices[-1], int):
                    #注意最后一个维度是数字
                    slices = list(slices)
                    slices[-1] = slice(slices[-1], slices[-1] + 1, 1)
                    slices = tuple(slices)
                    slices = slices + (slice(None, None, None), )
                    expand_dim = True
                else:
                    assert False, "not supported"
                    # x = x.unsqueeze(-1)
                if expand_dim:
                    grad_output = grad_output.unsqueeze(-1)
                    self.x_shape = self.x_shape + (1, )
                    sizes = []
                    begins = []
                    ends = []
                    steps = []
                    dims = []
                    for dim, s in enumerate(slices):
                        if isinstance(s, int):
                            s = slice(s, s + 1, 1)
                            # squeeze_dims.append(dim)
                        if isinstance(s, jt.Var):
                            assert False, "jt.Var not supported"
                        start, stop, step = s.indices(self.x_shape[dim])
                        size = (stop - start - 1) // step + 1
                        # stride = self.stride(x, dim) * step
                        sizes.append(size)
                        steps.append(step)
                        begins.append(start)
                        ends.append(stop)
                        dims.append(dim)
                    if not sizes:
                        sizes = [1]
                        steps = [1]
                attr_code = f"""
                op.jt_name = "stridedsliceassignv2";
                StrideAttr *attr = new StrideAttr();
                attr->begins = {{ {", ".join(map(str, begins))} }};
                attr->ends = {{ {", ".join(map(str, ends))} }};
                attr->steps = {{ {", ".join(map(str, steps))} }};
                attr->axes = {{ {", ".join(map(str, dims))} }};
                op.op_attr.reset(attr);
                """
                inputs = [grad_output]
                outputs = [jt.zeros(self.x_shape, dtype=grad_output.dtype)]
                result = acl_cmd("StridedSliceAssignV2",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
                result.sync()
                if expand_dim:
                    result = result.squeeze(-1)
                return result, None
            elif self.type_ == 'mask':
                return self.mask.float()
                pass
            else:
                assert False, f"grad not implemented for {self.type_}"

    def getitem_acl(x, slices, return_x=None):
        return GetItemACL()(x, slices, return_x)

    class SetItemACL(Function):

        def __init__(self):
            self.type_ = 'notype'
            self.value_var = True

        def stride(self, x, dim):
            stride = 1
            for i in range(dim + 1, len(x.shape)):
                stride *= x.shape[i]
            return stride

        def execute(self, x, slices, value):
            self.x_shape = x.shape
            self.input_slice = slices
            if not isinstance(value, jt.Var):
                self.value_var = False
            if isinstance(slices, jt.Var):
                if slices.dtype == "bool":
                    slices_len = slices.sum().item()
                    if slices_len == 0:
                        return x
                    if isinstance(value, int) or isinstance(value, float):
                        value = jt.full((slices_len, ), value)
                    assert slices.shape == x.shape, "setitem shape not match"
                    assert len(value.shape) == 1, "value shape must be 1D"
                    assert value.shape[
                        0] == slices_len, "value shape length must be equal to slices sum"
                    self.type_ = 'mask'
                    self.value_shape = value.shape
                    inputs = [value, slices]
                    outputs = [x.clone()]
                    attr_code = f"""
                    op.jt_name = "inplacemaskedscatter";
                    """
                    result = acl_cmd("InplaceMaskedScatter",
                                     inputs=inputs,
                                     outputs=outputs,
                                     attr_code=attr_code)[0]
                    return result

            # assert isinstance(value,jt.Var), "value must be jt.Var"
            # self.value_shape = value.shape
            if not isinstance(slices, tuple):
                slices = (slices, )
            slices = list(slices)
            for i, s in enumerate(slices):
                if isinstance(s, int) and s < 0:
                    slices[i] = x.shape[i] + s
            slices = tuple(slices)
            slices_list = list(slices)
            #check slices contains slice type
            contains_slice = False
            for s in slices:
                if not isinstance(s, jt.Var) and (isinstance(s, slice)
                                                  or s == Ellipsis):
                    contains_slice = True
                    break
            if not contains_slice:
                indices = []
                value_shape = []
                slices_len = len(slices)
                boardcast_shape = caculate_shape(slices_list[0])
                for ii in range(1, len(slices)):
                    dd, boardcast_shape = can_broadcast_and_shape(
                        boardcast_shape, caculate_shape(slices_list[ii]))
                    assert dd is True, "can not broadcast"
                value_shape = boardcast_shape
                value_shape += x.shape[slices_len:]
                if value_shape == []:
                    value_shape = [1]
                if isinstance(value, int) or isinstance(value, float):
                    value = jt.full(value_shape, value)
                self.value_shape = value_shape
                for ii in slices:
                    indices.append(jt.Var(ii).int32())
                if isinstance(slices[0], jt.Var) or isinstance(
                        slices[0], int) or isinstance(
                            slices[0], list) or isinstance(slices[0], tuple):
                    self.indices = indices
                    self.type_ = 'index'
                    attr_code = f"""
                    op.jt_name = "indexputimpl";
                    """
                    inputs = [value] + indices
                    outputs = [x.clone()]
                    result = acl_cmd("IndexPutImpl",
                                     inputs=inputs,
                                     outputs=outputs,
                                     attr_code=attr_code)[0]
                    # result.sync()
                    return result
                assert "not support"
            assert contains_slice, "slice type error"
            x_dim = len(x.shape)
            slices = list(slices)
            for s in slices:
                if not isinstance(s, jt.Var) and s == Ellipsis:
                    slices = slices[:slices.index(s)] + [
                        slice(None, None, None)
                    ] * (x_dim - len(slices) + 1) + slices[slices.index(s) +
                                                           1:]
                    break
            slices = tuple(slices)
            self.input_slice = slices
            if len(slices) < x_dim:
                slices += (slice(None, None, None), ) * (x_dim - len(slices))
            sizes = []
            begins = []
            ends = []
            steps = []
            dims = []
            #适配华为奇怪的要求，最后一个维度的step必须是1
            expand_dim = False
            if isinstance(slices[-1], slice):
                if slices[-1].step is not None and slices[-1].step != 1:
                    slices = slices + (slice(None, None, None), )
                    expand_dim = True

            elif isinstance(slices[-1], int):
                #注意最后一个维度是数字
                slices = slices + (slice(None, None, None), )
                expand_dim = True
                # value = value.unsqueeze(-1)
            else:
                assert False, "not supported"
            x_shape = list(x.shape)
            if expand_dim:
                x_shape.append(1)
                x = x.unsqueeze(-1)
                value = value.unsqueeze(-1)

            squeeze_dims = []
            if isinstance(value, jt.Var):
                for dim, s in enumerate(slices):
                    if isinstance(s, int):
                        s = slice(s, s + 1, 1)
                        squeeze_dims.append(dim)

                for dim in squeeze_dims:
                    value = value.unsqueeze(dim)
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                if isinstance(s, jt.Var):
                    assert False, "jt.Var not supported"
                start, stop, step = s.indices(x_shape[dim])
                size = (stop - start - 1) // step + 1
                # stride = self.stride(x, dim) * step
                sizes.append(size)
                steps.append(step)
                begins.append(start)
                ends.append(stop)
                dims.append(dim)
            if not sizes:
                sizes = [1]
                steps = [1]
            if isinstance(value, int) or isinstance(value, float):
                value = jt.full(sizes, value)
            self.type_ = 'slicev2'
            attr_code = f"""
            op.jt_name = "stridedsliceassignv2";
            StrideAttr *attr = new StrideAttr();
            attr->begins = {{ {", ".join(map(str, begins))} }};
            attr->ends = {{ {", ".join(map(str, ends))} }};
            attr->steps = {{ {", ".join(map(str, steps))} }};
            attr->axes = {{ {", ".join(map(str, dims))} }};
            op.op_attr.reset(attr);
            """
            self.value_shape = value.shape
            inputs = [value]
            outputs = [x.clone()]
            result = acl_cmd("StridedSliceAssignV2",
                             inputs=inputs,
                             outputs=outputs,
                             attr_code=attr_code)[0]
            if expand_dim:
                result = result.squeeze(-1)
            # result.sync()
            return result

        def grad(self, grad_output):
            value_grad = None
            if self.value_var:
                value_grad = grad_output[self.input_slice]
            grad_output[self.input_slice] = jt.zeros(self.value_shape)
            return grad_output, None, value_grad

    def setitem_acl(x, slices, value):
        res = SetItemACL()(x, slices, value)
        return x.assign(res)

    class BmmACL(Function):

        def __init__(self, trans_x2=False):
            super(BmmACL, self).__init__()
            self.trans_x2 = trans_x2

        def execute(self, x1, x2):
            self.input = [x1, x2]
            result = acl_cmd(
                "BatchMatMul", [x1, x2],
                output_dtypes=[x1.dtype],
                output_shapes=[
                    x1.shape[:-1] +
                    x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] +
                    x2.shape[-1:]
                ],
                attr_code="op.jt_name=\"bmm_trans_1\";"
                if self.trans_x2 else "op.jt_name=\"bmm\";")[0]

            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            if len(x1) != len(x2):
                reshape_grad_x2 = True
            else:
                reshape_grad_x2 = False
            grad_x1 = acl_cmd("BatchMatMul", [grad_output, x2],
                              output_dtypes=[x1.dtype],
                              output_shapes=[
                                  grad_output.shape[:-1] +
                                  x2.shape[-2:-1] if not self.trans_x2 else
                                  grad_output.shape[:-1] + x1.shape[-1:]
                              ],
                              attr_code="op.jt_name=\"bmm_trans_1\";" if
                              not self.trans_x2 else "op.jt_name=\"bmm\";")[0]
            if self.trans_x2:
                if reshape_grad_x2:
                    output_shape = grad_output.shape[1:-2] + grad_output.shape[
                        -1:] + x1.shape[-1:]
                    grad_x2 = acl_cmd(
                        "BatchMatMul", [
                            grad_output.reshape(-1, grad_output.shape[-1]),
                            x1.reshape(-1, x1.shape[-1])
                        ],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"bmm_trans_0\";")[0]
                else:
                    output_shape = grad_output.shape[:-2] + grad_output.shape[
                        -1:] + x1.shape[-1:]
                    grad_x2 = acl_cmd(
                        "BatchMatMul", [grad_output, x1],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"bmm_trans_0\";")[0]
            else:
                if reshape_grad_x2:
                    output_shape = x1.shape[1:-2] + x1.shape[
                        -1:] + grad_output.shape[-1:]
                    grad_x2 = acl_cmd(
                        "BatchMatMul", [
                            x1.reshape(-1, x1.shape[-1]),
                            grad_output.reshape(-1, grad_output.shape[-1])
                        ],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"bmm_trans_0\";")[0]
                else:
                    output_shape = x1.shape[:-2] + x1.shape[
                        -1:] + grad_output.shape[-1:]
                    grad_x2 = acl_cmd(
                        "BatchMatMul", [x1, grad_output],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"bmm_trans_0\";")[0]
            if len(grad_x1.shape) > len(x1.shape):
                grad_x1 = grad_x1.sum(0)
            if len(grad_x2.shape) > len(x2.shape):
                grad_x2 = grad_x2.sum(0)
            return grad_x1, grad_x2

    def bmm_acl(x1, x2):
        return BmmACL()(x1, x2)

    def bmm_transpose_acl(x1, x2):
        return BmmACL(True)(x1, x2)

    class MatmulACL(Function):

        def __init__(self, trans_x2=False):
            super(MatmulACL, self).__init__()
            self.trans_x2 = trans_x2

        def execute(self, x1, x2):
            self.input = [x1, x2]
            result = acl_cmd(
                "MatMul", [x1, x2],
                output_dtypes=[x1.dtype],
                output_shapes=[
                    x1.shape[:-1] +
                    x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] +
                    x2.shape[-1:]
                ],
                attr_code="op.jt_name=\"matmul_trans_1\";"
                if self.trans_x2 else "op.jt_name=\"matmul\";")[0]
            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            if len(x1) != len(x2):
                reshape_grad_x2 = True
            else:
                reshape_grad_x2 = False
            grad_x1 = acl_cmd(
                "MatMul", [grad_output, x2],
                output_dtypes=[x1.dtype],
                output_shapes=[
                    grad_output.shape[:-1] + x2.shape[-2:-1]
                    if not self.trans_x2 else grad_output.shape[:-1] +
                    x2.shape[-1:]
                ],
                attr_code="op.jt_name=\"matmul_trans_1\";"
                if not self.trans_x2 else "op.jt_name=\"matmul\";")[0]

            if self.trans_x2:
                if reshape_grad_x2:
                    output_shape = grad_output.shape[1:-2] + grad_output.shape[
                        -1:] + x1.shape[-1:]
                    grad_x2 = acl_cmd(
                        "MatMul", [
                            grad_output.reshape(-1, grad_output.shape[-1]),
                            x1.reshape(-1, x1.shape[-1])
                        ],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"matmul_trans_0\";")[0]
                else:
                    output_shape = grad_output.shape[:-2] + grad_output.shape[
                        -1:] + x1.shape[-1:]
                    grad_x2 = acl_cmd(
                        "MatMul", [grad_output, x1],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"matmul_trans_0\";")[0]
            else:
                if reshape_grad_x2:
                    output_shape = x1.shape[1:-2] + x1.shape[
                        -1:] + grad_output.shape[-1:]
                    grad_x2 = acl_cmd(
                        "MatMul", [
                            x1.reshape(-1, x1.shape[-1]),
                            grad_output.reshape(-1, grad_output.shape[-1])
                        ],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"matmul_trans_0\";")[0]
                else:
                    output_shape = x1.shape[:-2] + x1.shape[
                        -1:] + grad_output.shape[-1:]
                    grad_x2 = acl_cmd(
                        "MatMul", [x1, grad_output],
                        output_dtypes=[x2.dtype],
                        output_shapes=[output_shape],
                        attr_code="op.jt_name=\"matmul_trans_0\";")[0]
            return grad_x1, grad_x2

    def matmul_acl(x1, x2):
        return MatmulACL()(x1, x2)

    def matmul_transpose_acl(x1, x2):
        return MatmulACL(True)(x1, x2)

    class TransPoseACL(Function):

        def __init__(self):
            super(TransPoseACL, self).__init__()

        def execute(self, x, *dim):
            self.input = x
            if len(dim) == 1 and isinstance(dim[0], Sequence):
                dim = dim[0]
            elif len(dim) == 2:
                axes = list(range(x.ndim))
                a, b = dim
                axes[a], axes[b] = axes[b], axes[a]
                dim = axes

            attr_code = f"""
            op.jt_name = "transpose";
            ReduceAttr *attr = new ReduceAttr();
            attr->axes = {{ {", ".join(map(str, dim))} }};
            op.op_attr.reset(attr);
            """
            # calculate output shape
            output_shape = [x.shape[i] for i in dim]
            output = acl_cmd("Transpose", [x],
                             output_dtypes=[x.dtype],
                             output_shapes=[jt.empty(output_shape).shape],
                             attr_code=attr_code)[0]
            self.dim = dim
            return output

        def grad(self, grad_output):
            dim = list(range(grad_output.ndim))
            for i, p in enumerate(self.dim):
                dim[p] = i
            output_shape = [grad_output.shape[i] for i in dim]
            attr_code = f"""
            op.jt_name = "transpose";
            ReduceAttr *attr = new ReduceAttr();
            attr->axes = {{ {", ".join(map(str, dim))} }};
            op.op_attr.reset(attr);
            """
            output = acl_cmd("Transpose", [grad_output],
                             output_dtypes=[grad_output.dtype],
                             output_shapes=[jt.empty(output_shape).shape],
                             attr_code=attr_code)[0]
            return output

    def transpose_acl(x, *dim):
        return TransPoseACL()(x, *dim)

    class FlashAttentionACL(Function):

        def __init__(self,
                     headnum,
                     layout="BNSD",
                     prefix=None,
                     qstart=None,
                     kvstart=None,
                     scale=1.0,
                     prob=1.0,
                     pretokens=2147483647,
                     nexttokens=2147483647,
                     innerprecise=0,
                     sparsemode=0,
                     psetype=1):
            self.headnum = headnum
            self.layout = layout
            self.scale = scale
            self.prob = prob
            self.pretokens = pretokens
            self.nexttokens = nexttokens
            self.innerprecise = innerprecise
            self.sparsemode = sparsemode
            self.psetype = psetype
            self.prefix = prefix
            self.qstart = qstart
            self.kvstart = kvstart

        def execute(
            self,
            q,
            k,
            v,
            realshift=None,
            dropMask=None,
            paddingMask=None,
            attenMask=None,
        ):
            if self.layout == 'BSH':
                B, SQ, H = q.shape
                SKV = k.shape[1]
                N = self.headnum
                D = H / N
            elif self.layout == 'SBH':
                SQ, B, H = q.shape
                SKV = k.shape[0]
                N = self.headnum
                D = H / N
            elif self.layout == 'BSND':
                B, SQ, N, D = q.shape
                SKV = k.shape[1]
            elif self.layout == 'BNSD':
                B, N, SQ, D = q.shape
                SKV = k.shape[2]
            else:
                raise ValueError(f"got invalid input layout {self.layout}")

            output_shape = (B, N, SQ, 8)

            self.q = q
            self.k = k
            self.v = v

            self.prefix = self.prefix if self.prefix else [0 for _ in range(B)]
            self.qstart = self.qstart if self.qstart else [0 for _ in range(B)]
            self.kvstart = self.kvstart if self.kvstart else [
                0 for _ in range(B)
            ]

            self.hasRealshift = (not realshift == None)
            self.hasDropmask = (not dropMask == None)
            self.hasPaddingmask = (not paddingMask == None)
            self.hasAttenmask = (not attenMask == None)

            # 待定，目前设为nullptr
            self.realshift = realshift if realshift else jt.zeros(
                B, N, SQ, SKV)
            self.dropMask = dropMask if dropMask else jt.ones(B, N, SQ, SKV)
            self.paddingMask = paddingMask if paddingMask else jt.zeros(
                B, N, SQ, SKV)
            self.attenMask = attenMask if attenMask else jt.zeros(SQ, SKV)

            attr_code = f"""
            op.jt_name = "flashattention";
            FlashAttentionAttr *attr = new FlashAttentionAttr();
            attr->scale = {self.scale};
            attr->keepProb = {self.prob};
            attr->preToken = {self.pretokens};
            attr->nextToken = {self.nexttokens};
            attr->headNum = {self.headnum};
            attr->inputLayout = "{self.layout}";
            attr->innerPrecise = {self.innerprecise};
            attr->sparseMode = {self.sparsemode};
            attr->psetype = {self.psetype};
            attr->prefix = {{ {", ".join(map(str, self.prefix))} }};
            attr->qStartIdx = {{ {", ".join(map(str, self.qstart))} }};
            attr->kvStartIdx = {{ {", ".join(map(str, self.kvstart))} }};
            attr->hasRealshift = {"true" if self.hasRealshift else "false"};
            attr->hasDropmask = {"true" if self.hasDropmask else "false"};
            attr->hasPaddingmask = {"true" if self.hasPaddingmask else "false"};
            attr->hasAttentmask = {"true" if self.hasAttenmask else "false"};
            op.op_attr.reset(attr);
            """

            inputs = [
                q, k, v, self.realshift, self.dropMask, self.paddingMask,
                self.attenMask
            ]

            result = acl_cmd(
                "FlashAttention",
                inputs,
                output_dtypes=["float", "float", q.dtype],
                output_shapes=[output_shape, output_shape, q.shape],
                attr_code=attr_code)

            self.maxout = result[0]
            self.sumout = result[1]
            self.attenout = result[2]

            return self.attenout

        def grad(self, dy):
            attr_code = f"""
            op.jt_name = "flashattentionbackward";
            FlashAttentionAttr *attr = new FlashAttentionAttr();
            attr->scale = {self.scale};
            attr->keepProb = {self.prob};
            attr->preToken = {self.pretokens};
            attr->nextToken = {self.nexttokens};
            attr->headNum = {self.headnum};
            attr->inputLayout = "{self.layout}";
            attr->innerPrecise = {self.innerprecise};
            attr->sparseMode = {self.sparsemode};
            attr->psetype = {self.psetype};
            attr->prefix = {{ {", ".join(map(str, self.prefix))} }};
            attr->qStartIdx = {{ {", ".join(map(str, self.qstart))} }};
            attr->kvStartIdx = {{ {", ".join(map(str, self.kvstart))} }};
            attr->hasRealshift = {"true" if self.hasRealshift else "false"};
            attr->hasDropmask = {"true" if self.hasDropmask else "false"};
            attr->hasPaddingmask = {"true" if self.hasPaddingmask else "false"};
            attr->hasAttentmask = {"true" if self.hasAttenmask else "false"};
            op.op_attr.reset(attr);
            """
            inputs = [
                self.q, self.k, self.v, dy, self.realshift, self.dropMask,
                self.paddingMask, self.attenMask, self.maxout, self.sumout,
                self.attenout
            ]

            result = acl_cmd(
                "FlashAttentionBackward",
                inputs,
                output_dtypes=[self.q.dtype, self.k.dtype, self.v.dtype],
                output_shapes=[self.q.shape, self.k.shape, self.v.shape],
                attr_code=attr_code)
            return result

    class ReLUACL(Function):

        def __init__(self):
            super(ReLUACL, self).__init__()

        def execute(self, x):
            self.input = x
            result = acl_cmd("ReLU", [x],
                             output_dtypes=[x.dtype],
                             output_shapes=[x.shape],
                             attr_code="op.jt_name=\"unary\";")[0]
            return result

        def grad(self, grad_output):
            mask = acl_cmd("GreaterThan", [self.input, 0],
                           output_dtypes=[self.input.dtype],
                           output_shapes=[self.input.shape],
                           attr_code="op.jt_name=\"binary\";")[0]
            grad_input = acl_cmd("Multiply", [grad_output, mask],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code="op.jt_name=\"binary\";")[0]
            return grad_input

    class ReLU(jt.nn.Module):

        def __init__(self):
            super(ReLU, self).__init__()

        def execute(self, x):
            return ReLUACL()(x)

    def reluacl(x):
        return ReLUACL()(x)

    class LeakyReLUACL(Function):

        def __init__(self):
            super(LeakyReLUACL, self).__init__()

        def execute(self, x, negative_slope=0.01):
            self.input = x
            attr_code = f"""
            op.jt_name = "leakyrelu";
            LeakyReluAttr *attr = new LeakyReluAttr();
            attr->negativeSlope = {negative_slope};
            op.op_attr.reset(attr);
            """
            result = acl_cmd("LeakyReLU", [x],
                             output_dtypes=[x.dtype],
                             output_shapes=[x.shape],
                             attr_code=attr_code)[0]
            self.negative_slope = negative_slope
            return result

        def grad(self, grad_output):
            attr_code = f"""
            op.jt_name = "leakyrelubackward";
            LeakyReluAttr *attr = new LeakyReluAttr();
            attr->negativeSlope = {self.negative_slope};
            attr->selfIsResult = false;
            op.op_attr.reset(attr);
            """
            grad_input = acl_cmd("LeakyReLUBackward",
                                 [grad_output, self.input],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code=attr_code)[0]
            return grad_input

    class LeakyReLU(jt.nn.Module):

        def __init__(self, negative_slope=0.01):
            super(LeakyReLU, self).__init__()
            self.negative_slope = negative_slope

        def execute(self, x):
            return LeakyReLUACL()(x, self.negative_slope)

    def leaky_reluacl(x, scale=0.01):
        return LeakyReLUACL()(x, scale)

    class DropoutACL(Function):

        def __init__(self):
            super(DropoutACL, self).__init__()

        def execute(self, x, p=0.5, is_train=False):
            self.input = x
            num_elements = x.numel()
            aligned_elements = (num_elements + 127) // 128 * 128
            mask_shape = (aligned_elements // 8, )
            attr_code = f"""
            op.jt_name = "dropout";
            DropoutAttr *attr = new DropoutAttr();
            attr->p = {p};
            attr->train = {"true" if is_train else "false"};
            attr->seed = 0;
            attr->offset = 0;
            op.op_attr.reset(attr);
            """
            result = acl_cmd("Dropout", [x],
                             output_dtypes=[x.dtype, "uint8"],
                             output_shapes=[x.shape, mask_shape],
                             attr_code=attr_code)
            self.maskout = result[1]
            return result[0]

        def grad(self, grad_output):
            attr_code = f"""
            op.jt_name = "dropoutbackward";
            DropoutAttr *attr = new DropoutAttr();
            attr->scale = 1.0;
            op.op_attr.reset(attr);
            """
            grad_input = acl_cmd("DropoutBackward",
                                 [grad_output, self.maskout],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code=attr_code)[0]
            return grad_input

    class SiLUACL(Function):

        def __init__(self):
            super(SiLUACL, self).__init__()

        def execute(self, x):
            inputs = [x]
            self.input = x
            outputs = [jt.empty(x.shape, x.dtype)]
            attr_code = f"""
            op.jt_name = "silu";
            """
            result = acl_cmd("SiLU",
                             inputs=inputs,
                             outputs=outputs,
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            attr_code = f"""
            op.jt_name = "silubackward";
            """
            inputs = [grad_output, self.input]
            outputs = [jt.empty(grad_output.shape, grad_output.dtype)]
            grad_input = acl_cmd("SiLUBackward",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
            return grad_input

    class SiLU(jt.nn.Module):

        def __init__(self):
            super(SiLU, self).__init__()

        def execute(self, x):
            return SiLUACL()(x)

    class SigmoidACL(Function):

        def __init__(self):
            super(SigmoidACL, self).__init__()

        def execute(self, x):
            inputs = [x]
            outputs = [jt.empty(x.shape, x.dtype)]
            attr_code = f"""
            op.jt_name = "sigmoid";
            """
            result = acl_cmd("Sigmoid",
                             inputs=inputs,
                             outputs=outputs,
                             attr_code=attr_code)[0]
            self.output = result
            return result

        def grad(self, grad_output):
            attr_code = f"""
            op.jt_name = "sigmoidbackward";
            """
            inputs = [grad_output, self.output]
            outputs = [jt.empty(grad_output.shape, grad_output.dtype)]
            grad_input = acl_cmd("SigmoidBackward",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
            return grad_input

    class Sigmoid(jt.nn.Module):

        def __init__(self):
            super(Sigmoid, self).__init__()

        def execute(self, x):
            return SigmoidACL()(x)

    class EmbeddingACL(Function):

        def __init__(self):
            super(EmbeddingACL, self).__init__()

        def execute(
            self,
            indices,
            weight,
        ):
            inputs = [weight, indices]
            self.indices = indices
            self.weight_shape = weight.shape
            output_shape = list(indices.shape) + list(weight.shape[1:])
            outputs = [jt.empty(output_shape, weight.dtype)]
            attr_code = f"""
            op.jt_name = "embedding";
            """
            result = acl_cmd("Embedding",
                             inputs=inputs,
                             outputs=outputs,
                             attr_code=attr_code)[0]
            return result

        def grad(self, grad_output):
            inputs = [grad_output, self.indices]
            outputs = [jt.empty(self.weight_shape, grad_output.dtype)]
            attr_code = f"""
            op.jt_name = "embeddingbackward";
            EmbeddingAttr *attr = new EmbeddingAttr();
            attr->numEmbeddings = {self.weight_shape[0]};
            op.op_attr.reset(attr);
            """
            grad_weight = acl_cmd("EmbeddingBackward",
                                  inputs=inputs,
                                  outputs=outputs,
                                  attr_code=attr_code)[0]
            return None, grad_weight

    class Embedding(jt.nn.Module):

        def __init__(self,
                     num_embeddings,
                     embedding_dim,
                     padding_idx=None,
                     dtype="float32"):
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.weight = jt.init.gauss(
                [self.num_embeddings, self.embedding_dim], dtype)
            if padding_idx is not None:
                self.weight[padding_idx] = 0

        def execute(self, x):
            res = embedding(x, self.weight)
            return res

    class Dropout(jt.nn.Module):

        def __init__(self, p=0.5, is_train=False):
            super(Dropout, self).__init__()
            self.p = p
            self.is_train = is_train

        def execute(self, x):
            return DropoutACL()(x, self.p, self.is_train)

    def dropoutacl(x, p=0.5, is_train=False):
        return DropoutACL()(x, p, is_train)

    def warp(origin_func, new_func, name=None):

        def warpper(*args, **kwargs):
            if jt.flags.use_acl:
                return new_func(*args, **kwargs)
            if name == 'setitem':
                result = args[0].assign(origin_func(*args, **kwargs))
            else:
                result = origin_func(*args, **kwargs)
            return result

        if isinstance(origin_func, type):
            class WrappedClass(origin_func):
                def __init__(self, *args, **kwargs):
                    super(WrappedClass, self).__init__(*args, **kwargs)

                def __call__(self, *args, **kwargs):
                    return warpper(*args, **kwargs)

            return WrappedClass

        else:
            return warpper

    jt.triu = warp(jt.triu, triu_acl)
    jt.triu_ = warp(jt.triu, triu_acl)
    jt.Var.triu = lambda x: jt.triu(x)
    jt.Var.triu_ = lambda x: jt.triu_(x)
    jt.nn.conv2d = warp(jt.nn.conv2d, ConvACL())
    jt.nn.Conv2d = warp(jt.nn.Conv2d, Conv2D)
    jt.nn.Conv = warp(jt.nn.Conv, Conv2D)
    jt.nn.Pool = warp(jt.nn.Pool, PoolACL)

    jt.flip = warp(jt.flip, flip_acl)
    jt.Var.flip = lambda x, dim_vector=0: jt.flip(x, dim_vector)
    jt.concat = warp(jt.concat, concat)

    jt.gather = warp(jt.gather, gather_acl)

    jt.cumsum = warp(jt.cumsum, cumsum_acl)
    jt.index = warp(jt.index, index_acl)
    jt.Var.index = lambda x, dim=None: jt.index(x.shape, dim)

    jt.scatter = warp(jt.scatter, scatter_acl)
    jt.Var.scatter = lambda x, dim, index, src, reduce="void": jt.scatter(
        x, dim, index, src, reduce)

    jt.floor_int = warp(jt.floor_int, floor_int_acl)
    jt.Var.floor_int = lambda x: jt.floor_int(x)

    jt.getitem = warp(jt.contrib.getitem, getitem_acl)
    fake_getitem = jt.Var.getitem
    jt.Var.getitem = lambda x, slices, return_x=None: warp(
        fake_getitem, getitem_acl)(x, slices)
    jt.Var.slice_var = lambda x, slices, return_x=None: warp(
        fake_getitem, getitem_acl)(x, slices)
    jt.Var.__getitem__ = lambda x, slices, return_x=None: warp(
        fake_getitem, getitem_acl)(x, slices)

    jt.setitem = warp(jt.contrib.setitem, setitem_acl)
    fake_setitem = jt.Var.setitem
    jt.Var.setitem = lambda x, slices, value: warp(
        fake_setitem, setitem_acl, name='setitem')(x, slices, value)
    jt.Var.__setitem__ = lambda x, slices, value: warp(
        fake_setitem, setitem_acl, name='setitem')(x, slices, value)

    jt.nn.bmm = warp(jt.nn.bmm, bmm_acl)
    jt.bmm = warp(jt.bmm, bmm_acl)
    jt.nn.matmul = warp(jt.matmul, matmul_acl)
    jt.matmul = warp(jt.matmul, matmul_acl)
    jt.nn.matmul_transpose = warp(jt.nn.matmul_transpose, matmul_transpose_acl)
    jt.nn.bmm_transpose = warp(jt.nn.bmm_transpose, bmm_transpose_acl)
    jt.bmm_transpose = warp(jt.bmm_transpose, bmm_transpose_acl)

    jt.transpose = warp(jt.transpose, transpose_acl)
    fake_transpose = jt.transpose
    jt.Var.transpose = lambda x, *dim: warp(fake_transpose, transpose_acl)(x, *dim)
    # jt.Var.permute = lambda x: warp(fake_transpose, transpose_acl)(x)
    # jt.Var.t = lambda x: warp(fake_transpose, transpose_acl)(x)

    # jt.nn.relu = warp(jt.nn.relu, relu)
    # jt.nn.ReLU = warp(jt.nn.ReLU, ReLU)

    # jt.nn.leaky_relu = warp(jt.nn.leaky_relu, leaky_relu)
    # jt.nn.LeakyReLU = warp(jt.nn.LeakyReLU, LeakyReLU)

    def silu(x):
        return SiLUACL()(x)

    # jt.nn.silu = warp(jt.nn.silu, silu)
    # jt.nn.SiLU = warp(jt.nn.SiLU, SiLU)

    def sigmoid(x):
        return SigmoidACL()(x)

    # jt.sigmoid = warp(jt.sigmoid, sigmoid)
    # jt.nn.Sigmoid = warp(jt.nn.Sigmoid, Sigmoid)

    def embedding(indices, weight):
        return EmbeddingACL()(indices, weight)

    # jt.nn.embedding = warp(jt.nn.embedding, embedding)
    # jt.nn.Embedding = warp(jt.nn.Embedding, Embedding)
    # jt.nn.dropout = warp(jt.nn.dropout, dropout)
    # jt.nn.Dropout = warp(jt.nn.Dropout, Dropout)

    jt.nn.FlashAttention = warp(jt.nn.FlashAttention, FlashAttentionACL)
