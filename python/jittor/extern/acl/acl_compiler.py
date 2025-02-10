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
import numpy as np

from typing import Union
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
        # Skip files in hccl directory
        if "hccl" in name:
            continue
        # if "acl_op_exec" in name or "_op_acl.cc" in name:
        if "acl_op_exec" in name or "_op_acl.cc" in name or "utils.cc" in name:
            compiler.extra_core_files.append(name)
        else:
            cc_files2.append(name)
    cc_files = cc_files2
    ascend_toolkit_home = os.getenv('ASCEND_TOOLKIT_HOME')

    #print(ascend_toolkit_home)
    #print(acl_compiler_home)
    cc_flags += f" -MD -DHAS_CUDA -DIS_ACL  \
    -I{ascend_toolkit_home}/include/ \
    -I{ascend_toolkit_home}/include/acl/ \
    -I{ascend_toolkit_home}/include/aclnn/ \
    -I{ascend_toolkit_home}/include/aclnnop/ \
    -I{acl_compiler_home} -lascendcl -lacl_op_compiler \
    -I{acl_compiler_home}/aclnn \
    -I{acl_compiler_home}/aclops \
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

def change_function():
    import jittor as jt
    from jittor import Function
    from .aclops.flashattention_op import FlashAttentionACL
    from .aclops.conv_op import ConvACL
    from .aclops.pool_op import PoolACL
    from .aclops.nantonum_op import NanToNumACL
    from .aclops.stack_op import StackACL
    from .aclops.rope_op import RopeACL
    from .aclops.softmax_op import SoftmaxACL
    from .aclops.sigmoid_op import SigmoidACL
    from .aclops.silu_op import SiLUACL
    from .aclops.dropout_op import DropoutACL
    from .aclops.relu_op import LeakyReLUACL
    from .aclops.flip_op import FlipACL
    from .aclops.concat_op import ConcatACL
    from .aclops.gather_scatter_op import GatherACL
    from .aclops.cumsum_op import CumsumACL
    from .aclops.index_op import IndexACL
    from .aclops.gather_scatter_op import ScatterACL
    from .aclops.where_op import WhereACL
    from .aclops.where_op import NonzeroACL
    from .aclops.floor_op import FloorIntACL
    from .aclops.getitem_op import GetItemACL
    from .aclops.setitem_op import SetItemACL
    from .aclops.bmm_op import BmmACL
    from .aclops.matmul_op import MatmulACL
    from .aclops.transpose_op import TransPoseACL

    from .aclops.triu_op import TriuACL

    def triu_acl(x, diagonal=0):
        return TriuACL()(x, diagonal)

    from .aclops.conv_op import ConvACL

    def conv_acl(x,
                 weight,
                 bias=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        return ConvACL()(x, weight, bias, stride, padding, dilation, groups)

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


    from .aclops.flip_op import FlipACL
    def flip_acl(x, dim):
        return FlipACL()(x, dim)

    from .aclops.concat_op import ConcatACL
    def concat(x, dim=0):
        return ConcatACL()(x, dim)

    from .aclops.gather_scatter_op import GatherACL

    def gather_acl(input, dim, index):
        return GatherACL()(input, dim, index)

    def any_acl(input):
        if jt.sum(input != 0).item() > 0:
            return jt.array([True])
        else:
            return jt.array([False])

    from .aclops.cumsum_op import CumsumACL

    def cumsum_acl(input, dim=-1):
        return CumsumACL()(input, dim)

    def cumprod_acl(x, dim=None):
        x = jt.log(x)
        x = cumsum_acl(x, dim=dim)
        return jt.exp(x)

    from .aclops.index_op import IndexACL

    def index_acl(inshape: Union[jt.Var, list], dim=None, dtype="int32"):
        if isinstance(inshape, jt.Var):
            inshape = inshape.shape
        return IndexACL()(inshape, dim, dtype)

    from .aclops.gather_scatter_op import ScatterACL
    def scatter_acl(input, dim, index, src, reduce='void'):
        return ScatterACL()(input, dim, index, src, reduce)

    from .aclops.where_op import WhereACL

    def where_acl(condition, x=None, y=None):
        return WhereACL()(condition, x, y)

    from .aclops.where_op import NonzeroACL

    def nonzero_acl(x):
        return NonzeroACL()(x)

    from .aclops.floor_op import FloorIntACL

    def floor_int_acl(x):
        return FloorIntACL()(x)

    from .aclops.getitem_op import GetItemACL

    def getitem_acl(x, slices, return_x=None):
        # Transform numpy int to int
        if isinstance(slices, (np.int8, np.int16, np.int32, np.int64)):
            slices = int(slices)
        if hasattr(np, 'int128') and isinstance(slices, np.int128):
            slices = int(slices)
        if hasattr(np, 'int256') and isinstance(slices, np.int256):
            slices = int(slices)

        ## If not related to `None`, directly use `GetItemACL`
        if slices is not None and (not isinstance(slices, Iterable)
                                   or all([s is not None for s in slices])):
            return GetItemACL()(x, slices, return_x)

        ## If related to `None`, filter out `None` first, then use `GetItemACL`, and finally insert `None` (new dimensions) back

        # Transform to tuple
        if isinstance(slices, int) or isinstance(slices, slice):
            slices = (slices, )
        assert isinstance(slices, tuple)

        def get_insert_positions(slices):
            result = []
            pos = 0

            not_none_cnt = len(slices) - slices.count(None)
            for s in slices:
                if isinstance(s, int):
                    continue
                elif s is None:
                    result.append(pos)
                    pos += 1
                elif s == Ellipsis:
                    pos += 1 + x.ndim - not_none_cnt
                else:
                    pos += 1

            return result

        insert_positions = get_insert_positions(slices)
        slices_without_none = tuple(s for s in slices if s is not None)
        result = GetItemACL()(x, slices_without_none, return_x)

        for i in insert_positions:
            result = result.unsqueeze(i)

        return result


    from .aclops.setitem_op import SetItemACL

    def setitem_acl(x, slices, value):
        res = SetItemACL()(x, slices, value)
        return x.assign(res)


    from .aclops.bmm_op import BmmACL

    def bmm_acl(x1, x2):
        return BmmACL()(x1, x2)

    def bmm_transpose_acl(x1, x2):
        return BmmACL(True)(x1, x2)


    from .aclops.matmul_op import MatmulACL

    def matmul_acl(x1, x2):
        return MatmulACL()(x1, x2)

    def matmul_transpose_acl(x1, x2):
        return MatmulACL(True)(x1, x2)

    from .aclops.transpose_op import TransPoseACL

    def transpose_acl(x, *dim):
        return TransPoseACL()(x, *dim)

    from .aclops.relu_op import ReLUACL
    class ReLU(jt.nn.Module):

        def __init__(self):
            super(ReLU, self).__init__()

        def execute(self, x):
            return ReLUACL()(x)

    def relu(x):
        return ReLUACL()(x)

    from .aclops.relu_op import LeakyReLUACL

    class LeakyReLU(jt.nn.Module):

        def __init__(self, negative_slope=0.01):
            super(LeakyReLU, self).__init__()
            self.negative_slope = negative_slope

        def execute(self, x):
            return LeakyReLUACL()(x, self.negative_slope)

    def leaky_relu(x, scale=0.01):
        return LeakyReLUACL()(x, scale)

    from .aclops.dropout_op import DropoutACL

    class Dropout(jt.nn.Module):

        def __init__(self, p=0.5, is_train=False):
            super(Dropout, self).__init__()
            self.p = p
            self.is_train = is_train

        def execute(self, x):
            return DropoutACL()(x, self.p, self.is_train)

    def dropout_acl(x, p=0.5, is_train=False):
        return DropoutACL()(x, p, is_train)

    from .aclops.silu_op import SiLUACL

    def silu_acl(x):
        return SiLUACL()(x)

    class SiLU(jt.nn.Module):

        def __init__(self):
            super(SiLU, self).__init__()

        def execute(self, x):
            return SiLUACL()(x)

    from .aclops.sigmoid_op import SigmoidACL
    
    def sigmoid_acl(x):
        return SigmoidACL()(x)

    class Sigmoid(jt.nn.Module):

        def __init__(self):
            super(Sigmoid, self).__init__()

        def execute(self, x):
            return SigmoidACL()(x)

    # class Embedding(jt.nn.Module):

    #     def __init__(self,
    #                  num_embeddings,
    #                  embedding_dim,
    #                  padding_idx=None,
    #                  dtype="float32"):
    #         self.num_embeddings = num_embeddings
    #         self.embedding_dim = embedding_dim
    #         self.padding_idx = padding_idx
    #         self.weight = jt.init.gauss(
    #             [self.num_embeddings, self.embedding_dim], dtype)
    #         if padding_idx is not None:
    #             self.weight[padding_idx] = 0

    #     def execute(self, x):
    #         res = embedding_acl(x, self.weight)
    #         return res
    
    class Softmax(jt.nn.Module):

        def __init__(self):
            super(Softmax, self).__init__()

        def execute(self, x, dim):
            return SoftmaxACL()(x, dim)

    def softmax_acl(x, dim):
        return SoftmaxACL()(x, dim)

    from .aclops.rope_op import RopeACL
    def rope_acl(xq, xk, freqs_cis=None, freq_sin=None, freq_cos=None):
        return RopeACL()(xq, xk, freqs_cis, freq_sin, freq_cos)

    from .aclops.stack_op import StackACL
    def stack_acl(x, dim=0):
        return StackACL()(x, dim)

    from .aclops.nantonum_op import NanToNumACL
    
    def isnan_acl(x):
        tonum = NanToNumACL()(x, -1.0)
        return jt.not_equal(x, tonum).logical_and(
            jt.not_equal(tonum, jt.ones_like(x)))

    def isinf_acl(x):
        tonum = NanToNumACL()(x, 1.0)
        return jt.not_equal(x, tonum).logical_and(
            jt.not_equal(tonum, jt.ones_like(x)))

    def warp(origin_func, new_func, name=None):

        if isinstance(origin_func, type):

            class WrappedClass(origin_func, new_func):

                def __init__(self, *args, **kwargs):
                    if jt.flags.use_acl:
                        new_func.__init__(self, *args, **kwargs)
                    else:
                        origin_func.__init__(self, *args, **kwargs)

                def execute(self, *args, **kwargs):
                    if jt.flags.use_acl:
                        return new_func.execute(self, *args, **kwargs)
                    elif name == 'setitem':
                        return args[0].assign(origin_func(*args, **kwargs))
                    else:
                        return origin_func.execute(self, *args, **kwargs)

            return WrappedClass

        else:

            def warpper(*args, **kwargs):
                if jt.flags.use_acl:
                    return new_func(*args, **kwargs)
                elif name == 'setitem':
                    return args[0].assign(origin_func(*args, **kwargs))
                else:
                    return origin_func(*args, **kwargs)

            return warpper

    jt.triu = warp(jt.triu, triu_acl)
    jt.triu_ = warp(jt.triu, triu_acl)
    jt.Var.triu = jt.triu
    jt.Var.triu_ = lambda x, diagonal=0: x.assign(x.triu(diagonal))
    jt.nn.conv2d = warp(jt.nn.conv2d, conv_acl)
    jt.nn.Conv2d = warp(jt.nn.Conv2d, Conv2D)
    jt.nn.Conv = warp(jt.nn.Conv, Conv2D)

    jt.nn.Pool = warp(jt.nn.Pool, PoolACL)

    jt.flip = warp(jt.flip, flip_acl)
    jt.Var.flip = lambda x, dim_vector=0: jt.flip(x, dim_vector)
    jt.concat = warp(jt.concat, concat)
    jt.stack = warp(jt.stack, stack_acl)

    jt.gather = warp(jt.gather, gather_acl)
    jt.any = warp(jt.any, any_acl)
    jt.Var.any = jt.any

    jt.cumsum = warp(jt.cumsum, cumsum_acl)
    jt.cub_cumsum = jt.cumsum
    jt.Var.cumsum = jt.cumsum
    jt.Var.cub_cumsum = jt.cumsum

    jt.cumprod = warp(jt.cumprod, cumprod_acl)
    jt.index = warp(jt.index, index_acl)
    jt.Var.index = jt.index

    jt.scatter = warp(jt.scatter, scatter_acl)
    jt.Var.scatter = lambda x, dim, index, src, reduce="void": jt.scatter(
        x, dim, index, src, reduce)

    jt.where = warp(jt.where, where_acl)
    jt.nonzero = warp(jt.nonzero, nonzero_acl)
    jt.misc.nonzero = warp(jt.misc.nonzero, nonzero_acl)
    jt.Var.nonzero = jt.misc.nonzero
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

    fake_matmul = jt.Var.matmul
    jt.nn.bmm = warp(jt.nn.bmm, bmm_acl)
    jt.bmm = warp(jt.bmm, bmm_acl)
    jt.nn.matmul = warp(jt.matmul, matmul_acl)
    jt.matmul = warp(jt.matmul, matmul_acl)
    jt.nn.matmul_transpose = warp(jt.nn.matmul_transpose, matmul_transpose_acl)
    jt.nn.bmm_transpose = warp(jt.nn.bmm_transpose, bmm_transpose_acl)
    jt.bmm_transpose = warp(jt.bmm_transpose, bmm_transpose_acl)
    jt.Var.__matmul__ = lambda x, y: warp(fake_matmul, matmul_acl)(x, y)

    jt.transpose = warp(jt.transpose, transpose_acl)
    fake_transpose = jt.transpose
    jt.Var.transpose = lambda x, *dim: warp(fake_transpose, transpose_acl)(x, *
                                                                           dim)
    # jt.Var.permute = lambda x: warp(fake_transpose, transpose_acl)(x)
    # jt.Var.t = lambda x: warp(fake_transpose, transpose_acl)(x)

    jt.nn.relu = warp(jt.nn.relu, relu)
    jt.nn.ReLU = warp(jt.nn.ReLU, ReLU)

    jt.nn.leaky_relu = warp(jt.nn.leaky_relu, leaky_relu)
    jt.nn.LeakyReLU = warp(jt.nn.LeakyReLU, LeakyReLU)

    # jt.nn.silu = warp(jt.nn.silu, silu_acl)
    # jt.nn.SiLU = warp(jt.nn.SiLU, SiLU)

    jt.sigmoid = warp(jt.sigmoid, sigmoid_acl)
    jt.nn.Sigmoid = warp(jt.nn.Sigmoid, Sigmoid)

    # from .aclops.embedding_op import EmbeddingACL
    # def embedding_acl(indices, weight):
    #     return EmbeddingACL()(indices, weight)

    # jt.nn.embedding = warp(jt.nn.embedding, embedding_acl)
    # jt.nn.Embedding = warp(jt.nn.Embedding, Embedding)
    jt.nn.dropout = warp(jt.nn.dropout, dropout_acl)
    jt.nn.Dropout = warp(jt.nn.Dropout, Dropout)

    jt.nn.softmax = warp(jt.nn.softmax, softmax_acl)

    # from .aclops.norms_op import BatchNormACL,LayerNormACL
    # jt.nn.BatchNorm = warp(jt.nn.BatchNorm, BatchNormACL)
    # jt.nn.LayerNorm = warp(jt.nn.LayerNorm, LayerNormACL)

    jt.nn.FlashAttention = warp(jt.nn.FlashAttention, FlashAttentionACL)
    jt.isnan = warp(jt.isnan, isnan_acl)
    jt.isinf = warp(jt.isinf, isinf_acl)
    jt.Var.isnan = jt.isnan
    jt.Var.isinf = jt.isinf

    jt.nn.rotary_emb = rope_acl
