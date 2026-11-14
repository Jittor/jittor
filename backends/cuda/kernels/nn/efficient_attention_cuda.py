"""Directly compile the upstream CUTLASS memory-efficient CUDA attention kernels.

No native Torch package is imported or linked. The optional source roots are
explicit: JITTOR_EFFICIENT_ATTN_SRC (a PyTorch include tree or source checkout)
and JITTOR_CUTLASS_PATH (CUTLASS's root or include directory).
"""
import ctypes
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import subprocess

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._core.flags import _output_requires_grad
from jittor._runtime.dispatch import register_kernel

_HERE = Path(__file__).resolve().parent
_MEMORY = Path('ATen/native/transformers/cuda/mem_eff_attention')
_DTYPES = {'float32': 0, 'float16': 1, 'bfloat16': 2}


class _Metadata(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        'batches', 'heads', 'queries', 'keys', 'head_dim', 'value_dim', 'causal', 'dtype')]
    _fields_ += [(name, ctypes.c_int64) for name in (
        'bias_batch_stride', 'bias_head_stride', 'bias_row_stride')]
    _fields_ += [('scale', ctypes.c_float), ('dropout', ctypes.c_float)]


def _source_roots():
    raw = os.environ.get('JITTOR_EFFICIENT_ATTN_SRC')
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    source = next((base for base in (root, root/'include', root/'aten/src')
                   if (base/_MEMORY/'kernel_forward.h').is_file()), None)
    if source is None:
        raise RuntimeError('JITTOR_EFFICIENT_ATTN_SRC must contain the upstream CUDA attention headers')
    raw_cutlass = os.environ.get('JITTOR_CUTLASS_PATH')
    root_cutlass = Path(raw_cutlass).expanduser().resolve() if raw_cutlass else root/'third_party/cutlass'
    cutlass = next((base for base in (root_cutlass, root_cutlass/'include')
                    if (base/'cutlass/cutlass.h').is_file()), None)
    if cutlass is None:
        raise RuntimeError('Set JITTOR_CUTLASS_PATH to the CUTLASS include tree')
    return source, cutlass


def _hash_tree(digest, root):
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())


@functools.lru_cache(maxsize=None)
def _library(source, cutlass, architectures=None):
    """Compile once per source/toolchain identity, propagating build failures."""
    import pybind11
    from jittor.build import compiler
    import sysconfig

    # Resolve the compatibility headers from this checkout, without a Torch import.
    checkout = Path(jt.__file__).resolve().parents[2]
    shim = checkout/'compat/shim/cpp_extension/include'
    if not shim.is_dir():
        shim = Path(jt.__file__).resolve().parent/'compat/shim/cpp_extension/include'
    digest = hashlib.sha256()
    for path in (_HERE/'efficient_attention.cu', _HERE/'efficient_attention.h'):
        digest.update(path.read_bytes())
    for path in (Path(source)/_MEMORY, Path(cutlass)/'cutlass', shim):
        _hash_tree(digest, path)
    architectures = architectures or tuple(int(arch) for arch in jt.flags.cuda_archs)
    identity = (compiler.nvcc_path, compiler.cc_path, architectures, source, cutlass)
    digest.update(repr(identity).encode())
    directory = Path(jt.flags.cache_path)/'efficient_attention'/digest.hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'kernels.so'
    with (directory/'build.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = directory/('kernels.%d.so' % os.getpid())
            includes = [str(_HERE), source, cutlass, str(shim), pybind11.get_include(),
                        sysconfig.get_path('include')]
            command = [compiler.nvcc_path, '-std=c++17', '-O3', '-shared',
                       '-Xcompiler', '-fPIC', '--expt-relaxed-constexpr', '--extended-lambda',
                       '--compiler-bindir', compiler.cc_path, '-w']
            for arch in architectures:
                command += ['-gencode', 'arch=compute_%d,code=sm_%d' % (arch, arch)]
            command += ['-I'+path for path in includes]
            command += [str(_HERE/'efficient_attention.cu'), '-o', str(temporary)]
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError('Building CUTLASS efficient attention failed:\n'+done.stderr[-12000:])
            os.replace(temporary, target)
    library = ctypes.CDLL(str(target))
    library.jt_efficient_attention_workspace.argtypes = [ctypes.POINTER(_Metadata)]
    library.jt_efficient_attention_workspace.restype = ctypes.c_int64
    library.jt_efficient_attention_error.restype = ctypes.c_char_p
    return library


def _supports(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, enable_gqa=False, **kwargs):
    dtype = dtype_name(query.dtype)
    if dtype not in _DTYPES or not all(dtype_name(x.dtype) == dtype and len(x.shape) == 4
                                     for x in (query, key, value)):
        return False
    if dtype == 'bfloat16' and not kwargs.get('allow_reduced_precision', False):
        return False
    if not 0 <= float(dropout_p) < 1:
        return False
    architectures = tuple(int(arch) for arch in jt.flags.cuda_archs)
    if not architectures or min(architectures) < 80:
        return False
    b, h, m, d = map(int, query.shape)
    if tuple(key.shape[:2]) != (b, h) or tuple(value.shape[:3]) != tuple(key.shape[:3]):
        return False
    if key.shape[-1] != d or not all(0 < int(x) for x in (*query.shape, *key.shape, *value.shape)):
        return False
    if any(int(dim) > 128 or int(dim) % 8 for dim in (d, value.shape[-1])):
        return False
    if attn_mask is not None:
        if (attn_mask.requires_grad or len(attn_mask.shape) > 4
                or dtype_name(attn_mask.dtype) not in ('bool', dtype)):
            return False
        shape = (1,)*(4-len(attn_mask.shape))+tuple(attn_mask.shape)
        target = (b, h, m, int(key.shape[-2]))
        if any(a not in (1, expected) for a, expected in zip(shape, target)):
            return False
    return _source_roots() is not None


def _bias(mask, batch, heads, queries, keys, dtype):
    if mask is None:
        return None, (0, 0, 0)
    shape = (1,)*(4-len(mask.shape))+tuple(mask.shape)
    bias = mask.reshape(shape)
    if dtype_name(bias.dtype) == 'bool':
        bias = jt.ternary(bias, jt.zeros_like(bias, dtype='float32'),
                          jt.full_like(bias, float('-inf'), dtype='float32'))
    else:
        bias = bias.float32()
    if shape[-1] == 1:
        bias = bias.broadcast((*shape[:-1], keys))
    padded = (keys+7)//8*8
    if padded != keys:
        bias = jt.nn.pad(bias, (0, padded-keys))
    bias = bias.cast(dtype).contiguous().stop_grad()
    sizes = tuple(bias.shape)
    strides = (sizes[1]*sizes[2]*padded, sizes[2]*padded, padded)
    return bias, tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides))


class _Attention(jt.Function):
    def __init__(self, metadata, bias, library, seed):
        self.metadata, self.bias, self.library, self.seed = metadata, bias, library, seed

    def _source(self):
        a = self.metadata
        values = [a.batches, a.heads, a.queries, a.keys, a.head_dim, a.value_dim, a.causal,
                  a.dtype, a.bias_batch_stride, a.bias_head_stride, a.bias_row_stride]
        # Construction constants only; tensor values stay device inputs.
        functions = ''
        for alias, name in (('forward', 'jt_efficient_attention_forward'),
                             ('backward', 'jt_efficient_attention_backward'),
                             ('error', 'jt_efficient_attention_error')):
            address = ctypes.cast(getattr(self.library, name), ctypes.c_void_p).value
            functions += 'auto %s = reinterpret_cast<decltype(&%s)>(uintptr_t(%dULL));\n' % (
                alias, name, address)
        return (functions+'JtEfficientAttention args = {'+','.join(map(str, values))+', '
                +repr(float(a.scale))+'f, '+repr(float(a.dropout))+'f};\n'
                'int device; CHECK(cudaGetDevice(&device) == cudaSuccess);\n'
                'auto stream = jittor::cuda_compute_stream(device);\n')

    def execute(self, query, key, value):
        a = self.metadata
        inputs = [query, key, value]+([] if self.bias is None else [self.bias])
        bias = 'nullptr' if self.bias is None else 'in3_p'
        seed = 'nullptr'
        if self.seed is not None:
            seed = 'reinterpret_cast<const int64_t*>(in%d_p)' % len(inputs)
            inputs.append(self.seed)
        output_shape = (a.batches, a.queries, a.heads, a.value_dim)
        output, lse, _accumulator, saved_seed = jt.code(
                             [output_shape, (a.batches, a.heads, (a.queries+31)//32*32),
                              output_shape, (2,)],
                             [query.dtype, 'float32', 'float32', 'int64'], inputs,
                             cuda_header='#include "'+str(_HERE/'efficient_attention.h')+'"\n#include "stream_compat.h"',
                             cuda_src=self._source()+
                             'CHECK(forward(&args, in0_p, in1_p, in2_p, '+bias+
                             ', out0_p, out1_p, out2_p, '+seed+
                             ', reinterpret_cast<int64_t*>(out3_p), stream) == 0) << error();')
        self.saved = query, key, value, output, lse, saved_seed
        return output

    def grad(self, grad_output):
        query, key, value, output, lse, saved_seed = self.saved
        a = self.metadata
        workspace_size = self.library.jt_efficient_attention_workspace(ctypes.byref(a))
        if workspace_size < 0:
            raise RuntimeError(self.library.jt_efficient_attention_error().decode())
        workspace = jt.empty((max(1, (workspace_size+3)//4),), dtype='float32').stop_grad()
        inputs = [query, key, value, output, lse, grad_output.cast(query.dtype).contiguous(), workspace,
                  saved_seed]
        if self.bias is not None:
            inputs.append(self.bias)
        bias = 'nullptr' if self.bias is None else 'in8_p'
        result = jt.code([tuple(query.shape), tuple(key.shape), tuple(value.shape),
                          (a.batches, a.heads, a.queries)], [query.dtype]*3+['float32'], inputs,
                         cuda_header='#include "'+str(_HERE/'efficient_attention.h')+'"\n#include "stream_compat.h"',
                         cuda_src=self._source()+
                         'CHECK(backward(&args, in0_p, in1_p, in2_p, '+bias+
                         ', in3_p, in4_p, in5_p, out0_p, out1_p, out2_p, out3_p, in6_p, '
                         'reinterpret_cast<const int64_t*>(in7_p), stream) == 0) '
                         '<< error();')
        for gradient in result[:3]:
            gradient._set_first_order_only()
        return result[:3]


def _efficient_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False, **kwargs):
    if not _supports(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, **kwargs):
        return None
    source, cutlass = _source_roots()
    library = _library(str(source), str(cutlass), tuple(int(arch) for arch in jt.flags.cuda_archs))
    b, h, m, d = map(int, query.shape)
    n, dv = int(key.shape[-2]), int(value.shape[-1])
    dtype = dtype_name(query.dtype)
    bias, strides = _bias(attn_mask, b, h, m, n, dtype)
    metadata = _Metadata(b, h, m, n, d, dv, int(is_causal), _DTYPES[dtype], *strides,
                         float(d**-.5 if scale is None else scale), float(dropout_p))
    with jt.flag_scope(amp_reg=0):
        seed = ((jt.random((2,))*2147483647.).int64().stop_grad() if dropout_p else None)
    operands = [jt.transpose(x, (0, 2, 1, 3)).contiguous() for x in (query, key, value)]
    operation = _Attention(metadata, bias, library, seed)
    output = (operation(*operands) if _output_requires_grad(query, key, value)
              else operation.execute(*operands))
    return jt.transpose(output, (0, 2, 1, 3))


register_kernel('nn.fused_attention', 'cuda', _efficient_attention,
                supports=_supports, priority=20)
register_kernel('nn.memory_efficient_attention', 'cuda', _efficient_attention,
                supports=_supports, priority=20)
