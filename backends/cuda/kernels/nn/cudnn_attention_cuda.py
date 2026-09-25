"""cuDNN's fused attention as the CUDA ``nn.fused_attention`` kernel.

The math path of ``scaled_dot_product_attention`` builds the whole
``[..., Lq, Lk]`` score matrix and its softmax in device memory. At the SD1.5
UNet's 64x64 resolution that is 1 GiB per call and a fifth of the device time
of a UNet step, where PyTorch runs a fused kernel that never writes the scores
out. cuDNN 9 ships such a kernel -- forward and backward, float16 and
bfloat16, causal masks, additive biases -- behind its frontend graph API.

The frontend is header-only C++17, and Jittor's JIT compiles C++14, so the
graph code lives in ``cudnn_sdpa.cc``, built once into a small shared library
with a C interface; the ``jt.code`` operators below only call into it. It needs
the ``nvidia-cudnn-frontend`` wheel for its headers. Without it, or where
cuDNN answers that it cannot run a shape, this kernel declines and attention
takes the path it took before.
"""

import ctypes
import hashlib
import importlib.util
import os
import subprocess
import threading

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core.flags import _output_requires_grad
from jittor._runtime.dispatch import register_kernel

_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cudnn_sdpa.cc")
_DESCRIPTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                           "libraries", "cudnn", "include", "cudnn_descriptor.h")
_DTYPES = {"float16": 0, "bfloat16": 1}

_lock = threading.Lock()
_state = {}
_supported_shapes = {}


def _frontend_include():
    """The cudnn-frontend header directory, or None when it is not installed."""
    spec = importlib.util.find_spec("cudnn")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        include = os.path.join(os.path.dirname(location), "include")
        if os.path.exists(os.path.join(include, "cudnn_frontend.h")):
            return include
    return None


def _include_dirs(frontend):
    from jittor.build import compiler
    dirs = [frontend]
    if compiler.cuda_wheel_stack is not None:
        dirs += compiler.cuda_wheel_stack.include_dirs()
    dirs += [compiler.cuda_include,
             os.path.join(compiler.cuda_include, "..", "targets", "x86_64-linux", "include")]
    return [d for d in dirs if d and os.path.isdir(d)]


def _build(frontend):
    """Compile ``cudnn_sdpa.cc`` into the cache once, and load it globally."""
    from jittor.build import compiler
    includes = _include_dirs(frontend)
    with open(_SOURCE, "rb") as source:
        digest = hashlib.sha1(source.read())
    digest.update("\0".join(includes).encode())
    target = os.path.join(jt.flags.cache_path, "cudnn_sdpa_%s.so" % digest.hexdigest()[:16])
    if not os.path.exists(target):
        partial = "%s.%d.tmp" % (target, os.getpid())
        command = [compiler.cc_path, "-std=c++17", "-O2", "-shared", "-fPIC",
                   *("-I" + d for d in includes), _SOURCE, "-o", partial]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError("building cuDNN attention failed:\n%s\n%s"
                               % (" ".join(command), done.stderr[-4000:]))
        os.replace(partial, target)
    return ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL)


def _library():
    """The loaded library, or None when the cudnn frontend is not installed."""
    with _lock:
        if "library" not in _state:
            frontend = _frontend_include()
            if frontend is None:
                _state["library"] = None
            else:
                # cuDNN has to be loaded (globally) before the library that
                # leaves its symbols undefined.
                from jittor._runtime.backend_libraries import get_library_ops
                if get_library_ops("cudnn") is None:
                    _state["library"] = None
                else:
                    _state["library"] = _build(frontend)
        return _state["library"]


_HEADER = """
#include "{descriptor}"
extern "C" {{
const char* jt_cudnn_sdpa_last_error();
int jt_cudnn_sdpa_supported(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, float, int, int, int64_t, int64_t, int64_t*);
int64_t jt_cudnn_sdpa_forward_workspace(cudnnHandle_t, int, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, float, int, int, int64_t, int64_t);
int jt_cudnn_sdpa_forward(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, float, int, int, int64_t, int64_t,
    void*, void*, void*, void*, void*, void*, void*);
int64_t jt_cudnn_sdpa_backward_workspace(cudnnHandle_t, int, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, float, int);
int jt_cudnn_sdpa_backward(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, float, int, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
}}
"""

# Shapes are read from the inputs at run time, so one compiled operator serves
# every shape; only the dtype, the scale and the two switches are baked in.
_DIMS = ("in0->shape[0], in0->shape[1], in1->shape[1], in0->shape[2], "
         "in1->shape[2], in0->shape[3]")


def _header():
    return _HEADER.format(descriptor=os.path.abspath(_DESCRIPTOR))


def _bias_dims(bias):
    return (0, 0) if bias is None else (int(bias.shape[0]), int(bias.shape[1]))


def _supported(query, key, scale, causal, training, bias=None):
    bias_b, bias_h = _bias_dims(bias)
    shape = (tuple(query.shape), tuple(key.shape), _jittor_dtype_name(query.dtype),
             float(scale), bool(causal), bool(training), bias_b, bias_h, jt.flags.device_id)
    answer = _supported_shapes.get(shape)
    if answer is None:
        # Asked through the executor, with the handle and device the real call
        # will use: cuDNN's own check_support is the only complete answer.
        dtype = _DTYPES[shape[2]]
        probe = jt.code(
            (1,), "int32", [query, key],
            cuda_header=_header(),
            cuda_src=f"""
            cudnnHandle_t handle = jittor::cudnn_bind_stream();
            int64_t workspace = 0;
            int ok = jt_cudnn_sdpa_supported(handle, {dtype}, {_DIMS}, {float(scale)!r}f,
                                             {int(causal)}, {int(training)},
                                             {bias_b}, {bias_h}, &workspace);
            cudaMemcpy(out0_p, &ok, sizeof(int), cudaMemcpyHostToDevice);
            """)
        answer = bool(probe.item())
        _supported_shapes[shape] = answer
    return answer


def _forward(query, key, value, scale, causal, training, bias=None):
    dtype = _DTYPES[_jittor_dtype_name(query.dtype)]
    bias_b, bias_h = _bias_dims(bias)
    stats_shape = tuple(query.shape[:3]) + (1,)
    outputs = jt.code(
        [query.shape, stats_shape] if training else [query.shape],
        [query.dtype, "float32"] if training else [query.dtype],
        [query, key, value] + ([] if bias is None else [bias]),
        cuda_header=_header(),
        cuda_src=f"""
        cudnnHandle_t handle = jittor::cudnn_bind_stream();
        int64_t size = jt_cudnn_sdpa_forward_workspace(handle, {dtype}, {_DIMS},
            {float(scale)!r}f, {int(causal)}, {int(training)}, {bias_b}, {bias_h});
        if (size < 0) LOGf << "cuDNN attention forward:" << jt_cudnn_sdpa_last_error();
        jittor::CudnnWorkspace workspace(size);
        if (jt_cudnn_sdpa_forward(handle, {dtype}, {_DIMS}, {float(scale)!r}f,
                {int(causal)}, {int(training)}, {bias_b}, {bias_h},
                in0_p, in1_p, in2_p, {"nullptr" if bias is None else "in3_p"}, out0_p,
                {"out1_p" if training else "nullptr"}, workspace.ptr))
            LOGf << "cuDNN attention forward:" << jt_cudnn_sdpa_last_error();
        """)
    return outputs[0], (outputs[1] if training else None)


def _backward(query, key, value, out, grad_out, stats, scale, causal):
    dtype = _DTYPES[_jittor_dtype_name(query.dtype)]
    return jt.code(
        [query.shape, key.shape, value.shape],
        [query.dtype, key.dtype, value.dtype],
        [query, key, value, out, grad_out, stats],
        cuda_header=_header(),
        cuda_src=f"""
        cudnnHandle_t handle = jittor::cudnn_bind_stream();
        int64_t size = jt_cudnn_sdpa_backward_workspace(handle, {dtype}, {_DIMS},
            {float(scale)!r}f, {int(causal)});
        if (size < 0) LOGf << "cuDNN attention backward:" << jt_cudnn_sdpa_last_error();
        jittor::CudnnWorkspace workspace(size);
        if (jt_cudnn_sdpa_backward(handle, {dtype}, {_DIMS}, {float(scale)!r}f, {int(causal)},
                in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out0_p, out1_p, out2_p,
                workspace.ptr))
            LOGf << "cuDNN attention backward:" << jt_cudnn_sdpa_last_error();
        """)


class _CudnnAttention(jt.Function):
    """Fused attention that saves one float per query row for its backward."""

    def execute(self, query, key, value, scale, causal):
        self.scale, self.causal = scale, causal
        out, stats = _forward(query, key, value, scale, causal, True)
        self.saved = (query, key, value, out, stats)
        return out

    def grad(self, grad_out):
        query, key, value, out, stats = self.saved
        grad_out = grad_out.cast(query.dtype)
        grad_query, grad_key, grad_value = _backward(
            query, key, value, out, grad_out, stats, self.scale, self.causal)
        return grad_query, grad_key, grad_value, None, None


def _mask_bias(attn_mask, query, key):
    """The mask as an additive [b or 1, h or 1, Lq, Lk] bias, or None if it is not one."""
    shape = tuple(int(size) for size in attn_mask.shape)
    if len(shape) < 2 or len(shape) > 4:
        return None
    shape = (1,) * (4 - len(shape)) + shape
    batch, heads, query_length = (int(query.shape[i]) for i in range(3))
    if shape[2:] != (query_length, int(key.shape[2])) \
            or shape[0] not in (1, batch) or shape[1] not in (1, heads):
        return None
    mask = attn_mask.reshape(shape)
    if _jittor_dtype_name(mask.dtype) == "bool":
        negative = jt.array(float("-inf")).cast(query.dtype).broadcast(shape)
        return jt.ternary(mask, jt.zeros(shape, query.dtype), negative)
    if "float" not in _jittor_dtype_name(mask.dtype):
        return None
    return mask.cast(query.dtype)


def _cudnn_fused_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None):
    """Run attention through cuDNN, or return None to decline."""
    if float(dropout_p or 0.0) != 0.0:
        return None
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        return None
    # Equal head counts: `nn.scaled_dot_product_attention` has no enable_gqa,
    # so a mismatch is the caller's error to raise on the path below (the Torch
    # frontend expands grouped heads before it gets here).
    if tuple(key.shape) != tuple(value.shape) or query.shape[0] != key.shape[0] \
            or query.shape[1] != key.shape[1] or query.shape[3] != key.shape[3]:
        return None
    dtype = _jittor_dtype_name(query.dtype)
    if dtype not in _DTYPES or _jittor_dtype_name(key.dtype) != dtype \
            or _jittor_dtype_name(value.dtype) != dtype:
        return None
    if _library() is None:
        return None
    scale = float(scale) if scale is not None else float(query.shape[3]) ** -0.5
    training = _output_requires_grad(query, key, value)
    if attn_mask is None:
        if not _supported(query, key, scale, is_causal, training):
            return None
        if training:
            return _CudnnAttention.apply(query, key, value, scale, bool(is_causal))
        return _forward(query, key, value, scale, bool(is_causal), False)[0]
    # A mask only without a backward: a row every key is masked out of has no
    # softmax, which the composite answers with 0 and zero gradient; cuDNN's
    # backward would carry the NaN instead.
    if training:
        return None
    bias = _mask_bias(attn_mask, query, key)
    if bias is None or not _supported(query, key, scale, is_causal, False, bias):
        return None
    out = _forward(query, key, value, scale, bool(is_causal), False, bias)[0]
    # The same 0 for a fully masked row as the composite gives.
    top = bias.max([-1], keepdims=True)
    live = jt.logical_not(jt.isinf(top) & (top < 0)).broadcast(out.shape)
    return jt.ternary(live, out, jt.zeros_like(out))


def _supports(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return _jittor_dtype_name(query.dtype) in _DTYPES


# Not `dtypes=`: the registry filters on every tensor argument, and a bool mask
# would take a float16 call off this kernel. `supports` looks at the query
# only, which also keeps float32 for `fused_attention_f32_cuda.py`.
register_kernel("nn.fused_attention", "cuda", _cudnn_fused_attention, supports=_supports)
