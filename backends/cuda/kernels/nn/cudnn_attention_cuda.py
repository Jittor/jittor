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
import math
import functools
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
        # The CUDA 13 compiler components do not include the optional
        # cuDNN sibling. Match the headers to the loaded cuDNN wheel before
        # considering headers exposed by an external nvcc installation.
        dirs += compiler.cuda_wheel_stack.include_dirs("cudnn")
        dirs += compiler.cuda_wheel_stack.include_dirs()
    dirs += [compiler.cuda_include,
             os.path.join(compiler.cuda_include, "..", "targets", "x86_64-linux", "include")]
    return list(dict.fromkeys(d for d in dirs if d and os.path.isdir(d)))


def _build(frontend):
    """Compile ``cudnn_sdpa.cc`` into the cache once, and load it globally."""
    from jittor.build import compiler
    # Frontend headers emit NVRTC references even when cuDNN itself was loaded
    # through the legacy toolkit path. Resolve that dependency explicitly;
    # loading the extension must not rely on another framework preloading it.
    compiler.preload_cuda_library("nvrtc", required=True)
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
        library = _state["library"]
    if library is not None and not _state.get("checked"):
        _state["checked"] = True
        _warn_if_unusable()
    return library


def _warn_if_unusable():
    """Say so when cuDNN answers no to a shape it always supports.

    A frontend and a runtime from different majors -- 1.30 against cuDNN 8.9,
    say -- build no graph at all. Every call then declines and attention
    silently runs as matmuls, which measured 2.4x slower on Qwen3-0.6B prefill
    (36.4 against 88.4 ms) with nothing in the log to connect cause and effect.
    """
    try:
        with jt.no_grad():
            query = jt.randn((1, 2, 16, 64)).bfloat16().cuda()
            key = jt.randn((1, 2, 16, 64)).bfloat16().cuda()
            usable = _supported(query, key, 0.125, False, False, None, 0)
    except Exception:
        return
    if usable:
        return
    jt.LOG.w("cuDNN fused attention answered no to a shape it always supports; "
             "the cuDNN runtime and nvidia-cudnn-frontend are probably from "
             "different major versions. Attention runs as matmuls instead.")


_HEADER = """
#include "{descriptor}"
extern "C" {{
const char* jt_cudnn_sdpa_last_error();
int jt_cudnn_sdpa_supported(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, float, int, int, int, int64_t, int64_t, int64_t*);
int64_t jt_cudnn_sdpa_forward_workspace(cudnnHandle_t, int, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, float, int, int, int, int64_t, int64_t);
int jt_cudnn_sdpa_forward(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, float, int, int, int, int64_t, int64_t,
    void*, void*, void*, void*, void*, void*, void*);
int64_t jt_cudnn_sdpa_backward_workspace(cudnnHandle_t, int, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, float, int, int, int64_t, int64_t);
int jt_cudnn_sdpa_backward(cudnnHandle_t, int, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, float, int, int, int64_t, int64_t, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*);
}}
"""

# Shapes are read from the inputs at run time, so one compiled operator serves
# every shape; only the dtype, the scale, the switches and the layout are baked
# in. `layout` has a bit per tensor (q 1, k 2, v 4) for one given to the kernel
# as [batch, seq, heads, head_dim] -- the tensor a transpose view was taken of
# -- rather than [batch, heads, seq, head_dim]; see cudnn_sdpa.cc.
_Q, _K, _V, _PACKED_QKV = 1, 2, 4, 8


def _dims(layout):
    if layout & _PACKED_QKV:
        return ("in0->shape[0], in0->shape[3], in0->shape[3], in0->shape[1], "
                "in0->shape[1], in0->shape[4]")
    q_seq, k_seq = layout & _Q, layout & _K
    return ("in0->shape[0], in0->shape[%d], in1->shape[%d], in0->shape[%d], "
            "in1->shape[%d], in0->shape[3]" % (2 if q_seq else 1, 2 if k_seq else 1,
                                               1 if q_seq else 2, 1 if k_seq else 2))


def _logical(t, seq_major):
    if len(t.shape) == 5:
        b, s, _, h, d = (int(size) for size in t.shape)
        return (b, h, s, d)
    b, x, y, d = (int(size) for size in t.shape)
    return (b, y, x, d) if seq_major else (b, x, y, d)


def _dense_storage_layout(tensor):
    """Return ``(root, strides, offset)`` for a non-negative-stride view of a dense Var."""
    strides = tuple(int(s) for s in tensor._storage_strides())
    if (any(int(n) == 0 for n in tensor.shape)
            or strides[-1] != 1 or any(s < 0 for s in strides)):
        return None
    root = tensor
    seen = set()
    while not root._storage_is_contiguous():
        if id(root) in seen:
            return None
        seen.add(id(root))
        parent = root._view_base_object()
        if not isinstance(parent, jt.Var):
            return None
        root = parent
    if (_jittor_dtype_name(root.dtype) != _jittor_dtype_name(tensor.dtype)
            or int(root.placement_backend) != int(tensor.placement_backend)):
        return None
    if root.device_id >= 0 and tensor.device_id >= 0 and root.device_id != tensor.device_id:
        return None
    offset = int(tensor._storage_offset()) - int(root._storage_offset())
    end = offset + sum((int(n) - 1) * s for n, s in zip(tensor.shape, strides))
    if offset < 0 or end >= int(root.numel()):
        return None
    return root, strides, offset


def _packed_qkv_view(query, key, value):
    """Return a dense ``(b, s, 3, h, d)`` view if ``q, k, v`` are slices of one
    packed QKV projection ``[b, s, 3, h, d]`` (or ``[b, s, 3*h*d]``), else None.
    """
    if tuple(query.shape) != tuple(key.shape) or tuple(query.shape) != tuple(value.shape):
        return None
    b, h, s, d = (int(size) for size in query.shape)
    if b <= 0 or h <= 0 or s <= 0 or d <= 0:
        return None
    if all(t._storage_is_contiguous() for t in (query, key, value)):
        return None
    layouts = [_dense_storage_layout(t) for t in (query, key, value)]
    if any(layout is None for layout in layouts):
        return None
    roots = [layout[0] for layout in layouts]
    if roots[0].var_ptr != roots[1].var_ptr or roots[0].var_ptr != roots[2].var_ptr:
        return None
    root = roots[0]
    if int(root.numel()) != b * s * 3 * h * d:
        return None
    offsets = [layout[2] for layout in layouts]
    if offsets != [0, h * d, 2 * h * d]:
        return None
    expected = (3 * s * h * d, d, 3 * h * d, 1)
    for _, st, _ in layouts:
        if len(st) != 4:
            return None
        for dim_size, actual_s, exp_s in zip((b, h, s, d), st, expected):
            if dim_size > 1 and actual_s != exp_s:
                return None
    return root.detach().reshape((b, s, 3, h, d))


def _physical(t):
    """(tensor, seq_major): the [b, s, h, d] source of a [b, h, s, d] transpose
    view, which the kernel reads in place, or `t` itself."""
    if not isinstance(t, jt.Var):
        return t, False
    if tuple(t._transpose_view_axes()) == (0, 2, 1, 3):
        return t._transpose_view_source(), True
    # Whose view record went with its root's holder (see
    # `_producer_transpose_axes`): the graph still knows.
    if tuple(t._producer_transpose_axes()) == (0, 2, 1, 3):
        return t._input(0), True
    return t, False


def _header():
    return _HEADER.format(descriptor=os.path.abspath(_DESCRIPTOR))


def _bias_dims(bias):
    return (0, 0) if bias is None else (int(bias.shape[0]), int(bias.shape[1]))


def _bias_uses_io_dtype(library):
    if "bias_uses_io_dtype" not in _state:
        probe = library.jt_cudnn_sdpa_bias_uses_io_dtype
        probe.argtypes = []
        probe.restype = ctypes.c_int
        _state["bias_uses_io_dtype"] = bool(probe())
    return _state["bias_uses_io_dtype"]


def _supported(query, key, scale, causal, training, bias=None, layout=0):
    bias_b, bias_h = _bias_dims(bias)
    shape = (tuple(query.shape), tuple(key.shape), _jittor_dtype_name(query.dtype),
             float(scale), bool(causal), bool(training), bias_b, bias_h, jt.flags.device_id,
             layout)
    answer = _supported_shapes.get(shape)
    if answer is None:
        lib = _library()
        if lib is None:
            return False
        dtype = _DTYPES[shape[2]]
        b, h, sq, d = _logical(query, layout & _Q)
        _, hk, skv, _ = _logical(key, layout & _K)
        fn = getattr(lib, "jt_cudnn_sdpa_supported_host", None)
        if fn is not None:
            fn.argtypes = [
                ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
            ]
            fn.restype = ctypes.c_int
            answer = bool(fn(dtype, b, h, hk, sq, skv, d, float(scale),
                             int(causal), int(training), int(layout),
                             int(bias_b), int(bias_h)))
        else:
            from jittor._runtime import step_capture
            if step_capture.tracing():
                return False
            inputs = [query] if (layout & _PACKED_QKV) else [query, key]
            probe = jt.code(
                (1,), "int32", inputs,
                cuda_header=_header(),
                cuda_src=f"""
                cudnnHandle_t handle = jittor::cudnn_bind_stream();
                int64_t workspace = 0;
                int ok = jt_cudnn_sdpa_supported(handle, {dtype}, {_dims(layout)}, {float(scale)!r}f,
                                                 {int(causal)}, {int(training)}, {layout},
                                                 {bias_b}, {bias_h}, &workspace);
                cudaMemcpy(out0_p, &ok, sizeof(int), cudaMemcpyHostToDevice);
                """)
            answer = bool(probe.item())
        _supported_shapes[shape] = answer
    return answer


def _forward(query, key, value, scale, causal, training, bias=None, layout=0):
    dtype = _DTYPES[_jittor_dtype_name(query.dtype)]
    bias_b, bias_h = _bias_dims(bias)
    packed = bool(layout & _PACKED_QKV)
    b, h, sq, d = _logical(query, layout & _Q)
    out_shape = (b, sq, h, d) if packed else query.shape
    stats_shape = (b, h, sq, 1)
    if _HEADER_TEXT[0] is None:
        _HEADER_TEXT[0] = _header()
    inputs = ([query] if packed else [query, key, value]) + ([] if bias is None else [bias])
    outputs = jt.code(
        [out_shape, stats_shape] if training else [out_shape],
        [query.dtype, "float32"] if training else [query.dtype],
        inputs,
        cuda_header=_HEADER_TEXT[0],
        cuda_src=_forward_source(dtype, layout, float(scale), bool(causal), bool(training),
                                 bias is not None, bias_b, bias_h))
    return outputs[0], (outputs[1] if training else None)


#: `_header()`, formatted once: the descriptor's path does not move.
_HEADER_TEXT = [None]


@functools.lru_cache(maxsize=256)
def _forward_source(dtype, layout, scale, causal, training, has_bias, bias_b, bias_h):
    # A model calls attention with the same settings in every layer, and
    # formatting this was a third of building the operator.
    dims = _dims(layout)
    if layout & _PACKED_QKV:
        q_ptr = "in0_p"
        k_ptr = "(in0_p + (int64_t)in0->shape[3] * in0->shape[4])"
        v_ptr = "(in0_p + 2LL * (int64_t)in0->shape[3] * in0->shape[4])"
        bias_ptr = "in1_p" if has_bias else "nullptr"
    else:
        q_ptr, k_ptr, v_ptr = "in0_p", "in1_p", "in2_p"
        bias_ptr = "in3_p" if has_bias else "nullptr"
    return f"""
        cudnnHandle_t handle = jittor::cudnn_bind_stream();
        int64_t size = jt_cudnn_sdpa_forward_workspace(handle, {dtype}, {dims},
            {scale!r}f, {int(causal)}, {int(training)}, {layout}, {bias_b}, {bias_h});
        if (size < 0) LOGf << "cuDNN attention forward:" << jt_cudnn_sdpa_last_error();
        jittor::CudnnWorkspace workspace(size);
        if (jt_cudnn_sdpa_forward(handle, {dtype}, {dims}, {scale!r}f,
                {int(causal)}, {int(training)}, {layout}, {bias_b}, {bias_h},
                {q_ptr}, {k_ptr}, {v_ptr}, {bias_ptr}, out0_p,
                {"out1_p" if training else "nullptr"}, workspace.ptr))
            LOGf << "cuDNN attention forward:" << jt_cudnn_sdpa_last_error();
        """


def _backward(query, key, value, out, grad_out, stats, scale, causal, layout=0, bias=None):
    dtype = _DTYPES[_jittor_dtype_name(query.dtype)]
    dims = _dims(layout)
    bias_b, bias_h = _bias_dims(bias)
    return jt.code(
        [query.shape, key.shape, value.shape],
        [query.dtype, key.dtype, value.dtype],
        [query, key, value, out, grad_out, stats] + ([] if bias is None else [bias]),
        cuda_header=_header(),
        cuda_src=f"""
        cudnnHandle_t handle = jittor::cudnn_bind_stream();
        int64_t size = jt_cudnn_sdpa_backward_workspace(handle, {dtype}, {dims},
            {float(scale)!r}f, {int(causal)}, {layout}, {bias_b}, {bias_h});
        if (size < 0) LOGf << "cuDNN attention backward:" << jt_cudnn_sdpa_last_error();
        jittor::CudnnWorkspace workspace(size);
        if (jt_cudnn_sdpa_backward(handle, {dtype}, {dims}, {float(scale)!r}f, {int(causal)},
                {layout}, {bias_b}, {bias_h},
                in0_p, in1_p, in2_p, in3_p, in4_p, in5_p,
                {"in6_p" if bias is not None else "nullptr"}, out0_p, out1_p, out2_p,
                workspace.ptr))
            LOGf << "cuDNN attention backward:" << jt_cudnn_sdpa_last_error();
        """)


class _CudnnAttention(jt.Function):
    """Fused attention that saves one float per query row for its backward."""

    # The tensors are the physical ones (see `_physical`); every gradient
    # comes back in its tensor's own layout, and the output in the query's.
    def execute(self, query, key, value, bias, scale, causal, layout):
        self.scale, self.causal, self.layout = scale, causal, layout
        out, stats = _forward(query, key, value, scale, causal, True, bias, layout)
        self.saved = (query, key, value, out, stats, bias)
        return out

    def grad(self, grad_out):
        query, key, value, out, stats, bias = self.saved
        grad_out = grad_out.cast(query.dtype)
        grad_query, grad_key, grad_value = _backward(
            query, key, value, out, grad_out, stats, self.scale, self.causal, self.layout, bias)
        return grad_query, grad_key, grad_value, None, None, None, None


class _CudnnRoundedAttention(_CudnnAttention):
    """CuDNN forward with a native composite backward for short BF16 calls.

    CuDNN rounds the scaled dS before its GEMMs; Torch Flash rounds dS first
    and scales the FP32 GEMM results. Reuse the canonical native attention
    gradient with that rounding boundary, rather than claiming the CuDNN
    fused backward satisfies the stricter Torch gradient budget.
    """

    def _run_call(self, query, key, value, *args, **kw):
        # Keep the original graph and values when a caller rebinds its holder.
        self.inputs = tuple(jt.ops.reshape(tensor, tensor.shape)
                            for tensor in (query, key, value))
        return super()._run_call(query, key, value, *args, **kw)

    def grad(self, grad_out):
        from jittor.nn.functional.attention import (
            _attention_gradients, _attention_probabilities,
        )

        def logical(tensor, flag):
            return tensor.transpose(0, 2, 1, 3) if self.layout & flag else tensor

        query, key, value = (logical(tensor, flag)
                             for tensor, flag in zip(self.inputs, (_Q, _K, _V)))
        out = logical(self.saved[3], _Q)
        with jt.flag_scope(amp_reg=0):
            probabilities = _attention_probabilities(query, key, None, False, self.scale)
            gradients = _attention_gradients(
                query, key, value, probabilities, logical(grad_out, _Q), self.scale,
                rounded_output=out)
        return tuple(logical(tensor, flag) for tensor, flag in
                     zip(gradients, (_Q, _K, _V))) + (None, None, None, None)


def _mask_bias(attn_mask, query_dims, key_dims, dtype):
    """The mask as a float32 [b or 1, h or 1, Lq, Lk] bias, or None to decline.

    `query_dims` and `key_dims` are the logical [b, h, s, d] shapes.
    """
    shape = tuple(int(size) for size in attn_mask.shape)
    if len(shape) < 1 or len(shape) > 4:
        return None
    shape = (1,) * (4 - len(shape)) + shape
    batch, heads, query_length = query_dims[:3]
    target = (batch, heads, query_length, key_dims[2])
    if any(size not in (1, full) for size, full in zip(shape, target)):
        return None
    shape = shape[:2] + target[2:]
    original = (1,) * (4 - len(attn_mask.shape)) + tuple(attn_mask.shape)
    mask = attn_mask.reshape(original).broadcast(shape)
    if _jittor_dtype_name(mask.dtype) == "bool":
        negative = jt.array(float("-inf")).float32().broadcast(shape)
        return jt.ternary(mask, jt.zeros(shape, "float32"), negative)
    if "float" not in _jittor_dtype_name(mask.dtype):
        return None
    # Match the softmax compute domain. Casting a float32 -1e9 padding
    # bias to half would turn a finite value into an infinity.
    return mask.float32()


def _accuracy_admitted(query):
    # Executable cuDNN graphs are not necessarily accurate enough: BF16
    # small-head attention exceeded the public output/gradient budgets.
    # Keep that whole head-width domain on FP32 opmath, including inference
    # and GQA. This is an accuracy policy, not a cuDNN hardware limitation.
    return not (_jittor_dtype_name(query.dtype) == "bfloat16"
                and len(query.shape) == 4 and int(query.shape[-1]) < 32)


def _reduced_precision_admitted(query, key, value, attn_mask, dropout_p, is_causal):
    # Match the native probability rounding of fused Torch SDPA for short,
    # unmasked BF16 attention. Native callers retain FP32 opmath by default.
    # A single key has exactly zero Q/K derivatives; keep that exact math
    # path rather than introducing fused cancellation residuals.
    shapes = [tuple(t.shape) for t in (query, key, value)]
    return (
        attn_mask is None and float(dropout_p or 0.0) == 0.0 and not is_causal
        and all(len(shape) == 4 for shape in shapes)
        and all(_jittor_dtype_name(t.dtype) == "bfloat16" for t in (query, key, value))
        and shapes[0][:2] == shapes[1][:2] == shapes[2][:2]
        and shapes[0][0] > 0 and shapes[0][1] > 0
        and shapes[1] == shapes[2]
        and shapes[0][-1] == shapes[1][-1] in (8, 16, 24)
        and 0 < shapes[0][-2] <= 64 and 1 < shapes[1][-2] <= 64
    )


def _cudnn_fused_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, *, grouped=False,
                           allow_reduced_precision=False, enable_gqa=False):
    """Run attention through cuDNN, or return None to decline.

    ``grouped``: `key` and `value` may have fewer heads than `query`, each
    shared by a group of query heads, which cuDNN reads in place.
    """
    if float(dropout_p or 0.0) != 0.0:
        return None
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        return None
    if tuple(key.shape) != tuple(value.shape) or query.shape[0] != key.shape[0] \
            or query.shape[3] != key.shape[3]:
        return None
    # Equal head counts through `nn.fused_attention`: it has no enable_gqa, so
    # a mismatch there is the caller's error to raise on the path below.
    heads, key_heads = int(query.shape[1]), int(key.shape[1])
    if key_heads != heads and not ((grouped or enable_gqa) and key_heads > 0 and heads % key_heads == 0):
        return None
    dtype = _jittor_dtype_name(query.dtype)
    if dtype not in _DTYPES or _jittor_dtype_name(key.dtype) != dtype \
            or _jittor_dtype_name(value.dtype) != dtype:
        return None
    reduced_precision = (allow_reduced_precision and _reduced_precision_admitted(
        query, key, value, attn_mask, dropout_p, is_causal))
    if not (_accuracy_admitted(query) or reduced_precision):
        return None
    training = _output_requires_grad(query, key, value)
    if attn_mask is not None:
        if _output_requires_grad(attn_mask):
            return None
        from jittor._runtime import step_capture
        # Backward mask admission checks whether finite row offsets absorb
        # log(Lk) via a host readback; decline during capture when training.
        # Inference mask preparation is pure device graph ops and capture-safe.
        if step_capture.tracing() and training:
            return None
    library = _library()
    if library is None:
        return None
    io_bias = attn_mask is not None and _bias_uses_io_dtype(library)
    if io_bias and _jittor_dtype_name(attn_mask.dtype) not in ("bool", dtype):
        # The public mask may carry more precision than Q/K/V.
        return None
    scale = float(scale) if scale is not None else float(query.shape[3]) ** -0.5
    packed_qkv = None if training else _packed_qkv_view(query, key, value)
    if packed_qkv is not None:
        query = key = value = packed_qkv
        q_seq = k_seq = v_seq = True
        layout = _PACKED_QKV
    else:
        (query, q_seq), (key, k_seq), (value, v_seq) = (_physical(t) for t in (query, key, value))
        layout = (_Q if q_seq else 0) | (_K if k_seq else 0) | (_V if v_seq else 0)

    def logical(out):
        # The output is laid out as the query is; hand it back as the
        # [b, h, s, d] view the caller asked for.
        return out.transpose(0, 2, 1, 3) if q_seq else out

    if attn_mask is None:
        if not _supported(query, key, scale, is_causal, training, layout=layout):
            return None
        if training:
            implementation = _CudnnRoundedAttention if reduced_precision else _CudnnAttention
            return logical(implementation.apply(
                query, key, value, None, scale, bool(is_causal), layout))
        return logical(_forward(query, key, value, scale, bool(is_causal), False,
                                layout=layout)[0])
    # Constant masks can use cuDNN's biased backward.
    bias, live = _mask_terms(attn_mask, _logical(query, q_seq), _logical(key, k_seq),
                             query.dtype, bool(is_causal), check_absorbed=training,
                             replace_infinity=not io_bias)
    if bias is None or not _supported(query, key, scale, is_causal, training, bias, layout):
        return None
    if io_bias:
        bias = _cast_io_bias(bias, query.dtype)
    if training:
        out = _CudnnAttention.apply(query, key, value, bias, scale, bool(is_causal), layout)
    else:
        out = _forward(query, key, value, scale, bool(is_causal), False, bias, layout)[0]
    # Empty rows were made safe before cuDNN computes statistics. This masks
    # their outputs and upstream gradients to zero in the kernel's layout.
    if q_seq:
        live = live._storage_permute(jt.NanoVector((0, 2, 1, 3)))
    return logical(jt.ternary(live.broadcast(out.shape), out, jt.zeros_like(out)))


#: What the last mask was turned into: (key, mask, bias, live rows). One
#: entry, dropped before the next mask's bias is built, so two are never
#: alive at once: a step builds one mask and the next step a new one.
_MASK_TERMS = []


def _cast_io_bias(bias, dtype):
    from jittor._runtime import step_capture
    cap = step_capture._ACTIVE
    if cap is None:
        return bias.cast(dtype)
    key = ("io", bias.var_ptr, _jittor_dtype_name(dtype))
    cached = cap.mask_terms.get(key)
    if cached is not None and cached[0] is bias:
        return cached[1]
    casted = bias.cast(dtype)
    cap.mask_terms[key] = (bias, casted)
    return casted


def _mask_terms(attn_mask, query_dims, key_dims, dtype, causal=False, *,
                check_absorbed=True, replace_infinity=True):
    """The additive bias for `attn_mask` and which query rows keep a key.

    Built once per mask, not once per layer. A decoder hands every layer the
    same mask, and Transformers' SDPA wrapper slices it for each one
    (``attention_mask[:, :, :, :kv_len]``) -- a new view of the same data
    every time. Turning it into a bias and scanning it for fully masked rows
    in each of Qwen3-0.6B's 28 layers was 2.1 ms of a 27 ms prefill, the
    whole difference to PyTorch, and eight more graph nodes per layer per
    decoded token. A view is recognised by the Var it views and where in it
    it looks; anything else only by being the same object.
    """
    from jittor._runtime import step_capture
    cap = step_capture._ACTIVE
    tracing = step_capture.tracing()
    base = attn_mask._view_base_id() if attn_mask._is_view() else -1
    storage_ver = tuple(attn_mask._saved_storage_version()[1:])
    key = (base, attn_mask.var_ptr if base < 0 else 0, storage_ver,
           bool(causal), bool(check_absorbed), bool(replace_infinity),
           tuple(attn_mask.shape), attn_mask._storage_offset(),
           tuple(attn_mask._storage_strides()), _jittor_dtype_name(attn_mask.dtype),
           _jittor_dtype_name(dtype), tuple(query_dims), tuple(key_dims))
    if cap is not None:
        entry = cap.mask_terms.get(key)
        if entry is not None and (base >= 0 or entry[0] is attn_mask):
            return entry[1], entry[2]
    elif not tracing:
        for entry in _MASK_TERMS:
            if entry[0] == key and (base >= 0 or entry[1] is attn_mask):
                return entry[2], entry[3]
        _MASK_TERMS.clear()
    # Bias and saved statistics have a float32 ABI, independent of an
    # outer autocast preference.
    with jt.flag_scope(amp_reg=0):
        bias = _mask_bias(attn_mask, query_dims, key_dims, dtype)
        if bias is None:
            return None, None
        effective = bias
        if causal:
            hidden = jt.triu(jt.ones((query_dims[2], key_dims[2]), dtype="bool"), diagonal=1)
            negative = jt.array(float("-inf")).float32().broadcast(bias.shape)
            effective = jt.ternary(hidden, negative, bias)
        top = effective.max([-1], keepdims=True)
        live = jt.logical_not(jt.isinf(top) & (top < 0))
        if check_absorbed and key_dims[2] > 1:
            # The fused backward reconstructs exp(score - logsumexp). If a
            # finite row offset absorbs log(Lk) in float32, that subtraction
            # loses its normalizer. Keep the explicit-softmax path for these
            # rows; -inf rows are handled separately below.
            absorbed = jt.isfinite(top) & (top + math.log(key_dims[2]) == top)
            if bool(absorbed.any().item()):
                if cap is not None:
                    cap.mask_terms[key] = (attn_mask if base < 0 else None, None, None)
                elif not tracing:
                    _MASK_TERMS.append((key, attn_mask if base < 0 else None, None, None))
                return None, None
        # Preserve the FP32-bias path's finite sentinel. The IO-typed path
        # accepts -inf and can preserve it exactly, including in float16.
        if replace_infinity:
            neg_inf = jt.isinf(bias) & (bias < 0)
            bias = jt.ternary(neg_inf, jt.array(-1e9).float32().broadcast(bias.shape), bias)
        bias = jt.ternary(live, bias, jt.zeros_like(bias))
    if cap is not None:
        cap.mask_terms[key] = (attn_mask if base < 0 else None, bias, live)
    elif not tracing:
        _MASK_TERMS.append((key, attn_mask if base < 0 else None, bias, live))
    return bias, live


def _supports(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, allow_reduced_precision=False, enable_gqa=False):
    return _jittor_dtype_name(query.dtype) in _DTYPES and (
        _accuracy_admitted(query) or (allow_reduced_precision and _reduced_precision_admitted(
            query, key, value, attn_mask, dropout_p, is_causal)))


# Not `dtypes=`: the registry filters on every tensor argument, and a bool mask
# would take a float16 call off this kernel. `supports` looks at the query
# only, which also keeps float32 for `fused_attention_f32_cuda.py`.
register_kernel("nn.fused_attention", "cuda", _cudnn_fused_attention, supports=_supports)
register_kernel("nn.cudnn_attention", "cuda", _cudnn_fused_attention, supports=_supports)


def _cudnn_fused_attention_gqa(query, key, value, **kwargs):
    return _cudnn_fused_attention(query, key, value, grouped=True, **kwargs)


# Grouped-query attention without repeating the key and value heads first: a
# copy of every key and value head per query head, two operators and two
# kernels per layer of every decoded token.
register_kernel("nn.fused_attention_gqa", "cuda", _cudnn_fused_attention_gqa,
                supports=_supports)
