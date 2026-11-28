"""Private CUDA inference kernels for gated activations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel

from ._inference import cached_source


def _silu_and_mul_supported(x):
    if not isinstance(x, jt.Var):
        return False
    if _output_requires_grad(x):
        return False
    if _jittor_dtype_name(x.dtype) not in ("float16", "bfloat16", "float32"):
        return False
    try:
        shape = tuple(int(size) for size in x.shape)
    except Exception:
        return False
    return bool(shape) and all(size > 0 for size in shape) and shape[-1] % 2 == 0


@optional_kernel("nn.silu_and_mul", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_silu_and_mul_supported)
def _silu_and_mul_cuda(x):
    """Return fused ``silu(x[..., :d]) * x[..., d:]`` for CUDA inference."""
    shape = tuple(int(size) for size in x.shape)
    gated_size = shape[-1] // 2
    return _silu_and_mul_rows(x, shape[:-1] + (gated_size,), gated_size, shape[-1], 0)


def _silu_and_mul_rows(x, output_shape, gated_size, pitch, offset, *, value=None, gelu=False):
    activation = ("(0.5f * gate) * (1.0f + tanhf(0.7978845608028654f * "
                  "(gate + 0.044715f * (gate * gate * gate))))" if gelu
                  else "gate / (1.0f + expf(-gate))")
    value_ptr = "in1_p" if value is not None else "nullptr"
    name = "gelu_and_mul" if gelu else "silu_and_mul"
    cuda_src = cached_source(r"""
    __global__ static void %(name)s_vec16(
            const in0_type* __restrict__ x, const in0_type* __restrict__ value,
            out0_type* __restrict__ y,
            int64_t total_vecs) {
        constexpr int VEC = 16 / sizeof(in0_type);
        constexpr int GATED_VECS = %(gated_size)d / VEC;
        constexpr int INPUT_VECS = %(input_size)d / VEC;
        const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
        uint4* __restrict__ yv = reinterpret_cast<uint4*>(y);
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total_vecs; index += stride) {
            int64_t row = index / GATED_VECS;
            int col = (int)(index - row * GATED_VECS);
            int64_t base = row * INPUT_VECS + col + %(offset)dLL / VEC;
            uint4 g_raw = xv[base];
            uint4 v_raw = %(separate)d ? reinterpret_cast<const uint4*>(value)[index]
                                       : xv[base + GATED_VECS];
            uint4 out_raw;
            const in0_type* ge = reinterpret_cast<const in0_type*>(&g_raw);
            const in0_type* ve = reinterpret_cast<const in0_type*>(&v_raw);
            out0_type* oe = reinterpret_cast<out0_type*>(&out_raw);
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float gate = static_cast<float>(ge[k]);
                float value = static_cast<float>(ve[k]);
                float activated = static_cast<float>(
                    in0_type(%(activation)s));
                oe[k] = out0_type(activated * value);
            }
            yv[index] = out_raw;
        }
    }
    __global__ static void %(name)s(
            const in0_type* __restrict__ x, const in0_type* __restrict__ value,
            out0_type* __restrict__ y, int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int dim = (int)(index %% %(gated_size)d);
            int64_t row = index / %(gated_size)d;
            int64_t input_base = row * %(input_size)d + %(offset)dLL;
            float gate = static_cast<float>(x[input_base + dim]);
            float other = static_cast<float>(
                %(separate)d ? value[index] : x[input_base + %(gated_size)d + dim]);
            // Preserve the public SiLU dtype boundary before gating, as
            // silu(x[..., :d]) * x[..., d:] does without this fusion.
            float activated = static_cast<float>(
                in0_type(%(activation)s));
            y[index] = out0_type(activated * other);
        }
    }
    int64_t total = out0->num;
    int threads = 256;
    constexpr int VEC = 16 / sizeof(in0_type);
    if (total && VEC > 1 && (%(gated_size)d %% VEC == 0)
            && (%(input_size)d %% VEC == 0) && (%(offset)d %% VEC == 0)
            && ((((size_t)in0_p | (size_t)%(value_ptr)s | (size_t)out0_p) & 15) == 0)) {
        int64_t total_vecs = total / VEC;
        int blocks = (int)((total_vecs + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        %(name)s_vec16<<<blocks, threads>>>(in0_p, %(value_ptr)s, out0_p, total_vecs);
    } else if (total) {
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        %(name)s<<<blocks, threads>>>(in0_p, %(value_ptr)s, out0_p, total);
    }
    CHECK(0 == cudaGetLastError());
    """, {
        "gated_size": gated_size,
        "input_size": pitch,
        "offset": offset,
        "name": name,
        "activation": activation,
        "value_ptr": value_ptr,
        "separate": value is not None,
    })
    return _stop_grad_outputs(
        jt.code(output_shape, x.dtype, [x] if value is None else [x, value], cuda_src=cuda_src))


@optional_kernel("nn.silu_and_mul.split", "cuda")
def _silu_and_mul_split_cuda(gate, value):
    """Consume two adjacent equal-width slices of a pitched CUDA matrix."""
    operands = _adjacent_gate_slices(gate, value)
    if operands is not None:
        return _silu_and_mul_rows(*operands)
    return None


def _adjacent_gate_slices(gate, value):
    if _output_requires_grad(gate, value) or jt.flags.amp_reg:
        return None
    shape = tuple(gate.shape)
    if (len(shape) < 2 or shape != tuple(value.shape) or any(n <= 0 for n in shape)
            or gate.dtype != value.dtype
            or _jittor_dtype_name(gate.dtype) not in ("float16", "bfloat16")
            or gate._storage_is_contiguous() or value._storage_is_contiguous()):
        return None
    # A common live view owner makes the offsets comparable.
    owner = gate._view_base_object()
    other = value._view_base_object()
    if owner is None or other is None or owner.var_ptr != other.var_ptr:
        return None
    strides = tuple(gate._storage_strides())
    if strides != tuple(value._storage_strides()) or strides[-1] != 1:
        return None
    width = shape[-1]
    if value._storage_offset() - gate._storage_offset() != width:
        return None

    from .fused_attention_f32_cuda import _dense_storage_layout

    layout = _dense_storage_layout(gate)
    if layout is None:
        return None
    base, _, offset = layout
    offset_end = offset + sum((n - 1) * s for n, s in zip(shape, strides)) + width
    if offset_end >= int(base.numel()):
        return None
    pitch = None
    rows = 1
    for dim, stride in reversed(tuple(zip(shape[:-1], strides[:-1]))):
        if dim > 1:
            if pitch is None:
                pitch = stride
            if stride != pitch * rows:
                return None
        rows *= dim
    if pitch is None or pitch < 2 * width:
        return None
    return base, shape, width, pitch, offset


@optional_kernel("nn.gelu_and_mul.inference", "cuda")
def _gelu_and_mul_cuda(gate, value, *, dense_gate):
    if (_output_requires_grad(gate, value) or jt.flags.amp_reg
            or gate.dtype != value.dtype or tuple(gate.shape) != tuple(value.shape)
            or _jittor_dtype_name(gate.dtype) not in ("float16", "bfloat16")
            or not gate.numel()):
        return None
    original_gate = gate._input(0) if gate._is_pending_contiguous() else gate
    original_value = value._input(0) if value._is_pending_contiguous() else value
    operands = _adjacent_gate_slices(original_gate, original_value)
    if operands is not None:
        return _silu_and_mul_rows(*operands, gelu=True)
    if dense_gate and gate._storage_is_contiguous() and value._storage_is_contiguous():
        from jittor.nn.functional._parallel_linear import observe_gated_projections

        observe_gated_projections(gate, value)
        width = gate.shape[-1]
        return _silu_and_mul_rows(gate, gate.shape, width, width, 0, value=value, gelu=True)
    return None


__all__ = []
