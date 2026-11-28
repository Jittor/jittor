"""A vector copy for concatenating CUDA views with dense channels."""
from functools import lru_cache

from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import optional_kernel


def _offset(shape, strides, offset, concat_axis, start=0):
    terms = [f"{offset - start * strides[concat_axis]}LL", "column"]
    span = 1
    for axis in range(len(shape) - 2, -1, -1):
        size, stride = shape[axis], strides[axis]
        if size > 1 and stride:
            terms.append(f"(row / {span} % {size}) * {stride}LL")
        span *= size
    return " + ".join(terms)


@lru_cache(maxsize=128)
def _source(shape, axis, left_axis, layouts):
    span = 1
    for size in shape[axis + 1:-1]:
        span *= size
    left = _offset(shape, *layouts[0], axis)
    right = _offset(shape, *layouts[1], axis, start=left_axis)
    return f"""
__global__ static void cat_strided_rows(const in0_type* a, const in1_type* b,
        out0_type* out, const int count) {{
    constexpr int pack_width = 16 / sizeof(out0_type);
    constexpr int row_packs = {shape[-1]} / pack_width;
    const int step = blockDim.x * gridDim.x;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < count; p += step) {{
        const int row = p / row_packs, column = p % row_packs * pack_width;
        const bool first = row / {span} % {shape[axis]} < {left_axis};
        const long long index = first ? ({left}) : ({right});
        const out0_type* source = first ? a : b;
        if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(out)) & 15u) == 0) {{
            reinterpret_cast<uint4*>(out)[p] = *reinterpret_cast<const uint4*>(source + index);
        }} else {{
            #pragma unroll
            for (int i = 0; i < pack_width; ++i) out[p * pack_width + i] = source[index + i];
        }}
    }}
}}
constexpr int pack_width = 16 / sizeof(out0_type);
const int count = out0->num / pack_width;
cat_strided_rows<<<std::min((count + 255) / 256, 4096), 256>>>(in0_p, in1_p, out0_p, count);
CHECK(cudaGetLastError() == cudaSuccess);
"""


@optional_kernel("tensor.concat.strided_inference", "cuda")
def concat_strided_inference(values, axis):
    import jittor as jt
    from jittor.backends.cuda.kernels.nn.fused_attention_f32_cuda import _dense_storage_layout

    if not jt.flags.no_grad or len(values) != 2 or all(v._storage_is_contiguous() for v in values):
        return None
    a, b = values
    dtype = dtype_name(a.dtype)
    widths = {"bfloat16": 8, "float16": 8, "float32": 4}
    if dtype not in widths or b.dtype != a.dtype:
        return None
    if axis >= a.ndim - 1 or not (1 << 16 <= a.numel() + b.numel() <= 1 << 24):
        return None
    pack = widths[dtype]
    if int(a.shape[-1]) % pack:
        return None
    layouts = [_dense_storage_layout(v) for v in values]
    if any(layout is None for layout in layouts):
        return None
    if any(offset % pack or any(s % pack for s in stride[:-1])
           for _, stride, offset in layouts):
        return None
    shape = list(a.shape)
    shape[axis] += b.shape[axis]
    source = _source(tuple(shape), axis, a.shape[axis],
                     tuple((stride, offset) for _, stride, offset in layouts))
    return jt.code(shape, a.dtype, [root for root, _, _ in layouts], cuda_src=source).stop_grad()
