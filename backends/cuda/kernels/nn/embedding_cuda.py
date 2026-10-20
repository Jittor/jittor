"""Dense FP32 embedding gradients with stable, segmented CUDA accumulation."""

import jittor as jt
from jittor._core.dtypes import var_dtype_name
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


class _EmbeddingSum(jt.Function):
    def execute(self, gradient, indices, weight_rows):
        self.indices = indices
        self.gradient_shape = tuple(gradient.shape)
        channels = int(gradient.shape[-1])
        count = int(indices.numel())
        # CUDA argsort uses CUB's stable radix sort: equal keys retain their
        # original positions. Do not substitute the CPU std::sort path here.
        order, keys = jt.argsort(indices)
        if count >= 256:
            return self._chunked_sum(
                gradient, keys, order, weight_rows, channels, count)
        return jt.code(
            (weight_rows, channels), "float32",
            [keys, order, gradient.contiguous()],
            cuda_src=f"""
            __global__ static void embedding_segment_sum(
                    const in0_type* keys, const in1_type* order,
                    const in2_type* gradient, out0_type* result) {{
                for (int64_t begin = blockIdx.x; begin < {count}; begin += gridDim.x) {{
                    if (begin && keys[begin-1] == keys[begin]) continue;
                    int64_t token = keys[begin];
                    for (int64_t c = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
                         c < {channels}; c += (int64_t)gridDim.y * blockDim.x) {{
                        float value = 0.0f;
                        for (int64_t r = begin; r < {count} && keys[r] == token; ++r)
                            value += gradient[(int64_t)order[r] * {channels} + c];
                        result[token * {channels} + c] = value;
                    }}
                }}
            }}
            CHECK(0 == cudaMemsetAsync(out0_p, 0, (size_t)out0->num * sizeof(out0_type),
                                      cudaStreamPerThread));
            embedding_segment_sum<<<dim3({min(count, 65535)},
                {min((channels+255)//256, 65535)}), 256>>>(in0_p, in1_p, in2_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )

    @staticmethod
    def _chunked_sum(gradient, keys, order, weight_rows, channels, count):
        # Fixed 64-position chunks bound serial work even for one repeated
        # token. Only segment starts and chunk boundaries write scratch; the
        # finish kernel reads exactly those locations through a graph edge.
        chunk = 64
        partial = jt.code(
            (count, channels), "float32", [keys, order, gradient.contiguous()],
            cuda_src=f"""
            __global__ static void embedding_chunk_partial(
                    const in0_type* keys, const in1_type* order,
                    const in2_type* gradient, out0_type* partial) {{
                for (int64_t begin = blockIdx.x; begin < {count}; begin += gridDim.x) {{
                    if (begin % {chunk} && keys[begin-1] == keys[begin]) continue;
                    int64_t token = keys[begin];
                    int64_t end = (begin / {chunk} + 1) * {chunk};
                    for (int64_t c = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
                         c < {channels}; c += (int64_t)gridDim.y * blockDim.x) {{
                        float value = 0.0f;
                        for (int64_t r = begin;
                             r < end && r < {count} && keys[r] == token; ++r)
                            value += gradient[(int64_t)order[r] * {channels} + c];
                        partial[begin * {channels} + c] = value;
                    }}
                }}
            }}
            embedding_chunk_partial<<<dim3({min(count, 65535)},
                {min((channels+255)//256, 65535)}), 256>>>(in0_p, in1_p, in2_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )
        return jt.code(
            (weight_rows, channels), "float32", [keys, partial],
            cuda_src=f"""
            __global__ static void embedding_chunk_finish(
                    const in0_type* keys, const in1_type* partial, out0_type* result) {{
                for (int64_t begin = blockIdx.x; begin < {count}; begin += gridDim.x) {{
                    if (begin && keys[begin-1] == keys[begin]) continue;
                    int64_t token = keys[begin];
                    for (int64_t c = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
                         c < {channels}; c += (int64_t)gridDim.y * blockDim.x) {{
                        float value = partial[begin * {channels} + c];
                        for (int64_t r = (begin / {chunk} + 1) * {chunk};
                             r < {count} && keys[r] == token; r += {chunk})
                            value += partial[r * {channels} + c];
                        result[token * {channels} + c] = value;
                    }}
                }}
            }}
            CHECK(0 == cudaMemsetAsync(out0_p, 0, (size_t)out0->num * sizeof(out0_type),
                                      cudaStreamPerThread));
            embedding_chunk_finish<<<dim3({min(count, 65535)},
                {min((channels+255)//256, 65535)}), 256>>>(in0_p, in1_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )

    def grad(self, gradient):
        return gradient[self.indices].reshape(self.gradient_shape), None, None


class _Embedding(jt.Function):
    def execute(self, indices, weight):
        self.indices = indices
        self.weight_rows = int(weight.shape[0])
        return weight[indices]

    def grad(self, gradient):
        # Validate the ORIGINAL index before normalising negatives. Otherwise
        # -1 and V-1 could write the same row from different segments, or an
        # invalid index below -V could be normalised twice into a legal one.
        indices = jt.code(
            (int(self.indices.numel()),), "int64",
            [self.indices.contiguous()],
            cuda_header='#include "ops/composite/index_bounds.h"',
            cuda_src=f"""
            __global__ static void embedding_canonical_indices(
                    const in0_type* indices, out0_type* canonical, int64_t count) {{
                for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < count; i += (int64_t)blockDim.x * gridDim.x)
                    canonical[i] = jittor::bounded_index(
                        (int64_t)indices[i], {self.weight_rows}, 0, nullptr);
            }}
            embedding_canonical_indices<<<
                {min((int(self.indices.numel())+255)//256, 65535)}, 256>>>(
                in0_p, out0_p, in0->num);
            CHECK(0 == cudaGetLastError());
            """,
        ).stop_grad()
        return None, _EmbeddingSum.apply(gradient, indices, self.weight_rows)


def _supports_embedding(input, weight, padding_idx=None, max_norm=None,
                        norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    if not (isinstance(input, jt.Var) and isinstance(weight, jt.Var)):
        return False
    return (
        var_dtype_name(weight) == "float32" and var_dtype_name(input) in ("int32", "int64")
        and _output_requires_grad(weight)
        and len(weight.shape) == 2 and all(int(d) > 0 for d in weight.shape)
        and 0 < int(input.numel()) <= 2147483647
        and max_norm is None
        and not scale_grad_by_freq and not sparse
        and get_library_ops("cub") is not None
    )


@optional_kernel("nn.embedding", "cuda", dtypes=("float32", "int32", "int64"),
                 supports=_supports_embedding)
def _embedding_cuda(input, weight, padding_idx=None, max_norm=None,
                    norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    result = _Embedding.apply(input, weight)
    if padding_idx is not None:
        # Match the functional fallback: retain nonzero padding data in the
        # forward result while excluding its contribution to weight gradients.
        # Module construction owns padding-index normalization/validation.
        keep = (input != padding_idx).unsqueeze(-1).float32()
        result = result * keep + (result * (1.0 - keep)).stop_grad()
    return result
