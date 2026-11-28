"""Tensor-core CUDA SDPA for inference with too few query tiles to fill the GPU."""

import ctypes
from functools import lru_cache
import math
import os
from pathlib import Path

import jittor as jt
from jittor._core.dtypes import dtype_name

from .fused_attention_f32_cuda import _dense_storage_layout


_TILE_QUERIES = 16
_MAX_SPLITS = 64


@lru_cache(maxsize=16)
def _multiprocessors(device):
    try:
        driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    except OSError:
        return 0
    # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, from cuda.h.
    ordinal, count = ctypes.c_int(), ctypes.c_int()
    if (driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(ordinal), device)
            or driver.cuDeviceGetAttribute(ctypes.byref(count), 16, ordinal)):
        return 0
    return count.value


@lru_cache(maxsize=1)
def _header():
    return Path(__file__).with_name("split_kv_attention.cuh").read_text(encoding="utf-8")


@lru_cache(maxsize=64)
def _source(head_dim, value_dim, depth_pitch, value_pitch, tile_keys, mask_type, bf16, seq_major):
    mask = "in3_p" if mask_type else "nullptr"
    mask_type = mask_type or "float"

    def layout(prefix):
        return "{" + ", ".join(f'int64_t(data["{prefix}{i}"])' for i in range(4)) + "}"
    mask_ptr = mask if mask == 'nullptr' else 'in3_p + int64_t(data["m_offset"])'
    args = f"{layout('q')}, {layout('k')}, {layout('v')}, {layout('m')}"
    return f"""
    const int queries = int(data["queries"]), keys = int(data["keys"]);
    const int heads = int(data["heads"]), group = int(data["group"]);
    const int splits = int(data["splits"]), prefixes = int(data["prefixes"]);
    const dim3 grid((queries + {_TILE_QUERIES - 1}) / {_TILE_QUERIES}, prefixes, splits);
    jt_split_kv::partial_attention<in0_type, {mask_type}, {int(mask_type == "bool")},
        {int(bf16)}, {head_dim}, {value_dim}, {depth_pitch}, {value_pitch}, {tile_keys}>
        <<<grid, 128>>>(in0_p + int64_t(data["q_offset"]),
                        in1_p + int64_t(data["k_offset"]),
                        in2_p + int64_t(data["v_offset"]),
                        {mask_ptr},
                        {args}, queries, keys, heads, group, float(data["scale"]),
                        out1_p, reinterpret_cast<float2*>(out2_p));
    CHECK(cudaGetLastError() == cudaSuccess);
    jt_split_kv::combine_attention<out0_type, {value_dim}, {int(seq_major)}>
        <<<prefixes * queries, {max(32, ((value_dim + 31) // 32) * 32)}>>>(
            out1_p, reinterpret_cast<const float2*>(out2_p), out0_p, queries, heads, splits);
    CHECK(cudaGetLastError() == cudaSuccess);
    """


def _input_layout(tensor, skip_copy):
    if skip_copy and tensor._is_pending_contiguous():
        layout = _input_layout(tensor._input(0), False)
        if layout is not None:
            return layout
    layout = _dense_storage_layout(tensor)
    if layout is not None or tuple(tensor._producer_transpose_axes()) != (0, 2, 1, 3):
        return layout
    # Native permutation Vars may still have their producer but no Python
    # view owner. Read that producer's dense sequence-major input.
    source = tensor._input(0)
    if not source._storage_is_contiguous():
        return None
    strides = tuple(source._storage_strides())
    return source.detach(), (strides[0], strides[2], strides[1], strides[3]), 0


def split_kv_attention(query, key, value, mask, scale):
    """Return an inference result or decline the layout or tile geometry."""
    batch, heads, queries, head_dim = map(int, query.shape)
    _, kv_heads, keys, _ = map(int, key.shape)
    value_dim = int(value.shape[-1])
    if (not batch or not kv_heads or not 0 < queries <= 2 * _TILE_QUERIES
            or not all(0 < d <= 128 and d % 8 == 0 for d in (head_dim, value_dim))
            or heads % kv_heads or not math.isfinite(scale)):
        return None
    depth_pitch, value_pitch = (max(16, 1 << (d - 1).bit_length()) for d in (head_dim, value_dim))
    # Leave room for the query and probability tiles in the static shared memory budget.
    tile_keys = 128 if depth_pitch + value_pitch <= 128 else 64
    splits = (keys + tile_keys - 1) // tile_keys
    if not 4 <= splits <= _MAX_SPLITS:
        return None
    architectures = jt.flags.cuda_archs
    if not architectures or min(architectures) < 80:
        return None
    prefixes = batch * heads
    blocks = prefixes * ((queries + _TILE_QUERIES - 1) // _TILE_QUERIES)
    device = int(query.device_id)
    sms = _multiprocessors(device if device >= 0 else int(jt.core.current_device()))
    # Splitting trades a second launch and partial result traffic for occupancy.
    if not 0 < blocks <= sms // 2:
        return None
    if prefixes * queries * splits * (value_dim + 2) * 4 > 16 << 20:
        return None

    skip_copy = jt.flags.keep_graph == 2
    layouts = [_input_layout(t, skip_copy) for t in (query, key, value)]
    if any(layout is None for layout in layouts):
        return None
    mask_type = ""
    if mask is not None:
        dtype = dtype_name(mask.dtype)
        if mask.ndim < 1 or mask.ndim > 4 or dtype not in ("bool", "float32", dtype_name(query.dtype)):
            return None
        mask_shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
        if any(size not in (1, full) for size, full in zip(mask_shape, (batch, heads, queries, keys))):
            return None
        layout = _dense_storage_layout(mask)
        if layout is None:
            return None
        root, strides, offset = layout
        strides = (0,) * (4 - len(strides)) + strides
        layouts.append((root, tuple(0 if n == 1 else s for n, s in zip(mask_shape, strides)), offset))
        mask_type = "bool" if dtype == "bool" else "in3_type"

    # Preserve the contiguous output of a contiguous query and the usual
    # sequence-major output of a transposed projection.
    expected = (queries * heads * head_dim, head_dim, heads * head_dim, 1)
    seq_major = all(n == 1 or s == want for n, s, want in zip(
        query.shape, query._storage_strides(), expected))
    out_shape = (batch, queries, heads, value_dim) if seq_major else (batch, heads, queries, value_dim)
    data = dict(queries=queries, keys=keys, heads=heads, group=heads // kv_heads,
                splits=splits, prefixes=prefixes, scale=scale)
    for name, (_, strides, offset) in zip("qkvm", layouts):
        for axis, stride in enumerate(strides):
            data[f"{name}{axis}"] = stride
        data[f"{name}_offset"] = offset
    if mask is None:
        data.update(m0=0, m1=0, m2=0, m3=0)
    parts = prefixes * queries * splits
    result, _, _ = jt.code(
        [out_shape, (parts, value_dim), (parts, 2)],
        [query.dtype, "float32", "float32"],
        [layout[0] for layout in layouts],
        cuda_header=_header(),
        cuda_src=_source(head_dim, value_dim, depth_pitch, value_pitch, tile_keys,
                         mask_type, dtype_name(query.dtype) == "bfloat16", seq_major),
        data=data,
    )
    result.stop_grad()
    return result._storage_permute((0, 2, 1, 3)) if seq_major else result
