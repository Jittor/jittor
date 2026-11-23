// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <limits>
#include "core/var.h"
#include "cub_argsort_op.h"
#include <vector>
#include "core/executor.h"
#include "runtime/device.h"
#include "ops/composite/argsort_op.h"
#ifdef JIT_cuda
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#endif

namespace jittor {

#ifndef JIT
CubArgsortOp::CubArgsortOp(Var* x, Var* indexes, Var* offsets, bool descending, NanoString dtype)
    : CubArgsortOp(x, indexes, offsets, descending, dtype, false) {}

CubArgsortOp::CubArgsortOp(Var* x, Var* indexes, Var* offsets, bool descending, NanoString dtype,
                         bool uniform_rows)
    : x(x), indexes(indexes), offsets(offsets), descending(descending), rowwise(false) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    USER_CHECK(offsets->dtype()==ns_int32);
    if (uniform_rows && construction_target_backend(x) == BackendId::Cuda
            && x->dtype() == ns_float32 && x->shape.size() > 0
            && x->num > 0 && x->num <= std::numeric_limits<int>::max()) {
        const int64 width = x->shape[x->shape.size() - 1];
        const int64 rows = width > 0 ? x->num / width : 0;
        // Segmented radix sort gives a wide segment to one CTA per pass.
        // A device-wide radix sort per row uses the GPU for these shapes.
        const bool grouped = rows > 8 && rows <= 16 && width >= 131072 && width <= 262144
            && (dtype == ns_int32 || dtype == ns_int64);
        rowwise = grouped || (rows > 0 && rows <= 8
            && width >= std::max<int64>(65536, rows * 16384));
    }
    y = create_output(nullptr, dtype);
    y_key = create_output(nullptr, x->dtype());
    set_flag(OpFlags::_manual_set_vnbb);
    y->set_flag(VarFlags::_needed_by_backward);
}

VarPtr CubArgsortOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return ArgsortOp::get_grad(out, dout, v, v_index, v->shape.size()-1, y);
}

void CubArgsortOp::infer_shape() {
    USER_CHECK(x->shape.size() == indexes->shape.size());
    int n = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        USER_CHECK(x->shape[i] == indexes->shape[i]);
        if (i < x->shape.size() - 1) {
            n *= x->shape[i];
        }
    }
    USER_CHECKop(offsets->shape.size(),==,1);
    USER_CHECKop(offsets->shape[0],==,n + 1);
    y->set_shape(x->shape);
    y_key->set_shape(x->shape);
}

void CubArgsortOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Tindexes:" << indexes->dtype();
    jk << "«Toffsets:" << offsets->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«ROWWISE=" << JK::hex(rowwise);
    if (rowwise) jk << "«DESCENDING=" << JK::hex(descending);
    jk << "«FUNC:";
    if (descending)
        jk << "SortPairsDescending";
    else
        jk << "SortPairs";
}

#else // JIT
#ifdef JIT_cuda
#if ROWWISE
__global__ static void pack_wide_sort_keys(const float* values, uint64_t* keys, int count, int width) {
    const int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned bits = __float_as_uint(values[index]);
    // Match the stable radix order, with both zero signs in the same group.
    const unsigned canonical = (bits & 0x7fffffffu) == 0 ? 0 : bits;
    unsigned ordered = canonical ^ ((canonical >> 31) ? 0xffffffffu : 0x80000000u);
    if (DESCENDING) ordered = ~ordered;
    keys[index] = (uint64_t(index / width) << 32) | ordered;
}

__global__ static void gather_wide_sort_values(const float* source, const Ty* index,
                                               float* destination, int count, int width) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) destination[i] = source[(i / width) * width + index[i]];
}

static void packed_wide_argsort(const float* source, const Tindexes* input_indices,
                                float* values, Ty* indices, int rows, int width) {
    const int count = rows * width;
    int row_bits = 0;
    for (int max_row = rows - 1; max_row > 0; max_row >>= 1) ++row_bits;
    const int end_bit = 32 + row_bits;
    const auto stream = reinterpret_cast<cudaStream_t>(
        backend_stream({BackendId::Cuda, current_device()}, BackendStreamKind::Compute).handle);

    size_t workspace_bytes = 0;
    const auto query = cub::DeviceRadixSort::SortPairs(
        nullptr, workspace_bytes, static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        input_indices, indices, count, 0, end_bit, stream);
    CHECK(query == cudaSuccess) << "CUDA packed radix sort setup failed: " << cudaGetErrorString(query);
    // The temporary buffer passed to CUB starts after two key arrays.
    const size_t key_bytes = (2 * size_t(count) * sizeof(uint64_t) + 255) & ~size_t(255);
    size_t allocation;
    auto& allocator = *runtime_executor().temp_allocator;
    auto* memory = static_cast<char*>(allocator.alloc(key_bytes + workspace_bytes, allocation));
    auto* keys = reinterpret_cast<uint64_t*>(memory);
    auto* sorted = keys + count;
    const int blocks = (count + 255) / 256;
    pack_wide_sort_keys<<<blocks, 256, 0, stream>>>(source, keys, count, width);
    auto result = cub::DeviceRadixSort::SortPairs(
        memory + key_bytes, workspace_bytes, keys, sorted, input_indices, indices, count, 0, end_bit, stream);
    if (result == cudaSuccess) {
        gather_wide_sort_values<<<blocks, 256, 0, stream>>>(source, indices, values, count, width);
        result = cudaGetLastError();
    }
    allocator.free(memory, key_bytes + workspace_bytes, allocation);
    CHECK(result == cudaSuccess) << "CUDA packed radix sort failed: " << cudaGetErrorString(result);
}
#endif

void CubArgsortOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ indexesp = indexes->ptr<Tindexes>();
#if ROWWISE
    auto* __restrict__ values = y_key->ptr<Tx>();
    auto* __restrict__ indices = y->ptr<Ty>();
    const int width = x->shape[x->shape.size() - 1];
    const int rows = x->num / width;
    // Keep each packed radix sort within the same key workspace bound even
    // when the batch needs more than eight rows.
    constexpr int max_packed_items = 2 * 1024 * 1024;
    if (rows > 1 && (rows > 8 || x->num <= max_packed_items)
            && (y->dtype() == ns_int32 || y->dtype() == ns_int64)) {
        const int group = std::min(8, max_packed_items / width);
        for (int first = 0; first < rows; first += group) {
            const int start = first * width;
            packed_wide_argsort(xp + start, indexesp + start, values + start, indices + start,
                                std::min(rows - first, group), width);
        }
        return;
    }
    size_t scratch_bytes = 0;
    CHECK(cub::DeviceRadixSort::@FUNC@@(nullptr, scratch_bytes,
        xp, values, indexesp, indices, width) == cudaSuccess);
    size_t allocation;
    auto* scratch = runtime_executor().temp_allocator->alloc(scratch_bytes, allocation);
    auto error = cudaSuccess;
    for (int row = 0; row < rows && error == cudaSuccess; ++row) {
        const int start = row * width;
        error = cub::DeviceRadixSort::@FUNC@@(scratch, scratch_bytes,
            xp + start, values + start, indexesp + start, indices + start, width);
    }
    runtime_executor().temp_allocator->free(scratch, scratch_bytes, allocation);
    CHECK(error == cudaSuccess) << "CUDA radix sort failed:" << cudaGetErrorString(error);
#else
    auto* __restrict__ offsetsp = offsets->ptr<Toffsets>();

    int num_items = 1, num_segments = 1;
    for (int i = 0; i < x->shape.size(); ++i) {
        num_items *= x->shape[i];
        if (i < x->shape.size() - 1) {
            num_segments *= x->shape[i];
        }
    }
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();

    // Determine temporary device storage requirementse = NULL;
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, y_keyp, indexesp, yp,
        num_items, num_segments, offsetsp, offsetsp + 1);
    // Allocate temporary storage
    size_t allocation;
    d_temp_storage = runtime_executor().temp_allocator->alloc(temp_storage_bytes, allocation);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::@FUNC@@(d_temp_storage, temp_storage_bytes,
        xp, y_keyp, indexesp, yp,
        num_items, num_segments, offsetsp, offsetsp + 1);
    runtime_executor().temp_allocator->free(d_temp_storage, temp_storage_bytes, allocation);
#endif
}
#endif // JIT_cuda
#endif // JIT

} // jittor
