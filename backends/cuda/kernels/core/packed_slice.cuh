#pragma once

#include <algorithm>
#include <cstdint>
#include "core/var.h"
#include "core/var_slices.h"

namespace jittor {

template<class Pack, class Index>
__global__ static void packed_slice_write(Pack* target, const Pack* source,
                                         Index rows, Index columns, Index pitch) {
    const Index row_step = Index(gridDim.x) * blockDim.y;
    const Index column_step = Index(gridDim.y) * blockDim.x;
    for (Index row = Index(blockIdx.x) * blockDim.y + threadIdx.y;
         row < rows; row += row_step) {
        for (Index column = Index(blockIdx.y) * blockDim.x + threadIdx.x;
             column < columns; column += column_step) {
            target[row * pitch + column] = source[row * columns + column];
        }
    }
}

template<class Pack, class Index, class T>
static void launch_packed_slice(T* target, const T* source,
                                Index rows, Index columns, Index pitch) {
    const Index width = sizeof(Pack) / sizeof(T);
    columns /= width;
    pitch /= width;
    constexpr int threads = 256;
    int column_threads = 1;
    while (column_threads < threads && column_threads < columns)
        column_threads *= 2;
    const int row_threads = threads / column_threads;
    const dim3 block(column_threads, row_threads);
    const dim3 grid(std::min<Index>((rows - 1) / row_threads + 1, 4096),
                    std::min<Index>((columns - 1) / column_threads + 1, 65535));
    packed_slice_write<<<grid, block>>>(
        reinterpret_cast<Pack*>(target), reinterpret_cast<const Pack*>(source),
        rows, columns, pitch);
}

// A plain unit-step interval through a dense tensor is a pitched copy.
// Handle the common single-interval assignment directly without computing
// a slice index for every element. The ordinary scatter owns other indices.
template<class Index, class T>
static bool try_packed_slice_write(const Var* out, const Var* data,
                                  const VarSlices& slices, T* target, const T* source) {
    const int rank = out->shape.size();
    if (rank == 0 || data->num <= 0 || data->shape.size() != rank || slices.n > rank ||
        !out->is_contiguous() || !data->is_contiguous())
        return false;
    const auto dst_address = reinterpret_cast<uintptr_t>(target);
    const auto src_address = reinterpret_cast<uintptr_t>(source);
    if (dst_address < src_address + data->size && src_address < dst_address + out->size)
        return false;

    int cut = -1;
    Index start = 0;
    for (int axis = 0; axis < rank; ++axis) {
        const auto extent = out->shape[axis];
        int64 begin = 0, end = extent;
        if (axis < slices.n) {
            const auto& item = slices.slices[axis];
            if (!item.is_slice())
                return false;
            if (item.slice.mask != 7) {
                if (item.slice.step != 1)
                    return false;
                begin = item.slice.start;
                end = item.slice.stop;
            }
        }
        if (begin < 0 || end > extent || end < begin || data->shape[axis] != end - begin)
            return false;
        if (begin != 0 || end != extent) {
            if (cut >= 0)
                return false;
            cut = axis;
            start = begin;
        }
    }
    if (cut < 0)
        cut = 0;
    Index inner = 1;
    for (int axis = cut + 1; axis < rank; ++axis)
        inner *= out->shape[axis];
    const Index columns = inner * data->shape[cut];
    const Index pitch = inner * out->shape[cut];
    const Index rows = data->num / columns;
    auto* destination = target + start * inner;
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(destination) | src_address |
        (uintptr_t(columns) * sizeof(T)) | (uintptr_t(pitch) * sizeof(T));
    if constexpr (sizeof(T) < 16) {
        if ((alignment & 15) == 0) {
            launch_packed_slice<uint4>(destination, source, rows, columns, pitch);
            return true;
        }
    }
    if constexpr (sizeof(T) < 8) {
        if ((alignment & 7) == 0) {
            launch_packed_slice<uint2>(destination, source, rows, columns, pitch);
            return true;
        }
    }
    return false;
}

} // namespace jittor
