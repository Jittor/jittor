"""Class-aware greedy non-maximum suppression on CUDA."""

import jittor as jt
from jittor._runtime.dispatch import register_kernel


_HEADER = r"""
namespace jittor {
constexpr int nms_word_bits = 32;
constexpr int nms_mask_threads = 256;

template<class T>
__global__ static void find_nms_groups(
        const T* classes, int size, int2* groups, int* count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= size || (row && classes[row - 1] == classes[row])) return;
    int begin = row + 1, end = size;
    while (begin < end) {
        const int mid = begin + (end - begin) / 2;
        if (classes[mid] == classes[row]) begin = mid + 1;
        else end = mid;
    }
    groups[atomicAdd(count, 1)] = make_int2(row, begin);
}

__global__ static void set_nms_ranges(const int2* groups, const int* count, int2* ranges) {
    for (int group = blockIdx.x; group < *count; group += gridDim.x) {
        const int2 span = groups[group];
        for (int i = span.x + threadIdx.x; i < span.y; i += blockDim.x)
            ranges[i] = span;
    }
}

template<class T, class Math>
__device__ static inline bool nms_overlap(const T* a, const T* b, float threshold) {
    const T width = max(T(0), min(a[2], b[2]) - max(a[0], b[0]));
    const T height = max(T(0), min(a[3], b[3]) - max(a[1], b[1]));
    const Math intersection = Math(width) * Math(height);
    const Math area_a = (Math(a[2]) - Math(a[0])) * Math(a[3] - a[1]);
    const Math area_b = (Math(b[2]) - Math(b[0])) * Math(b[3] - b[1]);
    return intersection / (area_a + area_b - intersection) > threshold;
}

template<class T, class Math>
__global__ static void build_nms_masks(
        const T* boxes, const int2* ranges, int size, int words, uint32_t* masks,
        float threshold) {
    constexpr int rows_per_block = nms_mask_threads / nms_word_bits;
    const int row = blockIdx.x * rows_per_block + threadIdx.x / nms_word_bits;
    if (row >= size) return;
    const int2 span = ranges[row];
    const int length = span.y - span.x;
    if (length > words * nms_word_bits) return;
    const int lane = threadIdx.x % nms_word_bits;
    const int end_word = (length + nms_word_bits - 1) / nms_word_bits;
    for (int word = (row - span.x) / nms_word_bits + blockIdx.y;
             word < end_word; word += gridDim.y) {
        const int other = span.x + word * nms_word_bits + lane;
        const bool suppressed = other > row && other < span.y &&
            nms_overlap<T, Math>(boxes + int64_t(row) * 4, boxes + int64_t(other) * 4, threshold);
        const uint32_t bits = __ballot_sync(0xffffffffu, suppressed);
        if (lane == 0) masks[int64_t(row) * words + word] = bits;
    }
}

template<class T, class Math>
__global__ static void select_nms_boxes(
        const T* boxes, const int* score_ranks, const int2* groups, const int* group_count,
        const uint32_t* masks, uint8_t* flags, bool* keep, int words, float threshold) {
    extern __shared__ uint32_t removed[];
    for (int group = blockIdx.x; group < *group_count; group += gridDim.x) {
        const int2 span = groups[group];
        const int length = span.y - span.x;
        if (length > words * nms_word_bits) {
            // Bounded scratch for large classes. Compare only surviving rows.
            for (int i = span.x + threadIdx.x; i < span.y; i += blockDim.x) flags[i] = 0;
            for (int i = span.x; i < span.y; ++i) {
                __syncthreads();
                if (flags[i]) continue;
                if (threadIdx.x == 0) keep[score_ranks[i]] = true;
                for (int j = i + 1 + threadIdx.x; j < span.y; j += blockDim.x)
                    if (nms_overlap<T, Math>(boxes + int64_t(i) * 4, boxes + int64_t(j) * 4, threshold))
                        flags[j] = 1;
            }
            __syncthreads();
            continue;
        }
        const int active_words = (length + nms_word_bits - 1) / nms_word_bits;
        for (int word = threadIdx.x; word < active_words; word += blockDim.x) removed[word] = 0;
        __syncthreads();
        for (int word = 0; word < active_words; ++word) {
            const int base = span.x + word * nms_word_bits;
            const int valid = min(span.y - base, nms_word_bits);
            const uint32_t valid_bits = 0xffffffffu >> (nms_word_bits - valid);
            while (const uint32_t next = valid_bits & ~removed[word]) {
                const int lane = __ffs(next) - 1;
                const int row = base + lane;
                if (threadIdx.x == 0) keep[score_ranks[row]] = true;
                const uint32_t* mask = masks + int64_t(row) * words;
                __syncthreads();
                for (int j = word + threadIdx.x; j < active_words; j += blockDim.x)
                    removed[j] |= mask[j] | (j == word ? 1u << lane : 0u);
                __syncthreads();
            }
        }
        __syncthreads();
    }
}
}  // namespace jittor
"""

_LAUNCH = r"""
    using Math = typename std::conditional<std::is_same<in0_type, double>::value, double, float>::type;
    const int size = in0_shape0;
    const int words = out3_shape1;
    auto* groups = reinterpret_cast<int2*>(out1_p);
    auto* ranges = reinterpret_cast<int2*>(out2_p);
    auto* count = out4_p;
    CHECK(0 == cudaMemsetAsync(count, 0, sizeof(int)));
    CHECK(0 == cudaMemsetAsync(out0_p, 0, size * sizeof(bool)));
    constexpr int threads = 128;
    const int group_blocks = min(size, 256);
    const float threshold = data["threshold"];
    find_nms_groups<<<(size - 1) / threads + 1, threads>>>(in1_p, size, groups, count);
    set_nms_ranges<<<group_blocks, threads>>>(groups, count, ranges);
    const dim3 mask_blocks((size - 1) / (nms_mask_threads / nms_word_bits) + 1, min(words, 4));
    build_nms_masks<in0_type, Math><<<mask_blocks, nms_mask_threads>>>(
        in0_p, ranges, size, words, out3_p, threshold);
    select_nms_boxes<in0_type, Math><<<group_blocks, 64, words * sizeof(uint32_t)>>>(
        in0_p, in2_p, groups, count, out3_p, out5_p, out0_p, words, threshold);
    CHECK(0 == cudaGetLastError());
"""


def _batched_nms_cuda(boxes, scores, classes, iou_threshold):
    size = boxes.shape[0]
    order, _ = jt.argsort(scores, descending=True, stable=True)
    group_order, sorted_classes = jt.argsort(classes[order], stable=True)
    sorted_boxes = boxes[order[group_order]].contiguous()
    words = (min(size, 2048) + 31) // 32
    keep, *_ = jt.code(
        [(size,), (size, 2), (size, 2), (size, words), (1,), (size,)],
        ["bool", "int32", "int32", "uint32", "int32", "uint8"],
        [sorted_boxes, sorted_classes, group_order],
        cuda_header=_HEADER, cuda_src=_LAUNCH, backend="cuda", data={"threshold": iou_threshold})
    return order[jt.where(keep)[0]].int64()


register_kernel("vision.batched_nms", "cuda", _batched_nms_cuda,
                dtypes={"float16", "float32", "float64", "int32", "int64"},
                supports=lambda boxes, scores, classes, threshold: boxes.shape[0] <= 2**31 - 1)
