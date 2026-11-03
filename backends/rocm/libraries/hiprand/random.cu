#include "random.h"
#include <hiprand/hiprand_kernel.h>

namespace jittor {
namespace {
template<class T, bool NORMAL>
__global__ void generate(T* output, int64_t count, uint64_t seed, uint64_t offset) {
    constexpr bool PAIRS = NORMAL || sizeof(T) == 8;
    constexpr int RAW = sizeof(T) == 8 ? 4 : (NORMAL ? 2 : 1);
    const int64_t units = PAIRS ? (count + 1) / 2 : count;
    const int64_t first = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (first >= units) return;
    hiprandStatePhilox4_32_10_t state;
    hiprand_init(seed, 0, offset + first * RAW, &state);
    if (sizeof(T) == 8) {
        double2 value = NORMAL ? hiprand_normal2_double(&state) : hiprand_uniform2_double(&state);
        output[2 * first] = T(value.x);
        if (2 * first + 1 < count) output[2 * first + 1] = T(value.y);
    } else if (NORMAL) {
        float2 value = hiprand_normal2(&state);
        output[2 * first] = T(value.x);
        if (2 * first + 1 < count) output[2 * first + 1] = T(value.y);
    } else {
        output[first] = T(hiprand_uniform(&state));
    }
}

template<class T>
void launch(void* output, int64_t count, bool normal,
            uint64_t seed, uint64_t offset, hipStream_t stream) {
    const int64_t units = normal || sizeof(T) == 8 ? (count + 1) / 2 : count;
    const unsigned blocks = unsigned((units + 255) / 256);
    if (normal) generate<T, true><<<blocks, 256, 0, stream>>>(static_cast<T*>(output), count, seed, offset);
    else generate<T, false><<<blocks, 256, 0, stream>>>(static_cast<T*>(output), count, seed, offset);
}
} // namespace

hipError_t hiprand_generate(void* output, int64_t count, bool use_double, bool normal,
                           uint64_t seed, uint64_t offset, hipStream_t stream) {
    if (!count) return hipSuccess;
    if (use_double) launch<double>(output, count, normal, seed, offset, stream);
    else launch<float>(output, count, normal, seed, offset, stream);
    return hipGetLastError();
}
} // namespace jittor
