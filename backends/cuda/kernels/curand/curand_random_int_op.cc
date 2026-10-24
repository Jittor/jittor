#include "curand_random_int_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <limits>
#ifdef JIT_cuda
#include <curand_kernel.h>
#endif

namespace jittor {
#ifndef JIT
CurandRandomIntOp::CurandRandomIntOp(NanoVector shape, int64 low, int64 high,
        bool high_none, RandomGenerator* generator) {
    // Only this native Philox owner is supported. No implicit host-generator
    // substitution, no CPU fallback and no second PRNG.
    USER_CHECK(construction_target_backend() == BackendId::Cuda)
        << "random_int64 requires the CUDA backend";
    USER_CHECK(generator && generator->state)
        << "random_int64 requires an explicit CUDA Generator";
    USER_CHECK(generator->state->device_id == current_device())
        << "random_int64 Generator and output device differ";
    USER_CHECK(high_none || low < high) << "random_int64 requires low < high";
    const int64 maximum = std::numeric_limits<int64>::max();
    const int64 minimum = std::numeric_limits<int64>::min();
    full_range = high_none && low == minimum;
    range = high_none ? uint64(maximum) - uint64(low) + uint64(1)
                      : uint64(high) - uint64(low);
    USER_CHECK(full_range || range != 0) << "random_int64 invalid range";
    base_bits = uint64(low);
    // Match the public non-FBCODE Torch CUDA raw-width policy, not Torch's
    // launch/subsequence allocation or serialized state. Modulo has the same
    // finite-width bias; this is deliberately not rejection sampling.
    raw_per_value = full_range || range >= (uint64(1) << 28) ? 2 : 1;
    int64 num = 1;
    for (uint i=0; i<shape.size(); ++i) {
        USER_CHECK(shape[i] >= 0) << "random_int64 negative dimension";
        int64 next;
        USER_CHECK(!__builtin_mul_overflow(num, int64(shape[i]), &next))
            << "random_int64 element count overflows int64";
        num = next;
    }
    USER_CHECK(num <= maximum / int64(sizeof(int64)))
        << "random_int64 byte count overflows int64";
    int64 raw;
    USER_CHECK(!__builtin_mul_overflow(num, int64(raw_per_value), &raw))
        << "random_int64 raw count overflows int64";
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    output = create_output(shape, ns_int64);
    USER_CHECK(output->num == num) << "random_int64 shape count mismatch";
    // Everything that can reject the request above precedes stream commit.
    philox_seed = uint64(generator->state->seed_);
    philox_offset = uint64(generator->state->reserve_cuda(raw));
}

void CurandRandomIntOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
}
#else
#ifdef JIT_cuda
// Each i owns exactly raw_per_value consecutive words of the existing
// subsequence-0 stream. No retry, alignment rounding, hidden curand4 tail,
// or random-word reads outside the reservation; launch shape is irrelevant.
__global__ static void philox_integer_kernel(int64* x, int64 num,
        unsigned long long seed, unsigned long long offset,
        unsigned long long base, unsigned long long range,
        int words, bool full) {
    const int64 first = int64(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64 stride = int64(gridDim.x) * blockDim.x;
    for (int64 i=first; i<num; i+=stride) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, 0, offset + uint64(i) * uint64(words), &state);
        uint64 raw = uint64(curand(&state));
        if (words == 2) raw = (raw << 32) | uint64(curand(&state));
        const uint64 bits = full ? raw : base + raw % range;
        // Both casts are representable. No signed subtraction/addition
        // overflow and no out-of-range unsigned-to-signed conversion.
        const uint64 sign = uint64(1) << 63;
        x[i] = bits < sign ? int64(bits)
                          : (-int64(0x7fffffffffffffffLL) - 1)
                            + int64(bits - sign);
    }
}

void CurandRandomIntOp::jit_run() {
    const int64 num = output->num;
    if (num == 0) return;
    const int block = 256;
    const int64 needed = (num + block - 1) / block;
    const unsigned grid = unsigned(needed < 65535 ? needed : 65535);
    philox_integer_kernel<<<grid, block>>>(output->ptr<int64>(), num,
        philox_seed, philox_offset, base_bits, range, raw_per_value, full_range);
    checkCudaErrors(cudaGetLastError());
}
#else
void CurandRandomIntOp::jit_run() {
    USER_CHECK(false) << "random_int64 has no CPU implementation";
}
#endif
#endif
} // namespace jittor
