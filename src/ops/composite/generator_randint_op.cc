#include <cstdint>
#include <limits>
#include "core/var.h"
#include "ops/composite/generator_randint_op.h"
#include "utils/philox.h"

namespace jittor {

#ifndef JIT
GeneratorRandintOp::GeneratorRandintOp(NanoVector shape, NanoString dtype,
        int64 low, int64 high, int64 seed, int64 offset)
    : low(low), high(high), seed(seed), offset(offset) {
    USER_CHECK(dtype == ns_int32 || dtype == ns_int64)
        << "generator_randint supports int32 and int64, got " << dtype;
    USER_CHECK(high > low)
        << "generator_randint expects high > low, got [" << low << ", " << high << ")";
    if (dtype == ns_int32) {
        USER_CHECK(low >= std::numeric_limits<int32_t>::min()
            && high <= int64(std::numeric_limits<int32_t>::max()) + 1)
            << "generator_randint bounds do not fit int32";
    }
    USER_CHECK(offset >= 0) << "generator_randint expects non-negative offset, got " << offset;
    output = create_output(shape, dtype);
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
}

void GeneratorRandintOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
}

#else

#ifdef JIT_cpu
static inline uint64 generator_randint_sample(uint64 seed,
        uint64 element, uint64 span, uint64 threshold, uint64 attempt=0) {
    const int pair = int(element & 1);
    const uint64 block_index = element >> 1;
    while (true) {
        auto block = philox4x32_10(seed, block_index, attempt++);
        auto product = philox_mul_wide(philox_uint64(block, pair), span);
        if (product.low >= threshold) return product.high;
    }
}

void GeneratorRandintOp::jit_run() {
    auto* out = output->ptr<T>();
    const uint64 seed_value = uint64(seed);
    const uint64 span = uint64(high) - uint64(low);
    const uint64 threshold = (uint64(0) - span) % span;
    index_t i = 0;

    // Align to an even logical element, then consume both 64-bit lanes of
    // each Philox block. Each element keeps a stable counter/lane mapping, so
    // splitting a draw across calls produces the same stream.
    if (i < output->num && ((uint64(offset) + uint64(i)) & 1)) {
        out[i] = T(uint64(low) + generator_randint_sample(seed_value,
            uint64(offset) + uint64(i), span, threshold));
        ++i;
    }
    for (; i + 1 < output->num; i += 2) {
        const uint64 element = uint64(offset) + uint64(i);
        const uint64 block_index = element >> 1;
        auto block = philox4x32_10(seed_value, block_index, 0);
        auto product0 = philox_mul_wide(philox_uint64(block, 0), span);
        auto product1 = philox_mul_wide(philox_uint64(block, 1), span);
        if (product0.low < threshold) {
            product0.high = generator_randint_sample(
                seed_value, element, span, threshold, 1);
        }
        if (product1.low < threshold) {
            product1.high = generator_randint_sample(
                seed_value, element + 1, span, threshold, 1);
        }
        out[i] = T(uint64(low) + product0.high);
        out[i + 1] = T(uint64(low) + product1.high);
    }
    if (i < output->num) {
        out[i] = T(uint64(low) + generator_randint_sample(seed_value,
            uint64(offset) + uint64(i), span, threshold));
    }
}
#else
#error "generator_randint accelerator kernels must be provided by the backend"
#endif

#endif
}
