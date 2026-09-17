#include <random>
#include "core/var.h"
#include "ops/composite/generator_randint_op.h"

namespace jittor {
#ifndef JIT
GeneratorRandintOp::GeneratorRandintOp(NanoVector shape, int64 low, int64 high,
                                       int64 seed, int64 offset, NanoString dtype) {
    USER_CHECK(low < high) << "generator_randint expects low < high, got"
        << low << high;
    USER_CHECK(static_cast<uint64>(high) - static_cast<uint64>(low)
               <= (uint64(1) << 32))
        << "generator_randint supports ranges no wider than 2^32, got"
        << (static_cast<uint64>(high) - static_cast<uint64>(low));
    USER_CHECK(offset >= 0) << "generator_randint expects non-negative offset, got"
        << offset;
    USER_CHECK(dtype == ns_int32 || dtype == ns_int64)
        << "generator_randint expects int32 or int64, got" << dtype;
    this->low = low;
    this->high = high;
    this->seed = seed;
    this->offset = offset;
    output = create_output(shape, dtype);
    set_flag(OpFlags::_cpu);
}

void GeneratorRandintOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
}
#else
#ifdef JIT_cpu
void GeneratorRandintOp::jit_run() {
    std::mt19937 engine(static_cast<uint32>(seed));
    engine.discard(offset);
    const uint64 range = static_cast<uint64>(high) - static_cast<uint64>(low);
    auto* out = output->ptr<T>();
    for (index_t i = 0; i < output->num; ++i) {
        const uint64 sample = static_cast<uint64>(engine()) % range;
        out[i] = static_cast<T>(low + static_cast<int64>(sample));
    }
}
#else
#error "generator_randint is CPU-only"
#endif
#endif
}
