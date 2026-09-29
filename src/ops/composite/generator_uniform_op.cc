#include <random>
#include <cstring>
#include "core/var.h"
#include "ops/composite/generator_uniform_op.h"
namespace jittor {
#ifndef JIT
GeneratorUniformOp::GeneratorUniformOp(NanoVector shape, NanoString dtype, double low, double high, int64 seed, int64 offset, int64 precision_bits) {
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64) << "generator_uniform expects float32 or float64, got" << dtype;
    USER_CHECK(low <= high) << "uniform bounds must satisfy low <= high";
    USER_CHECK(offset >= 0) << "generator_uniform expects non-negative offset, got" << offset;
    USER_CHECK(precision_bits > 0 && precision_bits <= 53) << "invalid uniform precision" << precision_bits;
    this->seed=seed; this->offset=offset; this->precision_bits=precision_bits; this->low=low; this->high=high; output=create_output(shape,dtype); set_flag(OpFlags::_cpu);
}
void GeneratorUniformOp::jit_prepare(JK& jk) { jk << "«T:" << output->dtype(); jk << "«B:" << precision_bits; }
#else
#ifdef JIT_cpu
void GeneratorUniformOp::jit_run() {
    std::mt19937 engine(static_cast<uint32>(seed)); engine.discard(offset);
    auto* out=output->ptr<T>();
    for (index_t i=0; i<output->num; ++i) {
        uint64 value=engine();
        if (precision_bits > 32) value=(value<<32)|engine();
        uint64 mask=(uint64(1)<<precision_bits)-1;
        T divisor=T(1.0)/T(uint64(1)<<precision_bits);
        T unit=T(value&mask)*divisor;
        T result=unit*T(high-low)+T(low);
        if (precision_bits == 8) {
            float rounded=static_cast<float>(result);
            uint32 bits;
            std::memcpy(&bits, &rounded, sizeof(bits));
            bits += 0x7fff + ((bits >> 16) & 1);
            bits &= 0xffff0000;
            std::memcpy(&rounded, &bits, sizeof(bits));
            result=static_cast<T>(rounded);
        }
        out[i]=result;
    }
}
#else
#error "generator_uniform is CPU-only"
#endif
#endif
}
