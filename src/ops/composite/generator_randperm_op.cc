#include <random>
#include "core/var.h"
#include "ops/composite/generator_randperm_op.h"
namespace jittor {
#ifndef JIT
GeneratorRandpermOp::GeneratorRandpermOp(int64 n, int64 seed, int64 offset, NanoString dtype) {
    USER_CHECK(n >= 0) << "generator_randperm expects non-negative n, got" << n;
    USER_CHECK(offset >= 0) << "generator_randperm expects non-negative offset, got" << offset;
    USER_CHECK(dtype == ns_int32 || dtype == ns_int64) << "generator_randperm expects int32 or int64, got" << dtype;
    this->n=n; this->seed=seed; this->offset=offset; output=create_output({n}, dtype); set_flag(OpFlags::_cpu);
}
void GeneratorRandpermOp::jit_prepare(JK& jk) { jk << "«T:" << output->dtype(); }
#else
#ifdef JIT_cpu
void GeneratorRandpermOp::jit_run() {
    std::mt19937 engine(static_cast<uint32>(seed)); engine.discard(offset);
    auto* out=output->ptr<T>();
    for (index_t i=0; i<n; ++i) out[i]=static_cast<T>(i);
    for (index_t i=0; i+1<n; ++i) { index_t z=i+engine()%(n-i); auto v=out[i]; out[i]=out[z]; out[z]=v; }
}
#else
#error "generator_randperm is CPU-only"
#endif
#endif
}
