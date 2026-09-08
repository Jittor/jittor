#include "ops/composite/contiguous_op.h"
#include "core/var.h"

namespace jittor {
#ifndef JIT
ContiguousOp::ContiguousOp(Var* input) : x(input) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::element);
    set_flag(OpFlags::_manual_set_vnbb);
    if (x->is_contiguous()) { forward(x); return; }
    y = create_output(nullptr, x->dtype());
}
void ContiguousOp::infer_shape() { y->set_shape(x->shape); }
VarPtr ContiguousOp::grad(Var*, Var* dout, Var*, int) { return dout; }
void ContiguousOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype() << "«DIM=" << JK::hex1(x->shape.size());
}
#else
void ContiguousOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    @for(d, 0, DIM, index_t xshape@d = x->shape[@d];)
    @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
    index_t num = y->num;
    for (index_t i=0; i<num; ++i) {
        index_t rem = i;
        index_t offset = 0;
        @for(d, DIM-1, -1, -1,
            offset += (rem % xshape@d) * xstride@d;
            rem /= xshape@d;
        )
        yp[i] = xp[offset];
    }
}
#endif
}
