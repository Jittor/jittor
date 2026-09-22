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
    // Which axes move the physical index at all. A view's zero-stride axes
    // contribute nothing to the offset, so saying so in the key lets the copy
    // skip them instead of dividing them away once per element; an all-zero
    // pattern (a scalar broadcast) then reads element 0 for every i. There is
    // deliberately no `offset = i` shortcut here, unlike the elementwise ops:
    // this kernel may be handed a view whose every stride is zero, where `i`
    // would index past the four bytes the source actually owns.
    jk << "«XSMASK=" << JK::hex(x->stride_pattern());
}
#else
void ContiguousOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    @for(d, 0, DIM, index_t xshape@d = x->shape[@d];)
    @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
    // The product of the shapes above each axis, so one axis' index is a single
    // division away (`i / xabove@d % xshape@d`) instead of a chain of them:
    // `(i/a)/b == i/(a*b)` for non-negative integers. See KI-CODEGEN-001, and
    // keep the macro arguments comma-free -- the parser splits them on commas.
    index_t xabove@{DIM-1} = 1;
    @for(d, DIM-2, -1, -1, index_t xabove@d = xabove@{d+1} * xshape@{d+1};)
    index_t num = y->num;
    for (index_t i=0; i<num; ++i) {
        index_t offset = 0;
        @for(d, 0, DIM,
            @if(XSMASK>>d&1,
                offset += @if(d, (i / xabove@d % xshape@d), (i / xabove@d)) * xstride@d;
            )
        )
        yp[i] = xp[offset];
    }
}
#endif
}
