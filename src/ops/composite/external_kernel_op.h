#pragma once
#include "core/op.h"
#include "core/kernel_func.h"

namespace jittor {
// A fusion boundary with explicit read operands and freshly allocated writes.
// Compilation and launch belong to the provider; scheduling and storage belong
// to the executor, exactly as for other native operators.
struct ExternalKernelOp : Op {
    static constexpr bool accepts_storage_strides = true;
    vector<Var*> read_buffers, write_buffers;
    KernelFunc launch;
    string backend;

    // @attrs(multiple_outputs)
    ExternalKernelOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes,
                     vector<Var*>&& inputs, KernelFunc&& launch, string backend);
    const char* name() const override { return "external_kernel"; }
    void run() override;
    VarPtr grad(Var*, Var*, Var*, int) override;
};
} // namespace jittor
