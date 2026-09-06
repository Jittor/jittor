#include "var.h"
#include "ops/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_where(Var* condition, NanoString) {
    return backend_ops(accelerator_backend_id()).execution.prefer_compaction_kernel
        || condition->shape.size() > 1 || std::abs(condition->num) > 4096;
}

RegisterOpCapability<vector<VarPtr>, Var*, Var*, NanoString, bool> arg_reduce(
    accelerator_backend_id(), OpCapability::SegmentedArgReduce, "cub_arg_reduce");
RegisterOpCapability<vector<VarPtr>, Var*, Var*, Var*, bool, NanoString> argsort(
    accelerator_backend_id(), OpCapability::SegmentedArgsort, "cub_argsort");
RegisterOpCapability<vector<VarPtr>, Var*, NanoString> where(
    accelerator_backend_id(), OpCapability::Where, "cub_where", supports_where);
}
} // namespace jittor
#endif
