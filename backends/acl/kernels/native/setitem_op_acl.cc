#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "setitem_op_acl.h"

namespace jittor
{
    InplaceMaskedScatterOpRunner::InplaceMaskedScatterOpRunner() : BaseOpRunner("InplaceMaskedScatter")
    {
    }

    void InplaceMaskedScatterOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // inputs: base(0), mask(1), value(2); output: out(0).
        // Copy base -> out on aclstream first (base is a tracked input, so its
        // data is materialized before this runs), then masked-scatter in place.
        // Previously the base (x.clone()) was a write-only output whose copy was
        // an untracked dependency -> non-masked elements could read a stale
        // buffer (wrong, run-order-dependent results).
        aclrtMemcpyAsync(out_[0]->mem_ptr, out_[0]->size,
                         in_[0]->mem_ptr, in_[0]->size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
        ret = aclnnInplaceMaskedScatterGetWorkspaceSize(outputTensors[0], inputTensors[1], inputTensors[2], &workspaceSize, &executor);

        launch(ret, aclnnInplaceMaskedScatter, true);
        return;
    }

    IndexPutImplOpRunner::IndexPutImplOpRunner() : BaseOpRunner("IndexPutImpl")
    {
    }

    void IndexPutImplOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // inputs: base(0), value(1), index0(2), index1(3), ...; output: out(0).
        // Copy base -> out on aclstream first (base is a tracked input, so its
        // data is materialized before this runs), then index-put in place on
        // out. Previously the base was a write-only `x.clone()` output whose
        // clone shares the base allocation, so the index-put mutated the base
        // buffer out of order; on view write-through (`x[:] = result`) the
        // native setitem planner then correctly rejects the self-overlapping
        // copy.
        CHECK(in_.size() >= 2);
        CHECK(in_[0]->is_contiguous());
        CHECK(in_[0]->size == out_[0]->size);
        aclrtMemcpyAsync(out_[0]->mem_ptr, out_[0]->size,
                         in_[0]->mem_ptr, in_[0]->size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
        auto input_num = in_.size();
        std::vector<aclTensor *> indexTensorList = {};
        for (int i = 2; i < input_num; i++)
        {
            indexTensorList.push_back(inputTensors[i]);
        }
        auto indexTensorListInput = aclCreateTensorList(indexTensorList.data(), input_num - 2);
        ret = aclnnIndexPutImplGetWorkspaceSize(outputTensors[0], indexTensorListInput, inputTensors[1], false, true, &workspaceSize, &executor);

        launch(ret, aclnnIndexPutImpl, true);
        return;
    }

}
