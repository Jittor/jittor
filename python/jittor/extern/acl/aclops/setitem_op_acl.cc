#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "ops/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/transpose_op.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "executor.h"
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "setitem_op_acl.h"

namespace jittor
{
    InplaceMaskedScatterOpRunner::InplaceMaskedScatterOpRunner() : BaseOpRunner("InplaceMaskedScatter")
    {
    }

    void InplaceMaskedScatterOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
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

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnInplaceMaskedScatter(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnInplaceMaskedScatter failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

    IndexPutImplOpRunner::IndexPutImplOpRunner() : BaseOpRunner("IndexPutImpl")
    {
    }

    void IndexPutImplOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto input_num = in_.size();
        std::vector<aclTensor *> indexTensorList = {};
        for (int i = 1; i < input_num; i++)
        {
            indexTensorList.push_back(inputTensors[i]);
        }
        auto indexTensorListInput = aclCreateTensorList(&indexTensorList[0], input_num - 1);
        ret = aclnnIndexPutImplGetWorkspaceSize(outputTensors[0], indexTensorListInput, inputTensors[0], false, true, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnIndexPutImpl failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

}