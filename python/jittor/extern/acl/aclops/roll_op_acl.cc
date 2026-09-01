#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "common.h"
#include "acl_jittor.h"
#include "mem/allocator.h"
#include "aclnnop/aclnn_roll.h"
#include "roll_op_acl.h"

namespace jittor
{
    RollOpRunner::RollOpRunner() : BaseOpRunner("Roll")
    {
    }

    void RollOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto shifts_array = aclCreateIntArray(shifts.data(), shifts.size());
        auto dims_array = aclCreateIntArray(dims.data(), dims.size());
        ret = aclnnRollGetWorkspaceSize(
            inputTensors[0], shifts_array, dims_array, outputTensors[0],
            &workspaceSize, &executor);
        checkRet(ret);

        if (workspaceSize > 0)
            mallocWorkSpace(workspaceSize);

        ret = aclnnRoll(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("%s: aclnnRoll failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        aclDestroyIntArray(dims_array);
        aclDestroyIntArray(shifts_array);
    }
}
