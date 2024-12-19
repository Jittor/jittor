#pragma once
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
#include "dropout_op_acl.h"

namespace jittor
{
    DropoutOpRunner::DropoutOpRunner() : BaseOpRunner("Dropout")
    {
    }

    void DropoutOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
        ret = aclnnDropoutGetWorkspaceSize(inputTensors[0], attr->p, attr->train, attr->seed, attr->offset, outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnDropout(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnDropout failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }

    DropoutBackwardOpRunner::DropoutBackwardOpRunner() : BaseOpRunner("DropoutBackward")
    {
    }

    void DropoutBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
        ret = aclnnDropoutBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], attr->scale, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnDropoutBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnDropoutBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }

}