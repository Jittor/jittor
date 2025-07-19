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
#include "relu_op_acl.h"

namespace jittor
{
    LeakyReLUOpRunner::LeakyReLUOpRunner() : BaseOpRunner("LeakyReLU")
    {
    }

    void LeakyReLUOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        aclScalar *negativeSlope = nullptr;

        auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
        negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
        ret = aclnnLeakyReluGetWorkspaceSize(inputTensors[0], negativeSlope, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnLeakyRelu failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyScalar(negativeSlope);
        return;
    }

    LeakyReLUBackwardOpRunner::LeakyReLUBackwardOpRunner() : BaseOpRunner("LeakyReLUBackward")
    {
    }

    void LeakyReLUBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        aclScalar *negativeSlope = nullptr;

        auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
        negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
        ret = aclnnLeakyReluBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], negativeSlope, attr->selfIsResult, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnLeakyReluBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnLeakyReluBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyScalar(negativeSlope);
        return;
    }

}