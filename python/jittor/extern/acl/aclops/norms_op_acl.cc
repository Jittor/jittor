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
#include "norms_op_acl.h"

namespace jittor
{
    BatchNormOpRunner::BatchNormOpRunner() : BaseOpRunner("BatchNorm")
    {
    }

    void BatchNormOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<BatchNormAttr *>(op_attr.get());
        ret = aclnnBatchNormGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], inputTensors[4], attr->is_train, attr->momentum, attr->eps, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnBatchNorm(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnBatchNorm failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }

    BatchNormBackwardOpRunner::BatchNormBackwardOpRunner() : BaseOpRunner("BatchNormBackward")
    {
    }

    void BatchNormBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<BatchNormAttr *>(op_attr.get());
        bool outputMask[3] = {true, true, true};
        aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
        ret = aclnnBatchNormBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], inputTensors[4], inputTensors[5], inputTensors[6], attr->is_train, attr->eps, outMask, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnBatchNormBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }

    LayerNormOpRunner::LayerNormOpRunner() : BaseOpRunner("LayerNorm")
    {
    }

    void LayerNormOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<LayerNormAttr *>(op_attr.get());
        aclIntArray *normalizedShape = nullptr;
        normalizedShape = aclCreateIntArray(attr->normalizedShape.data(), attr->size);
        ret = aclnnLayerNormGetWorkspaceSize(inputTensors[0], normalizedShape, inputTensors[1], inputTensors[2], attr->eps, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnLayerNorm(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnLayerNorm failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        aclDestroyIntArray(normalizedShape);

        return;
    }

}