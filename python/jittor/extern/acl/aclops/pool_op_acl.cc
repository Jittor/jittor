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
#include "pool_op_acl.h"

namespace jittor
{
    MaxpoolOpRunner::MaxpoolOpRunner() : BaseOpRunner("Maxpool")
    {
        use_nchw = true;
    }

    void MaxpoolOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {

        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *dilations = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
        ret = aclnnMaxPool2dWithIndicesGetWorkspaceSize(inputTensors[0], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnMaxPool2dWithIndices(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMaxPool2dWithIndices failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(dilations);
        aclDestroyIntArray(kernel_size);

        return;
    }


    AvgpoolOpRunner::AvgpoolOpRunner() : BaseOpRunner("Avgpool")
    {
        use_nchw = true;
    }

    void AvgpoolOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {

        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        ret = aclnnAvgPool2dGetWorkspaceSize(inputTensors[0], kernel_size, strides, pads, attr->poolCeil, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnAvgPool2d failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(kernel_size);

        return;
    }


    MaxpoolBackwardOpRunner::MaxpoolBackwardOpRunner() : BaseOpRunner("MaxpoolBackward")
    {
        use_nchw = true;
    }

    void MaxpoolBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {

        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *dilations = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
        ret = aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnMaxPool2dWithIndicesBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMaxPool2dWithIndicesBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(dilations);
        aclDestroyIntArray(kernel_size);

        return;
    }



    AvgpoolBackwardOpRunner::AvgpoolBackwardOpRunner() : BaseOpRunner("AvgpoolBackward")
    {
        use_nchw = true;
    }

    void AvgpoolBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        ret = aclnnAvgPool2dBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], kernel_size, strides, pads, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnAvgPool2dBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnAvgPool2dBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(kernel_size);

        return;
    }

}