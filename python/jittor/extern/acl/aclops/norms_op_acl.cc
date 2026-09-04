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
#include "aclnnop/aclnn_rms_norm_grad.h"
#include "norms_op_acl.h"

namespace jittor
{
    static void setupNormTensorDescs(
        const vector<Var *> &vars,
        vector<vector<int64_t>> &shapes,
        vector<aclTensor *> &tensors,
        int nchwPrefix)
    {
        for (auto *var : vars)
        {
            vector<int64_t> shape;
            for (int j = 0; j < var->shape.size(); j++)
                shape.push_back(var->shape[j]);
            shapes.push_back(shape);
        }
        for (int idx = 0; idx < vars.size(); idx++)
        {
            tensors.push_back(nullptr);
            auto ret = CreateAclTensor(
                shapes[idx], vars[idx]->mem_ptr, vars[idx]->size,
                get_dtype(vars[idx]->dtype()), &tensors[idx],
                idx < nchwPrefix);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    BatchNormOpRunner::BatchNormOpRunner() : BaseOpRunner("BatchNorm")
    {
    }

    void BatchNormOpRunner::setupInputDesc()
    {
        setupNormTensorDescs(in_, inputShapes, inputTensors, 1);
    }

    void BatchNormOpRunner::setupOutputDesc()
    {
        setupNormTensorDescs(out_, outputShapes, outputTensors, 1);
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

    void BatchNormBackwardOpRunner::setupInputDesc()
    {
        setupNormTensorDescs(in_, inputShapes, inputTensors, 2);
    }

    void BatchNormBackwardOpRunner::setupOutputDesc()
    {
        setupNormTensorDescs(out_, outputShapes, outputTensors, 1);
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
        aclDestroyBoolArray(outMask);

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

        launch(ret, aclnnLayerNorm, true);
        aclDestroyIntArray(normalizedShape);

        return;
    }

    LayerNormBackwardOpRunner::LayerNormBackwardOpRunner() : BaseOpRunner("LayerNormBackward")
    {
    }

    void LayerNormBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        // inputs : gradOut(0), input(1), mean(2), rstd(3), weight(4), bias(5)
        // outputs: gradInput(0), gradWeight(1), gradBias(2)
        auto attr = dynamic_cast<LayerNormAttr *>(op_attr.get());
        aclIntArray *normalizedShape = aclCreateIntArray(attr->normalizedShape.data(), attr->size);
        bool outputMask[3] = {true, true, true};
        aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);

        ret = aclnnLayerNormBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], normalizedShape, inputTensors[2],
            inputTensors[3], inputTensors[4], inputTensors[5], outMask,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnLayerNormBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnLayerNormBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        aclDestroyIntArray(normalizedShape);
        aclDestroyBoolArray(outMask);

        return;
    }

    GroupNormOpRunner::GroupNormOpRunner() : BaseOpRunner("GroupNorm")
    {
    }

    void GroupNormOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<GroupNormAttr *>(op_attr.get());
        ret = aclnnGroupNormGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            attr->batch, attr->channels, attr->spatialSize, attr->groups,
            attr->eps, outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        checkRet(ret);
        if (ret != ACL_SUCCESS)
            return;

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnGroupNorm(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnGroupNorm failed. ERROR: %d\n", name.c_str(), ret); return);
        syncRun();
    }

    GroupNormBackwardOpRunner::GroupNormBackwardOpRunner()
        : BaseOpRunner("GroupNormBackward")
    {
    }

    void GroupNormBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<GroupNormAttr *>(op_attr.get());
        bool outputMaskValues[3] = {true, true, true};
        std::unique_ptr<aclBoolArray, decltype(&aclDestroyBoolArray)> outputMask(
            aclCreateBoolArray(outputMaskValues, 3), aclDestroyBoolArray);
        CHECK_RET(outputMask != nullptr,
                  LOG_PRINT("%s: aclCreateBoolArray failed.\n", name.c_str()); return);
        ret = aclnnGroupNormBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3],
            inputTensors[4], attr->batch, attr->channels, attr->spatialSize,
            attr->groups, outputMask.get(), outputTensors[0], outputTensors[1],
            outputTensors[2], &workspaceSize, &executor);
        checkRet(ret);
        if (ret != ACL_SUCCESS)
            return;

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnGroupNormBackward(
            workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnGroupNormBackward failed. ERROR: %d\n", name.c_str(), ret); return);
        syncRun();
    }

    RmsNormOpRunner::RmsNormOpRunner() : BaseOpRunner("RmsNorm")
    {
    }

    void RmsNormOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<RmsNormAttr *>(op_attr.get());
        ret = aclnnRmsNormGetWorkspaceSize(
            inputTensors[0], inputTensors[1], attr->eps,
            outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        launch(ret, aclnnRmsNorm, true);

        return;
    }

    RmsNormGradOpRunner::RmsNormGradOpRunner() : BaseOpRunner("RmsNormGrad")
    {
    }

    void RmsNormGradOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        ret = aclnnRmsNormGradGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3],
            outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        launch(ret, aclnnRmsNormGrad, true);

        return;
    }

}
