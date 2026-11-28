#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include <cstring>
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
#include "aclnnop/aclnn_rms_norm_grad.h"
#include "norms_op_acl.h"

namespace jittor
{
    static void setupNormTensorDescs(
        const string &name,
        vector<aclTensor *> &pool,
        const vector<Var *> &vars,
        vector<vector<int64_t>> &shapes,
        vector<aclTensor *> &tensors,
        int nchwPrefix)
    {
        const size_t n = vars.size();
        shapes.resize(n);
        for (size_t idx = 0; idx < n; idx++)
        {
            auto &shape = shapes[idx];
            const auto &vs = vars[idx]->shape;
            shape.resize(vs.size());
            for (size_t j = 0; j < vs.size(); j++)
                shape[j] = vs[j];
        }
        tensors.resize(n, nullptr);
        for (size_t idx = 0; idx < n; idx++)
        {
            tensors[idx] = nullptr;
            auto ret = AcquireAclTensor(
                pool, shapes[idx], vars[idx]->mem_ptr, vars[idx]->size,
                get_dtype(vars[idx]->dtype()), &tensors[idx],
                static_cast<int>(idx) < nchwPrefix);
            if (ret != ACL_SUCCESS) LOGf << name << ": normalization input tensor creation failed. ERROR:" << ret;
        }
    }

    BatchNormOpRunner::BatchNormOpRunner() : BaseOpRunner("BatchNorm")
    {
    }

    void BatchNormOpRunner::setupInputDesc()
    {
        setupNormTensorDescs(name, scratch->descriptors, in_, inputShapes, inputTensors, 1);
    }

    void BatchNormOpRunner::setupOutputDesc()
    {
        setupNormTensorDescs(name, scratch->descriptors, out_, outputShapes, outputTensors, 1);
    }

    void BatchNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<BatchNormAttr *>(op_attr.get());
        aclTensor *running_mean_tensor = inputTensors[3];
        aclTensor *running_var_tensor = inputTensors[4];
        if (outputTensors.size() >= 5)
        {
            if (out_[3]->mem_ptr != in_[3]->mem_ptr && out_[3]->size > 0)
            {
                auto st = aclrtMemcpyAsync(
                    out_[3]->mem_ptr, out_[3]->size, in_[3]->mem_ptr, out_[3]->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << name << ": running_mean D2D copy failed. ERROR:" << st;
            }
            if (out_[4]->mem_ptr != in_[4]->mem_ptr && out_[4]->size > 0)
            {
                auto st = aclrtMemcpyAsync(
                    out_[4]->mem_ptr, out_[4]->size, in_[4]->mem_ptr, out_[4]->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << name << ": running_var D2D copy failed. ERROR:" << st;
            }
            running_mean_tensor = outputTensors[3];
            running_var_tensor = outputTensors[4];
        }
        ret = aclnnBatchNormGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            running_mean_tensor, running_var_tensor,
            attr->is_train, attr->momentum, attr->eps,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);

        launch(ret, aclnnBatchNorm, true);

        return;
    }

    BatchNormBackwardOpRunner::BatchNormBackwardOpRunner() : BaseOpRunner("BatchNormBackward")
    {
    }

    void BatchNormBackwardOpRunner::setupInputDesc()
    {
        setupNormTensorDescs(name, scratch->descriptors, in_, inputShapes, inputTensors, 2);
    }

    void BatchNormBackwardOpRunner::setupOutputDesc()
    {
        setupNormTensorDescs(name, scratch->descriptors, out_, outputShapes, outputTensors, 1);
    }

    void BatchNormBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<BatchNormAttr *>(op_attr.get());
        bool outputMask[3] = {true, true, true};
        aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
        ret = aclnnBatchNormBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], inputTensors[4], inputTensors[5], inputTensors[6], attr->is_train, attr->eps, outMask, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);

        launch(ret, aclnnBatchNormBackward, true);
        aclDestroyBoolArray(outMask);

        return;
    }

    LayerNormOpRunner::LayerNormOpRunner() : BaseOpRunner("LayerNorm")
    {
    }

    void LayerNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

    void LayerNormBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

        launch(ret, aclnnLayerNormBackward, true);
        aclDestroyIntArray(normalizedShape);
        aclDestroyBoolArray(outMask);

        return;
    }

    GroupNormOpRunner::GroupNormOpRunner() : BaseOpRunner("GroupNorm")
    {
    }

    void GroupNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<GroupNormAttr *>(op_attr.get());
        ret = aclnnGroupNormGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            attr->batch, attr->channels, attr->spatialSize, attr->groups,
            attr->eps, outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnGroupNorm, true);
    }

    GroupNormBackwardOpRunner::GroupNormBackwardOpRunner()
        : BaseOpRunner("GroupNormBackward")
    {
    }

    void GroupNormBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<GroupNormAttr *>(op_attr.get());
        bool outputMaskValues[3] = {true, true, true};
        std::unique_ptr<aclBoolArray, decltype(&aclDestroyBoolArray)> outputMask(
            aclCreateBoolArray(outputMaskValues, 3), aclDestroyBoolArray);
        if (!outputMask) LOGf << name << ": aclCreateBoolArray failed";
        ret = aclnnGroupNormBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3],
            inputTensors[4], attr->batch, attr->channels, attr->spatialSize,
            attr->groups, outputMask.get(), outputTensors[0], outputTensors[1],
            outputTensors[2], &workspaceSize, &executor);
        launch(ret, aclnnGroupNormBackward, true);
    }

    RmsNormOpRunner::RmsNormOpRunner() : BaseOpRunner("RmsNorm")
    {
    }

    void RmsNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        const auto *input = in_[0];
        const auto rank = input->shape.size();
        if (input->dtype() == ns_float16 && rank
                && input->shape[rank - 1] > 0 && input->shape[rank - 1] < 16
                && input->num > input->shape[rank - 1]
                && std::strncmp(acl_soc_name(), "Ascend310P", 10) == 0)
            LOGf << "RmsNorm on Ascend310P rejects float16 normalized widths below 16"
                 << "with multiple rows: the vendor kernel can return incorrect rows;"
                 << "use the portable RMSNorm path";
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

    void RmsNormGradOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnRmsNormGradGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3],
            outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        launch(ret, aclnnRmsNormGrad, true);

        return;
    }

}
