#pragma once
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
#include "pool_op_acl.h"
#include "aclnnop/aclnn_adaptive_avg_pool2d.h"
#include "aclnnop/aclnn_adaptive_avg_pool2d_backward.h"

namespace jittor
{
    AdaptiveAvgPool2dOpRunner::AdaptiveAvgPool2dOpRunner()
        : BaseOpRunner("AdaptiveAvgPool2d") { use_nchw = true; }

    void AdaptiveAvgPool2dOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<AdaptiveAvgPool2dAttr *>(op_attr.get());
        CHECK(attr);
        auto output_size = aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size());
        ret = aclnnAdaptiveAvgPool2dGetWorkspaceSize(inputTensors[0], output_size,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnAdaptiveAvgPool2d, true);
        aclDestroyIntArray(output_size);
    }

    AdaptiveAvgPool2dBackwardOpRunner::AdaptiveAvgPool2dBackwardOpRunner()
        : BaseOpRunner("AdaptiveAvgPool2dBackward") { use_nchw = true; }

    void AdaptiveAvgPool2dBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1],
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnAdaptiveAvgPool2dBackward, true);
    }

    MaxpoolOpRunner::MaxpoolOpRunner() : BaseOpRunner("Maxpool")
    {
        use_nchw = true;
    }

    void MaxpoolOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

        launch(ret, aclnnMaxPool2dWithIndices, true);

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

    void AvgpoolOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {

        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        ret = aclnnAvgPool2dGetWorkspaceSize(inputTensors[0], kernel_size, strides, pads, attr->poolCeil, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnAvgPool2d, true);

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(kernel_size);

        return;
    }

    MaxpoolBackwardOpRunner::MaxpoolBackwardOpRunner() : BaseOpRunner("MaxpoolBackward")
    {
        use_nchw = true;
    }

    void MaxpoolBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

        launch(ret, aclnnMaxPool2dWithIndicesBackward, true);

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

    void AvgpoolBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *kernel_size = nullptr;

        auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
        kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
        strides = aclCreateIntArray(attr->poolStrides.data(), 2);
        pads = aclCreateIntArray(attr->poolPads.data(), 2);
        ret = aclnnAvgPool2dBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], kernel_size, strides, pads, attr->poolCeil, attr->countIncludePad, attr->divisorOverride, 0, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnAvgPool2dBackward, true);

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(kernel_size);

        return;
    }

}
