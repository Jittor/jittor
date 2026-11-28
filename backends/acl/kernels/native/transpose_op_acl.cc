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
#include "transpose_op_acl.h"

namespace jittor
{
    TransposeOpRunner::TransposeOpRunner() : BaseOpRunner("Transpose")
    {
    }

    void TransposeOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());

        aclIntArray *dim = nullptr;

        dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
        bool keepdims = attr->keepdims;

        ret = aclnnPermuteGetWorkspaceSize(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnPermute, true);

        aclDestroyIntArray(dim);

        return;
    }

    ContiguousPermuteOpRunner::ContiguousPermuteOpRunner() : BaseOpRunner("Transpose")
    {
    }

    void ContiguousPermuteOpRunner::setupInputDesc()
    {
        inputShapes.resize(1);
        inputShapes[0].assign(base_shape, base_shape + perm_rank);
        inputTensors.assign(1, nullptr);
        auto *v = in_[0];
        auto status = AcquireAclTensor(
            scratch->descriptors, inputShapes[0], v->mem_ptr, v->size,
            get_dtype(v->dtype()), &inputTensors[0], false, nullptr);
        if (status != ACL_SUCCESS)
            LOGf << name << ": contiguous permute input descriptor failed. ERROR:" << status;
    }

    void ContiguousPermuteOpRunner::setupOutputDesc()
    {
        outputShapes.resize(1);
        outputShapes[0].assign(out_shape, out_shape + perm_rank);
        outputTensors.assign(1, nullptr);
        auto *v = out_[0];
        auto status = AcquireAclTensor(
            scratch->descriptors, outputShapes[0], v->mem_ptr, v->size,
            get_dtype(v->dtype()), &outputTensors[0], false, nullptr);
        if (status != ACL_SUCCESS)
            LOGf << name << ": contiguous permute output descriptor failed. ERROR:" << status;
    }

    void ContiguousPermuteOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        aclIntArray *dim = aclCreateIntArray(perm, perm_rank);
        ret = aclnnPermuteGetWorkspaceSize(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnPermute, true);
        aclDestroyIntArray(dim);
    }

    ContiguousCopyOpRunner::ContiguousCopyOpRunner() : BaseOpRunner("Transpose")
    {
    }

    void ContiguousCopyOpRunner::setupInputDesc()
    {
        auto *v = in_[0];
        inputShapes.resize(1);
        inputShapes[0].resize(v->shape.size());
        for (size_t i = 0; i < v->shape.size(); ++i)
            inputShapes[0][i] = v->shape[i];
        inputTensors.assign(1, nullptr);
        auto status = AcquireAclTensor(
            scratch->descriptors, inputShapes[0], v->mem_ptr, v->size,
            get_dtype(v->dtype()), &inputTensors[0], false, v);
        if (status != ACL_SUCCESS)
            LOGf << name << ": contiguous copy input descriptor failed. ERROR:" << status;
    }

    void ContiguousCopyOpRunner::setupOutputDesc()
    {
        auto *v = out_[0];
        outputShapes.resize(1);
        outputShapes[0].resize(v->shape.size());
        for (size_t i = 0; i < v->shape.size(); ++i)
            outputShapes[0][i] = v->shape[i];
        outputTensors.assign(1, nullptr);
        auto status = AcquireAclTensor(
            scratch->descriptors, outputShapes[0], v->mem_ptr, v->size,
            get_dtype(v->dtype()), &outputTensors[0], false, nullptr);
        if (status != ACL_SUCCESS)
            LOGf << name << ": contiguous copy output descriptor failed. ERROR:" << status;
    }

    void ContiguousCopyOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnInplaceCopyGetWorkspaceSize(
            outputTensors[0], inputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnInplaceCopy, true);
    }

}
