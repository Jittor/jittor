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
#include <aclnnop/aclnn_index_copy.h>
#include "gather_scatter_op_acl.h"

namespace jittor
{
    GatherOpRunner::GatherOpRunner() : BaseOpRunner("Gather")
    {
    }

    void GatherOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
        ret = aclnnGatherGetWorkspaceSize(inputTensors[0], attr->dim, inputTensors[1], outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnGather, true);
        return;
    }

    IndexCopyOpRunner::IndexCopyOpRunner() : BaseOpRunner("IndexCopy")
    {
    }

    void IndexCopyOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
        // Some CANN IndexCopy paths also update selfRef (observed for dim=0
        // on 310P). Own the copy explicitly and only mutate the output.
        CHECK(in_[0]->is_contiguous());
        CHECK(in_[0]->size == out_[0]->size);
        auto copied = aclrtMemcpyAsync(out_[0]->mem_ptr, out_[0]->size,
            in_[0]->mem_ptr, in_[0]->size, ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
        CHECK(copied == ACL_SUCCESS) << "IndexCopy base copy failed: " << copied;
        ret = aclnnInplaceIndexCopyGetWorkspaceSize(outputTensors[0], attr->dim,
            inputTensors[1], inputTensors[2], &workspaceSize, &executor);
        launch(ret, aclnnInplaceIndexCopy, true);
    }

    ScatterOpRunner::ScatterOpRunner() : BaseOpRunner("Scatter")
    {
    }

    void ScatterOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<ScatterAttr *>(op_attr.get());
        ret = aclnnScatterGetWorkspaceSize(inputTensors[0], attr->axis, inputTensors[1], inputTensors[2], attr->reduction, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnScatter, true);
        return;
    }

}
