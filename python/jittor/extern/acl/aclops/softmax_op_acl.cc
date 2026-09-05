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
#include "runtime/device.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "softmax_op_acl.h"

namespace jittor
{
    SoftmaxOpRunner::SoftmaxOpRunner() : BaseOpRunner("Softmax")
    {
    }

    void SoftmaxOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<SoftmaxAttr *>(op_attr.get());
        ret = aclnnSoftmaxGetWorkspaceSize(inputTensors[0], aclDataType(attr->dim), outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSoftmax, true);

        return;
    }

    SoftmaxBackwardOpRunner::SoftmaxBackwardOpRunner() : BaseOpRunner("SoftmaxBackward")
    {
    }

    void SoftmaxBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<SoftmaxAttr *>(op_attr.get());
        ret = aclnnSoftmaxBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], attr->dim, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSoftmaxBackward, true);

        return;
    }

}
