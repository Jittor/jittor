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

        launch(ret, aclnnDropout, true);

        return;
    }

    DropoutBackwardOpRunner::DropoutBackwardOpRunner() : BaseOpRunner("DropoutBackward")
    {
    }

    void DropoutBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
        ret = aclnnDropoutBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], attr->scale, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnDropoutBackward, true);

        return;
    }

}
