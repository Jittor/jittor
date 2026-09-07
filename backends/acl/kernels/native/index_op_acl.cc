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
#include "index_op_acl.h"

namespace jittor
{
    RangeOpRunner::RangeOpRunner() : BaseOpRunner("Range")
    {
    }

    void RangeOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        aclScalar *start = nullptr;
        aclScalar *end = nullptr;
        aclScalar *step = nullptr;

        auto attr = dynamic_cast<RangeAttr *>(op_attr.get());
        int64_t startValue = attr->start;
        int64_t endValue = attr->end;
        int64_t stepValue = attr->step;
        start = aclCreateScalar(&startValue, aclDataType::ACL_INT64);
        end = aclCreateScalar(&endValue, aclDataType::ACL_INT64);
        step = aclCreateScalar(&stepValue, aclDataType::ACL_INT64);

        ret = aclnnRangeGetWorkspaceSize(start, end, step, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnRange, true);

        aclDestroyScalar(start);
        aclDestroyScalar(end);
        aclDestroyScalar(step);
        return;
    }

}
