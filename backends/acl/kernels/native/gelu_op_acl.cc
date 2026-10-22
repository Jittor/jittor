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
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_gelu_v2.h>
#include <aclnnop/aclnn_gelu_backward.h>
#include <aclnnop/aclnn_gelu_backward_v2.h>
#include "gelu_op_acl.h"

namespace jittor
{
    // V2 names both public GELU algorithms explicitly. The portable formulas
    // issue an ACL launch for each elementwise node rather than one GELU.
    GeluOpRunner::GeluOpRunner() : BaseOpRunner("Gelu")
    {
    }

    void GeluOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto *attr = dynamic_cast<GeluAttr *>(op_attr.get());
        const int64_t approximate = attr ? attr->approximate : 0;
        CHECK(approximate == 0 || approximate == 1);
        ret = aclnnGeluV2GetWorkspaceSize(inputTensors[0], approximate, outputTensors[0],
                                          &workspaceSize, &executor);
        launch(ret, aclnnGeluV2, true);
        return;
    }

    GeluBackwardOpRunner::GeluBackwardOpRunner() : BaseOpRunner("GeluBackward")
    {
    }

    void GeluBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto *attr = dynamic_cast<GeluAttr *>(op_attr.get());
        const int64_t mode = attr ? attr->approximate : 0;
        CHECK(mode == 0 || mode == 1);
        // CANN's backward API takes a non-const string, unlike the integer
        // enum used by its forward API. Each runner receives the same mode.
        static char exact[] = "none";
        static char tanh[] = "tanh";
        char *approximate = mode ? tanh : exact;
        ret = aclnnGeluBackwardV2GetWorkspaceSize(inputTensors[0], inputTensors[1], approximate,
                                                  outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnGeluBackwardV2, true);
        return;
    }
}
