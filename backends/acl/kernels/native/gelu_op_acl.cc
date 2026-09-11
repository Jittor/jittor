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
    // The exact (erf) GELU. Jittor's portable definition is five elementwise
    // ops -- half * x * (1 + erf(x * inv_sqrt2)) -- and the ACL fused path
    // issues one aclnn launch per elementwise node, so a model that uses GELU
    // paid five launches plus their gradients per activation.
    GeluOpRunner::GeluOpRunner() : BaseOpRunner("Gelu")
    {
    }

    void GeluOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // approximate 0 is the exact erf form, which is what jittor's
        // approximate="none" means; aclnnGelu (v1) is the tanh approximation
        // and disagrees with it by ~5e-4. aclnnGeluBackward is already exact.
        ret = aclnnGeluV2GetWorkspaceSize(inputTensors[0], 0, outputTensors[0],
                                          &workspaceSize, &executor);
        launch(ret, aclnnGeluV2, true);
        return;
    }

    GeluBackwardOpRunner::GeluBackwardOpRunner() : BaseOpRunner("GeluBackward")
    {
    }

    void GeluBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // Same story as the forward: the v1 backward computes the tanh
        // approximation, so name the exact form explicitly. The parameter is a
        // non-const char*, hence the local buffer.
        static char approximate[] = "none";
        ret = aclnnGeluBackwardV2GetWorkspaceSize(inputTensors[0], inputTensors[1], approximate,
                                                  outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnGeluBackwardV2, true);
        return;
    }
}
