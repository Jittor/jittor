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
#include <aclnnop/aclnn_swish.h>
#include <aclnnop/aclnn_swish_backward.h>
#include <aclnnop/aclnn_swi_glu.h>
#include "silu_op_acl.h"

namespace jittor
{
    SiLUOpRunner::SiLUOpRunner() : BaseOpRunner("SiLU")
    {
    }

    void SiLUOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnSiluGetWorkspaceSize(inputTensors[0], outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSilu, true);

        return;
    }

    SiLUBackwardOpRunner::SiLUBackwardOpRunner() : BaseOpRunner("SiLUBackward")
    {
    }

    void SiLUBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnSiluBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSiluBackward, true);

        return;
    }

    SwishOpRunner::SwishOpRunner() : BaseOpRunner("Swish")
    {
    }

    void SwishOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnSwishGetWorkspaceSize(
            inputTensors[0], nullptr, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSwish, true);

        return;
    }

    SwishBackwardOpRunner::SwishBackwardOpRunner() : BaseOpRunner("SwishBackward")
    {
    }

    void SwishBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnSwishBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], nullptr, outputTensors[0],
            &workspaceSize, &executor);

        launch(ret, aclnnSwishBackward, true);

        return;
    }

    SwiGluOpRunner::SwiGluOpRunner() : BaseOpRunner("SwiGlu")
    {
    }

    void SwiGluOpRunner::executeOp(
        AclOpRegistry::const_iterator &it)
    {
        ret = aclnnSwiGluGetWorkspaceSize(
            inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnSwiGlu, true);

        return;
    }

    SoftplusOpRunner::SoftplusOpRunner() : BaseOpRunner("Softplus")
    {
    }

    void SoftplusOpRunner::executeOp(
        AclOpRegistry::const_iterator &it)
    {
        struct ScopedAclScalar
        {
            aclScalar *ptr = nullptr;
            ~ScopedAclScalar()
            {
                if (ptr)
                    aclDestroyScalar(ptr);
            }
        };
        auto *attr = dynamic_cast<SoftplusAttr *>(op_attr.get());
        float beta_val = attr ? static_cast<float>(attr->beta) : 1.0f;
        float threshold_val = attr ? static_cast<float>(attr->threshold) : 20.0f;
        ScopedAclScalar beta_scalar{aclCreateScalar(&beta_val, aclDataType::ACL_FLOAT)};
        ScopedAclScalar threshold_scalar{aclCreateScalar(&threshold_val, aclDataType::ACL_FLOAT)};
        ret = aclnnSoftplusGetWorkspaceSize(
            inputTensors[0], beta_scalar.ptr, threshold_scalar.ptr,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnSoftplus, true);
        return;
    }

    RsqrtOpRunner::RsqrtOpRunner() : BaseOpRunner("Rsqrt")
    {
    }

    void RsqrtOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnRsqrtGetWorkspaceSize(
            inputTensors[0], outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnRsqrt, true);
        return;
    }

}
