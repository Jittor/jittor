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
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "random_op_acl.h"

namespace jittor
{
    RandomOpRunner::RandomOpRunner() : BaseOpRunner("RandomUniform")
    {
        name = "RandomUniform";
    }

    RandomOpRunner::RandomOpRunner(const string &_name) : BaseOpRunner(_name)
    {
        name = _name;
    }

    void RandomOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<RandomAttr *>(op_attr.get());
        AclExecuteLauncher launcher;
        if (name == "RandomUniform")
        {
            ret = aclnnInplaceUniformGetWorkspaceSize(outputTensors[0], 0.0, 1.0, attr->seed, attr->offset, &workspaceSize, &executor);

            launcher = aclnnInplaceUniform;
        }
        else if (name == "RandomNormal")
        {
            ret = aclnnInplaceNormalGetWorkspaceSize(outputTensors[0], 0.0, 1.0, attr->seed, attr->offset, &workspaceSize, &executor);

            launcher = aclnnInplaceNormal;
        }
        else
        {
            LOGf << "Not supported random type : " << name;
        }
        launch(ret, launcher, true);
        return;
    }
}
