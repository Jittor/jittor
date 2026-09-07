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

#include "unary_op_acl.h"

namespace jittor
{
    UnaryOpRunner::UnaryOpRunner() : BaseOpRunner("unary")
    {
        use_nchw = false;
        is_group_op = true;
    }

    void UnaryOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        if (name == "Cast")
            ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
        else
            ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);

        // Preserve the historical asynchronous unary path while moving its
        // workspace/execute/error tail behind the common launcher contract.
        launch(ret, it->second.executeFunc, false);
    }
}
