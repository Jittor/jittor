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
#include "binary_op_acl.h"

namespace jittor
{
    BinaryOpRunner::BinaryOpRunner() : BaseOpRunner("binary")
    {
        use_nchw = false;
        is_group_op = true;
    }

    void BinaryOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        aclScalar *alpha = nullptr;

        if (name == string("Add") || name == string("Sub"))
        {
            if (get_dtype(in_[0]->dtype()) == ACL_FLOAT)
            {
                float alphaValue = 1.0;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_FLOAT16)
            {
                __fp16 alphaValue = 1.0;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT64)
            {
                int64_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT32)
            {
                int alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT8)
            {
                int8_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT16)
            {
                int16_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT8)
            {
                uint8_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT16)
            {
                uint16_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT32)
            {
                uint32_t alphaValue = 1;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_BOOL)
            {
                bool alphaValue = true;
                alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
            }
            else
            {
                LOGf << "Not supported dtype: " << in_[0]->dtype();
            }

            CHECK_RET(alpha != nullptr, return);
            ret = it->second.getWorkspaceSizeFuncAdd(inputTensors[0], inputTensors[1], alpha, outputTensors[0], &workspaceSize, &executor);
        }
        else

        {
            ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
        }

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyScalar(alpha);
        return;
    }
}