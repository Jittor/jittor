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
    template <typename T, aclDataType DType>
    struct UnitScalarHolder
    {
        T value = static_cast<T>(1);
        aclScalar *scalar = aclCreateScalar(&value, DType);

        ~UnitScalarHolder()
        {
            aclDestroyScalar(scalar);
        }
    };

    template <typename T, aclDataType DType>
    static aclScalar *getUnitScalar()
    {
        // aclnn launches asynchronously and may still read the scalar after
        // executeFunc returns. Cache one immutable scalar per dtype so its
        // lifetime covers every queued Add/Sub invocation.
        static UnitScalarHolder<T, DType> holder;
        return holder.scalar;
    }

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
                alpha = getUnitScalar<float, ACL_FLOAT>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_FLOAT16)
            {
                alpha = getUnitScalar<__fp16, ACL_FLOAT16>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_BF16)
            {
                alpha = getUnitScalar<float, ACL_FLOAT>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT64)
            {
                alpha = getUnitScalar<int64_t, ACL_INT64>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT32)
            {
                alpha = getUnitScalar<int, ACL_INT32>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT8)
            {
                alpha = getUnitScalar<int8_t, ACL_INT8>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_INT16)
            {
                alpha = getUnitScalar<int16_t, ACL_INT16>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT8)
            {
                alpha = getUnitScalar<uint8_t, ACL_UINT8>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT16)
            {
                alpha = getUnitScalar<uint16_t, ACL_UINT16>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_UINT32)
            {
                alpha = getUnitScalar<uint32_t, ACL_UINT32>();
            }
            else if (get_dtype(in_[0]->dtype()) == ACL_BOOL)
            {
                alpha = getUnitScalar<bool, ACL_BOOL>();
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

        launch(ret, it->second.executeFunc, true);

        return;
    }
}
