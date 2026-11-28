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
#include "binary_op_acl.h"

namespace jittor
{
    // ACL reads the alpha scalar as raw bytes in the tensor's dtype, so the
    // float16 unit scalar needs a two-byte host value. `__fp16` is an ARM-only
    // compiler extension and `_Float16` does not exist on x86-64 before
    // GCC 12, so an x86-64 host build cannot name a 16-bit float type. Carry
    // the IEEE binary16 encoding instead; 0x3c00 is 1.0.
    struct Float16Unit
    {
        uint16_t bits;

        explicit Float16Unit(int value) : bits(0x3c00)
        {
            ASSERT(value == 1) << "Float16Unit only encodes the unit scalar";
        }
    };

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
        // the execute launcher returns. Cache one immutable scalar per dtype so its
        // lifetime covers every queued Add/Sub invocation.
        static UnitScalarHolder<T, DType> holder;
        return holder.scalar;
    }

    BinaryOpRunner::BinaryOpRunner() : BaseOpRunner("binary")
    {
        use_nchw = false;
        is_group_op = true;
    }

    void BinaryOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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
                alpha = getUnitScalar<Float16Unit, ACL_FLOAT16>();
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

            if (!alpha) LOGf << name << ": failed to create binary scalar alpha";
        }
        AclWorkspaceArguments args;
        args.x = inputTensors[0];
        args.y = inputTensors[1];
        args.output = outputTensors[0];
        args.alpha = alpha;
        ret = it->second.workspace(args, &workspaceSize, &executor);
        launch(ret, it->second.launcher(), true);

        return;
    }

    BinaryScalarOpRunner::BinaryScalarOpRunner() : BaseOpRunner("Add")
    {
        use_nchw = false;
        is_group_op = true;
    }

    namespace
    {
        // Small per-thread cache of host aclScalar handles keyed by (aclDataType,
        // raw 64-bit value). Constants like 1.0, 0.0, -1.0, 1e-6, 0.5 repeat
        // hundreds of times per token; caching avoids aclCreateScalar /
        // aclDestroyScalar on every scalar binary launch.
        struct HostScalarCache
        {
            static constexpr size_t kCap = 64;
            struct Entry
            {
                aclDataType dtype = ACL_DT_UNDEFINED;
                uint64_t bits = 0;
                aclScalar *scalar = nullptr;
            };
            Entry entries[kCap];
            size_t count = 0;

            ~HostScalarCache()
            {
                for (size_t i = 0; i < count; ++i)
                    if (entries[i].scalar)
                        aclDestroyScalar(entries[i].scalar);
            }

            aclScalar *get_or_create(aclDataType dt, const void *data, size_t nbytes, bool &cached)
            {
                uint64_t bits = 0;
                std::memcpy(&bits, data, nbytes);
                for (size_t i = 0; i < count; ++i)
                {
                    if (entries[i].dtype == dt && entries[i].bits == bits)
                    {
                        cached = true;
                        return entries[i].scalar;
                    }
                }
                aclScalar *s = aclCreateScalar(const_cast<void *>(data), dt);
                if (s && count < kCap)
                {
                    entries[count++] = {dt, bits, s};
                    cached = true;
                }
                else
                {
                    cached = false;
                }
                return s;
            }
        };

        thread_local HostScalarCache tl_host_scalar_cache;
    }

    void BinaryScalarOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        struct ScopedAclScalar
        {
            aclScalar *ptr = nullptr;
            bool owned = false;
            ~ScopedAclScalar()
            {
                if (owned && ptr) aclDestroyScalar(ptr);
            }
        };

        float bf16_as_fp32 = 0.0f;
        aclDataType scalar_acl_dt = ACL_DT_UNDEFINED;
        const void *val_ptr = scalar_data;
        size_t val_bytes = 0;
        if (scalar_dtype == ns_float32)
        {
            scalar_acl_dt = ACL_FLOAT;
            val_bytes = 4;
        }
        else if (scalar_dtype == ns_float16)
        {
            scalar_acl_dt = ACL_FLOAT16;
            val_bytes = 2;
        }
        else if (scalar_dtype == ns_bfloat16)
        {
            uint16_t raw_bits = 0;
            std::memcpy(&raw_bits, scalar_data, sizeof(raw_bits));
            uint32_t f32_bits = static_cast<uint32_t>(raw_bits) << 16;
            std::memcpy(&bf16_as_fp32, &f32_bits, sizeof(bf16_as_fp32));
            scalar_acl_dt = ACL_FLOAT;
            val_ptr = &bf16_as_fp32;
            val_bytes = 4;
        }
        else if (scalar_dtype == ns_int32)
        {
            scalar_acl_dt = ACL_INT32;
            val_bytes = 4;
        }
        else if (scalar_dtype == ns_int64)
        {
            scalar_acl_dt = ACL_INT64;
            val_bytes = 8;
        }
        else
        {
            LOGf << name << ": unsupported scalar dtype " << scalar_dtype;
        }
        bool is_cached = false;
        aclScalar *raw_scalar = tl_host_scalar_cache.get_or_create(
            scalar_acl_dt, val_ptr, val_bytes, is_cached);
        ScopedAclScalar other{raw_scalar, !is_cached};
        if (!other.ptr) LOGf << name << ": failed to create host aclScalar";

        if (name == "Add" || name == "Sub")
        {
            aclScalar *alpha = nullptr;
            const auto dt = get_dtype(in_[0]->dtype());
            if (dt == ACL_FLOAT || dt == ACL_BF16)
                alpha = getUnitScalar<float, ACL_FLOAT>();
            else if (dt == ACL_FLOAT16)
                alpha = getUnitScalar<Float16Unit, ACL_FLOAT16>();
            else if (dt == ACL_INT32)
                alpha = getUnitScalar<int, ACL_INT32>();
            else if (dt == ACL_INT64)
                alpha = getUnitScalar<int64_t, ACL_INT64>();
            else
                LOGf << "Not supported dtype: " << in_[0]->dtype();
            if (name == "Add")
            {
                ret = aclnnAddsGetWorkspaceSize(
                    inputTensors[0], other.ptr, alpha, outputTensors[0],
                    &workspaceSize, &executor);
                launch(ret, aclnnAdds, true);
            }
            else
            {
                ret = aclnnSubsGetWorkspaceSize(
                    inputTensors[0], other.ptr, alpha, outputTensors[0],
                    &workspaceSize, &executor);
                launch(ret, aclnnSubs, true);
            }
            return;
        }
        if (name == "Mul")
        {
            ret = aclnnMulsGetWorkspaceSize(
                inputTensors[0], other.ptr, outputTensors[0],
                &workspaceSize, &executor);
            launch(ret, aclnnMuls, true);
            return;
        }
        if (name == "RealDiv")
        {
            ret = aclnnDivsGetWorkspaceSize(
                inputTensors[0], other.ptr, outputTensors[0],
                &workspaceSize, &executor);
            launch(ret, aclnnDivs, true);
            return;
        }
        LOGf << "BinaryScalarOpRunner: unsupported op " << name;
    }
}
