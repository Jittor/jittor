#include <unordered_map>
#include <string>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include "utils.h"
#include "aclnn/aclnn.h"

namespace jittor
{
    aclDataType get_dtype(NanoString s)
    {
        if (s == ns_bfloat16)
            return ACL_BF16;
        if (s == ns_float32)
            return ACL_FLOAT;
        if (s == ns_float16)
            return ACL_FLOAT16;
        if (s == ns_int64)
            return ACL_INT64;
        if (s == ns_int32)
            return ACL_INT32;
        if (s == ns_int8)
            return ACL_INT8;
        if (s == ns_int16)
            return ACL_INT16;
        if (s == ns_uint8)
            return ACL_UINT8;
        if (s == ns_uint16)
            return ACL_UINT16;
        if (s == ns_uint32)
            return ACL_UINT32;
        if (s == ns_bool)
            return ACL_BOOL;
        if (s == ns_complex64)
            return ACL_COMPLEX64;
        LOGf << "Not supported dtype: " << s;
        return ACL_FLOAT;
    }

    aclError CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                             aclDataType dataType, aclTensor **tensor, bool use_nchw)
    {
        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        if (shape.size() == 0)
            strides = {};
        // 调用aclCreateTensor接口创建aclTensor
        *tensor = nullptr;
        if (use_nchw)
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                      shape.data(), shape.size(), deviceAddr);
        else
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), deviceAddr);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }

    aclError CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                      aclDataType dataType, aclTensor **tensor, bool use_nchw)
    {
        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        if (shape.size() == 0)
            strides = {};
        int n = shape.size();
        if (n > 1)
        {
            std::swap(shape[n - 1], shape[n - 2]);
            std::swap(strides[n - 1], strides[n - 2]);
        }
        // 调用aclCreateTensor接口创建aclTensor
        *tensor = nullptr;
        if (use_nchw)
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                      shape.data(), shape.size(), deviceAddr);
        else
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), deviceAddr);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }
}
