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

    std::unordered_map<string, int> op_idx_map =
        {
            {"Add", 1},
            {"Sub", 2},
            {"Expand", 3},
            {"Cast", 4},
            {"Unary", 5},
            {"Binary", 6},
            {"BatchMatMul", 7},
            {"MatMul", 8},
            {"ReduceSum", 9},
            {"ReduceMean", 10},
            {"ReduceMax", 11},
            {"ReduceMin", 12},
            {"RandomUniform", 13},
            {"RandomNormal", 14},
            {"Nonzero", 15},
            {"Select", 16},
            {"Where", 17},
            {"Triu", 18},
            {"Transpose", 19},
            {"Conv2d", 20},
            {"Conv2dBackward", 21},
            {"Maxpool", 22},
            {"MaxpoolBackward", 23},
            {"Avgpool", 24},
            {"AvgpoolBackward", 25},
            {"Flip", 26},
            {"Concat", 27},
            {"Gather", 28},
            {"Cumsum", 29},
            {"Scatter", 30},
            {"Floor", 31},
            {"Index", 32},
            {"SliceV2", 33},
            {"IndexPutImpl", 34},
            {"IndexPutImplAccumulate", 35},
            {"StridedSliceAssignV2", 36},
            {"Range", 37},
            {"LeakyReLU", 38},
            {"LeakyReLUBackward", 39},
            {"Dropout", 40},
            {"DropoutBackward", 41},
            {"SiLU", 42},
            {"SiLUBackward", 43},
            {"Sigmoid", 44},
            {"SigmoidBackward", 45},
            {"Embedding", 46},
            {"EmbeddingBackward", 47},
            {"InplaceMaskedScatter", 48},
            {"MaskedSelect", 49},
            {"SplitWithSize", 50},
            {"FlashAttention", 51},
            {"FlashAttentionBackward", 52},
            {"Softmax", 53},
            {"SoftmaxBackward", 54},
            {"BatchNorm", 55},
            {"BatchNormBackward", 56},
            {"LayerNorm", 57},
            {"RotaryPosEmb", 58},
            {"Stack", 59},
            {"NanToNum", 60},
    };

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
