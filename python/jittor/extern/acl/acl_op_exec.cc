// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
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

namespace jittor
{
    int CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                        aclDataType dataType, aclTensor **tensor, bool use_nchw = false)
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
        if (use_nchw)
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                      shape.data(), shape.size(), deviceAddr);
        else
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), deviceAddr);
        return 0;
    }

    struct AclOpRunner
    {
        string name;
        string jt_name;
        vector<Var *> in_;
        vector<Var *> out_;
        std::unique_ptr<AclOpAttr> op_attr;

        AclOpRunner(const string &name) : name(name)
        {
        }

        ~AclOpRunner()
        {
        }

        aclDataType get_dtype(NanoString s)
        {
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
            LOGf << "Not supported dtype: " << s;
            return ACL_FLOAT;
        }

        void add(Var *v, bool is_input)
        {

            if (is_input)
            {
                in_.push_back(v);
            }
            else
            {
                out_.push_back(v);
            }
            return;
        }

        template <typename T>
        std::vector<T> createVector(int64_t size)
        {
            return std::vector<T>(size, 0);
        }

        void run()
        {
            // LOGir << name << " " << jt_name;
            auto it = aclOpFuncMap.find(name);
            if (it == aclOpFuncMap.end())
            {
                LOGir << "Not supported op: " << name;
                throw std::runtime_error("Unsupported operation type.");
            }

            // 0. 算子的输入、输出、需要的attr定义
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::vector<int64_t>> outputShapes;

            // for reduce
            std::vector<int64_t> axes;
            aclIntArray *dim = nullptr;
            bool keepdims;

            bool use_nchw = false;

            auto input_num = in_.size();
            auto output_num = out_.size();

            for (int input_idx = 0; input_idx < input_num; input_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < in_[input_idx]->shape.size(); j++)
                {
                    shape.push_back(in_[input_idx]->shape[j]);
                }
                inputShapes.push_back(shape);
            }
            for (int output_idx = 0; output_idx < output_num; output_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < out_[output_idx]->shape.size(); j++)
                {
                    shape.push_back(out_[output_idx]->shape[j]);
                }
                outputShapes.push_back(shape);
            }

            // 1. 创建aclTensor和aclScalar，不同算子可能不一样，需要根据具体API的接口定义修改
            std::vector<aclTensor *> inputTensors;
            std::vector<aclTensor *> outputTensors;

            // for add and sub
            aclScalar *alpha = nullptr;

            // for expand
            aclIntArray *size = nullptr;

            // for conv
            aclIntArray *strides = nullptr;
            aclIntArray *pads = nullptr;
            aclIntArray *outPads = nullptr;
            aclIntArray *dilations = nullptr;
            int ret = -1;

            // for maxpool
            aclIntArray *kernel_size = nullptr;

            // for range
            aclScalar *start = nullptr;
            aclScalar *end = nullptr;
            aclScalar *step = nullptr;

            // for leaky_relu
            aclScalar *negativeSlope = nullptr;

            if (name == string("Add") || name == string("Sub"))
            {

                if (get_dtype(in_[0]->dtype()) == ACL_FLOAT)
                {
                    float alphaValue = 1.0;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_FLOAT16)
                {
                    float alphaValue = 1.0;
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
            }

            if (jt_name == "conv" || jt_name == "conv2d" || jt_name == "conv2dbackward" || jt_name == "maxpool" || jt_name == "maxpoolbackward" || jt_name == "avgpool" || jt_name == "avgpoolbackward")
                use_nchw = true;

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }

            if (jt_name == "reduce" || jt_name == "transpose")
            {
                auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
                dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                keepdims = attr->keepdims;
                if (name == string("ReduceMax") || name == string("ReduceMin") || name == string("ReduceMean") || name == string("ReduceProd"))
                {
                    if (attr->axes.size() == in_[0]->shape.size())
                        outputShapes[0] = {};
                }
            }

            if (jt_name == "range")
            {
                auto attr = dynamic_cast<RangeAttr *>(op_attr.get());
                int64_t startValue = attr->start;
                int64_t endValue = attr->end;
                int64_t stepValue = attr->step;
                start = aclCreateScalar(&startValue, aclDataType::ACL_INT64);
                end = aclCreateScalar(&endValue, aclDataType::ACL_INT64);
                step = aclCreateScalar(&stepValue, aclDataType::ACL_INT64);
            }

            if (jt_name == "conv2dbackward")
            {
                for (int idx = 0; idx < 2; idx++)
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
                // biasgrad nd format
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[2], out_[2]->mem_ptr, out_[2]->size, get_dtype(out_[2]->dtype()), &outputTensors[2], false);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }
            else
            {
                for (int idx = 0; idx < output_num; idx++)
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }

            // 2. 调用CANN算子库aclnnxxxGetWorkspaceSize的接口，两段式接口的第一个
            uint64_t workspaceSize = 0;
            aclOpExecutor *executor;

            if (name == string("Add") || name == string("Sub"))
                ret = it->second.getWorkspaceSizeFuncAdd(inputTensors[0], inputTensors[1], alpha, outputTensors[0], &workspaceSize, &executor);
            else if (name == string("Expand"))
            {
                size = aclCreateIntArray(&outputShapes[0][0], outputShapes[0].size());
                ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], size, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Cast"))
                ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "unary")
                ret = it->second.getWorkspaceSizeFuncUnary(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "binary")
                ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "bmm" || jt_name == "matmul")
                ret = it->second.getWorkspaceSizeFuncMatmul(inputTensors[0], inputTensors[1], outputTensors[0], 1, &workspaceSize, &executor);
            else if (name == string("ReduceSum") || name == string("ReduceMean"))
            {
                ret = it->second.getWorkspaceSizeFuncReduceSum(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("ReduceMax") || name == string("ReduceMin"))
            {
                ret = it->second.getWorkspaceSizeFuncAmax(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
            }
            // else if (name == string("ReduceProd"))
            // {
            //     ret = it->second.getWorkspaceSizeFuncReduceProd(inputTensors[0], dim, false, outputTensors[0], &workspaceSize, &executor);
            // }
            else if (name == string("RandomUniform") || name == string("RandomNormal"))
            {
                auto attr = dynamic_cast<RandomAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncRandom(outputTensors[0], 0.0, 1.0, attr->seed, attr->offset, &workspaceSize, &executor);
            }
            else if (name == string("Select") || name == string("Where"))
            {
                ret = it->second.getWorkspaceSizeFuncSelect(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Triu"))
            {
                auto attr = dynamic_cast<TriuAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], aclDataType(attr->diagonal), outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Transpose"))
            {
                ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Conv2d"))
            {
                auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                strides = aclCreateIntArray(attr->convStrides.data(), 2);
                pads = aclCreateIntArray(attr->convPads.data(), 2);
                outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                dilations = aclCreateIntArray(attr->convDilations.data(), 2);
                aclTensor *bias = nullptr;
                if (input_num == 3)
                    bias = inputTensors[2];

                ret = it->second.getWorkspaceSizeFuncConv(inputTensors[0], inputTensors[1], bias, strides, pads, dilations, false, outPads, attr->group, outputTensors[0], 0, &workspaceSize, &executor);
            }
            else if (name == string("Conv2dBackward"))
            {
                auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                strides = aclCreateIntArray(attr->convStrides.data(), 2);
                pads = aclCreateIntArray(attr->convPads.data(), 2);
                outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                dilations = aclCreateIntArray(attr->convDilations.data(), 2);
                bool outputMask[3] = {true, true, true};
                if (input_num == 3)
                {
                    outputMask[2] = false;
                }
                aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
                auto biasSizes = aclCreateIntArray(&outputShapes[2][0], outputShapes[2].size());
                ret = it->second.getWorkspaceSizeFuncConvBackward(inputTensors[0], inputTensors[1], inputTensors[2], biasSizes, strides, pads, dilations, false, outPads, attr->group, outMask, 0, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);
            }
            else if (name == string("Maxpool"))
            {
                auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                pads = aclCreateIntArray(attr->poolPads.data(), 2);
                dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
                ret = it->second.getWorkspaceSizeFuncMaxPool(inputTensors[0], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], outputTensors[1], &workspaceSize, &executor);
            }
            else if (name == string("MaxpoolBackward"))
            {
                auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                pads = aclCreateIntArray(attr->poolPads.data(), 2);
                dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
                ret = it->second.getWorkspaceSizeFuncMaxPoolBackward(inputTensors[0], inputTensors[1], inputTensors[2], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Avgpool"))
            {
                auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                pads = aclCreateIntArray(attr->poolPads.data(), 2);
                ret = it->second.getWorkspaceSizeFuncAvgPool(inputTensors[0], kernel_size, strides, pads, attr->poolCeil, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("AvgpoolBackward"))
            {
                auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                pads = aclCreateIntArray(attr->poolPads.data(), 2);
                ret = it->second.getWorkspaceSizeFuncAvgPoolBackward(inputTensors[0], inputTensors[1], kernel_size, strides, pads, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Flip"))
            {
                auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
                dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Concat"))
            {
                std::vector<aclTensor *> concatTensorList = {};
                for (int i = 0; i < input_num; i++)
                {
                    concatTensorList.push_back(inputTensors[i]);
                }
                auto concatTensorListInput = aclCreateTensorList(&concatTensorList[0], input_num);
                auto attr = dynamic_cast<ConcatAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncConcat(concatTensorListInput, attr->dim, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Gather"))
            {
                auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncGather(inputTensors[0], attr->dim, inputTensors[1], outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Cumsum"))
            {
                auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncCumsum(inputTensors[0], attr->dim, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Scatter"))
            {
                auto attr = dynamic_cast<ScatterAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncScatter(inputTensors[0], attr->axis, inputTensors[1], inputTensors[2], attr->reduction, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Floor"))
            {
                ret = it->second.getWorkspaceSizeFuncUnary(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Index"))
            {
                auto indexTensorList = aclCreateTensorList(&inputTensors[1], input_num - 1);
                ret = it->second.getWorkspaceSizeFuncIndex(inputTensors[0], indexTensorList, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("SliceV2"))
            {
                auto attr = dynamic_cast<StrideAttr *>(op_attr.get());
                auto begins = aclCreateIntArray(attr->begins.data(), attr->begins.size());
                auto ends = aclCreateIntArray(attr->ends.data(), attr->ends.size());
                auto steps = aclCreateIntArray(attr->steps.data(), attr->steps.size());
                auto axes = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                ret = it->second.getWorkspaceSizeFuncSliceV2(inputTensors[0], begins, ends, axes, steps, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("IndexPutImpl"))
            {
                std::vector<aclTensor *> indexTensorList = {};
                for (int i = 1; i < input_num; i++)
                {
                    indexTensorList.push_back(inputTensors[i]);
                }
                auto indexTensorListInput = aclCreateTensorList(&indexTensorList[0], input_num - 1);
                ret = it->second.getWorkspaceSizeFuncIndexPutImpl(outputTensors[0], indexTensorListInput, inputTensors[0], false, true, &workspaceSize, &executor);
            }
            else if (name == string("StridedSliceAssignV2"))
            {
                ret = it->second.getWorkspaceSizeFuncStridedSliceAssignV2(outputTensors[0], inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], inputTensors[4], &workspaceSize, &executor);
            }
            else if (jt_name == "range")
            {
                ret = it->second.getWorkspaceSizeFuncRange(start, end, step, outputTensors[0], &workspaceSize, &executor);
            }
            else if (jt_name == "leakyrelu")
            {
                auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
                negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
                ret = it->second.getWorkspaceSizeFuncLeakyRelu(inputTensors[0], negativeSlope, outputTensors[0], &workspaceSize, &executor);
            }
            else if (jt_name == "leakyrelubackward")
            {
                auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
                negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
                ret = it->second.getWorkspaceSizeFuncLeakyReluBackward(inputTensors[0], inputTensors[1], negativeSlope, attr->selfIsResult, outputTensors[0], &workspaceSize, &executor);
            }
            else if (jt_name == "dropout")
            {
                auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncDropout(inputTensors[0], attr->p, attr->train, attr->seed, attr->offset, outputTensors[0], outputTensors[1], &workspaceSize, &executor);
            }
            else if (jt_name == "dropoutbackward")
            {
                auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncDropoutBackward(inputTensors[0], inputTensors[1], attr->scale, outputTensors[0], &workspaceSize, &executor);
            }
            else
                LOGir << "not supported op " << jt_name;

            // for debug
            if (ret != ACL_SUCCESS)
            {
                auto tmp_err_msg = aclGetRecentErrMsg();
                LOGir << name << ", " << tmp_err_msg;
            }

            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);

            // 4. 根据第一段接口计算出的workspaceSize申请device内存
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }

            // 5. 调用aclnnxx第二段接口
            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
            // destroy tensor
            for (int idx = 0; idx < input_num; idx++)
            {
                aclDestroyTensor(inputTensors[idx]);
            }
            for (int idx = 0; idx < output_num; idx++)
            {
                aclDestroyTensor(outputTensors[idx]);
            }
            // destroy scalar
            aclDestroyScalar(alpha);
            aclDestroyScalar(start);
            aclDestroyScalar(end);
            aclDestroyScalar(step);
            aclDestroyScalar(negativeSlope);

            // destroy IntArray
            aclDestroyIntArray(size);
            aclDestroyIntArray(dim);
            aclDestroyIntArray(strides);
            aclDestroyIntArray(pads);
            aclDestroyIntArray(outPads);
            aclDestroyIntArray(dilations);
            aclDestroyIntArray(kernel_size);

            return;
        }
    };

    void free_var_mem(Var *v);

    unordered_map<uint32, string> opname_map = {
        // unary op
        {ns_cast, "Cast"},
        {ns_negative, "Neg"},
        {ns_abs, "Abs"},
        {ns_exp, "Exp"},
        {ns_log, "Log"},
        {ns_sqrt, "Sqrt"},
        {ns_ceil, "Ceil"},
        {ns_floor, "Floor"},
        {ns_round, "Round"},
        // m(round_int)
        // m(floor_int)
        // m(ceil_int)
        {ns_sin, "Sin"},
        {ns_cos, "Cos"},
        {ns_tan, "Tan"},
        {ns_asin, "Asin"},
        {ns_acos, "Acos"},
        {ns_atan, "Atan"},
        {ns_sinh, "Sinh"},
        {ns_cosh, "Cosh"},
        {ns_tanh, "Tanh"},
        {ns_asinh, "Asinh"},
        {ns_acosh, "Acosh"},
        {ns_atanh, "Atanh"},
        {ns_sigmoid, "Sigmoid"},
        {ns_erf, "Erf"},
        {ns_erfinv, "Erfinv"},
        {ns_logical_not, "LogicalNot"},
        {ns_bitwise_not, "BitwiseNot"},
        // binary op
        {ns_pow, "Pow"},
        {ns_maximum, "Maximum"},
        {ns_minimum, "Minimum"},
        {ns_add, "Add"},
        {ns_subtract, "Sub"},
        {ns_multiply, "Mul"},
        {ns_divide, "RealDiv"},
        {ns_floor_divide, "FloorDiv"},
        {ns_mod, "Mod"},
        {ns_less, "Less"},
        {ns_less_equal, "LessEqual"},
        {ns_greater, "Greater"},
        {ns_greater_equal, "GreaterEqual"},
        {ns_equal, "Equal"},
        {ns_not_equal, "NotEqual"},
        {ns_left_shift, "LeftShift"},
        {ns_right_shift, "RightShift"},
        {ns_logical_and, "LogicalAnd"},
        {ns_logical_or, "LogicalOr"},
        {ns_logical_xor, "LogicalXor"},
        {ns_bitwise_and, "BitwiseAnd"},
        {ns_bitwise_or, "BitwiseOr"},
        {ns_bitwise_xor, "BitwiseXor"},

    };

    void fallback_cpu(Op *op)
    {
        LOGy << "!!! fallback_cpu " << op;
        use_cuda = 0;
        for (auto v : op->inputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, exe.allocator);
            }
        }
        for (auto v : op->outputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, exe.allocator);
            }
        }
        op->flags.set(NodeFlags::_cpu);
        op->flags.set(NodeFlags::_cuda, 0);
        if (op->name() == string("fused"))
        {
            auto fop = (FusedOp *)op;
            for (auto op : fop->ops)
            {
                op->flags.set(NodeFlags::_cpu);
                op->flags.set(NodeFlags::_cuda, 0);
            }
        }
        op->do_run();
        use_cuda = 1;
    }

    /*
        check compile
        if compiled: exec
        else: compile
            check is fused
                check is relay
                else
                    compile func = try exec
                        if failed: fallback_cpu
            else
                try compile
                if failed: fallback_cpu
    */

    extern jit_op_entry_t (*do_compile_hook)(Op *);
    jit_op_entry_t do_compile_inner(Op *op);

    void try_exec_and_fallback_cpu(Op *op)
    {
        LOGv << "try_exec_and_fallback_cpu " << op;
        auto fop = (FusedOp *)op;

        vector<Var *> new_alloced;
        int fallback = 0;
        try
        {
            for (Op *op : fop->ops)
            {
                for (auto out : op->outputs())
                {
                    if (out->mem_ptr)
                        continue;
                    out->alloc(exe.allocator);
                    new_alloced.push_back(out);
                }
                if (op->name() == string("unary"))
                {
                    auto uop = (UnaryOp *)op;
                    AclOpRunner op("...");
                    op.add(uop->x, true);
                    op.add(uop->y, false);
                    auto iter = opname_map.find(uop->ns);
                    ASSERT(iter != opname_map.end()) << "op " << uop->ns << " not found";
                    op.name = iter->second;
                    op.jt_name = uop->name();
                    op.run();
                }
                else if (op->name() == string("binary"))
                {
                    auto bop = (BinaryOp *)op;
                    AclOpRunner op("...");
                    op.add(bop->x, true);
                    op.add(bop->y, true);
                    op.add(bop->z, false);
                    auto iter = opname_map.find(bop->ns);
                    ASSERT(iter != opname_map.end()) << "op " << bop->ns << " not found";
                    op.name = iter->second;
                    op.jt_name = bop->name();

                    if (bop->x->dtype() == ns_bool and bop->y->dtype() == ns_bool)
                    {
                        // BitwiseOr, BitwiseAnd, BitwiseXor -> LogicalOr, LogicalAnd, LogicalXor
                        if (bop->ns == ns_bitwise_or)
                        {
                            op.name = "LogicalOr";
                        }
                        else if (bop->ns == ns_bitwise_and)
                        {
                            op.name = "LogicalAnd";
                        }
                        else if (bop->ns == ns_bitwise_xor)
                        {
                            op.name = "LogicalXor";
                        }
                    }
                    op.run();
                }
                else if (op->name() == string("ternary"))
                {
                    auto top = (TernaryOp *)op;
                    AclOpRunner op("Select");
                    op.add(top->cond, true);
                    op.add(top->x, true);
                    op.add(top->y, true);
                    op.add(top->z, false);
                    op.run();
                }
                else if (op->name() == string("array"))
                {
                    auto aop = (ArrayOp *)op;
                    aclrtMemcpy(aop->output->mem_ptr, aop->output->size, aop->ptr<void>(), aop->output->size, ACL_MEMCPY_HOST_TO_DEVICE);
                }
                else if (op->name() == string("reduce"))
                {
                    auto rop = (ReduceOp *)op;
                    AclOpRunner op("");
                    if (rop->ns == ns_add)
                        op.name = "ReduceSum";
                    else if (rop->ns == ns_multiply)
                        // TODO unsupported the multi dim
                        op.name = "ReduceProd";
                    else if (rop->ns == ns_maximum)
                        op.name = "ReduceMax";
                    else if (rop->ns == ns_minimum)
                        op.name = "ReduceMin";
                    else if (rop->ns == ns_mean)
                        op.name = "ReduceMean";
                    else
                        LOGf << "op " << rop->ns << " not supported";
                    op.jt_name = "reduce";
                    op.add(rop->x, true);

                    ReduceAttr *attr = new ReduceAttr();
                    for (int i = 0; i < rop->x->shape.size(); i++)
                        if (rop->reduce_mask & (1 << i))
                            attr->axes.push_back(i);
                    if (rop->x->shape.size() == rop->y->shape.size())
                        attr->keepdims = true;
                    else
                        attr->keepdims = false;

                    op.op_attr.reset(attr);
                    op.add(rop->y, false);
                    op.run();
                }
                else if (op->name() == string("broadcast_to"))
                {
                    auto bop = (BroadcastToOp *)op;
                    AclOpRunner op("Expand");
                    op.jt_name = "expand";

                    NanoVector xshape, xshape_bk = bop->x->shape;
                    NanoVector zshape = bop->z->shape;
                    for (int i = 0; i < zshape.size(); i++)
                    {
                        if (bop->bcast_mask & (1 << i))
                        {
                            xshape.push_back(1);
                        }
                        else
                        {
                            xshape.push_back(zshape[i]);
                        }
                    }
                    bop->x->shape = xshape;
                    op.add(bop->x, true);
                    // bop->x->shape = xshape_bk;
                    op.add(bop->z, false);
                    op.run();
                    bop->x->shape = xshape_bk;
                }
                else if (op->name() == string("fuse_transpose"))
                {
                    // replace fuse_transpose with transpose
                    auto top = (TransposeOp *)op;
                    AclOpRunner op("Transpose");
                    op.add(top->x, true);
                    op.add(top->y, false);
                    op.jt_name = "transpose";

                    ReduceAttr *attr = new ReduceAttr();
                    for (int i = 0; i < top->axes.size(); i++)
                        attr->axes.push_back(top->axes[i]);
                    op.op_attr.reset(attr);

                    op.run();
                }
                else
                {
                    LOGf << "op " << op->name() << " not supported";
                }
            }
        }
        catch (std::exception &e)
        {
            fallback = 1;
            LOGir << "fallback cpu" << e.what();
        }
        for (auto v : new_alloced)
        {
            free_var_mem(v);
        }
        if (fallback)
        {
            fallback_cpu(op);
        }
    }

    extern int current_seed;
    extern int64 current_offset;

    static unordered_map<string, std::function<void(Op *)>> acl_ops = {
        {"curand_random", [&current_seed, &current_offset](Op *op)
         {
             auto _op = (RandomOp *)op;
             AclOpRunner runner(_op->type == ns_uniform ? "RandomUniform" : "RandomNormal");
             auto out = op->output(0);
             RandomAttr *attr = new RandomAttr();
             attr->seed = current_seed;
             attr->offset = current_offset;
             runner.jt_name = "random";
             runner.op_attr.reset(attr);

             runner.add(out, false);
             runner.run();
             current_offset += out->numel();
         }},
    };

    static void exec_mapped_acl_ops(Op *op)
    {
        auto iter = acl_ops.find(op->name());
        if (iter != acl_ops.end())
        {
            LOGv << "exec acl op " << op->name() << op;
            iter->second(op);
        }
        else
        {
            LOGf << "op " << op->name() << " not supported";
        }
    }

    static jit_op_entry_t acl_do_compile(Op *op)
    {
        LOGv << "compile" << op;
        OpCompiler oc(op);
        string *src = &oc.src;
        for (auto op_type : op_types)
            op_type->post_pass(&oc);
        string src_after_passes;
        // if is fused op
        if (oc.op)
        {
            TunerManager tm(&oc);
            src_after_passes = tm.tune();
            src = &src_after_passes;
        }
        op->compile_optimize(*src);
        if (!op->flags.get(NodeFlags::_cuda))
        {
            LOGv << "compile cpu";
            return oc.compile(op->get_jit_key(get_jk()), *src);
        }
        if (op->name() == string("fused"))
        {
            FusedOp *fop = (FusedOp *)op;
            // if is a relayed op
            if (fop->context->vrm.relay_groups.size())
            {
                LOGv << "relay fused op";
                return oc.compile(op->get_jit_key(get_jk()), *src);
            }
            else
            {
                return &try_exec_and_fallback_cpu;
            }
        }
        else if (op->name() == string("code"))
        {
            CodeOp *cop = (CodeOp *)op;
            if (cop->cuda_src.find("acl") != string::npos)
            {
                LOGv << "compile acl op";
                return oc.compile(op->get_jit_key(get_jk()), *src);
            }
            else
            {
                return &exec_mapped_acl_ops;
            }
        }
        else
        {
            LOGv << "compile finish" << op;
            return &exec_mapped_acl_ops;
        }
        return do_compile_inner(op);
    }

    // from op_register.cc
    extern unordered_map<string, OpInfo> op_info_map;

    void init_acl_ops()
    {
        do_compile_hook = acl_do_compile;
        vector<string> to_erase;
        for (auto &kv : op_info_map)
        {
            if (startswith(kv.first, "cu") && acl_ops.count(kv.first) == 0)
            {
                to_erase.push_back(kv.first);
            }
        }
        for (auto &k : to_erase)
        {
            LOGv << "op not supported: " << k << ", erase it.";
            op_info_map.erase(k);
        }
    }

} // jittor