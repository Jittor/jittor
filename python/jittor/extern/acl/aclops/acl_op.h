#pragma once
#include "utils.h"

namespace jittor
{
    extern int sync_run;
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
                LOGir << "aclOpFuncMap Not supported op: " << name;
                throw std::runtime_error("Unsupported operation type.");
            }

            // 0. 算子的输入、输出、需要的attr定义
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::vector<int64_t>> outputShapes;

            // for reduce
            // std::vector<int64_t> axes;
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

            // for layernorm
            aclIntArray *normalizedShape = nullptr;

            // for range
            aclScalar *start = nullptr;
            aclScalar *end = nullptr;
            aclScalar *step = nullptr;

            // for leaky_relu
            aclScalar *negativeSlope = nullptr;

            if (jt_name == "conv" || jt_name == "conv2d" || jt_name == "conv2dbackward" || jt_name == "maxpool" || jt_name == "maxpoolbackward" || jt_name == "avgpool" || jt_name == "avgpoolbackward")
                use_nchw = true;

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "bmm_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0) || (jt_name == "bmm_trans_0" && idx == 0))
                {
                    auto ret = CreateFakeTransAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
                else
                {
                    auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }

            // if (jt_name == "reduce" || jt_name == "transpose")
            if (jt_name == "transpose")
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

            // if (jt_name == "range")
            // {
            //     auto attr = dynamic_cast<RangeAttr *>(op_attr.get());
            //     int64_t startValue = attr->start;
            //     int64_t endValue = attr->end;
            //     int64_t stepValue = attr->step;
            //     start = aclCreateScalar(&startValue, aclDataType::ACL_INT64);
            //     end = aclCreateScalar(&endValue, aclDataType::ACL_INT64);
            //     step = aclCreateScalar(&stepValue, aclDataType::ACL_INT64);
            // }

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
            int op_idx;
            if (jt_name == "binary" && name != "Add" && name != "Sub")
                op_idx = 6;
            else if (jt_name == "unary" && name != "Cast")
                op_idx = 5;
            else
                op_idx = op_idx_map.find(name)->second;

            // LOGir << name << " " << jt_name;
            // LOGir<<op_idx;

            // 4. 根据第一段接口计算出的workspaceSize申请device内存
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }

            // 5. 调用aclnnxx第二段接口
            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. （固定写法）同步等待任务执行结束
            // if(sync_run) {
            //     ret = aclrtSynchronizeStream(aclstream);
            //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);
            // }

            // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
            // destroy tensor
            // for (int idx = 0; idx < input_num; idx++)
            // {
            //     aclDestroyTensor(inputTensors[idx]);
            // }
            // for (int idx = 0; idx < output_num; idx++)
            // {
            //     aclDestroyTensor(outputTensors[idx]);
            // }
            // destroy scalar
            // aclDestroyScalar(start);
            // aclDestroyScalar(end);
            // aclDestroyScalar(step);
            // aclDestroyScalar(negativeSlope);

            // // destroy IntArray
            // aclDestroyIntArray(size);
            // aclDestroyIntArray(dim);
            // aclDestroyIntArray(strides);
            // aclDestroyIntArray(pads);
            // aclDestroyIntArray(outPads);
            // aclDestroyIntArray(dilations);
            // aclDestroyIntArray(kernel_size);
            // aclDestroyIntArray(normalizedShape);

            return;
        }
    };
}