#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>

namespace jittor
{
    int CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                        aclDataType dataType, aclTensor **tensor, bool use_nchw);

    int CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                 aclDataType dataType, aclTensor **tensor, bool use_nchw);

    struct TernaryOpRunner
    {
        string name;
        string jt_name;
        vector<Var *> in_;
        vector<Var *> out_;
        std::unique_ptr<AclOpAttr> op_attr;

        TernaryOpRunner()
        {
            jt_name = "ternary";
        }

        ~TernaryOpRunner()
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

        virtual void run()
        {
            // auto it = aclOpFuncMap.find(name);
            // if (it == aclOpFuncMap.end())
            // {
            //     LOGir << "aclOpFuncMap Not supported op: " << name;
            //     throw std::runtime_error("Unsupported operation type.");
            // }

            // 0. 算子的输入、输出、需要的attr定义
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::vector<int64_t>> outputShapes;

            auto input_num = in_.size();
            auto output_num = out_.size();
            bool use_nchw = false;

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
            int ret = -1;

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }

            for (int idx = 0; idx < output_num; idx++)
            {
                outputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }

            uint64_t workspaceSize = 0;
            aclOpExecutor *executor;
            ret = aclnnSWhereGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
            // ret = it->second.getWorkspaceSizeFuncSelect(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
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
            ret = aclnnSWhere(workspaceAddr, workspaceSize, executor, aclstream);
            // ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. （固定写法）同步等待任务执行结束
            // ret = aclrtSynchronizeStream(aclstream);
            // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);

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
            return;
        }
    };
}
