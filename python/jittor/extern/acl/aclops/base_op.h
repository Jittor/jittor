#pragma once
#include "utils.h"
#include "acl_jittor.h"

namespace jittor
{

    class BaseOpRunner
    {
    protected:
        vector<Var *> in_;
        vector<Var *> out_;

        int ret = -1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        bool is_group_op = false;

        std::vector<std::vector<int64_t>> inputShapes;
        std::vector<std::vector<int64_t>> outputShapes;

        std::vector<aclTensor *> inputTensors;
        std::vector<aclTensor *> outputTensors;

    public:
        string name;
        string jt_name;
        std::unique_ptr<AclOpAttr> op_attr;
        bool use_nchw;

        BaseOpRunner(const string &name = "") : name(name) {}
        virtual ~BaseOpRunner() = default;

        // Common functionality for adding input/output variables
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

        void setupInputDesc()
        {
            auto input_num = in_.size();
            for (int input_idx = 0; input_idx < input_num; input_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < in_[input_idx]->shape.size(); j++)
                {
                    shape.push_back(in_[input_idx]->shape[j]);
                }
                inputShapes.push_back(shape);
            }

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }
        }

        void cleanupDesc()
        {
            auto input_num = in_.size();
            auto output_num = out_.size();
            for (int idx = 0; idx < input_num; idx++)
            {
                aclDestroyTensor(inputTensors[idx]);
            }
            for (int idx = 0; idx < output_num; idx++)
            {
                aclDestroyTensor(outputTensors[idx]);
            }
        }

        virtual void setupOutputDesc()
        {
            auto output_num = out_.size();

            for (int output_idx = 0; output_idx < output_num; output_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < out_[output_idx]->shape.size(); j++)
                {
                    shape.push_back(out_[output_idx]->shape[j]);
                }
                outputShapes.push_back(shape);
            }

            for (int idx = 0; idx < output_num; idx++)
            {
                outputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }
        }

        void syncRun()
        {
            ret = aclrtSynchronizeStream(aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);
        }

        void checkRet(aclnnStatus ret)
        {
            if (ret != ACL_SUCCESS)
            {
                auto tmp_err_msg = aclGetRecentErrMsg();
                LOGir << name << ", " << tmp_err_msg;
            }

            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
        }

        // Base run method with common operator lookup logic
        void run()
        {
            if (is_group_op)
            {
                auto it = aclOpFuncMap.find(name);
                if (it == aclOpFuncMap.end())
                {
                    LOGir << "aclOpFuncMap Not supported op: " << name;
                    throw std::runtime_error("Unsupported operation type.");
                }
                setupInputDesc();
                setupOutputDesc();
                executeOp(it);
                cleanupDesc();
            }
            else
            {
                auto it = aclOpFuncMap.find(name);
                setupInputDesc();
                setupOutputDesc();
                executeOp(it);
                cleanupDesc();
            }
        }

    protected:
        // Virtual method for specific operator execution
        virtual void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) = 0;
        void cleanupAttr();
    };

}
