#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner() : BaseOpRunner("binary")
        {
        }

    protected:
        bool is_group_op = true;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override
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
                ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);

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
    };
}
