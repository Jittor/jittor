#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ReduceOpRunner : public BaseOpRunner
    {
        int op_idx; // Specific to reduce operations

        ReduceOpRunner();

    protected:
        ReduceAttr *attr;
        aclIntArray *dim;
        bool keepdims;

        void setupOutputDesc() override;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}