#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class BatchMatMulOpRunner : public BaseOpRunner
    {

    protected:
        void setupInputDesc() override;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        BatchMatMulOpRunner();
    };
}