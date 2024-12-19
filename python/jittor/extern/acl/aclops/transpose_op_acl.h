#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class TransposeOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        TransposeOpRunner();
    };

}