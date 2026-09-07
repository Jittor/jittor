#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class InplaceMaskedScatterOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        InplaceMaskedScatterOpRunner();
    };

    class IndexPutImplOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        IndexPutImplOpRunner();
    };
}