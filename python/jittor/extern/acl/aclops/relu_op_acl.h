#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class LeakyReLUOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        LeakyReLUOpRunner();
    };

    class LeakyReLUBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        LeakyReLUBackwardOpRunner();
    };

}