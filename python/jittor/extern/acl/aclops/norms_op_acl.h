#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class BatchNormOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        BatchNormOpRunner();
    };

    class BatchNormBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        BatchNormBackwardOpRunner();
    };

    class LayerNormOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        LayerNormOpRunner();
    };

}