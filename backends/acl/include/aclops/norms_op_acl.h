#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class BatchNormOpRunner : public BaseOpRunner
    {

    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        BatchNormOpRunner();
    };

    class BatchNormBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
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

    class LayerNormBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        LayerNormBackwardOpRunner();
    };

    class GroupNormOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        GroupNormOpRunner();
    };

    class GroupNormBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        GroupNormBackwardOpRunner();
    };

    class RmsNormOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        RmsNormOpRunner();
    };

    class RmsNormGradOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        RmsNormGradOpRunner();
    };

}
