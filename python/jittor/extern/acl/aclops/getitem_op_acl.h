#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class MaskedSelectOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        MaskedSelectOpRunner();
    };

    class IndexOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        IndexOpRunner();
    };

    class SliceV2OpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        SliceV2OpRunner();
    };

    class IndexPutImplAccumulateOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        IndexPutImplAccumulateOpRunner();
    };

    class StridedSliceAssignV2OpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        StridedSliceAssignV2OpRunner();
    };

}