#pragma once

#include "base_op.h"
#include "utils.h"

namespace jittor
{
    class AdamWListOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        AdamWListOpRunner();
    };

    // The flat AdamW's gradient buffer: every parameter's gradient copied,
    // back to back, into one tensor. aclnnCat cannot do this (its inputs must
    // agree in every dimension but the concat axis) and reshaping each
    // gradient first costs one op construction per parameter -- 201 of them,
    // ~10 ms of host on a BERT-base step. One op with N inputs instead: the
    // runner issues one D2D copy per input, which the caller's flat update
    // reads on the same stream.
    class FlatConcatOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FlatConcatOpRunner();
    };
}
