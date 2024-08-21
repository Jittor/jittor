// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include <acl/acl.h>

std::string acl_error_to_string(aclError error);

namespace jittor
{

    EXTERN_LIB uint64_t acl_jittor_tid;
    EXTERN_LIB aclrtStream aclstream;

    void acl_jittor_op_compiler(string &filename, string &src, bool is_acl, string &extra_flags);

    struct AclOpFunctions
    {
        // for Unary
        std::function<aclnnStatus(aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncUnary;
        // for Cast
        std::function<aclnnStatus(aclTensor *, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncCast;
        // for Bianry
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncBinary;
        // for Add and Sub
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncAdd;
        // for Expand and permute
        std::function<aclnnStatus(aclTensor *, aclIntArray *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncExpand;
        // for bmm and matmul
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, int8_t, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncMatmul;
        // for conv
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclIntArray *, int64_t, aclTensor *, int8_t, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncConv;
        // for reducesum, mean
        std::function<aclnnStatus(aclTensor *, aclIntArray *, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncReduceSum;
        // for amax and amin
        std::function<aclnnStatus(aclTensor *, aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncAmax;
        // for conv backward
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclIntArray *, int, aclBoolArray *, int8_t, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncConvBackward;
        // for proddim
        std::function<aclnnStatus(aclTensor *, int64_t, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncProdDim;
        // for select
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncSelect;
        // for random_uniform and random_normal
        std::function<aclnnStatus(aclTensor *, int64_t, int64_t, int64_t, int64_t, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncRandom;

        std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)> executeFunc;

        // 添加一个默认构造函数
        AclOpFunctions() = default;

        // for Unary
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)> execf)
            : getWorkspaceSizeFuncUnary(gwsf), executeFunc(execf) {}

        // for Cast
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)> execf)
            : getWorkspaceSizeFuncCast(gwsf), executeFunc(execf) {}

        // for Binary
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncBinary(gwsf), executeFunc(execf) {}
        // for Add and Sub
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncAdd(gwsf), executeFunc(execf) {}

        // for Expand
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncExpand(gwsf), executeFunc(execf) {}

        // for Matmul
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, int8_t, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncMatmul(gwsf), executeFunc(execf) {}

        // for conv
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclIntArray *, int64_t, aclTensor *, int8_t, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncConv(gwsf), executeFunc(execf) {}

        // for reducesum, mean
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncReduceSum(gwsf), executeFunc(execf) {}

        // for amax amin
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncAmax(gwsf), executeFunc(execf) {}

        // for conv backward
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclIntArray *, int, aclBoolArray *, int8_t, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncConvBackward(gwsf), executeFunc(execf) {}

        // for proddim
        AclOpFunctions(std::function<aclnnStatus(const aclTensor *, int64_t, bool, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncProdDim(gwsf), executeFunc(execf) {}

        // for select
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncSelect(gwsf), executeFunc(execf) {}

        // for random_normal
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, int64_t, int64_t, int64_t, int64_t, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncRandom(gwsf), executeFunc(execf) {}
    };

    static std::unordered_map<std::string, AclOpFunctions> aclOpFuncMap = {
        {"Abs", AclOpFunctions(aclnnAbsGetWorkspaceSize, aclnnAbs)},
        {"Exp", AclOpFunctions(aclnnExpGetWorkspaceSize, aclnnExp)},
        {"Log", AclOpFunctions(aclnnLogGetWorkspaceSize, aclnnLog)},
        {"Sqrt", AclOpFunctions(aclnnSqrtGetWorkspaceSize, aclnnSqrt)},
        {"Ceil", AclOpFunctions(aclnnCeilGetWorkspaceSize, aclnnCeil)},
        {"Floor", AclOpFunctions(aclnnFloorGetWorkspaceSize, aclnnFloor)},
        {"Round", AclOpFunctions(aclnnRoundGetWorkspaceSize, aclnnRound)},
        {"Sin", AclOpFunctions(aclnnSinGetWorkspaceSize, aclnnSin)},
        {"Cos", AclOpFunctions(aclnnCosGetWorkspaceSize, aclnnCos)},
        {"Tan", AclOpFunctions(aclnnTanGetWorkspaceSize, aclnnTan)},
        {"Asin", AclOpFunctions(aclnnAsinGetWorkspaceSize, aclnnAsin)},
        {"Acos", AclOpFunctions(aclnnAcosGetWorkspaceSize, aclnnAcos)},
        {"Atan", AclOpFunctions(aclnnAtanGetWorkspaceSize, aclnnAtan)},
        {"Sinh", AclOpFunctions(aclnnSinhGetWorkspaceSize, aclnnSinh)},
        {"Cosh", AclOpFunctions(aclnnCoshGetWorkspaceSize, aclnnCosh)},
        {"Tanh", AclOpFunctions(aclnnTanhGetWorkspaceSize, aclnnTanh)},
        {"Asinh", AclOpFunctions(aclnnAsinhGetWorkspaceSize, aclnnAsinh)},
        {"Acosh", AclOpFunctions(aclnnAcoshGetWorkspaceSize, aclnnAcosh)},
        {"Atanh", AclOpFunctions(aclnnAtanhGetWorkspaceSize, aclnnAtanh)},
        {"Sigmoid", AclOpFunctions(aclnnSigmoidGetWorkspaceSize, aclnnSigmoid)},
        {"Erf", AclOpFunctions(aclnnErfGetWorkspaceSize, aclnnErf)},
        {"Erfinv", AclOpFunctions(aclnnErfinvGetWorkspaceSize, aclnnErfinv)},
        {"LogicalNot", AclOpFunctions(aclnnLogicalNotGetWorkspaceSize, aclnnLogicalNot)},
        {"BitwiseNot", AclOpFunctions(aclnnBitwiseNotGetWorkspaceSize, aclnnBitwiseNot)},
        {"Neg", AclOpFunctions(aclnnNegGetWorkspaceSize, aclnnNeg)},
        {"Cast", AclOpFunctions(aclnnCastGetWorkspaceSize, aclnnCast)},
        {"Maximum", AclOpFunctions(aclnnMaximumGetWorkspaceSize, aclnnMaximum)},
        {"Minimum", AclOpFunctions(aclnnMinimumGetWorkspaceSize, aclnnMinimum)},
        {"Add", AclOpFunctions(aclnnAddGetWorkspaceSize, aclnnAdd)},
        {"Sub", AclOpFunctions(aclnnSubGetWorkspaceSize, aclnnSub)},
        {"Mul", AclOpFunctions(aclnnMulGetWorkspaceSize, aclnnMul)},
        {"RealDiv", AclOpFunctions(aclnnDivGetWorkspaceSize, aclnnDiv)},
        {"FloorDiv", AclOpFunctions(aclnnFloorDivideGetWorkspaceSize, aclnnFloorDivide)},
        {"LessEqual", AclOpFunctions(aclnnLeTensorGetWorkspaceSize, aclnnLeTensor)},
        {"Less", AclOpFunctions(aclnnLtTensorGetWorkspaceSize, aclnnLtTensor)},
        {"GreaterEqual", AclOpFunctions(aclnnGeTensorGetWorkspaceSize, aclnnGeTensor)},
        {"Greater", AclOpFunctions(aclnnGtTensorGetWorkspaceSize, aclnnGtTensor)},
        {"Equal", AclOpFunctions(aclnnEqTensorGetWorkspaceSize, aclnnEqTensor)},
        {"NotEqual", AclOpFunctions(aclnnNeTensorGetWorkspaceSize, aclnnNeTensor)},
        {"LogicalAnd", AclOpFunctions(aclnnLogicalAndGetWorkspaceSize, aclnnLogicalAnd)},
        {"LogicalOr", AclOpFunctions(aclnnLogicalOrGetWorkspaceSize, aclnnLogicalOr)},
        {"LogicalXor", AclOpFunctions(aclnnLogicalXorGetWorkspaceSize, aclnnLogicalXor)},
        {"BitwiseAnd", AclOpFunctions(aclnnBitwiseAndTensorGetWorkspaceSize, aclnnBitwiseAndTensor)},
        {"BitwiseOr", AclOpFunctions(aclnnBitwiseOrTensorGetWorkspaceSize, aclnnBitwiseOrTensor)},
        {"BitwiseXor", AclOpFunctions(aclnnBitwiseXorTensorGetWorkspaceSize, aclnnBitwiseXorTensor)},
        {"Pow", AclOpFunctions(aclnnPowTensorTensorGetWorkspaceSize, aclnnPowTensorTensor)},
        {"Expand", AclOpFunctions(aclnnExpandGetWorkspaceSize, aclnnExpand)},
        {"MatMul", AclOpFunctions(aclnnMatmulGetWorkspaceSize, aclnnMatmul)},
        {"BatchMatMul", AclOpFunctions(aclnnBatchMatMulGetWorkspaceSize, aclnnBatchMatMul)},
        {"Conv2D", AclOpFunctions(aclnnConvolutionGetWorkspaceSize, aclnnConvolution)},
        {"ReduceMax", AclOpFunctions(aclnnAmaxGetWorkspaceSize, aclnnAmax)},
        {"ReduceMin", AclOpFunctions(aclnnAminGetWorkspaceSize, aclnnAmin)},
        {"ReduceSum", AclOpFunctions(aclnnReduceSumGetWorkspaceSize, aclnnReduceSum)},
        {"Triu", AclOpFunctions(aclnnTriuGetWorkspaceSize, aclnnTriu)},
        {"Conv2d", AclOpFunctions(aclnnConvolutionGetWorkspaceSize, aclnnConvolution)},
        {"Conv2dBackward", AclOpFunctions(aclnnConvolutionBackwardGetWorkspaceSize, aclnnConvolutionBackward)},
        {"ReduceMean", AclOpFunctions(aclnnMeanGetWorkspaceSize, aclnnMean)},
        // {"ReduceProd", AclOpFunctions(aclnnProdDimGetWorkspaceSize, aclnnProdDim)},
        {"Select", AclOpFunctions(aclnnSWhereGetWorkspaceSize, aclnnSWhere)},
        {"RandomUniform", AclOpFunctions(aclnnInplaceRandomGetWorkspaceSize, aclnnInplaceRandom)},
        {"RandomNormal", AclOpFunctions(aclnnInplaceNormalGetWorkspaceSize, aclnnInplaceNormal)},
        {"Transpose", AclOpFunctions(aclnnPermuteGetWorkspaceSize, aclnnPermute)},
    };

    struct AclOpAttr
    {
        virtual ~AclOpAttr() {}
    };

    struct ConvAttr : AclOpAttr
    {
        vector<int64_t> convStrides;
        vector<int64_t> convPads;
        vector<int64_t> convOutPads;
        vector<int64_t> convDilations;
        bool convWithBias;
        bool is_transposed;
        int64_t group;

        // 析构函数
        ~ConvAttr()
        {
            convStrides.clear();
            convPads.clear();
            convOutPads.clear();
            convDilations.clear();
        }
    };

    struct ReduceAttr : AclOpAttr
    {
        vector<int64_t> axes;
        // for proddim
        int64_t prod_dim;
        bool keepdims;

        ~ReduceAttr()
        {
            axes.clear();
        }
    };

    struct RandomAttr : AclOpAttr
    {
        int64_t seed, offset;

        ~RandomAttr()
        {
        }
    };

    struct TriuAttr : AclOpAttr
    {
        int64_t diagonal;

        ~TriuAttr()
        {
        }
    };

}