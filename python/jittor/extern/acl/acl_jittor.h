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
    EXTERN_LIB void *workspaceAddr;

    void mallocWorkSpace(uint64_t size);

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
        // for Expand, permute, flip
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
        // for select, where
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncSelect;
        // for random_uniform and random_normal
        std::function<aclnnStatus(aclTensor *, int64_t, int64_t, int64_t, int64_t, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncRandom;
        // for maxpool
        std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncMaxPool;
        // for maxpool backward
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncMaxPoolBackward;
        // for avgpool
        std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, bool, bool, int64_t, int8_t, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncAvgPool;
        // for concat
        std::function<aclnnStatus(aclTensorList *, uint64_t, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncConcat;
        // for gather
        std::function<aclnnStatus(aclTensor *, uint64_t, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncGather;
        // for cumsum
        std::function<aclnnStatus(aclTensor *, uint64_t, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncCumsum;
        // for scatter
        std::function<aclnnStatus(aclTensor *, uint64_t, aclTensor *, aclTensor *, uint64_t, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncScatter;
        // for index
        std::function<aclnnStatus(aclTensor *, aclTensorList *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncIndex;
        // for stridesliceassign
        std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncStridedSliceAssignV2;
        // for slicev2
        std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncSliceV2;
        // for indexputimpl
        std::function<aclnnStatus(aclTensor *, aclTensorList *, aclTensor *, bool, bool, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncIndexPutImpl;
        // for range
        std::function<aclnnStatus(aclScalar *, aclScalar *, aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **)> getWorkspaceSizeFuncRange;


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

        // for Expand, flip
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

        // for select, where
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncSelect(gwsf), executeFunc(execf) {}

        // for random_normal
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, int64_t, int64_t, int64_t, int64_t, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncRandom(gwsf), executeFunc(execf) {}

        // for maxpool
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncMaxPool(gwsf), executeFunc(execf) {}

        // for maxpool backward
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, bool, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncMaxPoolBackward(gwsf), executeFunc(execf) {}

        // for avgpool
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, bool, bool, int64_t, int8_t, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncAvgPool(gwsf), executeFunc(execf) {}

        // for concat
        AclOpFunctions(std::function<aclnnStatus(aclTensorList *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncConcat(gwsf), executeFunc(execf) {}

        // for gather
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, int64_t, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncGather(gwsf), executeFunc(execf) {}

        // for cumsum
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, int64_t, aclDataType, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncCumsum(gwsf), executeFunc(execf) {}

        // for scatter
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, uint64_t, aclTensor *, aclTensor *, uint64_t, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncScatter(gwsf), executeFunc(execf) {}

        // for index
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensorList *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncIndex(gwsf), executeFunc(execf) {}

        // for stridesliceassignv2
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncStridedSliceAssignV2(gwsf), executeFunc(execf) {}

        // for slicev2
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclIntArray *, aclIntArray *, aclIntArray *, aclIntArray *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncSliceV2(gwsf), executeFunc(execf) {}

        // for indexputimpl
        AclOpFunctions(std::function<aclnnStatus(aclTensor *, aclTensorList *, aclTensor *, bool, bool, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncIndexPutImpl(gwsf), executeFunc(execf) {}

        // for range
        AclOpFunctions(std::function<aclnnStatus(aclScalar *, aclScalar *, aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **)> gwsf,
                       std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, const aclrtStream)> execf)
            : getWorkspaceSizeFuncRange(gwsf), executeFunc(execf) {}
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
        {"Maxpool", AclOpFunctions(aclnnMaxPool2dWithIndicesGetWorkspaceSize, aclnnMaxPool2dWithIndices)},
        {"MaxpoolBackward", AclOpFunctions(aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize, aclnnMaxPool2dWithIndicesBackward)},
        {"Flip", AclOpFunctions(aclnnFlipGetWorkspaceSize, aclnnFlip)},
        {"Concat", AclOpFunctions(aclnnCatGetWorkspaceSize, aclnnCat)},
        {"Gather", AclOpFunctions(aclnnGatherGetWorkspaceSize, aclnnGather)},
        {"Cumsum", AclOpFunctions(aclnnCumsumGetWorkspaceSize, aclnnCumsum)},
        {"Index", AclOpFunctions(aclnnIndexGetWorkspaceSize, aclnnIndex)},
        {"Scatter", AclOpFunctions(aclnnScatterGetWorkspaceSize, aclnnScatter)},
        {"Where", AclOpFunctions(aclnnSWhereGetWorkspaceSize, aclnnSWhere)},
        {"Floor", AclOpFunctions(aclnnFloorGetWorkspaceSize, aclnnFloor)},
        {"StridedSliceAssignV2", AclOpFunctions(aclnnStridedSliceAssignV2GetWorkspaceSize, aclnnStridedSliceAssignV2)},
        {"SliceV2", AclOpFunctions(aclnnSliceV2GetWorkspaceSize, aclnnSliceV2)},
        {"IndexPutImpl", AclOpFunctions(aclnnIndexPutImplGetWorkspaceSize, aclnnIndexPutImpl)},
        {"Range", AclOpFunctions(aclnnRangeGetWorkspaceSize, aclnnRange)},
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

    struct PoolAttr : AclOpAttr
    {
        vector<int64_t> kernel_size;
        vector<int64_t> poolStrides;
        vector<int64_t> poolPads;
        vector<int64_t> poolDilations;
        bool poolCeil;

        // 析构函数
        ~PoolAttr()
        {
            kernel_size.clear();
            poolStrides.clear();
            poolPads.clear();
            poolDilations.clear();
        }
    };

    struct ConcatAttr : AclOpAttr
    {
        int64_t tensorNum;
        int64_t dim;

        ~ConcatAttr()
        {
        }
    };

    struct GatherAttr : AclOpAttr
    {
        int64_t dim;

        ~GatherAttr()
        {
        }
    };

    struct ScatterAttr : AclOpAttr
    {
        int64_t axis;
        int64_t reduction;

        ~ScatterAttr()
        {
        }
    };

    struct StrideAttr : AclOpAttr
    {
        vector<int64_t> begins;
        vector<int64_t> ends;
        vector<int64_t> steps;
        vector<int64_t> axes;
        ~StrideAttr()
        {
            begins.clear();
            ends.clear();
            steps.clear();
            axes.clear();
        }
    };

    struct RangeAttr : AclOpAttr
    {
        int64_t start;
        int64_t end;
        int64_t step;

        ~RangeAttr()
        {
        }
    };

}