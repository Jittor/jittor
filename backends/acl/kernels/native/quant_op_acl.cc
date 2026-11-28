#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "core/common.h"
#include "core/var.h"
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "quant_op_acl.h"

namespace jittor
{
    DynamicBlockQuantOpRunner::DynamicBlockQuantOpRunner() : BaseOpRunner("DynamicBlockQuant")
    {
    }

    void DynamicBlockQuantOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        USER_CHECK(acl_soc_supports_fp8())
            << "DynamicBlockQuant requires Ascend950-class hardware, got" << acl_soc_name();
        auto *attr = dynamic_cast<DynamicBlockQuantAttr *>(op_attr.get());
        USER_CHECK(attr != nullptr) << "DynamicBlockQuant requires DynamicBlockQuantAttr";
        USER_CHECK(inputTensors.size() == 1 && outputTensors.size() == 2)
            << "DynamicBlockQuant expects 1 input and 2 outputs";
        aclDataType dst_type = get_dtype(out_[0]->dtype());
#if JT_ACL_HAS_FP8_OPS
        char round_mode[] = "rint";
        ret = aclnnDynamicBlockQuantGetWorkspaceSize(
            inputTensors[0],
            attr->minScale,
            round_mode,
            static_cast<int64_t>(dst_type),
            attr->rowBlockSize,
            attr->colBlockSize,
            outputTensors[0],
            outputTensors[1],
            &workspaceSize,
            &executor);
        launch(ret, aclnnDynamicBlockQuant, true);
#else
        (void)dst_type;
        USER_CHECK(false) << "DynamicBlockQuant requires CANN 9.2+ FP8 headers";
#endif
    }

    DynamicMxQuantOpRunner::DynamicMxQuantOpRunner() : BaseOpRunner("DynamicMxQuant")
    {
    }

    void DynamicMxQuantOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        USER_CHECK(acl_soc_supports_fp8())
            << "DynamicMxQuant requires Ascend950-class hardware, got" << acl_soc_name();
        auto *attr = dynamic_cast<DynamicMxQuantAttr *>(op_attr.get());
        USER_CHECK(attr != nullptr) << "DynamicMxQuant requires DynamicMxQuantAttr";
        USER_CHECK(inputTensors.size() == 1 && outputTensors.size() == 2)
            << "DynamicMxQuant expects 1 input and 2 outputs";
        aclDataType dst_type = get_dtype(out_[0]->dtype());
#if JT_ACL_HAS_FP8_OPS
        char round_mode[] = "rint";
        ret = aclnnDynamicMxQuantGetWorkspaceSize(
            inputTensors[0],
            attr->axis,
            round_mode,
            static_cast<int64_t>(dst_type),
            attr->blockSize,
            attr->scaleAlg,
            outputTensors[0],
            outputTensors[1],
            &workspaceSize,
            &executor);
        launch(ret, aclnnDynamicMxQuant, true);
#else
        (void)dst_type;
        USER_CHECK(false) << "DynamicMxQuant requires CANN 9.2+ FP8 headers";
#endif
    }

    QuantMatmulOpRunner::QuantMatmulOpRunner() : BaseOpRunner("QuantMatmul")
    {
    }

    static aclTensor *acquire_transposed_2d_tensor(
        std::vector<aclTensor *> &pool, void *addr, int64_t rows, int64_t cols, aclDataType dt)
    {
        // Storage is contiguous [rows, cols]; logical view is transposed [cols, rows].
        int64_t view_dims[2] = {cols, rows};
        int64_t strides[2] = {1, cols};
        int64_t storage_dims[2] = {rows, cols};
        aclTensor *out = nullptr;
        auto st = AcquireRawAclTensor(
            pool, view_dims, 2, dt, strides, storage_dims, 2, addr, &out);
        return st == ACL_SUCCESS ? out : nullptr;
    }

    static aclTensor *acquire_transposed_mx_scale_tensor(
        std::vector<aclTensor *> &pool, void *addr, int64_t n, int64_t k_half, int64_t pair, aclDataType dt)
    {
        // Storage is contiguous [N, K/64, 2]; logical view is [K/64, N, 2] with strides {2, K/32, 1}.
        int64_t view_dims[3] = {k_half, n, pair};
        int64_t strides[3] = {pair, k_half * pair, 1};
        int64_t storage_dims[3] = {n, k_half, pair};
        aclTensor *out = nullptr;
        auto st = AcquireRawAclTensor(
            pool, view_dims, 3, dt, strides, storage_dims, 3, addr, &out);
        return st == ACL_SUCCESS ? out : nullptr;
    }

    void QuantMatmulOpRunner::setupInputDesc()
    {
        auto *attr = dynamic_cast<QuantMatmulAttr *>(op_attr.get());
        USER_CHECK(attr != nullptr) << "QuantMatmul requires QuantMatmulAttr";
        USER_CHECK(in_.size() >= 4) << "QuantMatmul expects at least 4 inputs (x1, x2, x1Scale, x2Scale)";

        inputShapes.resize(in_.size());
        inputTensors.assign(in_.size(), nullptr);

        for (size_t i = 0; i < in_.size(); ++i)
        {
            Var *v = in_[i];
            auto &shape = inputShapes[i];
            shape.resize(v->shape.size());
            for (int d = 0; d < v->shape.size(); ++d)
                shape[d] = v->shape[d];

            aclDataType dt = get_dtype(v->dtype());

            if (i == 0 && attr->transposeX1)
            {
                USER_CHECK(v->shape.size() == 2 && v->is_contiguous())
                    << "QuantMatmul transposeX1 expects contiguous 2-D x1";
                inputTensors[i] = acquire_transposed_2d_tensor(
                    scratch->descriptors, v->mem_ptr, v->shape[0], v->shape[1], dt);
                USER_CHECK(inputTensors[i] != nullptr) << "QuantMatmul failed to create transposed x1 descriptor";
                continue;
            }

            if (i == 1)
            {
                bool strided_trans_view =
                    !v->is_contiguous() && v->shape.size() == 2 &&
                    v->storage_stride(0) == 1 && v->storage_stride(1) == v->shape[0];
                if (attr->transposeX2)
                {
                    USER_CHECK(v->shape.size() == 2 && v->is_contiguous())
                        << "QuantMatmul transposeX2 expects contiguous 2-D x2 [N, K]";
                    inputTensors[i] = acquire_transposed_2d_tensor(
                        scratch->descriptors, v->mem_ptr, v->shape[0], v->shape[1], dt);
                    USER_CHECK(inputTensors[i] != nullptr) << "QuantMatmul failed to create transposed x2 descriptor";
                    continue;
                }
                if (strided_trans_view)
                {
                    // Zero-copy view of a transposed [N, K] tensor passed as logical [K, N].
                    inputTensors[i] = acquire_transposed_2d_tensor(
                        scratch->descriptors, v->mem_ptr, v->shape[1], v->shape[0], dt);
                    USER_CHECK(inputTensors[i] != nullptr) << "QuantMatmul failed to create strided x2 descriptor";
                    continue;
                }
            }

            if (i == 3 && attr->transposeX2Scale)
            {
                USER_CHECK(v->is_contiguous()) << "QuantMatmul transposeX2Scale expects contiguous x2Scale";
                if (v->shape.size() == 2)
                {
                    inputTensors[i] = acquire_transposed_2d_tensor(
                        scratch->descriptors, v->mem_ptr, v->shape[0], v->shape[1], dt);
                }
                else if (v->shape.size() == 3)
                {
                    inputTensors[i] = acquire_transposed_mx_scale_tensor(
                        scratch->descriptors, v->mem_ptr, v->shape[0], v->shape[1], v->shape[2], dt);
                }
                else
                {
                    USER_CHECK(false) << "QuantMatmul transposeX2Scale expects rank 2 or 3 scale, got"
                                      << v->shape.size();
                }
                USER_CHECK(inputTensors[i] != nullptr) << "QuantMatmul failed to create transposed x2Scale descriptor";
                continue;
            }

            auto st = AcquireAclTensor(
                scratch->descriptors, inputShapes[i], v->mem_ptr, v->size,
                dt, &inputTensors[i], false, v);
            USER_CHECK(st == ACL_SUCCESS && inputTensors[i] != nullptr)
                << "QuantMatmul input descriptor creation failed for index" << i << "status:" << st;
        }
    }

    void QuantMatmulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        USER_CHECK(acl_soc_supports_fp8())
            << "QuantMatmul requires Ascend950-class hardware, got" << acl_soc_name();
        auto *attr = dynamic_cast<QuantMatmulAttr *>(op_attr.get());
        USER_CHECK(attr != nullptr) << "QuantMatmul requires QuantMatmulAttr";
        USER_CHECK(inputTensors.size() >= 4 && outputTensors.size() == 1)
            << "QuantMatmul expects 4 or 5 inputs and 1 output";
        aclTensor *bias_tensor = (attr->hasBias && inputTensors.size() >= 5) ? inputTensors[4] : nullptr;
#if JT_ACL_HAS_FP8_OPS
        ret = aclnnQuantMatmulV5GetWorkspaceSize(
            inputTensors[0],
            inputTensors[1],
            inputTensors[2],
            inputTensors[3],
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            bias_tensor,
            false,
            false,
            attr->groupSize,
            outputTensors[0],
            &workspaceSize,
            &executor);
        launch(ret, aclnnQuantMatmulV5, true);
#else
        (void)bias_tensor;
        USER_CHECK(false) << "QuantMatmul requires CANN 9.2+ FP8 headers";
#endif
    }

    Fp8LinearOpRunner::Fp8LinearOpRunner() : BaseOpRunner("QuantMatmul")
    {
    }

    void Fp8LinearOpRunner::setupInputDesc()
    {
        // Descriptors are constructed directly in executeOp with explicit 2-D
        // flattened views and transposed weight/scale views, avoiding redundant
        // generic setup/cleanup work.
        inputTensors.clear();
    }

    void Fp8LinearOpRunner::setupOutputDesc()
    {
        outputTensors.clear();
    }

    void Fp8LinearOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        (void)it;
        auto &pool = scratch->descriptors;
        struct PooledAclTensor
        {
            std::vector<aclTensor *> &pool;
            aclTensor *ptr = nullptr;
            ~PooledAclTensor()
            {
                if (ptr)
                    RecycleAclTensor(pool, ptr);
            }
        };
        USER_CHECK(acl_soc_supports_fp8())
            << "Fp8Linear requires Ascend950-class hardware, got" << acl_soc_name();
        USER_CHECK(in_.size() == 3 && out_.size() == 1 && qx_scratch && sx_scratch)
            << "Fp8Linear expects 3 inputs, 1 output and 2 scratch buffers";
        Var *x = in_[0];
        Var *w = in_[1];
        Var *ws = in_[2];
        Var *y = out_[0];
        USER_CHECK(x->is_contiguous() && w->is_contiguous() && ws->is_contiguous() && y->is_contiguous())
            << "Fp8Linear requires contiguous operands";

        const int64_t k_dim = w->shape[1];
        const int64_t n_dim = w->shape[0];
        const int64_t m_dim = x->num / k_dim;
        const aclDataType x_dt = get_dtype(x->dtype());
        const aclDataType q_dt = get_dtype(w->dtype());
        const aclDataType y_dt = get_dtype(y->dtype());

        auto make_contig = [&](const int64_t *dims, uint64_t rank, aclDataType dt, void *ptr) {
            int64_t strides[4];
            strides[rank - 1] = 1;
            for (int64_t d = static_cast<int64_t>(rank) - 2; d >= 0; --d)
                strides[d] = strides[d + 1] * dims[d + 1];
            aclTensor *t = nullptr;
            auto st = AcquireRawAclTensor(pool, dims, rank, dt, strides, dims, rank, ptr, &t);
            USER_CHECK(st == ACL_SUCCESS && t != nullptr)
                << "Fp8Linear descriptor creation failed, status:" << st;
            return t;
        };

        const int64_t x_dims[2] = {m_dim, k_dim};
        const int64_t y_dims[2] = {m_dim, n_dim};
        PooledAclTensor tx{pool, make_contig(x_dims, 2, x_dt, x->mem_ptr)};
        PooledAclTensor tqx{pool, make_contig(x_dims, 2, q_dt, qx_scratch->mem_ptr)};
        PooledAclTensor tsx{pool, nullptr};
        PooledAclTensor tw{pool, acquire_transposed_2d_tensor(pool, w->mem_ptr, n_dim, k_dim, q_dt)};
        USER_CHECK(tw.ptr != nullptr) << "Fp8Linear weight descriptor creation failed";
        PooledAclTensor tsw{pool, nullptr};
        PooledAclTensor ty{pool, make_contig(y_dims, 2, y_dt, y->mem_ptr)};

        int64_t group_size = 0;
#if JT_ACL_HAS_FP8_OPS
        char round_mode[] = "rint";

        if (mode == 0)
        {
            const int64_t k_blk = k_dim / block_size;
            const int64_t n_blk = n_dim / block_size;
            const int64_t sx_dims[2] = {m_dim, k_blk};
            tsx.ptr = make_contig(sx_dims, 2, ACL_FLOAT, sx_scratch->mem_ptr);
            tsw.ptr = acquire_transposed_2d_tensor(pool, ws->mem_ptr, n_blk, k_blk, ACL_FLOAT);
            USER_CHECK(tsw.ptr != nullptr) << "Fp8Linear scale descriptor creation failed";
            group_size = (1LL << 32) | (block_size << 16) | block_size;

            ret = aclnnDynamicBlockQuantGetWorkspaceSize(
                tx.ptr, 0.0, round_mode, static_cast<int64_t>(q_dt),
                1, block_size, tqx.ptr, tsx.ptr, &workspaceSize, &executor);
            launch(ret, aclnnDynamicBlockQuant, false);
        }
        else
        {
            const int64_t k_half = k_dim / (2 * block_size);
            const int64_t sx_dims[3] = {m_dim, k_half, 2};
            tsx.ptr = make_contig(sx_dims, 3, ACL_FLOAT8_E8M0, sx_scratch->mem_ptr);
            tsw.ptr = acquire_transposed_mx_scale_tensor(pool, ws->mem_ptr, n_dim, k_half, 2, ACL_FLOAT8_E8M0);
            USER_CHECK(tsw.ptr != nullptr) << "Fp8Linear MX scale descriptor creation failed";
            group_size = (1LL << 32) | (1LL << 16) | block_size;

            ret = aclnnDynamicMxQuantGetWorkspaceSize(
                tx.ptr, -1, round_mode, static_cast<int64_t>(q_dt),
                block_size, 0, tqx.ptr, tsx.ptr, &workspaceSize, &executor);
            launch(ret, aclnnDynamicMxQuant, false);
        }

        workspaceSize = 0;
        executor = nullptr;
        ret = aclnnQuantMatmulV5GetWorkspaceSize(
            tqx.ptr, tw.ptr, tsx.ptr, tsw.ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            false, false, group_size, ty.ptr, &workspaceSize, &executor);
        launch(ret, aclnnQuantMatmulV5, true);
#else
        (void)group_size;
        USER_CHECK(false) << "Fp8Linear requires CANN 9.2+ FP8 headers";
#endif
    }
} // namespace jittor
