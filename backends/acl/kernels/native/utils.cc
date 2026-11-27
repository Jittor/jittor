#include <cstring>
#include <unordered_map>
#include <string>
#include <vector>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <aclnn/acl_meta.h>
#include <Python.h>
#include <pystate.h>
#include "utils.h"
#include "core/var.h"
#include "aclnn/aclnn.h"

namespace jittor
{
    const char* acl_soc_name()
    {
        // Cached: the SoC cannot change within a process, and aclrtGetSocName
        // needs the ACL context that any tensor-descriptor caller already set
        // up. Callers only reach this on the ACL execution path.
        static const char* name = [] {
            const char* value = aclrtGetSocName();
            return value ? value : "";
        }();
        return name;
    }

    bool acl_soc_supports_bfloat16()
    {
        static const bool value =
            std::strncmp(acl_soc_name(), "Ascend310P", 10) != 0;
        return value;
    }

    bool acl_soc_supports_fp8()
    {
        static const bool value =
            std::strncmp(acl_soc_name(), "Ascend950", 9) == 0;
        return value;
    }

    bool acl_soc_supports_fp32_cube()
    {
        static const bool value =
            std::strncmp(acl_soc_name(), "Ascend310", 9) != 0;
        return value;
    }

    void acl_check_fp32_cube(int cube_math_type, NanoString dtype, const char* op_name)
    {
        if (cube_math_type != 0 || dtype != ns_float32
            || acl_soc_supports_fp32_cube())
            return;
        USER_CHECK(false) << op_name << ": FP32 cube math is unsupported on"
                          << acl_soc_name()
                          << "with KEEP_DTYPE; enable the acl_allow_hf32 flag"
                             " (or jt.acl_allow_hf32) to opt in to"
                             " ALLOW_FP32_DOWN_PRECISION";
    }

    aclDataType get_dtype(NanoString s)
    {
        if (s == ns_bfloat16)
        {
            USER_CHECK(acl_soc_supports_bfloat16())
                << "bfloat16 is not supported on" << acl_soc_name()
                << ": the installed CANN op package has no DT_BFLOAT16 kernels";
            return ACL_BF16;
        }
        if (s == ns_float32)
            return ACL_FLOAT;
        if (s == ns_float16)
            return ACL_FLOAT16;
        if (s == ns_int64)
            return ACL_INT64;
        if (s == ns_int32)
            return ACL_INT32;
        if (s == ns_int8)
            return ACL_INT8;
        if (s == ns_int16)
            return ACL_INT16;
        if (s == ns_uint8)
            return ACL_UINT8;
        if (s == ns_uint16)
            return ACL_UINT16;
        if (s == ns_uint32)
            return ACL_UINT32;
        if (s == ns_bool)
            return ACL_BOOL;
        if (s == ns_complex64)
            return ACL_COMPLEX64;
        if (s == ns_float8_e4m3fn)
        {
            USER_CHECK(acl_soc_supports_fp8())
                << "float8_e4m3fn is not supported on" << acl_soc_name()
                << ": native FP8 requires Ascend950-class hardware";
#if JT_ACL_HAS_FP8_OPS
            return ACL_FLOAT8_E4M3FN;
#else
            return static_cast<aclDataType>(36);
#endif
        }
        if (s == ns_float8_e5m2)
        {
            USER_CHECK(acl_soc_supports_fp8())
                << "float8_e5m2 is not supported on" << acl_soc_name()
                << ": native FP8 requires Ascend950-class hardware";
#if JT_ACL_HAS_FP8_OPS
            return ACL_FLOAT8_E5M2;
#else
            return static_cast<aclDataType>(35);
#endif
        }
        if (s == ns_float8_e8m0fnu)
        {
            USER_CHECK(acl_soc_supports_fp8())
                << "float8_e8m0fnu is not supported on" << acl_soc_name()
                << ": native MXFP8 scale requires Ascend950-class hardware";
#if JT_ACL_HAS_FP8_OPS
            return ACL_FLOAT8_E8M0;
#else
            return static_cast<aclDataType>(37);
#endif
        }
        LOGf << "Not supported dtype: " << s;
        return ACL_FLOAT;
    }

    // A jittor Var shape is a NanoVector, which cannot hold more than 15 axes,
    // so the descriptor argument buffers live on the stack. Building them in
    // std::vectors cost two malloc/free pairs per tensor per launch.
    static constexpr size_t kMaxAclRank = 16;

    static void apply_storage_strides(const int64_t* shape, size_t rank,
                                      int64_t* strides, const Var* storage) {
        if (!storage || storage->is_contiguous()) return;
        int source = int(storage->shape.size())-1;
        for (int axis=int(rank)-1; axis>=0; --axis) {
            if (source < 0) {
                USER_CHECK(shape[axis] == 1) << "ACL descriptor cannot reshape strided storage";
                strides[axis] = 0;
            } else {
                USER_CHECK(shape[axis] == storage->shape[source])
                    << "ACL descriptor cannot reshape strided storage";
                strides[axis] = storage->storage_stride(source--);
            }
        }
        USER_CHECK(source < 0) << "ACL descriptor dropped storage dimensions";
    }

    namespace {

    // The exact argument set aclCreateTensor and aclInitTensor both take.
    struct AclTensorLayout {
        int64_t strides[kMaxAclRank];
        int64_t packed_storage = 0;
        const int64_t* storage_dims = nullptr;
        size_t storage_rank = 0;
        aclFormat format = aclFormat::ACL_FORMAT_ND;
    };

    void describe_tensor(const int64_t* shape, size_t rank, bool use_nchw,
                         const Var* storage, AclTensorLayout& layout) {
        USER_CHECK(rank <= kMaxAclRank)
            << "ACL descriptor rank is above the supported maximum:" << rank;
        // 计算连续tensor的strides
        if (rank) {
            layout.strides[rank - 1] = 1;
            for (int64_t i = int64_t(rank) - 2; i >= 0; i--)
                layout.strides[i] = shape[i + 1] * layout.strides[i + 1];
        }
        apply_storage_strides(shape, rank, layout.strides, storage);
        layout.storage_dims = shape;
        layout.storage_rank = rank;
        if (storage && !storage->is_contiguous()) {
            layout.packed_storage = storage->storage_span_bytes() / storage->dsize();
            layout.storage_dims = &layout.packed_storage;
            layout.storage_rank = 1;
        }
        layout.format = use_nchw
            ? (rank == 5 ? aclFormat::ACL_FORMAT_NCDHW : aclFormat::ACL_FORMAT_NCHW)
            : aclFormat::ACL_FORMAT_ND;
    }

    // The pool lives in the runner's leased scratch, so the launch path costs
    // no thread-local lookup per descriptor. Its size is bounded by the widest
    // operator that scratch block has ever served.
    constexpr size_t kMaxPooledTensors = 256;

    } // namespace

    aclError CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                             aclDataType dataType, aclTensor **tensor, bool use_nchw, const Var* storage)
    {
        AclTensorLayout layout;
        describe_tensor(shape.data(), shape.size(), use_nchw, storage, layout);
        // 调用aclCreateTensor接口创建aclTensor
        *tensor = nullptr;
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, layout.strides, 0,
                                  layout.format, layout.storage_dims, layout.storage_rank, deviceAddr);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }

    aclError CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                      aclDataType dataType, aclTensor **tensor, bool use_nchw, const Var* storage)
    {
        // strides and the storage view follow the original shape; the last two
        // axes are only swapped afterwards, exactly as before.
        AclTensorLayout layout;
        describe_tensor(shape.data(), shape.size(), use_nchw, storage, layout);
        int n = shape.size();
        if (n > 1)
        {
            std::swap(shape[n - 1], shape[n - 2]);
            std::swap(layout.strides[n - 1], layout.strides[n - 2]);
            // storage_dims aliases shape unless the storage was packed above,
            // so the swap is already reflected in the storage view.
        }
        // 调用aclCreateTensor接口创建aclTensor
        *tensor = nullptr;
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, layout.strides, 0,
                                  layout.format, layout.storage_dims, layout.storage_rank, deviceAddr);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }

    aclError AcquireRawAclTensor(std::vector<aclTensor *> &pool,
                                 const int64_t *view_dims, size_t rank,
                                 aclDataType dataType, const int64_t *strides,
                                 const int64_t *storage_dims, size_t storage_rank,
                                 void *deviceAddr, aclTensor **tensor,
                                 aclFormat format)
    {
        *tensor = nullptr;
        while (rank > 0 && !pool.empty()) {
            aclTensor* recycled = pool.back();
            pool.pop_back();
            auto status = aclInitTensor(recycled, view_dims, rank, dataType,
                                        strides, 0, format,
                                        storage_dims, storage_rank, deviceAddr);
            if (status == ACL_SUCCESS) {
                *tensor = recycled;
                return ACL_SUCCESS;
            }
            aclDestroyTensor(recycled);
        }
        *tensor = aclCreateTensor(view_dims, rank, dataType, strides, 0,
                                  format, storage_dims, storage_rank, deviceAddr);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }

    aclError AcquireFakeTransAclTensor(std::vector<aclTensor *> &pool,
                                       std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                       aclDataType dataType, aclTensor **tensor, bool use_nchw, const Var* storage)
    {
        AclTensorLayout layout;
        describe_tensor(shape.data(), shape.size(), use_nchw, storage, layout);
        int n = shape.size();
        if (n > 1)
        {
            std::swap(shape[n - 1], shape[n - 2]);
            std::swap(layout.strides[n - 1], layout.strides[n - 2]);
        }
        return AcquireRawAclTensor(pool, shape.data(), shape.size(), dataType,
                                   layout.strides, layout.storage_dims, layout.storage_rank,
                                   deviceAddr, tensor, layout.format);
    }

    aclError AcquireAclTensor(std::vector<aclTensor *> &pool,
                              const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                              aclDataType dataType, aclTensor **tensor, bool use_nchw, const Var* storage)
    {
        AclTensorLayout layout;
        describe_tensor(shape.data(), shape.size(), use_nchw, storage, layout);
        return AcquireRawAclTensor(pool, shape.data(), shape.size(), dataType,
                                   layout.strides, layout.storage_dims, layout.storage_rank,
                                   deviceAddr, tensor, layout.format);
    }

    void RecycleAclTensor(std::vector<aclTensor *> &pool, aclTensor *tensor)
    {
        if (!tensor) return;
        if (pool.size() >= kMaxPooledTensors) {
            aclDestroyTensor(tensor);
            return;
        }
        pool.push_back(tensor);
    }
}
