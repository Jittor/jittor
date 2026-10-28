#pragma once
#include <unordered_map>
#include <string>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include "type/nano_string.h"
#include "aclnn/aclnn.h"

namespace jittor
{
    struct Var;
    aclDataType get_dtype(NanoString s);

    // CANN SoC identity, queried once per process. The inference series
    // (Ascend310*) has a smaller op package: no DT_BFLOAT16 kernels, and a
    // cube unit that cannot run FP32 with KEEP_DTYPE.
    const char* acl_soc_name();
    bool acl_soc_supports_bfloat16();
    bool acl_soc_supports_fp32_cube();
    // Fails unless FP32 cube math can run on this SoC or the caller opted
    // into down-conversion, so matmul/bmm/conv report the opt-in identically.
    void acl_check_fp32_cube(int cube_math_type, NanoString dtype, const char* op_name);

    aclError CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                             aclDataType dataType, aclTensor **tensor, bool use_nchw = false, const Var* storage = nullptr);

    aclError CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                      aclDataType dataType, aclTensor **tensor, bool use_nchw = false, const Var* storage = nullptr);

    // Descriptor recycling for the per-operator launch path.
    //
    // AcquireAclTensor takes a descriptor out of the caller's pool and rewrites
    // *every* field of it through aclInitTensor -- view dims, rank, dtype,
    // strides, offset, format, storage dims and data pointer -- so a recycled
    // object cannot carry a shape, a dtype or an address from the operator that
    // used it last. There is no cache key and nothing is matched: the pool only
    // removes the aclCreateTensor/aclDestroyTensor pair. A rank-0 descriptor is
    // always built fresh, because CANN reads viewDimsNum == 0 as "leave the
    // view alone". RecycleAclTensor returns a descriptor to the pool; it
    // accepts any aclTensor, including one built by CreateAclTensor.
    aclError AcquireAclTensor(std::vector<aclTensor *> &pool,
                              const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                              aclDataType dataType, aclTensor **tensor, bool use_nchw = false,
                              const Var* storage = nullptr);
    void RecycleAclTensor(std::vector<aclTensor *> &pool, aclTensor *tensor);
}
