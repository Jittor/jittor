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

    aclError CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                             aclDataType dataType, aclTensor **tensor, bool use_nchw = false, const Var* storage = nullptr);

    aclError CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                      aclDataType dataType, aclTensor **tensor, bool use_nchw = false, const Var* storage = nullptr);
}
