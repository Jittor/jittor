#pragma once
#include <unordered_map>
#include <string>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include "misc/nano_string.h"

namespace jittor
{
    aclDataType get_dtype(NanoString s);

    extern std::unordered_map<string, int> op_idx_map;
    int CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                        aclDataType dataType, aclTensor **tensor, bool use_nchw = false);

    int CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                 aclDataType dataType, aclTensor **tensor, bool use_nchw = false);
}