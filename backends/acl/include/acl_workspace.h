#pragma once

#include "core/common.h"

namespace jittor {

EXTERN_LIB void* acl_workspace_address();
EXTERN_LIB void* mallocWorkSpace(uint64_t size);
EXTERN_LIB void releaseWorkSpace();
EXTERN_LIB void release_all_acl_workspaces() noexcept;

} // namespace jittor
