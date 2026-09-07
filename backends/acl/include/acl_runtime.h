#pragma once
#include "core/common.h"
#include <acl/acl.h>

namespace jittor {

EXTERN_LIB int acl_runtime_current_device();
EXTERN_LIB aclrtStream acl_current_stream();
EXTERN_LIB void shutdown_acl_backend() noexcept;

} // namespace jittor
