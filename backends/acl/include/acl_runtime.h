#pragma once
#include "core/common.h"
#include <acl/acl.h>

namespace jittor {

EXTERN_LIB int acl_runtime_current_device();
EXTERN_LIB aclrtStream acl_current_stream();
EXTERN_LIB void shutdown_acl_backend() noexcept;
// Reserve one native ACL random stream segment on the current device.
EXTERN_LIB void reserve_acl_random(int device, int64 count, uint64& seed, uint64& offset);

} // namespace jittor
