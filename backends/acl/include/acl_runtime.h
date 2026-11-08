#pragma once
#include "core/common.h"
#include <acl/acl.h>

namespace jittor {

EXTERN_LIB int acl_runtime_current_device();
EXTERN_LIB aclrtStream acl_current_stream();
EXTERN_LIB void shutdown_acl_backend() noexcept;
// Reserve one native ACL random stream segment on the current device.
EXTERN_LIB void reserve_acl_random(int device, int64 count, uint64& seed, uint64& offset);

// Recorded device graphs (aclmdlRI) alive or being recorded. A recording
// re-issues the addresses it saw -- the operator workspace, the host scalars
// its copies read -- so while one lives a block that would be replaced is
// kept instead of freed, and no stream is waited on for it (the stream may be
// the one being recorded). Kept blocks go when the last graph does.
EXTERN_LIB int acl_live_graphs();
EXTERN_LIB void acl_scalar_cache_release_retired();

} // namespace jittor
