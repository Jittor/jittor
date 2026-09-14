#pragma once

#include "core/common.h"
#include <acl/acl.h>
#include "aclnn/acl_meta.h"

namespace jittor {

// CANN's foreach operators take their coefficient (`alpha`, `scalar`) as a
// one-element *device* tensor, so a runner that wants to launch them has to get
// a float onto the card first. This owns that staging area: one pinned host
// ring and one device ring per device, drained and released with the backend.
//
// `tensors` receives `count` freshly created one-element aclTensors, which the
// caller destroys; the memory behind them belongs to the ring.
EXTERN_LIB void acl_stage_foreach_coefficients(const float* values, int count,
                                               aclTensor** tensors);
EXTERN_LIB void release_all_acl_foreach_coefficients() noexcept;

} // namespace jittor
