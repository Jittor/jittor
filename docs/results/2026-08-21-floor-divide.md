# 负整数 floor division CPU/CUDA 修复验证

- Status: CPU/CUDA verified; NPU/ROCm real-device verification pending
- Date: 2026-08-21
- Baseline: `85aad9f3`
- Owner: binary operator maintainers
- Review when: the common operator code generator, integer dtype inference, or
  an unverified accelerator backend becomes available

## Finding and correction

Native `jt.floor_divide` generated plain C++ integer division. C++ truncates
toward zero, so both CPU and CUDA returned `-1` for `-5 // 3`, while Python,
NumPy, and Torch return `-2`. The existing OpInfo samples used only non-negative
dividends and therefore could not signal the known defect.

The common operator code generator now calls a shared header-only helper. It
computes the truncating quotient and remainder, then subtracts one only when the
remainder is nonzero and its sign differs from the divisor. This keeps exact
integer arithmetic, avoids a float conversion that would lose int64 precision,
and compiles into the generated CPU/CUDA kernel rather than falling back to the
host. OpInfo now includes negative dividends and negative divisors and uses
`numpy.floor_divide` as its independent oracle.

## Verification

Environment and cache isolation match the accompanying
[MMCV report](2026-08-21-mmcv-cuda-typed-tensors.md): Python 3.11.15, CUDA 12.2,
NVIDIA driver 595.84, and an RTX 4090, with separate CPU/CUDA run roots and
serialized first JIT compilation.

Before the fix, the fixed vector produced six wrong values on both CPU and
CUDA:

```text
got [-2, -2, -1, 0, 0, 0, -1, -2, -2]
ref [-3, -2, -2, -1, 0, -1, -2, -2, -3]
```

Results after the fix:

| Gate | Result |
| --- | --- |
| Fixed CPU regression: signed widths, uint8, operator spelling, broadcast | `2 passed` |
| Fixed real-device CUDA regression | `2 passed` |
| CPU floor/pow OpInfo and type-system selection | `6 passed, 2 skipped` |
| CUDA floor/pow OpInfo selection | `4 passed` |
| CPU type-system and Torch-promotion expansion | `55 passed, 2 skipped` |
| CUDA Torch-promotion expansion | `25 passed` |
| `JIT_cuda + IS_ACL` helper syntax compile | passed |
| `tests/structure` | `210 passed, 2 skipped` |
| `bash agent/scripts/check_repo_layout.sh` | passed |

Generated CUDA source contains both
`#include "type/floor_divide_compute.h"` and a device-kernel call to
`jittor::_floor_divide`; the result therefore does not rely on CPU fallback.

No real NPU or ROCm device was available. The known-issues entry remains active
until the same fixed vectors and OpInfo samples pass on both backends.
