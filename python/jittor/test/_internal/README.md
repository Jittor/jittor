# jittor test infrastructure (`jittor.test._internal`)

A modern, PyTorch-style test system for jittor. It exists to **measure real bugs** —
especially the *silent-wrong backward* bugs that forward-only, jittor-vs-jittor tests
miss. It is a faithful adaptation of `torch.testing._internal` (OpInfo + gradcheck +
device-type parametrization); see `agent/skills/jittor-dev-context/design-test-system.md`
for the full rationale and the audit that motivated it.

## Pieces

| module | role | torch analogue |
|---|---|---|
| `common_utils.py` | `JittorTestCase` (`assertEqual`, tolerance policy), `make_tensor`, dtype groups, `parametrize` | `common_utils.py` |
| `common_device_type.py` | `instantiate_device_type_tests`, `@ops`/`@dtypes`/`@onlyCPU`/`@onlyCUDA`/`@onlyNPU`/`@skipCUDAIf` | `common_device_type.py` |
| `gradcheck.py` | `gradcheck` / `gradgradcheck` / `numerical_vjp` — the **backward oracle** (numerical vs analytical Jacobian, float64) | `torch.autograd.gradcheck` |
| `opinfo/core.py` | `OpInfo`, `SampleInput`, `UnaryUfuncInfo`, `BinaryUfuncInfo`, `ReductionOpInfo`, `DecorateInfo`/`skip`/`xfail` | `opinfo/core.py` |
| `opinfo/definitions/*.py` | per-domain `OpInfo` lists (auto-discovered); `_refs.py` shared numpy refs | `opinfo/definitions/` |
| `common_methods_invocations.py` | `op_db` — aggregate of all definitions | same |
| `report.py` | coverage matrix (op × {fwd, bwd, gradgrad}) + SKIP reporting | — |

The generic drivers live one level up in `../test_ops.py` (`TestCommon` forward-vs-ref,
`TestGradients` gradcheck), generated over `op_db × device × dtype`.

## Why a green test means something

- **Forward** is compared to an INDEPENDENT numpy `ref` (not jittor-vs-jittor).
- **Backward** is `gradcheck`: numerical (finite-diff) vs analytical (`jt.grad`) Jacobian,
  in float64 on CPU. Forward already pinned to numpy ⇒ a correct gradcheck *transitively
  proves the backward matches torch*, no live-torch needed.
- **float32 stability** (norm small-variance cancellation) is a *separate* test
  (`../test_norm.py`): gradcheck runs in float64 where the cancellation doesn't occur.

## Add an operator (the whole point — it's a few lines)

Drop an entry in the right `opinfo/definitions/*.py` (auto-discovered, no central edit):

```python
from ._refs import *                       # make_tensor, SampleInput, np, jt, nn, F
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo

op_db = [
    UnaryUfuncInfo("tanh", ref=np.tanh, op=jt.tanh),              # elementwise: fwd+bwd auto
    OpInfo("matmul", op=jt.matmul, ref=np.matmul,                 # custom samples
           sample_inputs_func=my_sample_fn),
    OpInfo("argmax", op=jt.argmax, ref=np.argmax,
           supports_autograd=False),                             # non-differentiable
]
```

Rules of thumb:
- A `SampleInput`'s `input` and any **float-Var positional args** are differentiated by
  gradcheck; pass int64 indices / python scalars as `kwargs` so they stay fixed.
- `supports_autograd=False` for integer/bool/argmax ops; `supports_gradgrad=False` for ops
  whose backward is a non-differentiable `jt.Function` (norms) or piecewise (activations).
- Keep differentiated tensors small (≲24 elems): gradcheck is O(numel) forward passes.
- jittor narrows `jt.array(float64)`→float32 and `int64`→int32 — pin dtype with
  `jt.array(a, dtype=...)` (`make_tensor` already does).

## Run

```bash
JT=$(which python)   # use the jittor env's python; PYTHONPATH=<repo>/python
$JT -m jittor.test.test_ops -v                 # op battery (fwd all devices, grad CPU)
$JT -m jittor.test.test_ops -k layer_norm      # one op
$JT -m jittor.test.test_norm                   # norm float32 backward stability
$JT -m jittor.test.test_regression             # §4 silent-wrong locks
$JT -m jittor.test.test_kernel_traps           # §4-B inherent-behavior traps
$JT -m jittor.test._internal.report            # coverage matrix
JITTOR_TEST_DEVICES=cpu ...                     # restrict device matrix
cache_name=cardN CUDA_VISIBLE_DEVICES=N ...     # isolate parallel runs (no cache contention)
```
