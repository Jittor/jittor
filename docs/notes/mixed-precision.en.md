# Mixed-precision training

Training with float16 or bfloat16 trades numerical range for memory and speed. This
page explains how to enable each of Jittor's two modes, when loss scaling is
needed, and where accumulation still happens in float32.

Mixed precision involves two independent things; understanding them separately
avoids the traps:

1. **Which operators drop to half precision** — decided by the executor tier
   (`jt.flags.auto_mixed_precision_level`; in torch compatibility mode,
   `torch.autocast`).
2. **How gradients avoid being flushed away** — decided by loss scaling
   (`jt.amp.GradScaler`; in torch compatibility mode, `torch.amp.GradScaler`).

Only float16 needs the second. bfloat16's exponent is as wide as float32's, so
gradients do not underflow.

## Why float16 needs loss scaling

The smallest normal number in float16 is 6e-5. A gradient that is unremarkable in
float32 — say 1e-7 — is 0 in float16. Backward propagation thus silently drops
every small gradient, the model stops learning, and nothing raises.

Multiplying the loss by a large constant before backward lifts the whole gradient
distribution into float16's range; dividing the gradients back afterwards leaves
the parameter update completely unchanged. That is all loss scaling is.

The constant cannot be hard-coded: too large and it overflows to inf, too small
and it still underflows, and which side is right depends on the model and how far
training has progressed. So it is chosen dynamically — start high, halve it and
**skip the step** whenever a non-finite gradient appears (those gradients are
already garbage), then double it again after several clean steps in a row.

## Native mode

```python
import jittor as jt

model = MyModel()
for p in model.parameters():
    p.assign(p.float16())

opt = jt.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scaler = jt.amp.GradScaler()          # bfloat16 训练时传 enabled=False

for x, y in data:
    loss = loss_fn(model(x), y)
    opt.backward(scaler.scale(loss.float32()))   # 注意 .float32()，见下
    scaler.step(opt)                  # 先 unscale，梯度非有限就跳过这一步
    scaler.update()                   # 增长或回退 scale
```

**`scale()` must be applied to a float32 loss.** Multiplication keeps the input's
dtype, the default scale is 65536, and float16's maximum is 65504 — so a direct
`scaler.scale(loss)` overflows to inf before backward even begins. Every
subsequent step then sees non-finite gradients, skips, and halves the scale until
it reaches 1, with no training happening from start to finish and no error
reported. torch behaves the same way (its `_scale` is a float32 tensor; a
half-precision tensor times a float32 tensor promotes, while times a python float
does not), but torch users rarely hit it: autocast keeps the loss function in
float32. The native style of converting the whole model to float16 does hit it.

`loss.float32()` only affects the scaling step; backward is still float16 from the
first operator onward. When the scale is large enough to necessarily overflow
float16, `scale()` emits one `RuntimeWarning` saying so.

With `enabled=False` every method is an identity operation, so the same training
loop can run at any precision without branches.

If you need to insert something between unscale and step (gradient clipping is the
most common reason — it must see gradients at their real magnitude), call
`scaler.unscale_(opt)` yourself and `step` knows not to do it again.

| Method | Effect |
| --- | --- |
| `scale(loss)` | multiplies by the current scale; also accepts a list/tuple/iterable of Vars |
| `unscale_(opt)` | divides that optimizer's gradients back in place and records whether non-finite values appeared |
| `step(opt)` | unscales first if needed; skips and returns `None` if gradients are non-finite |
| `update()` | multiplies by `backoff_factor` (not below 1) on overflow, otherwise by `growth_factor` after `growth_interval` consecutive good steps |
| `state_dict()` / `load_state_dict()` | five keys, with the same key names as torch, so checkpoints are mutually readable |

The tier (which operators drop precision) is a switch on the executor, scoped with
`jt.flag_scope`:

```python
with jt.flag_scope(auto_mixed_precision_level=5):
    ...
```

Tiers 4/5/6 keep parameters in float32 while dropping gradients and intermediates
to float16; 1–3 basically do not reduce precision. The tier only changes dtype
inference and does no loss scaling — the two must be enabled separately.

## torch compatibility mode

`torch.autocast` and `torch.amp.GradScaler` are both present, with signatures
matching torch:

```python
import torch

scaler = torch.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

`torch.amp.GradScaler` is `jt.amp.GradScaler` with torch's argument order — one
algorithm, two entry points, so a change to the scaling policy never lands on only
one side. `torch.cuda.amp.GradScaler` and `torch.cpu.amp.GradScaler` are its
subclasses, keeping the pre-torch-2.3 old order with `init_scale` first.

`is_autocast_enabled()` with no argument asks about **cuda**, consistent with
torch: inside `with torch.autocast("cpu")` it returns `False`; pass `"cpu"` to ask
about cpu.

One known difference: `torch.autocast(dtype=torch.bfloat16)` computes as float16
in regions that are entirely float32, because jittor's amp register keeps
bfloat16 only when a bfloat16 operand already exists. To stay in bfloat16, convert
the module itself with `.to(torch.bfloat16)`. The runtime emits one
`RuntimeWarning` saying so.

## Where accumulation still happens in float32

Reducing precision only lowers **storage and multiplication**; accumulation is
always wider — this matches torch's rules and is the precondition for half
precision being usable:

- Matrix multiplication and convolution: cuBLAS/cuDNN use `CUBLAS_COMPUTE_32F`,
  and the generic `(a*b).sum(k)` fallback path likewise uses float32 intermediates.
- `sum` / `mean` / `std` / `norm` / `softmax` / `log_softmax` / `cumsum`: float32
  accumulators.
- LayerNorm / GroupNorm / InstanceNorm: statistics (mean, variance, rsqrt) are
  computed in float32.
- A reduction's **output** dtype keeps the input's half precision and does not
  quietly become float32 — `x.sum(-1)`, `x.max(-1)` and `x.prod(-1)` all return
  float16 for float16 inputs.

CPU and CUDA give the same answer. Measured (error relative to a float64
closed-form solution on the same group of already-rounded inputs; alongside are
the numbers from real torch 2.13 on the same inputs):

| Operator | dtype | Device | Jittor | torch 2.13 |
| --- | --- | --- | --- | --- |
| `matmul` (K=512) | fp16 | cpu | 7.759e-3 | 7.759e-3 |
| `matmul` | bf16 | cpu | 6.175e-2 | 6.175e-2 |
| `layer_norm` | fp16 | cpu/cuda | 9.750e-4 | 9.750e-4 |
| `sum` / `mean` (4096) | bf16 | cpu | 1.780e-3 | 1.780e-3 |
| `softmax` | bf16 | cpu | 8.677e-4 | 8.677e-4 |

`max` / `min` use the same expression as float32 in half precision, so NaN
propagation rules are identical in both operand positions, on both devices, at all
three floating widths — `jt.maximum(5, nan)` is NaN, and `x.max()` on an array
containing NaN is NaN.

## End-to-end expectations

Same model, same initial values, same schedule, final loss:

| Precision | Final loss |
| --- | --- |
| float32 | 0.13503 |
| bfloat16 | 0.13867 |
| float16 + GradScaler | 0.13501 |
| `auto_mixed_precision_level` 4/5/6 | 0.13318–0.13403 |

Real torch 2.13 under the same conditions gives distributions of 0.4% and 3.4% for
float32 / bf16 autocast / fp16 autocast+GradScaler. Half-precision training
converging to roughly the same place as float32 is normal; a difference of an
order of magnitude means something is accumulating in half precision, which is a
bug, not the price of precision.

## Related

- [Numerics contract](numerics-contract.en.md): the general rules for NaN,
  infinity, subnormals and reduction precision.
- [float32 accumulation precision](float32-precision-policy.en.md): float32
  operands' own tf32/bf16 tiers.
- [Devices and placement](device-placement.en.md): `.cuda()` / `.cpu()` and which
  device optimizer state lives on.
