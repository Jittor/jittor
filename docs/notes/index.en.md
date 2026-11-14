# How Jittor works

[Documentation](../index.en.md) · [中文](index.md)

These pages explain behavior you need to account for when writing or tuning a model. Start with the relevant contract; implementation details follow the usage and limitations.

## Devices and numerics

Read devices and placement before mixing backends. For low precision, distinguish the dtype of inputs from the precision used to accumulate results. The numerics contract collects special-value and cross-device rules.

- [Devices and placement](device-placement.en.md)
- [Mixed-precision training](../guides/mixed-precision.en.md)
- [float32 accumulation precision](numerics/float32-precision-policy.en.md)
- [Complex dtypes](numerics/complex-dtype.en.md)
- [Numerics contract](numerics/numerics-contract.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

device-placement.en
numerics/float32-precision-policy.en
numerics/complex-dtype.en
numerics/numerics-contract.en
```
-->

## Runtime behavior

Understand when work executes, inspect runtime state without changing it, and learn how CUDA library plans are cached. To measure time or memory, use the [performance workflow](../performance/index.en.md).

- [Pipelined lazy execution](runtime/pipelined-execution.en.md)
- [Runtime introspection](runtime/runtime-introspection.en.md)
- [CUDA library plan caches](runtime/cuda-library-plan-caches.en.md)
- [CUTLASS memory-efficient attention](runtime/efficient-attention.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

runtime/pipelined-execution.en
runtime/runtime-introspection.en
runtime/cuda-library-plan-caches.en
runtime/efficient-attention.en
```
-->
