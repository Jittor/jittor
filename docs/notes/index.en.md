# How Jittor works

This section explains how Jittor runs -- the places where behaviour surprises
you unless it is written down: which device a tensor really lives on, when
low-precision accumulation applies, why two devices produce different numbers,
and why lazy execution leaves the GPU idle.

It is written for people who can already run a model and now care whether the
results are correct and the speed is reasonable. For how the code is organised,
how errors are categorised, and how operator sources are written, see the
[development documentation](../development/index.en.md).

```{toctree}
:maxdepth: 1

device-placement.en
numerics-contract.en
float32-precision-policy.en
cuda-library-plan-caches.en
runtime-introspection.en
mixed-precision.en
complex-dtype.en
pipelined-execution.en
profiling.en
native-sharded-checkpoint.en
native-zero-optimizer.en
native-tensor-parallel.en
native-fsdp.en
```

## Index by question

| Symptom | Read |
| --- | --- |
| Which device is a tensor on? How do `.cpu()` / `.cuda()` work? | [Devices and placement](device-placement.en.md) |
| CPU and CUDA disagree; NaN, infinity, subnormals, reduction precision | [Numerics contract](numerics-contract.en.md) |
| TF32, fp32 matmul precision, `float32_matmul_precision` | [float32 accumulation precision](float32-precision-policy.en.md) |
| Who owns cuFFT/cuTT/cuDNN plan caches, when are they released, how to observe them | [CUDA library plan caches](cuda-library-plan-caches.en.md) |
| How tests ask "is CUDA available", "what is the current policy", "how many Vars leaked" | [Runtime introspection](runtime-introspection.en.md) |
| Enabling fp16 / bf16 training, loss scaling, `GradScaler` | [Mixed-precision training](mixed-precision.en.md) |
| How far complex-number support goes | [Complex dtypes](complex-dtype.en.md) |
| Why GPU utilisation stays low, and what `auto_flush_ops` is | [Pipelined lazy execution](pipelined-execution.en.md) |
| Where a step's time and memory go; host-bound or device-bound | [Performance and memory profiling](profiling.en.md) |
