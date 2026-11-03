# Guides

[Documentation](../index.en.md) · [中文](index.md)

Choose a guide for the task you want to complete. Check platform coverage before running on a new device; a successful installation alone does not establish numerical or training support.

## Hardware and prerequisites

Start with platform support and known limitations, then configure your device. The Corex page covers detection only; it does not claim hardware validation.

- [Platform Support](hardware/platform-support.en.md)
- [Known Limitations](known-limitations.en.md)
- [Ascend 910B: Configuration and Verification](hardware/ascend-910b.en.md)
- [Ascend 310P: Configuration, Capability Boundaries, and Verification](hardware/ascend-310p.en.md)
- [Iluvatar Corex: Discovery and Follow-up Verification](hardware/corex.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

hardware/platform-support.en
known-limitations.en
hardware/ascend-910b.en
hardware/ascend-310p.en
hardware/corex.en
```
-->

## Distributed training

New native scripts start with single-machine training through `jtrun`. The MPI guide is for existing MPI programs; model and optimizer sharding are explained in the mechanism notes.

- [Native Single-Machine Multi-GPU Training](distributed/distributed-training.en.md)
- [Distributed Training with MPI](distributed/distributed-mpi.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

distributed/distributed-training.en
distributed/distributed-mpi.en
```
-->

## Debugging and integration

Use debugging for failures and unexpected values. The C++ console guide covers embedding the Python runtime in a C++ application.

- [Debugging Jittor Programs](debugging.en.md)
- [C++ Console Interface](cpp-console.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

debugging.en
cpp-console.en
```
-->

## Measure and optimize

Follow the [performance workflow](../performance/index.en.md): measure a baseline, profile the bottleneck, then adjust memory or execution settings.

## Distributed state

Choose the state to shard: optimizer state with ZeRO, linear-layer features with tensor parallelism, parameters with FSDP, or files with sharded checkpoints. These capabilities have separate backend and composition limits; file sharding alone is not training sharding.

- [Native ZeRO stage-one optimizer](distributed/native-zero-optimizer.en.md)
- [Native tensor-parallel linear layers](distributed/native-tensor-parallel.en.md)
- [Native FSDP storage, backward and checkpoint ownership](distributed/native-fsdp.en.md)
- [Native sharded checkpoint files](distributed/native-sharded-checkpoint.en.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

distributed/native-zero-optimizer.en
distributed/native-tensor-parallel.en
distributed/native-fsdp.en
distributed/native-sharded-checkpoint.en
```
-->

## Mixed-precision training

Start with [mixed precision](mixed-precision.en.md) for autocast and gradient scaling, then check the device and operator limits.

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

mixed-precision.en
```
-->