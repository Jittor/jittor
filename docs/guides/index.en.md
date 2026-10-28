# Guides

Task-oriented guides: running on specific hardware, debugging problems, and
tuning memory and speed.

```{toctree}
:maxdepth: 1

platform-support.en
known-limitations.en
debugging.en
memory-optimization.en
performance-comparison.en
distributed-mpi.en
distributed-training.en
ascend-910b.en
ascend-310p.en
corex.en
cpp-console.en
```

| Guide | When to use it |
| --- | --- |
| [Platform and backend support](platform-support.en.md) | How far a combination (OS, Python, CUDA/CANN, torch version) is supported and where it was verified |
| [Known limitations](known-limitations.en.md) | Check whether the pitfall you are about to hit is already recorded |
| [Debugging](debugging.en.md) | Error localisation, abnormal gradients, compilation failures |
| [Memory optimization](memory-optimization.en.md) | Out of memory, or a larger batch is needed |
| [Performance comparison](performance-comparison.en.md) | Reproducible speed comparisons with other frameworks |
| [MPI distributed](distributed-mpi.en.md) | Multi-process, multi-device training |
| [Native multi-device training](distributed-training.en.md) | `jtrun`-launched native NCCL data parallelism (without compat.torch) |
| [Ascend 910B](ascend-910b.en.md) | Running on Huawei Ascend NPUs |
| [Ascend 310P](ascend-310p.en.md) | Running on Ascend inference cards (capability boundaries and the reduced-precision switch) |
| [Iluvatar Corex](corex.en.md) | Running on Iluvatar GPUs |
| [C++ console](cpp-console.en.md) | Calling the C++ interface directly and embedding the runtime |
