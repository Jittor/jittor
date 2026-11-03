# Jittor documentation

[中文](index.md)

Jittor is a deep-learning framework built on just-in-time compilation. Use the
native Jittor API, or run supported PyTorch code through `jittor-compat`.

## Get started

If you are new to Jittor, follow these steps:

1. [Installation and environment](getting-started/installation.en.md): choose an installation method and check your interpreter and compiler.
2. [Your first training example](getting-started/quickstart.en.md): run a forward pass, gradients and parameter updates on CPU.
3. [Continue learning](getting-started/index.en.md): choose native Jittor or Torch compatibility tutorials.

For an existing PyTorch project, start with [Torch compatibility](compatibility/index.en.md).
For libraries such as vLLM and Transformers, also check the versions and devices
supported by the corresponding [JittorEcosystem project](https://github.com/JittorEcosystem).

## Find a task

| What you want to do | Start here |
| --- | --- |
| Configure hardware, run distributed training or diagnose a problem | [User guides](guides/index.en.md) |
| Look up functions, classes and parameters | [API reference](api/index.en.md) |
| Understand devices, precision and lazy execution | [Core concepts](notes/index.en.md) |
| Profile time and memory or compare performance | [Performance](performance/index.en.md) |
| Change the framework, add features or run tests | [Developer guide](development/index.en.md) |
| Read migration notes, find community projects or explore research | [Project information](project/index.en.md) |

## Check before you run

[Platform support](guides/hardware/platform-support.en.md) distinguishes continuous tests
from historical validation. [Known limitations](guides/known-limitations.en.md) explains
current usage boundaries. Historical measurements apply only to their recorded commit
and environment; they do not establish acceptance of the current revision.

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 2
:caption: Use Jittor

getting-started/index.en
guides/index.en
compatibility/index.en
performance/index.en
```
-->

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference and development

api/index.en
notes/index.en
development/index.en
project/index.en
```
-->

[Source](https://github.com/Jittor/jittor/tree/2.0-refactor) ·
[Report an issue](https://github.com/Jittor/jittor/issues) ·
[Forum](https://discuss.jittor.org/)
