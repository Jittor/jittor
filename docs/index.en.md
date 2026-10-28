# Jittor documentation

[中文（默认）](index.md) | English

Jittor is a deep-learning framework built on just-in-time compilation and
meta-operators. The frontend exposes a Python dynamic-graph interface; the
backend compiles operators into C++ and CUDA code tuned for the actual
workload.

This site covers installation and first steps, tutorials, the API reference,
usage guides, Torch compatibility, framework internals, and contributor
documentation.

```{toctree}
:maxdepth: 2
:caption: Getting started

guides/index.en
tutorials/index.en
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index.en
compatibility/index.en
notes/index.en
performance/index.en
results/index.en
```

```{toctree}
:maxdepth: 2
:caption: Project

development/index.en
research/index.en
releases/index.en
community/index.en
contributing.en
```

## Where to start

| What you want to do | Where to go |
| --- | --- |
| Install and run a first model | [Install and quick start](https://github.com/Jittor/jittor#install), [tutorials](tutorials/index.en.md) |
| Look up a function | [API reference](api/index.en.md) |
| Run PyTorch code on Jittor | [Torch compatibility](compatibility/index.en.md) |
| Run on Ascend, Iluvatar, or multiple devices | [Guides](guides/index.en.md) |
| Understand memory, precision, or asynchronous execution | [How Jittor works](notes/index.en.md) |
| Investigate an error or performance problem | [Debugging guide](guides/debugging.en.md), [performance](performance/index.en.md) |
| See phenomena and conclusions others have already checked | [Verification results](results/index.en.md) |
| Check whether a system / Python / backend combination was verified | [Platform support](guides/platform-support.en.md) |
| Look up a known numerical, performance, or compatibility gap | [Known limitations](guides/known-limitations.en.md) |
| Upgrade from 1.x | [2.0 release notes](releases/2.0.en.md) |
| Contribute code to Jittor | [Contributing guide](contributing.en.md), [development documentation](development/index.en.md) |

## Project links

- [Website](https://cg.cs.tsinghua.edu.cn/jittor/)
- [Source repository](https://github.com/Jittor/jittor)
- [Issue tracker](https://github.com/Jittor/jittor/issues)
- [Forum](https://discuss.jittor.org/)
