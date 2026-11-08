# Use Torch compatibility

[Documentation](../index.en.md) · [中文](index.md)

For existing PyTorch code, `jittor-compat` provides Torch-style tensors, models and optimizers backed by Jittor. Check the APIs you need, then select downstream integrations by their pinned versions and device support records.

- First attempt: [install the compatibility package](../getting-started/installation.en.md), then run the [CPU training example](torch.en.md).
- Make an application start with `import torch`: [installation and deployment](torch-shim.en.md).
- Use vLLM, Transformers or another library: consult its [JittorEcosystem](https://github.com/JittorEcosystem) repository for versions, tutorials and support scope.

These interfaces are not an unconditional replacement for every PyTorch version, model and device. Importing a library does not establish that its training, distributed or accelerator paths have been validated.

<!-- sphinx-only
```{toctree}
:hidden:

torch.en
torch-shim.en
```
-->

## Change the compatibility implementation

Maintainers should read the [compatibility principles](../development/compatibility/principles.en.md), then the [architecture](../development/compatibility/torch-architecture.en.md) and [call lowering](../development/compatibility/torch-lowering.en.md). These explain ownership and acceptance requirements; they are not prerequisites for running an application.
