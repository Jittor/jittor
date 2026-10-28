# Torch compatibility

Jittor provides a Torch-compatible frontend: point `import torch` at Jittor and
code written for PyTorch runs on Jittor.

```{toctree}
:maxdepth: 1

torch.en
torch-shim.en
torch-architecture.en
torch-lowering.en
principles.en
```

| Document | Contents |
| --- | --- |
| [Torch-compatible API](torch.en.md) | Supported interface range, semantic differences, and unsupported parts |
| [Torch shim](torch-shim.en.md) | How the shim is deployed, what it affects, and how to diagnose it |
| [Layered architecture](torch-architecture.en.md) | Unified computation graph, framework lowering, and operator optimization |
| [Call lowering](torch-lowering.en.md) | How PyTorch calls become IR nodes, and the new API surface |
| [Compatibility principles](principles.en.md) | Acceptance gates: ownership, activation boundary, fallback, and evidence |

The compatibility layer only adapts **spelling and signatures**; the capability
itself comes from Jittor. When Jittor does not provide a capability, the layer
reports a clear error instead of silently returning a suspicious result.
