# Torch 兼容

Jittor 提供一层 Torch 兼容前端：把 `import torch` 指向 Jittor，让按 PyTorch 写的
代码在 Jittor 上运行。

```{toctree}
:maxdepth: 1

torch
torch-shim
```

| 文档 | 内容 |
| --- | --- |
| [Torch 兼容 API](torch.md) | 支持的接口范围、语义差异与不支持的部分 |
| [Torch shim](torch-shim.md) | shim 的部署方式、生效范围与排查方法 |

兼容层只负责**拼写与签名的适配**；能力本身由 Jittor 框架提供。如果某个能力
Jittor 没有，兼容层会明确报错，而不是静默给出可疑结果。
