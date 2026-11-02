# 下游集成

[English](README.en.md)

下游专属实现、配套测试、教程和支持范围在 [JittorEcosystem](https://github.com/JittorEcosystem) 的独立仓库维护。本目录只提供导航，不是可安装的发行包。

| 仓库 / 安装包 | Python 模块 |
|---|---|
| [jittor-transformers](https://github.com/JittorEcosystem/jittor-transformers) | `jittor_transformers` |
| [jittor-torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) | `jittor_torchmetrics` |
| [jittor-tensordict](https://github.com/JittorEcosystem/jittor-tensordict) | `jittor_tensordict` |
| [jittor-verl](https://github.com/JittorEcosystem/jittor-verl) | `jittor_verl` |
| [jittor-deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) | `jittor_deepspeed` |
| [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm) | `jittor_vllm` |

在相应仓库中按其快速开始文档安装。每个项目独立固定上游版本，使用 `jittor.module_patches` 注册；仅安装所需项目即可。核心和 `jittor-compat` 不依赖 adapter 合集。

通用计算问题修回 Jittor，Torch 公共语义问题修回 compat，第三方库私有行为留在对应集成仓库。设备与功能支持以各仓的实际验收记录为准。

主仓只运行核心和兼容层门禁；下游 host、官方测试和设备验收在各仓执行。主仓的显式生态对拍任务仍可使用独立安装包，例如 `ecosystem_deepspeed` 与 `ecosystem_vllm` 分别要求 `JITTOR_DEEPSPEED_SOURCE` 与 `JITTOR_VLLM_SOURCE` 指向对应仓库 checkout。
