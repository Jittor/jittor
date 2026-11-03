# 下游集成

[English](README.en.md)

下游专属实现、配套测试、教程和支持范围在 [JittorEcosystem](https://github.com/JittorEcosystem) 的独立仓库维护。本目录只提供导航，不是可安装的发行包。

| 仓库 | Python 模块 / 入口 |
|---|---|
| [jittor-transformers](https://github.com/JittorEcosystem/jittor-transformers) | `jittor_transformers` |
| [jittor-datasets](https://github.com/JittorEcosystem/jittor-datasets) | `jittor_datasets` |
| [jittor-tokenizers](https://github.com/JittorEcosystem/jittor-tokenizers) | `jittor_tokenizers` |
| [jittor-safetensors](https://github.com/JittorEcosystem/jittor-safetensors) | `jittor_safetensors` |
| [jittor-accelerate](https://github.com/JittorEcosystem/jittor-accelerate) | 官方 API 与验证工具 |
| [jittor-ray](https://github.com/JittorEcosystem/jittor-ray) | 官方 API 与验证工具 |
| [jittor-torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) | `jittor_torchmetrics` |
| [jittor-tensordict](https://github.com/JittorEcosystem/jittor-tensordict) | `jittor_tensordict` |
| [jittor-verl](https://github.com/JittorEcosystem/jittor-verl) | `jittor_verl` |
| [jittor-deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) | `jittor_deepspeed` |
| [jittor-trl](https://github.com/JittorEcosystem/jittor-trl) | `jittor_trl` |
| [jittor-ms-swift](https://github.com/JittorEcosystem/jittor-ms-swift) | `jittor_ms_swift` |
| [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm) | `jittor_vllm` |

在相应仓库中按其快速开始文档安装。每个项目独立固定上游版本；需要运行时补丁的集成使用 `jittor.module_patches` 注册，验证型仓库直接运行官方 API 与配套测试。仅安装所需项目即可。

通用计算问题修回 Jittor，Torch 公共语义问题修回 compat，第三方库私有行为留在对应集成仓库。设备与功能支持以各仓的实际验收记录为准。

主仓只运行核心和兼容层门禁；下游 host、官方测试和设备验收在各仓执行。主仓的显式生态对拍任务仍可使用独立安装包，例如 `ecosystem_deepspeed` 与 `ecosystem_vllm` 分别要求 `JITTOR_DEEPSPEED_SOURCE` 与 `JITTOR_VLLM_SOURCE` 指向对应仓库 checkout。
