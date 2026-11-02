# Downstream integrations

[中文](README.md)

Library-specific implementations, tests, tutorials and support records live in independent [JittorEcosystem](https://github.com/JittorEcosystem) repositories. This directory is an index, not an installable distribution.

| Repository | Python module / entry |
|---|---|
| [jittor-transformers](https://github.com/JittorEcosystem/jittor-transformers) | `jittor_transformers` |
| [jittor-datasets](https://github.com/JittorEcosystem/jittor-datasets) | `jittor_datasets` |
| [jittor-tokenizers](https://github.com/JittorEcosystem/jittor-tokenizers) | `jittor_tokenizers` |
| [jittor-safetensors](https://github.com/JittorEcosystem/jittor-safetensors) | `jittor_safetensors` |
| [jittor-accelerate](https://github.com/JittorEcosystem/jittor-accelerate) | Official APIs and validation tools |
| [jittor-ray](https://github.com/JittorEcosystem/jittor-ray) | Official APIs and validation tools |
| [jittor-torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) | `jittor_torchmetrics` |
| [jittor-tensordict](https://github.com/JittorEcosystem/jittor-tensordict) | `jittor_tensordict` |
| [jittor-verl](https://github.com/JittorEcosystem/jittor-verl) | `jittor_verl` |
| [jittor-deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) | `jittor_deepspeed` |
| [jittor-trl](https://github.com/JittorEcosystem/jittor-trl) | `jittor_trl` |
| [jittor-ms-swift](https://github.com/JittorEcosystem/jittor-ms-swift) | `jittor_ms_swift` |
| [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm) | `jittor_vllm` |

Follow each repository's quickstart to install it. Each project pins its upstream versions. Integrations needing runtime patches register through `jittor.module_patches`; validation repositories run official APIs and their accompanying tests. Install only the projects you need.

General computation belongs in Jittor, public Torch semantics in compat, and library-private behavior in the corresponding integration. Feature and hardware support is bounded by each repository's execution evidence.

Main-repository gates cover core and compat. Downstream host, upstream and device suites run in their own repositories. Explicit ecosystem comparisons may still use installed integrations: `ecosystem_deepspeed` and `ecosystem_vllm` require `JITTOR_DEEPSPEED_SOURCE` and `JITTOR_VLLM_SOURCE` respectively, pointing to the independent checkouts.
