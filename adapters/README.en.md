# Downstream integrations

[中文](README.md)

Library-specific implementations, tests, tutorials and support records live in independent [JittorEcosystem](https://github.com/JittorEcosystem) repositories. This directory is an index, not an installable distribution.

| Repository / distribution | Python module |
|---|---|
| [jittor-transformers](https://github.com/JittorEcosystem/jittor-transformers) | `jittor_transformers` |
| [jittor-torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) | `jittor_torchmetrics` |
| [jittor-tensordict](https://github.com/JittorEcosystem/jittor-tensordict) | `jittor_tensordict` |
| [jittor-verl](https://github.com/JittorEcosystem/jittor-verl) | `jittor_verl` |
| [jittor-deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) | `jittor_deepspeed` |
| [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm) | `jittor_vllm` |

Follow each repository's quickstart to install it. Each project pins its upstream versions and registers through `jittor.module_patches`; install only the projects you need. Core and `jittor-compat` do not depend on an adapter bundle.

General computation belongs in Jittor, public Torch semantics in compat, and library-private behavior in the corresponding integration. Feature and hardware support is bounded by each repository's execution evidence.

Main-repository gates cover core and compat. Downstream host, upstream and device suites run in their own repositories. Explicit ecosystem comparisons may still use installed integrations: `ecosystem_deepspeed` and `ecosystem_vllm` require `JITTOR_DEEPSPEED_SOURCE` and `JITTOR_VLLM_SOURCE` respectively, pointing to the independent checkouts.
