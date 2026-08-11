# Agent 结果索引

本目录只保存人工整理、可长期阅读的验证与性能报告。原始日志、JSONL、模型、
图片、视频和编译缓存统一放在
`${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}`，不纳入主仓库。

## 兼容性与功能验证

- [2026-08-11 源码架构重构第十、十一批：misc 形状域与 FSDP2 兼容层](2026-08-11-source-architecture-misc-fsdp2-refactor.md)
- [2026-08-11 源码架构重构第九批：legacy pooling](2026-08-11-source-architecture-legacy-pool-refactor.md)
- [2026-08-11 源码架构重构第八批：pooling 覆盖层](2026-08-11-source-architecture-pooling-refactor.md)
- [2026-08-11 源码架构重构第七批：padding](2026-08-11-source-architecture-padding-refactor.md)
- [2026-08-11 源码架构重构第六批：convolution 类阶段](2026-08-11-source-architecture-convolution-layers-refactor.md)
- [2026-08-11 源码架构重构第五批：convolution 第一阶段](2026-08-11-source-architecture-convolution-refactor.md)
- [2026-08-11 源码架构重构第四批：RNN](2026-08-11-source-architecture-rnn-refactor.md)
- [2026-08-11 源码架构重构第三批：normalization](2026-08-11-source-architecture-normalization-refactor.md)
- [2026-08-11 源码架构重构第二批：nn.py](2026-08-11-source-architecture-nn-refactor.md)
- [2026-08-11 源码架构重构第一批](2026-08-11-source-architecture-refactor.md)
- [2026-08-10 仓库工作区整理](2026-08-10-repository-cleanup.md)
- [2026-07-16 CUDA 12 component wheel 验证](2026-07-16-cuda-pip-wheels.md)
- [2026-07-05 复数 CUDA 审计](2026-07-05-complex-cuda-audit.md)
- [2026-07-05 Diffusers 视频生成（Jittor）](2026-07-05-diffusers-video-jittor.md)
- [2026-07-05 Diffusers 视频生成（PyTorch 基线）](2026-07-05-diffusers-video-torch-baseline.md)
- [2026-07-12 `test_example` 验证](2026-07-12-test-example-validation.md)
- [2026-07-12 TorchQuantum README 验证](2026-07-12-torchquantum-validation.md)

## 性能

- [2026-07-07 Transformers 初始性能分析](2026-07-07-transformers-performance.md)
- [2026-07-10 Transformers 性能深挖总览](2026-07-10-transformers-performance-deep-dive.md)

Transformers 专项报告：

- [clip_grad_norm_](transformers/2026-07-11-clip-grad-norm.md)
- [Transformer block](transformers/2026-07-11-e2e-models.md)
- [HF tiny models](transformers/2026-07-11-hf-tiny-models.md)
- [LayerNorm](transformers/2026-07-11-layernorm.md)
- [optimizer step](transformers/2026-07-11-optimizer-step.md)
- [SDPA 与同步](transformers/2026-07-11-sdpa-sync.md)
- [softmax 调度边界](transformers/2026-07-11-softmax-boundaries.md)
- [softmax/GELU](transformers/2026-07-11-softmax-gelu.md)

新增报告使用 `YYYY-MM-DD-topic.md`；专题较多时在主题子目录中保持同一命名规则。
