# Agent 结果索引

本目录只保存人工整理、可长期阅读的验证与性能报告。原始日志、JSONL、模型、
图片、视频和编译缓存统一放在
`${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}`，不纳入主仓库。

## 兼容性与功能验证

- [2026-08-22 CUDA parallel-range 与常见网络独立 oracle 验证](2026-08-22-cuda-parallel-range-network-oracle.md)
- [2026-08-22 完整 CUDA 后端、dtype、设备对拍与 OpInfo 门禁](2026-08-22-cuda-test-suite.md)
- [2026-08-22 完整 CPU native/Torch 双会话验证](2026-08-22-complete-cpu-test-suite.md)
- [2026-08-21 分布采样与独立 PyTorch oracle 测试收口](2026-08-21-distribution-oracle-tests.md)
- [2026-08-21 原生与 Torch 类型系统测试隔离](2026-08-21-test-mode-type-system.md)
- [2026-08-21 JIT 浮点常量与小数 padding 修复](2026-08-21-jit-float-constants.md)
- [2026-08-21 Jupyter SIGCHLD 与并行编译复核](2026-08-21-jupyter-sigchld.md)
- [2026-08-21 CUDA rFFT 序列风险复核](2026-08-21-rfft-sequence-review.md)
- [2026-08-21 浮点 NaN 比较 CPU/CUDA 修复验证](2026-08-21-ieee-nan-comparisons.md)
- [2026-08-21 原生与 Torch 模式启动隔离修复验证](2026-08-21-native-torch-mode-isolation.md)
- [2026-08-21 median 原生与 Torch 模式 CPU/CUDA 修复验证](2026-08-21-median.md)
- [2026-08-21 负整数 floor division CPU/CUDA 修复验证](2026-08-21-floor-divide.md)
- [2026-08-21 MMCV/MMEngine CUDA typed tensor 导入兼容](2026-08-21-mmcv-cuda-typed-tensors.md)
- [2026-08-18 对拍框架、下游库覆盖与测试套件模式切分](2026-08-18-todo-parity-and-suite.md)
- [2026-08-12 Python 3.12 与旧接口兼容验证](2026-08-12-python312-native-compatibility.md)
- [2026-08-12 仓库结构现代化：交付验收报告（独立复核）](2026-08-12-repository-modernization-review.md)
- [2026-08-12 仓库结构现代化最终验收](2026-08-12-repository-modernization-final.md)
- [2026-08-11 仓库结构现代化阶段 6：杂物清理与发行边界](2026-08-11-repository-modernization-cleanup.md)
- [2026-08-11 仓库结构现代化阶段 5：测试外移与制品验收](2026-08-11-repository-modernization-test-migration.md)
- [2026-08-11 仓库结构现代化阶段 4：兼容层四层分离](2026-08-11-repository-modernization-compatibility-layers.md)
- [2026-08-11 仓库结构现代化阶段 3：领域包收敛](2026-08-11-repository-modernization-domain-packages.md)
- [2026-08-11 仓库结构现代化阶段 2：工具链、交付与性能基建](2026-08-11-repository-modernization-tooling.md)
- [2026-08-11 仓库结构现代化阶段 1：打包、部署与 wheel 基线](2026-08-11-repository-modernization-packaging.md)
- [2026-08-11 仓库结构现代化阶段 0：目标架构 RFC](2026-08-11-repository-modernization-rfc.md)
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
