# Jittor 技能目录

这里只列可复用的调试、验证和维护技能，按当前任务选用。
仓库规则见根 [AGENTS.md](../AGENTS.md)，用户与开发文档见 [docs](../docs/index.md)。
任务、问题状态和运行结果在 GitHub issue/PR 维护。

## 验证与仓库工具

- [git-worktree-shared-state](skills/git-worktree-shared-state/SKILL.md)：多个 worktree 之间共用与不共用的 git 状态。
- [jittor-worktree-verification](skills/jittor-worktree-verification/SKILL.md)：确认跑的是本 worktree 的代码。
- [verifying-a-gate-actually-ran](skills/verifying-a-gate-actually-ran/SKILL.md)：相信一次红或绿之前先证明门禁真的跑了。
- [gate-tier-budget](skills/gate-tier-budget/SKILL.md)：门禁分层与按算术校验的耗时预算。
- [codebase-wide-fix-as-rule](skills/codebase-wide-fix-as-rule/SKILL.md)：把全树修复写成结构规则并用反例证明它有牙。
- [pure-code-motion-refactor](skills/pure-code-motion-refactor/SKILL.md)：证明拆分或搬移代码行为逐字不变。

## 核心 C++、构建与绑定

- [jittor-core-cpp-edit-loop](skills/jittor-core-cpp-edit-loop/SKILL.md)：改 `src/` 的编辑、重编、验证循环。
- [jittor-build-change-verification](skills/jittor-build-change-verification/SKILL.md)：构建系统改动的冷/热缓存、并发与切 flag 验证。
- [jittor-build-time-capabilities](skills/jittor-build-time-capabilities/SKILL.md)：同一源码为何编出能力不同的构建。
- [jittor-pyjt-bindings](skills/jittor-pyjt-bindings/SKILL.md)：pyjt 绑定生成器与转换层的验证。
- [jittor-single-kernel-cold-profile](skills/jittor-single-kernel-cold-profile/SKILL.md)：单个生成 kernel 的冷编译分阶段计时。
- [jit-compile-failure-attribution](skills/jit-compile-failure-attribution/SKILL.md)：JIT 编译失败是否归因到闯祸的算子。
- [core-invariant-property-tests](skills/core-invariant-property-tests/SKILL.md)：给图、liveness 与执行器写属性测试。
- [graph-inference-to-explicit-record](skills/graph-inference-to-explicit-record/SKILL.md)：把图反推改成在产生处显式记录。
- [process-global-state-and-optin](skills/process-global-state-and-optin/SKILL.md)：找出全进程副作用并改成可撤销的 opt-in。
- [hang-that-holds-the-gil](skills/hang-that-holds-the-gil/SKILL.md)：定位持有 GIL 的挂死并改成有界等待。

## 正确性对拍与语义

- [jittor-op-parity-oracle](skills/jittor-op-parity-oracle/SKILL.md)：Python 层算子的数值对拍基准与容差。
- [jittor-grad-silence-probe](skills/jittor-grad-silence-probe/SKILL.md)：判断梯度是否被静默吞掉、查询反向图形状。
- [optimizer-semantics-diffing](skills/optimizer-semantics-diffing/SKILL.md)：优化器语义（累积、裁剪、偏差修正）的对拍。
- [native-api-sweep](skills/native-api-sweep/SKILL.md)：原生公开接口面的存续、数值与显存扫描。
- [jittor-allocator-flag-matrix](skills/jittor-allocator-flag-matrix/SKILL.md)：分配器缺陷的 flag 矩阵与竞争复现。
- [multi-device-verification](skills/multi-device-verification/SKILL.md)：多卡上证明张量真的在声称的设备上。
- [jittor-distributed-verification](skills/jittor-distributed-verification/SKILL.md)：单机多进程验证 MPI/NCCL 与 rendezvous。
- [torch-checkpoint-stride-oracle](skills/torch-checkpoint-stride-oracle/SKILL.md)：PyTorch checkpoint 的 storage/stride 读取对拍。

## 性能

- [jittor-performance-profiling](skills/jittor-performance-profiling/SKILL.md)：选择核心与 compat 性能探针；工具在 `benchmarks/` 维护。

- [jittor-core-planning-cost](skills/jittor-core-planning-cost/SKILL.md)：核心记账改动的规划开销定价。
- [jittor-graph-construction-cost](skills/jittor-graph-construction-cost/SKILL.md)：Python 前端建图的主机开销。
- [codegen-optimization-in-effect](skills/codegen-optimization-in-effect/SKILL.md)：证明指令级 codegen 优化真的生成且有收益。
- [cuda-elementwise-bandwidth-roofline](skills/cuda-elementwise-bandwidth-roofline/SKILL.md)：逐元素 kernel 的带宽屋顶线与同口径对比。
- [cuda-reduction-strategy-comparison](skills/cuda-reduction-strategy-comparison/SKILL.md)：CUDA 归约三种策略的选择与测量。
- [cuda-backend-choice-proof](skills/cuda-backend-choice-proof/SKILL.md)：证明 CUDA 库算子选到了预期算法、精度或缓存键。

## 后端与真实设备验证

- [hccl-on-device-verification](skills/hccl-on-device-verification/SKILL.md)：真实多卡 HCCL 的数值、竞态和同步策略验收。
- [isolated-validation](skills/isolated-validation/SKILL.md)：建立隔离缓存并核对源码或 wheel 的实际导入来源。

### 无目标硬件时

- [acl-host-syntax-check](skills/acl-host-syntax-check/SKILL.md)：无 CANN 主机上对 ACL C++ 做真实 TU 语法检查。
- [static-launch-equivalence](skills/static-launch-equivalence/SKILL.md)：去样板前后设备调用的静态等价性证明。
- [cuda-negative-path-verification](skills/cuda-negative-path-verification/SKILL.md)：真正跑 CUDA 负向用例并判断绿色是否算数。

## Torch 兼容层

- [torch-shim-noop-audit](skills/torch-shim-noop-audit/SKILL.md)：判断一个 torch API 是真实现还是签名齐全的空操作。
- [torch-api-cohort-promotion](skills/torch-api-cohort-promotion/SKILL.md)：把一族 torch API 写成模块级一等对象并登记保真度。
- [jittor-torch-diff](skills/jittor-torch-diff/SKILL.md)：shim 与真 PyTorch 的双解释器前反向对拍与梯度调试。

## 主仓与下游的交界

- [downstream-library-adaptation](skills/downstream-library-adaptation/SKILL.md)：将问题分流到核心、通用 compat 或独立集成仓库，并确定对应回归证据。
- 各库的版本组合、adapter、官方测试、教程、基准与功能/芯片支持表由 [JittorEcosystem](https://github.com/JittorEcosystem) 对应仓库维护。主仓不再复制每个项目的运行手册。
