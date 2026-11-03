# Jittor 开发技能

按当前任务选用；仓库规则只在 [AGENTS.md](../AGENTS.md) 维护，公开说明见
[开发文档](../docs/development/index.md)。这里不存项目状态、协作手册或历次运行报告。

每个技能的 `SKILL.md` 说明用途、操作与验收边界；附带工具放 `scripts/`，
较长的条件性协议放 `references/` 并从技能入口链接。诊断探针不等于通过门禁，
主机检查也不代表硬件验收。任务、缺陷与验证结果记录到 GitHub issue/PR。

## 环境与验收

- [isolated-validation](skills/isolated-validation/SKILL.md)：隔离源码、wheel、worktree 与缓存，核对实际导入来源。
- [verifying-a-gate-actually-ran](skills/verifying-a-gate-actually-ran/SKILL.md)：核对收集集、执行结论、skip 和基线差异。
- [gate-tier-budget](skills/gate-tier-budget/SKILL.md)：维护 core/smoke/full 的选择和算术预算。
- [codebase-wide-fix-as-rule](skills/codebase-wide-fix-as-rule/SKILL.md)：把同族缺陷变成有负向验证的结构规则。
- [pure-code-motion-refactor](skills/pure-code-motion-refactor/SKILL.md)：检查搬移的代码、导出、初始化与调用路径。

## 核心、构建与运行时

- [jittor-build-change-verification](skills/jittor-build-change-verification/SKILL.md)：核心重编、配置能力、缓存与导入成本。
- [jittor-pyjt-bindings](skills/jittor-pyjt-bindings/SKILL.md)：绑定生成、类型转换与对象生命周期。
- [jit-compile-failure-attribution](skills/jit-compile-failure-attribution/SKILL.md)：JIT 编译错误的真实算子归属。
- [core-invariant-property-tests](skills/core-invariant-property-tests/SKILL.md)：图关系、显式记录、遍历与生命周期不变量。
- [jittor-allocator-flag-matrix](skills/jittor-allocator-flag-matrix/SKILL.md)：分配器与并发存储问题诊断。
- [process-global-state-and-optin](skills/process-global-state-and-optin/SKILL.md)：可撤销的全局状态、注册和 hooks。
- [hang-that-holds-the-gil](skills/hang-that-holds-the-gil/SKILL.md)：有界等待、GIL 与挂死定位。

## 数值与 Torch 兼容语义

- [jittor-op-parity-oracle](skills/jittor-op-parity-oracle/SKILL.md)：独立前向/梯度参考、输入污染与反向图查询。
- [optimizer-semantics-diffing](skills/optimizer-semantics-diffing/SKILL.md)：多步更新、梯度累积、裁剪与参数组。
- [torch-checkpoint-stride-oracle](skills/torch-checkpoint-stride-oracle/SKILL.md)：checkpoint 的 storage、stride、别名和 dtype。
- [torch-api-cohort-promotion](skills/torch-api-cohort-promotion/SKILL.md)：一族 compat API 的实现归属和对象身份。
- [torch-shim-noop-audit](skills/torch-shim-noop-audit/SKILL.md)：区分真实行为、合法 identity 和空操作。
- [downstream-library-adaptation](skills/downstream-library-adaptation/SKILL.md)：核心、compat 与独立生态仓库的问题分流。

## 设备与通信

- [multi-device-verification](skills/multi-device-verification/SKILL.md)：多卡指针、分配器和事件设备归属。
- [jittor-distributed-verification](skills/jittor-distributed-verification/SKILL.md)：MPI/NCCL、rendezvous 与 Native FSDP 验收。
- [hccl-on-device-verification](skills/hccl-on-device-verification/SKILL.md)：真实 Ascend 多卡通信与同步策略。
- [acl-host-syntax-check](skills/acl-host-syntax-check/SKILL.md)：无 CANN 主机上的 ACL TU 与 launcher ABI 检查。
- [static-launch-equivalence](skills/static-launch-equivalence/SKILL.md)：设备调用包装重构的静态证明边界。
- [cuda-negative-path-verification](skills/cuda-negative-path-verification/SKILL.md)：真正执行 CUDA 负向路径并清理 worker。

## 性能与生成代码

- [jittor-performance-profiling](skills/jittor-performance-profiling/SKILL.md)：构图、规划、执行和通用性能入口。
- [jittor-single-kernel-cold-profile](skills/jittor-single-kernel-cold-profile/SKILL.md)：合成单核的冷编译分阶段计时。
- [codegen-optimization-in-effect](skills/codegen-optimization-in-effect/SKILL.md)：源码、机器指令与实际收益的证据链。
- [cuda-elementwise-bandwidth-roofline](skills/cuda-elementwise-bandwidth-roofline/SKILL.md)：逐元素流量、copy 带宽与 profiler 归因。
- [cuda-reduction-strategy-comparison](skills/cuda-reduction-strategy-comparison/SKILL.md)：同一实际 kernel 的归约正确性与性能。
- [cuda-backend-choice-proof](skills/cuda-backend-choice-proof/SKILL.md)：CUDA 库算法、精度与缓存键的命中证据。

下游库的固定版本、adapter、官方测试、教程、基准和功能/芯片支持表由
[JittorEcosystem](https://github.com/JittorEcosystem) 对应仓库维护。
