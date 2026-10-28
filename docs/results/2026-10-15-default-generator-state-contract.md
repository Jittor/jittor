# 默认生成器的 CPU 状态、私有副本与设备隔离

- Status: 限定 CPU 与真实双 CUDA 设备的默认生成器契约通过。
- Date: 2026-10-15。
- Baseline: 同步 `7a1e66b3858279b40eaec0c4895c6059ce8eb8db` 后冻结修改；实际待提交源码哈希在下述 `source-freeze-release.json`，提交后逐文件映射 Git blobs，历史结果不重标为干净基线测量。
- Owner: Native RNG 与 Torch frontend maintainers。
- Review when: 默认/显式生成器、CPU 状态编码、CUDA seed owner、pickle 或安装依赖顺序变化。

## 问题与修复

独立 binary Torch `2.11.0+cu128` 的默认生成器与显式对象具有同一个公开类型；
默认对象的 `manual_seed` 与 `seed` 只修改全局 CPU 流。pickle/copy 得到的对象
持有私有流，不能继续操作进程默认流。同一后端的默认/私有生成器可交换完整状态。

同步基线的最小实际复现与之不符：`default_generator` 为另一代理类，
两种播种方法会重置 CUDA 状态；pickle 副本播种会修改全局 CPU 状态；
默认/私有 CPU 状态互相恢复均被拒绝。CPU 与实际 CUDA 的原始日志均保留。

现在由 `compat/torch/generator.py` 定义一个公开 Generator owner。默认对象持有
全局 CPU owner 的视图，显式及反序列化对象持有原有私有 native Generator。
默认对象的播种使用已有 `rng_manual_seed("cpu", ...)`，保留
`torch.manual_seed` 的全局播种语义。seed 校验在修改状态前完成，不修改 Python/NumPy 流。
原 `installers.tensor.Generator` 继续导入同一个类，保留旧 Python 类查找路径。
旧 `_DefaultGenerator` 代理没有完整状态的历史 pickle 不在可恢复范围内；
不会把这类空状态对象无声映射为可续接的私有流。

CPU 的 Torch 状态共用 runtime 已有 `JITTOR_RNG_1` 全引擎编码与严格解析器；
新增 native 私有 getter 提供该编码。原生 `jt.Generator.get_state()` 的既有 frame、
CUDA offset 与 Philox 路径保留。读取旧显式 CPU frame（有/无 stream tag）仍被检查，
CPU/CUDA 错设备、损坏和尾随字节拒绝继续由原 owner 执行。
没有新增 PRNG、随机计算内核、Python 状态解析器或跨框架字节转换。

## 实际验证

候选 CPU 显式隐藏 CUDA；CUDA 使用实际 RTX 4090、NVCC 12.2.140、strict kernel math，
最终回归暴露两张设备。unit 缓存与 benchmark 分开；首次编译串行完成。
新契约检查都在禁止 fallback 的 scope 中执行。正确性运行不授予性能结论。

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| 独立 Torch CPU | 20 passed / 3 CUDA skipped | 同一 23 条契约的原测试主体 |
| 独立 Torch CUDA | 23 passed / 0 skipped | binary extension，拒绝 Jittor import |
| 候选 CPU 扩大回归 | 83 passed / 39 skipped | 默认、显式、fork、sampler、安装 owner 与十项结构 fixture |
| 候选真实 CUDA 扩大回归 | 112 passed / 0 skipped | 同组回归及既有双设备条目；每个可见 CUDA 流均有真实续接检查 |
| native CPU random/generator | 14 passed / 21 skipped | CPU；其余为 CUDA/可选 Torch 节点 |
| native CUDA random/generator | 33 passed / 2 skipped | 实际 CUDA、两种旧 CPU frame、原生 counter 保留；可选 Torch 节点未执行 |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | native 98 + Torch 224，两个 mode exit 0 |

新契约覆盖 copy/deepcopy/pickle 副本独立、完整位宽 seed、默认/私有 CPU 状态
互换、保存前未读回的随机操作、恢复后真实随机序列，以及三个默认对象方法对实际
CUDA 后续序列的保护。每次比较完整数组与状态；不是只比较 seed。

独立参考复用新测试的全部函数主体与参数。仅移除候选的 Jittor 导入与 fallback/sync
fixture，替换为真实 Torch 的状态保存恢复 fixture；每次拒绝 Jittor 导入并检查真实
`torch._C` 扩展。转换后的脚本与原测试 SHA256 单独保留，没有替换测试里的计算、
预期值或 device。跨框架比较的是这些行为契约，不要求不同 PRNG 的样本数值相同。

初次 Python owner 合并触发真实导入循环，已移入依赖中立的 Generator 模块后复验。
最初直接改变 native CPU get_state 编码时，原生 CUDA 回归发现 CPU handle 的既有
CUDA counter 路径丢失续接位置（1 failed / 32 passed / 2 skipped）。最终保留 native
公开 frame、单独提供 Torch CPU getter 后，同一回归 33 passed / 2 skipped。
原失败未删除，也不计为最终通过。

扩大到 compat 结构套件时，另有两项旧 fixture 失败：仍要求已删除的全局 reseed 入口，
以及最小 NativeModule stand-in 缺少现有 child-view 协议。在精确 7a 基线的归档
源码上用 stdlib unittest 独立复现同样两个失败。fixture 现在检查私有 draw 委派
且不全局 reseed，并加载真实 native child-view descriptor 检查副本绑定；
没有修改 NN 生产代码。移出 Generator 时 shape helper 的 NumPy 重导出依赖
也显式保留，模块导入和安装 owner 会在最终回归重新执行。

## 证据与边界

未版本化目录：`$JITTOR_LAB_ROOT/_state/default-generator-integration-7a1/`。
`sync.json`、baseline/oracle probes、两轮 source freeze、最终 CPU/CUDA/core 日志、
原始失败与 `oracle-test-binding-release.json` 保存实际身份和验证范围。
`integration-manifest.json` 记录最终工具/结构检查、原始文件 SHA256 与最终提交映射；
该目录只引用按白名单保存的字段，不收集整个进程环境。

源文件已有的 Ruff 诊断与基线逐项比较，不新增诊断；新 Generator owner 和新测试
完整通过 Ruff。最终布局、manifest 和完整结构门禁另记该 manifest。

这批只修复默认对象类型归属、CPU 状态互换、私有副本和设备播种隔离。
它不授予完整 C-extension ABI、所有 Generator 方法、整数采样精度、cuDNN RNN
完整位宽 seed、本表之外的双 CUDA/Ascend/ROCm 场景或跨 Torch RNG 字节互通。
R33 的作者 rebase 与维护者最终处置仍未执行，不因本报告关闭。
