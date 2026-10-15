# 上游 PR 处置与 NumPy ABI 实测

状态：2026-10-15 只读复核官方 PR 状态与当前源码处置；历史 Linux CPU NumPy 1/2
回归仍绑定下面的原测量基线，本轮没有重跑数值或更改 GitHub PR 状态。Windows MSVC、
CUDA 13 打包栈和 Ascend 的新验收不属于本轮。
维护者：Jittor 核心与兼容层维护者。
历史初始冻结基线：`f02e7aec0b43daf1b275a97eab340de7c4e917d3`，同步自
`origin/2.0-refactor` 后新建独立分支，保留此前检查点与图重放分支。
历史数值集成基线：`a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`；原基线测试完整收尾后
冻结本树，fetch 并 fast-forward，保留三个本地测试/报告文件，没有 stash 或丢弃修改。
当前只读复核基线：`4a4a480e628dae634a9d660f320c3aa957341e99`。新任务先确认干净且
无本树测试，再同步 `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` 并另建分支；
此前 TRELLIS/TP refs 保留。官方 PR 快照时间为 `2026-10-15T01:56:19Z`。随后
确认无本树测试，将唯一未提交报告备份/hash 后无损 fast-forward 至 4a4；Windows
官方 API 于 `2026-10-15T02:15:23Z` 补查。本轮只更新本报告，原 NumPy/shape/jtune
测试与历史日志不改写，没有新增计算、绑定、适配或运行时验收。
复查条件：PR head/base 更新，NumPy C-API、shape 代码生成、RNG owner、FSDP FULL
或 CUDA wheel 解析协议变化。

## 官方状态与源码处置

先读取既有报告与 PR API 快照，再从 GitHub 官方 REST API 重新获取十个 PR 的
base/head/state，并重新获取 #18031 的三页 209 文件和其余七个小 PR 的完整 files
patch。十个 PR 的 state、head/base ref 与旧快照一致：九个 open，#18042 merged。
下面 SHA 是官方 PR 响应，不把 PR 的历史 base
当作目标分支的最新 tip。完整响应、HTTP headers 和补充 Git diff 保存在外部实验区。
`#18031` 的 209 个文件本轮完成三页重新获取；`#18042` 复用原审计已分页的
277 文件快照，本轮确认其 head 未变并重新检查 merge ancestry。`#18028` 的
3184 文件是 API 的 changed_files，超过 REST files 接口的 3000 文件上限；完整
内容比较使用已经获取的 Git commits。当前 Git 的默认 rename 检测把
`git diff --name-status 9eb696d9...e5cffe1a` 计为 2935 条，`--no-renames` 则为
3717 条（323 M、2294 A、1100 D）。这些计数口径不同，均不冒充 REST 的 3184；
完整文件清单、stat 和 merge-base 已保存，没有把第一页当完整差异。

| PR（官方来源） | 当前状态；base → head | 对冻结树的处置 |
| --- | --- | --- |
| [#18031](https://github.com/Jittor/jittor/pull/18031) | open；`2.0-refactor/e184a85f` → `cgq_transformers/e5cffe1a`；209 files | 需在最新 2.0-refactor 上重新整理剩余差异。多模态、FSDP、原生分布式和 RNG 已有不同程度的整合，原始差异同时携带旧目录和后续 owner 冲突，不能按标题授予整体通过。 |
| [#18028](https://github.com/Jittor/jittor/pull/18028) | open；`2.0/9eb696d9` → 同一 `e5cffe1a`；3184 files | 对当前目标 base 不适用，建议由作者/维护者标记为 #18031 的重复评审入口；保留 head，在最新 refactor base 上只评审剩余功能。 |
| [#18033](https://github.com/Jittor/jittor/pull/18033) | open；`master/06f5d3d2` → `1cf2a4e1`；3 files | 已覆盖：`python/jittor/contrib/loss_ssl/` 和 `tests/nn/test_loss_ssl.py`；整合提交 `8224841d4`。现有 CPU 数值、梯度及联合训练用例通过，不重复建立旧 `loss_ssl` 顶层目录。 |
| [#18029](https://github.com/Jittor/jittor/pull/18029) | open；`master/06f5d3d2` → `ef7619fb`；1 file | 已覆盖：`python/jittor/tools/jtune.py` 使用 `.format(cmd=cmd)`，并明确报告命令失败。已有 unknown-command 回归保留历史真实错误信息验证。 |
| [#18024](https://github.com/Jittor/jittor/pull/18024) | open；`master/5e412c63` → `2eb4233c`；1 file | 原补丁对当前实现不适用：`backends/cuda/kernels/cufft/cufft_fft_op.cc` 已使用 `CufftPlanKey`，没有原来的 `std::array` 使用点。没有因缺少无用 include 修改源码；此源码判断不代表新 GCC12/CUDA 实测。 |
| [#18022](https://github.com/Jittor/jittor/pull/18022) | open；`master/3fe57d77` → `b7f16ef8`；5 files | NumPy ABI 已覆盖且历史真实 CPU 验证通过（原测量基线）；当前以 dtype 推导字节数，避免补丁的描述符私有偏移读法。其 normalization 部分也已由更完整的稳定归一化 owner 覆盖；旧 1.3.x 版本号改动不适用于 2.0。 |
| [#18021](https://github.com/Jittor/jittor/pull/18021) | open；`master/3fe57d77` → `fa80a654`；3 files | property 与变长维度标注已覆盖；保留当前 `Tuple[int, ...]`，不移植 PR 的固定单元素 `Tuple[int]`。原生返回类型仍是 NanoVector，见下面的实际绑定证据。 |
| [#18020](https://github.com/Jittor/jittor/pull/18020) | open；`master/3fe57d77` → `c934ac2e`；3 files | 旧 packed `data.gz`/`data.o`/`vdp` 路径在当前 build/core 源码中已不存在，不能原样移植旧编译器 patch。Windows MSVC 导入仍待真实 Windows 构建、导入与数值验收，Linux 不能关闭该硬件事项。 |
| [#18041](https://github.com/Jittor/jittor/pull/18041) | open；`2.0-refactor/cded3102` → `7ccc796c`；27 files | uv lock/dev group 已存在；原 f02 基线仍缺 CUDA13，最终 a1 基线已整合下述固定 CUDA13.0 独立方案。PR 的混合 CUDA12 runtime、双 CCCL 和 Cu12/NCCL ranges 与该方案仍不同，需独立评估，不能记为 PR 整体已接受。 |
| [#18042](https://github.com/Jittor/jittor/pull/18042) | closed/merged；`2.0-refactor/cc193436` → `73bffaed`；277 files | 已合入：官方 `merge_commit_sha=659e6a47f42466ba407f2d453c0d1224ba5ca0c6`，Git ancestry 检查确认它是冻结基线祖先。其代码导入不自动消除后续文档化限制。 |

初次官方 branch-ref API 确认当时 `2.0-refactor` tip 为 `3ba5bb771…`；
最终 Git fetch 已同步后续 `4a4a480e6…`。初次官方响应中的
`2.0` tip 为 `88ae21d1fad87438dfd17815d9f0400e7a521d5f`，两个 PR 的 head ref
均为完整 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`。因此 #18028 与 #18031
是同一源码 head 的不同 base 评审入口，不是两组独立功能。#18031 的 API mergeable
为 unknown/null，#18028 为 false/dirty；只记录该响应，不把它当作重新整理已通过。
#18042 的 merge commit 仍是当前基线祖先。对 master-target PR 的“已覆盖/不适用”
判断仅指当前 2.0-refactor；没有据此关闭仍需单独维护的旧 master 版本。

集成 owner 另提供已完成的 `98da22192aa886c584f9d4aef73a45c3810a5fb7`
及 `docs/results/2026-10-15-cuda13-toolchain-packaging.md`：固定 CUDA 13.0
五个 toolchain wheel、C++17、设备属性 API、exact `libcudart.so.13` 链接，以及
真实 uv 安装后的 CUDA 数值/梯度验证；原 cu122 fingerprint 和 23 项回归保留。
该方案已整合到最终 a1 基线，对应公开[CUDA13 验证报告](2026-10-15-cuda13-toolchain-packaging.md)；它没有接受 #18041
对 Cu12/NCCL ranges 的整体放宽，也不授予任意 CUDA13/运行库组合通过。

NumPy PR 中的 `instance_norm`/`layer_norm` 中心化方差和低精度 FP32 中间量
已在 `nn/functional/normalization.py` 的统一 `_ln_normalize`、稳定自定义 backward
及 dtype 恢复路径体现。模块调用该 owner，避免再次恢复旧 `nn.py` 或重新拼装
一个只修 forward 的实现。本次没有重新授予 normalization 的 CUDA/NPU 覆盖。

## cgq 的重新整理方案与必须保留的 API 超集

保留原 `cgq_transformers` ref，不把其完整文件覆盖到最新树。建议在最新
`origin/2.0-refactor` 上新建评审分支，先用
`git log --left-right --cherry-pick --no-merges origin/2.0-refactor...origin/cgq_transformers`
列出 patch-ID 不等价的提交，再对照目标 owner 逐项重放尚缺的功能。保留作者与原提交
出处，将旧 `refactor-wip/results/` 证据按当前公开结果目录规则整理。
需要审查全部依赖，而非将 209 文件或 3184 文件简单缩为同一批 cherry-pick。

3ba 冻结时的 Git 实际 merge-base 为 `e184a85fe9d7704e8319627a2382cd3477ea35bf`；
`rev-list --left-right --count 3ba5bb771…...e5cffe1a…` 得到 208/84，排除 merge 并
按 patch-ID 去重后为 205/72。**72 是待语义审查的提交数，不是 72 个尚缺功能**。
最终 4a4 基线重新计算得到 ancestry 209/84、patch-ID 去重 206/72，merge-base 不变；
两份结果分别封存。PR #18031 的 209 条路径在 3ba 布局中有 46 条不存在，包含已退休的 refactor-wip
记录、已拆分测试和可能迁移过的 RNG 文件；路径不存在也不等于能力缺失。

重新整理的具体审阅单元如下；保持原 head/ref，建议在最新目标上逐组形成小提交，
不执行一次覆盖式导入或自动重放全部 84 个旧提交：

| 审阅单元 | 当前保留点与待办 | 可复用证据 |
| --- | --- | --- |
| Transformers 多模态/算子/adapter | 对照实际模型族、全部梯度及训练输入；缺能力到 native，Torch API 到 compat，库私有差异才进 adapter。旧大模型/速度记录不能替代当前矩阵 | [现有矩阵](2026-10-15-transformers-model-family-matrix.md)：4.56.2、8/374 模型族和 2/46 任务、CPU/CUDA 各九个 tiny L3 场景，明确绑定 8e388 测量源 |
| FSDP/Module/optimizer/checkpoint | 保留共享 native FSDP 数学、FULL gather materialize、FQN/group 转换、刷新 optimizer shard 绑定、native Module strict/copy 路由和别名/生命周期修复 | [FULL bf16 续训](2026-10-15-bf16-fsdp2-checkpoint-continuation.md)、[选项](2026-10-15-checkpoint-state-options.md)、[生命周期修复](2026-10-15-fsdp-compat-lifecycle.md)；原 native FSDP 报告的两个 compat 失败已有该后续修复，不能继续泛称未修 |
| RNG/显式 Generator | 保留 CPU engine、CUDA seed/offset、ACL 每 device 状态及同步/校验 owner。64-bit manual_seed 边界和 generator-aware randint/randperm 仍需按具体入口筛选，不能用第二套 Generator/Philox owner 覆盖已有 native owner | `src/runtime/rng_state.cc`、`random_generator.{cc,h}` 与 curand owner；`tests/ops/test_random_generator.py`、compat generator/RNG 回归及 [RNG/FULL 报告](2026-10-14-checkpoint-rng-fsdp2-continuation.md) |
| 原生训练与工程文件 | 对照现有 process-group/launcher/DDP、ZeRO1、TP、FSDP 和 checkpoint 公共入口；不回填旧 AGENTS、目录、MANIFEST、结果看板或另一套 collective/optimizer 数学 | 下面分项的真实 CPU/单机两 rank 报告；多机与任意组合仍另外验收 |

实际 head 源码有三类需要明确处理的差异：

- cgq 已实现 `get_cpu_rng_state`/`set_cpu_rng_state` 的完整 CPU engine 快照，
  **不能根据其过期的 `_MISC_DETAILS`「seed-only」描述断言运行时只恢复 seed**。
  它新增的 `set_cpu_seed(uint64)`、64 位 seed 范围检查和 lazy work flush 应逐项
  评估为待保留行为。当前 compat `_manual_seed` 仍调用 int 参数的 `jt.set_seed`，
  并非已经接上 uint64 `rng_manual_seed`；本次只读源码核查未新增这个边界的运行时
  复现或修复。显式 Generator 则已有 native CPU engine 与 CUDA Philox owner，但 compat
  `_draw_from_generator` 的私有 stream 路径仍在 NumPy 中抽样；不能仅看输出被放到
  CUDA 就授予该路径 device-native 支持。移植前需按入口验证并复用现有数学。最新树已有 `jt.core.rng_state/set_rng_state/rng_manual_seed`
  及统一 runtime owner，重新整理时将行为落到该 owner，避免恢复旧全局 `eng`
  和 `current_offset`。cgq 的 `JITTOR_CPU_RNG_V2` 与主树 `JITTOR_RNG_1` 是不同的
  opaque 格式，不能直接互读，也不能转为 seed-only 来消除格式冲突。
- 最新树保留 CPU 完整流位置、CUDA cuRAND seed/offset 和每个 ACL device 的
  seed/counter 隔离；`set_rng_states` 先完整校验再修改状态，CPU 恢复不重置设备流。
  cgq 的 deterministic setter 是 no-op/getter 固定 False，不能覆盖当前直接读写
  backend policy 的实现。继续守住格式/设备拒绝、lazy/async 同步和实际流续接回归。
- 当前 `checkpoint_state` owner 的 `get/set_optimizer_state_dict`、FQN/group
  映射、合并 model+optimizer API、CPU offload 与 rank-zero 广播均须保留。
  cgq 的 combined setter 只加载 model、getter 直接返回 optimizer 原始索引字典，
  不能覆盖该超集。保留 FULL CPU-offload gather 在各 rank 丢弃前 materialize，
  load 后刷新 FSDP optimizer shard 绑定，以及 importable-but-UNIMPLEMENTED 的
  `DefaultSavePlanner`/`DefaultLoadPlanner` 边界；不把 FULL 成功计为 SHARDED I/O。

主树已有 [CPU/CUDA RNG 与两 rank FULL 续训证据](2026-10-14-checkpoint-rng-fsdp2-continuation.md)。
`3d93c2ca91fe2a24cdb8945f3534e116489b72f8` 的真实两 rank bf16 Accelerate FULL
续训回归也已整合到最终 a1 基线，见[bf16 续训报告](2026-10-15-bf16-fsdp2-checkpoint-continuation.md)。
最终树另含 StateDictOptions 的选项/metadata/严格加载验收，见
[检查点选项报告](2026-10-15-checkpoint-state-options.md)。重新整理 cgq 时一同保留这些
更完整的 owner 契约；本次 CPU PR 审计没有重新执行该 CUDA 验收。
截至本轮基线，“TP/ZeRO/原生分片 checkpoint 均未完成”的旧总括已过期：

- [原生分片训练 checkpoint](2026-10-15-native-sharded-checkpoint.md) 已有 native
  Module/optimizer/四类 RNG 的实际 CPU 与两 rank CUDA 保存恢复、BF16 原始位、
  完整性/缺口拒绝及独立参考续训；这是 checkpoint 存储能力，不据此声称训练 sharding。
- [ZeRO1](2026-10-15-native-zero-optimizer.md) 已验证实际 owned moments、多个组、
  累积/DDP 去重归约、CPU/单机双 rank CUDA 与新对象续训；参数/梯度仍复制，
  固定 ownership/world，不能写成 ZeRO2/3 或任意后端通过。
- [原生 TP](2026-10-15-native-tensor-parallel.md) 与
  [BF16/AdamW 后续验收](2026-10-15-native-tensor-parallel-adamw-bf16.md) 已有真实
  Column/Row local 参数、通信伴随梯度、独立完整 Torch 参考和 fresh resume；
  后续共享 Linear 复验保持已冻结的 BF16 预算；最新 tanh 整合后真实 CUDA/NCCL
  两 rank 各 13 项复验也通过，仍限公开声明的均匀 Linear 分片组合。
- [原生 FSDP](2026-10-15-native-fsdp.md) 已有真实 FP32 参数/状态分片、FP32/BF16
  双 rank 训练与 fresh-launcher 精确续训；[saved-output tanh 后续验证](2026-10-15-tanh-fsdp-bfloat16.md)
  在 `cc45f12a216af639faf1e7bd3fc2ff916ce16653` 固定原 fixture/预算后，80/80 个
  BF16 梯度数组逐位匹配独立完整 Torch，全部 368 个 fresh-resume 数组精确一致。
  独立真实 Torch FSDP2 的 396 个数组也逐位等于其 dense oracle。原报告的 11.2242%
  小梯度差异保留为旧源码结果，不再作为当前未修缺陷；实际 CUDA/F0 证据见后续报告。

这些分项都不是两台真实 host 的证据。多节点、任意 FSDP/TP/ZeRO 组合、world-size
迁移、完整 Torch SHARDED planner 和各未声明的设备/精度范围继续独立保留。

## 本轮复核的复用边界

#18033 的功能在 `jittor.contrib.loss_ssl`，#18029 在当前 jtune 的显式
`AssertionError("unknown cmd: {cmd}".format(cmd=cmd))`；保留当前 2.0 公共布局，
不复制旧 `jittor.loss_ssl` 目录。#18022 的 NumPy 版本化 CopyInto slot 与 dtype
推导字节数、稳定 normalization，以及 #18021 的 property/`Tuple[int, ...]`
仍在当前 owner 中。原 generated binding 的 runtime NanoVector 与静态 tuple 近似
必须继续区分；本轮没有重新授予任何 NumPy ABI 二进制通过。

与交付历史回归的 `0cce2377482083282b5d81aee1df43c30fe57b95` 逐字节对照的
13 个 runtime/test 文件中，12 个完全一致。唯一不同的 `py_converter.h` 只增加
GradCallback 的 `python_callable` 记录；这一差异也完整保留，未把它说成整文件相同。
该对照证明报告引用的具体源码/测试仍可复用，**不替代新基线的实际 NumPy1/2 执行**。
下节的 Linux 数值结果保留原测量日期与 a1 基线。

#18020 的旧 packed-data 路径仍没有当前移植点。当前新增 Windows workflow 和
`tools/release/check_windows_msvc.py` 要求真实 Windows、cl.exe、仓外安装 wheel、
实际编译/训练及零 fallback，Linux 负向 guard 不能产生 passed certificate。
官方 contents API 取得 `windows.yml@4a4a480e6…`，与当前文件逐字节一致；
SHA-256 为 `73e028e312b178404a7de928b90a39adf025c98df6b1df2e3578053fff2ca543`。
指定 `windows.yml` 的 runs API 返回 HTTP 404，不能单独据此推断零运行。补查公开
workflow catalog 为 HTTP 200，仅列出三个动态 Copilot workflow；全仓库 runs API
为 HTTP 200，`total_count=7` 且完整返回七条，均为 Copilot，没有 Windows 构建；
限定 `branch=2.0-refactor` 的 runs API 为 HTTP 200、`total_count=0`。该可见范围内
没有可授予 R16 的 Windows job、source-SHA 构建日志或 passed artifact，不能把
workflow 配置或 Copilot success 当作 MSVC 通过。这里只记录公开 API 当次可见结果，
不声称其他私有系统或未来运行不存在。维护者需提供指定 source SHA 的真实 Windows
结果、原始编译日志和该工具的 passed JSON 后才能关闭硬件边界；本轮未触发 workflow。

## 历史 NumPy 与真实 generated binding 验证

使用独立 Python 3.11.16 环境、pytest 7.4.4；NumPy 2.4.6 安装到任务私有 target，
没有修改共享环境。NumPy 1.26.4 与 2.4.6 分别使用独立 CPU JIT 缓存，首编译串行。
所有原生执行显式固定当前 worktree 的 `PYTHONPATH`，`JT_USE_CUDA=0/use_cuda=0`，
`JT_BUILD_NVCC_PATH=''`，并记录实际 package 文件与 active cache 路径。

初次 probe 对 int64 默认输出 int32 的断言失败，归因于已有
`auto_convert_64_to_32=1`，与 NumPy ABI 无关。保留原始失败输出，随后用显式 dtype
验证字节复制，没有更改默认转换行为。最终覆盖 bool、8/16/32/64 位 signed/unsigned
整数、float16/32/64 和 complex64/128 的 C-contiguous、负 stride 和 Fortran 布局。
新增测试分别从 Var 和 RingBuffer 读回所有 42 组数组，断言 dtype 和全部元素值；
负 stride/Fortran 路径实际进入 C-API `PyArray_CopyInto`。另检查标量和空数组。
独立 probe 还验证 Var.data 在原 Var wrapper 删除后仍保留数据。
可能崩溃的 C-API 回归通过项目 `run_python_child(..., crash_isolated=True)` 隔离。

真实 active `gen/pyjt_jit_op_maker.cc` 的 `shape` getset 条目调用：

```cpp
return to_py_object<NanoVector>(((GET_RAW_PTR(VarHolder,self))->shape()));
```

runtime descriptor 是 `getset_descriptor`，返回值是 `jittor_core.NanoVector`，
不是 callable，也不是 Python tuple；`tuple(var.shape)` 给出预期整数维度。
新增回归覆盖 rank 0、1、3、10。`src/core/var_holder.h` 的
`@pyjt(__get__shape)`、`tools/build/gen_pyi.py` 的 property 生成与当前 `.pyi`
变长形状声明一致。Tuple 标注是返回值的静态近似，**不是精确 runtime 类型保证**；
后续若要求完全准确，应由 typing owner 为 NanoVector/整数序列输出单独建模，避免
为标注改变原生返回 API 或把所有 NanoVector 输入一并收窄。

| 实际验证 | 结果 |
| --- | --- |
| NumPy 2.4.6 私有 probe：42 array copies、RingBuffer CopyInto、0D/empty、view lifetime、shape | 通过 |
| 新 NumPy ABI + shape 回归，NumPy 1.26.4 | 2 passed，0 skip |
| 同一回归，NumPy 2.4.6 | 2 passed，0 skip |
| NumPy 2.4.6：既有 array、SSL、jtune、pyjt binding protocol | 71 passed，7 accelerator skips |
| 新 unknown-command 与既有 jtune 回归，NumPy 2.4.6 | 5 passed，0 skip |
| 官方 CPU `core` runner，NumPy 2.4.6、native/Torch 独立 session | 322 passed，42 skipped，1 既有 xfail；两 session exit 0 |
| 最终 a1 基线：同源 NumPy2 ABI/shape、jtune、SSL、稳定 norm 回归 | 31 passed，11 CUDA skips |
| 最终 a1 基线：同源 NumPy1 ABI/shape 回归 | 2 passed，0 skip |
| 最终 a1 基线：官方 CPU `core` runner，NumPy2、双 mode | 322 passed，42 skipped，1 既有 xfail；两 session exit 0 |

官方 runner 初次因禁用 pytest plugin autoload 后遗漏显式 timeout plugin 而
报 `--timeout=900` 不识别，原始 exit 4 日志保留；随后显式设置
`PYTEST_PLUGINS=pytest_timeout,xdist.plugin` 重跑以上完整 core。该环境没有 cmake，
warmup 明确报告 oneDNN 未启用，计算使用原生 generic CPU kernels；不授予 oneDNN 验收。

原 f02 基线完整 structure：1430 passed、6 skipped、2 failed；两项失败准确对应
本次新增报告尚未登记的公共 `docs/results/index.md` 与 generated `MANIFEST.in`，
两文件由集成 owner 负责。首次 structure 因 jit_utils 更新按明确 exit 3 协议重启，
尚未执行用例的原始输出也保留。没有将这些结果记为全绿或 ACL 失败。

同步后最终 a1 基线完整 structure：1439 passed、6 skipped、2 failed；失败仍精确
对应本报告的 index 与 generated manifest 登记，没有新增源码契约失败。
最终 layout、staged/unstaged whitespace 检查通过。两次完整 structure 均存在两个
源码安装解析用例的环境 skip（此私有解释器未安装 Jittor），不将它们记为执行通过。
最终 NumPy1/2 定向、core 和 structure 的原始日志分别以 `a1-` 为前缀保存。

## 复现与边界

历史数值实验根目录为 `$JITTOR_LAB_ROOT/_state/upstream-pr-disposition/`：
`api/` 保存官方响应与完整 Git 文件统计，`private/numpy2/` 保存隔离 NumPy 安装，
`logs/` 保存 probe 与测试原始输出。源码、wheels、缓存和 API JSON 没有放入文档树。
本轮只读 API/headers、branch refs、Git 关系与 owner 字节对照另存于
`$JITTOR_LAB_ROOT/_state/upstream-pr-disposition-refresh/2026-10-15/`，未版本化；
`api/`、`snapshot.json`、完整 rename/no-rename 文件清单、Git 关系与字节对照的
最终 `*-4a4.json` 和 `windows-api-summary.json` 可直接复核。所有网络请求仅为 GET，没有
发送评论、消息、修改 PR/base 或向远端推送。在各自 NumPy 环境中固定待测源码后运行：

```bash
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' \
  PYTHONPATH=python python -m pytest -q tests/bindings/test_numpy_abi.py
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' \
  PYTHONPATH=python python -m pytest -q tests/tools/test_jtune.py tests/nn/test_loss_ssl.py
```

这些结论只授予已执行的 Linux CPU NumPy/SSL/CLI/binding 范围。PR 当前 open/merged
状态与硬件准入分别记录；需要作者 rebase、维护者标记 superseded 或真实 Windows/
CUDA13/NPU 验收的事项，没有通过发送 GitHub 评论或改变 PR 状态代替。
