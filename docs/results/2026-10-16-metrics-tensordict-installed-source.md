# Metrics 与 TensorDict：固定官方安装证据

- 状态：全新的 CPU/CUDA 组件测量；共享 schema 准入见下文。
- 复查日期：2026-10-16。
- 维护者：Torch 兼容与下游验证维护者。
- 测量源码：`91242395f555f3c24f60df665c3ea66e13d01caf`。
- 测量远端基线：`53d710647b0b44564af6fe476e0b16cf298b53b2`；在所有测量进程终止后，另行同步到 `2f8895f1a656bcdd66c280dbdbe222eb8268f56e`。后续同步不重标这些测量。
- 复查条件：库源码/wheel 固定版本、安装 metadata、注册表提取、组件/训练 builder、精度/设备/fallback 处理、快照观测或源码闭包校验变化。

本批关闭了一个实际的安装证据缺口。它改动测量与校验工具，不改模型数学、优化器、adapter 或数值预算。[历史报告](2026-10-15-metrics-tensordict-scorecards.md)保留其原始 `90ee` 数组与历史审计器结论。一次使用当时 `4ce` 守卫的只读审计接受了其两行 Metrics，即使既有非 Python 资源哈希声明被自洽地替换；其两行 TensorDict 已经因为生成的 `_version.py` 缺少源码权威而失败。这些是不同的发现。在本批更严格的守卫下，全部四条未改动的历史行都因缺少固定官方 wheel 证据被拒绝。没有向那些旧测量回填任何安装或平台字段。

## 实际组件范围

每次测量与收集都从干净的固定 checkout 重新枚举完整的已声明顶层公共 API 轴。不使用可选依赖过滤或更小的手写分母。

| 库与上游 | 已声明轴 | 实际维护规格 | CPU | CUDA |
| --- | --- | --- | --- | --- |
| TorchMetrics 1.7.4, `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 105 个顶层公共 API | MeanAbsoluteError + MeanSquaredError；FP32 prediction 与 target `(4,8)`；输出与两个输入梯度；零参数；种子 0 与固定权重种子 7 | L2, 6 对数组, 最大绝对 0 | L2, 6 对数组, 最大绝对 0 |
| TensorDict 0.10.0, `c34264a58b12471dc3e2913ea586b32431872d4c` | 60 个顶层公共 API | TensorDict + 实际 lazy_stack；batch `[2,3]`，FP32 sample `(2,3,4)`；整数/bool/lazy 索引与 Linear(4,3)；两个参数梯度与 sample 梯度；种子 0/7 加 lr .001 的三次 SGD 步 | L3, 29 对数组, 最大绝对 9.5367431640625e-7 | L3, 29 对数组, 最大绝对 7.152557373046875e-7 |

未改动的验收预算是绝对 .002 或缩放相对 .01，并带原有逐阶段全局下限。独立 NumPy 审计检查了全部 70 个实际数组、有限值、精确的字段/shape/dtype 集合、每个请求的非零梯度、两个参数在两个运行时中的真实更新、输入扰动下输出改变，以及 CPU/CUDA 之间完整相等的初始数组。它还检查了 20 个不同的测量 PID、148 条张量观测，以及所有 worker 之间一致的已提交源码映射。这些映射绑定 1,609 个选定的受跟踪框架/探针文件，包括 Git 符号链接目标文本；一份单独的完整 Git 归档与基于前置条件的 bundle 保留测量树。

组件覆盖在 CPU 与 CUDA 上各为 **L2 下 2/105 个 Metrics API** 与 **L3 下 2/60 个 TensorDict API**：core 0，broad 4 行 device/scenario，tail 0。其余 103/58 个导出未验证。TensorDict wheel 的原生扩展纳入安装权威；这不主张所有扩展 kernel 在此组合中都被执行。不由此得出整库、分布式、持久化、其他 dtype/后端、L4 或 L5 结论。

## 实际平台与安装权威

| 平台字段 | CPU | CUDA |
| --- | --- | --- |
| 操作系统 / CPU | Linux 6.1.188, glibc 2.34；AMD EPYC 9554 | 同一主机与 CPU |
| 已分配卡 / 映射 | 零, `CUDA_VISIBLE_DEVICES=''`；回读可用性为 false | 一张 NVIDIA GeForce RTX 4090；每个 worker 产物中记录实际物理 UUID 与可见/逻辑映射 |
| 驱动 / 工具链 / 编译器 | 驱动/CUDA/nvcc/CANN 不适用；g++ 11.5.0 | 驱动 580.178.04；原生 CUDA/nvcc 12.2.140；独立 Torch CUDA 12.8；g++ 11.5.0；CANN 不适用 |
| 精度 / 原生 flag | use_cuda/use_acl 0/0；TF32 与 cuDNN benchmark 关闭；highest | use_cuda/use_acl 1/0；两个 TF32 开关与 cuDNN benchmark 关闭；highest |
| 图 / 线程 | 默认图 replay 1，保留 268435456 字节；运行时与 OMP/MKL/OpenBLAS 线程 2 | 相同图策略与线程设置 |
| 独立二进制 oracle | 自有前缀中的真实 Torch 2.11.0+cpu | 另一自有前缀中的真实 Torch 2.11.0+cu128 |
| 速度 / 整进程内存 | 未为 L4 测量 | 未为 L4 测量 |

每个实际 worker 都捕获共享的 `_runtime_platform.py` 头部、选定 NVIDIA distribution 版本、编译器回读、精度、原生 flag、可执行文件/PID、完整运行时路径与源码闭包。candidate/reference 的 HOME、JITTOR_HOME、TMPDIR 与 XDG_CACHE_HOME 相互分离；benchmark 与单测编译缓存相互分离，初次编译串行执行。每个 CUDA 快照都校验真实驱动指针序号与已分配物理 UUID；原生快照还要求实际设备驻留/dispatch。两个设备都禁止 fallback 并回读计数为零。补充的 `/proc` 库映射覆盖十个实际 CUDA 测量 PID 中的九个，排除编译器子进程；它们佐证全新运行时前缀，从不替代张量证据。

四个新的私有计算环境完成依赖解析并通过 `pip check`；NumPy 为 1.26.4，SciPy 为 1.13.1。没有修改任何共享安装或旧封存前缀，也没有复制 JIT 缓存。独立 Torch 安装另外与其固定官方 wheel 输入、实际 METADATA 与全部 `torch/` 载荷文件匹配：CPU 11,781 个、CUDA 11,677 个文件，排除字节码缓存。实际 RECORD 文件被哈希，其 `torch/` 行被校验载荷哈希与大小。这些安装计数不是模型覆盖。

整个 wheel 的 SHA256 绑定输入归档。这里的已安装载荷校验覆盖完整 `torch/` 包，而非该 wheel 的每个前缀成员：`functorch/`、`torchgen/` 与其他 dist-info 载荷文件在这个附加 oracle 证明之外。本批使用正常的私有 pip 安装并保留其实际 pip RECORD 字节；它不会用归档 RECORD 覆盖已安装文件。

下游固定版本为官方 PyPI 构件：

| 包 | 整个官方 wheel SHA256 | 实际包载荷 |
| --- | --- | --- |
| TorchMetrics 1.7.4 | `9298ad0e893b0cf2956bee95b0f7eecdc65205ab84ddb4f6762eff157c240518` | 341 个受跟踪 Python 源码，346 个已安装文件，包括非 Python 资源 |
| TensorDict 0.10.0 cp311 manylinux | `bccb6a7d7f02a8c83ca41815943bfa284da7d28019ab4276135809bda5ba05a6` | 36 个受跟踪 Python 源码，42 个已安装文件，包括生成的 `_version.py` 与 `_C.so` |

新的 `_metric_td_installation.py` 保留库固定版本与 TensorDict 的实际 setuptools-scm 声明。干净上游 release tag 在精确 commit 处校验；固定 wheel 的生成版本为 0.10.0，其实际 commit_id 为 **None**，而不是虚构的上游 SHA。其字节仍由固定整 wheel 哈希保护。共享的 `_installed_source_proof.py` 重新校验选定的完整包载荷、选定 distribution 根、原始 wheel METADATA 与该包的实际 pip RECORD 哈希/大小行；vLLM 通过薄调用方保留其原固定版本与公共 helper 行为。

## 复现与审计

使用维护的 [`component_measure.py`](../../agent/skills/torch-compat-repo-runbook/scripts/component_measure.py)、[`verify_repo.py`](../../agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py) 与 [`training_probe.py`](../../agent/skills/torch-compat-repo-runbook/scripts/training_probe.py)。把 core/frontend 正常安装进 candidate 前缀，并保留一个单独的真正二进制 Torch 前缀。固定上面两个上游 commit、官方远端与 wheel；获取 TensorDict 的实际 `v0.10.0` release tag 并校验其 peeled commit。无 tag 的 checkout 诚实地登记为 `source@SHA`，并在此 release profile 下被拒绝。

每个运行时的可执行 launcher 必须在 `verify_repo` 净化子进程环境之后恢复其自身的四个运行时路径，然后转发到：

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/component_measure.py \
  --root <clean-measurement-checkout> --repo <torchmetrics-or-tensordict> \
  --checkout <clean-pinned-upstream> --wheel <fixed-official-wheel> -- "$@"
```

对 CPU，显式隐藏 CUDA，并把 canonical/legacy 两个 CUDA flag 置零、两个 nvcc 路径置空。对 CUDA，选择已分配的物理卡，启用严格原生 wheel 校验，并在**两个** launcher 中显式恢复实际的 `JT_BUILD_NVCC_PATH`：独立 oracle 的净化处理否则会移除它。使用 TF32 off、cuDNN benchmark off、highest、离线 flag 与相同的线程设置。转发 base `--seed 0 --repeats 3 --no-memory`，以种子 7 与原始完整参考权重重复维护 runner，并用 `--steps 3 --learning-rate .001` 与相同权重运行 TensorDict 训练。使用全新输出目录；collector 拒绝改动既有比较。

```bash
scripts=agent/skills/torch-compat-repo-runbook/scripts
python "$scripts/collect_component_scorecard.py" --root <auditor-checkout> \
  --upstream torchmetrics=<clean-metrics> --upstream tensordict=<clean-tensordict> \
  --wheel torchmetrics=<metrics-wheel> --wheel tensordict=<tensordict-wheel> \
  --run torchmetrics=<cpu-metrics> --run torchmetrics=<cuda-metrics> \
  --run tensordict=<cpu-tensordict> --run tensordict=<cuda-tensordict> \
  --platform cpu=<cpu-worker-derived-platform> --platform cuda=<cuda-worker-derived-platform> \
  --baseline 91242395f555f3c24f60df665c3ea66e13d01caf \
  --report docs/results/2026-10-16-metrics-tensordict-installed-source.md \
  --command '<complete retained two-runtime matrix invocation>' --output <new-scorecard-json>
```

collector 重新推导两个清单并调用实际 `audit_coverage.scorecard`；`final-scorecard.json`（SHA256 `80f3cb67e9a9321474fa548908c38737b304fd8aa4b6d4fd4c294cebdd5896b3`）包含**四条接受记录、零拒绝**及其绑定。共享平台文件保留稳定的实际二进制身份；进程特定 PID 留在各自原始 packet 中。它不能为已测量协议替换为无 CPU 的 CUDA 行或仅 metadata 声明。

## 回归证据与边界

定向安装/组件/平台测试通过 107 个 case，包括对实际资源/生成版本/原生载荷字节的自洽改动、诚实的 RECORD 修复与匹配的捕获声明：不可变 wheel 仍拒绝全部替换。另有共享源码/vLLM/Accelerate 测试通过 131 项。四个原始 c997 vLLM 实际 CPU/CUDA 安装 packet 在提取后保持逐字节相等的绑定。原始 c7 Accelerate 审计仍为两个接受/两个阻塞/零拒绝；全部 48 个原始平台头部原样重放。这是格式/守卫兼容性，不是数值重跑，也不是旧阻塞证据的升级。

第一次 CPU 尝试通过 Metrics，然后在 TensorDict 执行前停止，因为全新 checkout 缺少实际 release tag。第一次 CUDA 尝试在参考平台捕获阶段停止，因为外部 launcher 未恢复显式编译器路径。两次尝试与原始调用脚本都完好保留；成功的 v2 测量使用独立输出目录与同一干净 `91242395f` 源码。实际的复制构件负例也在省略固定 wheel、既有资源/生成载荷声明被自洽改动、显式 distribution 版本改为 9.9，或已提交源码哈希被改动时拒绝全部四行。移除一个完整参数梯度数组会被组件 collector 拒绝。移除 CPU 行会保留独立有效的 CUDA 数值记录，但产生零加速器覆盖分子；这不会把缺失的 CPU 证据变成运行时失败。

在后续干净校验源码 `2e0bdac97585df049da56d7a61dccaef5aeff3b3` 上，专用单测环境通过 CPU core：322 passed、42 skipped 与一个预期失败。完整结构检查以 1,628 passed、四个 skipped 与两个失败结束：新报告的中央 toctree 条目与生成的 MANIFEST 登记。这两个文件仍由集成归属方负责；这不是完整通过的 structure 主张。完整原始输出与精确失败节点保留在 `mandatory-status.json` 中。在所有进程终止后，远端 `bc4dc0bc01a3930d3fad22326da0f9b8c70d444f` 在 `0bcf640b140e764db828ef81bb57e67c93f46338` 处被整合；该同步不重标任何数值或 core/structure 结果。

全部原始数组、metadata、命令、官方 wheel 选择、环境锁、负例实验、独立复核器与源码归档/bundle 未版本化地位于 `$JITTOR_LAB_ROOT/_state/metric-td-installed-source/20261016/`；`evidence-seal.json` 绑定保留的文件。本批未执行或触发托管 nightly、未改动仓库默认分支，也未授予未见过的硬件/后端覆盖。维护节点仍是原有的 Metrics 与 TensorDict CPU/CUDA `EcosystemParity` 节点。

## 独立集成审计

集成审计重新哈希了交付封条中的每个文件：421 个文件与
110,079,967 字节，封条 SHA256 为
`7073bd2146bbe2d456cdc7fa0224fff4443edbac98ca20ad0d9e7fee1727298a`。
它独立重算了全部 70 对数组、全部 20 个不同 worker 身份、
148 条物理观测与 1,609 个选定源码哈希，对照原始
`91242395f` Git 对象。两个参数在每次 TensorDict 训练步都发生变化；
记录的梯度在两个 float32 ULP 内重现 SGD 更新。
完整历史归档也与全部 2,906 个受跟踪 Git 文件逐字节
匹配，包括可执行模式与符号链接目标；其基于前置条件的 bundle
对照保留的 Git 历史校验。重新枚举两个真实上游
checkout 复现了两个完整清单，包括生成的 TensorDict
版本权威。共同审计器仍接受四行、拒绝零行。
原始测量保留其原始源码 SHA；集成不
创造新的数值或整库覆盖。

干净集成提交 `67e166018cd5ec1c05413d2daf1c7302b2461970`
通过 260 项定向 host 测试，CPU core 322 passed / 42 skipped / 一个既有
xfail，以及扩展结构门禁 1,628 passed / 六个 skipped。布局、
生成 manifest、Ruff 与 diff 检查通过；全部 74 个既有 nox 函数
AST 保持不变。独立集成封条包含 28 个文件与
52,728,692 字节，SHA256
`7ffc1a2416cbcab5707d8aac6cf7b3710bccd397e213c74e929e1c1076e440df`。
一个 precommit 源码测试正确地拒绝了新 helper 在
先前 Git HEAD 中缺失；未改动的测试对照实际已提交源码通过。
