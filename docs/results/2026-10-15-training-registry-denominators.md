# 训练 registry 分母

- 状态：源码枚举已验证；枚举不授予任何数值或设备
  支持级别。
- 日期：2026-10-15。
- 基线：`9d9310e3c9a4af737702e12d2d6d47062fa015cd`；实现为包含
  本报告的提交。
- 维护者：下游兼容层维护者。
- 复查条件：上游 registry 布局或 SHA、registry 抽取或
  scorecard 成员统计变化。

## 缺失的轴与抽取

维护中的抽取器此前只报告 ms-swift 的 model/tuner 登记
与 verl 的 advantage-estimator/policy-loss 登记。它遗漏了它们的 trainer、
CLI、declared task/RLHF 与蒸馏轴，尽管这些在同一干净源码 checkout 中
同样存在。新的抽取读取受跟踪的 Python AST，不导入下游库
或张量 runtime。每份 inventory 仍记录完整上游 SHA、remote、版本
及其所用源文件的哈希。

普通 trainer 与 Megatron trainer 以及 CLI 路由是各自独立的轴。声明的
`rlhf_type` 别名仍是独立轴：SimPO 出现在参数中
并不额外产生一个 trainer 类。只有参数模块拥有的字段
才计入该轴；模型元数据字段不定义 CLI 选项。
verl Hydra 条目包含训练、评估与生成，并以源路径、
函数与 config 名称标识。它们不是算法验收。

缺失或动态 registry、不支持的 Literal 声明、有歧义的轴
与非法蒸馏名称会报错，而不是静默缩小
分母。没有新增手工维护的模型或算法名称列表。

## 固定源码结果

全部五份 inventory 都从干净的受跟踪源码 checkout 重新推导：

| 仓库 | 版本 | 完整上游 SHA | 独立分母轴 |
| --- | --- | --- | --- |
| Transformers | 4.56.2 | `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` | 374 个模型族；46 个 task mapping |
| PEFT | 0.17.1 | `53c25fe4fdd7c6aa4b40db0560815ea570d32303` | 26 个方法；6 个任务 |
| Diffusers | 0.35.1 | `0f252be0ed42006c125ef4429156cb13ae6c1d60` | 79 个 pipeline 模块族；249 个 pipeline 导出；37 个 AutoPipeline 族；3 个 AutoPipeline task mapping |
| ms-swift | 4.5.2 | `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` | 222 个模型族；11 个 tuner；13 个普通 trainer；5 个 Megatron trainer；12 条普通 CLI 路由；4 条 Megatron CLI 路由；5 个 declared task；9 个 RLHF 选项 |
| verl | 0.10.0.dev | `e5a627cf55c94d85406556854c2c1a22db4ff5e7` | 14 个 advantage estimator；12 个 policy loss；5 个 Hydra 条目；8 个蒸馏 loss |

这些计数是彼此独立的分母，不能相加成一个支持率。
特别地，Diffusers 较小的 AutoPipeline 矩阵不能替代其
完整 pipeline 分母。用这些 inventory 生成、且没有
数值证据的 scorecard 保留了每个分母，并将每个分子都赋为零。

## 复现与检查

```bash
python -S agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py \
  --repo ms-swift --checkout <clean-pinned-ms-swift-checkout> \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/ms-swift-registry.json"
python -S tests/tools/test_training_registries.py -v
python -S tests/tools/test_downstream_coverage.py -v
```

八个纯源码回归用例通过，包含动态与有歧义
registry 反例、非法名称，以及上游新增方法别名
但不改变 trainer 覆盖的变化。既有 18 项覆盖/审计契约
也通过。源码 fixture 中不可用的张量 runtime 导入证明
枚举不执行上游代码。完整五源码 scorecard CLI
成功完成，没有被接受的证据行，所有分子为零。
完整结构选择通过：1,432 passed、six skipped、无
失败。两项 skip 涉及纯源码解释器中的发行物
解析；其余是 declared 或 accelerator 能力 skip。初次
运行在新 host fixture 中捕获到收集期的 `exec_module`；将其
替换为普通 import 修复了测试契约，而未放宽门禁。布局
与空白检查通过。未版本化结构日志为
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/training-registry-structure-recheck.log`。
CPU core runner 也通过：322 passed、42 skipped 与一个既有预期
失败，两个子进程均返回零。其未版本化日志为
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/training-registry-core.log`。

使用 Python 3.11。未版本化基线与最终 inventory、源码哈希
与空 scorecard 保留在
`$JITTOR_LAB_ROOT/_state/refactor-goal/r21-registry-audit/` 下。文件名为
`baseline-ms-swift.json`、`baseline-verl.json`、`final-<repo>.json` 与
`empty-scorecard.json`。这些是 registry 证据，不是模型 benchmark 数据。

## vLLM 与 Omni 中央 inventory 后续

基于 `1eaf2678d` 的后续工作将仅 host 的 vLLM 抽取器接入
同一 `registry_inventory.py` CLI 与 schema。实际干净上游 checkout
被重新枚举；没有导入任何张量 runtime 或下游包：

| 仓库 | 来源身份 | 完整上游 SHA | 独立分母轴 |
| --- | --- | --- | --- |
| vLLM | 确切发布 tag `v0.29.0` | `98dff2a81d747d1dba01a47f939f48c3526d4206` | 376 个模型架构；40 个 CustomOp 层声明 |
| vLLM Omni | `source@446c2b5474dabffccf9a0034836eeab77c62b4b8` | `446c2b5474dabffccf9a0034836eeab77c62b4b8` | 92 个自有模型架构；68 个扩散模型；64 个 pipeline 模型类型 |

这两份 inventory 分别绑定 589 与 172 个被扫描源码哈希，并保留
动态注册位置与缺失实现目标。Omni 继承的
vLLM registry 需要单独固定的 vLLM inventory；它不会被静默
加入自有模型分母。每个轴仍保持独立。CustomOp
成员资格或某一层的数值结果不能认证引擎或模型。

当生成的 package 版本文件缺失时，只有所选提交上确切最终发布
tag 才提供发布版本。近邻 tag 或 release
candidate 不够；纯源码身份仍为 `source@<full SHA>`。
多个不同的确切发布版本会 fail closed。该源码标签
不替代独立记录的已安装发行物版本，这些
枚举也不授予数值 scorecard 行或 adapter 版本支持。

用现有 CLI 复现，使用 `--repo vllm` 或 `--repo vllm-omni` 与
对应的干净固定 checkout。该 helper 的 14 项契约与中央
覆盖/身份/文档检查一并通过：77 passed，无 skip。
它们覆盖 dirty source、未知改动、缺失目标、继承范围、
release-tag 歧义与上游重新枚举。原始 inventory 与日志
未版本化，位于 `$JITTOR_LAB_ROOT/_state/leaf-view-integration/`：
`vllm029-inventory.json`、`omni-inventory.json` 与
`vllm-registry-contracts-corrected.log`。

## SCM wheel 身份后续

基于 `a5932248f`，共享审计器额外支持针对
`source@SHA` inventory 证明真实 SCM 发行物。它分别保留源码
标签与实际安装版本。带哈希的原始 wheel 必须
具有正确的发行物名称/版本，并包含每个重新生成的
`package_sources` 文件及其确切哈希。仅有 registry/目标哈希
不够。两个 runtime 的 base、扰动与
训练测量都必须绑定该 wheel 及其实际已安装源码
哈希。该路径不放宽与已发布版本的匹配。

实际干净源码 Omni wheel 的版本为
`0.29.0rc2.dev58+g446c2b547`，而不是 release `0.29.0`。其 SHA256 为
`c6a43c816c95f99f751dd9aa18c36ff94461d508da002a2a328c543e14b2c0de`。
172 个 registry/目标源码哈希仍是限定的分母证据。
额外的整包 inventory 绑定全部 1474 个受跟踪 Python 文件；
每个文件都与 wheel 及独立安装目标匹配。该 wheel 唯一
生成的 Python 文件 `_version.py` 也绑定到其安装哈希。
发行物元数据一致。这仅是一次
身份检查：不从这些文件准入任何 Omni 数值、设备或模型覆盖。原始证明未版本化，位于
`$JITTOR_LAB_ROOT/_state/scorecard-scm-identity/omni-full-wheel-source-installed-proof.json`；
更早的 172 文件子集检查另行保留。
共享契约还覆盖错误的源码/包/版本、缺失 wheel、
release 版本绕过以及缺少逐 runtime/逐训练安装
绑定；这些都会拒绝，而不削弱原始数值协议。
