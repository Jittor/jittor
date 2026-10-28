# VERL 维护版 CPU nightly 与独立安装前缀

- 状态：nightly 源码接线已实现；等效的维护版 CPU 会话在本地通过。不声明运行过 GitHub 托管工作流。
- 日期：2026-10-16
- 基线提交：全新环境准备 `d80422c89d29f2f6dd050847a2a5b0fda72194b5`；成功测量的源码 `302900315cf3c4bd9449f166d1e1632feacfea94`。本报告是更晚的文档改动。
- 维护者：Jittor Torch 兼容性维护者
- 复查条件：VERL 固定版本、依赖 profile、已安装源码证明、组件用例、数值预算、工作流或 nox 会话发生变化。

## 维护入口与范围

`.github/workflows/ecosystem.yml` 中既有的维护版 CPU 矩阵现在包含 `verl`，与 PEFT 和 Swift
并列；更早的通用生态作业仍然保留。它调用真实的
`tools/provision_ecosystem_nightly.py --library verl --run` 入口，而不再依赖开发者本地的
包目录。可变的检出、wheel、三个环境、HOME、临时文件、缓存和日志都位于显式的外部 state
根目录或 CI 的 `RUNNER_TEMP` 之下。

两个必需的节点仍是
`compat/tests/torch/test_ecosystem_verl_training.py::EcosystemVERLTraining::test_verl_distillation_k3`
和该类中的 `test_verl_ppo_gae_vanilla`。它们运行实际的上游已注册 k3、GAE 和 vanilla
策略损失函数。这是组件级训练，不是 Ray worker、rollout、Hydra 入口点、CPU/后端 extra 或
完整的 VERL trainer。没有缩减任何算法、梯度请求、训练步或数值阈值。

上游是位于 `e5a627cf55c94d85406556854c2c1a22db4ff5e7` 的干净官方 VERL 源码。清单从该
检出重新生成：14 个优势估计器、12 个策略损失、8 个蒸馏损失和 5 个 Hydra 入口点。有界的
CPU 场景覆盖 `gae` **1/14**、`vanilla` **1/12** 和 `k3` **1/8**；入口点仍为 **0/5**。
这些维度相互独立，不会被累加成一个整库百分比。

## 全新真实环境与已复现的失败

在 `d80422c89` 上执行的 provisioning 新建了 candidate、oracle 和 controller 环境，分别
安装了 133、131 和 17 个发行版。运行时 profile 使用官方 PyTorch **2.11.0+cpu**、NumPy
**2.2.6**、Transformers **5.5.3**、Ray **2.54.1**、TensorDict **0.10.0**、PEFT
**0.17.1**、Accelerate **1.10.1**、Datasets **3.6.0** 和 PyArrow **19.0.1**。完整的基础
依赖闭包由 pip 解析、保存为 lock，并在三个环境中检查。Controller 构建工具与运行时
setuptools 约束保持分离。可选的完整 trainer CPU/后端 extra 未启用。

实际的 oracle 在不带 shim 的情况下加载了其二进制 Torch 扩展，并报告 CUDA 不可用。
NumPy 2 / PyArrow / Datasets 的互操作被执行。CUDA 被显式隐藏；候选选择原生 CPU 后端和
`error` 回退策略，计数为零。源码构建的 VERL wheel 是 `verl-0.10.0.dev0-py3-none-any.whl`，
SHA256 `609f257d53ea96dd31861af485f2dee9531abbcb9d49b2c69484871d716fba9b`。
源码/模块版本 `0.10.0.dev` 与实际分发版本 `0.10.0.dev0` 保持不变，并通过既有的 PEP440
和完整已安装 wheel 证明进行检查。

第一次真实的维护版尝试在输入扰动与训练之前就**两个节点都失败**，因为比较器要求依赖的
`origin` 字符串完全相同。分离的前缀具有相同的实际已安装载荷，但绝对路径不同。其
**2 failed / 0 skipped，126.01 s** 的结果、原始代码和原始文件予以封存；它不是训练成功。
一个单独的仅 base 诊断检查了九对张量。

修复首先重新哈希五个实际依赖包目录。记录的 origin 必须是真实的绝对文件。VERL 以及任何
既有的逐依赖发行版绑定都必须与各自一侧的 origin 根、版本和完整已安装载荷一致。只有在此
之后，跨运行时身份比较才省略 `origin` 和 `package` 位置。所有源码、wheel、分发版本、
载荷和新引入的身份字段仍被检查。原始元数据保留实际位置。主机侧反例拒绝缺失或无效的
origin、错误前缀的绑定、被改动的真实或记录载荷、伪造的源码/wheel/METADATA、范围变化和
未匹配的新字段；它们是元数据契约，不是数值证据。

## 成功的新测量

在干净源码 `302900315` 上，同样私有准备的环境运行了一个**全新的**完整维护版会话：
**2 passed，0 skipped，205.12 s**。搭建身份仍为 `d80422c89`；provisioner 和依赖安装代码
在这两个提交之间没有变化。既有的任务自有 benchmark 缓存被正常复用，未复制二进制，也未
声称是另一次冷构建。其首次编译此前已串行执行。单测缓存分离。三个环境在新的测量之后都
再次通过 `pip check`。

| 必需的 CPU 节点 | 参数 / 请求的输入梯度 | 所有阶段的数组对 | 最大绝对差异 |
| --- | --- | ---: | ---: |
| packed k3 蒸馏 | student weight+bias / features+teacher | 39 | `7.152557373046875e-7` |
| GAE + vanilla PPO loss | policy weight+bias / features | 33 | `2.384185791015625e-7` |

每个节点运行固定初始权重，种子为 **0、7 和 19**，检查扰动输出确实改变，然后执行原始的
**三次 SGD 更新**，学习率 **0.001**。所有请求的参数和输入梯度都被检查；两个参数都必须
实际更新。基础门禁保留其原有的归一化输出 **0.002** / 梯度 **0.01** 限值和梯度块下限。
扰动与训练字段保留其原有的 `max_abs <= 0.002 + 0.01 * max(reference_abs_max, 1e-8)`
判据。表中报告的是观察到的误差，并不定义替代预算。

该 harness 现在为这两个节点保留 **36 个文件、13,426,995 字节**，包括每一侧的 NPZ 和实际
源码元数据、初始权重以及两份 `gate-complete.json` 记录。只有在全部四个完整阶段通过后
才写入记录；失败的目录保留其原始数据，但不带成功标记。运行后的复核器重新验证了每个
绑定文件的哈希、每一侧的已安装源码以及**全部 72 对张量**，包括完整的训练字段。实际的
后代进程记录保留严格的环境允许列表，包括私有 HOME/JITTOR_HOME/TMPDIR 和隐藏 CUDA 的
选择；不输出完整的环境转储。

## 复现与检查

从干净的检出出发，使用健康的 Python 3.11 和外部 state 根目录：

```bash
python tools/provision_ecosystem_nightly.py --library verl \
  --state-root "$STATE/fresh-verl" --run
```

可通过 `--source-cache` 和 `--wheel-cache` 提供只读的精确源码与 wheel 输入；依赖解析、
源码验证、实际安装和 pip 检查仍会执行。常规的
`python -m nox -s ecosystem_verl_training` 入口在显式配置独立解释器和已安装包路径后
仍可复用，如 [VERL skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/verl-torch-compat/SKILL.md)
中所述。

nox 会话在一次性会话暂存区之外，每次调用分配一个新的证据目录。`JITTOR_VERL_EVIDENCE_DIR`
可选择另一个外部输出根；重复的用例输出会被拒绝而不是覆盖。最终比较之后的失败由一个
负向测试覆盖，该测试检查保留的原始文件和成功标记的缺失。

在 `302900315` 上，相关主机检查通过 **85 / 0 skipped**，子进程契约通过 **8**，规范 Ruff
和布局检查通过，完整结构加 compat 结构在 **223.09 s** 内通过 **1,624 / 4 skipped**。
第一次结构运行有一处新的测试 fixture 解释器拼写失败（**1,623 passed / 4 skipped /
1 failed**）；它在成功运行之前被修复。两次中间的结构启动在执行测试之前以规定的
`jit_utils` 重建状态退出；同样的正确配置命令随后通过。这些未成功的调用保留在证据中。

在每次测量和结构进程终止之后，整合了 origin `9c6ee1dde`，未替换其 R33 改动。VERL 辅助
函数和被测组件源码字节保持不变。主机检查再次通过 **85 / 0 skipped**，规范 Ruff 和布局
通过；本次文档同步不会给 `302900315` 的数值或结构结果重新贴标。

未纳入版本控制的证据位于 `$JITTOR_LAB_ROOT/_state/ecosystem-verl-nightly/20261015/`：
`verl-v1/provision.json`、其 lock/日志和 `upstream-proof.json` 保留全新搭建；
`failed-v1-manifest.json` 保留第一次真实失败；`verl-v2/source.json` 绑定全部 2,882 个受
跟踪源码条目（包括 Git 符号链接目标文本）；`verl-v2/measurements/`、
`recomputed-comparisons.json`、实际进程记录和 `unit-full/` 保留新的检查。最终的
`manifest.json` 封存交付证据。历史的 `73333a677` CPU/CUDA 评分卡保持不变。这次新的 CPU
维护版运行不会被重新贴标为新的公共评分卡、CUDA 运行、整库支持、性能结果或实际的 GitHub
计划执行。

## 中央整合复核（2026-10-16）

整合负责人审阅了工作流、provisioner、维护版 harness 和严格的逐前缀身份改动，随后在
`902e733ed` 整合它们。原始搭建与测量仍为 `d80422c89` 和 `302900315`；本次整合不会给这些
执行重新贴标。不导入 Jittor 的独立重放重新检查了**72 对完整张量**、全部八个阶段封存和
两个实际已安装源码绑定。全部 **2882 个原始 Git blob**（包括符号链接目标文本）都与冻结的
测量快照一致。完整的交付封存匹配全部 **149 个文件 / 20273226 字节**，manifest SHA256
`284bab3ec2aafec0d5bfcfd82d86c50d3723385d11f8f106f464dcb84d19d0d7`。
三个实际的 candidate、oracle 和 controller 环境在交付后也通过了独立的 `pip check`。

整合后的主树通过 **74 项定向主机契约**，核心 **322 passed / 42 skipped / 一个既有预期
失败**，完整结构加 compat 结构 **1622 passed / 六个 skipped**（208.99 s）。六个实际结构
跳过中包含两个解释器安装解析检查；它们不会被改述为成功的已安装包检查。改动 Python 的
Ruff、布局、生成的 manifest 和 diff 检查通过。验证脚本、原始复核和终端日志未纳入版本
控制，位于 `$JITTOR_LAB_ROOT/_state/verl-nightly-root-integration-13a/20261016/`。
没有任何旧的数值包或交付封存被修改。上述全部完整 trainer、托管计划、新 CUDA、共享
评分卡和性能边界仍然有效。
