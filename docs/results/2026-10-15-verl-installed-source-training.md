# CPU 与 CUDA 上的已安装来源 VERL 组件训练

- 状态：两个原始 tiny 组件协议在 CPU 与 CUDA 上均通过完整梯度、两次输入扰动与三步 SGD；限定的组件 L3。
- 日期：2026-10-15。
- 基线提交（测量源码）：`73333a677dae984a3f577afb7d1586aecac5f490`；版本证明 schema 前置 `dbea21762d2df71b7c67dcf2ba9abb4f642944da`。本报告是更晚的文档提交。
- 上游：干净 VERL `e5a627cf55c94d85406556854c2c1a22db4ff5e7`，模块版本 `0.10.0.dev`，实际发行物版本 `0.10.0.dev0`。
- 维护者：Jittor Torch 兼容层与下游验证维护者。
- 复查条件：上游 registry、已安装 payload、解释器/provider、精度、组件 builder、测量工具或训练协议变化。

本次新运行保留[更早的 VERL 协议](2026-10-15-verl-registered-component-oracles.md)的完整范围，
并新增实际已安装包证明。历史测量保留在原始源码身份下；
未回填任何历史元数据。没有使用 adapter、复制算法、上游补丁或模型下载。

## 实际源码与已安装包

未修改的干净 checkout 被归档并构建为 `verl-0.10.0.dev0-py3-none-any.whl`，SHA256
`3bd60a64f76de6ff889951b1e994b1d08fa8aaff34b3b22da49a23e61006af0c`。
这是由固定源码自行构建的 wheel，不是官方 PyPI wheel。两个解释器都导入
该已安装 wheel payload。inventory 记录 360 个 Python 源码与 462 个受跟踪包文件；
wheel 携带 413 个包文件，含全部 360 个 Python 文件与 53 个数据文件。其余 49 个
受跟踪非 Python 文件不被表示为已安装。每个 runtime 阶段都绑定全部源码哈希、
全部 413 个实际已安装 payload 哈希、干净上游 SHA、wheel SHA 与实际发行物元数据。

模块的字面 `.dev` 版本保持不变。wheel METADATA 将其正规化为 `.dev0` 是合法的。
共享证明保留两个字符串，并只通过严格的
`packaging.Version` 相等、完整 wheel 证明与每个 runtime
及阶段的显式 `distribution_version` 接受该差异。版本缺失或不等即失败。字面相等与既有 SCM 证明保留其先前
契约；纯源码 inventory/audit 入口不会急切 import packaging。

## 范围、梯度与冻结预算

| Case | 真实已登记执行与 shape | 完整梯度范围 | 三步字段 |
| --- | --- | --- | --- |
| `verl_ppo_gae_vanilla` | `gae` + `vanilla`；3→1 Linear；features 4×6×3；49 值 loss/advantages/returns 输出 | `policy.weight` (1×3)、`policy.bias` (1)、features (4×6×3) | 21 |
| `verl_distillation_k3` | 已登记 packed `k3`；28×3 features 与 28×1 teacher；4×6 token 输出 | `student.weight` (1×3)、`student.bias` (1)、features 与 teacher | 24 |

GAE 保留其实际 `no_grad` 语义。比较其 advantages 与 returns；被请求的输入
梯度来自 policy 分支。Packed TensorDict 输入故意没有共同 batch
维。k3 配置不启动蒸馏引擎。teacher 与 student 两侧的
输入梯度都检查。每个 runtime 必须为每个被请求梯度产生精确完整的键、shape、FP32 dtype、
有限值与一个非零张量。缺失、多余、为零或可广播
但 shape 不等的字段都失败。

Base seed 0 与扰动 7/19 使用相同初始权重。CUDA 还复用 CPU 参考
初始权重。扰动后的输出必须改变。训练使用 seed 0、确定性 step 输入 seed
1/2/3、学习率 0.001 的三次真实 SGD 更新，两个命名参数都保持可训练并
发生改变。每步比较输出、loss、全部参数/输入梯度与更新后参数。
保留原始阶段预算：

- Base seed 0 使用 `max_abs_error / max(reference_field_max, gradient_block_max * 0.001, 1e-6)`，输出严格小于 0.002、梯度严格小于 0.01。梯度块下限仅适用于梯度字段。
- 扰动与训练使用原始的逐字段和：`max_abs_error <= 0.002 + 0.01 * max(reference_field_max, 1e-8)`。

没有模型级缩放、参数数量稀释、收敛声明或改变的容差。

## 平台、独立 oracle 与实际设备

| 属性 | CPU | CUDA |
| --- | --- | --- |
| Runtime | Python 3.11；NumPy 2.2.6；Transformers 5.5.3；TensorDict 0.10.0；Ray 2.54.1 | 相同 |
| Oracle | 真实 PyTorch 2.11.0+cu128 及其真实二进制扩展；加速器被遮蔽 | 相同，实际 CUDA 12.8 provider |
| 候选 | 经 Torch 兼容层的 Jittor 原生 CPU | 真实已分配的 NVIDIA RTX 4090；NVCC 12.2.140；驱动 580.178.04 |
| 精度 | FP32，TF32 OFF，matmul `highest` | FP32，TF32 matmul/cuDNN ON，matmul `high` |
| 条件 | 两个 runtime 线程，cuDNN benchmark OFF，严格 native kernel math，离线 | 相同 |
| 候选执行 | CPU 分发与 CPU 物理 storage | CUDA 分发与设备物理 storage |

候选 fallback 策略为 `error`，实际计数为零。观察器在两个 runtime 与设备上共记录 304 个张量
观测，包括每个导出输出、梯度与训练
参数快照。候选观测在同步与分发后检查物理位置；
它们不会把张量搬到某设备以制造表面驻留。CPU 与 CUDA 的 benchmark 缓存与
单测缓存分别独立。不作 L4 或速度结论。

两次初次 provider 配置失败分开保留。旧共享参考安装
缺少 `_strobelight/compile_time_profiler.py`，发生在任何数值运行之前。未修改任何共享包。
随后安装了私有完整官方 PyTorch wheel；其首次 import 选到了
不兼容的 CUDA 12.2 runtime。为该私有 prefix 提供标准的相邻真实 NVIDIA CUDA 12.8
包布局后修复了 provider 选择，未改动 Torch wheel 字节。成功运行的
私有 wheel SHA256 为 `c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`；
验证了 11,841 个原始 wheel 文件，包括二进制 `_C` SHA256
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
全部 2,227 个 Python 源码路径与 wheel 的确切源集合匹配。全部 17 个实际映射的 NVIDIA 库
都被取哈希。正式运行后的检查重新哈希了原始 wheel payload、映射的 provider 与测量
源码，均未变化。这些失败是环境失败，不是通过的数值测量。

## 结果与维护门禁

| 设备 / 组件 | 全阶段被比较数组对 | 最大绝对误差 | 最大 base 归一化误差 |
| --- | --- | --- | --- |
| CPU / GAE + vanilla | 33 | 2.38418579e-7 | 3.65876907e-6 |
| CPU / packed k3 | 39 | 7.15255737e-7 | 5.00747165e-7 |
| CUDA / GAE + vanilla | 33 | 2.38418579e-7 | 0 |
| CUDA / packed k3 | 39 | 2.98023224e-7 | 2.98727005e-7 |

两个维护节点都运行完整三个 seed 与三步训练，包含必需的依赖、
独立解释器、完整源码/wheel、设备与零 fallback 检查。CPU nox 实际
完成 **2 passed, 0 skipped**（pytest 375.98 秒）；CUDA nox 实际完成 **2 passed, 0 skipped**
（pytest 521.53 秒）。冷 nox session 分别约 12 与 14 分钟。这些是
opt-in 维护门禁；它们不确立 nightly 执行。

通用 scorecard 重算全部 144 个数组对并检查全部 32 个 runtime 阶段元数据文件后，
接受全部四条记录，零 rejected 或 blocked。
自动 registry 分母在各设备上分别仍为 advantage estimators **1/14**
（`gae`）、policy losses **1/12**（`vanilla`）、distillation losses **1/8**（`k3`）、Hydra entrypoints **0/5**。
只有这些命名的 tiny 组件场景获得 L3。Ray/rollout worker、完整 PPO trainer、Hydra
训练、模型级蒸馏、其他平台与整库覆盖仍未验证。

源码验证：97 项聚焦 host 契约通过；CPU core **322 passed, 42 skipped, 1 xfailed**；
`tests/structure` **1510 passed, 4 skipped**；lint 与布局通过。Host 负向测试覆盖严格
版本正规化、两个 runtime/阶段证明、完整 payload、缺失/多余张量字段、
shape/dtype/非有限/零梯度、错误标记/case/非零退出与错误物理驻留。
对实际元数据副本的另外十三种改动都被拒绝。原始证据
未变。稍后一次隔离文档检查有 14 项通过，一项为本新报告的中央 toctree
登记待完成；该索引与中央 manifest 由集成方负责。一次未使用专用缓存环境的
初次文档检查启动被停止并单独保留；
它不计为通过的验证运行。

## 复现与检查证据

维护 session 使用预先配置的独立解释器与干净的源码构建 wheel；
它不安装或下载下游包。配置每个解释器优先导入实际
已安装 VERL payload 及其声明的 NumPy 2 / Transformers 5 依赖。

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_ECOSYSTEM_PACKAGE_SITE="$VERL_SITE" JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$VERL_SITE" \
  JITTOR_VERL_UPSTREAM="$VERL_CHECKOUT" JITTOR_VERL_WHEEL="$VERL_WHEEL" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_verl_training
# CUDA: use a fresh cache, JITTOR_TEST_DEVICES=cuda, explicit CUDA_VISIBLE_DEVICES,
# and nvcc_path pointing to the real allocated CUDA toolchain.
```

确切的环境变量名与实际命令数组也保存在命令产物中；
在其他机器上复现时使用该 session 声明的必需变量。实际节点：
`compat/tests/torch/test_ecosystem_verl_training.py::EcosystemVERLTraining::test_verl_ppo_gae_vanilla`
与 `::test_verl_distillation_k3`。

原始产物位于仓库之外：
`$JITTOR_LAB_ROOT/_state/verl-source-bound/20261015/`。成功数据位于
`formal-73333a677dae-provider-v3/{cpu,cuda}/`；更早的目录保留两次配置失败。
`clean-source-freeze-v1.json` 绑定测量源码。`scorecard-v1.json` 含重新生成的
inventory、记录与审计器结果。`verl-sealed-raw-manifest-v1.json` 固定 240 个文件（127,073,470
字节），SHA256 `ea4a86547d4f3f96a97d5010c746fc425ed6ad0dc016b777c057f09ec6b4b32e`。
最终证据 manifest 另绑定最终 scorecard 与文档身份。
全部原始数组、元数据、源码证明与保留的失败日志仍以其实际
源码与 provider 身份保存。


## 主整合验证

集成方同步了 `1e98e127e406f1924f3c3729f7f58254637b4455`，重放三个
封存改动，并为本次验证冻结干净 `6f25bcf5957414c78e6d45c4238626589d80b02a`。
nox 冲突通过保留既有主线前缀完整并追加
原 VERL session 解决。全部 71 个既有函数 AST（含 Swift/PEFT）保持一致；
Swift 的模块名映射也完整保留。中央发现现在会验证两个实际
`EcosystemVERLTraining` 方法；缺少任一方法或其类/session 会 fail closed。
发现不会分配执行级别，也不添加合成 CASES。

原最终封存的 263 个文件（948,221,295 字节）经独立重新哈希，未变。
当前审计器接受同样的四条原始记录，零拒绝。一次独立
NumPy 重算按原始严格的 base 相对与扰动/训练逐字段预算检查原始 144 个数组对的每一个：最大绝对误差
`7.152557373046875e-7`。其生产者仍为 `73333a677...`，包括原始设备与
provider。私有 oracle prefix 中全部 11,841 个原始官方 Torch wheel 文件也经
重新哈希未变（1,624,262,090 字节）；这遵循原证明对 pip 重写 RECORD 的排除，
而不是断言 RECORD 未变。

主线验证使用新的私有候选 venv 与新的按设备 unit cache。
原始候选 prefix、oracle/provider、已安装 VERL payload 与上游 checkout 都是
只读输入。CPU 显式遮蔽 CUDA；CUDA 使用另一张单独分配的真实卡。首次
CPU 与 CUDA core 编译串行执行。新的主线结果与
原始正式 scorecard 分开：

| 主线验证 | 结果 |
| --- | --- |
| 聚焦版本、VERL、中央门禁契约 | 83 passed |
| 共享已安装证明、组件、Swift、Diffusers、PEFT 契约 | 77 passed |
| 实际 CPU `ecosystem_verl_training` | 2 passed / 0 skipped, 292.03 s pytest |
| 实际 CUDA `ecosystem_verl_training` | 2 passed / 0 skipped, 504.48 s pytest |
| 规范 CPU core | 322 passed / 42 skipped / 1 existing xfail，两种模式 exit 0 |
| 完整 `tests/structure compat/tests/structure` | 1622 passed / 6 skipped, 206.81 s |

六项结构 skip 为两项 accelerator、两项 declared 与两项在未安装该
发行物的解释器中的已安装 native 解析检查；它们不授予任何安装等级。
新/修改 Python 文件的 lint、仓库布局、MANIFEST 与空白检查通过。
维护门禁运行两个原始完整协议、seed 0/7/19 与三次真实更新。
其临时原始数据成功后正常删除；这是新的维护门禁运行，
不是新的持久 scorecard，也不是对原始 733 个数组的重标。nightly 准备
仍是独立任务；这些 opt-in 运行本身不确立定时执行。

未版本化整合日志、命令、干净源码冻结、新旧门禁发现证明、
独立数组重算与私有 provider 重新哈希位于
`$JITTOR_LAB_ROOT/_state/verl-main-integration-1e98/`。上述全部边界仍然有效：
只有这些组件获得所述等级，不含完整 trainer、Hydra 或速度声明。
