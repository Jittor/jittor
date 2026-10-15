# PEFT LoRA：完整安装来源与维护训练门禁

- 状态：CPU ①与真实 CUDA ③的原完整 tiny LoRA 范围通过；维护 nox 自动执行数值验收。
- 日期：2026-10-15。
- 实测源码：`1738f4a09d178becfad989f7bef7915729735890`，两设备同一 clean 提交。
- 维护者：Jittor core maintainers。
- 复查条件：PEFT/Transformers 固定源码或 wheel、共享 case/runner、训练 probe、兼容计算、来源采集或评分器变更。

本次重新测量旧报告已经声明的 `peft_lora_llama` 全部范围，补齐每个实际计算进程的
完整安装包与上游源码绑定。没有给旧 raw 补填身份，没有调整框架数学或容差，未新增 adapter。
旧 [2026-10-14 数值报告](2026-10-14-downstream-oracle-scorecards.md) 保留自己的源码与证据；
它缺少的新来源字段不能由本次安装目录反向补证。

## 固定来源与真实运行环境

| 项目 | 实际身份 |
| --- | --- |
| PEFT | 0.17.1，官方源码 `53c25fe4fdd7c6aa4b40db0560815ea570d32303` |
| Transformers | 4.56.2，官方源码 `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| 独立 oracle | Python 3.11，二进制 PyTorch `2.11.0+cu128`；真实 `_C` 扩展路径与 SHA256 逐进程记录 |
| CPU | AMD EPYC 9554，Linux/glibc 2.34；双方显式 CPU，候选关闭 CUDA/oneDNN/MPI/NCCL |
| CUDA | NVIDIA GeForce RTX 4090，driver 580.178.04；双方真实分配指针属于同一物理卡 |
| 编译与精度 | GCC 11.5，候选真实 NVCC 12.2；FP32，CUDA 两侧 TF32 matmul/cuDNN 开启、benchmark 关闭，CPU 两侧关闭 TF32 |
| 线程与隔离 | 双方各 2 线程；runtime × device 各自 HOME/JITTOR_HOME/TMPDIR/XDG/shim cache，计算前后路径一致 |

实际已映射 `libcudart` 的 API 查询为 oracle **12080**、候选 **12020**。
候选使用独立真实 cu122 package 前缀；不是将 cu128 库重标为 cu122。
每个进程记录实际首选 NVIDIA metadata 和已加载库的绝对路径/哈希；候选 cuBLAS
12.2.5.6、runtime 12.2.140、cuDNN 8.9.7.29，oracle 对应 12.8.4.1、12.8.90、9.19.0.56。
本 case 是单进程，不据此声明 NCCL 通信支持；导入时映射库也不等于使用其计算路径。

两个原始官方 PyPI wheel 的 SHA256：

- PEFT：`3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e`。
- Transformers：`79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`。

每个正式 worker 都在计算前后重新核对实际 import 根、`importlib.metadata` 首选发行物、
官方 wheel 的 METADATA/RECORD 字节以及全部已安装 package payload：PEFT **161** 文件
（158 Python，加 typed/C++/CUDA），Transformers **2173** 文件（2147 Python，含 C++/CUDA/头文件等）。
Transformers 上游额外的 GPT2 `CONVERSION.md` 与 T5 `download_from_gcp.sh` 未进入官方 wheel，
在库存中保留明确排除原因；未将它们伪记为已安装运行文件。
每个 worker 绑定 **1557** 个框架/兼容层/相关工具源码哈希，包括目录符号链接的 target 文本
与实际 compat owner。另封存实测 clean 提交全部 **2835** 个 Git 文件的 mode/blob/字节哈希。

## 原完整数值范围

复用现有公开 case、生态 runner 与 `training_probe.py`，没有复制模型数学。
模型是两层 tiny Llama：hidden 64、intermediate 128、2 attention/KV heads、vocab 128；
`q_proj`/`v_proj` 使用 LoRA rank 4、alpha 8、dropout 0。输入为 `(2, 8)` int64 token，
输出 `(2, 8, 128)` FP32 logits，离线运行。

每设备六个真实进程：base 的两 runtime、seed 7 输入扰动的两 runtime、三步训练的两 runtime。
全部进程加载同一份 oracle 初始权重，含 **29 参数和 1 rotary buffer**。
base/perturb 沿用 runner 原契约，临时请求 **全部 29 参数梯度**；训练保持原 PEFT 分区，
**8 个 LoRA 参数可训练、21 个参数冻结**，三步手动 SGD、learning rate 0.001。
输入 token 是整数，不请求 input grad；没有把 0 个请求写成浮点输入梯度能力。
每步检查输出、loss、全部 8 参数梯度和全部 29 更新后参数，并逐字节确认冻结参数不变、
可训练参数实际更新。输入扰动必须实际改变输出。

| 设备 | base：输出+29 梯度 | seed 7 扰动 | 三步轨迹 | 最大绝对误差 | 最大原量级归一化误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU ① | 30 | 30 | 117 | 1.4543533e-5 | 3.1351752e-7 |
| CUDA ③ | 30 | 30 | 117 | 1.0300159e-2 | 2.2198918e-4 |

独立 collector 与当前统一 auditor 重新生成全部来源库存并重算后：**2 accepted、0 blocked、0 rejected**，
两设备共 **354 array pairs**。原预算不变：`max_abs <= 0.002` 或
`max_abs / 原整组参考数组最大量级 <= 0.01`；并非逐元素相对误差声明。
CUDA 最大差异来自扰动组，三步轨迹最大绝对误差仅 **5.9627928e-7**。
CPU 三步轨迹最大绝对误差 **2.3841858e-7**。
候选各阶段都使用 `fallback_policy=error`、计数 **0**，同时观察实际 input/output/parameter/gradient
设备；CUDA 额外通过 driver 指针属性与 UUID 核对实际分配。oracle 的 fallback 字段只是旧 runner
占位，真实 oracle 身份由独立二进制及实际设备证明，不把该占位视作 Jittor fallback 度量。

覆盖按固定真实注册表分别计数：PEFT 方法 **1/26 @ 53c25fe4**（LORA）、任务 **1/6**
（CAUSAL_LM）；挂靠 Transformers 模型族 **1/374 @ cd74917f**（llama）、任务 **1/46**
（causal LM mapping）。这些轴不能相加，不代表整库支持。

## 可复用入口与独立复算

新增 `tools/run_peft_gate.py` 和维护入口 `nox -s ecosystem_peft`。
controller 必须完成六个 worker，并自动调用同一 strong collector 检查全部 **177** 数值字段、
输入扰动、冻结参数和来源绑定后才退出成功。预检 `--preflight` 只跑 base 两进程，明确不授 L3；
nox 拒绝透传 phase/preflight 等缩小范围参数，仅允许指定输出与缓存路径。

实际 formal-v5 使用下列公开入口，环境与完整命令保存在外置 `run-gate-v5.sh`：

```bash
export JITTOR_CI_PYTHON=<candidate-python> REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_PEFT_CHECKOUT=<clean-official-peft-checkout>
export JITTOR_TRANSFORMERS_CHECKOUT=<clean-official-transformers-checkout>
export JITTOR_PEFT_WHEEL=<official-peft-wheel>
export JITTOR_TRANSFORMERS_WHEEL=<official-transformers-wheel>
export JITTOR_TEST_DEVICES=cpu  # 完成 CPU 后以 cuda 独立运行
# CUDA 另显式分配 CUDA_VISIBLE_DEVICES 和真实 nvcc/provider；首次 JIT 串行。
<audit-python> -m nox -s ecosystem_peft -- \
  --output <new-external-output> --cache-root <private-benchmark-cache>
```

普通 `test_ecosystem_parity.py` 的 PEFT CPU/CUDA 节点依然是既有 base/一步 parity 门禁；
不能把它当成完整 L3 自动门禁。本次完整 L3 的实际入口是上述独立 nox session，
尚未声称 scheduled nightly 工作流已选择该新 session。

正式测量全部终态后，用独立进程重读 NPZ/JSON、完整源码、wheel 和来源记录：

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_peft_scorecards.py \
  --run "$PEFT_STATE/formal-v5/cpu" --run "$PEFT_STATE/formal-v5/cuda" \
  --peft-checkout "$JITTOR_PEFT_CHECKOUT" --transformers-checkout "$JITTOR_TRANSFORMERS_CHECKOUT" \
  --peft-wheel "$JITTOR_PEFT_WHEEL" --transformers-wheel "$JITTOR_TRANSFORMERS_WHEEL" \
  --baseline 1738f4a09d178becfad989f7bef7915729735890 \
  --report docs/results/2026-10-15-peft-installed-source-training.md \
  --output <new-external-audit-output>
```

`PEFT_STATE=$JITTOR_LAB_ROOT/_state/peft-source-bound/20261015`（未版本化）。
formal-v5 的 execution/validation、12 个实际 worker JSON/NPZ/log、独立 scorecard、
源/环境/库绑定和失败历史由该目录的最终 manifest 绑定；不将 raw 或环境放进仓库。

## 负例与保留边界

13 个 host 契约通过，覆盖非 Python payload/RECORD/metadata 错站、实际 observation 缺失、
完整范围和 nox 参数约束、数值错误导致 maintained controller 失败等。
另对真实 formal CPU 元数据/NPZ 的外置副本验证 **7 个拒绝场景**：缺 Transformers 源码、
缺 RECORD、缺 output residency、实际私有 HOME 偏离、梯度数值错误、缺请求梯度、冻结参数
只改 1 ULP（透明副本更新其自身 digest 以单独检验冻结不变量）。原始 raw 始终只读。

首次 formal-v1 在目录 symlink 的来源快照处报 `IsADirectoryError`，尚未计算；原 rc/源码/log 保留。
修为 target 文本加真实 compat 文件双绑定。preflight-v2 暴露旧 collector 对 RECORD/完整观测校验
不够，已以负例修复；base 输出实际 key 为 `__output__`，误用 `output` 的检查也已修正。
formal-v3 CPU 六 worker 与外部数值验收通过，但当时 controller 尚未自动验数，不以该旧 nox 退出码
授自动 L3。随后 ea897 加入自动检查；远端来源审计修复安全同步后重新进行同源 formal-v5。
所有旧数据保留原基线，不与本次统一。

本次只关闭这个 tiny LoRA L3 范围的来源与可复用完整训练门禁缺口。不授 L4 性能、adapter
保存/恢复、Trainer、BF16、分布式训练、量化、生成、其他 tuner 或真实大模型能力；
未验证 NPU/ROCm。依赖环境缺失、真实设备缺失和来源不符都应显式失败，不能用 import、
host 契约或 CPU fallback 充当数值验收。

## 主树整合复验

同步 `9ba68263b267c78799ec254e2b71abd865f41094` 后，代码与本页整合为
`508cdd07e67f0c5f1fbc6aa1411185a024a38068`。在该 clean 提交的独立冻结工作树、
全新私有 benchmark 缓存中重跑 CPU/CUDA 完整维护 nox：12 个真实 worker 全部成功，
每设备自动检查 177 数组；独立进程重算后仍为 2 accepted / 0 blocked / 0 rejected、
354 array pairs。误差与上表一致，两侧实际已映射 CUDA runtime 仍分别为 12080/12020，
候选全部阶段 fallback=0。每个 worker 绑定 1558 个运行源码，完整 Git 库存为 2838 文件。
这组新测量保留 508c 身份，不重标原 1738 记录，也不重复增加覆盖分子。

主树定向 77 项、core 322 passed / 42 skipped / 1 existing xfail 通过。
首次完整结构检查发现新 host 测试在收集时加载模块、直接引用解释器的两项违规；
修为共享延迟加载基类与标准 child-process helper。中间一次只给一个测试类延迟加载导致
三项 fixture 失败，原日志保留；最终测试与结构契约 55 项通过，完整结构
1622 passed / 6 skipped。六项跳过为两项加速器、两项声明与两项当前解释器未安装原生包。
这些修正只涉及测试，冻结测量工作树的运行工具与计算源码始终未变。

主树新测量、独立评分卡、完整源码与原始证据清单位于未版本化的
`$JITTOR_LAB_ROOT/_state/peft-main-integration-9ba/`；原始 1738 的证据保持只读，
另独立核对 235 原始/支撑文件、2375 实际安装文件/库及全部 2835 Git 文件的 mode/blob/哈希。
