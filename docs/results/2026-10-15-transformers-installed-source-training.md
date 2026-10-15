# Transformers：完整安装来源与原训练矩阵复测

- 状态：CPU ①与真实 CUDA ③原完整范围通过；独立统一评分 20 accepted、0 blocked、0 rejected。
- 日期：2026-10-15。
- 实测源码：`b4e46151dff102b0fe92697d7c1a7e53acaf7a3b`，全部正式进程同一 clean 提交。
- 开工整合远端：`9ba68263b267c78799ec254e2b71abd865f41094`，另明确合入已完成的 PEFT 工具作为前置。
- 维护者：Jittor Torch compatibility maintainers。
- 复查条件：固定上游或 wheel、模型构造器、共享 runner/training probe、计算实现、来源采集或评分器变化。

本次对原 TF18 与 Longformer2 的 **10 个 case × CPU/CUDA × 四阶段 × 两个独立 runtime**
重新测量，补齐实际安装包与上游源码绑定。保留原模型、全部参数和浮点输入梯度、两次输入扰动、
三步训练及数值预算；没有修改模型数学、共享 runner、兼容计算或新增 adapter。
旧二十行缺少完整的当时安装来源，当前评分器拒绝其新来源准入的事实已独立封存；旧数值证据不被
改写或倒填。本报告只描述新的正式进程和原始产物。

## 固定来源与实际平台

| 项目 | 实际身份 |
| --- | --- |
| Transformers | 官方 4.56.2，clean 源码 `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| 独立 oracle | Python 3.11、NumPy 1.26.4、完整官方二进制 Torch `2.11.0+cu128`，私有 prefix |
| CPU | AMD EPYC 9554，Linux/glibc 2.34；双方显式 CPU，候选关闭 CUDA/oneDNN/MPI/NCCL |
| CUDA | 一张 NVIDIA GeForce RTX 4090，driver 580.178.04；正式 80 个进程使用同一物理卡 |
| 编译与精度 | GCC 11.5.0、候选 NVCC 12.2.140；请求 FP32，CUDA 两侧开启 TF32 matmul/cuDNN、关闭 benchmark；CPU 关闭 TF32 |
| 隔离 | 双方各 2 线程，runtime × device 各自 HOME/JITTOR_HOME/TMPDIR/XDG/shim cache；不共用单测缓存 |

每个正式进程在计算前后核对实际 import 根、首选发行物、官方 METADATA/RECORD 和全部
**2173 个 Transformers payload 文件**，含 2147 Python 与 C++/CUDA/头文件等。
上游 package 有 2175 文件：GPT-2 `CONVERSION.md` 和 T5 `download_from_gcp.sh`
未进入官方 wheel，库存明确记录这两个非运行文件，未伪记为已安装。
每个 worker 绑定 **1562 个框架/兼容层/相关工具源码哈希**；另记录实测提交全部 **2842 个
Git 文件**的 mode/blob/字节哈希，符号链接按 target 文本处理，并绑定实际 compat owner。

官方 Transformers wheel SHA256：
`79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`。
oracle 使用完整官方 Torch wheel，SHA256：
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`，
逐一验证全部 **11842 个 payload 文件**、RECORD 哈希和大小，并再次检查实际私有 prefix。
实际 `_C` SHA256 为 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
oracle 无 shim marker；没有改 Torch wheel 字节或伪造版本。

实际已映射 `libcudart` 的 API 查询是候选 **12020**、oracle **12080**，分别保留自己的
真实路径、哈希及首选 NVIDIA metadata。候选 cuBLAS/runtime/cuDNN metadata 为
12.2.5.6 / 12.2.140 / 8.9.7.29，oracle 为 12.8.4.1 / 12.8.90 / 9.19.0.56。
私有 oracle 采用标准相邻 NVIDIA 依赖布局，只读引用真实 cu128 namespace 与真实发行 metadata，
没有把 cu128 重标为 cu122。另绑定候选实际 NVCC/G++、关键头文件、runtime 及 metadata/RECORD。
库被映射不等于模型使用该库全部计算能力；本次无通信、NCCL 或 jtrun 能力声明。

## 原完整数值范围

每设备四阶段：base seed 0、输入扰动 seed 7、输入扰动 seed 19、三步训练；每阶段两个真实进程。
双方加载完整 oracle 初始参数与 buffer。额外独立核对 CPU/CUDA 的 **20 套初始权重、870 个
参数/buffer 数组逐值字节相同**，全部模型配置、输入规格及阶段 seed 一致。

| Case | 参数 / 可训练 / 冻结 | base 与每组扰动数组 | 三步数组 | 每设备数组总数 |
| --- | ---: | ---: | ---: | ---: |
| BERT | 39 / 39 / 0 | 40 | 240 | 360 |
| CLIP 多模态 | 78 / 78 / 0 | 80 | 477 | 717 |
| GPT-2 | 28 / 28 / 0 | 29 | 174 | 261 |
| Llama | 21 / 21 / 0 | 22 | 132 | 198 |
| Llama RoPE 2048 | 12 / 12 / 0 | 13 | 78 | 117 |
| Mixtral MoE | 41 / 41 / 0 | 42 | 252 | 378 |
| T5 | 47 / 47 / 0 | 48 | 288 | 432 |
| ViT | 40 / 40 / 0 | 42 | 249 | 375 |
| Whisper | 89 / 88 / 1 | 91 | 540 | 813 |
| Longformer 32768 | 29 / 29 / 0 | 31 | 183 | 276 |
| 合计 | | | | **3927** |

RoPE 保留真实长度 **2048**、max position 512、linear factor 4；CLIP 包含两个塔及像素输入梯度；
Mixtral 保留四专家/top-2、全部专家与路由器梯度；Whisper base/扰动请求全部 89 参数梯度，
训练保持 88 可训练及一个冻结位置嵌入，并检查特征输入梯度。
Longformer 保留 `[1,32768]` token、`[1,32768,32]` 嵌入噪声及其梯度、window 256、
global attention、全部 29 参数梯度，实际 primary output 有 **1048608** 个元素。
没有用短序列替代它。

复用公开 `training_probe.py`：三步手动 SGD、learning rate 0.001、原固定完整权重、
step input seeds 1/2/3。每步比较输出、loss、全部可训练参数梯度、全部浮点输入梯度和全部参数；
冻结参数必须逐值字节不变，可训练参数必须实际更新，两组扰动必须实际改变双方输出。
另对两个设备各 32 份实际特殊场景产物检查四专家非零梯度、Longformer global q/k/v 与噪声梯度、
浮点输入梯度形状和 Whisper 冻结不变量；没有增加 worker 或覆盖分子。

## 原预算与结果

TF9 保留原**整阶段参考数组最大量级**归一化：`max_abs <= atol` 或
`max_abs / phase_reference_max <= rtol`。base 两设备均为 0.002/0.01；CPU 扰动/训练仍为
0.002/0.01，CUDA 扰动/训练仍为 0.005/0.02。这不是逐元素相对误差声明。

Longformer 保留原每数组规则：梯度/输入梯度以
`max(reference_array_max, 0.001 * step_gradient_block_max, 1e-6)` 归一化；输出、loss、参数
使用自身数组量级。CPU forward/backward 限制 0.002/0.01，CUDA 0.005/0.02，均保留原严格
小于判据。巨大输出不能掩盖小梯度。新比较器另只读复算全部 **7854 个历史数组对**确认预算未变，
这些旧数组不算本次新来源测量。

| 正式设备 | Cases / 独立进程 / 数组对 | TF9 最大原全阶段归一化误差 | Longformer 最大原数组/梯度块归一化误差 | 候选 fallback |
| --- | ---: | ---: | ---: | ---: |
| CPU ① | 10 / 80 / 3927 | 1.7956447e-6 | 9.8483680e-6 | error / 0 |
| CUDA ③ | 10 / 80 / 3927 | 1.1524663e-2 | 2.7364785e-5 | error / 0 |

CUDA 最大 TF9 差异在 **T5 seed 19 的 `decoder.block.0.layer.2.DenseReluDense.wi.weight`
梯度**：最大绝对误差 **0.3834403**、原阶段量级归一化 **1.1524663e-2**，在原该阶段
**2e-2** 预算内。保留该实际精度边界，不通过更换输入、关闭 TF32 或放宽预算减小它。
CPU/CUDA Longformer 最大绝对误差分别 **0.0166015625 / 0.060546875**；其归一化规则与 TF9 不同。
CUDA 检查实际 input/output/parameter/全部请求 gradient 的驻留、driver 指针属性与 UUID；
所有正式阶段同一物理卡。oracle 的 fallback 字段是旧 runner 占位，不当作 Jittor fallback 度量。

独立 collector/统一评分器重新读取全部产物后：**20 accepted、0 blocked、0 rejected，7854 数组对**。
完整官方 Torch 的 11842 个文件在正式运行后再次全量核验通过。两个正式维护 nox 均已退出成功，CPU/CUDA
完整运行分别约 33/40 分钟，包含来源核对与首次 JIT；原始逐 worker timing 保留，仅作诊断，
不据此授 L4 或性能比值。

固定源码的覆盖轴仍分别为 **9/374 模型族、2/46 task mappings @ cd74917f**；
两个 Llama 场景只计一个模型族。Longformer 保留原 broad，CLIP/MoE/RoPE 保留 tail。
来源证明更完整不意味着新增模型或整库能力。

## 可维护入口与独立复算

新增 `tools/run_transformers_gate.py`、专用 collector/helper 和
`nox -s ecosystem_transformers_source`，复用既有 case/runner/train 数学。
维护入口强制全部十 case、四阶段、80 个进程；每 case 自动检查完整来源、权重、观测、扰动、
冻结参数和全部数值，worker rc0 不能单独通过。非空 output 拒绝复用；`--preflight` 只是两个
BERT base 进程，不授 L3。nox 只允许透传 output/cache 路径，拒绝缩小 case/phase。

```bash
export JITTOR_CI_PYTHON=<candidate-python> REAL_TORCH_PYTHON=<genuine-binary-torch-python>
export JITTOR_TRANSFORMERS_CHECKOUT=<clean-official-transformers-checkout>
export JITTOR_TRANSFORMERS_WHEEL=<official-transformers-wheel>
export JITTOR_TEST_DEVICES=cpu
<audit-python> -m nox -s ecosystem_transformers_source -- \
  --output <new-external-cpu-output> --cache-root <private-benchmark-cache>
# CPU 完成后，显式分配一张真实 CUDA 卡和实际 nvcc/provider，使用新的 output。
export JITTOR_TEST_DEVICES=cuda
<audit-python> -m nox -s ecosystem_transformers_source -- \
  --output <new-external-cuda-output> --cache-root <private-benchmark-cache>
```

原 `test_ecosystem_parity.py` 与 `test_ecosystem_longformer.py` 节点保留自己的既有 parity 范围；
不能将其当作上述完整 L3 自动门禁。本次没有声称 scheduled nightly 已选择新 nox session。
独立复算命令：

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_transformers_scorecards.py \
  --run "$TF_STATE/formal-v5/cpu" --run "$TF_STATE/formal-v5/cuda" \
  --transformers-checkout "$JITTOR_TRANSFORMERS_CHECKOUT" \
  --transformers-wheel "$JITTOR_TRANSFORMERS_WHEEL" \
  --baseline b4e46151dff102b0fe92697d7c1a7e53acaf7a3b \
  --report docs/results/2026-10-15-transformers-installed-source-training.md \
  --output <new-external-audit-output>
```

`TF_STATE=$JITTOR_LAB_ROOT/_state/transformers-source-bound/20261015`（未版本化）。
正式 `formal-v5/{cpu,cuda}`、独立 `independent-scorecard-v5`、源码/安装包/库绑定、失败历史和
最终 `evidence-manifest-v5.json` 均在此处；canonical 环境入口是 `run-formal-cpu-v5.sh` 与
`run-formal-cuda-v5.sh`，独立复算封装为 `recheck-v5.sh`。
原始日志、二进制和缓存不进入仓库。

## 反例、失败历史与边界

12 个 host 契约通过，覆盖两种数值归一化、Longformer 梯度 floor、真实来源路径、私有缓存/设备
路由，以及数值错误导致 controller 失败；不当作真实 L3 证据。
另以实际 metadata 副本验证九个拒绝反例：缺像素梯度、缺 output、缺完整安装来源、错误 RECORD、
非零 fallback、HOME 偏离、seed19 缺安装哈希、seed19 精度改变、训练源码哈希改变。
独立专用 collector 核验 base/seed7/seed19/train 的每个真实进程，未依赖旧中央评分器额外扰动
role 尚未核验的漏洞。

preflight-v1 在额外平台 helper 的来源路径处失败，尚未计算；修复后加直接文件路径契约。
v3 CPU 完整矩阵和 CUDA base 预检通过，但随后首个正式 CUDA oracle 因共享 Torch 符号链接
target 缺 `compile_time_profiler.py` 导入失败。保留原失败、目录与链接身份，不推断是谁改动。
新完整官方 Torch 私有 prefix 首次导入又暴露 ELF 查找落入旧 CUDA runtime；保留该 v4
undefined-symbol 失败和 loader trace。采用真实相邻 cu128 依赖布局后，以新 v5 输出完整重跑
CPU→CUDA，未改 Torch wheel 文件，也未将旧 CPU 数字重标为 v5。
初选卡在预检后仅剩约 3.56 GiB 显存，协调另一空闲卡重新预检后才启动正式 CUDA；
所有 80 个正式 CUDA 进程始终同卡。可选 gdb/路径诊断保留在 raw，成功以实际 rc/数值为准。

本次只关闭原十场景 FP32 L3 的来源与完整维护门禁缺口，不授 Trainer、生成、optimizer 类、
checkpoint/resume、完整预训练家族、BF16、多 rank、NPU/ROCm 或 L4 性能能力。
