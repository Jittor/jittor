# Transformers：全新的维护版 CPU nightly 验证

- 状态：完成维护版 CPU nightly profile 的本地执行；经独立审计。
- 复查日期：2026-10-16。
- 维护者：Torch 兼容性与下游验证维护者。
- 测量源码：`2f8895f1a656bcdd66c280dbdbe222eb8268f56e`。
- 环境搭建源码：`13b05c8b1951622ab12511e2936243086d5408ba`。
- 独立审计检出：`bc4dc0bc01a3930d3fad22326da0f9b8c70d444f`。
- 复查条件：依赖固定版本、provisioning、用例范围、源码/安装验证、精度、数值预算或训练构建器发生变化。

这是完整 `ecosystem_transformers_source` 会话的一次全新 CPU 执行。原始 nox 进程在全部十个
用例、四个阶段和 80 个 worker 完成后以零退出。其控制器在执行前后验证被测检出。独立的
只读审计重复每项原始数值断言、完整请求的梯度集合、初始权重绑定和三步轨迹。它既不改变
测量树，也不改变原始数组，并且不会把它们重新贴标到更晚的源码修订。

## 实际范围与结果

每个用例都运行种子 0、种子 7 与 19 的固定权重输入扰动，以及学习率 .001、输入种子 1/2/3
的三步 SGD。完整的参考参数与缓冲区被提供给两个运行时。base 与训练种子 0 的初始数组也
完全一致。原始 TF 的相位全局预算仍为绝对 .002 或缩放相对 .01；Longformer 保留其原有的
逐字段/梯度块归一化，前向 .002、反向 .01。

| 维护用例 | 参数 / 可训练 | 实际输入范围 | 数组对 | 最大缩放误差 |
| --- | --- | --- | --- | --- |
| BERT | 39 / 39 | token IDs `(2,8)` | 360 | 3.06e-7 |
| CLIP 多模态 | 78 / 78 | token IDs `(2,8)` 与像素 `(2,3,16,16)` | 717 | 7.27e-7 |
| GPT-2 | 28 / 28 | token IDs `(2,8)` | 261 | 3.20e-7 |
| Llama | 21 / 21 | token IDs `(2,8)` | 198 | 5.63e-7 |
| Llama RoPE | 12 / 12 | token IDs `(1,2048)` | 117 | 1.80e-6 |
| Mixtral MoE | 41 / 41 | token IDs `(2,32)` | 378 | 2.56e-7 |
| T5 | 47 / 47 | encoder/decoder token IDs `(2,8)` | 432 | 7.13e-7 |
| ViT | 40 / 40 | 像素 `(2,3,32,32)` | 375 | 5.17e-7 |
| Whisper | 89 / 88 | 特征 `(2,16,64)` 与 decoder IDs `(2,8)` | 813 | 3.83e-7 |
| Longformer | 29 / 29 | IDs `(1,32768)` 与嵌入噪声 `(1,32768,32)` | 276 | 9.85e-6 |

独立的 NumPy 计算与冻结的维护版比较器在全部 **3,927 个数组对**上逐字段一致。其中包含
2,541 个参数梯度对与 24 个输入梯度对；所有请求的梯度都存在、有限，且在两个运行时中
均非零。形状、dtype 和完整数组名集合一致。两种扰动在每个运行时中都改变输出。全部 2,538
次可训练参数更新都对照每个运行时保存的 SGD 梯度和前值进行检查；最大更新重放差异为
1.49e-8。Whisper 的那一个冻结参数在所有六份保存的训练快照中都逐位不变。

最大跨运行时绝对差异是 Longformer 中的 .0166015625。其原有的归一化规则通过；若改用另一个
仅绝对值的 .002 规则，将错误描述该契约。所有相位特定的误差都保留在独立审计中。这十个
CPU 场景在固定源码上自动重新导出的 **374** 个 `MODEL_MAPPING_NAMES` 模型中，覆盖九个
选定的已声明模型族。两个 Llama 场景计为一个模型族。这是有界的 L3 证据，不含整库声明，
也不是新的成对 CPU/CUDA 评分卡。

## 实际平台与源码权威

| 字段 | 记录的 CPU 执行 |
| --- | --- |
| OS / CPU | Linux 6.1.188，glibc 2.34；AMD EPYC 9554 |
| 加速器 / 卡数 | CPU 张量驻留；分配卡数为零 |
| 驱动 / CUDA / nvcc / CANN | 对本次 CPU 执行不适用 |
| 编译器 / 精度 | g++ 11.5.0；FP32；TF32 与 cuDNN benchmark 关闭；highest |
| 线程 / 图 | runtime 与 OMP/MKL/OpenBLAS 线程 2；候选默认图重放 |
| 独立 oracle | 独立私有前缀中的真正二进制 Torch 2.11.0+cpu |
| 运行时边界 | 候选回退策略 error，观察计数为零；各运行时 HOME/JIT/临时/缓存路径分离 |
| 速度 / 整进程内存 | 未为 L4 测量 |

全部 80 个实际 worker PID 各不相同。它们的 36,810 次张量观察都驻留在 CPU，具有原始的
输入/输出/参数/梯度形状和 FP32 计算。控制器记录空的 `CUDA_VISIBLE_DEVICES`，Git 绑定的
CPU 启动器显式将可见性置空并把两个 CUDA 标志置零。单个 worker 元数据不单独记录
`cuda.is_available()`；该回读既未被推断也未被回填。oracle provisioning 的健康检查另行
断言 CUDA 不可用。

控制器的完整 2,905 个受跟踪 Git blob 哈希（包括符号链接目标文本）与测量提交及未改动的
检出一致。每个 worker 的 1,575 个选定框架/探针源码哈希一致；该选定映射不是完整的 Python
导入清单。完整的历史 Git 归档另行保留。

干净的官方 Transformers 4.56.2 检出为
`cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`。其固定的官方 wheel SHA256 为
`79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`。实际导入包的完整
2,173 个文件在两个前缀中都与 wheel 和受跟踪源码一致。只有两个此前已声明的非运行时省略项，
即 GPT-2 的 `CONVERSION.md` 和 T5 的 `download_from_gcp.sh`，被排除在 wheel/源码映射之外。

既有的 TF profile 在真实的完整依赖安装之后，会有意恢复固定 purelib wheel 的 METADATA/
RECORD 字节。两个实际的 TF RECORD 文件都与归档收据相等；它们**不是未改动过的 pip 生成
RECORD**。另外，官方 Torch CPU wheel SHA256
`8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` 会对照全部 11,945 个
原始已安装归档文件复核（除自未哈希 RECORD 之外）：700,036,604 字节，包括 `torch/`、
`functorch/`、`torchgen/` 和 dist-info 载荷。Torch 实际的 pip RECORD 哈希与大小与每个
原始载荷文件一致。三个私有环境都独立通过 `pip check`；保留的 lock 和实际的发行版
METADATA 哈希描述了其他依赖，但不声明对每一个都做了完整的 wheel 载荷复核。

## 重放与证据

从干净的检出出发，使用健康的 Python 3.11 和新的外部 state 目录：

```bash
python tools/provision_ecosystem_nightly.py --library transformers \
  --python <healthy-python-3.11> \
  --state-root "$JITTOR_LAB_ROOT/_state/transformers-nightly/<new-run>" --run
```

这会准备分离的 controller/candidate/oracle 环境、解析完整依赖、检查它们，并运行完整的
维护版会话。只读的源码/wheel 缓存是可选的输入；JIT 缓存必须保持私有且绝不复制。该会话
只允许指定输出/缓存路径，因此十个用例或四个阶段无法被筛选掉。既有的[源码绑定训练证据](2026-10-15-transformers-installed-source-training.md)
保留其自身的历史身份。

原始未纳入版本控制的证据位于 `$JITTOR_LAB_ROOT/_state/transformers-nightly/20261016/`，
包括 `run-formal-v1.py`、`formal-controller-v1.json`、`formal-terminal-v1.json` 和
`fresh-v2/logs/027.log`。独立输出位于
`$JITTOR_LAB_ROOT/_state/transformers-nightly-independent/20261016/`：
`full-v2/numeric-audit.json`、逐字段比较、源码归档、完整 Torch 安装审计、平台/安装摘要
以及 `evidence-seal.json`。审计的第一次尝试错误地把既有后端字典当成字符串；该审计器
失败予以保留，其修正后的 v2 在不改变原始数据或预算的情况下通过。

本次执行在本地证明了维护版 CPU profile。它不证明 GitHub 的托管计划已执行，不新增 CUDA
测量，不升级历史 CUDA 记录，也不确立 L4/L5、完整模型族或未见后端的覆盖。

独立证据封存在整合期间再次重新哈希，绑定 572 个文件和 919,783,355 字节，SHA256
`bcc385d75c48872244df28ac5f7f0008bdbfaea1d8bfc85c792016ad19606112`。
