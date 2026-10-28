# 新进程中的 Accelerate checkpoint 恢复

- Status: 有界协议已验证；world-two CUDA 配对准入仍被阻塞
- Date: 2026-10-15
- Baseline commit: `93d2cfa7dd6c69ce6f5cb0e2d310dcf80f06e1e0`（干净的 final-v6 测量源）
- Upstream: Accelerate 1.10.1，官方干净提交 `623cc0ba588d192c77758939a6e340adf07edd4f`
- Owner: Jittor Torch 兼容维护者
- Review when: checkpoint、RNG、optimizer、scheduler、FSDP2、launcher、协议捕获或独立 auditor 归属变化；在其他硬件/工具链组合上重跑。

维护中的 Accelerate gate 现在由实际的 producer 进程保存，等待该进程与 launcher 终止后，再以新建/新 prepare 的 model、optimizer 与 scheduler 启动独立的 consumer。consumer 只通过 `Accelerator.load_state` 读取持久化文件；它不读取 producer 的观测值，也不接收活的 Python 对象。PID、nonce、单调的进程/launcher 边界、完整的 checkpoint 哈希以及两个阶段的原始数组均单独审计。

现有 fixture 与容差保持不变：world one 使用 Linear(3,2)、AdamW 加 StepLR；CUDA world-two 使用 FSDP2 FULL 状态，Linear(4,3)、AdamW 且无 scheduler，分 FP32 与 BF16 两组。每个 fixture 使用原始 seed-0 输入与 seed-7 扰动，保存前三次更新、续训两次更新。每个参数梯度与输入梯度都与独立的真实 Torch 进程比较，`atol=2e-6`、`rtol=2e-5`；恢复与自身 runtime 续训保持精确。保存前实际推进 Python MT、NumPy MT、Torch CPU 以及每个 CUDA RNG 状态，随后完整恢复，并验证后续样本。consumer 刻意使用不同的预加载 seed/state。每个可见 CUDA generator 在保存前与恢复后都被采样，包括 oracle 进程的非当前卡；推进按 generator 逐个检查，而不是只比较状态列表的汇总。

历史 b7 测量仍属同一 prepared 对象的证据，不因新的协议 ID `accelerate-checkpoint-fresh-process` 与 `accelerate-fsdp2-full-fresh-process` 而重标。CLI 默认仍支持 legacy 协议；维护中的 nox 默认选择 `--fresh-process`。

## 实际环境

oracle 是带二进制 `_C` 扩展的真实 Torch 2.11.0+cu128，与绑定源码的 Jittor Torch shim 分离。两者使用完全相同的干净 Accelerate 1.10.1 源码/包证明。CUDA 硬件为 NVIDIA RTX 4090 SM89，驱动 580.178.04。native provider 使用实际 NVCC 12.2.140 与十个真实固定版本的 CUDA 组件 wheel，其中包括 cuDNN 8.9.7.29 与 NCCL 2.18.3。这些 wheel 下载到隔离的任务前缀；官方 PyPI SHA256 与 wheel RECORD 校验验证实际安装的头文件/库。真实 Torch 环境保留其自身的 CUDA 12.8 组件栈。实际的逐进程已加载库观测可区分这两个栈；没有任何元数据或库被重标。

CPU 执行使用 AMD EPYC 9554，Linux 6.1/glibc 2.34，Python 3.11.16 与 GCC 11.5.0。CPU 与 CUDA 各自使用维护中的 nox 入口。native 双 rank CUDA 使用公开的 `python -m jittor.distributed.launch` 入口（等价于控制台 `jtrun`）；真实 Torch 使用其自身的 torchrun。串行暖机在正式启动前编译每个 rank 的真实 native 核心与 NCCL 扩展。每个 rank 拥有自己的 HOME/JITTOR_HOME/TMPDIR、XDG/shim 与扩展缓存，而通信器控制只在其实际启动范围内共享。

## 复现

准备独立的候选解释器与真实 Torch 解释器、官方固定版本的 Accelerate checkout 以及一致可用的 native CUDA 安装。通过现有 gate 输入设置解释器与 checkout：

```bash
export JITTOR_CI_PYTHON=<candidate-python>
export REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_ACCELERATE_CHECKOUT=<official-pinned-checkout>
export JITTOR_LAB_ROOT=<external-lab-root>
export JITTOR_CUDA_WHEEL_STRICT=1
JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_accelerate -- \
  --output <empty-external>/cpu --cache-root <private-benchmark-cache>
CUDA_VISIBLE_DEVICES=<allocated-gpus> JITTOR_TEST_DEVICES=cuda \
  python -m nox -s ecosystem_accelerate -- \
  --output <empty-external>/cuda --cache-root <private-benchmark-cache>
```

gate 会拒绝非空的 fresh-process 输出目录，包括部分失败的情况。失败的 producer 永远不会启动其 consumer。会话级显式的 strict provider/NVCC/NCCL 输入在 nox 隔离下仍然有效。测量与单测使用不同缓存；原始日志、checkpoint、组件 wheel 与编译产物均为外部且未版本化。

## 结果与限制

| 设备 | World | 精度 | 数组对 | 参数/输入梯度 | 最大绝对差 | 最大缩放差 | 配对准入 |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| cpu | 1 | no | 438 | 28/14 | 5.9604645e-08 | 1.7745026e-07 | L3 接受 |
| cuda | 1 | no | 438 | 28/14 | 2.9802322e-08 | 1.9864779e-07 | L3 接受 |
| cuda | 2 | bf16 | 820 | 56/28 | 1.4901161e-08 | 6.4892491e-08 | 阻塞：无 CPU world-two 对应项 |
| cuda | 2 | no | 820 | 56/28 | 1.4901161e-08 | 8.9234059e-08 | 阻塞：无 CPU world-two 对应项 |

公共 auditor 接受两个 world-one L3 记录，阻塞两个已独立验证的 CUDA world-two L3 记录，拒绝零个。它检查 2,516 对跨 runtime 数组、全部 168 个请求的参数梯度字段与 84 个输入梯度字段。全部 48 个实际阶段 worker 均完成，零跳过；24 个 native worker 报告 fallback 时出错且计数为零。真实 Torch 没有 Jittor fallback 计数器，记为不适用。共有 64 个持久化 checkpoint 文件。

BF16 FSDP2 用例观察到实际的 BF16 Linear 输入/权重/输出，同时保留 FP32 主参数与 optimizer 状态。每个自身 runtime 保存的 model、optimizer、存在的 scheduler、完整 RNG 状态与下一组样本/续训比较都精确；独立数值预算保持不变。native CUDA world two 使用公开 jtrun。该协议不使用较早的 CPU MPI 路径。

最终源码通过 60 个 host 准入契约，以及完整结构加这些契约：**1,568 passed, 6 skipped**。checkpoint-owner 的 CPU 选择在其独立单测解释器中通过 33 项测试、1 项可选的 Accelerate skip，新增的 rotation 选择通过 5 项测试。两个维护中的正式解释器都具备必需的 Accelerate，并执行所有请求节点而不跳过。CPU core 在未变的 checkpoint-owner 源码 `0823f271d` 上通过 **322**、跳过 **42**，并保留 **1 xfail**；后续变更只影响捕获/auditor 字段。七个实际原始产物的可丢弃副本被拒绝：相同 PID、变更时刻、错误的完整 RNG、初始偏移的 CPU RNG、非零 fallback、未激活的第二个 CUDA generator 以及重叠的 launcher 生命周期。历史 b7 重算以其原有的同对象语义保持 2 accepted/2 blocked/0 rejected。

外部未版本化证据位于 `$JITTOR_LAB_ROOT/_state/accelerate-fresh-process/20261015/`：`formal-v6/{cpu,cuda}`、`final-scorecards-v6`、`preflight-v6a`、`provider-watch-v6.jsonl` 与 `final-audit-negative-controls-v6.json`。完整的 `final-v6-evidence.json` SHA256 为 `8646545e61bcda2d7df7bd4235fa1360f5fdf514951d128fd03a4fffd32b0591`；它绑定 1,567 个源码文件、342 个原始文件、全部 48 个阶段身份与 38 个实际映射的库文件。每个 CUDA 阶段 PID 都有实际的 map 观测。十六个 native NVIDIA 库载荷与官方 wheel RECORD 匹配。所有 native 阶段元数据都报告解析后的精确 wheel pin。

native 实际 runtime/NCCL 版本为 12020/21803，真实 Torch 版本为 12080/22809。launcher 与串行暖机 worker 还映射外部 NVCC 工具链单独提供的 `libcudart.so.12.2.140`（API 版本同为 12020）；48 个正式阶段 PID 均未映射该额外文件。其实际路径/哈希单独保留，且不被标记为 wheel 载荷。

world-one CPU 与对应 CUDA 记录可被公共全语义协议配对准入。CUDA world-two FP32/BF16 没有同语义的 CPU FSDP2 对应项，因此必须保持在配对准入的阻塞状态，即使独立 CUDA 一致与严格的 fresh-process 恢复都成功。CPU world one 不能替代 CPU world two。这些都是有界协议记录，不构成 registry 分母或整个 Accelerate 覆盖的声明。本工作不建立 L4 速度/内存结论，也没有 NPU/ROCm 证据。

在包含本报告与相关代码引用的隔离 checkout 中，配合已准备的 NumPy/stdlib 审计依赖，重算独立一致性：

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_accelerate_scorecards.py \
  --run <run-state>/formal-v6/cpu --run <run-state>/formal-v6/cuda \
  --upstream <official-pinned-checkout> \
  --baseline 93d2cfa7dd6c69ce6f5cb0e2d310dcf80f06e1e0 \
  --report docs/results/2026-10-15-accelerate-fresh-process.md \
  --command "independent raw recheck" --output <new-external-audit-output>
```

## 保留的失败与范围变化

- formal-v2 的 native CUDA 在训练前失败，因为 nox 丢弃了调用方的 strict wheel 选项，较旧的候选环境选择了 CUDA 12.8 元数据而非 native cu122。真实 Torch 侧的成功运行不使该配置获得准入。
- formal-v3 的张量测试暴露了继承的 nox XDG/shim 缓存路由：preflight 将 native HOME 改写为共享的临时会话目录。严格的外部 auditor 拒绝了该证据。后续源码只更改 benchmark 环境路由，失败的原始文件保留。

- 最初的 v3 双 rank consumer 还暴露了 DCP 初始化缺陷：零学习率的 AdamW 初始化步骤会轮换 native FSDP 公共分片视图，使该步骤之前缓存的 FQN 身份映射失效。checkpoint owner 现在在初始化与状态提取之后解析名称。本地实际张量轮换测试保留权重、零 moments、step 与学习率；最终的双 rank 公共 API 运行验证真实 FSDP 用例。
- v4 已通过所有有界张量/严格恢复场景，但 oracle 非当前的可见 CUDA generator 仍停留在初始偏移。v5 增加实际的按设备推进与后续样本验证；v4 不被重标为这一更强的协议。

- v5 通过有界数学与恢复检查，但其包清单理解让后续被遮蔽的 site 覆盖了实际选中的 CUDA 发行版本。现在的捕获为每个包记录 `importlib.metadata.version(name)`，并有真实的竞争 `dist-info` 优先级测试。六进程 preflight 在冻结 final v6 之前检查了两个环境、真实映射库与 wheel RECORD 的一致性、私有 HOME/XDG 路径与所有 RNG 流。v5 元数据保持不变，不构成 provider 来源一致的证据。

所有较早阶段与失败配置保留各自的源码身份。最终证据只绑定明确列出的最终干净源码与终态运行。

## 后续：固定版本上游的 CPU world-two FSDP2 选择

这项单独的源码与 runtime 探针不改变上述 v6 测量 SHA、原始元数据、数组计数或准入结果。它考察干净官方
Accelerate 1.10.1 于 `623cc0ba588d192c77758939a6e340adf07edd4f` 的公开 CPU 对应路径，使用真实二进制
PyTorch `2.11.0+cu128`，CUDA 可见性为空。每种精度用两个真实 `torchrun`/Gloo 进程运行
`Accelerator(cpu=True, fsdp_plugin=FullyShardedDataParallelPlugin(fsdp_version=2,
state_dict_type='FULL_STATE_DICT', auto_wrap_policy='NO_WRAP'))` 并 prepare 一个真实 Linear。

上游 `state.py` 在第 795/808 行选择 `MULTI_CPU` 与 Gloo。其在第 976–1005 行的 FSDP 提升位于仅限加速器的
分布式类型分支内；`MULTI_CPU` 则进入第 1014 行的独立分支。第 1098 行的 `is_fsdp2` 要求 `DistributedType.FSDP`。
对精度 `no` 与 `bf16`，全部四个实际 worker 都报告 `MULTI_CPU`、`is_fsdp2=False`、状态中没有 `fsdp_plugin`，
并得到一个实际的 `torch.nn.parallel.distributed.DistributedDataParallel` prepared model。其 CPU 参数保留
完整的 `(3, 4)` / `(3,)` 形状且没有 `to_local`；真实 backward 完成。因此传入 FSDP2 plugin 并不能通过这一
固定版本的公开选择路径建立 CPU FSDP2。

该探针既不主张未来所有 Accelerate 版本或直接 Torch FSDP API 都缺少 CPU 支持，也不以这一 DDP 结果替代
维护中的 CUDA FSDP2 协议。CPU world one 与 CPU world-two DDP 都无法满足其同语义的 CPU world-two FSDP2
配对；两个 CUDA 行保持阻塞。没有更改候选 launcher、核心实现或 auditor 配对规则。未使用 GPU 或 Jittor JIT。

外部证据位于
`$JITTOR_LAB_ROOT/_state/legacy-scorecard-source-audit/20261015/cpu-fsdp2/`。
`manifest.json` 的 SHA256 `4df33e8b247b5b6696fa0b948b23474fdaa1c71b0390ab1b357c9808fcddf279`
绑定全部四个 PID/rank 结果 JSON 文件、两份终端日志、精确的可执行文件/环境命令、该探针、官方上游源码文件
以及真实 Torch 的 `_C` 二进制/导入文件。这是 provider 选择的否定结果，不是新的训练或 fresh 恢复 scorecard。

## 主树整合与空 CUDA 状态集合

主树同步 `8bb59861de5bbf98954728a331a4efe42ea8a389` 后整合上述代码，另以干净
`0fe68129f09dea2f4b5585dde52dd149bb1db4b5` 运行完整维护 nox CPU→CUDA 协议。
两个 rank 的首次核心/扩展 JIT 串行完成。CPU 四组、CUDA 十二组与全部 48 个真实
阶段 worker 均成功；producer 全部退出后才启动 consumer。独立重算仍为 2,516
数组、168 参数梯度和 84 输入梯度，原预算未变；两个单 rank L3 接受，两个双 rank
CUDA 记录仍因上述 CPU FSDP2 限制而不授配对通过。所有 2,826 个 Git 文件（包含
目录符号链接的目标文本）、64 个 checkpoint 文件与正式 CUDA worker 的真实库映射
均另行绑定。历史 93d 测量保持原身份，不重标为主树结果。

该运行完全终态后，另修复 CPU-only 的空 CUDA 状态集合：无可见 CUDA 设备的
`torch.cuda.get_rng_state_all()` 返回 `[]`；`set_rng_state_all([])` 与空 tuple
均正常返回，不初始化 CUDA，不修改 CPU/Python/NumPy 流。此前候选三个契约全部失败，
同一测试在独立 binary Torch 上三个全部通过。修复只在零设备或空集合时提前返回，
有状态的解析与 native 恢复继续由原 owner 执行。标量 CUDA state API 与无设备的
非空集合不因此获得新的支持声明。

修复后定向 CPU **43 passed / 19 skipped**，实际双 CUDA **59 passed / 3 skipped**；
CUDA 运行的三个 skip 是明确只适用于无可见 CUDA 进程的空集合节点。现有真实设备状态、
全位宽播种与默认 Generator 隔离回归均实际执行。此空集合改动未参与前述 0fe 正式测量，
不把旧测量重新标为最终提交。

主树未版本化证据目录：
`$JITTOR_LAB_ROOT/_state/accelerate-fresh-integration-8bb/`。`formal/{cpu,cuda}`、
`main-scorecards`、`measured-main-source.json`、`provider-watch.jsonl` 和
`sealed-main-evidence.json` 保留完整测量身份；`integration-manifest.json` 另记录最终
core/结构/工具检查、冻结文件与最终提交的 Git blob 映射。三个 `source-suite` 入口是
维护 gate 指向真实测试文件的符号链接，绑定目标文本并逐字节匹配 0fe 的 Git 内容，
不冒充独立复制的测量文件。旧 unit 解释器缺失完整 safetensors 的导入失败保留；
依赖完整的私有解释器定向 72 项通过。首次封装工具拒绝这些源码链接及后续自身日志
引用的问题已修正，旧草稿保留，不改变原始训练数组或数值准入。
