# DeepSpeed 配对 BF16 Engine 协议

- Status: 已验证有界的 CPU/CUDA 配对 L3 协议；不作 registry 覆盖主张
- Date: 2026-10-15
- Measurement commit: `f787cc0b34b8b24e04d79cc55aacf746c01036b8`，每个用例前后均干净
- Upstream: 官方 DeepSpeed `f5af15c2b4da10411e340850c00ba0cb5b14fd87`
- Installed version: `0.19.7+f5af15c`；保留开发版本
- Owner: Jittor 核心 / Torch 兼容维护者
- Review when: Engine/ZeRO 源码、adapter pin、native autograd/存储/optimizer、
  BF16 Linear/LayerNorm、进程组、公开 launcher 或协议校验器变化

## 验证范围

这是一个有界的训练/checkpoint 协议，没有 registry 分母，也不作性能主张。
两个设备使用相同的残差 Linear/GELU/LayerNorm 模型、332 个可训练 BF16 参数元素、
两个可微输入、world size 二、ZeRO1/2、AdamW、GAS 二、六个 microstep 与三次更新。
两个原始输入偏移 `0` 与 `.125` 都执行。每个 rank 的输入不同，并拥有真实的 166 元素
FP32 主 optimizer 分区。在第四个 microstep 之后，新 Engine 加载保存的
checkpoint 并重复 microstep 四/五。所有观测到的 optimizer 状态、主
分区、输出、loss、梯度与更新后的模型参数都必须与其
未中断运行逐位一致。

专用的 CPU 对应项使用实际的 OpenMPI 5.0.11；独立 PyTorch 使用
Gloo。这证明了该协议下的 CPU MPI 后端。它**不**确立
native Gloo 或 CPU `jtrun` 支持，也不推荐其他训练 launcher。
公开的 `jtrun` owner 当前支持 NCCL/HCCL。CUDA 候选训练使用
`python -m jittor.distributed.launch`（等价于控制台入口）与实际
NCCL；独立二进制 PyTorch 在独立进程中使用 NCCL。

每个候选与 oracle 进程实际设置 seed 零，然后用原始确定性 fixture
完整覆盖权重。`.125` 输入偏移不会被
重命名为不同的 RNG seed。快照在 DeepSpeed 转换之后、训练之前观测实际的初始 BF16
训练权重；全部 332 个值精确匹配。
临时 FP32 fixture 在 BF16 转换之前存在既有的一个 ULP 融合/分离舍入差异
（至多 `1.49e-8`）。转换前的失败哈希诊断
被保留；改变快照位置不改变任何模型数学或权重。

每个 microstep 记录实际输入/目标与全部传入参数与输入
梯度。ZeRO1 的公开完整梯度快捷方式返回 BF16；ZeRO2 在
GAS 边界的完整 getter 返回 FP32 累加存储。梯度 dtype 与完整预期的
schema 独立于跨 runtime 键相等性进行检查。初始 BF16 输出
可能因 LayerNorm 而对加性扰动不敏感；后续输出、
每个 microstep 的 loss 与梯度都会响应实际扰动。校验器
检查完整训练轨迹，而不改变原始输入。

## 结果与平台

| 设备 / stage | 比较的数值张量 | 参数 / 输入梯度张量 | 逐位梯度 | 最大缩放前向/状态误差 | 最大缩放梯度误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / ZeRO1 | 1056 | 512 / 64 | 569 / 576 | 1.37151e-5 | 5.67769e-4 |
| CPU / ZeRO2 | 928 | 384 / 64 | 428 / 448 | 5.49490e-5 | 3.33526e-4 |
| CUDA / ZeRO1 | 1056 | 512 / 64 | 576 / 576 | 1.75847e-7 | 0 |
| CUDA / ZeRO2 | 928 | 384 / 64 | 448 / 448 | 1.75847e-7 | 0 |

计数对两个扰动与两个 rank 相对于独立 oracle 进行汇总。
原有的共享预算保持不变：forward/state 为 `.002`，梯度为 `.01`，
使用各类别原有的比较下限。CPU 一致性为近似；在该固定矩阵中
CUDA 梯度逐位相等。每个进程通过 68
（ZeRO1）或 60（ZeRO2）个严格续训字段，并更新全部 166 个主元素。
所有候选运行都在禁止 fallback 下执行，fallback 增量为零，
所请求设备的输入、参数、输出与梯度均已物化。

主机为 Linux x86-64、AMD EPYC 9554、GCC 11.5.0 与 Python 3.11。native CPU 使用
oneDNN 3.9.1。CUDA 使用两张分配的 RTX 4090，SM89，驱动 580.178.04 与
实际的 native pip-cu122/nvcc 12.2.140 provider。oracle 是独立加载的
PyTorch `2.11.0+cu128`，带真实 `_C` 扩展与 SHA256。CUDA 精度开关
报告 TF32 matmul/cuDNN 启用、cuDNN benchmark 禁用、float32 matmul 为 `high`；
CPU 报告 TF32 禁用与 `highest`。这些是记录的实际设置，不是
对无关 FP32 负载的主张。逐进程 manifest 保留 GPU 驱动 UUID、
分配、编译器输出与已安装 NVIDIA wheel 元数据；仅凭已安装元数据
不被视为已加载 CUDA provider。

两个 rank 共享用例/runtime 的通信器控制目录，而 HOME、
JITTOR_HOME 与 TMPDIR 保持私有。首次核心/扩展构建串行。
CPU 后端检查显式保留父进程的 CPU affinity，而不是
OpenMPI 默认的单核绑定。CUDA 的 rank/设备/控制分配来自
公开的 `jtrun`；`-S` wrapper 在 site hook 之前选择缓存，然后 exec
正常解释器。rank 启动命令与放置控制作为原始数据被哈希。

## 源码与证据

官方 wheel 由干净的分离上游 checkout 独立重建，
使用二进制 Torch、`DS_BUILD_OPS=0` 与真实的 CPU 加速器选择。
原始与重建的 wheel 包含完全相同的 1002 个包文件：1001 个
受跟踪源码载荷加上上游 `setup.py` 生成的元数据。每个包
字节（包括生成的 SCM 版本/已安装算子/构建 Torch 元数据）都一致。
原始 wheel 的 SHA256 为
`467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`。

源码描述符展开 accelerator、op_builder 与 csrc 的官方 Git 目录
符号链接。仅 Python 的源码列表会漏掉真实的 JIT 扩展
源码。它记录 34 个被普通包数据规则排除的官方文件
（docs、非包的 AIO/GDS 开发者测试与一个 MPS `.metal` 文件）；这并不表示
支持 MPS。缺失或被修改的 C++/Python 载荷、多余文件、可执行的生成
元数据、脏源码与非官方远端都会被拒绝。独立重建的
wheel 字节与构建日志保留在 Git 之外；不会从已安装的依赖元数据
制造新的包版本。

原始数据未版本化，位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`：
`protocol-formal-v1/{cpu,cuda}/zero{1,2}/AdamW/bfloat16/world2/perturb-*` 包含
实际的 NPZ/日志/launcher/协议/比较产物与完整源码 manifest。
每个 manifest 绑定全部 1574 个受跟踪的 native/compat/backend/adapter/nox
runtime 文件。历史失败诊断与旧 CUDA 证据保留其原始基线；
它们不被重标为这次干净公开 launcher 测量。

使用已安装的独立解释器、显式的
oneDNN/工具链路径与全新的外部产物目录，按
[DeepSpeed runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/deepspeed-torch-compat/SKILL.md)所述复现专用门禁。CUDA 需要
显式分配的两卡 `CUDA_VISIBLE_DEVICES`。CPU MPI 后端验证
还需要真实的 OpenMPI 工具链；它不是 CPU `jtrun` 命令。
两种情况下维护入口都是 `nox -s ecosystem_deepspeed_multirank`。

`collect_deepspeed_scorecard.py` 重查每个实际的 rank/场景产物并
调用公共的 `audit_coverage.scorecard` 协议分发器。协议记录使用
`deepspeed-zero-training`、空的 registry 轴/引用，并在 CPU/CUDA 上使用相同的完整语义
描述符；MPI/Gloo/NCCL 与 launcher 留在平台元数据中。
缺失精确的 CPU 对应项必须阻塞 CUDA 准入。L4/L5 与整库
支持百分比都不从这些极小的 L3 协议测量中推断。
位于 `5cd468ac4f4f678cd0f73e62309a0ef348558fd2` 的实际公共审计接受了四个
记录，拒绝零个、阻塞零个：CPU 与 CUDA 的 ZeRO1/2 均为 L3，两个 CUDA 记录都有精确
同语义的 CPU 对应项。registry 覆盖数组为
空。产物位于外部状态根下的 `paired-scorecard-v1/{evidence,scorecard}.json`；
完整的源码/原始/checkpoint/日志封存为
`paired-protocol-v1-evidence.json`。

在干净 `c93eb2c0690b9541063330685fb7ac6560eb8812` 上的最终公共审计
独立重新生成同样的四个接受记录、零拒绝/阻塞，
并在 `paired-scorecard-v2/{evidence,scorecard}.json` 中给出空的 registry 覆盖。
测量源码仍为 `f787cc0b34b8b24e04d79cc55aacf746c01036b8`。
不可变的最终证据封存的 SHA256 为
`f008faff4bfaa4cfd540b62067d69ec031b98749ac9ab04984bf52e3b6ae52cc`；
它绑定 829 个原始文件、1574 个被测 runtime 源码哈希、全部 32 个正式 NPZ 文件
与 48 个实际 checkpoint 文件。它还保留原始诊断、
独立 wheel 重建、两次公共审计与全部十一个负向控制。

在实际 NPZ/门禁产物的可丢弃副本上，九项破坏性测试在哈希重新绑定后
被公共 auditor 拒绝：两个 runtime 同时丢弃一个梯度、一个 ULP 的恢复 moment
差异、错误的完整梯度 dtype、
fallback、oracle shim 身份、CPU 后端错标、交换 CUDA UUID、私有
CUDA launcher 与 stage 错标。因此拒绝的原因不是普通的哈希不匹配。
另外，CUDA 缺少 CPU 或只有不同阶段的 CPU
记录时保持阻塞，准入层级为 null 并保留独立 L3。

## 回归边界

源码封存之前：native/Torch core `322 passed / 42 skipped / 1 existing xfail`，
25 项 launch/control 契约与 30 项定向结构检查通过。完整结构
运行有 1502 通过与六项跳过：两项因禁用插件自动加载导致的 xdist 失败
用已声明的插件重跑成功；两项 MPI/math 报告登记
失败由后续上游整合解决。两份原始日志都保留。
在干净的测量源码上，专用 nox CPU 与 CUDA 各通过全部四个用例
且无跳过。launch/scorecard host 测试通过 46 例；加上共享的
Accelerate 协议回归，78 个 host 用例通过。Ruff/layout 通过。
在上游同步并加入本报告之后，完整结构有 1504 通过与六项
跳过；其两项失败是本新报告的中心索引/MANIFEST 登记，
仍属整合 owner 的责任。没有数值测试因这些文档失败而被跳过
或给予更弱的阈值。

第一次扩大的单 rank FP32 回归暴露了一个过宽的新传输
断言：native singleton WORLD API 在没有活跃 MPI/NCCL 通信器时
仍保留其 legacy 默认 `mpi` 标签。CPU 与 CUDA 各在执行训练/严格续训之后
于该断言处失败 12 例。比较器现在只在 world size 大于一时
要求实际匹配的 native transport；
完整双 rank 规则与数值预算不变。Singleton 标签
不构成分布式能力的证明。原始失败日志保持未修改。

在干净 `c93eb2c0690b9541063330685fb7ac6560eb8812` 上，修复后的旧 Engine
矩阵通过全部 12 个 CPU 用例（118.99 秒）与全部 12 个实际 CUDA 用例
（140.47 秒），零跳过且源码/HEAD 不变。对应的
`protocol-legacy-{cpu,cuda}-v2.log`、`legacy-regression-v2-source.json` 与
`legacy-regression-v2-result.json` 与原始失败分开封存。
同一干净源码通过 78 项 host 契约、Ruff 与布局。保留的
`singleton-engine-v1/` 中的直接 singleton Engine 产物验证两个设备上的真实
train/save/新 Engine 加载，每个 runtime 124 个严格续训字段，候选
fallback 增量为零。其 native `mpi` 标签与 `JT_BUILD_USE_MPI=0` 并存；
该 singleton 元数据不得被解读为活跃 MPI 通信器的证据。

Stage3、optimizer/参数 offload、其他 optimizer、pipeline/MoE/elastic/DeepCompile、
大模型、任意 BF16 shape、非 Linux 平台与 CPU 统一启动
都不在本协议范围内。既有的近似 post-accumulate 调度不会被
提升为精确的 PyTorch worker 调度。其他加速器仍未验证。

## 主树整合验证

整合到同步后的 `4ccd72fafe51496190671dc93548c852be2974c1`。
原始的八次干净测量保留 `f787cc0b34b8b24e04d79cc55aacf746c01036b8`；
singleton 比较器修复为 `6268a15d6`，共享 collector/validator
交付为 `c93eb2c06`。没有原始数值观测被重标。

一项独立整合审计重新打开全部 32 个 rank/runtime/场景 NPZ 文件，
在未变的 `.002`/`.01` 预算下重算 3,968 对数值字段，
并逐字节检查 2,048 对同 runtime 续训字段。CPU 有
997/1,024 个逐位梯度字段；CUDA 有 1,024/1,024。独立的共享
collector 也重新生成完整的源码/wheel 闭包与全部四个记录：
四 accepted、零 blocked 或 rejected，registry 覆盖为空。官方
重建脚本/日志及其 wheel SHA256 经检查；重建
checkout 在固定上游提交处保持干净。

在整合后的树上，旧的单进程 FP32 Adam/AdamW × ZeRO0/1/2 × 两个
偏移回归通过全部 12 个 CPU 与 12 个实际 CUDA 节点，零跳过。Launch、
scorecard、Accelerate 与 registry host 回归通过 131 个节点。完整结构
通过 1,506 项，六项跳过；改动 Python 文件的 Ruff 与布局通过。
报告与生成的 manifest 登记解决了孤立树的两项错误。

整合环境失败被保留：第一次 CUDA 运行通过六个
节点、失败六个，因为真实 Torch 在另一个进程占用约 18 GiB 时
无法分配 ZeRO bucket。在空闲内存充足的
分配卡上重复未变的 12 节点协议则通过。第一次 CPU 尝试
在数值比较之前失败，因为真实 Torch 的扩展构建器
找不到 Ninja；其最初缺失的私有临时目录也
允许了临时路径回退。最终运行创建该目录并
使用既有的真实 Ninja 可执行文件，在独立的 unit 缓存中串行编译扩展。
这些环境修正不改变任何模型、
optimizer、bucket 规格或数值预算。

未版本化的独立与整合证据位于
`$JITTOR_LAB_ROOT/_state/deepspeed-paired-integration-4ccd/`，包括
`direct-array-audit.json`、`independent-scorecard-v1/`、源码检查与门禁日志。
这加强了有界的配对 Engine 结果；它不提供 CPU `jtrun`、
fresh-process 恢复、多机或整库结论。
