# 绑定源码的 Accelerate checkpoint 协议

- 状态：已验证受限 CPU/CUDA 数值协议与全新进程恢复协议；CPU 配对的 world-2 准入仍被阻塞
- 日期：2026-10-16
- 基线：`c7c2d37a085dad5e3ba9eb333a34c3cb570bfb09`
- 生产改动：`815be5daca4aa8cc1a9f569b66b00f78bf77bbaa`
- 维护者：Torch 兼容维护者
- 复查条件：Accelerate 版本/源码/wheel、协议仪器化、安装证据规则、checkpoint 数学、launcher 或 device provider 变化

上一版协议审计器在两侧都往既有 `installed_files_sha256` 映射中加入一个未绑定成员之后，接受了复制而来的真实 CPU 记录。它同时忽略了注入的 `distribution_version=9.9.9` 字段，但该字段在旧观测中并不存在：这第二个探针**不能**证明一个实际安装的 9.9.9 发行版通过了旧门禁。原始 metadata 只有模块版本、来源、安装文件哈希与源码哈希，没有实际 distribution METADATA/RECORD。原始的八条阶段观测未被改动。被证实的缺口是安装声明校验不完整，以及缺少实际 distribution 证明。

维护 worker 现在把真实导入的 Accelerate 包绑定到官方 1.10.1 wheel、完整包文件映射、实际同前缀 METADATA 与 RECORD 字节以及上游源码。审计器在计算数值结果之前重新校验这些字节、完整载荷与路径。在这些全新环境中，安装后的 RECORD 使用 CRLF；捕获时保留精确的 UTF-8 字节。新 helper 与既有 source-proof/dependency-report helper 都纳入了每个阶段实测源码哈希映射。门禁要求实际的官方 wheel；没有任何旧观测获得重建的 metadata。缺少这些实际测量字段的历史行仍属历史数值证据，不能获得当前严格的 installed-source 准入。

Accelerate 没有模型注册表分母。这些是显式的 `record_kind=protocol` 行，带空 axes 与注册表引用。checkpoint 协议通过并不建立完整的 Accelerate 支持。

## 冻结输入与实际环境

Accelerate 1.10.1 是干净的官方上游 `623cc0ba588d192c77758939a6e340adf07edd4f`；wheel SHA256 为 `3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11`。其完整包闭包由该 checkout 与官方 wheel 推导，并在每个实际 producer 与 consumer 进程中分别校验。

CPU 使用三个全新环境：controller、candidate 和独立的二进制 Torch 2.11.0+cpu oracle。CUDA 另用两个全新运行时环境：native candidate 的完整固定 CUDA 12.2 wheel 依赖栈，以及独立解析的官方 Torch 2.11.0+cu128 oracle。所有安装都走真实依赖解析并通过 `pip check`。candidate 的 Torch distribution metadata 来自真实官方 CPU wheel；实际数值执行显式使用 shim 与 native 后端。oracle 断言其二进制扩展存在且无 shim。

四个实际运行时前缀都做了完整的源码/二进制 RECORD 哈希与大小审计。生成的 Python 字节码被显式排除：早先一次审计保留了 NumPy 生成的 pyc 不匹配，而不是改动那个安装前缀。此外，新 oracle 的官方 Torch 与 19 个实际 CUDA 依赖 wheel 与 12,743 个已安装包的源码/二进制文件匹配。包含该 payload 的三个前缀中的 CPU Torch 载荷，以及 candidate 的十个 CUDA 依赖 wheel，分别与其官方 wheel 载荷匹配。打包 metadata 与被重定位的脚本仍单独受 RECORD 绑定。这些检查并不断言每个已安装的可选库都被执行过。

冻结框架的 2,887 个 Git blob（包括 `python/jittor/compat` 符号链接的目标文本）同时匹配实际源码树与 setup 快照。一个 live observer 记录了 165 个选定 CUDA 库映射快照，并重新哈希了 20 个未变化的映射库。实际 candidate/oracle 的 NCCL、CUDA runtime 与 cuBLAS 路径指向各自新的私有前缀；旧的共享 provider 未被替换。实际二进制 Torch 的分配、矩阵乘、同步与驱动 UUID 探针在 CUDA 运行前都已在两张指定卡上通过。

## 数值与恢复范围

原协议、模型、优化器/调度器、固定初始权重、seed-0/7 输入、完整梯度与预算保持不变：逐元素 `atol=2e-6`、`rtol=2e-5`；保存前三次更新，之后两次更新。每个 producer 在新的 consumer 构造全新 model/optimizer 对象并加载 checkpoint 之前终止。world one 另外使用 StepLR 并恢复全新 scheduler；维护后的 world-two FULL 协议没有 scheduler。参考实现使用实际 torchrun；candidate 多 rank 执行使用公共 `python -m jittor.distributed.launch`，即 jtrun 归属方。

| 设备 / 协议 | 数组对数 | 参数梯度 | 输入梯度 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| CPU, world 1, FP32 | 438 | 28 / 28 | 14 / 14 | 5.960465e-8 |
| CUDA, world 1, FP32 | 438 | 28 / 28 | 14 / 14 | 2.980232e-8 |
| CUDA, world 2 FULL, FP32 | 820 | 56 / 56 | 28 / 28 | 1.490116e-8 |
| CUDA, world 2 FULL, BF16 | 820 | 56 / 56 | 28 / 28 | 1.490116e-8 |

全部 2,516 次跨运行时数组比较通过，包括全部 168 个参数梯度字段与 84 个输入梯度字段。自身运行时恢复的 model/optimizer、Python/NumPy/CPU RNG、CUDA 行上每个可见的 CUDA RNG 流、world-one scheduler 与续训轨迹都精确一致。BF16 行保留了对实际计算 dtype/device 以及 FP32 master 参数/动量的维护检查。每个必需的 CUDA 阶段都实际执行，并带真实 device 断言与 fallback 错误/计数为零检查。

一个全新 CPU oracle torchrun/Gloo 探针成功初始化了两个 rank，但公共 Accelerate 在 CPU 上没有选择 FSDP2。因此 CPU world 1 只与 CUDA world 1 配对。两个 world-2 CUDA 协议保有其独立的数值/恢复事实，但对 CPU 配对的共同准入仍然阻塞；没有任何 DDP 替代或直接原生 CPU FSDP 被计入该协议。

实际共享的 `audit_coverage.scorecard` 准入：**2 接受 / 2 阻塞 / 0 拒绝**。CPU 与 CUDA world-one 行具有验证过的 L3；两个 CUDA FULL 行具有独立的已验证 L3，但因其精确的 CPU 协议对应项缺失而以 `level=null` 准入。没有产生注册表分子。既有注册表校验器未改动，其通用负例契约通过。

共同审计器另外接受了实际的新 CPU 对照，并在八条阶段记录被一致重新绑定后拒绝了四个一次性副本：实际 distribution 版本 1.10.1 改为 9.9.9、额外的未绑定安装载荷、改动/重新哈希的 METADATA 文本，以及官方 wheel 构件被移除。未改动的历史 v6 CPU 行因缺少 wheel 构件被单独拒绝；没有向该历史行提供任何新 metadata。结果见 `common-negative-results.json`。

## 复现与证据

所有原始观测、installed-source 证明、脚本与日志均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ecosystem-evidence-next/20261016/` 下。CPU 输出为 `formal-cpu/`；最终 CUDA 输出为 `formal-cuda-v2/`；最初的 `formal-cuda/` 尝试保持不可变。第一次 CUDA 尝试失败，原因是外部启动脚本在既有归属方要求目录的位置提供了 NCCL 文件。在其自然终止之后，修正后的脚本把两个解释器重跑到新输出，只使用其自身的正常 benchmark 缓存。没有复制任何 JIT 缓存。controller-only 解释器中的一次 collector 尝试因缺少 NumPy 失败；其输出被保留，实际重算使用了准备好的独立 CPU 解释器。

使用仓库 runbook 中维护的命令，配合匹配的已准备解释器与 `JITTOR_ACCELERATE_WHEEL`/`--wheel`。新的硬件运行请选择全新输出与独立 benchmark 缓存，为 CPU 显式隐藏 CUDA，为 CUDA 分配两张健康卡，并保留所有环境/源码证明。本结果既不建立多主机执行、分片 checkpoint 格式、性能水平，也不覆盖其他 Accelerate API。

冻结基线处的维护检查：`tools/run_test_suite.py --tier core` 完成，322 passed、42 skipped、1 xfailed；专用协议测试与通用 downstream scorecard host 测试完成，134 passed 且无 skip；完整 `tests/structure` 加 `compat/tests/structure` 完成，1,624 passed、四个 skip；仓库布局检查通过。新单测 home/cache 与 benchmark home/cache 相互分离，首次编译串行执行。今后无框架的纯 host 检查使用 `JITTOR_TORCH_SHIM=0`；本次实际运行使用模式 1，并保留了其额外的冷单测编译。在加入本报告之后，文档/私有路径检查为 18 passed，另有一个预期的集成前置条件：`TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists` 报告这个新页面未出现在中央 results toctree 中。集成归属方必须登记它并重新生成中央 manifest；测试与索引都没有被绕过。

不可变的初步测量封条为 `measurement-seal-v2.json`：564 个文件 / 105,020,755 字节，SHA256 `637cf98d1a7e52c8f5aefbf064dfbd5510864c9ad3b2cec2d74c7a9b014b3b0c`。它排除了当时仍在运行的单元日志；最终维护/common-scorecard 产物另行封存。初步 v1 的自身日志不匹配被保留为一次失败的封存尝试。独立的维护者审计另外核实了全部 564 个文件、全部 2,887 个源码 blob、2,516 对数组、168/84 个梯度、48 个不同的正式 worker，以及 1,520 项自身运行时精确检查。其 2,336 条计算设备观测把 280 个标量 CPU 优化器步数计数器与实际计算张量区分开；它们不要求每个优化器控制标量都驻留 CUDA。

独立审计未版本化，位于 `$JITTOR_LAB_ROOT/_state/accelerate-main-integration-4ce/20261016/independent-c7-audit-v3.json`，SHA256 `4c5e8757397ad574d18a37e12c66e1cc1e803def7f748184da5aec8f2c0a8ecf`。

中央集成独立重新哈希了最终封条的完整 682 个文件
（117907054 字节），其 manifest SHA256 为
`173107537e5a2567ede0975c65abedcc14cd38d50ac7ba2140aa71031fd8bf94`。
这补充了上面独立的原始 2516 数组、48 worker 与完整
安装/源码审计；整合本报告既不改变原始测量基线，也不改变
其数值预算。

最终中央检查：182 项定向 host 测试，core 322 passed / 42 skipped /
1 既有 xfail，扩展结构 1628 passed / 6 skipped；布局、manifest
与 Ruff 通过。当前共同准入在未改动的原始记录上独立复现出 2 个接受的 L3，
2 个因 CPU 未配对而阻塞的独立 L3，以及 0 个拒绝。只有 FlashAttention 与 Accelerate 的 nox 函数体发生了变化；
其余 72 个与同步后的集成基线完全相同。
