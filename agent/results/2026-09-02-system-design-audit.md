# Jittor 2.0 设计审计（2026-09-02）

基于本轮实测（nsys 按步切窗、执行器分相计时、cuDNN 独立微基准、gdb 回溯、门禁日志）与源码核对，
列出当前系统在设计层面不合理、需要改进的地方。每条附证据、后果与建议，并标注处理状态。

严重度：**关键** = 直接决定能否达标或能否用；**主要** = 系统性成本或脆弱点；**次要** = 局部或可绕过。
统计：关键 4，主要 14，次要 9。状态为本文提交时的状态。

## A · 执行模型

元算子加统一惰性图是框架的核心资产，问题在于它被"整步先建后跑"的单线程串行执行方式束缚。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| A1 | 整步图在 CPU 上建完才发射，建图期间 GPU 空转 | BERT 一步 GPU 空闲 6.0 ms（PyTorch 2.6），Jittor kernel 总时间反而更少（24.1 对 27.3 ms） | 所有 CUDA 用例共有的结构性损失。已加 `auto_flush_ops`（默认 128，仅 CUDA）流水化，transformer 类全部追平。长期应把执行器做成异步（独立线程或按段提交），而不是靠持有变量创建时的启发式触发。 | 关键 | 已缓解 |
| A2 | 反向建图不参与流水 | `jt.grad` 内部不创建持有变量，发射从不触发；UNet 反向建图 6.5 ms 期间 GPU 闲置 | 尝试在 `grad()` 循环内把已完成的梯度提前发射，同环境 A/B 三个用例全部变慢（Llama +3 ms）：切碎反向图损失跨段融合并增加执行器调用，**不采用**。更根本的做法是让执行器提供"提交部分图"的显式接口，并降低 `jt.grad` 与 `Function` 回调自身的开销。 | 主要 | 已验证，不采用 |
| A3 | 图结构不变量只存在于代码里 | `add_hold_vars` 急切分支硬编码 `op->name()=="tape"`；`Tapes` 构造函数对已执行的算子加输入边导致 double free | "某些算子必须保持挂起"应成为节点标志并由执行器统一尊重，而不是散落的字符串比较。已加断言与单测。 | 主要 | 已修 |
| A4 | 执行器每次运行重做全部调度工作 | 每步 BFS、并查集融合、两级拓扑排序、逐融合算子构造 jit key 字符串再查 `string_view_map`；UNet 约 1.5 ms/步 | 训练循环的图逐步相同：按图结构哈希缓存融合划分、执行顺序与各段入口指针；jit key 改为结构化哈希。 | 主要 | 未动 |
| A5 | 算子内部同步与流水冲突 | 6 个算子文件内含 `cudaDeviceSynchronize`（动态形状类）；transformers 的 `.item()` 会清空整个已发射队列 | 同步点应只等待所需子图（事件级同步）；动态形状算子应走主机侧形状缓存或延迟形状。 | 次要 | 未动 |

## B · 算子后端

库调用方式停留在 cuDNN 7 时代的用法，成本被惰性图的"先建后跑"掩盖了很久。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| B1 | cuDNN legacy API 每次调用重做规划 | 微基准：`GetWorkspaceSize` 59 µs + `ConvolutionForward` 49 µs，描述符只占 3 µs；算子内实测 fwd 199 / bwd_x 91 / bwd_w 103 µs，UNet 一步 20 ms 花在卷积的 CPU 侧 | 改为 backend graph API + 按（种类、形状、步长、精度）键的执行计划缓存，每次调用 12 µs；UNet 1.48× → 1.10×。conv3d 三处仍是 legacy，应同样迁移。 | 关键 | 已实现（2D） |
| B2 | 明确拒绝 cuDNN 9 | `compile_extern.py` 因 RNN 使用已移除的 legacy API 而对 major ≥ 9 直接抛错；PyTorch 侧已是 cuDNN 9 引擎 | 把 RNN 迁到 v8 RNN API 后放开版本限制；否则卷积/注意力引擎与 PyTorch 长期存在代差。 | 主要 | 未动 |
| B3 | 算子级自动微分损坏而无人察觉 | `CudnnConvOp::grad` 与 `BackwardWOp::grad` 用 `xformat=="ncdhw"/"nchw"` 判断布局，算子实际收到的是 `"abcd"/"acdb"`；`nn/backends/cudnn.py` 的 docstring 写着 "autodiff through the raw op is broken" 并用 `Function` 绕过。同类：`jit_run` 把权重维度按物理顺序交给 cuDNN，NHWC 直接调用找不到算法 | 通过布局串读维度（已修）。每个带 `grad()` 的算子都应有对 CPU 参考的梯度单测（已加 `test_cudnn_conv_plan`），绕过缺陷不应替代修复。 | 主要 | 已修 |
| B4 | 精度策略分散且与 PyTorch 语义不对齐 | `use_tensorcore`（0–3 多义编码）、`cuda_allow_tf32`、`cuda_allow_cudnn_tf32` 三个 flag 交叉决定 compute type；`cudnn_benchmark` 三态 −1/0/1；PyTorch 默认 TF32 卷积，Jittor 默认全 fp32 | 收敛为一个 fp32 矩阵精度策略（highest / high / medium），matmul 与卷积共用，shim 映射到 `set_float32_matmul_precision`；默认值作为明确决策写进文档。 | 主要 | 未动 |
| B5 | 算子源码经文本解析器处理 | KernelIR 把 op 的 .cc 当文本解析：`#include`、`_Pragma`、格式串里的 `%` 都会失败；历史上 `_Pragma` 让 1199 个 CUDA 用例挂掉 | 把只用于代码生成的 JIT 区段与普通 C++ 分离，或给解析器加逃逸机制；至少要有明确的编写规范与错误定位。 | 主要 | 未动 |
| B6 | 字符串键的算法缓存、满后退化 | `fwd_algo_cache` 等为 `unordered_map<string,…>`，每次调用拼字符串；`max_cache_size`（100）满后每次重新启发式，曾造成一步 663 次 cuDNN 查询 | POD 结构体哈希做键（新计划缓存已采用），容量按形状数而非固定 100。 | 次要 | 部分已修 |

## C · 设备与全局状态

整个运行时建立在"一个进程、一张卡、一条流、一个线程"的假设上，且这个假设以 80 个全局 flag 的形式散布在代码里。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| C1 | 单进程只能用一张卡，设备不是张量属性 | `Var` 无 device 字段，只有全局 `use_cuda`；`device_id` 是带 setter 的全局 flag（`cuda_flags.cc`）走环境变量；shim 的 `Tensor.device` 对任何 CUDA 张量恒返回 `cuda:0`，`.to("cuda:1")` 丢卡号 | 设备需成为 Var/分配器/执行器的一等属性，PyTorch 有多少（`set_device`、`cuda:N`、跨卡拷贝、流）就要多少。由并行子任务专门处理。 | 关键 | 另行处理 |
| C2 | 全局可变状态过多、读取机制不统一 | 80 个 `DEFINE_FLAG`；`hold_vars`、`exe`、`sync_ptr`、`tflag_count` 均为全局单线程假设；C++ flag 与 Python 层 `use_parallel_op_compiler` 各有一套环境变量读取方式 | 把执行相关状态收进一个 Runtime/Context 对象；flag 分为启动期配置与运行期开关并统一环境变量语义。 | 主要 | 未动 |
| C3 | 单流执行 | 所有 kernel 与拷贝在默认流；内存复用的安全性依赖流序 | 多卡与 H2D/D2H 重叠都需要流与事件模型；应与 C1 一起设计。 | 次要 | 未动 |

## D · 构建、缓存与进程

JIT 缓存与进程管理的设计让"跑门禁"和"改代码"互斥，也让一次误杀变成小时级的排查。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| D1 | 缓存原地重建，源码改动影响所有在跑进程 | 改 `python/jittor/src` 后每个新起的进程都在原目录重编译核心；门禁运行期间不能改源码；`custom_ops` 把全部 cuDNN 算子按所有源码哈希编成一个 .so，改一个算子全量重编 | 缓存目录按源码内容哈希分版本（写入新目录、切换指针），算子库按文件粒度或按算子分片编译。 | 主要 | 未动 |
| D2 | 全局文件锁 + import 时的子进程探测可自死锁 | `compiler.py` 每次 import 都 `getstatusoutput(query_cuda_cc)`；父进程持 `jittor.lock` 时子进程也要锁，父进程停在 `pipe_read`；被杀进程留下孤儿持锁，本日两次各卡 40 分钟 | 探测结果落盘缓存；子进程不得再取同一把锁；锁带超时与持有者诊断。 | 主要 | 未动 |
| D3 | 默认缓存被所有会话共享，磁盘满时报错像随机失败 | 未设 `JITTOR_HOME` 的任何 import 都重建 `~/.cache/jittor`；根分区曾 100%，缓存被截断后表现为散布失败加段错误 | 写入前检查可用空间并给出明确错误；文档要求实验设置独立 `JITTOR_HOME`。 | 次要 | 未动 |
| D4 | 并行算子编译器在门禁中被禁用求稳 | 所有门禁脚本 `use_parallel_op_compiler=0` | 要么修到可信并默认开启，要么删除。 | 次要 | 未动 |

## E · Torch 兼容层

28k 行以 monkeypatch 方式装配 `torch` 命名空间，很多语义靠标记链和进程级注册表维持。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| E1 | 没有统一的"视图 / 存储"模型 | 切片共享靠 `_torch_index_parent`/`_torch_data_owner` 标记链逐级回写；`param[:rows].data.copy_(w)` 曾静默丢权重，补了一级回写 | 给 shim 一个显式的存储对象（base + offset + strides），所有原地操作对它生效；否则每种新写法都是一个新漏洞。 | 主要 | 已修一处 |
| E2 | 反向叶子靠进程级注册表，且被 optimizer 裁剪 | `jt._torch_leaf_params` 全局；有 optimizer 时 `backward()` 有意丢弃不在 optimizer 里的参数；无 optimizer 的手写循环靠首次前向登记才有梯度 | 叶子应由张量自身的 `requires_grad` 与图连通性决定，与 optimizer 解耦。 | 主要 | 已缓解 |
| E3 | 设置项映射不透明 | `cudnn.allow_tf32` 的 setter 用 `_jittor_cudnn_init` 门控，读起来像只在初始化前生效，实际行为需要实验确认 | 每个 `torch.backends.*` 设置项与 Jittor flag 的映射应有表格化定义与单测。 | 次要 | 未动 |
| E4 | 结构测试把实现细节钉死 | 23 个 `tests/structure` 文件逐字段断言 `__dict__`；一个内部标记被迫改为外置 `WeakSet` | 断言公共契约（属性、行为）而非私有字段集合。 | 次要 | 未动 |

## F · 测试与度量

门禁太慢、口径靠人记，速度判定的前提条件没有被工具化。

| # | 问题 | 证据 | 后果 / 建议 | 严重度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| F1 | 门禁耗时与无分层 | 原生门禁 40 分钟（16 核）、CUDA 门禁 2 小时；`test_device_parity.py` 单文件占大头 | 分 smoke / full 两层；对拍类测试按算子抽样并行化。 | 主要 | 未动 |
| F2 | 速度门禁未固化公平条件 | 默认 3 次计时；CPU 对比曾被父进程的 `OMP_PROC_BIND` 绑到单核污染 14 倍且两轮复现一致；harness 只校验 TF32 一致，不校验线程数/绑核 | harness 记录并断言两侧的线程数、亲和性掩码与精度策略；重复次数默认 ≥10。 | 主要 | 部分已修 |
| F3 | pytest 覆盖 PYTHONPATH | `pyproject.toml` 的 `pythonpath=["python"]` 让任何 pytest 都导入主树，副本/工作树无法用 pytest 验证 | 改为 conftest 按环境变量决定，或文档写明 `-o pythonpath=` 用法。 | 次要 | 未动 |
| F4 | 微基准没有"关闭图优化"的正式开关 | CSE、死码消除、未物化让手写微基准给出 17–50 TFLOPS 的不可能数字 | 提供计时 API：固定输入池、全量物化、剔除首编译，并在文档里列出陷阱。 | 次要 | 未动 |
| F5 | 异步错误报告延迟且致命 | 63 处 `LOGf`；CUDA 异步错误只提示 "rerun with JT_SYNC=1"；错误后进程退出时 "terminate called without an active exception" | 默认记录最近发射算子的 Python 位置（低开销环形缓冲）；析构路径不得抛异常。 | 次要 | 未动 |

## 优先级建议

- **先做能改变达标结果的**：B1（卷积后端迁移，含 conv3d）直接决定 UNet 与 vLLM 解码能否 ≤1.07×；C1（多卡）决定框架能否被真实训练任务采用。
- **再做消除系统性成本的**：A4（执行器计划缓存）、B2（cuDNN 9）、B4（精度策略统一）。
- **基础设施类三项（D1、D2、F1）** 不影响用户结果，但决定后续每一项改动的验证周期，本轮为此损失的机器时间以小时计。

范围：不含 NPU/ROCm 后端；C1 由并行子任务详细处理。
