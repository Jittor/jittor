# 已知限制

[本栏目目录](index.md) · [English](known-limitations.en.md)

本页汇总[历史记录](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/known-issues.md)中影响数值、性能与 API 可用性的限制。KI 编号用于查找原始复现和证据；本次整理没有重新验收或关闭这些条目。判断当前版本时应查最新 GitHub issue/PR，并按实际版本复现。

限制修复后同步更新本页并链接验证证据。纯测试基础设施问题不列入用户指南。

## 数值与语义

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-DTYPE-002` | `jt.array` 隐式构造会把 64 位 NumPy 值收窄到 32 位 | 当前默认行为，有显式逃生口；混 float64 数据时要显式指定 dtype |
| `KI-SEMANTICS-003` | 浮点比较的后端验证不完整 | CPU/CUDA 与 NPU float32 已验证，其余 NPU dtype 与 ROCm 未验证 |
| `KI-OPS-002` | 整数 floor-division 的后端验证不完整 | CPU/CUDA/NPU 已验证，ROCm 待验证 |
| `KI-COMPLEX-001` | 部分复数算子与高阶导数仍有缺口 | complex64/complex128 是原生 dtype；具体 FFT、后端与导数阶限制见[复数 dtype](../notes/numerics/complex-dtype.md) |

## 性能与显存

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-CODEGEN-001` | 真实输入的广播是跨步视图，融合逐元素 kernel 为此每元素付一次除法与取模 | 内存受限的广播加法约为稠密形态的 4.22x |
| `KI-CODEGEN-003` | CUDA 代码生成的归约在 UNet 形状上落后 PyTorch | diffusers UNet2D 训练步 |
| `KI-CODEGEN-004` | 一个 UNet2D 步的融合逐元素 kernel 慢约 10% | 同上 |
| `KI-TUNER-001` | 手写元算子乘积形态下 matmul/conv relay 不触发，落到通用 kernel | 用户常走的路径已修；手写形态仍慢很多 |
| `KI-OPS-006` | NaN 正确的 CPU max/min 归约只有一半速度 | 结果正确，吞吐减半（已改善一半） |
| `KI-EXEC-004` | 流水执行多付 43% 峰值显存 | 已测量并接受的取舍，见[流水执行](../notes/runtime/pipelined-execution.md) |
| `KI-EXEC-010` | 反向构图不参与流水提交 | 构反向图期间设备空闲 |
| `KI-DIST-001` | FSDP2 扁平分片的峰值高于未分片模型 | 数值正确；省不下显存，见 `compat/tests/fsdp2/test_fsdp_memory.py` |
| `KI-BACKEND-011` | oneDNN CPU 路径只有 float32，单次调用开销未测 | 非 float32 的 CPU 卷积/矩阵乘走通用 kernel |

## Torch 兼容缺口

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-COMPAT-001` | Torch 命名空间里混有原生专用助手与被导入的符号 | `dir(torch)` 比真 torch 多；已登记在 Torch API 清单里 |
| `KI-COMPAT-002` | `torch.ops.aten` 没有通用原生算子面 | 适配器只注册其需要的有界算子；不能据导入成功推断任意 aten 调用受支持 |
| `KI-COMPAT-003` | nested tensor 不是 `Tensor` | `isinstance(torch.nested.as_nested_tensor(...), torch.Tensor)` 为假 |
| `KI-COMPAT-004` | Tensor 别名缺少部分原地修改保护与动态可训练性语义 | 视图自动微分行为仍有差异；工厂叶子问题已关闭，不能沿用旧结论 |
| — | 没有 Lightning 兼容层 | `pytorch_lightning` / `lightning` 不被支持；自研的 `jittor.lightning` 已于 2026-07 删除 |
| — | 编译过的 PyTorch 扩展装不进来 | `mmcv.ops`、`torch_npu` 等针对 PyTorch C++ ABI 编译的包；架构边界，见 [Torch 兼容](../compatibility/torch.md) |

## 后端与硬件

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-BACKEND-001` | 窄整数 sum/max/min 在 NPU 上缺原子操作 | 这些组合在 NPU 上 skip |
| `KI-BACKEND-002` | 组合出的 `atan2` 在 NPU 上可能崩 | NPU skip |
| `KI-BACKEND-003` | 复数 `irfft` 在 NPU 上可能卡住 | NPU skip |
| `KI-BACKEND-006` | `cublas_test` / `cudnn_test` 被派发到主机且主机没有 kernel | 两个库自检算子跑不了；库本身可用 |
| `KI-BACKEND-012` | ACL 描述符缓存没接进任何 runner，launcher 与属性迁移没在昇腾设备上跑过 | 这几条 ACL 路径只有主机侧证据 |
| `KI-BACKEND-013` | 7 条后端梯度没有梯度测试，1 条未实现 | HCCL / ROCm / ACL |
| `KI-BACKEND-016` | ACL 上 `sd15_unet_train` 偶发崩在 507035（非确定性） | 该训练 workload 在昇腾上时成时败；`jt.flags.auto_graph_replay=0` 可绕开 |
| `KI-BACKEND-017` | 昇腾上 device-bound 的单核密集 workload 比 torch_npu 慢 | 训练/发射受限图（`qwen3_train` 快约 11-13%）持平或更快，`qwen3_prefill`/`vit` 这类慢 |
| `KI-COMPILER-001` | 并行编译器可能破坏进程状态 | 文件级编译池死锁已于 2026-09-16 修复，算子级状态仍开口；见[并行编译器段错误](../development/known-issues/parallel-compiler-segfault.md) |
| `KI-COMPILER-007` | CPU-only 无 CUDA 的 import 在退出时报堆损坏 | 目前只在门禁内复现 |
| `KI-EXEC-003` | cuDNN 自动调优没有与执行调度隔离 | 同一模型同一输入的训练数值会静默变化 |
| `KI-EXEC-005` | 另一个线程在批次中途释放的 var 会让该批次失败 | 多线程下罕见且目前无法验证 |
| `KI-EXEC-008` | CUDA 卷积测试文件偶发在前向存活计数下溢时中止 | 整个进程中止 |

等硬件才能验收的后端（ROCm、天数 Corex、多卡 HCCL、跨机）列在
[平台支持](hardware/platform-support.md)与
[历史记录](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/deferred-hardware.md)。
