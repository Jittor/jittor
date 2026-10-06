# 已知限制

这一页是**用户可见的开口项**的摘要：会影响你拿到的数值、速度，或者某个 API 直接不
可用的事情。每条带一个编号，指向
[问题总账](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
——**总账才是唯一真源**，有复现命令、测量条件、已被推翻的猜想和当前进展；这里只给
一句话和它的影响。

只收**开口**项。已修的条目留在总账里（它们记着怎么被发现的）。纯测试基础设施、
门禁耗时与静态检查覆盖这类条目不在这里。

## 数值与语义

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-DTYPE-002` | `jt.array` 隐式构造会把 64 位 NumPy 值收窄到 32 位 | 当前默认行为，有显式逃生口；混 float64 数据时要显式指定 dtype |
| `KI-DTYPE-003` | Python float 与 float64 张量运算时按 float32 参与 | 静默丢 29 位尾数，float64 计算要留意 |
| `KI-OPS-012` | 半精度 max/min 归约从有限字面量折叠 | CUDA half 上可能给出错值（CPU 那半已修并验证） |
| `KI-OPS-014` | 一行 key 全是 `finfo.min` 时，float32 融合注意力的反向当成概率全为 1 | 前向正确；只在整行被 mask 掉时出现 |
| `KI-SEMANTICS-003` | 浮点比较的后端验证不完整 | CPU/CUDA 与 NPU float32 已验证，其余 NPU dtype 与 ROCm 未验证 |
| `KI-OPS-002` | 整数 floor-division 的后端验证不完整 | CPU/CUDA/NPU 已验证，ROCm 待验证 |
| `KI-COMPLEX-001` | 原生复数能力有缺口 | 部分线性代数与 FFT 走实部/虚部桥接，`complex128` 未支持，见[复数 dtype](../notes/complex-dtype.md) |
| `KI-COMPILER-008` | 用户可达的 JIT 源码检查报成内部错误 | 异常能捕获，但类别与消息指向框架内部而不是你的输入 |

## 性能与显存

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-CODEGEN-001` | 真实输入的广播是跨步视图，融合逐元素 kernel 为此每元素付一次除法与取模 | 内存受限的广播加法约为稠密形态的 4.22x |
| `KI-CODEGEN-003` | CUDA 代码生成的归约在 UNet 形状上落后 PyTorch | diffusers UNet2D 训练步 |
| `KI-CODEGEN-004` | 一个 UNet2D 步的融合逐元素 kernel 慢约 10% | 同上 |
| `KI-CODEGEN-002` | `split{i}` 与 `parallel` 两个循环选项不能同时用 | 编译错误，不会算错；挡住一类 tuner 配置 |
| `KI-TUNER-001` | 手写元算子乘积形态下 matmul/conv relay 不触发，落到通用 kernel | 用户常走的路径已修；手写形态仍慢很多 |
| `KI-OPS-006` | NaN 正确的 CPU max/min 归约只有一半速度 | 结果正确，吞吐减半（已改善一半） |
| `KI-MEM-002` | 读一个设备张量会把它搬回主机 | 读过的张量之后的算子可能在 CPU 上继续 |
| `KI-EXEC-004` | 流水执行多付 43% 峰值显存 | 已测量并接受的取舍，见[流水执行](../notes/pipelined-execution.md) |
| `KI-EXEC-010` | 反向构图不参与流水提交 | 构反向图期间设备空闲 |
| `KI-COMPAT-009` | float32 张量除以 Python float 时按 float64 计算 | 结果与 PyTorch 一致，代价是多一次宽类型运算 |
| `KI-DIST-001` | FSDP2 扁平分片的峰值高于未分片模型 | 数值正确；省不下显存，见 `compat/tests/fsdp2/test_fsdp_memory.py` |
| `KI-BACKEND-011` | oneDNN CPU 路径只有 float32，单次调用开销未测 | 非 float32 的 CPU 卷积/矩阵乘走通用 kernel |

## Torch 兼容缺口

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-COMPAT-001` | Torch 命名空间里混有原生专用助手与被导入的符号 | `dir(torch)` 比真 torch 多；已登记在 Torch API 清单里 |
| `KI-COMPAT-002` | `torch.ops.aten` 没有原生算子面 | `import vllm_omni` 到不了头 |
| `KI-COMPAT-003` | nested tensor 不是 `Tensor` | `isinstance(torch.nested.as_nested_tensor(...), torch.Tensor)` 为假 |
| `KI-COMPAT-004` | 工厂函数的结果不是反向叶子 | `torch.ones(3, requires_grad=True).is_leaf` 为假；梯度本身正确 |
| `KI-COMPAT-006` | `load_state_dict` 共享源张量，原地优化器会同时改两个模型 | CUDA 上静默：EMA 副本一类的第二个模型会被一起更新 |
| `KI-COMPAT-007` | `no_grad` 下 `w.copy_(x)` 且 `x` 需要梯度时，`w` 的梯度丢失 | 反向后 `w.grad` 仍是 `None` |
| `KI-COMPAT-008` | `torch.optim.SGD` 不接受 `foreach=` / `fused=` | 构造时 `TypeError`，不是静默忽略 |
| `KI-AUTOGRAD-003` | `register_hook` 让接收方误报自己的驻留设备 | 只影响自省，不影响计算 |
| — | 没有 Lightning 兼容层 | `pytorch_lightning` / `lightning` 不被支持；自研的 `jittor.lightning` 已于 2026-07 删除 |
| — | 编译过的 PyTorch 扩展装不进来 | `mmcv.ops`、`torch_npu` 等针对 PyTorch C++ ABI 编译的包；架构边界，见 [Torch 兼容](../compatibility/torch.md) |

## 后端与硬件

| 编号 | 现象 | 影响 |
| --- | --- | --- |
| `KI-BACKEND-001` | 窄整数 sum/max/min 在 NPU 上缺原子操作 | 这些组合在 NPU 上 skip |
| `KI-BACKEND-002` | 组合出的 `atan2` 在 NPU 上可能崩 | NPU skip |
| `KI-BACKEND-003` | 复数 `irfft` 在 NPU 上可能卡住 | NPU skip |
| `KI-BACKEND-009` | CUDA 编不出逻辑类与窄整数归约 | 一整族归约在 CUDA 上不可用 |
| `KI-BACKEND-006` | `cublas_test` / `cudnn_test` 被派发到主机且主机没有 kernel | 两个库自检算子跑不了；库本身可用 |
| `KI-BACKEND-012` | ACL 描述符缓存没接进任何 runner，launcher 与属性迁移没在昇腾设备上跑过 | 这几条 ACL 路径只有主机侧证据 |
| `KI-BACKEND-013` | 7 条后端梯度没有梯度测试，1 条未实现 | HCCL / ROCm / ACL |
| `KI-COMPILER-001` | 并行编译器可能破坏进程状态 | 文件级编译池死锁已于 2026-09-16 修复，算子级状态仍开口；见[并行编译器段错误](../development/known-issues/parallel-compiler-segfault.md) |
| `KI-COMPILER-007` | CPU-only 无 CUDA 的 import 在退出时报堆损坏 | 目前只在门禁内复现 |
| `KI-EXEC-003` | cuDNN 自动调优没有与执行调度隔离 | 同一模型同一输入的训练数值会静默变化 |
| `KI-EXEC-005` | 另一个线程在批次中途释放的 var 会让该批次失败 | 多线程下罕见且目前无法验证 |
| `KI-EXEC-007` | 四个线程写同一个参数会把一块分配释放两次 | 多线程写同一参数时可复现地中止 |
| `KI-EXEC-008` | CUDA 卷积测试文件偶发在前向存活计数下溢时中止 | 整个进程中止 |
| `KI-EXEC-009` | 被记录的多输出 `Function` 算子可能多释放一次反向存活计数 | 解释器退出时中止，或每次调用泄漏两个 Var；与环境相关 |

等硬件才能验收的后端（ROCm、天数 Corex、多卡 HCCL、跨机）列在
[平台支持](platform-support.md)与
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md)。
