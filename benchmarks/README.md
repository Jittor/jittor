# 性能基准与对照测量

性能基准套件统一放在本目录，按测量目的选择。兼容层的性能门禁、构图诊断与
可复用 profiling 工具由各自目录维护，在下面的统一目录中也能找到。

```bash
# 只读源码与套件清单，不导入框架、不触发 JIT，也不下载模型
python benchmarks/catalog.py
python benchmarks/catalog.py --suite asv --json
python benchmarks/catalog.py --suite torch-compat --json
```

目录列出用途、实际入口、运行时、声明的设备范围和工作负载；ASV 的名字与参数从源码
解析，真实模型清单复用其自己的 `--list --json`。路径失效或不能静态解析的 ASV 参数会
直接报错。**列出了设备不表示完成硬件验收**，实际运行仍遵守各套件的设备与环境协议。

| 套件 | 回答的问题 | 运行方式 |
| --- | --- | --- |
| [`asv/`](asv/) | 算子、小模型、优化器和传输在不同提交间有没有回归 | `nox -s benchmark` / `nox -s benchmark_cuda`；CI 只选取部分工作负载，其余手工运行 |
| [`torch_compat/`](torch_compat/README.md) | 同一份真实模型代码经 Jittor Torch 前端运行，与独立 PyTorch 相差多少 | 手工运行，设备选项为 CPU、CUDA、NPU；具体模型是否通过需看设备实测 |
| [`acl/`](acl/README.md) | 原生 ACL 后端与 `torch_npu` 的速度、数值和 CANN 发射开销 | 在真实昇腾设备上手工运行，含 `xval/` 与 `probes/` |
| [`ecosystem-speed`](../compat/tests/torch/test_ecosystem_speed.py) | 真实规模的生态前向与反向相对参考运行时多慢 | opt-in pytest 测量，显式设置比值阈值才作为门禁 |
| [`graph-build`](../tools/benchmarks/graph_build_phases.py) | 构图的主机开销在哪个阶段，执行是否混进计时 | 手工诊断，CPU/CUDA；阶段细分需要 profiling build |
| [`profiling`](../agent/skills/jittor-transformers-perf/SKILL.md) | 算子、训练、带宽与冷编译成本的瓶颈 | 按各 skill 的独立环境运行，工具的前端与计时口径各异 |

这些套件各自保留依赖、计时协议与设备要求；ASV 只扫描 `asv/`，不会导入手工测量脚本。
完整的计时、正确性和结果留存要求见[基准测试规范](../docs/performance/benchmarking.md)。

从仓库根目录运行：

```bash
# 按提交记录 CPU 性能回归
python -m nox -s benchmark

# 对比同一 Jittor Torch 前端的直接训练与 Accelerate 包装开销
python -m nox -s benchmark_accelerate

# 查看真实模型工作负载，无需导入框架或加载模型
python benchmarks/torch_compat/run.py --list

# 昇腾原生后端对照；两侧必须使用相同的精度配置
python benchmarks/acl/bench_jittor.py --hf32
python benchmarks/acl/bench_torch.py --hf32
```

编译缓存、模型缓存、原始日志和测量结果保存在
`${JITTOR_LAB_ROOT:-../jittor-lab}/_state/`，不写进本目录。
不同套件、并行运行和单元测试使用独立的 `JITTOR_HOME` 或 `cache_name`；
首次 JIT 编译串行完成。使用手工测量入口前，按对应套件说明准备隔离环境。

## 覆盖边界

这里覆盖的是选定的工作负载，不能代表整个框架都已有性能基线。新增基准前，先查看
`catalog.py --json` 的实际参数，避免把不同规模、训练阶段或精度误当成同一项。

| 维度 | 已有入口 | 仍缺的系统覆盖 |
| --- | --- | --- |
| 算子与模型 | ASV 的算子、卷积前向/反向、归约、tiny Llama；真实模型套件；ACL MLP/CNN/Transformer | 算子 shape × dtype 的普遍扫描，原生整模与 Torch 前端的统一分层对照 |
| 稳态与冷启动 | ASV 稳态；真实模型单列首步；single-kernel cold-profile | 端到端安装/导入/首编译的跨提交自动基线 |
| 设备 | ASV CPU/CUDA；真实模型 CPU/CUDA/NPU 入口；ACL 探针 | ROCm/Corex 的专用周期基线；NPU 的 ASV/CI 留存；跨设备的同规格矩阵 |
| 内存与传输 | ASV allocator/RSS、CUDA D2H；真实模型的设备进程内存采样 | 统一的 H2D/D2H、输入流水与 pinned memory 扫描，以及跨后端一致的峰值口径 |
| 训练与分布式 | SGD/AdamW；整模训练；opt-in Accelerate 单进程训练 | 多卡 collective、DDP/FSDP 扩展效率与通信重叠基线 |
| 数据加载 | Accelerate 的 resident TensorDataset 和整模预置输入池 | 文件 I/O、worker、预处理、传输重叠组成的端到端数据流水基线 |

CI 的选择以 [CPU workflow](../.github/workflows/cpu.yml)、
[CUDA workflow](../.github/workflows/cuda.yml) 与 [nox 会话](../noxfile.py) 为准。
手工工具能运行、真实设备通过、周期回归留存是三个不同层次；目录不把它们合并为
“已覆盖”。ASV 的 Accelerate 行要求独立的 shim 进程与依赖，属于显式 opt-in，
不会因为出现在清单中就加入默认 CPU/CUDA 会话。
