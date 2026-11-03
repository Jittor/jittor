# 核心与 compat 性能诊断

本目录保存可复用的算子、梯度管理与设备前端探针。它们是手工诊断工具，各自有独立的
设备与计时口径；源码存在不代表已经通过当前提交的硬件验收。模型或第三方库的完整
验收应到 [JittorEcosystem](https://github.com/JittorEcosystem) 对应仓库执行。

| 工具 | 用途 | 设备 / 前端 |
| --- | --- | --- |
| `benchmark_npu_frontends.py` | 同尺寸矩阵乘的原生/独立 compat 前端延迟与数值校验 | ACL NPU，分别起进程；检查驻留与零回退 |
| `audit_kernel_changes_gpu.py` | softmax、GELU、LayerNorm 边界与生成核诊断 | CUDA / 原生 Jittor |
| `probe_softmax_gelu.py` | softmax、GELU 的构图和执行开销 | CUDA / 原生 Jittor |
| `probe_large_softmax.py` | 大词表 streaming softmax 的候选核对照 | CUDA / 原生 Jittor；实验实现，不改生产选择器 |
| `probe_clip_grad_norm.py` | 梯度裁剪的实现消融 | CUDA / 原生张量与 compat 梯度助手 |
| `benchmark_grad_management.py` | clip/scaler 的同步开销 | CUDA / 原生张量与 compat 梯度助手 |

## 运行环境

解释器需安装当前核心；涉及 compat 的工具还需安装配套 `jittor-compat`。从仓库根目录
执行。为每个任务选择独立 `cache_name`，首次 JIT 串行预热；不要与测试共享缓存。

```bash
cache_name=softmax-audit CUDA_VISIBLE_DEVICES="$JITTOR_GPU" \
  benchmarks/profiling/run_perf_env.sh python \
  benchmarks/profiling/audit_kernel_changes_gpu.py \
  --task softmax-case --length 50257 --rows 2 --dtype float32
```

包装器保留用户的 `HOME`，将 JIT、临时文件与默认结果放到
`$JITTOR_LAB_ROOT/_state/profiling/<cache_name>/`；未设置 lab 根时使用仓库同级的
`jittor-lab/`。可用 `JTCUDA` 指定完整 CUDA SDK 根目录，设错会失败，不静默换工具链。

NPU 基准先按[硬件指南](../../docs/guides/hardware/ascend-910b.md)加载 CANN，设置已分配的
`ASCEND_RT_VISIBLE_DEVICES`。两个模式用相同输入、不同缓存和进程顺序测量：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
run_dir="$JITTOR_LAB_ROOT/_state/profiling/npu-matmul"
JITTOR_HOME="$run_dir/native-cache" python benchmarks/profiling/benchmark_npu_frontends.py \
  --mode native --output "$run_dir/native.json"
JITTOR_HOME="$run_dir/compat-cache" python benchmarks/profiling/benchmark_npu_frontends.py \
  --mode compat --output "$run_dir/compat.json"
```

数值、驻留或回退校验失败时不得形成成功报告。结果只覆盖所测矩阵尺寸与前向延迟，
不能外推为整个模型性能。具体输入选项看各脚本的参数定义；部分 CUDA 脚本在参数解析前
导入 Jittor，`--help` 也可能触发初始化。

## 计时与证据

- 记录提交、包来源、芯片、SDK、精度、输入尺寸与同步口径。
- 首次编译与稳态分开；预热输入驻留，保留所有输出/梯度直到同步。
- 比较原生 PyTorch 时使用独立进程，确认导入真实 PyTorch；不要将 `jittor` 别名成 `torch`。
- 每种候选实现先验证数值和所声明的梯度；运行不报错不等于正确。
- 原始结果留在 lab；复查结论记录到 issue/PR。完整规则见[性能基准规范](../../docs/performance/benchmarking.md)。
