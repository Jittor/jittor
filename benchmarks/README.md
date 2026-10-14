# 性能基准与对照测量

仓库的性能基准统一放在本目录，按测量目的选择套件：

| 套件 | 回答的问题 | 运行方式 |
| --- | --- | --- |
| [`asv/`](asv/) | 算子、小模型、优化器和传输在不同提交间有没有回归 | `nox -s benchmark` / `nox -s benchmark_cuda`，由 CI 留存结果 |
| [`torch_compat/`](torch_compat/README.md) | 同一份真实模型代码经 Jittor Torch 前端运行，与独立 PyTorch 相差多少 | 手工运行，支持 CUDA 与 NPU |
| [`acl/`](acl/README.md) | 原生 ACL 后端与 `torch_npu` 的速度、数值和 CANN 发射开销 | 在真实昇腾设备上手工运行，含 `xval/` 与 `probes/` |

这些套件各自保留依赖、计时协议与设备要求；ASV 只扫描 `asv/`，不会导入手工测量脚本。
完整的计时、正确性和结果留存要求见[基准测试规范](../docs/performance/benchmarking.md)。

从仓库根目录运行：

```bash
# 按提交记录 CPU 性能回归
python -m nox -s benchmark

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
