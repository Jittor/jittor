# CUDA float64 matmul 计算精度修复

- Status: Real-CUDA gate accepted
- Last reviewed: 2026-08-24
- Commit: `cdd12d04`
- Owner: CUDA linear-algebra maintainers
- Review when: cuBLAS compute-type selection, scalar ABI, or matmul routing changes

## 结论

CUDA 11+ 的三个 cuBLAS matmul 实现曾把非低精度输入的 compute type 默认设为
`CUBLAS_COMPUTE_32F`。对 float64 普通 matmul，这会触发
`CUBLAS_STATUS_NOT_SUPPORTED`；公开 2D/batched API 还以先转 float32、计算后转回
float64 的方式绕过错误，造成静默精度损失。

实现现在对 float64 选择 `CUBLAS_COMPUTE_64F` 和 double alpha/beta，覆盖普通、
strided-batched 与 acc matmul；Python 路由删除 float32 降精度绕行。batched 快路径
同时限定为相同浮点 dtype，其他输入继续走通用 dtype 语义。

## 复现

- 修复前完整 `tests/ops/test_matmul.py`：`1 failed, 12 passed`；float64 CUDA
  matmul 在 `cublasGemmEx` 返回 `CUBLAS_STATUS_NOT_SUPPORTED`。
- 修复前公开 2D 和 batched CUDA matmul 计算 `100000000 + 1` 均返回
  `100000000`，尽管输出 dtype 标记为 float64。

## 验证

- 真实 RTX 4090，CUDA 12.2：新增 2D、batched、acc float64 精度回归
  `1 passed in 3.99s`，三条路径均精确返回 `100000001`。
- cuBLAS forward/backward 与原 float64 tuner 回归：`4 passed in 98.58s`。
- 完整 matmul 模块：`13 passed in 15.87s`。
- float16/bfloat16 普通、batched 和 backward：`6 passed in 65.00s`。
- 仓库门禁：布局检查通过，`tests/structure` 为 `218 passed`。

首次 JIT 串行执行，并使用独立 `JITTOR_HOME`/`cache_name`；所有 CUDA 结果均在
声明的真实 device 上完成，没有使用 CPU fallback。
