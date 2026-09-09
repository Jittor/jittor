# CUDA 窄整数与逻辑 reduction 收口

- Status: CUDA accepted; NPU verification remains open
- Last reviewed: 2026-08-23
- Commit: `7eb506fd`
- Owner: reduction and CUDA backend maintainers
- Review when: CUDA atomic implementation, reduction dtype inference, or NPU
  atomic coverage changes

## 结论

CUDA 原生 `sum`、`prod`、`max`、`min` 现在支持维护的窄整数 dtype
`uint8`、`int8`、`int16`。CUDA 逻辑 `all`/`any` 的输入仍为 bool，但 reduction
输出是 int32，因此使用 int32 的 `atomicAnd`/`atomicOr`，不再需要跳过 CUDA。

实现使用 32 位 packed CAS 更新一个 byte 或 halfword，保留同一 32 位 word 中其余
输出的内容；`uint8` 的 max/min 身份值也加入 CUDA numeric limits。Torch 兼容层继续
把窄整数 `sum` 提升到 int32，以保持既有 Torch dtype 策略。

## 环境

- Python 3.11.15; Jittor 1.3.11.0; g++ 12.3.0
- NVIDIA GeForce RTX 4090, compute capability 8.9; driver 595.84
- CUDA toolkit 12.2.140
- `CUDA_VISIBLE_DEVICES=2`; `use_parallel_op_compiler=0`
- 独立缓存和原始产物位于 `${JITTOR_LAB_ROOT}/_state/reduction-atomic/20260823/`

## 验证

以下验证均在真实 CUDA device 执行，CPU 对拍由同一 parity 测试独立运行：

| Gate | Result |
| --- | ---: |
| `uint8/int8/int16` 四种 reduction，全量与两个维度 | `REDUCE_MATRIX_OK` |
| uint8 多输出 packed-CAS 压力样本（16 x 257） | `PACKED_CAS_STRESS_OK` |
| bool `all`/`any` 多输出压力样本与全量 reduction | `BOOL_REDUCE_CUDA_OK` |
| `tests/backends/parity/test_device_parity.py` 四个整数 reduction | `4 passed, 223 deselected` |
| 同一 parity 的 bool `all`/`any` | `2 passed, 225 deselected` |
| CUDA OpInfo bool reference | `2 passed, 906 deselected` |
| `tests/ops/test_ops.py` collection for integer reduction | `16 collected` |

修复前的独立复现为 NVCC 编译错误：`atomicAdd(&(uint8*), uint8)` 没有匹配的
overload；修复后同一缓存中的 CUDA 编译和上述 parity 均通过。完整 CUDA 套件、NPU
和 ROCm 不在本报告范围内；NPU 窄整数仍由 `KI-BACKEND-001` 管理。
