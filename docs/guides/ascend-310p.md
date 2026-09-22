# 昇腾 310P：配置、能力边界与验证

310P 是推理卡：cube 单元只支持 FP16/INT8，没有 BF16，CANN 为 `ascend310p`
编译的算子包也小于训练卡。本页只记录与 [昇腾 910B 指南](ascend-910b.md)
不同的部分；驱动检查、CANN 环境、JIT 缓存隔离、editable 安装等步骤完全
相同，请先照那份指南完成。

## 已验证的基线

| 组件 | 验证版本 |
| --- | --- |
| 设备 | 昇腾 310P3（`npu-smi` 单卡，`/dev/davinci0`） |
| 架构 | Linux x86_64 |
| CANN toolkit 与 `ccec` | 9.1.0 |
| Python | 3.12.3 |
| NumPy | 1.26.4 |

验证命令与结果：`tests/backends/acl/` 214 通过 / 11 跳过 / 0 失败，
`backend_fallback=error` 下零 CPU 回退，全部跳过已按能力归类。
模型级验收：Transformers 4.56.2 + Torch shim，Qwen3-0.6B FP16 greedy
生成零回退且三次重复一致（需 `acl_allow_hf32=1`，RoPE 的 FP32
`inv_freq @ position_ids` 走显式降精度），详见
`refactor-wip/results/transformers/2026-09-22-qwen3-ascend310p-fp16.md`。

## 与 910B 的能力差异

Jittor 在运行时通过 `aclrtGetSocName` 识别芯片
（`backends/acl/kernels/_soc.py`），以下为 310P 实测差异，不是猜测：

| 能力 | 310P 状态 | 行为 |
| --- | --- | --- |
| BF16 | 无 `DT_BFLOAT16` 核 | 构造/转换即报错，带明确原因 |
| FP32 matmul/bmm/conv | cube 不支持 `KEEP_DTYPE` | 默认报错；`acl_allow_hf32` 显式降精度 |
| `StridedSliceAssignV2`、`EmbeddingDenseBackward`、`RotaryPositionEmbeddingGrad`、`FlashAttentionScore`、`NanToNum`、固定窗 `Maxpool`/`Avgpool`、`IndexPutImplAccumulate` | CANN 未为 310P 编译 | 自动走设备侧组合实现 |
| 比较核对非有限值 | 不遵循 IEEE（`x==+inf` 命中 `{nan,+inf}`） | `isnan`/`isinf`/`isfinite`/`isposinf`/`isneginf`/`nan_to_num` 用等价不变量组合实现 |
| `aclnnAdaptiveAvgPool2d` | 可用但内部低精度累加（~4e-5） | 与 torch_npu 经同一 CANN 核的精度一致 |
| `RotaryPositionEmbedding` | `seq_len==1` 返回垃圾值（tiling 边界） | 该情形自动改走 `x*cos+R(x)*sin` 组合前向 |
| `rfft`（DFT 常数 FP32 matmul） | 无可用精度路径 | 声明为不支持 |

## FP32 矩阵运算：显式降精度开关

与 `torch.npu.matmul.allow_hf32` 对应，Jittor 提供 `acl_allow_hf32`，
三种打开方式等价：

```python
import jittor as jt

jt.acl_allow_hf32 = 1                            # 模块属性
# 或
jt.flags.acl_allow_hf32 = 1                    # flags
# 或局部生效
with jt.flag_scope(acl_allow_hf32=1):
    y = a @ b
```

也可用环境变量 `acl_allow_hf32=1`。打开后 FP32 matmul/bmm/conv 以
`ALLOW_FP32_DOWN_PRECISION` 执行（内部 FP16 拆分，实测最大误差约 1e-3）。
**不打开时 FP32 矩阵运算会立即报错而不是静默降精度**——这是刻意的：
全精度是默认承诺，降精度必须显式同意。910B 上该开关同样生效（自愿
降精度），但 FP32 默认路径本身可用。

## 已知限制

- `x[i]`（纯整数索引）的反向在 310P 上可能读到未初始化字节：这是
  `setitem_inplace` 图优化把梯度打包进零基座后的调度层缺陷，与 ACL 核
  无关。公开路径（`GetItemACL`）已绕行；原生 op 路径在测试中声明为
  预期失败，详见 `agent/manuals/known-issues.md` KI-BACKEND-011。
- bool mask 的 `where`/`masked_fill` 在 310P 上静默无效；涉及非有限值
  的路径已改用 `nonzero`+`IndexPutImpl`。
- 310P 不声明训练能力：缺少的反向算子由组合实现补齐并逐一对拍，
  但未做端到端训练验证。

## 快速验证

```bash
source /path/to/cann-9.1.0/set_env.sh
PYTHONPATH=python python -m pytest -q tests/backends/acl/
```

期望结果：214 通过、11 个按能力归类的跳过、0 失败。若 FP32 matmul 测试
报 `acl_allow_hf32` 相关错误，说明环境未按本页第二节打开开关——测试
内部已用 `_soc.supports_fp32_cube()` 自动加 opt-in，正常情况下不需要
手动设置。
