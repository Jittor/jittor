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

最初的算子验收：`tests/backends/acl/` 214 通过 / 11 跳过 / 0 失败，
`backend_fallback=error` 下零 CPU 回退，全部跳过已按能力归类。
模型级验收：Transformers 4.56.2 + Torch shim，Qwen3-0.6B FP16 greedy
生成零回退且三次重复一致（需 `acl_allow_hf32=1`，RoPE 的 FP32
`inv_freq @ position_ids` 走显式降精度），详见
`docs/results/2026-09-22-qwen3-ascend310p-fp16.md`。
当前 2.0 代码的同步性能对比、输入长度与快照权重约束见
[310P3 性能报告](../results/2026-10-12-qwen3-ascend310p-performance.md)。

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

FP16 无梯度推理可通过标准 SDPA 入口使用推理卡的注意力算子：causal、无显式
mask 的等长 prefill 走 `PromptFlashAttentionV3`，设备侧补齐到 128 的倍数，
完成后裁回有效 query；单 query decode 走 `IncreFlashAttentionV4`。这两个
接口与缺失的 `FlashAttentionScore` 是不同算子。Transformers 模型加载时选择
`attn_implementation="sdpa"`；不符合已验证条件的调用继续使用原有组合路径。

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

## FP16 推理权重预转换

线性层权重不再更新时，可以在设备上一次性转换为 CANN 的 NZ 布局，避免每个
decode 步骤重复转换。先加载普通 checkpoint、迁移到 ACL 并调用 `eval()`，再显式
转换；独立 Torch frontend 的模型需要传入其 `Linear` 类型：

```python
import torch
from jittor.backends.acl.inference import freeze_linear_layers_

model.eval()  # model 已加载权重并位于 ACL
with torch.no_grad():
    model = freeze_linear_layers_(model, linear_type=torch.nn.Linear)
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
```

原生 Jittor 模型使用 `jt.no_grad()` 和默认的 `linear_type=jt.nn.Linear`。
转换返回的根模块必须保留：根本身是 Linear 时也会被替换。当前只支持 FP16、
ACL、无梯度推理，训练调用会报错；普通 Linear 子类保持原样。
带 forward/pre hooks 或实例 forward/execute 覆盖的 Linear 会拒绝转换，避免
静默丢弃自定义计算；转换失败时原模块树保持完整。

转换后的 `FrozenLinear` 保存独立的权重快照，源权重随后被修改不会影响快照。
它只注册 `packed_weight` 和可选 bias buffer，不保留可训练 weight Parameter；
因此 `parameters()` 的统计会减少，但计算使用的权重数量和精度不变。转换后
`state_dict()` 是打包格式；要加载普通 checkpoint，需重新构建普通模型、加载后
再次转换。权重转换耗时应独立记录，不计入稳态 token 吞吐。
原本与 Embedding 共享的 Linear 权重会保存独立快照，因此转换可能增加设备内存占用。

## 已知限制

- `x[i]`（纯整数索引）的反向在 310P 上可能读到未初始化字节：这是
  `setitem_inplace` 图优化把梯度打包进零基座后的调度层缺陷，与 ACL 核
  无关。公开路径（`GetItemACL`）已绕行；原生 op 路径在测试中声明为
  预期失败，详见 `agent/manuals/known-issues.md` KI-BACKEND-022。
- bool mask 的 `where`/`masked_fill` 在 310P 上静默无效；涉及非有限值
  的路径已改用 `nonzero`+`IndexPutImpl`。
- 310P 不声明训练能力：缺少的反向算子由组合实现补齐并逐一对拍，
  但未做端到端训练验证。

## 快速验证

```bash
source /path/to/cann-9.1.0/set_env.sh
JITTOR_TORCH_SHIM=0 PYTHONPATH=python \
  JITTOR_HOME="$ACL_TEST_STATE/native-cache" \
  python -m pytest -q tests/backends/acl/ \
  --ignore=tests/backends/acl/test_acl_torch_compat.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  JITTOR_HOME="$ACL_TEST_STATE/torch-cache" \
  python -m pytest -q tests/backends/acl/test_acl_torch_compat.py
```

`ACL_TEST_STATE` 应指向 lab 下的独立运行目录。两种前端分进程执行，不能混合
收集。当前树新增了算子及生命周期回归，214/11 是历史计数；跳过须有能力分类，
以实际执行的设备测试为准。若 FP32 matmul 测试报 `acl_allow_hf32` 相关错误，测试
内部已用 `_soc.supports_fp32_cube()` 自动加 opt-in，正常情况下不需要
手动设置。
