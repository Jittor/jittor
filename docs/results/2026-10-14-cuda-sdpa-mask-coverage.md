# CUDA SDPA 融合支持矩阵与 mask 训练

- 状态：下述 eager 矩阵和边界已在真实 CUDA 验证；fusion 不等同于性能结论。
- 日期：2026-10-14。
- 同步基线：`3d6aac9db75fb7018fa51ae71b9647bf130929dd`，包含此前 AMP 修复和远端
  `7ed9b3b52b658d2a3c1ac390eba09087c5e80f95`；修复由本报告所属提交确定。
- 维护者：CUDA attention 与 Torch 兼容层维护者。
- 平台：RTX 4090 / SM89，驱动 580.178.04，CUDA 编译器 12.2.140、GCC 11.5、
  Python 3.11；实际 cuDNN 9.24.0、frontend 1.30.0。
- 复查条件：cuDNN/frontend 版本、flash source 配置、attention dispatch、mask 数据域、
  AMP 或 capture/replay 规则变化。未验证其他 GPU、ROCm、ACL/NPU。

## Eager 支持矩阵

每个输入组固定 batch=1、heads=2，形状按 `(Lq,Lk,head_dim)`：

| 名称 | 形状 |
| --- | --- |
| short_self | `(32,32,32)` |
| self | `(128,128,64)` |
| cross | `(33,47,64)` |
| wide_self | `(70,70,128)` |
| head160 | `(32,32,160)` |

每组枚举 FP32/FP16/bf16 × 无 mask/bool padding/additive dense × causal 开/关 ×
推理/训练，共 **180 个执行格**。bool padding 为 `[1,1,1,Lk]`，末尾四分之一 key
不可见；additive dense 为 `[Lq,Lk]` 的有限 FP32 bias。输入为确定性 NumPy 正态
样本，Q/K/V 和 upstream gradient 的标准差 0.5，additive bias 的标准差 0.2。

下表是同步输出、执行反向并检查 Q/K/V 梯度后记录的实际 provider。选中函数后
返回 `None` 不计为 fusion；复合实现也必须在 CUDA 上执行。

| dtype / head_dim | mask | causal | 推理 | 训练 |
| --- | --- | --- | --- | --- |
| FP32 / 32、64、128 | 三种 | 两种 | float32 tile fused | float32 tile fused |
| FP32 / 160 | 三种 | 两种 | composite | composite |
| FP16、bf16 / 32、64、128 | 三种 | 两种 | cuDNN fused | cuDNN fused |
| FP16、bf16 / 160 | 三种 | 两种 | cuDNN fused | composite |

基线为 48 格 float32 tile fused、56 格 cuDNN、76 格 composite；修复后为
**48、108、24**，其中 **52 格从 composite 转为实际 cuDNN fusion**。剩余 24 格
分别来自 float32 tile 的 head_dim≤128 限制，以及该平台 cuDNN backward 对
head_dim=160 的拒绝。后者的真实 probe 返回
`hidden_dim should be less than or equal to 128`，不是静态注册推断。

本次没有配置官方 FlashAttention source，native flash selector 实际未返回结果；
此矩阵不声称验证了 flash extension 的融合能力。没有新 adapter 或依赖加入仓库。

## 修改与其他执行场景

cuDNN backward 接口增加与 forward 一致的 constant bias；bias 用 FP32 保持计算域，
避免 `-1e9` 等有限值在转为 FP16 后变成 Inf。支持 1D 和查询轴广播的 padding
mask。mask 准备固定在 FP32 AMP 域，非 view mask 更新会使缓存失效。

全 `-inf` 行，以及 causal 与 mask 相交后没有可见 key 的行，先给 cuDNN 有限的
内部统计，再在外层把该行输出与 upstream gradient 置零；输出与 dQ 严格为零，
dK/dV 与独立闭式参考一致。额外回归还验证了 grouped heads 的 constant mask。

| 场景 | 实际路径与证据 |
| --- | --- |
| 常规 eager mask | 上面的 180 格及其 Q/K/V gradients；不包含 capture 状态 |
| 整行巨大有限 bias | 显式 softmax composite。新融合候选对 `-1e9` 整行的 dQ 曾出现约 31 倍峰值误差；GPU 上检测到 FP32 row offset 吸收 `log(Lk)` 时拒绝并缓存该结果，FP16/bf16 与真 Torch 复验恢复到约 `5.6e-4` / `4.5e-3` |
| 同一行部分列为 `-1e9`、仍有可见 key | padding mask 继续 cuDNN fusion；不会按最大绝对 mask 值一概拒绝 |
| FP16/bf16 cuDNN mask 的 capture/replay tracing | 保留 graph-only composite，避免 mask 值域准入的 host scalar read；四次不同输入/不同 padding mask 实测 `captured=1,replayed=2`，输出和全部梯度对闭式参考通过 |
| learned mask | cuDNN bridge 不返回 mask gradient，明确拒绝，保留原路径 |

Capture 场景与 eager 比例分别验收；180 格比例不代表捕获状态的融合率。以上 replay
证据对应框架图重放，未据此声明 CUDA Graph recording。无 mask 的既有 cuDNN
forward/backward 路径保持使用原有选择逻辑。

## 独立参考与验证

真实 PyTorch **2.11.0+cu128** 在独立进程、同一张 CUDA 卡执行 MATH SDPA；确认
`torch._C` 是真实二进制扩展且没有 shim 安装标记。Torch MATH 拒绝同时传入 mask
与 `is_causal=True`，这些组合的 oracle 显式把 top-left causal 合入 mask 后调用
`is_causal=False`；这是 Jittor 当前组合语义的数学参考，不声明直接 Torch API
支持该参数组合。

Jittor 运行启用 `forbid_backend_fallbacks()`，同步检查输出及三个梯度的实际
`location()==device`。误差为 `max(abs(got-want))/max(abs(want))`：

| dtype | 输出最大误差 | 三种梯度的最大误差 |
| --- | --- | --- |
| FP32 | `7.66e-7` | `7.80e-7` |
| FP16 | `7.73e-4` | `9.46e-4` |
| bf16 | `6.37e-3` | `7.64e-3` |

- CUDA cuDNN 定向：[test_cudnn_fused_attention.py](../../tests/backends/cuda/test_cudnn_fused_attention.py)，
  **20 passed，0 skipped，13.83 秒**；包含新 mask 边界、AMP、changed-input replay
  和既有大序列内存检查。
- Native 公共数学及 flash decline：`test_sdpa_memory_efficient.py` 与
  `test_sdpa_cuda_flash_kernel.py`，**26 passed，0 skipped，74.65 秒**，CPU/CUDA。
- Torch 公共 no-mask、causal、FP16 mask training、zero masked rows 定向：
  **4 passed，0 skipped，40.50 秒**，CPU/CUDA。另从 `import torch` 的 CUDA
  FP16/bf16 bool-mask 路径验证 `autograd.grad`、四个 golden 数组、实际 cuDNN
  provider 和 residency，运行时默认 `jt.flags.use_cuda=0`，由实际操作数确定 device。
- `bash tools/check_repo_layout.sh`、`git diff --check` 和 import layering 通过，
  循环内模块保持 165 个。
- 完整 `tests/structure`：**1427 passed、4 skipped、3 failed，234.42 秒**。
  一项为此基线的 `DropoutACL` 梯度登记过期；另外两项要求把新增结果报告接入
  results toctree 并生成 `MANIFEST.in`。这些共享文件由主任务整合时维护。两项
  skip 是私有 venv 未安装 core distribution 的解析检查，另两项为声明跳过。
  不据此声明完整结构门禁通过。
- Core 门禁由主任务在整合后统一执行；本任务未运行 full。

复现命令采用独立环境、`JT_CUDA_KERNEL_MATH=strict`、单卡
`CUDA_VISIBLE_DEVICES=<gpu>`、隔离 HOME/JITTOR_HOME/TMPDIR、首次 JIT 串行：

```bash
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICES=cuda PYTHONPATH=python python -m pytest -q \
  tests/backends/cuda/test_cudnn_fused_attention.py
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICES=cpu,cuda PYTHONPATH=python python -m pytest -q \
  tests/nn/test_sdpa_memory_efficient.py tests/nn/test_sdpa_cuda_flash_kernel.py
```

未版本化的原始 script、90 份 Torch golden arrays、JSON 和日志放在
`$JITTOR_LAB_ROOT/_state/refactor-goal/sdpa/`。矩阵 script 通过 `torch` / `jittor`
两种 runtime 参数分进程执行，`--root` 指向 golden 目录，Jittor 的 `--output`
指向结果 JSON。SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| matrix.py | `0557a06a8573f8da3edbe99b738a40db43bf5943c7ef23d91ae66730d311a4e7` |
| baseline.json | `4bcaee31e0b24178b7adc60e6b4f7bbf5214791a24eedd6c1b463ea0a43f538e` |
| final.json | `0e9ab7b41ae58f481f754b0842a49265da26ee75d68be59bb80799f71855bcdf` |

这是一张有限样本、dtype 和 device 的支持矩阵。padding mask 当前会转换为
`[bias_batch,bias_heads,Lq,Lk]` 的 FP32 bias，尚未使用压缩 sequence-length mask。
本报告没有做 latency 对比或推出显存比例，保留现有 `APPROXIMATE` fidelity；
也未把未修改的 float32 极端有限 mask 或逐元素 parity 已知问题宣称为已解决。
