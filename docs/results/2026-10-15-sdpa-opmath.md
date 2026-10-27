# SDPA 低精度累加与高阶梯度

- 状态：本文限定的 CPU/CUDA 数值及梯度回归已验证。
- 日期：2026-10-15；维护者：native attention 与 CUDA 后端维护者。
- 同步基线：`626dcd6823dfd4ee45200374fa447c39e91797f2`；修复由本报告所属提交确定。
- 复查条件：attention 数学路径、Function tape、AMP、cuDNN 或输入存储语义变化。

原 math 路径在低精度 QK、概率乘 V 和 backward 中提前舍入。保存的真实
`[1,4,2,8]` Q/K/V 复现中，BF16 前向最大归一化误差超过既有 `0.002` 预算。
现在 FP16/BF16 的 Q/K/V、softmax 及 backward 中间计算使用 FP32，仅在公开
输出及返回梯度处转回原 dtype；局部关闭 AMP，离开作用域后恢复调用方设置。
FP64 保持 FP64。

Backward 从概率与概率梯度重算行内积。每次 Function 调用在 tape 前保存原图
上的稳定 reshape holder，使二阶梯度连接到原输入，并避免调用方 `assign`
重绑定或复用同一 Function 实例覆盖前一次状态。这不是底层存储的深复制。

真实 cuDNN 对上述 BF16 小 head 输入可执行，但输出误差 `0.002513`、dQ
误差 `0.03273`，超过原前向/梯度 `0.002/0.01` 预算。因此 BF16 head_dim<32
统一走 native FP32 math，包括推理与 GQA；这是数值准入策略。FP16 和既有
D32 起的 cuDNN 准入条件保留。

独立二进制 Torch 2.13.0+cu130 的 forced-MATH 原始数组保持不变。
最终 CPU 的 FP16/BF16 默认及显式 math 输出和三路输入梯度全部逐位相等；
CUDA 的 FP16 显式 math、BF16 默认及显式 math 同样逐位相等。CUDA FP16
默认 cuDNN 路径输出/梯度最大归一化误差为 `0.000628/0.004728`，满足原预算。
这些是固定输入的正确性结果，不代表完整 Omni 模型验收或性能结论。

验证包括新增八项 CPU 用例，以及真实 CUDA 上新旧 attention 与完整 cuDNN
定向共 39 项，全部通过、无跳过；CUDA 显存门槛保持通过。用例覆盖 bool/additive
mask、全遮蔽行、AMP、小 head 公共路径、输入重绑定及同实例复用。
FP64 一阶和有限差分 HVP 用例明确运行在 CPU。Core 为 322 passed、42 skipped、
1 既有 xfail。

使用独立 CPU/CUDA 编译缓存，选择实际可用 CUDA 工具链；混合设备测试全局默认
CPU，由测试显式切换 CUDA，新增设备计算均校验 residency 并禁止 backend fallback：

```bash
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICE=cuda python -m pytest -q \
  tests/nn/test_sdpa_opmath.py tests/nn/test_sdpa_memory_efficient.py \
  tests/backends/cuda/test_cudnn_fused_attention.py
python tools/run_test_suite.py --tier core --serial-compile
```

平台为 Linux x86_64、Python 3.11、RTX 4090/SM89、nvcc 12.2.140。
原始日志、独立 Torch 数组和逐字段对拍在未版本化的
`$JITTOR_LAB_ROOT/_state/sdpa-opmath-integration-626/`；原 oracle 位于
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/saved-sdpa-v1/`。
其他 GPU、ROCm、NPU 及完整模型不在本次验收范围。
