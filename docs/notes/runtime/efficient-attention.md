# CUTLASS 高效注意力

[核心概念](../index.md) · [English](efficient-attention.en.md)

CUDA 后端可以直接编译 PyTorch 的 CUTLASS memory-efficient attention 头文件，
不导入或链接原生 Torch。使用前配置两个独立的源码目录：

```bash
export JITTOR_EFFICIENT_ATTN_SRC=/path/to/pytorch
export JITTOR_CUTLASS_PATH=/path/to/cutlass
```

第一个目录可以是 PyTorch 源码根目录、`aten/src` 或安装包的 `include` 目录；必须包含
`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`。第二个可以是
CUTLASS 根目录或 `include` 目录。编译还需要 CUDA、C++17 编译器和 pybind11 头文件。
保留源码版本与许可证；不把外部源码复制进 Jittor 仓库。初次编译串行执行，生成物
位于当前 JIT 缓存中，缓存身份包含源文件、头文件、工具链和目标架构。

源码内核要求 CUDA 架构 SM80 或更新。当前入口接受四维 Q/K/V，支持 FP32、FP16、BF16，head 和 value 维度是 8 的倍数且
不超过 128。布尔或加性 mask 可广播到 `[batch, heads, queries, keys]`；支持矩形序列、
左上对齐的 causal mask、dropout 及 Q/K/V 的一阶梯度，不保存完整注意力分数矩阵。
加性 mask 与 query 的 dtype 必须一致；Torch CUDA autocast 会按所选精度转换浮点输入。
mask 梯度、未覆盖的 head 布局或 dtype 由其它已启用路径处理；只允许该后端时会报错。

原生接口的 BF16 融合概率乘积需要 `allow_reduced_precision=True`；默认保留更严格的
FP32 中间精度契约。Torch 兼容前端按其融合注意力约定选择该精度。FP32 使用上游
`OpMultiplyAddFastF32`，与原生 Torch 的该 CUDA 内核一致；矩阵乘法的 TF32 开关不会
改写该分解算法。数值验证仍需独立高精度参考。

在已启用的 Torch 兼容前端中，可以明确选择后端：

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

`sdpa_kernel` 与旧版 `torch.backends.cuda.sdp_kernel` 保存、应用并恢复后端开关。
显式 math 选择绕过融合注意力；没有可用的已选择内核时抛出错误。自动选择保留默认策略。
接口接受后端集合；自定义优先顺序尚未实现。

前向、反向、工作区和随机数都由图节点持有。dropout 反向读取前向保存的 Philox 状态，
CUDA graph 重放通过设备状态刷新随机数。测试应核对输出、全部输入梯度、全遮蔽行和
更换输入/上游梯度后的图重放，并确认 dropout mask 每次刷新。
