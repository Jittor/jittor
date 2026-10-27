# 原生张量并行线性层

状态：有界的原生 column/row 线性实现。验证基线与设备范围记录在
[维护者结果](../results/2026-10-15-native-tensor-parallel.md)。当集合通信伴随算子、原生
优化器同步或 checkpoint 变化后重新检查。

`jittor.distributed.tensor_parallel.ColumnParallelLinear` 只保存本 rank 的输出行与本地
bias。`RowParallelLinear` 只保存输入列；它的 bias 是复制的，在部分输出相加之后再加上。
特征数必须能被 WORLD size 整除。原生 FP32 CPU 单进程，以及两 rank CUDA/NCCL 配上下面
的 BF16 CUDA 训练协议，是这里的验证范围；这些不是通用的 DTensor、序列并行、
张量并行 attention 或任意分布式算子。

```python
from jittor import nn
from jittor.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class MLP(nn.Module):
    def __init__(self):
        self.up = ColumnParallelLinear(256, 1024, gather_output=False)
        self.down = RowParallelLinear(1024, 256, input_is_parallel=True)

    def execute(self, x):
        return self.down(self.up(x).tanh())
```

在 CUDA 上，构造这些层之前先初始化原生 NCCL WORLD。原生启动器可以用
`jt.compile_extern.get_library("nccl", load=True)` 加载 NCCL 后端，并调用
`jittor.distributed.init_process_group("nccl", init_method="env://")`。
CPU 单进程执行不需要进程组设置。

每个 TP rank 都提供相同的输入与复制后的 loss，并且按相同顺序调用前向、反向与
checkpoint。这是模型并行，不是数据并行平均。各 TP rank 之间使用不相关的输入批违反这个
契约。`gather_output=True` 产出完整复制的 column 输出；`input_is_parallel=False` 把复制
的 row 输入切散。这些选项既可支持单独使用某一层，也支持 column/非线性/row 成对使用。

| 边界 | 前向 | 反向 |
| --- | --- | --- |
| 复制的 column 输入 | 恒等 | 汇总本地输入梯度贡献 |
| 复制的 row 输出 | 汇总部分输出 | 恒等 |
| gather column 输出 | gather 本地特征切片 | 切出匹配的本地梯度 |
| scatter row 输入 | 切出匹配的本地特征 | gather 本地输入梯度 |

这些 TP 专用的伴随算子使用原生 Function 与原生集合通信。特别地，复制的 row 输出在反向
中不会再做一次 all-reduce。原生优化器同步能识别 TP 参数，并对这些分片跳过旧的数据并行
归约与周期性广播。优化器算术仍由既有的原生优化器拥有；维护者的训练验证使用原生动量
SGD 与带 weight decay 的 AdamW。由 DDP 管理同一批 TP 参数是明确不支持的；独立的数据
并行 mesh 轴与混合进程组需要各自的实现。

原生 `checkpoint.save_training_state` / `load_training_state` 路径让真正分片的模型与
优化器叶子保持 rank 本地，并在修改之前校验 rank、world、方向、本地/全局形状与 FQN
元数据。只有布局里的 rank 字段是 rank 本地的；其余等分元数据必须跨 rank 一致，不兼容的
全局形状会使发布失败。新建的对象必须具有相同的切分与优化器绑定。这保持了训练状态；
它不会把 checkpoint 重新分片到另一个 world size。checkpoint 载荷包含真实的本地张量，
不会 gather 完整权重。复制的 row bias 及其优化器缓冲仍保留复制一致性检查；发生漂移的
bias 必须让 checkpoint 发布失败。优化器类、有序 FQN、分组超参数与 step 元数据仍然复制
并校验。

`partition()` 返回原始的所有权元数据，不暴露可变的实时存储元数据。参数在使用期间必须
保持其切分形状与 WORLD 设备。不均匀切分、空特征 rank、子组 TP、DP+TP、ZeRO+TP 或
FSDP+TP 组合、分片词表 loss、超出已验证 BF16 协议的混合精度策略、更大 world 与多主机
验收，仍然未验证或落在这一有界实现之外。
这里不声称全进程显存峰值或吞吐有所改善。

## BF16 CUDA 训练配 FP32 状态

[AdamW/BF16 结果](../results/2026-10-15-native-tensor-parallel-adamw-bf16.md)验证了 CUDA
线性乘积的实际 BF16 输入与权重操作数、FP32 master 参数、输入/参数梯度以及两个 AdamW
动量矩。原生混合精度使用既有的 flag；TP 不额外引入 autocast 上下文或优化器算术：

```python
import jittor as jt

optimizer = jt.optim.AdamW(model.parameters(), lr=0.003, fused=False)
with jt.flag_scope(auto_mixed_precision_level=3):
    with jt.flag_scope(amp_reg=jt.amp_flags.prefer_bfloat16):
        prediction = model(inputs)
    loss = ((prediction.float32() - targets.float32()) ** 2).mean()
    optimizer.backward(loss)
    optimizer.step()
```

让档位 3 在前向、反向与优化器更新期间都保持生效：核心用它把梯度按每个原始变量的 dtype
发布。只在前向选 BF16 偏好并不能保证 FP32 的参数梯度。只在 forward 期间选 BF16 偏好；
loss 与优化器算术保持 FP32。复制的 row bias 在相加之前被显式转换成乘积的计算 dtype，
用的是与普通函数式 linear 相同的原生辅助函数。row 归约携带 BF16 的部分输出；column
伴随算子汇总 FP32 的输入梯度贡献。

维护中的小 MLP 验收把全部五个训练步与一个独立的、未分片的真实 Torch 模型对拍，然后检查
在第三步保存、恢复新对象之后的精确原生态续跑。BF16 有一份单独冻结、针对特定 fixture 的
精度预算，由真实 Torch 全模型与同一套两 rank TP 分解标定得到。FP32 对拍保持它的严格容差。
BF16 验收不承诺跨框架逐位相等，也不给出通用的误差上界。

CPU 单进程测试覆盖对应的输出、梯度/状态与 checkpoint 语义。原生 CPU AMP 的通用收缩
可能保留更宽的输入操作数；这些测试不证明真实的 BF16 CPU GEMM 输入转换。CUDA GEMM 的
操作数 dtype 与存储在任何 NumPy 读回之前就被检查。其它 AMP 策略、混合 mesh 与更大或
多主机的 WORLD 仍在此验证之外。
