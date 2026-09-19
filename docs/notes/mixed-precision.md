# 混合精度训练

用 float16 或 bfloat16 训练，是拿数值范围换显存和速度。这一页说明 Jittor 两种
模式下各自怎么开、什么时候需要 loss scale、以及哪些地方仍然按 float32 累加。

混合精度有两件互相独立的事，分开理解就不会踩坑：

1. **哪些算子降到半精度**——由执行器的档位决定（`jt.flags.auto_mixed_precision_level`，
   torch 兼容模式下是 `torch.autocast`）。
2. **梯度怎么不被冲掉**——由 loss scale 决定（`jt.amp.GradScaler`，torch 兼容模式下
   是 `torch.amp.GradScaler`）。

只有 float16 需要第二件。bfloat16 的指数位和 float32 一样宽，梯度不会下溢。

## float16 为什么需要 loss scale

float16 的最小正规数是 6e-5。一个在 float32 里再普通不过的梯度——比如 1e-7——
在 float16 里就是 0。反向传播因此静悄悄地丢掉所有小梯度，模型不再学习，而且不报错。

把 loss 乘一个大常数再做反向，整个梯度分布被抬进 float16 的范围；反向之后把梯度
除回去，参数更新完全不变。这就是 loss scale 的全部内容。

常数不能写死：太大溢出成 inf，太小仍然下溢，而且哪边合适取决于模型和训练进行到
哪一步。所以要动态选——从高处起步，任何一步出现非有限梯度就减半并**跳过这一步**
（那批梯度已经是垃圾），连续若干步干净之后再翻倍。

## 原生模式

```python
import jittor as jt

model = MyModel()
for p in model.parameters():
    p.assign(p.float16())

opt = jt.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scaler = jt.amp.GradScaler()          # bfloat16 训练时传 enabled=False

for x, y in data:
    loss = loss_fn(model(x), y)
    opt.backward(scaler.scale(loss.float32()))   # 注意 .float32()，见下
    scaler.step(opt)                  # 先 unscale，梯度非有限就跳过这一步
    scaler.update()                   # 增长或回退 scale
```

**`scale()` 要作用在 float32 的 loss 上。** 乘法保持输入的 dtype，默认 scale 是
65536，而 float16 的最大值是 65504——直接 `scaler.scale(loss)` 在 loss 还没开始反向
时就溢出成 inf。之后每一步都发现非有限梯度、跳过、把 scale 减半，直到 scale 掉到 1，
训练从头到尾没有发生，而且不报错。torch 有同样的行为（它的 `_scale` 是 float32
张量，半精度张量乘 float32 张量会提升，乘 python float 不会），只是 torch 用户很少
遇到：autocast 让 loss 函数留在 float32。把模型整体转成 float16 的原生写法就会遇到。

`loss.float32()` 只影响 scale 这一步；反向从第一个算子起仍然是 float16。scale 大到
必然溢出 float16 时 `scale()` 会给一次 `RuntimeWarning` 说明这件事。

`enabled=False` 时每个方法都是恒等操作，所以同一个训练循环可以不加分支地跑在任意
精度下。

需要在 unscale 之后、step 之前插入别的操作（梯度裁剪是最常见的理由，它必须看到真实
量级的梯度），就自己调 `scaler.unscale_(opt)`，`step` 会知道不用再做一次。

| 方法 | 作用 |
| --- | --- |
| `scale(loss)` | 乘上当前 scale；也接受 Var 的列表/元组/可迭代对象 |
| `unscale_(opt)` | 原地把该优化器的梯度除回去，并记录是否出现非有限值 |
| `step(opt)` | 需要时先 unscale；梯度非有限则跳过并返回 `None` |
| `update()` | 溢出则乘 `backoff_factor`（不低于 1），否则连续 `growth_interval` 步后乘 `growth_factor` |
| `state_dict()` / `load_state_dict()` | 五个键，键名与 torch 一致，checkpoint 可互相读取 |

档位（哪些算子降精度）是执行器上的开关，用 `jt.flag_scope` 限定作用域：

```python
with jt.flag_scope(auto_mixed_precision_level=5):
    ...
```

档位 4/5/6 让参数保持 float32 而把梯度和中间结果降到 float16；1–3 基本不降精度。
档位只改 dtype 推断，不做 loss scale——两件事要分别开。

## torch 兼容模式

`torch.autocast` 和 `torch.amp.GradScaler` 都在，签名与 torch 一致：

```python
import torch

scaler = torch.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

`torch.amp.GradScaler` 就是 `jt.amp.GradScaler` 加上 torch 的参数顺序——同一套算法，
两个入口，缩放策略的修改不会只落在一边。`torch.cuda.amp.GradScaler` 和
`torch.cpu.amp.GradScaler` 是它的子类，保留 torch 2.3 之前 `init_scale` 在前的旧顺序。

`is_autocast_enabled()` 不带参数问的是 **cuda**，这和 torch 一致：在
`with torch.autocast("cpu")` 里面它返回 `False`，要问 cpu 就传 `"cpu"`。

一处已知差异：`torch.autocast(dtype=torch.bfloat16)` 在全 float32 的区域里会算成
float16，因为 jittor 的 amp register 只在已有 bfloat16 操作数时保持 bfloat16。要留在
bfloat16，用 `.to(torch.bfloat16)` 把模块本身转过去。运行时会给一次 `RuntimeWarning`
说明这件事。

## 哪些地方仍然按 float32 累加

降精度只降**存储和乘法**，累加一律更宽——这和 torch 的规则一致，也是半精度能用的
前提：

- 矩阵乘和卷积：cuBLAS/cuDNN 用 `CUBLAS_COMPUTE_32F`，走通用 `(a*b).sum(k)` 回退路径
  时同样用 float32 中间量。
- `sum` / `mean` / `std` / `norm` / `softmax` / `log_softmax` / `cumsum`：float32 累加器。
- LayerNorm / GroupNorm / InstanceNorm：统计量（均值、方差、rsqrt）在 float32 里算。
- 归约的**输出** dtype 保持输入的半精度，不会悄悄变成 float32——`x.sum(-1)`、
  `x.max(-1)`、`x.prod(-1)` 对 float16 输入都返回 float16。

CPU 和 CUDA 给同一个答案。实测（误差相对于同一组已舍入输入上的 float64 闭式解，
旁边是真实 torch 2.13 在同样输入上的数）：

| 算子 | dtype | 设备 | Jittor | torch 2.13 |
| --- | --- | --- | --- | --- |
| `matmul`（K=512） | fp16 | cpu | 7.759e-3 | 7.759e-3 |
| `matmul` | bf16 | cpu | 6.175e-2 | 6.175e-2 |
| `layer_norm` | fp16 | cpu/cuda | 9.750e-4 | 9.750e-4 |
| `sum` / `mean`（4096） | bf16 | cpu | 1.780e-3 | 1.780e-3 |
| `softmax` | bf16 | cpu | 8.677e-4 | 8.677e-4 |

`max` / `min` 在半精度下和 float32 用同一个表达式，NaN 的传播规则因此在两个操作数
位置、两种设备、三种浮点宽度上都一致——`jt.maximum(5, nan)` 是 NaN，含 NaN 的
`x.max()` 也是 NaN。

## 端到端的期望

同一个模型、同一组初值、同一个 schedule，最终 loss：

| 精度 | 最终 loss |
| --- | --- |
| float32 | 0.13503 |
| bfloat16 | 0.13867 |
| float16 + GradScaler | 0.13501 |
| `auto_mixed_precision_level` 4/5/6 | 0.13318–0.13403 |

真实 torch 2.13 在同样条件下 float32 / bf16 autocast / fp16 autocast+GradScaler 的
分布是 0.4% 和 3.4%。半精度训练收敛到和 float32 差不多的地方是正常的；差出一个
数量级说明某处在半精度里累加，那是 bug，不是精度的代价。

## 相关

- [数值契约](numerics-contract.md)：NaN、无穷、次正规数、归约精度的总规则。
- [float32 累加精度](float32-precision-policy.md)：float32 操作数自己的 tf32/bf16 档位。
- [设备与放置](device-placement.md)：`.cuda()` / `.cpu()` 与优化器状态在哪张卡上。
