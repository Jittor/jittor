---
name: cuda-reduction-strategy-comparison
description: 比较 Jittor CUDA 归约的三条策略（每线程原子、warp shuffle 归约、块内共享内存树形归约）该用哪条，以及怎么把它们量准。用于改 WarpReducePass / SharedReducePass / ReduceTuner、调 para_opt_level、或怀疑某个归约 kernel 慢的场合。含四种配置的实测数、切换开关的准确写法、以及为什么手写的 wall-clock 微基准在这里必然测错。
---

# CUDA 归约：三条策略，怎么选，怎么量

## 谁在改那条 atomicAdd

生成的 CUDA 归约 kernel 结尾是每个线程把私有部分和直接原子加到输出元素上。
有三个 pass 会动这条语句，按 `pass_manager.cc` 里的顺序：

| 顺序 | pass | 做什么 | 开关 |
| --- | --- | --- | --- |
| 117 | `AtomicTunerPass` | 把累加提到循环外，每个输出地址一次原子而不是每次迭代一次 | 总是开 |
| 118 | `SharedReducePass` | 重排线程范围让一个 block 覆盖整个归约，再以 warp shuffle → 每 warp 一个共享值 → 首 warp shuffle 折叠，`if (threadIdx.x==0)` 写一次 | **`para_opt_level >= 4`，默认 3，即默认关** |
| 120 | `WarpReducePass` | warp 内 `__shfl_down_sync` 折叠，每 warp 一次原子 | 默认开，`no_warp_reduce` 关 |

**先确认你以为在跑的那条真的在跑**——这三条的开关条件都不写在一起：

```python
jt.flags.para_opt_level          # 3 = 默认，SharedReducePass 直接 return
jt.flags.compile_options = {"no_warp_reduce": 1}   # 关 WarpReducePass
```

判据看**生成的源码**，不要看名字：`shared_reduce<` 在不在、`_wr_mask` 在不在。

## 实测数（RTX 4090，float32，profiler 报的设备时间，每 kernel 平均）

四维张量沿 `(0,2,3)` 归约（diffusers UNet 反向里最多的那一类）：

| 形状 | 只有原子 | + warp（今天的默认） | + 块内共享内存（lvl 4） |
| --- | --- | --- | --- |
| 8×384×32×32 | 157.0us | **15.7us** | 25.3us |
| 8×128×64×64 | 92.1us | **14.0us** | 31.3us |
| 16×192×32×32 | 159.2us | **15.0us** | 25.3us |
| 32×64×56×56 | 171.0us | **18.1us** | 34.8us |

结论，会反直觉，记下来省得重测：

1. **warp shuffle 比块内共享内存快 1.6–2.0 倍**。共享内存版本要一个 1024 项的
   `__shared__` 数组、六次 `__syncthreads()`，尾部那段 `warpReduce` 还走 volatile
   共享内存（每步一次读一次写）；shuffle 版本五条 `__shfl_down_sync`，全在寄存器里。
   「块内树形归约」听起来更高级，实测更慢。
2. 两者都远好过不优化（6–10 倍）。所以**「SharedReducePass 从没生效过」不等于
   「归约没优化」**——问题在 9eb696d9 之后已经被 WarpReducePass 解掉了。
3. 精度相反：块内树形归约 relerr ~2.3e-7，warp ~3.5e-7，纯原子 ~1.8e-6。
   要可复现的求和顺序时这一条才是理由。
4. **两个 pass 会改同一条语句。** SharedReducePass 留下的是
   `if (threadIdx.x == 0) atomicAdd(...)`，WarpReducePass 匹配的是
   `startswith(code, "atomicAdd")`，会把这条已经只剩一个活跃 lane 的原子再包一层
   shuffle。运行期 `__activemask() != 0xffffffff` 会走回退，**结果对但全是死代码**，
   实测每 kernel 多花 1.3–1.9us。判据：lvl 4 的源码里同时出现 `shared_reduce<`
   与 `_wr_mask` 就是踩上了。

## 3.22 两级混合路径复测（2026-09-03）

把旧的 1024 项共享树改成“两级 warp shuffle，中间只交换每 warp 一个值”后，level 4
路径从六次 `__syncthreads()` 降到两次。GPU profiler、30 次、同一输入的直接 A/B 为：

| 形状 | 默认 warp-only | 两级混合 level 4 | 混合 / warp |
| --- | ---: | ---: | ---: |
| 8×384×32×32 | 17.53us | 17.21us | 0.982 |
| 8×128×64×64 | 16.80us | 16.32us | 0.972 |
| 16×192×32×32 | 18.00us | 16.39us | 0.910 |
| 32×64×56×56 | 21.90us | 25.53us | 1.166 |
| 合计 | **74.23us** | 75.45us | **1.016** |

两边相对 NumPy 误差均不超过 `3.7e-7`，生成源码分别命中 `_wr_mask` 与
`shared_reduce<`/`if (threadIdx.x == 0)`。混合路径前三形状略快，但第四形状退化
16.6%，合计慢 1.64%，所以**不得改成默认**；它保留在 `para_opt_level >= 4` 供继续实验。
3.22 只有在代表形状不退化并满足完整 UNet 归约性能终点后才能关闭。

## 怎么量：不要写 wall-clock 微基准

Jittor 是惰性图。`t0=time(); y=x.sum(); t1=time()` 量到的是**建图**的时间，
kernel 还没跑；加一个 `.sync()` 又会把编译时间算进去。手写的微基准在这里
**必然**测错，而且错得像是「优化生效了」。

用 profiler 的设备时间，并且把编译赶到测量之外：

```python
jt.flags.use_cuda = 1
a = jt.random(shape); a.sync()
jt.reduce(a, "add", dims).sync()      # 先编译一次，不计入
jt.sync_all(True)
with jt.profile_scope(rerun=0) as rep:
    for _ in range(30):
        jt.reduce(a, "add", dims).sync()
    jt.sync_all(True)
hdr, rows = rep[0], rep[1:]
ni, ci, ti = hdr.index("Name"), hdr.index("Count"), hdr.index("TotalTime")
for r in rows:
    if "reduce" in r[ni]:
        print(float(r[ti]) / int(r[ci]) / 1000.0, "us")
```

三个要点：

- **`rep[0]` 是表头**，`rep[1:]` 才是数据行；`rep[i][hdr.index("FileName")]` 是
  生成源码的路径，拿它去 `open().read()` 检查 `shared_reduce<` / `_wr_mask`。
- **每次改 `para_opt_level` 都要换一个 `compile_options` 值**（例如
  `{"test_xxx": <序号>}`），否则第二次拿到的是缓存里上一个 level 编出来的 kernel，
  你会看到「改了没反应」。
- 每次都同时算一遍与 numpy 的相对误差。归约改的是求和顺序，**只看时间不看数值
  的对比毫无意义**——一个算错的 kernel 通常也更快。

## 走不走 JIT：先确认这个形状还在代码生成器里

`nn/backends/full_reduce_cuda.py` 把 `jt.Var.sum` / `jt.Var.mean` 猴补成了两级
CUB 折叠，**全量归约（不指定 dim）根本不进代码生成器**。所以

- 想量代码生成器产出的归约，用 `jt.reduce(x, "add", dims)`，不要用 `x.sum()`；
- 反过来，`x.sum()` 慢不慢与这三个 pass 无关，去看那个快路径。

同一个语义有两条实现、测试钉在其中一条上，是这个仓库反复出现的形状。量之前
先确认你打中的是哪一条。
