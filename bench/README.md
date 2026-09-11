# bench — Ascend NPU 的对照测量

这里放的是**手工运行**的测量代码：把 Jittor 的 ACL 后端和同机的
`torch + torch_npu` 放在同一组网络上比速度、比数值。它需要真实的昇腾设备和 CANN，
不会被 pytest 收集，也不进 CI。

按提交索引的留存性能结果属于 [`benchmarks/`](../benchmarks/)（ASV 套件，见
[`docs/performance/benchmarking.md`](../docs/performance/benchmarking.md)）。两者不要
共用一个 Jittor 编译缓存。

## 网络

两个框架的模型定义必须逐层对应，规格写在 [`models.md`](models.md) 里。改其中一个就要
同步改另一个，否则这里的所有数字都失去意义。

## 计时

```bash
python bench/bench_jittor.py --hf32
python bench/bench_torch.py  --hf32
```

各自打印每个 `模型/模式` 的 ms/step，最后一行是一整行 `JSON`，方便脚本采集。
`--hf32` 打开 cube 的 HF32 降精度路径，两边都要给，否则比的不是同一件事。

测量这台机器上容易踩的坑：

- **每次迭代都 `sync()` 会把发射和执行串起来**，把主机开销算进设备时间。要么像这两个
  脚本一样只在计时区间两端同步，要么构造 N 个操作再同步一次。
- **A/B 之间重新编译会带来很强的位置偏差**——先跑的那个总是赢。切运行时开关、只编译
  一次、交替跑。
- `jt.profiler` 的 `TotalTime` 对小算子**不是可相加的墙钟时间**。

## 数值交叉验证

```bash
python bench/xval/dump_jt.py    /tmp/xval --steps 20        # jittor 侧录制
python bench/xval/compare_th.py /tmp/xval --steps 20        # torch 侧回放并比对
python bench/xval/control_th.py /tmp/xval --steps 100       # 对照实验
```

`dump_jt.py` 把初始权重、输入、第 0 步的 loss 与梯度、训练 loss 轨迹和最终参数存成
`<模型>.npz`；`compare_th.py` 用**同一批权重和输入**在 torch 里重跑。两边的
submodule 命名不同，映射由 [`xval/_canon.py`](xval/_canon.py) 给出，并且会断言没有
任何可训练参数漏在映射之外。

怎么读这些数字：

- **第 0 步的 loss 和梯度是紧的那一项**——一次前向、一次反向，还没有优化器状态把差异
  带到下一步。fp32 下 loss 相对差 0…2e-07、梯度 6e-07…2e-06，就是 float32 舍入量级。
- **`--hf32` 下只看 loss**。HF32 的尾数只有约 10 位，K=4096 的矩阵乘上累积下来，梯度
  张量的最大元素相对差可以到 1e-2 量级；这是 HF32 本身，不是实现分歧。同样配置下
  loss 仍然对到 1e-06 以内。
- **长轨迹不是紧的那一项**。跑到 100 步，mlp 和 cnn 会明显分开而 transformer 不会。
  先跑 `control_th.py`：它让 torch 和 torch 自己比，唯一的差别是把初始权重扰动
  1e-7（一个 float32 舍入步）。如果两者量级相同，那个偏离就是网络的性质而不是框架的
  ——mlp 和 cnn 是纯 ReLU 网络，舍入级的差别迟早翻转某个 ReLU 的符号，那是离散跳变；
  transformer 用 GELU + LayerNorm，没有这种分支，100 步后仍停在 1e-07。

## CANN 探针

`aclprobe/` 下是绕开 Jittor、直接调 aclnn 的小程序，用来把「是 CANN 的下限还是我们的
开销」这类问题钉死。用 `ccec` 编译：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash bench/aclprobe/ascbuild.sh bench/aclprobe/binu_probe.cc /tmp/binu_probe
/tmp/binu_probe
```

| 探针 | 回答的问题 |
|---|---|
| `binu_probe.cc` | 同尺寸下一元与二元的差别。CANN 的二元并不比一元慢——这条证据指出慢的是我们自己的发射路径 |
| `muls_probe.cc` | 标量操作数：stride-0 视图、shape-[1] 张量、`aclScalar` 三种写法的代价 |
| `perm_probe.cc` | `aclnnPermute` 与「strided view 拷到 contiguous」在 11 种形状上的对比 |
| `bmm_strided.cc` | `aclnnBatchMatMul` 能不能直接吃排列过的 stride 视图（批维度的排列：不能） |
| `asc_numerics.cc` | 手写 AscendC elementwise kernel 与 aclnn 的逐位一致性 |
| `asc_fuse_bench.cc` | 融合成一个 AscendC kernel 相对 N 次 aclnn 发射的收益，以及 core 数的取舍 |

探针有意写得直白、自成一体：每个都是一个 `main`，自己分配缓冲、自己计时，好让人一眼
看出测的到底是什么。
