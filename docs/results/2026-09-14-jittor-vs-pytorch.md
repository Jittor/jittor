# Jittor vs 真 PyTorch 2.9.1：现在差在哪

- 状态：2026-09-14 的快照，之后未在本页更新。原生 Jittor 对 eager PyTorch：赢 6、
  平 5、输 1（本轮起点赢 2、平 1、输 9）；唯一输的是 prefill 训练（交替配对中位比值
  1.021），成因是流水没做满，未查清。对最好的 `torch.compile` 模式，更正后是赢 2、
  平 2、输 5（12 条里 9 条两臂状态一致可判）。整步训练的 CUDA Graph 回放在本轮只做了
  手工验证
- 日期：2026-09-14；2026-10 压缩归档
- 基线提交：分支 `perf/metaop-broadcast`（基于 `d40a2e97`）；最后一次更新的结论为
  `743fbe09`
- 验证范围：单张 CUDA 卡，原文未记卡型；同一 `metaop` lab、同一基线的
  [元算子报告](2026-09-12-cuda-metaop-launch-index-scalar.md)记为 H20（sm_90）。对照为
  PyTorch 2.9.1 + CUDA 12.9，两侧都不用 TF32（`allow_tf32=False`）。transformer / MLP /
  CNN 六种模型与形状 × 推理/训练；同机其他租户占满全部 GPU，只引用紧挨着量的比值。
  未覆盖：CPU、ROCm、NPU、多卡
- 维护者：Jittor 核心维护者
- 复查条件：图构建改成可重放、`auto_flush_bytes` 的判据、执行器发射路径或 CUDA 流
  语义变化

本页由约 900 行的逐日记录压缩而来：中间量法、被收回的结论与原始读数都在 Git 历史
（`git log -p -- docs/results/2026-09-14-jittor-vs-pytorch.md`）。lab 脚本
（`alt_pairs.sh`、`run_gate2.sh`、`vs_torch_pt.py`）在 `$JITTOR_LAB_ROOT/metaop-perf/`，
不进主仓库。

## 怎么测的

- **每个 case 独立进程、两边交替跑同一张卡、取多轮最小值。** 同一进程跑全部 case 时，
  前面 case 留下的分配器状态会让同一个 b8s256 训练量出 44.6 与 20.8 ms。
- **按整段预热，且要热够。** b8s256 训练要约 7 段（35 步）才进入稳态，只热 4 段时
  会停在 42 ms 的平台上；两边都改成先热 10 段。torch 几乎一开始就稳，这本身是用户
  可感知的差别（成因未查）。
- **两边同步纪律一致**：推理都在 `no_grad` 下，都只在一段 `steps` 结束时同步一次。
- **标签用整词匹配**：`"b1s1"` 是 `"b1s128"` 的子串，曾让 prefill 用了 decode 的步数。
- **边界上的 case 用五轮交替配对取中位比值**：这台机器的绝对值在两次运行间能跳 15%。

## 结果

### 对 eager PyTorch（ms，比值 = jittor / torch，小于 1 是 jittor 快）

| case | jittor | pt eager | pt compile | vs eager |
| --- | --- | --- | --- | --- |
| tf b8s256 训练 | 19.57 | 40.14 | 39.26 | **0.49** |
| cnn-6conv 推理 | 8.93 | 14.12 | 13.92 | **0.63** |
| tf d1024 decode 推理 | 0.44 | 0.68 | 0.58 | **0.65** |
| tf decode b1s1 推理 | 1.19 | 1.43 | 0.83 | **0.83** |
| cnn-6conv 训练 | 23.99 | 28.23 | 27.74 | **0.85** |
| mlp-1024x4 推理 | 0.37 | 0.40 | 0.38 | **0.93** |
| mlp-1024x4 训练 | 1.17 | 1.20 | 1.23 | 0.97 平 |
| tf b8s256 推理 | 13.17 | 13.41 | 13.00 | 0.98 平 |
| tf decode b1s1 训练 | 6.64 | 6.59 | 4.65 | 1.01 平 |
| tf d1024 decode 训练 | 3.53 | 3.49 | 3.55 | 1.01 平 |
| tf prefill b1s128 推理 | 2.86 | 2.82 | 1.78 | 1.02 平 |
| tf prefill b1s128 训练 | 8.02 | 7.58 | 7.57 | 1.06 |

b8s256 训练在 19.6 / 42.9 两个状态之间跳（预热不足，见上），这一轮落在快的一半。
prefill 训练 torch 侧在 7.26–8.64 之间摆（19%），单次表给不出判定，以下面的配对为准。

边界上的六条，五轮 torch/jittor 交替配对（`alt_pairs.sh`）的中位比值：

| case | 中位比值 | 区间 | 判定 |
| --- | --- | --- | --- |
| tf-d1024-L4 decode 推理 | 0.766 | 0.756–0.768 | 赢 |
| tf-d1024-L4 decode 训练 | 1.012 | 0.892–1.180 | 平 |
| tf-d512-L8 decode 推理 | 0.393 | 0.389–0.403 | 赢 |
| tf-d512-L8 decode 训练 | 0.997 | 0.991–1.134 | 平 |
| tf-d512-L8 prefill 推理 | 0.975 | 0.968–0.986 | 赢 |
| tf-d512-L8 prefill 训练 | 1.021 | 1.016–1.106 | **输** |

两条 decode 训练此前是 1.17 / 1.15，per-thread 流那笔通用收益（每算子地板
7.89 → 4.31 us）把它们拉平；推理侧的优势来自 CUDA Graph 回放。

### 对 `torch.compile`（更正后）

先前那张 compile 对比有两个 harness 错误，方向都对 jittor 有利：compile 侧写成了
`zero_grad(set_to_none=False)`（这也是「`reduce-overhead` 在全部训练用例上失败」的
真正成因，那条「jittor 回放没有这个失败模式」的说法已收回），且 compile 的数字是
单独一趟量的、跨了机器的两个性能状态。四条臂同 case 内交替重量后：

| case | vs 最好的 compile |
| --- | --- |
| cnn-6conv 推理 / 训练 | **0.67 / 0.92 赢** |
| mlp-1024x4 推理 / 训练 | 0.97 / 0.99 平 |
| tf decode b1s1 推理 | 1.42 输 |
| tf prefill b1s128 训练 | 1.42 输 |
| tf d1024 decode 训练 | 1.58 输 |
| tf prefill b1s128 推理 | 1.61 输 |
| tf decode b1s1 训练 | 3.01 输（6.62 对 2.20） |

compile 强在主机受限的 transformer 小步，jittor 强在设备受限、compile 帮不上 torch
的地方。

## 本轮落地的改动（都是通用改动，用户不用改代码）

| 改动 | 实测 |
| --- | --- |
| `zero_grad` 复用每个梯度的零，不再每步重建 | 1.031 → 0.037 ms |
| CUDA fused SGD（一个核走完一批参数，`momentum == 0` 也走） | decode b1s1 训练步 8.49 → 6.51 ms（torch 6.61） |
| 自动 flush 加尺寸判据 `auto_flush_bytes`（默认 8 MB） | 主机受限 1.09–1.16x；Resnet50 训练流水不变（146.1 对 145.9 ms） |
| 秩 ≥5 的 broadcast 形态 permute 把整个循环嵌套交给线程 | qkv permute 165.9 → 62.3 us；b8s256 一步设备时间 8.161 → 7.706 ms |
| 去掉 `+ - * /` 上只为复数标量存在的 Python 包装，转换搬进 `py_converter.h` | decode 一步省 0.160 ms（5.9%） |
| 要 `ordered` 的小块 H2D 改为流序拷贝（驱动对 ≤64 KB 阻塞拷贝要 2.3 ms，torch 同样付） | 2344 us → 1.5 us 发出 |
| `jt.graph_replay`：录一次推理调用、之后换输入重放 | d512-L8 decode 0.87x、prefill 0.79x（对 torch）；d1024 decode 1.10x |
| layer_norm 的核把一行读进寄存器复用 | 44.1 → 27.4 us（torch 29.7） |
| 全部 CUDA 工作挪到 `cudaStreamPerThread`（nvcc `--default-stream per-thread`、库句柄绑流）并加入捕获原语 | 每算子地板 7.89 → 4.25 us；回放第三次起录成设备图 |

最后一条（`28e8e1a7`）同时是 CUDA Graph 的前提：遗留默认流不可捕获。推理回放接进
自动路径后，tf-d1024-L4 decode 推理 0.852 → 0.538 ms，tf-d512-L8 decode 推理 0.575 ms。
整步训练（前向+反向+更新）手工捕获后对着 eager 30 步逐位相同，2.186 ms/步对 torch
eager 的 3.01–3.05（1.39x），主机成本从 3.5 ms 降到一次 2.24 us 的 launch。

## 量过、记下的诊断

- **decode 慢在主机建图。** 一步 3.08 ms 里 Python 建图 2.382 ms（70%），发射
  0.536 ms，规划只有 0.117 ms；一步 6426 次 Python/C 调用建约 100 个算子。把整个
  前端拿掉建图只省 38%（251.5 → 156.3 us/block），
  所以要赢必须不再每步重建图——这是图回放的依据。
- **每算子地板。** 回放一条 200 个算子的链，CUDA 每算子 7.89 us（CPU 后端 1.05，裸
  `cudaLaunchKernel` 2.0–2.6），与图大小、算子种类无关；大头是生成代码 `entry(this)`
  的 5.38 us。dlopen、遗留流隐式同步、队列打满、标量常量、profiler 钩子都量过不是。
- **b8s256 那 1.12x（一步 15.06 对 13.40 ms）** 一半是核函数（layernorm 1.49x、softmax 1.65x；gemm、
  transpose、elementwise 持平；先前「逐算子持平」的结论是用单次墙钟量的，已收回），
  一半是发射空隙（9% 对 torch 的 4%）。
- **d1024 decode 训练的 0.50 ms**：训练版 layer_norm 与 softmax 的主机成本是 torch 的
  4.2–5.6 倍（`jt.Function` 机制约占 61%），但即使追平也只到 torch 的 1.04–1.06；
  autodiff 记账本身不花钱。用 `jt.Function` 给 cuBLASLt 补反向是负收益，已撤回。
- **整步训练回放（不带 CUDA Graph）只值 1.08x**：保留图的 `run_exec_plan` 是 eager 的
  2.6 倍。保留的训练图会改写自己的叶子，不是幂等的——任何无关的 weak sync 扫到它就是
  悄悄多走一个优化器步。
- **prefill 训练那 2%**：主机 7.016 ms/步、设备约 5.35 ms，流水化实测 8.145 ms，约
  1.1 ms 没有重叠；`auto_flush` 全扫与步内同步拷贝都排除了。

## 门禁口径的两次更正

- **「1615 failed」是 harness 造的。** 把 `tests/backends/parity/test_device_parity.py`
  与 `tests/ops/test_ops.py` 写进同一次 pytest 调用时，前者留下的进程全局 Torch 模式
  状态让后者整片变红（parity 在前 99% 失败，ops 在前 20%）。一个文件一个进程跑。
  据此量出的「98 条 device-parity 转绿」作废。
- **`test_ops.py` 在 73% 处确定性崩溃**：`bilinear + align_corners=False` 上采样、
  前向已执行、第二次 `jt.grad(..., retain_graph=True)` 段错误，之后约 400 条从未运行。
  已入账并修复：KI-EXEC-006。绕开它之后，`test_ops.py` 从 286 降到 56 failed（其中
  28 条是环境缺 cupy）：参考里过时的 `np.atleast_1d`（189 条）、六个 reduce 包装对
  `keepdim`/`keepdims` 口径不一（21 条）、harness 只认 namedtuple 的多输出解包（20 条）。

## 复现

```bash
# 一个 case 一个进程、两边交替；lab 脚本不进主仓库
bash $JITTOR_LAB_ROOT/metaop-perf/alt_pairs.sh
# 门禁：一个文件一个进程
bash $JITTOR_LAB_ROOT/metaop-perf/run_gate2.sh
# 仓库内的回放与捕获回归
python -m pytest -q tests/core/test_graph_replay.py tests/core/test_graph_capture.py
```

## 未结事项（截至本快照）

- prefill 训练的 1.1 ms 未重叠，原因未找到。
- 训练步捕获在本轮未自动化；后来由 `StepCapture` 实现（`2f794e28`，见
  [真实模型对位报告](2026-09-24-torch-compat-real-models.md)）。
- 训练版 layer_norm/softmax 的 `jt.Function` 改成 `cuda_grad_src` 可省约 72–128 us/步，
  未做；`no_grad` 的 layer_norm 主机成本也是 torch 的 2.5 倍。
- `test_ops.py` 剩下 28 条：`reinterpret_view_op.cc:58` 字节数不符 16、
  `'float' object has no attribute 'astype'` 20（测试侧）、`cross_entropy_loss` 不收
  `label_smoothing` 10、`norm_p2` 默认维 8、`median` 参考 8、`rms_norm` 二阶梯度 4、
  融合算子执行失败 2（原文的分类计数，合计与 28 条不符，未复核）。
- jittor 训练要约 35 步才进入稳态，torch 几乎立刻稳定，成因未查。
