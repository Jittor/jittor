# 原生 TP AdamW 与 BF16 续训

状态：已验证本报告限定范围内的 CPU 语义，以及单机双 rank CUDA BF16
训练，源码 `479a96093fad9a8c05adf7252984503c37403022`。
维护者：原生 distributed owner。functional linear、CUDA
GEMM、AMP 梯度发布、collective adjoint、原生 optimizer 或
checkpoint ownership 变化后需复查。本报告延续
[FP32 momentum-SGD 结果](2026-10-15-native-tensor-parallel.md)。

工作树同步到 `d62e0deba6b6dc74e4e4d9aaa12bb6baefec2d95`，
随后把既有五个 TP 提交保留为 `642c60b1b`。验证冻结
该基线与本次改动；设备运行期间没有合并任何源码。
基线包含较新的 detach 与 CUDA RNG owner。本补丁未改动 functional
linear、其 AMP helper 与 CUDA Lt 实现。后续共享 BF16 epilogue 改动需要
用同样的固定输入测试与精度标准重跑；本结果不为其后的代码背书。

## 已确认缺陷与限定修复

`RowParallelLinear` 把 FP32 master bias 直接加到 BF16 reduce
结果上。在 CPU 上，权重 `[[1, 0], [1, 0]]`、输入
`[[0.10009765625, 0]]`、FP32 bias `[-0.0373, -0.0369]` 时，得到
`[0.06298828125, 0.06298828125]`。独立的 Torch autocast linear 产生
`[0.0625, 0.0634765625]`：它先把 bias 输入舍入到 BF16 再相加。
修复前 CUDA 已经给出与 Torch 相同的结果。这是 CPU
反例，不是复现到的 CUDA 数值缺陷。

TP 现在在 row reduction 之后调用既有的 `bias_for_compute_dtype` helper。
helper 与普通 linear owner 未改动。精确 CPU
与双 rank CUDA 回归验证输出、输入/权重/bias 梯度、
FP32 master bias 与零后端 fallback。未引入优化器数学，
也未引入第二套混合精度策略。

## 真实运行时协议

两个运行时执行同一个固定的 Linear(3, 6)、tanh、Linear(6, 4) 模型、
固定权重与五组输入/目标。原生端两种 gather/scatter
选项都运行。AdamW 使用学习率 `0.003`、betas `(0.8, 0.95)`、epsilon
`1e-6`、weight decay `0.02`，并禁用 fused/foreach 参考路径。

- 真实参考：独立 Python、Torch `2.11.0+cu128`、binary `torch._C`，
  不 import Jittor，也不带仓库 Python 路径。全模型 oracle 使用
  普通未分片的 Torch linear 层、autograd 与 AdamW。
- 原生端：Python 3.11.16、NumPy 1.26.4、CUDA 12.2.140、原生 NCCL 2.18.3、
  严格 CUDA 数学、两张真实 sm_89 GPU。测试与缓存和
  benchmark 缓存分开；每个 rank 在各自缓存中串行预热。
- 参考端显式禁用 TF32 与降精度 BF16 reduction。
  Dispatch 观测记录实际 linear GEMM 的输入、权重与 bias
  的 dtype/设备，而不只观测模型的最终输出。
- 原生外层 `auto_mixed_precision_level=3` 覆盖前向、反向与
  优化器更新；只有前向设置 `amp_reg=prefer_bfloat16`。实际 CUDA
  GEMM 的输入与权重操作数为 BF16，而 master、每份输入/参数
  梯度、一阶矩与二阶矩为 FP32。每个操作数在
  NumPy 拷贝前检查 CUDA dispatch、设备存储与 driver 上报的设备
  ordinal。在保留的初始探针中，仅前向的 BF16 标记并不能
  保证 FP32 梯度。
- 通信观测确认：五个原始步与
  两个恢复步中共有七次 BF16 row 输出求和 reduction，
  以及七次 FP32 column 输入梯度求和 reduction。legacy optimizer 的 reduction 与 broadcast
  在实际 owner 处计数，尽管每一步本应发生，计数仍为零。

每个 rank 拥有 28 个参数元素，以及每个 AdamW moment 中的 28 个元素，
而全模型有 52 个参数、每个 moment 52 个元素。moment
形状等于本地 master 形状；不会先分配完整 moment 张量
再切片。复制保存的 row bias 解释了与精确一半之间的差异。
激活与临时计算 cast 不属于优化器状态；
这些计数不能推出整进程峰值内存或加速结论。

每一步记录输出、loss、输入梯度、全部参数梯度、
更新后的参数与两个 moment：五个步累计每个 runtime/rank 95 个数组。
第三步后保存的 checkpoint 恢复到全新的原生 model
与 optimizer 对象。恢复后的两步在每一个字段上都与未中断的
原生轨迹精确相等，并且仍在独立全 Torch oracle 适用的
精度标准之内。

## 独立 BF16 标准

原 FP32 标准保持 `atol=2e-6, rtol=2e-5`；其 BF16
失败保留为诊断信息。BF16 标准在原生
修复后验收之前就已冻结。它使用独立的真实 Torch 对比：
全模型与同一 TP 分解比对，包含真实双 rank
Torch NCCL 2.28.9、两种 gather 选项、BF16 row 部分输出
reduction、FP32 column 输入梯度 reduction，以及 CPU singleton 校准。
全模型 oracle 仍是验收参考；TP 校准
不会用 Jittor 自比对取代它。

对每个数组，将最大绝对误差与
`8 * float32_epsilon + budget[field] * max(abs(full_reference_array))` 比较。
BF16 就近舍入的单位舍入误差为 `u = 1/256` 时，每个预算为
`(ceil(independent_Torch_envelope / u) + 2) * u`。两个单位的余量用于容纳
实现的舍入结合顺序。这是针对
该维护 fixture 的经验允许值，不是适用于任意模型的数值定理。
原生误差没有参与预算计算。

| 字段 | Torch 全模型 vs TP 最大相对 Linf | 冻结预算 | 原生 CUDA vs 全 Torch 最大相对 Linf |
| --- | ---: | ---: | ---: |
| 输出 | 0.005025 | 0.015625 | 0.005025 |
| 损失 | 0.002498 | 0.011719 | 0.000873 |
| 输入梯度 | 0.007692 | 0.015625 | 0.010956 |
| 参数梯度 | 0.010695 | 0.019531 | 0.010152 |
| 更新后参数 | 0.001858 | 0.011719 | 0.001998 |
| 一阶矩 | 0.006503 | 0.015625 | 0.006410 |
| 二阶矩 | 0.012947 | 0.023438 | 0.011875 |

数值汇总了两个 rank、两种 gather 选项与全部五个步；尺度按
每个独立的全参考切片计算。可复用的 Torch TP
校准脚本逐数组精确复现了独立冻结的校准结果。Torch TP FP32 与其全模型的最大
相对 Linf 差异为 `3.05e-7`；原生 CUDA FP32 与全 Torch oracle 的最大差异为
`3.11e-7`，全程通过原逐元素标准。

CPU BF16 测试覆盖原生 generic-contraction 输出与 FP32
状态/梯度语义，以及精确续训。CPU 路径不
承诺真实的 BF16 GEMM 输入 cast：其 generic contraction 可能保留
更宽的操作数。上文关于真实 BF16 计算的结论仅限 CUDA。
CPU 记录的最大字段误差为输入梯度相对 Linf `0.015385`，
在独立冻结的 `0.015625` fixture 预算之内。

## 命令与证据

按 worktree 与
distributed verification skill 的要求配置私有环境与缓存。`<native-python>` 必须能 import 本 checkout；
`REAL_TORCH_PYTHON` 必须是单独安装的真实 Torch 解释器。
不要把原生 Cu122/NCCL 的 loader 路径暴露给 Torch Cu128 参考。
helper 会在子进程中移除该路径与仓库 Python 路径。

```bash
REAL_TORCH_PYTHON=<real-torch-python> JITTOR_REQUIRE_REAL_TORCH=1 \
  JITTOR_TORCH_SHIM=0 JT_BACKEND=cpu JT_USE_CUDA=0 \
  PYTHONPATH=python <native-python> -m pytest -q \
  tests/nn/test_native_tensor_parallel.py

# The wrapper sets isolated JITTOR_HOME/TMPDIR from RANK before importing
# pytest/Jittor, then forwards its arguments to pytest. Warm each rank first.
# JITTOR_TP_TEST_DIR is shared; rank-local artifact paths are separate.
REAL_TORCH_PYTHON=<real-torch-python> JITTOR_REQUIRE_REAL_TORCH=1 \
  <native-python> python/jittor/distributed/launch.py --backend nccl -n2 \
  --device-ids=<gpu-a>,<gpu-b> --log-dir <state>/logs --timeout 300 -- \
  <native-python> <rank-cache-wrapper> \
  tests/backends/comm/nccl/test_native_tensor_parallel_nccl.py

# Independent TP calibration: run using real Torch's two-rank launcher,
# without native PYTHONPATH/LD_LIBRARY_PATH; repeat with --gather.
<real-torch-python> -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/_helpers/native_tp_oracle.py --mode tp --device cuda \
  --precision bf16 --output <state>/torch-tp.npz
```

CPU TP：**14 passed，0 skipped**（11.89 s）。CUDA TP：**每 rank 13 passed，
0 skipped**（18.60 s），launcher exit 0。包含此前全部八个
FP32 CUDA 用例、精确 row-bias 回归以及每 rank 四行 AdamW。
每个原生训练行的后端 fallback 计数均为零。
core runner：**322 passed，42 skipped，1 个既有 xfail**，exit 0
（native 98/41/1，Torch 模式 224/1）。仓库布局检查通过。
完整 structure：**1,459 passed，6 skipped，2 failed**（166.42 s）。
两个失败是 TP note 与两个结果页的中央 toctree
登记，以及生成的打包 manifest。这些中央文件
留给整合方处理；本次未改动。Import 分层、
循环检测与子进程契约通过。在中央登记步骤完成前，
本分支不声称结构门禁全绿。

原始日志、checkpoint、数组 trace、metadata 与冻结哈希
**未版本化**，位于 `$JITTOR_LAB_ROOT/_state/native-tensor-parallel-bf16/`。
`frozen-bf16-budget.json` 记录独立校准哈希与
标准公式；`mixed-shared-root-source.json` 记录受测的 1,115 个
原生 runtime 文件与六个归档的源码/测试哈希。CPU 日志为
`logs/cpu-tp-adamw-shared-root.log`；CUDA 日志为
`logs/formal-adamw-shared-root-two-rank/rank{0,1}.log`。正式 CUDA checkpoint
产物与 95 数组 trace 位于 `formal-shared-root-checkpoints/`。

最初的新 harness 错误地把 checkpoint 根目录放在
rank 本地产物目录下。由于各 rank 使用了不同的
checkpoint 身份，其 Store 等待失败；该次运行被停止并保留，
不给通过结论。修正后的 harness 使用共享 checkpoint 根目录与各自独立的
rank 本地观测。这不需要改动生产 checkpoint。
初始探针与一次 storage-observer 时序失败单独保留；
存储检查发生在 GEMM 消费其操作数之后、host 读回之前。

本报告不声称任意 TP 算子、attention/sequence parallelism、
非均匀切分、DP/ZeRO/FSDP 混合 mesh、更大 WORLD、多主机、非 CUDA 加速器、
其他 AMP 策略、进程峰值内存或吞吐验收通过。

## 共享 Linear 累加器复验

状态：真实 CPU 与双 rank CUDA 复验通过。同步的远端
基线为 `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e`。此前的 TP 改动
`479a96093fad9a8c05adf7252984503c37403022` 与
`b9c804386aaf4c56a186c823732d07b50bea1d8d` 被保留，并与
完整的共享 Linear 系列 `e64d4a27f`、`ead8bf803`、`e07a2e046`
和 `0c6837b90` 一起重放。得到的冻结源码父提交为
`6c194f0485943e5804e28b70bd237d2ac7c38f98`；此后唯一的测试改动
是把 CUDA 操作数 observer 移到实际 cuBLAS 调用处。精度预算、
输入、优化器与 checkpoint 数学仍是上文已验收的那一套。

带 bias 的 BF16 Linear 现在把 bias 加到 FP32 GEMM
结果后返回 BF16 cast。再通过最终结果的图去索引
已经无法定位该 GEMM。observer 包裹既有后端的 `cublas_matmul`，记录其实际输入
与输出，同步整个 Linear，并在任何 NumPy 转换前检查 dtype、CUDA dispatch、
常驻存储与 CUDA 指针 ordinal。它还
检查原生调用参数：带 bias 的 BF16 使用 `(False, True, True)` 与
FP32 结果；无 bias 的 BF16 保持 `(False, True)` 与 BF16 结果。FP32 调用
保持 FP32 操作数与结果。observer 释放其临时引用，
而不是保留此前的训练图。

各 rank 缓存串行预热后，在真实 CUDA 设备上运行
同一个双 rank 测试文件。CPU：**14 passed**，15.29 s。CUDA：**每 rank 13 passed**，
31.52 s，launcher exit 0。包含既有的八个 FP32 测试、
精确 Row bias 测试，以及 FP32/BF16 与 gather 的四种 AdamW 组合。
完整参数/输入梯度、FP32 master 与 moment、实际 BF16 GEMM
操作数、owned shard 大小、零 legacy optimizer collective、零 fallback，
以及三步后保存再以全新对象精确恢复两步，
仍全部受检。core 套件同样通过：**322 passed，42 skipped，1 个
既有 xfail**。

复用独立全模型 Torch oracle 与先前冻结的 BF16 预算，
未重新校准。跨两个 rank 与各 gather 变体，最大缩放误差为：

| 字段 | 共享 Linear 改动后 | 冻结允许值 |
| --- | ---: | ---: |
| 输出 | 0.00502513 | 0.015625 |
| 损失 | 0.00249812 | 0.01171875 |
| 输入梯度 | 0.01095618 | 0.015625 |
| 参数梯度 | 0.01015229 | 0.01953125 |
| 更新后参数 | 0.00239895 | 0.01171875 |
| 一阶矩 | 0.00650291 | 0.015625 |
| 二阶矩 | 0.01294701 | 0.0234375 |

这些是针对该 fixture 的正确性结果。fused column bias
舍入的改动会改变后续 TP 舍入，并不代表每个字段的
跨框架误差都会下降。严格 FP32 容差对 BF16 仍然失败，
仍作为诊断信息；不作性能或普适 BF16 误差界结论。

未版本化证据位于
`$JITTOR_LAB_ROOT/_state/native-tp-linear-revalidation/`：`frozen-source.json`、
`logs/cpu-14.log`、`logs/warm-rank{0,1}.log`、`logs/formal-launch.log`、
`logs/formal-two-rank/rank{0,1}.log`、`logs/core.log` 以及
`formal-checkpoints/adamw-*/rank*/{result.json,native-trace.npz}`。先前的
`native-tensor-parallel-bf16` 证据与已交付分支保持不变。
启动命令是前述命令配上这个新产物目录；
串行预热后复用了同样的每 rank 独立编译缓存。

完整 structure 运行结束于 **1459 passed，6 skipped，2
failed**（180.34 s）。两个失败属于中央整合工作：新的
Linear 与 TP BF16 结果页未出现在 results toctree 中，
以及根目录生成的 `MANIFEST.in` 需要重新生成。Import 分层/循环与
子进程契约通过。仓库布局通过。两个中央文件由父任务整合方
负责；本报告不把该次 structure 运行描述为全绿。

## 父任务整合

父任务在
`71c5be74ef6991b75fa79d73fe52eb04c0319d59` 之上，把已交付的 TP 改动与全部四个共享 Linear
提交同原生 FSDP/checkpoint ownership 合并。36 个交付文件与
其源码引用精确一致；433 个原始 TP 与 141 个共享 Linear
复验产物重新哈希成功。这些检查保持了
测量源码身份，而不是重标先前的双 rank 执行。

冻结的合并源码通过独立 oracle 的 singleton CPU TP
用例，计入 **56 项原生 owner 通过**；CPU core **322 passed**；真实 CUDA
Linear/cuBLAS **20 passed**；CUDA 兼容性 **24 passed**。中央
toctree 与 manifest 登记现已就位；完整 structure 通过
**1470**，含 6 个 skip。完整命令输出、skip 原因与源码哈希
未版本化，位于
`$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`。针对该 fixture 的
BF16 允许值与多节点限制保持不变。
