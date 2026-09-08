# FSDP2：分组、优化器复用与真实内存边界

- 状态：架构代码与数值验证完成；按用户最新优先级，剩余性能调优后移。
  7.13 显存验收仍未达标，不关闭整项。
- 复查日期：2026-09-08。
- Owner：兼容层 / 分布式维护者。
- 复查条件：FSDP flat 存储、优化器更新、allocator 或前端 holder 生命周期变动。

## 已实现

`DeviceMesh` 的输入现在是真实 rank 数组，`init_device_mesh` 才把 shape 转为
rank 数组。每个轴按全 rank 一致的顺序创建原生 ProcessGroup，切片共享对应组；
倒序 rank、二维切片、flatten、单组 from_group、局部 rank / coordinate 均来自
实际成员表。非法/重复/越界 rank 与不在 mesh 内的调用明确拒绝。

FSDP state 持有 shard / replicate 两个 group；gather、reduce-scatter、unused-grad
标志均使用对应 communicator。二维默认第一轴 replicate、第二轴 shard，显式
DataParallelMeshDims 可选择命名轴。范数按各梯度所属 shard group 分组归约，
再合并不同组与普通参数，replica 轴不重复计数。零范数先对每个逻辑参数判断全组
是否非零，避免把跨 rank 的同一参数重复计数。

SGD / Adam 数学实现在 `jittor.optim.algorithms`；native optimizer、FSDP 和
Torch Adam 调用同一更新函数。原生 Adam 的 epsilon 位置与 Torch / AdamW 位置
用显式策略区分。基类 step adapter 给 FSDP 显式 native_kind，用户 Adam 子类
可以执行自己的 step 并调用 super；直接绕过 override 调 FSDP helper 仍拒绝。

FSDP 参数不再保存 `_local_tensor=self`，也不保存强持自身的实例 bound method。
方法弱持有可弱引用的 state / entry，调用时解析当前 shard。独立 Tensor 的
shard / unshard / refresh 保留原前端类型；native Var 路径不强制转换类型。
删除了只写不读、每次 backward 扩大的 full-param-id 字典。

初始化分片、full checkpoint 加载和手动 full.grad 切片在成为本 rank 持久状态的
边界使用真正的 `Var.copy`。仅对完整参数做 slice+sync 会保留完整 backing allocation，
不能证明已经分片；后续同一 rank 的 flat entry views 仍共享该 rank 自己的存储。

SFRL 同时修复了大 segment 小尾段的实际复用：尾段按当前尺寸进入 small / large pool，
free 合并时从相邻块的真实 pool 摘除，合并后重新分类归池。不是降低 used 计数却留下
不可复用碎片。双向跨 pool 合并、尾段复用、share 生命周期和原底层 handle 释放的
三个 C++ 合同通过。

## 数值与生命周期证据

- Host DeviceMesh / 真实 ProcessGroup 类合同：3 passed。
- CPU optimizer kind / direct override 拒绝和共享数学：11 passed。
- 旧 alias 模式的 flat / nonflat 梯度切片、可见 full.grad 与手动 full.grad：
  3 passed。该测试文件明确测试 `import jittor as torch`，须用
  `JITTOR_TORCH_INDEPENDENT=0`；独立模式不会给 native Var 安装 `.grad`。
  fixture 改用真实可弱引用 StateRecord，并接收显式 group 参数。
- CUDA 两 rank：共享 Adam 四种 epsilon / decay 策略、SGD momentum / Nesterov
  与独立 NumPy 参考，以及倒序 mesh 自定义 Adam 三步轨迹，各 rank 3 passed。
- native_kind 最终接线后，自定义 Adam 节点两 rank 各 1 passed；旧 shard instance
  dict 的唯一 marker 在更新后立即释放，不依赖 gc.collect。
- CUDA 四 rank：真实 2×2 hybrid mesh 梯度与多组范数，各 rank 2 passed，0 skipped。
- 原有 flat SGD 两 rank 数值节点通过。
- 最终 owned-shard / SFRL 代码上的四 rank 小模型验收：hybrid gradient、自定义 Adam、
  嵌套分片 checkpoint 加载和原 flat SGD，各 rank 4 passed，0 skipped。
  原生 NestedModel 测试改用 execute；independent Torch 模式不会替 native Module
  自动安装 forward，因此这项测试不能依赖旧 alias 模式的副作用。

关键 nodeid：

```text
tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_mesh_reordered_shards_custom_adam_and_lifetime
tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_two_dimensional_mesh_hybrid_gradient
tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_mesh_norm_combines_groups_without_replicating
tests/distributed/test_fsdp_optimizer_math.py
tests/compat/torch/test_torch_optimizer_kind.py
```

## 显存：不能用较小的 Var 数字声称达标

同一模型（四层 512×512 Linear、batch=2、固定种子和输入、Adam 五步），full 与
shard 分别启动全新双 rank 进程。缓存可复用，进程内不做强制循环 GC。

| 模式 | allocator used 高水位 / B | CUDA Var 快照 / B | 第 2–5 步 CUDA 保留池 / B |
| --- | ---: | ---: | ---: |
| full，两 rank 相同 | 19,977,728 | 19,965,960 | 20,971,520 |
| flat shard，rank 0 | 29,406,720 | 18,919,428 | 44,040,192 |
| flat shard，rank 1 | 29,406,720 | 18,939,908 | 37,748,736 |
| nonflat shard，两 rank 相同 | 21,020,672 | 18,917,380 | 22,020,096 |

上表是修复 owned-shard / SFRL 尾段之前的诊断基线。最终代码复验：

| 模式 | allocator used 高水位 / B | CUDA Var 快照 / B | 第 2–5 步 CUDA 保留池 / B |
| --- | ---: | ---: | ---: |
| full，两 rank 相同 | 19,977,728 | 19,965,960 | 20,971,520 |
| flat shard，rank 0 | 23,123,456 | 17,868,804 | 41,943,040 |
| flat shard，rank 1 | 23,123,456 | 17,889,284 | 32,505,856 |

flat 的真实 used 高水位由 29,406,720 降至 23,123,456 B，但仍比 full 高约 15.7%。
五步训练没有持续逐步增长，但仍未满足降低峰值的要求。不能把较小的 CUDA Var 快照
当作峰值显存。显式给定 full baseline 的峰值断言已在最终代码上真实失败；
该性能缺口按用户指示留待后续调优，不继续为它扩展本批架构改动。

`core.get_peak_allocator_used_memory()` 只读现有 memory profiler 的
max_used_memory_size：executor 检查点记录的全部 SFRL used 高水位，含 CPU 与加速器，
不包含空闲缓存块；它不是驱动级或逐次分配 hook 的峰值。`get_mem_info().total_cuda_used`
实际包含 used+unused，故这里准确称为 CUDA 保留池。两项指标都保留，不能互相替代。

复验环境使用安装了当前 core / compat editable 的 Python，独立 JITTOR_HOME，
显式 `JITTOR_TORCH_INDEPENDENT=1`、`JITTOR_TORCH_SHIM=1`、`use_cuda=1`、
`use_mpi=1`、`use_nccl=1`、`use_mkl=0`。先单进程完成共享 cache 编译，再执行：

```bash
JITTOR_FSDP2_MEMORY_MODE=full mpirun -np 2 python -m pytest -q -s tests/distributed/test_fsdp_memory.py
JITTOR_FSDP2_MEMORY_MODE=shard JITTOR_FSDP2_REFERENCE_PEAK_BYTES=19977728 mpirun -np 2 python -m pytest -q -s tests/distributed/test_fsdp_memory.py
```

第二条应当在当前实现上失败；基线数值须先按同一机器第一条实测更新。未指定 reference
时测试只证明五步 trajectory 稳定，不声称峰值达标。原始日志未版本化，位于
`$JITTOR_LAB_ROOT/_state/fsdp-fast-7-13/` 的 `used-peak-{full,shard,nonflat}.log`、
`final-{full,shard,correctness}.log`、`allocator-tail-final.log`、`independent.log`、
`four-rank.log`、`cpu-optimizer.log`、`custom-adapter.log`、`final-cpu-fsdp-legacy.log`。

后续性能分区需继续定位 flat 更新重组与临时 allocation 的同时存活；不能仅改默认阈值
把 flat 藏起来。NPU / HCCL 和多机没有硬件验收证据，仍需上机执行
相同的 subgroup、数值和内存检查。
