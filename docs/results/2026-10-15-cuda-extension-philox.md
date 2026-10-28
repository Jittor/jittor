# CUDA 扩展 Philox 与受 checkpoint 约束的 RNG 归属

- Status: 实际 CUDA 扩展重放与融合 FlashAttention 训练回归已验证；与独立二进制 Torch FlashAttention 的对比仍待完成。
- Date: 2026-10-15。
- Synchronized baseline: `f2be6354924f595296c7d320a3f3f9304756e9d6`。
- Source fix: `fa25b2d43`。
- Owner: native CUDA RNG 与 Torch 兼容维护者。
- Review when: CUDA generator 归属、checkpoint 偏移、executor 入口、扩展 generator 接口或 FlashAttention 源码变化。

## 缺陷与共享 owner

官方 FlashAttention 扩展无法加载，因为兼容层 `CUDAGeneratorImpl` 头文件引用了已被
移除的 `jittor::current_offset` 符号。一个直接的模块加载诊断暴露了该错误。高层
loader 曾把它替换成后续的可选模块缺失；`bd38eba39` 单独保留了某次加载尝试的全部
错误。

扩展现在通过既有的 native 按设备 checkpoint 游标预留 Philox 位置。预留拒绝非法增量
与溢出，把增量向上取整到四个位置，并在被外部调用时进入 native executor。物理
cuRAND generator 随其记录的游标一同推进。只更新记录的游标会使 checkpoint 之后的
native/扩展混合续训变得不正确。没有新增第二个 RNG owner 或 checkpoint 格式。
初始的 CUDA generator seed 也来自既有的 native seed owner。

一个实际编译的 CUDA 扩展抽取 128 个 Philox 值。其测试覆盖 seed 重置、状态保存/恢复、
native 正态/扩展/native 均匀的交替续训、预留取整与非法增量的原子性。两项抽取测试
使用 `forbid_backend_fallbacks()` 并断言 CUDA 驻留。

## 平台与验证

Linux x86-64、Python 3.11.16、一块 NVIDIA GeForce RTX 4090（SM89）、驱动
580.178.04、GCC 11.5.0 与真实 NVCC 12.2.140。选中的物理卡映射为进程本地 CUDA
设备 0。CPU 进程显式隐藏 CUDA 并清除 nvcc；CUDA 进程选择 CUDA 并启用 native
所需的 FlashAttention。这些检查中 MKL、NCCL、MPI、CUTT 与并行算子编译均被禁用。
CUDA kernel math 为严格模式。这些是正确性检查；并发的独立作业排除了任何速度或
内存比较。

上游源码是官方 FlashAttention 2.7.4.post1，提交
`5231d95fe13733fb534c01895f7ea88c6a6c7793`，来自
`https://github.com/Dao-AILab/flash-attention.git`，CUTLASS 提交为
`c506e16788cb08416a4a57e11a9067beeee29420`。候选编译选择 head 维度
32/64 与 fp16/bf16。编译在每个缓存内串行；CPU core、结构与 CUDA 检查使用不同缓存。

| 同步源码上的检查 | 结果 | 外部日志 |
| --- | --- | --- |
| 新编译扩展契约加 CUDA RNG/fork_rng/generator-stream 回归 | 25 passed，零跳过 | `philox-synced-cuda.log` |
| 实际 native 必需的融合 FlashAttention 前向 | 1 passed，零跳过 | `candidate-native-synced.log` |
| 实际融合反向与 dropout 重放，fp16/bf16，head 维度 32/64 | 8 passed，零跳过 | `candidate-native-training-synced.log` |
| CPU core，native 与 Torch 进程模式 | 322 passed，42 skipped，1 项既有 xfail | `philox-synced-core.log` |
| 完整 Torch 模式结构套件 | 1460 passed，六项跳过 | `philox-synced-structure.log` |

融合反向测试用独立的 NumPy SDPA 数学比较全部 Q/K/V 梯度。Dropout 测试检查按 seed
精确重放的输出/梯度、有限非零梯度，以及随 RNG 推进而变化的输出。它们不确立与
二进制 PyTorch FlashAttention 实现或其 RNG 算法相等。多设备扩展 RNG 历史与 CUDA
graph 捕获不在本验收范围内。既有 xfail 涉及 complex dtype 占位符；六项结构 skip
是两项加速器检查、两项已声明检查与两个没有已安装 Jittor 发行版的孤立解释器。

## 复现与证据

准备固定版本的上游 checkout 与真实 CUDA 工具链，然后设置
`JITTOR_TORCH_SHIM=1`、绑定源码的 `PYTHONPATH`、`JITTOR_TEST_DEVICES=cuda`、
`CUDA_VISIBLE_DEVICES=<allocated-card>` 与私有的 `JITTOR_HOME`/`TMPDIR`。
设置 `JITTOR_FLASH_ATTN_JITTOR_SRC=<upstream-checkout>`、
`JITTOR_FLASH_ATTN_HEAD_DIMS=32,64`、`JITTOR_FLASH_ATTN_DTYPES=fp16,bf16`、
`JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1`，以及
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES=0` 与 `JITTOR_FLASH_ATTN_MIN_SCORES=0` 两项。

```bash
python -m pytest -q compat/tests/torch/test_cpp_extension_philox.py \
  compat/tests/torch/test_torch_cuda_rng_state.py \
  compat/tests/torch/test_torch_fork_rng.py compat/tests/torch/test_generator_streams.py
python -m pytest -q compat/tests/torch/test_torch_compat_attention.py \
  -k 'test_sdpa_native_flash_attn_fp16_cuda or test_sdpa_native_flash_attn_backward_fp16_cuda or test_sdpa_native_flash_attn_backward_bf16_cuda or test_sdpa_native_flash_attn_backward_hdim64_fp16_cuda or test_sdpa_native_flash_attn_backward_hdim64_bf16_cuda or test_native_flash_attn_dropout_replays_seed_and_backward or test_native_flash_attn_dropout_hdim64_fp16 or test_native_flash_attn_dropout_hdim64_bf16'
```

原始日志未版本化，位于
`$JITTOR_LAB_ROOT/_state/flash-attention-audit/logs/`。最初的符号失败在
`candidate-official-diagnostic.log` 中；`philox-native-first.log` 保留第一次绑定生成
编译失败。同步前成功的融合运行是 `candidate-native-fixed.log`；它仍绑定到
`aec3b683b` 加上四个外部哈希的 RNG 文件。在所有 worktree 测试停止后，快进到上述
基线保留了这些改动，并重新运行了同步后的检查。没有把编译超时当作终态结果。
