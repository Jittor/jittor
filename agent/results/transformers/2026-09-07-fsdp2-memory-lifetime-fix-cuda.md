# FSDP2 Var 元数据显存生命周期修复

- 状态：PASS（源码回归与真实 CUDA 生命周期验证通过；70B 完整长训练复验尚未执行）
- 验收日期：2026-09-07（Asia/Shanghai）
- 分支与提交：`feature/cgq_transformers@1a719099fe527c7e6e954d1a602535d6c19a4431`
- 共同基线：`508f816b81bdacba5ebdb72ce82b9bab3181b2ec`
- 环境：Python 3.11、Jittor `1.3.11.0`、CUDA toolkit 12.2/cuDNN 8、NVIDIA A800 80GB

## 结论

本轮修复了 Jittor Torch adapter 在真实 FSDP2 训练中因 Var 元数据强引用导致的显存生命周期
问题。修复后，FSDP full/shard/gradient Var 的替换不再通过 `_local_tensor` 建立 Var 之间的
强引用环；`to_local()`、`full_tensor()` 和 `redistribute()` 按 FSDP metadata 动态解析当前
值。公共 PyTorch/Transformers 代码不需要增加清理调用或修改训练循环。

CPU 定向 FSDP2 回归为 `32 tests, OK`。真实 CUDA 单进程 20 次连续 full/shard/gradient
metadata 替换中，`lived_vars` 全程为 2，分配器活跃增量保持 `4 MiB`，没有随 cycle 增长；
设备身份为 `cuda:0`（进程通过 `CUDA_VISIBLE_DEVICES=6` 绑定物理 GPU6），Jittor CUDA 已启用。
这证明了导致旧高水位增长的 metadata 生命周期在最小真实设备复现中已收敛。

## 根因与修复

旧实现为每个 FSDP Var 写入 `_local_tensor`。对于 `grad_shard` 和 `flat_shard`，这可能形成
`Var -> _local_tensor -> Var` 自引用；对于 all-gather 得到的 full parameter，则会继续保留
旧 shard 或旧计算图。Jittor extension Var 不会像普通 Python 对象那样可靠地通过循环 GC 回收，
所以每轮 unshard/reshard 都可能留下 CUDA allocation 和图节点。这是 Jittor adapter 为模拟
PyTorch DTensor/FSDP2 语义时引入的兼容层生命周期缺陷，不是 PyTorch FSDP 的已知必然行为。

提交 `1a719099` 做了两件事：

1. `_mark_fsdp_param_var()` 不再保存 `_local_tensor`，并清除旧 Var 上残留的普通 DTensor
   `_local_tensor`；三个 public method 改为从 state/entry 动态得到 local/full 值。
2. 新增 `test_fsdp_var_metadata_does_not_accumulate_replaced_vars`，并加强 full parameter、
   gradient shard 的 `_local_tensor` 断言，覆盖重复替换和 reshard 后释放。

此前已经落地的相关修复仍保持不变：初始 shard device-side materialize、冻结 shard/full
reference 释放、冻结叶子参数 registry 清理、冻结 forward output materialize，以及完整梯度
从 shard 重建。这些问题同样属于 Jittor adapter 与 Jittor Var 生命周期/惰性执行语义的差异，
不是 PyTorch 本身的通用缺陷。

## 验证证据

### CPU 定向回归

```text
Ran 32 tests in 3.390s
OK
```

覆盖 trainability、初始 shard 脱离 full parent、临时 full release、重复 full/shard/gradient
替换、冻结 forward output、flat/non-flat gradient、optimizer 和 public DTensor methods。

### 真实 CUDA 20-cycle 生命周期探针

隔离状态和原始日志位于：

`$JITTOR_LAB_ROOT/_state/transformers_70b_sft/diagnostics/memory-fix/cuda-metadata-current/`

关键输出：

```text
CUDA_FSDP_METADATA_OK {
  'device': 'cuda:0',
  'used_mib_first_last': [0.0, 0.0],
  'lived_vars': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  'allocator_live_delta': 4194304
}
```

`used_mib` 是 Jittor allocator 的该探针统计，在本配置下被整 MiB 四舍五入为 0；判断是否
增长以 `lived_vars` 和 allocator live delta 为准。探针使用真实 CUDA kernel/allocator，未启用
CPU fallback。

### 真实 checkpoint 小模型验证

修复后另有真实 `meta-llama/Llama-3.2-1B` safetensors checkpoint 的 CUDA/FSDP 训练验证：
真实 tokenizer 和权重加载成功，连续 8 steps loss finite，CUDA framework memory 约
`2507 MiB` 且全程稳定，`lived_vars` 从 `45` 降至 `27` 后保持稳定。该验证使用本地
checkpoint：

`/home/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08`

原始运行状态未版本化，保存在 `$JITTOR_LAB_ROOT/transformers_70b_sft/diagnostics/memory-fix/`
对应目录中。现有通用 70B harness 不直接接受该模型的 tied embedding（checkpoint 没有独立
`lm_head.weight`），因此没有把一次错误的 harness 启动当作模型修复失败。

## 修复前对照与边界

修复前提交 `fba9982421c42038f2a8391e7e96cf26a9e70a4b` 的 Llama 3.1 70B No Robots 正式
60-step run 曾记录 Jittor framework peak 从 `31.409 GiB` 增至 `67.854 GiB`，而 PyTorch
约 `27.171 GiB`；独立 `nvidia-smi` 峰值分别为 `68.493/33.946 GiB`。该数据是修复前证据，
用于说明问题和回归目标，不能作为 `1a719099` 的现状结论。修复后尚未重新执行六卡 70B
60-step 完整验收，因此本报告不宣称 70B 长训练已经达到固定显存高水位。

当前最强结论是：导致显存随 FSDP metadata replacement 增长的强引用根因已在 CPU 回归和
真实 CUDA 20-cycle 探针中消除，并在真实 Llama 1B checkpoint 的短训练中观察到稳定显存。
多 rank NCCL 端到端和 70B 完整长训练仍需在资源允许时复验；Jittor 的一般 allocator cache、
lazy graph 和算子编译固定成本也不能由本修复单独消除。

## 复现命令

CPU 定向回归：

```bash
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
unset JITTOR_CUDA
export nvcc_path=''
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/transformers_70b_sft/diagnostics/memory-fix/fsdp-unit-cpu/jittor-home"
export cache_name=fsdp-unit-cpu
python -m unittest tests.compat.torch.test_torch_compat_fsdp2
```

真实 CUDA 探针（每次使用新的 `JITTOR_HOME`/`cache_name`）：

```bash
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
export CUDA_VISIBLE_DEVICES=6
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/transformers_70b_sft/diagnostics/memory-fix/cuda-metadata-current/jittor-home"
export TMPDIR="$JITTOR_LAB_ROOT/_state/transformers_70b_sft/diagnostics/memory-fix/cuda-metadata-current/tmp"
export HOME="$JITTOR_LAB_ROOT/_state/transformers_70b_sft/diagnostics/memory-fix/cuda-metadata-current/home"
export cache_name=memory-fix-cuda-metadata-current
python - <<'PY'
import gc, types
import jittor as jt
import jittor.compat.torch
from jittor.compat.fsdp2 import shard as fsdp_shard

assert jt.has_cuda and int(jt.flags.use_cuda) == 1
with jt.flag_scope(use_cuda=1, use_stat_allocator=1, use_sfrl_allocator=0):
    owner = types.SimpleNamespace()
    entry = types.SimpleNamespace(owner=owner, attr="weight", shard=None,
                                  full_param=None, requires_grad=True)
    state = types.SimpleNamespace(true_fsdp_initialized=True, true_fsdp_flat=False,
                                  true_fsdp_unsharded=True, true_fsdp_params=(entry,),
                                  true_fsdp_module=None)
    live = []
    for _ in range(20):
        shard = jt.ones((1024 * 1024,), dtype="float32")
        entry.shard = shard; owner.weight = shard
        fsdp_shard._mark_fsdp_param_var(shard, state, entry, "shard")
        full = jt.ones((1024 * 1024,), dtype="float32")
        entry.full_param = full; owner.weight = full
        fsdp_shard._mark_fsdp_param_var(full, state, entry, "full")
        grad = jt.ones((1024 * 1024,), dtype="float32")
        fsdp_shard._mark_fsdp_param_var(grad, state, entry, "grad_shard")
        entry.full_param = None; owner.weight = entry.shard
        del shard, full, grad
        gc.collect(); jt.gc(); jt.sync_all(True)
        live.append(jt.liveness_info()["lived_vars"])
    assert len(set(live)) == 1, live
    print("CUDA_FSDP_METADATA_OK", live)
PY
```

