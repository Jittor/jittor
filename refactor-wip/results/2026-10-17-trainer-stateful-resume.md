# 显式 Stateful Trainer 的新进程恢复验收

- Status: PASS，仅限下述显式入口与有界 fixture；HF 全计划仍部分完成。
- Date: 2026-10-17，Asia/Shanghai。
- Baseline commit: `48de31b4157fa128c5e1e19b052fefa3be7ed098` 加本次源码和测试；完整 hash 在未版本化 manifest。
- Remote sync: 已 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为本地 HEAD 祖先，无需 merge。保留原工作树，未 stash、清理或合入另一条 refactor 远端。
- Owner: cgq / Jittor Torch 兼容层维护者。
- Review when: Trainer/Accelerate/torchdata 版本、prepared-loader lookahead、loader/sampler 状态协议、compat 公开入口（包括 `owned_runtime_hook`、`transaction`、`module_patcher`）或设备 RNG 实现变化。

## 入口和分流

默认 Trainer 的 workers=2 checkpoint 不保存完整数据加载状态，历史 CPU 复现中恢复段从随机增强开始偏移。直连 StatefulDataLoader 成功也不能证明 Trainer 恢复成功。本次按用户授权增加显式入口：

```python
from jittor_adapters.trainer_stateful import stateful_trainer_class

StatefulTrainer = stateful_trainer_class()
trainer = StatefulTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    stateful_dataset_id="fixed-dataset-and-transform-v1",
)
trainer.train()
# 新进程中重建相同配置，然后调用：
trainer.train(resume_from_checkpoint="output/checkpoint-5")
```

通用 loader 状态属于 `compat/torch/installers/data.py`；Trainer 专属 checkpoint 接线属于独立 `adapters/jittor_adapters/trainer_stateful.py`。入口不安装全局补丁，可传自定义 Trainer 子类；版本通过共享 `require_version` 拒绝未经验证的组合。本轮钉 Transformers **4.56.2**、Accelerate **1.10.1**。

入口设置公开 `accelerator.dataloader_config.use_stateful_dataloader`。callback 保存 **prepared loader** 的公开 `state_dict()`，从而计入 Accelerate 的一个 batch lookahead，而不是把底层已预读但未消费的 batch 当作完成。文件 `stateful_dataloader.pt` 与 Trainer checkpoint 同目录，含版本、配置、step/epoch、loader 状态、显式 loader/sampler generator。恢复时核对 Trainer 元数据，并在创建下一个训练 iterator 前加载；`ignore_data_skip=True` 防止重复跳过，callback 按已消费 microbatches 修正公开 epoch。

`stateful_dataloader_factory(trainer)` 可返回调用方拥有的、尚未 prepare 的 StatefulDataLoader，由 adapter 调公开 `accelerator.prepare`。native PyTorch 的验收 factory 使用 torchdata 的 RandomSampler/BatchSampler，并让 dataset transform 提供 Python/NumPy RNG 状态；普通 sampler 和任意 dataset 不会因启用开关而自动拥有完整恢复协议。Jittor 侧由逻辑线程 worker 捕获相同增强代码使用的全局 RNG。

普通 DataLoader 不保留已消费的完整 batch/worker 历史。显式 StatefulDataLoader 保存有限 epoch batch plan，默认 `max_stateful_indices=100000`，按实际样本索引数限制，拒绝无长度/无限计划。状态验证覆盖配置、cursor、索引范围、worker 分配、Python/NumPy RNG 和 generator；pending load 后再保存也保留完整状态。persistent iterator/pool 在直连 loader 层复用并维持逻辑 worker RNG。

## 环境和协议

两侧独立解释器使用匹配的 Python 3.11.3、Transformers 4.56.2、Accelerate 1.10.1、PEFT 0.17.1、Tokenizers 0.22.1、torchdata 0.11.0；oracle 为真实 PyTorch **2.6.0+cu124**，Jittor 为源码 **1.3.11.0**。torchdata 使用 `pip --no-deps` 安装，未替换 oracle Torch。runner 检查 runtime identity，离线生成固定 tiny BERT/LoRA 初始权重，Jittor 加载同一 fixture。

有界验收配置为 FP32、64 samples、batch size 2、G=2、RandomSampler、模型 dropout=0.2、workers=2/prefetch=2、Python random 与 NumPy legacy global 随机增强、sampler seed=821、loader seed=823。训练连续 **12 updates**，对照在第 **5** 步保存退出、由全新解释器恢复至第 **12** 步。持续运行与中断运行使用相同 `max_steps=12`，用 stop callback 中断，不改变 scheduler horizon。

恢复只与同 runtime 的连续轨迹比较，`atol=rtol=0`。800 个记录项覆盖样本顺序、增强 draws/输入、全部 trainable gradients、逐步参数、optimizer、scheduler 和最终模型张量；不要求 PyTorch 与 Jittor 的 RNG 序列相同。恢复段报告 `stateful_restored_step=5`、`global_step=12` 和 **14 个新 microbatches**，确实完成后续 7 次更新。

| Runtime / device | 比较结果 | 参数实际 device | fallback | 产物子目录 |
| --- | --- | --- | --- | --- |
| PyTorch CPU | 800/800 exact | cpu | 不适用（真实 PyTorch） | `pytorch-cpu-12/` |
| Jittor CPU | 800/800 exact | cpu | 0 | `jittor-cpu-12/` |
| PyTorch CUDA | 800/800 exact | cuda:0 | 不适用（真实 PyTorch） | `pytorch-cuda/` |
| Jittor CUDA | 800/800 exact | cuda:0 | 0 | `jittor-cuda/` |

CUDA 先检查旧日志、实时 GPU/进程、显存与 UUID，再执行单步 smoke；确认实际 CUDA 参数和 fallback=0 后才扩展。Jittor 使用系统 libstdc++ preload、MPI=0/NCCL=1、串行首次 JIT、隔离运行目录及已有 warm cache。共享 GPU 上存在其他负载，此处没有吞吐/性能结论。历史 continuous run 的 `gen_ops_curand_random` SIGSEGV 根因仍开放，后续通过不构成修复证明。

较早的 CPU 4 vs 2+2 也分别 272/272 exact。native 初版 fixture 缺少 checkpointable sampler/batch-sampler，曾有 87/78 项偏移；诊断产物原样保留，不并入最终通过结果。

## 维护性门禁与复现

新增 `adapters/tests/test_trainer_stateful_integration.py`，用独立双解释器和全新纯 CPU cache，执行 dataset=32/dropout=0.1、4 vs 2+2；两 runtime 各 **272/272 exact**。该文件连同 5 项门禁自身契约共 **6 passed、0 skipped**，537.78 秒包含冷编译。首次尝试遭遇 PyTorch resource-sharer 的 AF_UNIX 路径长度限制，已缩短实验 topic，失败日志保留；没有吞掉异常或转成 skip。

```bash
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
export REAL_TORCH_PYTHON="$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python"
export JITTOR_STATEFUL_PYTHON="$JITTOR_LAB_ROOT/transformers_compat/envs/jittor/bin/python"
export JITTOR_REQUIRE_TRAINER_STATEFUL=1 JITTOR_REQUIRE_REAL_TORCH=1
export JITTOR_TEST_REQUIRE_EXECUTION=1
export JITTOR_TORCH_SHIM=0 JITTOR_TORCH_STRICT_BOOTSTRAP=0
python -m pytest -q -s adapters/tests/test_trainer_stateful_integration.py
```

仓库还提供 `nox -s trainer_stateful`，校验两解释器路径并设置 require 环境；本轮直接执行了上面的底层 pytest 门禁，未重复执行 nox wrapper。依赖缺席在普通环境诚实 skip，在 require 环境为失败；不安装依赖、不联网、不回退到 shim oracle。

12 vs 5+7 使用同一 runner，将以下参数分别传给 continuous/first/resumed 三个独立进程；runtime/device/output 与环境按对应格配置，初始权重先由真实 PyTorch `initialize --runtime pytorch --weights-out ...` 生成：

```text
run --runtime <pytorch|jittor> --device <cpu|cuda> --weights <initial.npz>
    --max-steps 12 --sampler random --dropout 0.2 --dataset-size 64
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2
    --worker-random-transform --explicit-generators --stateful-dataloader
    --sampler-seed 821 --dataloader-seed 823 --output-dir <new-directory>
first:   追加 --stop-after 5
resumed: 追加 --resume-checkpoint <first/checkpoints/checkpoint-5> --start-step 5
merge --first <first/trajectory.npz> --resumed <resumed/trajectory.npz> --output <merged.npz>
compare --left <continuous/trajectory.npz> --right <merged.npz> --atol 0 --rtol 0
```

Trainer 恢复还依赖本任务此前已复现的 LambdaLR checkpoint 修复：将 functools.partial 等 callable 的状态字典保存，避免 weights_only 加载反序列化 callable；恢复保留当前构造的函数引用。该修复及定向 scheduler 回归一并提交，scheduler/helper 定向复验共 16 passed、0 skipped，未混入 AMP 或 native optimizer 改动。

loader 定向 **50 passed、0 skipped**；adapter 新 35 项加原 6 项 **41 passed**，包括配置拒绝、版本拒绝、缺文件、prepared state、元数据与公开边界的负向测试。runner 共用 `_training_state_helpers.py`，消除跨文件重复；其他未提交 trajectory runner 保留并改为导入同一 helper。

## 仓库门禁与支持边界

`core` 已跑：native **101 passed/43 skipped/1 xfailed**，Torch **217 passed/3 skipped/4 failed**。四项为独立 Tensor Linear 构造、boolean RHS 梯度、arange 默认 dtype、types.py 行预算，与此前观察到的四项相同；未进行干净 HEAD 归因对照。

完整 structure 初跑 **1390 passed/4 failed/6 skipped**。本任务的 `_named_tensors` 重复已修复，cleanup/nox/tier 定向复验 **54 passed/0 skipped**；其余三项涉及其他未提交 onboarding 的 toctree/MANIFEST 及 `py_converter.h` USER_CHECK 计数。完整 structure 未再次重复；布局检查通过，不宣称全仓绿。

当前入口仅验收单进程单设备、正 `max_steps`、首 epoch 内、完整累积窗口，恢复 checkpoint 必须在 epoch 结束前。多 epoch、accumulation tail、HPO、auto batch-size、model-only checkpoint、Seq2SeqTrainer、AMP、DDP/FSDP2 与真实数据指标仍不在本次验收内。默认 Trainer 历史 NOT_PASS 保留；直连 loader persistent 测试不证明 Trainer persistent/worker=4 组合已通过。

Jittor worker 仍为线程，RNG 范围仅 Python random/NumPy legacy global；用户其他线程并发访问这些全局 RNG、custom RNG、worker Torch RNG、mutable dataset hooks、IterableDataset、out-of-order 和非逐步 snapshot 均不在保证内。完整 batch plan 在上限内仍随样本数占用内存，不等于大数据流式 checkpoint。native torchdata 重建 worker 会额外消耗 loader generator：本次记录其 final hash，不要求其与 continuous 一致；sampler hash与首 epoch 训练轨迹仍 exact，不外推跨 epoch 等价。

## 未版本化证据

主产物根为 `$JITTOR_LAB_ROOT/_state/hf-stack-validation/trainer-stateful-20261017/`，比较日志在 `logs/{pytorch,jittor}-{cpu-12,cuda}-compare.log`；CPU 维护性门禁在 `$JITTOR_LAB_ROOT/_state/trainer-resume/cpu-5qlpw95k/`。loader 加固日志在 `hf-stack-validation/stateful-loader-hardening-20261017/logs/after-r2.log`。

主目录 `manifest.json` 记录基线/同步 SHA、最终源码与文档 hash、环境脚本、依赖记录、reports、轨迹、checkpoint 与日志 hash，包含集成门禁 summary。原始缓存、权重和失败诊断保留在 lab，没有写入仓库文档树。该 manifest 锚定具体 dirty-tree 验证输入，不将其他并行工作归入本任务提交。
