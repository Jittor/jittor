# verl Jittor 权重传输与 1-step PPO 真实 CUDA 门禁

- Status: Adapter/PPO two-rank verl and framework four-rank NCCL/FSDP2 gates accepted on real CUDA
- Last reviewed: 2026-08-26
- Jittor baseline: `cbac62c9`
- verl source: `3d66a3d7ca1cf783df949816ec6862d5a7af9406` plus existing adapter edits
- vLLM source: `51a99565c398c8320de8131e07731c75c52eb87c`
- Owner: verl/vLLM external-adapter maintainers
- Review when: verl bucket protocol, actor/critic engines, adapter transport, or rollout weight loader changes

## 结论

外部 `verl_jittor` adapter 的 numpy-over-ZMQ weight transport 已在真实 RTX 4090
上完成 sender/receiver 往返，不再只有 import 或函数替换证据。lazy import hook 对
canonical `verl.workers.rollout.vllm_rollout.bucketed_weight_transfer` 模块生效，实际
`BucketedWeightSender.async_send_weights` 与
`BucketedWeightReceiver.receive_weights` 均运行 adapter 实现。

探针发送 fp32、fp16、bf16 三个 CUDA 权重。sender 按 adapter 契约转为连续 fp32
host array，经 ZMQ REQ/REP 逐项传输；receiver 在 CUDA 上重建 Jittor tensor。名称、
shape 和逐元素值全部一致，`rotary_emb.inv_freq` 按协议被过滤。

同一 transport 随后连接到真实 vLLM V1 UniProc EngineCore 内的 Qwen3-0.6B。
receiver callback 调用模型原生 `load_weights`，将
`model.layers.0.input_layernorm.weight` 增加 `0.125`；EngineCore 模型参数读回
`delta=0.125000`，随后通过同一 loader 恢复原值并逐元素验证。

关闭 prefix cache 后，harness 还在更新前、更新后和恢复后分别执行真实 1-token
rollout 并读取 logprob。稳定复跑中 token 均为 `12095`；未更新模型的两次基线抖动
为 `0.001110`，权重更新令 logprob 改变 `-0.365747`，约为自然抖动的 329 倍；恢复
权重后 token 与 logprob 回到基线容差内。因此 transport 变化确实进入推理计算，
不是只更新了一个未被使用的参数表。

最后，sender/receiver 被放入两个真实 Ray `num_gpus=1` actor，分别绑定物理 GPU 2
和 GPU 3，并使用两个独立、串行预热的 Jittor cache。同一组 fp32/fp16/bf16 权重
跨 Ray 进程完成 ZMQ 往返，driver 收到的名称、dtype、shape 和逐元素值全部一致。

同一外置 adapter 随后完成 tiny Qwen3 的真实单卡 1-step PPO。模型保留 Qwen3
tokenizer 和完整词表，使用 2 层、hidden size 64、约 9.80M 参数的本地 checkpoint；
训练 batch 为 4，prompt/response 上限为 32/16，actor 与 critic 均为 FSDP，rollout
为 async vLLM V1/UniProc，优势估计为 GAE。完整 dense 路径依次执行 rollout、训练前
权重同步、old log-prob、reward、critic value、GAE、critic backward/optimizer、actor
backward/optimizer，以及训练后权重同步，最终 `training/global_step=1` 并正常退出。

最终一步的 actor/critic grad norm 分别为 `1.878197` 和 `0.368663`，不是空图或
CPU fallback。rollout 与训练侧概率差最大 `2.416e-8`、均值 `1.083e-8`，Pearson
相关系数 `0.9999851`；训练前后 vLLM 权重同步分别约 `0.55s` 和 `0.51s`。该运行
仍包含少量首次 JIT，因此 `395.99s/step` 只作为功能门禁耗时，不作为性能结果。

相同模型和训练配置随后以 `use_remove_padding=True` 完成优化态 1-step PPO。
`NanoVector.numel()` 和 TensorDict device 修复在实际 old log-prob、critic 与 actor
路径中均生效；最终 actor/critic grad norm 为 `1.918660/0.908953`，rollout 与训练侧
概率最大差 `2.379e-8`、均值 `1.067e-8`，Pearson 相关系数 `0.9999831`。训练前后
权重同步分别约 `1.29s/0.69s`，`training/global_step=1` 并正常退出。

最后，相同 batch 4/remove-padding/FSDP/GAE 链路在真实 Qwen3-0.6B 上完成 1-step
PPO。actor 与 critic 各为 `596.05M` 参数，rollout 仍使用 vLLM V1/UniProc；为适配
单张 24GB GPU，将 vLLM memory utilization 设为 `0.10`，并启用 Jittor 原生
save-mem，device limit 为 14 GiB。训练前/后权重同步约 `42.70s/19.37s`，最终
actor/critic grad norm 为 `22.363432/180.209122`，`training/global_step=1`。
rollout/训练概率差门禁有效，最大值 `0.035094`、均值 `0.003824`，Pearson 相关系数
`0.9993625`。

该规模首先暴露两个明确的峰值内存问题：vLLM `0.20` 时 old log-prob 的 tied
embedding transpose OOM；降至 `0.10` 后，Torch shim 的 Adam bias-correction 除法
又把 float32 二阶矩整体提升为 float64。后者在 `5267eba3` 修为 state-dtype 标量，
计算图回归确认 float32 AdamW 不再产生 float64 Var；完整 optimizer 与 FSDP2 回归
分别为 `22 passed` 和 `13 passed`。save-mem 负责余下的真实 float32 optimizer
峰值，未改变模型、batch、序列长度或训练 dtype。

同一 Qwen3-0.6B 配置随后在两张 RTX 4090 上完成 replicated compatibility 模式的
1-step PPO。两个 Ray worker 分别绑定一张真实 GPU、使用独立 Jittor cache，并各自
执行完整 batch 的模型路径；adapter 只让 rank 0 向 controller 回传结果，避免 verl
把两个相同的全量输出再次拼接。最终 packed token 三方计数均为 `141`，两个 worker
的四个 actor micro-batch 均为 `29/37/38/37`，`training/global_step=1`，actor/critic
grad norm 为 `21.367786/190.710083`。rollout 概率门禁有效，最大差 `0.023153`、
均值 `0.002405`，Pearson 相关系数 `0.9996167`。

这次双卡门禁修复了三个 adapter 进程边界问题。首先，Python spawn 会恢复父进程的
`sys.path`，令 EngineCore 同时加载 `device-2_3` 与子进程 device cache 中的两份
`jittor_core.so`，最终报 `Op fused not found`；EngineCore loader 现在先移除 cache
root 下的继承路径再导入 Jittor。其次，Ray worker 原先都使用可见卡列表
`device-2_3`，现在按 local rank 映射为 `device-2`/`device-3` 独立 cache。最后，
adapter 的 distributed shim 明确保持进程内 world size 为 1，而 verl controller 的
outer world size 为 2；旧 collect 会把两个完整输出从 `142` 拼成 `284`，但序列
offset 仍为 `142`。replicated 模式现在仅收集 rank 0，仍保留 rank 1 的实际执行。

该运行包含 GPU 3 的大量首次 shape-specialized JIT，trainer 记录
`1352.93s/step`，其中 actor update 为 `1047.46s`，因此只作为功能门禁耗时。actor
和 critic 峰值显存分别为 `20.81/21.14 GiB`；训练前 rank 0 权重同步日志为
`197.73s`，训练后为 `123.32s`。

框架级原生 FSDP2 随后通过 `jittor.distributed.launch` 在物理 GPU 2/3 上完成真实
双 rank NCCL smoke。`nn.Linear(4, 3)` 的 15 个参数被补齐为 16 个元素，每个 rank
只持有 8 个元素的 flat shard；all-gather 可逐元素还原完整初始参数，两个 rank 的
独立 loss 梯度经 reduce-scatter 求平均后执行 sharded SGD，一步更新与独立 NumPy
计算的最大误差为 `1.4901161e-8`，两个本地 shard 均发生有限非零更新。

该真实运行发现 flat gradient-sync 分支会被非 flat 分支的局部变量 `shard` 遮蔽同名
导入模块，因而抛出 `UnboundLocalError`。`6003beb9` 将局部梯度改名为
`shard_grad`，并新增 flat/non-flat slicing 单测、真实双 rank 测试和维护的
`nox -s nccl` 门禁。门禁逐 rank 串行预热独立 cache，关闭无关 MPI/CUTT/CUTLASS/MKL
探测，再并发执行两 rank pytest；从空 cache 完整运行正常以 `rc=0` 结束。

`cbac62c9` 将维护门禁和三个真实 NCCL/FSDP2 用例从固定 world-size 2 泛化为任意
`world_size >= 2`。在物理 GPU 4--7 上的单机 world-size 4 运行中，15 个参数补齐为
16 个元素，每 rank 仅持有 4 个 flat-shard 元素；四份不同输入的梯度经
reduce-scatter 求平均后完成 sharded SGD，all-gather 重组结果通过独立 NumPy 参考的
`rtol=2e-5, atol=2e-5` 门禁。SUM/MAX/MIN/PRODUCT、tensor/object gather、barrier、
嵌套分片和 full-state reload 也在四个 rank 全部通过。保留缓存的默认 world-size 2
复验仍为每 rank `3 passed`，说明原有入口未回归。

`a1ef6bdd` 进一步把该原生 world-size 2 路径接入 verl worker/controller，而不是继续
使用 replicated compatibility 模式。Ray worker 在启动时通过显式 opt-in 的
`JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1` 建立 Jittor NCCL communicator；canonical
`torch.distributed` 保持 rank `0/1`、world size `2`，actor 与 critic 均使用 verl
原生 `strategy=fsdp2`。父 FSDP module 不再重复管理已由子 module 分片的参数，完整
state dict 可从 rank 0 broadcast 后恢复每个 rank 的本地 flat/non-flat shard。

真实 tiny Qwen3 两卡 1-step PPO 随后在物理 GPU 2/3 上完整退出，最终
`training/global_step=1`。链路覆盖双 rank actor/critic FSDP2 初始化、初始权重同步、
async vLLM rollout、old log-prob、critic value、GAE、critic backward/optimizer、actor
backward/optimizer 和更新后权重同步。critic/actor grad norm 分别为
`0.265052/1.988398`，不是空图或单 rank fallback；rollout 与训练概率最大差
`2.7566e-8`、Pearson 相关系数 `0.9999843`，更新后权重同步约 `1.88s`。

接入过程还补齐了 verl 实际依赖的 Torch distributed 表面：Ray 动态 NCCL bootstrap、
WORLD/singleton process group、tensor/object gather、broadcast/barrier、SUM/AVG/MAX/MIN/
PRODUCT reduction、Store/c10d/rendezvous import，以及 FSDP2 的 ABC metaclass、
`Module.to_empty()`、真实 forward unshard hook 和 `torch.nn.utils.clip_grad` 私有梯度裁剪
入口。未实现的 Gloo、P2P 与 symmetric memory 保持 fail closed，不再报告假可用。

## 验证

- 独立 cache 首次完成 Jittor core、CUDA extern 与 MKL 初始化。
- 一次性最小复现输出：`VERL_WEIGHT_TRANSFER_OK 3 True`。
- 可复用 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/weight_transfer_smoke.py`。
- harness 热 cache：`verl weight transfer smoke OK weights=3`，约 8 秒。
- `$JITTOR_LAB_ROOT/verl_jittor/scripts/run_all.sh` 已把该脚本加入默认 CUDA gate，
  `--cpu-only` 时显式跳过；shell syntax 与 Python compile 检查通过。
- 真实模型应用 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/vllm_weight_apply_smoke.py`；设置
  `VERL_VLLM_MODEL` 时由 `run_all.sh` 追加执行。
- Qwen3-0.6B V1/UniProc：模型加载约 7 秒、engine warmup 约 2 秒，输出
  `verl vLLM weight apply smoke OK ... loaded=1 delta=0.125000`。
- 首次新增 rollout 图包含冷 JIT，耗时不作为性能结果；补齐 cache 后重复 harness
  输出 `score_delta=-0.365747 baseline_jitter=0.001110`。
- Ray 跨进程 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/ray_weight_transfer_smoke.py`，最终输出
  `verl Ray weight transfer smoke OK sender=2 receiver=3 weights=3`。
- Ray actor 设置 `DISABLE_MULTIPROCESSING=1`，使用核心已有的 host signal-ownership
  契约，避免 Ray 回收子进程触发 Jittor SIGCHLD quick-exit；operator compiler 保持串行。
- 1-step PPO 离线输入已生成 8 条 train/4 条 validation parquet；真实 Qwen tokenizer
  和 `RLHFDataset` 过滤后保留全部 8 条，并返回正确 raw chat/reward metadata。
- tiny Qwen3 单卡 dense PPO：`Training Progress: 100% 1/1`，actor/critic update、
  optimizer step 与更新后权重同步全部执行；最终 `response_length/mean=16`、
  `response/aborted_ratio=0`、`training/global_step=1`。
- tiny Qwen3 单卡 remove-padding PPO：`Training Progress: 100% 1/1`，同样完成
  rollout、old log-prob、GAE、critic/actor update 和更新后权重同步；最终
  `response_length/mean=16`、`response/aborted_ratio=0`、`training/global_step=1`。
- Qwen3-0.6B 单卡 remove-padding PPO：`Training Progress: 100% 1/1`，完成相同
  rollout、critic/actor optimizer 与更新后权重同步闭环；最终
  `response_length/mean=16`、`response/aborted_ratio=0`、`training/global_step=1`。
- Qwen3-0.6B 双卡 replicated compatibility PPO：两个 worker 分别使用物理 GPU 2/3
  和 `device-2`/`device-3` cache，均执行 141-token 全量路径及
  `29/37/38/37` actor micro-batch；controller 只收集 rank 0，最终
  `Training Progress: 100% 1/1`、`response/aborted_ratio=0`、
  `training/global_step=1`。
- EngineCore 污染 spawn 探针从 parent `device-2_3` cache 启动，在子进程切换单卡并
  导入 verl/vLLM target，确认只加载子进程 cache 的 Jittor core 后正常退出。
- `NanoVector.numel()` 定向测试 `3 passed`；真实 GPU cache 探针返回 `24`。
- `torch._C._nn._parse_to("cpu")` 现在返回 `device(type='cpu')`；真实 CUDA
  TensorDict construct/index/`.cpu()` 回归文件 `3 passed`。
- 直接双 rank FSDP2 smoke：rank 0/1 均报告 `world_size=2`、
  `flat_total_numel=15`、`flat_shard_numel=8` 和 `nccl_ops=true`；完整参数更新最大误差
  均为 `1.4901161193847656e-08`。
- Ray 两 GPU 动态 bootstrap 探针确认 actor 启动前无 `JT_NCCL_*`，初始化后 rank
  `0/1`、world size `2`；SUM 为 `3`，对象 gather 为 rank `0/1`，每 rank flat shard
  为 8 个元素，并可在同一 actor 导入、patch vLLM。
- tiny Qwen3 原生双 rank FSDP2 PPO：`Training Progress: 100% 1/1`，最终
  `training/global_step=1`、`response/aborted_ratio=0`，critic/actor update 和训练后
  vLLM 权重同步全部执行。
- 维护门禁 `CUDA_VISIBLE_DEVICES=2,3 python -m nox -s nccl` 从空 cache 通过；最终
  代码热态复验两个 rank 各为 `3 passed`，launcher 报告 `all ranks done, rc=0`。
- 泛化后的维护门禁以 `JITTOR_NCCL_WORLD_SIZE=4` 在四张真实 RTX 4090 上从独立
  rank cache 运行，rank 0/1/2/3 各 `3 passed`，launcher 为 `rc=0`，完整 nox
  会话约 21 分钟；随后保留缓存的默认 world-size 2 复验两个 rank 各 `3 passed`
  （约 22 秒）。
- 四 rank 命令设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`、
  `JITTOR_NCCL_WORLD_SIZE=4` 与 `nvcc_path=<nvcc>` 后执行 `python -m nox -s nccl`；
  环境为 Python 3.11.15、CUDA 12.2、NVIDIA driver 595.84 和四张 RTX 4090。
- 完整 FSDP2/gradient compatibility 回归 `24 passed`；`check_repo_layout.sh` 通过，完整
  `tests/structure` 为 `218 passed`。

## 隔离方法

verl 的 `vllm_rollout/__init__.py` 会同时加载完整 vLLM server 和无条件 NPU utility，
把 `vllm_ascend` 等与 transport 无关的可选依赖引入最小测试。harness 注册同名父
package 的真实 `__path__`，但不执行其 `__init__`，随后按 canonical module name
导入真实 `bucketed_weight_transfer.py`。因此 adapter 的 meta-path lazy hook 仍被
验证，同时没有用假的 sender/receiver 实现替代被测代码。

PPO 运行状态和编译 cache 均未版本化。最终 Hydra 配置与持久化 Ray worker 日志
保存在 `$JITTOR_LAB_ROOT/_state/verl-ppo/20260824/run-dense-final/`；其中配置、
TaskRunner stdout 和 stderr 的 SHA-256 分别为
`4fbe018a4cf8b46cb52edeb4768515f1810ab07eec3c32b849c02365be6caf26`、
`7129498ac277b2fe942047f6380118473bf0f2891c938d08a846eef7a0d58d1c` 和
`745c1370f56e03ea7488965c6eefa530dda6fa33f2d54b20c454c8542ffd4a35`。
remove-padding 运行的对应配置、stdout 和 stderr 保存在同级
`run-remove-padding-final/`，SHA-256 分别为
`c43733ab9da51afafb8021228a756033d635e9d6f24aacfe6a3fe49ee1561a84`、
`13fc4d2df67f82999f32c40c17aac7e7245e4bc0df94c0a113e758ae55dd2563` 和
`b7a4b36a7114d3122e7163faa381d2785e09a2d263682355cd8d5bc946f41410`。
Qwen3-0.6B 运行的对应配置、stdout 和 stderr 保存在同级
`run-qwen3-0.6b-swap/`，SHA-256 分别为
`2c39c9d95924a70ea49352862af209999b4ec46655b2bbf4ea228806ae884cfe`、
`e133540f31bea30d674b6ddae0070c1568e915e2572123459bccb4bddb1c8bfa` 和
`b182ad85a8073a5ece0320288734b2c7b93fca7ddd6ff85ba410e7218b900315`。
双卡 replicated 运行保存在同级 `run-qwen3-0.6b-2gpu-attempt9/`；Hydra 配置、
TaskRunner stdout/stderr 的 SHA-256 分别为
`8b46bbb659aa6cd58f71ae7f352741d5f078c3a99fa9b3065300da0a6de322e7`、
`992ed77f911efaf687606fc61f93b7a1f2640806be2f54dabd794c16d049238a` 和
`63ff796cb51337aba5fde28598c87a2bf1e6535c3bdc4105a251c5c21a480b49`。
rank 0/1 stderr 的 SHA-256 分别为
`d19929a86fc232b822dc5893c20209c57e34939eb3675fd882edb8917b34d7af` 和
`2123f9626b67f8c82bae2badb0c942bd930df75e7c0ac5df41596018dfe343bd`。
tiny checkpoint 的 `model.safetensors` SHA-256 为
`12cef412ee8181c27fb13143022c1b29b42663a66121047e5782d9b4afb7aae5`；模型和数据
均为本地离线产物。Qwen3-0.6B 权重 SHA-256 为
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
未版本化 adapter `sitecustomize.py`、bootstrap、nested shim 与 PPO harness
SHA-256 分别为
`e596deefe6ee776de5e2248e93649df8ea7140aebcbc9edce34e393467742f78`、
`b34f37e27821f384ec2a0b6f4a0553b587c831bcc5d61ef76e85d3de94b099b9`、
`36fe3af2de9ae980a8f929718ea5eda0bb54cea5d326cc3a76806ebbc6690175` 和
`8714d4e50a3868824ba7d30e7d936a6fdfd400b4b06777f876bb6cd7bf7616ba`。

原生 FSDP2 日志保存在
`$JITTOR_LAB_ROOT/_state/verl-fsdp2/20260825-native-nccl/`。直接 smoke 的 rank 0/1
日志 SHA-256 分别为
`e28cae6f6620d6127a376ad7941083882f4e518c8b41435d1e704056a1398633` 和
`b88660d4c5e020dc3830b43f341a58f1e041c0736d109eb01571b79d3b00f412`；维护 nox 门禁
的 rank 0/1 pytest 日志 SHA-256 分别为
`020f8aec55945e0543965caaecb4508528fece8a7b4eb0e5e0fc4b9c16058ad2` 和
`110fbe30df1e51228622979eb92464451cc0450c5c56073b104f0a8f16213b83`。

原生 verl 两 rank 运行保存在
`$JITTOR_LAB_ROOT/_state/verl-fsdp2/20260825-native-verl-ppo/attempt13/`。Hydra 配置、
TaskRunner stdout/stderr 的 SHA-256 分别为
`78e7db9dc853e8dabd80e139257fe76a48827e2eb005a26878833e5a4002b0a5`、
`d59ba2aff95c9ffab6d2ac1ff6f4dfd2c35e283ba939db28442cc6470f3e51bb` 和
`e3e014fdb4ddab461d5ef8d61e1393c916df81e4d0dd059c20505dda094a2783`。
最终代码的双 rank 日志位于同级 `nox-final3/`，rank 0/1 SHA-256 分别为
`dd4c52bb5488e95a7ade1cfe9a4af9b51f197f17df1c270ed52c4f1b011d5fe4` 和
`cb5e8a7ce5933cc15e3d96960f025394f967227f216616661aaebca9363ddbca`。
本次四 rank nox 日志持久化于
`$JITTOR_LAB_ROOT/_state/verl-fsdp2/20260826-native-nccl4/world4/`，rank 0--3
SHA-256 分别为
`779893710cd429294ff5541e42867d0c383abb3a985b6007e1fbfd579658bf82`、
`43c43d4714f444131596de6eda98ed3e9da5dd6c7fdd65c0b6d43754a4c0b430`、
`2528a6e22af522299bf43fd9caff9dbb3d685110c3c982e822a9c206efd37254` 和
`32af49e6908f995fa861dde238921a19b63f21b34e54da0af877e2c4290d8171`。
默认双 rank 保留缓存复验日志位于同级 `world2-retained/`，rank 0/1 SHA-256 分别为
`5008f015ea5b281d3f357050ac7a3f08cec35b7c8151759fceab819ac7e8b1ec` 和
`4ce79cc8575997480c27f80097da03f86235417f1d8961c11180e14b73cf68ae`。
未版本化 `verl_jittor/distributed_shim.py` 与 `vllm_jittor_ops/bootstrap.py` SHA-256
分别为 `e6fe5d7753eef110ed16bc701edd560c4c72f1eb67e31f9de278aa551158beb5` 和
`d4b6d124b8df969f65e4ae265a5b5fd1c7c9a09b4a7d0413973e0d1a68c17216`；PPO harness
仍为 `8714d4e50a3868824ba7d30e7d936a6fdfd400b4b06777f876bb6cd7bf7616ba`。

## 边界

完整 PPO 结论限定为 tiny Qwen3 与 Qwen3-0.6B、固定 reward；tiny 覆盖单卡
dense/remove-padding 和双卡原生 FSDP2，0.6B 覆盖单卡 remove-padding 与历史双卡
replicated compatibility。框架门禁证明单机 world-size 4 的 NCCL collective、FSDP
参数分片和一步线性层更新；原生 verl PPO 结论仍只到单机 world-size 2，不宣称性能
扩展，也不把历史 replicated 结果改写为原生分布式。多节点 verl、超过两 rank 的
verl、0.6B 原生 FSDP2、多 actor、真实 reward model、长序列、稳定热态性能，以及
NPU/ROCm 均不由本报告宣称通过。
