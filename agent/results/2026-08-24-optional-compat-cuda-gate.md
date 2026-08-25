# 可选依赖 fail-closed CUDA 门禁

- Status: Selected optional packages and native FlashAttention training accepted on real CUDA
- Last reviewed: 2026-08-25
- Commits: `566eae8e`, `2cf096d5`, `c2e340f8`, `90e00edd`, `9e69fa23`, `19820174`, `50fc95d5`, `a13cb06e`, `c8c43cf6`, `d500dc77`, `76b8a5a0`, `24cf00eb`, `95cd6f6c`, `c3e65b1d`, `fe97085a`, `1cd76dd9`, `f3df3274`, `797a6a97`, `5b838f0f`, `0f93f117`, `d77f04a2`, `1b47cbab`, `62251be1`, `1161a552`, `e0224e64`, `be036e7f`, `15bb9b74`, `f0b26e91`
- Owner: Torch compatibility and test-infrastructure maintainers
- Review when: optional package versions, Torch shim identity, or nox hardware
  environment contracts change

## 结论

项目现在提供 `python -m nox -s optional`，在预配置 CUDA 环境中 fail-closed
验证 TorchMetrics、mmcv-lite、MMEngine、PEFT、Safetensors、TensorDict 和部署的
FlashAttention adapter。session 在 pytest 前检查全部包，显式启用 Jittor Torch
shim 与离线模式，并在真实 CUDA 上运行五个兼容模块；缺少依赖时不再以 skip
充当通过。

PEFT 测试原先还要求 shim 的 `torch.__name__ == "torch"`，但部署契约是
`torch is jittor`，模块名保留 `jittor`，因此已安装 PEFT 时三个测试仍全部误跳过。
判断现在使用对象身份；`JITTOR_REQUIRE_OPTIONAL_DEPS=1` 下任何导入异常直接失败。

FlashAttention math fallback 原先接收 `dropout_p` 却没有传给 canonical SDPA，
会静默返回未 dropout 的结果。dense、packed 和 varlen fallback 现在都传递该参数，
并由真实部署 adapter 的 `dropout_p=1` 零输出回归锁定。

`optional` session 还支持显式 `JITTOR_FLASH_ATTN_JITTOR_SRC`。基础 optional 阶段
固定使用 deployed math adapter；随后独立的 native-required 阶段默认构建
hdim32/fp16 official kernel。这样 float32 math fallback 与 fp16 native-required 契约
不再共享互相矛盾的环境，native 构建或加载失败也不能用 fallback 满足门禁。

official backend 现在同时编译 forward、split-forward 与 backward kernel，不再定义
`FLASHATTENTION_DISABLE_BACKWARD/DROPOUT`。dense 和 varlen API 通过 `jt.Function`
保存 output、softmax LSE 与 RNG state，并在一阶反向中调用官方 `bwd/varlen_bwd`；
qkv-packed 训练绕过 inference-only direct packed fast path，经可微 split 复用同一
native backward。Torch SDPA 也允许无 mask 的 CUDA 训练与 `0 <= dropout_p < 1`
进入该 backend，未声明训练能力的第三方 backend 仍 fail closed 或回退 math。

C++ generator shim 原先每次返回固定 seed/offset，dropout 会重复同一 mask。现在
Philox state 读取 Jittor 全局 seed，并从共享 offset 分配 counter；重复
`manual_seed` 会把 offset 复位，连续调用则前进。forward 保存的 RNG state 直接传给
backward，确保同一 dropout mask 被重放。

## 环境

- Python 3.11.15，Jittor 1.3.11.0，真实 RTX 4090，CUDA toolkit 12.2.140。
- TorchMetrics 1.7.4、MMEngine 0.10.7、PEFT 0.17.1、Safetensors 0.8.0；
  TensorDict 0.10.0、FlashAttention adapter 2.7.4.post1；mmcv-lite 可从当前
  预配置环境导入。
- Native FlashAttention 源码提交 `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2`，
  从仓库边界外的官方 checkout 加载。
- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `JITTOR_TORCH_SHIM=1`、`use_cuda=1`、`use_parallel_op_compiler=0`。
- Native training gate 使用 `JITTOR_FLASH_ATTN_HEAD_DIMS=32`、
  `JITTOR_FLASH_ATTN_DTYPES=fp16`；nox 允许外部显式扩大这两个能力集合。
  当 dtype 集合包含 `bf16` 时，额外追加 bf16 dense backward 数值门禁；默认 fp16
  配置不会隐式扩大编译范围。
  外部 capability 与默认 `32/fp16` 取并集；包含 head dim 64、96、128、192 或 256
  时追加对应 fp16 dense backward 门禁，避免替换默认值后再触发第二次动态构建。

## 验证

- 修复前 `tests/compat/torch/test_peft.py`：`3 skipped`，实际 PEFT 导入探针成功。
- 修复后 PEFT：`3 passed in 126.50s`，覆盖 LoRA 冻结与梯度、200 步拟合、
  Safetensors adapter 保存/加载。
- TorchMetrics：`2 passed in 655.93s`，覆盖分类、回归、聚合和 required ops。
- mmcv-lite/MMEngine：`3 passed in 16.62s`，含 CUDA typed tensor 真实设备执行。
- 三模块同一 shim 会话：`8 passed, 1 warning in 17.83s`（warm cache）。
- TensorDict 与 FlashAttention 新增真实行为模块：`5 passed in 11.31s`，覆盖 CUDA
  构造/更新/index/lazy stack，以及 dense/packed/varlen attention、梯度和 dropout。
- 五模块加既有 loader/stub 契约同一 shim 会话：`16 passed, 1 warning in 28.37s`。
- 修复前真实 native 探针：forward 命中 official backend；`jt.grad` 警告输出不在图中
  并返回三个全零梯度，`dropout_p=0.2` 被明确拒绝。
- Native fused training 模式首次串行构建后，forward、数值 backward、dropout、GQA
  与 float32 cast 五项 `5 passed in 822.69s`；补齐 varlen 与 qkv-packed 后，热 cache
  七项同进程 `7 passed in 4.36s`。
- 独立 nox cache 的 native-required 七项 `7 passed in 96.18s`；完整 attention 模块
  在同一 source-required 环境 `41 passed in 499.76s`。dense SDPA 输出及 q/k/v 梯度
  对独立 NumPy softmax 导数，varlen 使用长度 3/4 两段对拍；qkv-packed 输出和 packed
  梯度与 dense native 完全一致。
- hdim32/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.011045/0.008340/0.011811/0.009316`，维护测试热 cache
  `1 passed in 21.16s`。默认 fp16 配置对该测试为显式 skip，`1 passed, 1 skipped`。
- hdim64/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.006614/0.009683/0.015307/0.008268`，维护测试热 cache
  `1 passed in 4.16s`。head dim 32 的 bf16 环境和 hdim64 的 fp16 环境均为
  `1 passed, 1 skipped`，证明 dtype/head dim 双条件 fail closed。
- hdim96/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.012326/0.012543/0.007809/0.009566`，维护测试热 cache
  `1 passed in 3.79s`。不匹配 head dim 时为 `1 passed, 1 skipped`；请求 `96/bf16`
  时 nox 选择 10 项，`head_dims=all,dtypes=all` 现精确选择 15 项且无重复。
- hdim128/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.007246/0.006798/0.010209/0.009305`，维护测试热 cache
  `1 passed in 4.16s`。不匹配 capability 时为 `1 passed, 1 skipped`；请求
  `128/bf16` 时 nox 选择 10 项，`all/all` 现精确选择 16 项且无重复。
- hdim192/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.008386/0.009887/0.008901/0.008372`，维护测试热 cache
  `1 passed in 3.77s`。不匹配 capability 时为 `1 passed, 1 skipped`；请求
  `192/bf16` 时 nox 选择 10 项，`all/all` 现精确选择 17 项且无重复。
- hdim256/bf16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.008780/0.008970/0.008186/0.010817`，维护测试热 cache
  `1 passed in 4.28s`。不匹配 capability 时为 `1 passed, 1 skipped`；请求
  `256/bf16` 时 nox 选择 10 项，`all/all` 现精确选择 18 项且无重复。
- hdim64/fp16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.000621/0.000998/0.001164/0.001183`，维护测试热 cache
  `1 passed in 3.08s`。默认 head dim 32 配置对该测试显式 skip；fake nox session
  确认请求 `64/bf16` 后 native 环境为 `32,64` 与 `fp16,bf16`，门禁为 9 项。
- hdim96/fp16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.000984/0.001061/0.001162/0.001060`，维护测试热 cache
  `1 passed in 3.90s`。默认 head dim 32 配置为 `1 passed, 2 skipped`；fake nox
  session 确认请求 96 后 capability 为 `32,96`，且只追加 hdim96 门禁，共 8 项。
- hdim128/fp16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.000882/0.001044/0.001305/0.000846`，维护测试热 cache
  `1 passed in 4.06s`。默认 head dim 32 配置为 `1 passed, 3 skipped`；fake nox
  session 确认请求 128 后 capability 为 `32,128`，且只追加 hdim128 门禁，共 8 项。
- hdim192/fp16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.001075/0.001532/0.001529/0.002105`，维护测试热 cache
  `1 passed in 3.65s`。默认 head dim 32 配置为 `1 passed, 4 skipped`；fake nox
  session 确认请求 192 后 capability 为 `32,192`，且只追加 hdim192 门禁，共 8 项。
- hdim256/fp16 dense forward/backward 独立构建与 NumPy 对拍通过；output/dq/dk/dv
  最大绝对误差分别为 `0.001176/0.001113/0.001430/0.001366`，维护测试热 cache
  `1 passed in 3.69s`。默认 head dim 32 配置为 `1 passed, 5 skipped`；fake nox
  session 确认请求 256 后 capability 为 `32,256`。
- Dropout 门禁使用 `p=0.25`：同 seed 的输出和三组梯度逐元素一致，不重置 seed 的
  下一调用输出变化；所有梯度有限且非零。`return_attn_probs` 只检查 shape，因为官方
  128 对齐工作区的有效序列外区域不保证初始化。
- dropout、varlen backward 与 qkv-packed backward 检查已抽取为 dtype 共用逻辑；
  hdim32/bf16 三项真实 CUDA `3 passed in 56.81s`，抽取后的 fp16 回归
  `3 passed in 4.32s`。bf16 环境同时收集六项时结果为 `3 passed, 3 skipped`，证明
  fp16 入口按 dtype fail closed。bf16 基础 nox 选择 11 项，`all/all` 现精确选择
  21 项且无重复。
- 三条训练变体检查进一步参数化 head dim。hdim64/fp16 为 `3 passed in 8.24s`，
  hdim64/bf16 为 `3 passed in 9.38s`，默认 hdim32/fp16 回归 `3 passed in 3.22s`。
  请求 `64/bf16` 时 nox 精确选择 19 项，`all/all` 现为 27 项且无重复。
- hdim96 的 dropout、varlen backward 与 qkv-packed backward 组合门禁在 fp16 为
  `1 passed in 14.97s`，bf16 为 `1 passed in 14.12s`；每个组合测试内部均执行三条
  训练路径。请求 `96/bf16` 时 nox native 阶段精确选择 15 项，`all/all` 现为
  29 项且无重复。
- hdim128 的同组三条训练路径在 fp16/bf16 同进程真实 CUDA 门禁为
  `2 passed in 3447.28s`；耗时包含 `32,128 × fp16,bf16` capability 组合的官方
  扩展冷构建。请求 `128/bf16` 时 nox native 阶段精确选择 15 项，`all/all` 现为
  31 项且无重复。
- hdim192 的 dropout、varlen backward 与 qkv-packed backward 在修复后热回归中
  fp16 为 `1 passed in 4.32s`，bf16 为 `1 passed in 3.11s`。hdim256 的 varlen 与
  qkv-packed backward 在两种 dtype 均通过；SM89 上带 dropout 的 backward 原先会
  返回全零或 NaN 梯度，现按官方能力边界在 dense、varlen、qkv-packed 三个入口
  forward 前 fail closed，同时保留 no-grad dropout forward。最终 fp16 为
  `1 passed in 3.16s`，bf16 为 `1 passed in 4.13s`。
- 请求 `192/bf16` 或 `256/bf16` 时 nox native 阶段均精确选择 15 项，`all/all`
  现为 35 项且无重复；默认 hdim32/fp16 dropout 回归 `1 passed in 4.04s`。
- 显式 mask 在 official fused API 不受支持，因此 Torch SDPA 的 native-required 门禁
  要求其精确回退 canonical math 路径。bool 下三角 mask 与 float32 additive bias 的
  output、dq/dk/dv 均对独立 NumPy 参考；fp16 为 `1 passed in 17.49s`，bf16 为
  `1 passed in 69.62s`，两种 dtype 的统计均为 `hits=0, misses={"mask": 1}`。
  默认 native 阶段现为 8 项，bf16 基础为 13 项，`all/all` 为 37 项且无重复。
- 修复前 official dense backward 的一阶 dq 非零（绝对值和 `246.6115`），但再次
  `jt.grad` 只警告无梯度并返回全零。Jittor core 现在提供可沿普通算子传播的
  first-order-only Var 标记；FlashAttention dense/varlen backward 输出打标，二阶
  在构图时明确报错，`stop_grad()` 同时清除标记以允许优化器切图。通用核心回归
  `1 passed in 82.84s`，完整 Function 与普通高阶回归 `35 passed, 2 skipped`。
- 真实 CUDA 高阶门禁覆盖 dense、varlen、qkv-packed 的非零一阶梯度与二阶拒绝，
  并执行两步 SGD 证明标记不泄漏到下一训练步，`1 passed in 79.76s`。默认 native
  九项同进程 `9 passed in 114.16s`；nox 默认为 9 项、bf16 基础为 14 项、
  `all/all` 为 38 项且无重复。
- 两步 smoke 同时复现高级优化器会把 fp16 参数/state 提升到 float32。base、SGD、
  Adam、AdamW、RMSprop、Adan 现在统一在 update 前 cast 回目标 dtype；fp16/bf16
  六优化器两步矩阵 `1 passed in 236.28s`，完整独立更新规则 `17 passed in 170.32s`。
- 训练 SDPA benchmark 现在区分 per-call latency 与 queued throughput，并支持可配置
  shape、causal、forced math/flash、生产 default 与预物化 BSHD direct。RTX 4090、
  fp16、`B=4,H=12,D=64` 的 per-call 中位数（毫秒）如下：

  | L | Jittor default / alternate forced | PyTorch flash | 结论 |
  | ---: | ---: | ---: | --- |
  | 128 | math `0.397` / flash `0.634` | `0.444` | short math 更快 |
  | 512 | default math `0.575` / flash `0.670` | `0.464` | 阈值避免更慢 native |
  | 1024 | default flash `0.730` / math `2.014` | `0.613` | native 比 math 快 `2.76x` |

  非 required、无 dropout 的训练现在默认在 `B*H*Lq*Lk < 2^24` 时选择 math；
  `JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES=0` 可禁用，required 门禁不受影响。L512
  默认相对强制 flash 中位数改善约 `14%`；L1024 仍 24/24 native hit。训练路径现在
  也按 publication token 缓存 capability-checked backend，环境 epoch/generation 变化
  仍会失效；L1024 由 `0.834` 降到 `0.730 ms`，约改善 `12.5%`。预物化 BSHD direct
  为 `0.675 ms`，剩余通用 layout/wrapper 成本约 `0.055 ms`。
- causal L1024 的 Jittor default flash/math/PyTorch flash 中位数为
  `0.711/2.082/0.561 ms`。queued L1024 中 Jittor wrapper 为 `0.725 ms`，PyTorch
  flash 为 `1.624 ms`；逐调用延迟仍分别约慢 `1.24x`（L512）与 `1.19x`（L1024），
  causal L1024 约慢 `1.27x`，
  因此性能只接受 workload dispatch 改善，不宣称总体追平。
- backend cache 的 4 项复用/失效回归通过；完整 official-source attention 模块从
  缓存前 `45 passed, 29 skipped in 483.97s` 降至缓存后
  `46 passed, 29 skipped in 9.87s`。optional nox 现为默认 11 项、bf16 16 项、
  `all/all` 40 项且无重复。
- L1024 同轮 per-call 分解显示 Jittor wrapper 的 graph/build 与 sync 中位数分别为
  `0.461/0.249 ms`，PyTorch flash 为 `0.404/0.180 ms`；预物化 BSHD direct 为
  `0.452/0.220 ms`。因此剩余约 `1.19-1.21x` 非 causal 差距同时来自 grad 构图与
  bridge/kernel 执行，不能用机械删除 clone 单点收口。
- 性能原始 31 行 JSONL 未版本化，位于
  `$JITTOR_LAB_ROOT/jittor_transformers_perf/results/flash_training_20260825.jsonl`，
  SHA-256 为 `b3f96180411b69ef8fb090866f70c95d7399dfa8e9c96a81c2c2000608d5cea4`。
- optional 两阶段的 retained nox cache：基础 TorchMetrics/MMCV/MMEngine/PEFT/
  TensorDict/FlashAttention 共 `14 passed, 1 warning in 18.44s`，native 阶段
  `7 passed in 96.18s`。fresh cache 首次 TorchMetrics 仍因主机满核在固定 600 秒内
  未完成 JIT；同 cache 以 1200 秒保护复跑为 `2 passed in 762.48s`。
- `nox -s optional -- tests/compat/torch/test_mmcv_compat.py`：依赖预检通过，
  `3 passed in 550.05s`，session 成功完成冷 cache 编排。
- 布局检查通过；`tests/structure`：`218 passed`；`noxfile.py` Ruff 检查通过。
- 从干净 `HEAD` 与叠加本任务三个运行文件分别构建 797-member wheel；精确比较为
  0 新增、0 删除，只变化 backend、generator header、SDPA installer 与派生 RECORD，
  四项 SHA-256 allowlist 后 wheel 内容门禁通过。默认 wheel 基线另有此前累积漂移，
  未在本任务中混入更新。

前一轮 cold-cache 组合运行在 20 分钟保护下因主机另一个长期满核进程而超时，没有
失败 traceback；相同隔离 cache 补齐编译后，上述模块和组合门禁全部通过。原始 cache
与运行状态均在 `$JITTOR_LAB_ROOT/_state/`，未进入仓库。

## 边界

Native fused 训练结论限定为 RTX 4090、无显式 attention mask。官方 fp16/bf16 的
head dim 32/64/96/128/192/256 均覆盖 dense forward/backward；hdim32/64 的两种
dtype 还覆盖 varlen/qkv-packed 一阶 backward 与 `p=0.25` dropout，hdim96/128/192
的两种 dtype 由组合门禁覆盖相同三条训练路径。hdim256 的两种 dtype 覆盖无 dropout
的 varlen/qkv-packed backward；官方仅在 SM80/SM90 支持大于 192 维的 dropout
backward，SM89 现在显式拒绝该组合而不再产生错误梯度。显式 mask 已验证正确回退
math 路径；二阶梯度已明确 fail closed，并不代表支持数值二阶。alibi、softcap、
逐调用热态性能和完整 Transformer 性能尚未由本报告宣称通过。NPU/ROCm
也未因本次 CUDA 结果获得任何通过结论。
