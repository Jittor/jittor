# 可选依赖 fail-closed CUDA 门禁

- Status: Selected optional packages and native FlashAttention training accepted on real CUDA
- Last reviewed: 2026-08-25
- Commits: `566eae8e`, `2cf096d5`, `c2e340f8`, `90e00edd`, `9e69fa23`, `19820174`, `50fc95d5`, `a13cb06e`, `c8c43cf6`, `d500dc77`, `76b8a5a0`, `24cf00eb`, `95cd6f6c`, `c3e65b1d`, `fe97085a`, `1cd76dd9`, `f3df3274`, `797a6a97`
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
head dim 32/64/96/128/192/256 均覆盖 dense forward/backward；其中只有 hdim32/fp16
和 hdim32/bf16 还覆盖 varlen/qkv-packed 一阶 backward 与 `p=0.25` dropout。
hdim64/96/128/192/256 dropout/varlen/packed、alibi、softcap、显式 mask、
二阶梯度、稳定热态性能和完整 Transformer 性能尚未由本报告宣称通过。NPU/ROCm
也未因本次 CUDA 结果获得任何通过结论。
