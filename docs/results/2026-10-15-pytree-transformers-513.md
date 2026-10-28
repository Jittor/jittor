# 自定义 pytree 节点与 Transformers 5.13.0

- 状态：以下限定契约的 CPU/CUDA 正确性已验证。
- 日期：2026-10-15
- 基线提交：`c92ae5cecca90801804e677b7200b2ef3b30b27c`（最终 GPT2 重放所用的干净源码）。
- Owner：Torch 兼容维护者。
- 复查条件：pytree 安装、自定义节点遍历、adapter 版本守卫、
  共享 ecosystem/训练 runner 或 Transformers 输出结构变化。

## 故障与归属

真实的 Transformers 5.13.0 发行版最初停在 adapter 的
版本守卫上。临时放行后，import 其 `ModelOutput` 因缺少
`torch.utils._pytree._dict_flatten_with_keys` 而失败。另一个自定义 `dict` 子类
还表明，既有 `register_pytree_node` 返回时没有注册任何东西：
flatten/unflatten 重建的是 `dict`，而不是注册的类。独立的 PyTorch 2.13.0
能重建该自定义类，并提供真实的 mapping/sequence key 条目。

这是共享 Torch API 契约。修复位于
`compat/torch/installers/utilities.py`：保存已注册回调；在
flatten、unflatten、mapping 与带 key 遍历中使用它们；保留已注册子类；
拒绝重复注册、缺失 keyed 回调与畸形的叶子计数。
内建 `OrderedDict` 与 `defaultdict` 保持其类型与 default factory。
公共与 legacy 注册使用同一 registry。TreeSpec JSON
序列化不在本次验证范围内。

adapter 现在接受精确版本 **5.13.0**，与此前已接受的
4.56.2 和 5.5.3 并列。host 契约仍拒绝未知版本；未启用宽泛的 5.x
范围。没有针对 ModelOutput 的专用补丁。

## 验证

oracle 是独立解释器中真实的 **PyTorch 2.13.0+cu130**。候选
执行使用本 checkout 与 Jittor Torch 前端，其报告的 Torch
API 版本仍为 **2.11.0**。两个框架版本都未被改写。双方
加载相同的真实 Transformers **5.13.0** 文件。Python 为 3.11；
NumPy 为 1.26.4。候选 CUDA 使用既有 CUDA 12.2 工具链；oracle
使用其真实的 CUDA 13.0 wheel 库。CPU 运行显式隐藏 CUDA，并把
Jittor 的 backend/use_cuda 设为 CPU/zero。

| 检查 | 结果 |
| --- | --- |
| 新的 custom-pytree 测试，真实 PyTorch CPU / CUDA | 9 / 9 passed |
| 候选 CPU custom-pytree 加既有 tree-map/owner 邻接 | 17 passed |
| 候选 CUDA custom-pytree 测试，含张量输入梯度 | 9 passed，无 skip |
| Adapter host 版本/probe 契约 | 7 passed |
| CPU core 套件 | 322 passed，42 skipped，1 个预期失败 |
| Torch 模式下完整 `tests/structure` | 1488 passed，4 skipped，无失败 |
| 仓库布局、新测试 Ruff、未新增实现 lint 诊断、diff 空白 | 通过 |

pytree 测试覆盖自定义 dict 与非 dict 节点、嵌套 key 路径与 `get`、
注册往返/map、真实张量输入梯度、重复注册、
缺失路径回调、叶子过少/过多、未注册子类行为、
legacy 注册，以及两个内建 mapping 子类。同一个新测试
文件也针对独立 PyTorch oracle 运行。

最终 GPT2 重放使用既有的 tiny GPT2 builder（两层、宽度 64、
两个 head、词表 128、长度 8、batch 2、dropout 0）。一个小 wrapper 使用
公共 `get_input_embeddings()` 入口加 `0.05 * embedding_noise`，因此
既有浮点输入梯度，也有完整的 28 参数梯度
集合。它为三个 SGD 步（`lr=0.001`、seed 7）各保存输出、loss、
每份参数梯度、输入梯度与更新后参数。没有参数被
冻结。两个设备/runtime 加载同一份初始状态产物：
`a82b8e3e9c20f57ed6da16920a87042ec0a2626f651a9913412ea44db96ea639`。

阈值在干净重放之前冻结：最大绝对误差除以
`max(max(abs(reference)), 1e-8)`，输出/loss/更新后参数限值 0.002，
参数/输入梯度限值 0.01。CUDA 沿用这些 CPU 限值。每个
数组还要求键相等、精确 shape 与有限值。TF32 关闭，
matmul 精度为 `highest`，原生 CUDA kernel 数学为 `strict`，runtime
线程数为 1。候选 fallback 策略为 `error`，两个设备的观测
fallback 计数均为零。

| 干净源码重放 | 比较的数组 | 最大归一化误差 | 结果 |
| --- | ---: | ---: | --- |
| CPU | 177 | 0.00000346416447 | 全部通过 |
| CUDA | 177 | 0.00090132669048 | 全部通过 |

CUDA 最大值是第零步 loss（参考 0.00048499566037，绝对误差
0.00000043713953）。次大的归一化误差是第二步 `c_attn.weight`
梯度的 0.00060066114。这些是最终重放测量；
更早的临时 dirty-source 输出单独保留，不用作这些数值的
替代。该证据支持受测的 tiny GPT2/ModelOutput
训练路径，不覆盖每个 Transformers 模型、vLLM engine、Omni 模型，也不构成速度
结论。GPU 共享与小工作负载使性能结论不成立。

## 复现与检查

提交的 API 测试可原样针对各解释器运行：

```bash
CUDA_VISIBLE_DEVICES='' JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  JT_BACKEND=cpu JT_USE_CUDA=0 use_cuda=0 \
  <jittor-python> -m pytest -q compat/tests/torch/test_pytree_custom_nodes.py
CUDA_VISIBLE_DEVICES=<gpu> JITTOR_TEST_DEVICE=cuda \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python JT_BACKEND=cuda JT_USE_CUDA=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_pytree_custom_nodes.py
env -u JITTOR_TORCH_SHIM -u PYTHONPATH CUDA_VISIBLE_DEVICES=<gpu> \
  JITTOR_TEST_DEVICE=cuda <real-torch-python> \
  compat/tests/torch/test_pytree_custom_nodes.py -v
# For the CPU oracle, hide CUDA and set JITTOR_TEST_DEVICE=cpu.
```

未版本化原始产物位于
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`：

- `logs/pytree-*-before.log` 保留原始独立 API 失败。
- `logs/pytree-*.log` 与 `logs/adapter-513-*.log` 包含定向/core/
  structure/布局结果。
- `tf513_training_check.py` 是围绕版本化 `training_probe.py` 的
  小型公共 embedding wrapper；`replay_tf513_clean.py` 启动四个隔离
  运行并检查固定阈值。它们是外部实验 fixture，
  不是新登记的 ecosystem 覆盖。
- `tf513-clean-c92ae5cec/` 包含两个 runtime 的 CPU/CUDA NPZ、metadata 与日志，
  以及逐数组比较 JSON 及其 manifest。每条 metadata 记录绑定
  干净源码 SHA、probe SHA、初始权重、依赖来源/版本、
  精度设置、实际后端与 fallback 计数。
- `pytree-final-manifest.json` 绑定 **31 个产物与六个源码文件**，含
  两个外部 fixture 脚本、其初始权重与全部最终输出。
  SHA256：`e48eb9d3e5bcb98427ec80ed535a229bd3fdbb1f002e1e8742ee4ac1222be4af`。
- 内层 clean-replay manifest 的 SHA256 为
  `1e706c2d679231ef2f96f27348ea29595a788562e848a89ef9a71faea74b4001`。

要用既有版本化 runner 复现该 wrapper，请注册一个外部
case：其 builder 用提供的 Torch 命名空间调用 `CASES` 中已注册的
`transformers_gpt2` builder，按
上文包裹其模型，加入输入规格 `embedding_noise=('float32', (2, 8, 64), None)`，然后
调用 `training_probe.main()`。用以下命令调用保留的外部 wrapper：

```bash
<python> <external-wrapper> --root <checkout> --case <external-case> \
  --runtime <torch-or-jittor> --device <cpu-or-cuda> --steps 3 --seed 7 \
  --weights <same-initial-state.npz> --output <outside-checkout.npz>
```

保留的 wrapper 与 replay 脚本提供本报告所用的精确
fixture 与环境。

在 `a5932248fdb06fb52293f5ddd8a357f4920195ff` 上整合时，独立
重哈希全部 31 个产物与六个匹配源码文件，并在每个设备上
重算全部 177 个数组比较。上述测量重放基线
不变。整合后的树通过 CPU core（322 passed，42 skipped，1 个
既有 xfail）、CPU owner/adapter 检查（24 passed）、CUDA owner 检查
（17 passed，fallback 增量零）以及合并的 structure/tools 检查
（1686 passed，六个环境/能力 skip）。既有实现 lint
诊断与父提交比较；未引入新诊断。
未版本化整合日志与独立数值审计位于
`$JITTOR_LAB_ROOT/_state/pytree-scm-integration/`。
