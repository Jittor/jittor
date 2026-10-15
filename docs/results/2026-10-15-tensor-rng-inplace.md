# Tensor 随机原位填充的 Generator 与 autograd

- Status: 限定 CPU / 真实单 CUDA 的两个 Tensor 方法通过；结构门禁详见下表。
- Date: 2026-10-15。
- Baseline: 同步 `origin/2.0-refactor` 的 `97d7d478a48ecc80f13f2858f0bc8fa78e377480`，保留已提交的 nn.init 修复与历史结果后，整合为 `19655e404a89b08f61d4585de6558f1234c30149`。
- Code: `11430ebac72bc214fb2562158974c96666352a5c`（生产语义）；`6afbca891d00f4fda560553086a12c3ff883d3b5`（测试工具提取）。
- Owner: Torch Tensor / native RNG maintainers。
- Review when: Generator factory、随机初始化、原位 storage 或 autograd graph owner 变化。

## 实际断点

在独立 binary Torch `2.11.0+cu128` CPU 上先执行最小 probe，
`Tensor.uniform_` / `normal_` 推进私有 Generator、保持全局流、完整状态恢复后重放，
并在 grad 模式中拒绝修改 requires-grad 的叶 tensor。
冻结候选基线上的同样调用则没有推进私有流、改变全局流、无法重放，
而且接受修改叶 tensor。另一个真实最小 probe 证明，
覆盖非叶 tensor 后，真 Torch 给旧输入完整零梯度并保留非叶身份；
候选将其重新绑定为叶 tensor，旧输入的 grad 为 None。
四份实际 oracle/candidate 基线日志保留，不替换为修复后的记录。

## 实现与边界

nn.init 已验证的 validation / dtype / device / draw / affine 逻辑提取成两个私有
values helper。两条公共 nn.init 入口继续自己的 no-grad 初始化与赋值；
Tensor 方法复用 helper，保留自己的原位语义，不调用另一条忽略 Generator 的消费者。
私有流仍由既有 factory 与 native CPU engine / CUDA Philox 执行；没有增加 PRNG。

Tensor 在任何抽样或计数预留前拒绝 grad 模式中的叶 tensor 与叶 tensor 的 view。
no_grad 允许初始化 trainable leaf，保持其身份与训练开关。
覆盖非叶 tensor 使用既有 zero_ 同款 false ternary 路径：新值与旧图保持零导数连接，
不会通过 `old * 0` 把 NaN/Inf 污染到新随机值。
写回复用既有 `_ip(..., preserve_storage=True)`，未修改共享 `_ip` 或 native autograd。

这批只声明四种真实浮点 dtype 的两个 Tensor 方法及共享 nn.init 回归。
不授予所有原位算子、其他初始化器、所有 Generator ABI、整数范围精度、
跨 Torch RNG 字节互通或未实测 ROCm/NPU。
首批报告与原始 source/oracle SHA 保持原测量身份，不因本次同步或 helper 抽取重标。

## 实际验证

新测试与独立 Torch 使用同一文件；只排除候选 native observer。
oracle 解释器测试前后拒绝 Jittor import，记录实际 binary extension、解释器、
版本、CUDA 可用设备和测试 SHA256。状态 / 输出 / 梯度均比较完整数组。
不同 PRNG 的跨框架样本不要求逐位相同；参考验证状态与原位行为契约、实际分布参数。

新测试覆盖 float16/bfloat16/float32/float64、全局 CPU/CUDA/Python/NumPy 流隔离、
完整重放与连续填充/工厂混合、默认 CPU Generator 身份、dtype bounds/std 原子拒绝、
错误 generator/device、冻结目标、叶/view 拒绝、非叶零导数（含原 NaN/Inf）及 retained view 写回。
候选所有新节点都在禁止 fallback 的 scope 内执行。
native observer 禁止 NumPy 抽样，检查原 native handle、float64 sampler dtype，
CUDA host readback 前用 driver pointer attribute 核实实际分配 ordinal。

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| 独立 Torch CPU | 41 passed / 41 CUDA skipped / 4 native observer deselected | 含追加 no_grad 非叶检查的完整通用 Tensor 测试 |
| 独立 Torch CUDA | 82 passed / 0 skipped / 4 native observer deselected | 真实单 CUDA |
| 候选 CPU | 145 passed / 103 skipped | 下列八个维护文件 |
| 候选 CUDA | 270 passed / 1 skipped | 同八文件，require CUDA / min execution；仅双设备项跳过 |
| 追加 no_grad 非叶检查 | CPU 2 passed / 2 CUDA skipped；CUDA 4 passed | 未修改生产源码，单独定向执行；old graph / pointer / pending reader |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | canonical native / Torch 两个模式 exit 0 |
| 最终共享测试工具回归 | CPU 74 passed / 74 CUDA skipped；CUDA 148 passed | nn.init + Tensor 两文件；额外 35 项去重/pytest 契约通过 |
| 最终独立 Torch | CPU 70 passed / 70 CUDA skipped；CUDA 140 passed | 同两文件全部通用测试，8 native observer deselected |
| 完整结构 / 布局 | 1622 passed / 6 skipped；布局通过 | clean 6afb 的 tests/structure + compat/tests/structure；本页登记前 |

最初一次扩大回归命令写错一个既有文件名，pytest exit 4、零节点执行；
该日志保留，随后按实际八个文件重新运行，不把收集失败算成通过。
初次完整结构还有两项 pytest 收集契约失败：跨 test 模块导入 fixture，且该 fixture 别名不被静态规则识别。
改为本地定义后又被去重检查拒绝。最终将公共逻辑放入 `_rng_init_support.py`，
两个测试只保留显式薄 fixture，不增加 allowlist。全部 test_* 函数的 AST 与先前实测逐个相等；
最终两文件仍在独立 Torch / 候选 CPU / 真 CUDA 上全部重验，生产两文件未变化。
这些原始失败均保留；另两项首批报告 docs/MANIFEST 登记失败在整合主线 9ba 后消除。
新测试与共享工具完整通过 Ruff；两个修改生产文件的既有 21 条诊断与基线逐项一致，无新增。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_tensor_rng_initializers.py \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_torch_init_package.py \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_default_generator_contract.py \
  compat/tests/torch/test_tensor_inplace_grad_views.py \
  compat/tests/torch/test_frozen_tensor_in_place.py \
  compat/tests/torch/test_tensor_inplace_storage.py
```

执行前仍须按环境手册配置独立 HOME/JITTOR_HOME/TMPDIR、明确 CPU 隐藏 CUDA 或
真实 CUDA 工具链与 strict fallback。缓存沿用同一 R33 消费者任务的独立 unit 缓存；
与 benchmark 分开，CPU→CUDA 顺序执行。GPU 同时存在其他进程，只授予正确性。

未版本化证据目录：`$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/tensor-followup/`。
`source-freeze.json` 对应八文件完整回归的测试源 SHA `7d8eec1939a181eafd20677b0af7f64bd66cade57df2eb1c12432cc7e4347be2`。
随后只在同一测试文件末尾追加四个 no_grad 非叶节点；两个生产文件逐字节不变，
原测试文件是追加后文件的精确字节前缀，原始副本与 SHA 仍保留。
`no-grad-recheck/` 记录追加后的独立 oracle 全部通用测试和候选的四个新增定向节点；
不重跑已过八文件或把旧日志改绑到新测试 SHA。
`source-freeze-extended.json` 分开记录两次测试源码身份。

修复测试收集/去重契约后，`common-helper-recheck/` 保存最终两文件 oracle 与候选 CUDA，
`common-helper-cpu.log` 保存候选 CPU 与 35 项定向结构门禁；
`source-freeze-common-helper.json` 绑定最终五个代码/测试/辅助文件，核实生产文件仍等于 114。
此时仅测试组织变化，所有 test_* 函数 AST 未变；也保留重验后的实际新 SHA，不回填旧记录。
最终结构在 clean 6afb、整合主线 9ba 的树上执行，旧八文件数学测量继续保留 19655 基线。

source freeze、baseline/oracle/final 日志、oracle metadata 与最终 manifest 绑定
实际测量 source / 原始文件哈希。测量时三个文件尚未提交，随后逐文件核对 Git blobs；
不将历史 dirty-source 结果改记为 clean-source。
中央结果索引与 MANIFEST 由集成维护者登记，最终文档树门禁需在登记后复验。

额外 CPU 最小审计确认：既有 nn.init._assign 对 requires_grad_(False) 的可逆冻结
仍可能解冻；首批测试只覆盖构造时停梯度态，不能扩大为所有冻结状态通过。
该独立旧缺陷另行修复，本批 Tensor 方法的可逆冻结 CPU 对照已保持 False；
不将其写成尚未完成的 nn.init CPU/CUDA 验收。
