# 尺寸工厂与标量 normal 的叶张量身份

- Status: 限定 CPU / 真实单 CUDA 的下列工厂契约通过；无速度或未测后端声明。
- Date: 2026-10-15。
- Baseline: 同步远端 `14aeca5df0bf841e99d6241e06b076e34ef08c05`；Tensor 随机填充整合后为 `74e39acd1fd4b211c34cc5460e623f4499760f6e`。
- Code: `d51c2a48cc4ea2367e41d0dca84e2051795eefb6`。实际验证时源码尚未提交，外置 source freeze 记录生产与测试 SHA256；提交后逐文件核对相等，不重标为 clean-source 测量。
- Owner: Torch factory / autograd maintainers。
- Review when: factory dtype/device 包装、叶注册、native detach/backward 或 Generator sampler owner 变化。

## 真实问题与修复

独立 binary Torch `2.11.0+cu128` 对 11 种工厂、FP32/FP64 和适用的默认/显式 Generator 共 32 个组合，均返回 `is_leaf=True`、`grad_fn=None`，并能在 backward 后取得完整 `.grad`。候选基线有九个不符组合：FP32 `zeros` / `ones` / `full`，两种 dtype 的 `linspace`，以及两种 dtype、两条 Generator 路径的标量 `normal`。它们暴露内部 broadcast/cast/affine 计算图，`.grad` 为 None。

此前 Tensor 随机填充集成的短 probe 用 `torch.ones(..., requires_grad=True)` 作为写入目标，却没有先检查 `is_leaf`。该目标在候选端实际不是叶张量，因此不能用其写入结果证明叶检查失败。补充观察确认：明确用 `torch.tensor` 创建的真实叶张量已被随机填充修复正确拒绝；私有 Generator 未推进的基线证据仍有效。原 probe 和补充观察均保留。

工厂包装器在最终 dtype 转换后、设置 requires-grad 和登记 backward holder 前，使用既有 native `detach` 切断创建过程的内部历史。显式 Generator 路径同样只处理抽样后的输出，不增加随机数、不修改 PRNG 或实现另一套数学。标量 `normal` 的判定在旧 shape 归一化前记录；Tensor mean/std overload 不进入新增的标量处理分支。

`linspace` 还有一层外部 dtype 包装：原来在登记叶 holder 后再 cast，FP64 形成新图节点，同 dtype cast 也可能使返回 holder 与已登记 holder 分离。现在由原 constructor 在 detach/登记前完成 dtype 转换，端点和步数转换保持原 owner。

## 实测范围

维护节点为 `compat/tests/torch/test_size_factory_leaf_contract.py`。同一份源文件在独立 Torch 与候选中执行，不要求不同 PRNG 的跨框架样本逐位相同。

- `empty`、`zeros`、`ones`、`full`、`arange`、`linspace`、`rand`、`randn`、标量 `normal`、浮点输出的 `randint` / `randperm`：FP32/FP64，显式 requires-grad 开关，普通与 `no_grad` 创建。
- 对默认无梯度输出随后启用梯度；检查真正叶身份、无 grad-fn、独立存储、完整梯度与两次 backward 累加。
- 五种显式 Generator 工厂：完整状态重放、私有流推进和全局 CPU/CUDA 流隔离。
- `zeros` / `ones` / `full` 的 FP16/BF16/FP32/FP64 真叶目标：`uniform_` / `normal_` 在抽样前拒绝，目标值和私有/全局状态逐数组不变。

候选新增节点均在 forbid-fallback scope 下执行，检查请求设备与真实 location；CUDA 在 host readback 前使用 driver pointer attribute 检查分配 ordinal。独立 oracle 前后拒绝 shim 和 Jittor import，核对真实 `_C` 扩展及测试文件 SHA。

| 验证 | 结果 | 说明 |
| --- | --- | --- |
| 独立 Torch 短矩阵 | 32/32 叶与梯度通过 | 候选修复前 9 个组合不符，修复后全通过 |
| 最终候选 CPU | 161 passed / 152 CUDA skipped | 新文件及既有 like / tensor / factory fidelity 四文件 |
| 最终候选 CUDA | 318 passed / 0 skipped | 同四文件，要求真实 CUDA 与至少 152 个 accelerator 节点 |
| 独立 Torch CPU | 222 passed / 222 CUDA skipped / 8 native-only deselected | 新工厂文件及 nn.init / Tensor 随机填充两文件 |
| 独立 Torch CUDA | 444 passed / 0 skipped / 8 native-only deselected | 与 CPU 同源同集合 |
| 工厂修复后的既有随机/原位八文件 | CPU 147 passed / 105 CUDA skipped；CUDA 274 passed / 1 双设备 skipped | before-review 源 freeze；后续 review 仅提前 overload 判定，最终四文件及 core 又重验 |
| 最终 CPU core | 322 passed / 42 skipped / 1 existing xfail | native 与 Torch 两模式 exit 0 |

最终四文件 CPU / CUDA 用时分别 5.21 / 12.54 秒。编译缓存、其他进程与规格不同，耗时仅作复现信息，不构成 L4。CPU 为显式隐藏 CUDA；CUDA 分配一个 RTX 4090，driver 580.178.04。候选 GCC 11.5.0 / NVCC 12.2.140、cu122 provider；oracle 为实际 cu128 Torch。所有新测试的输出与梯度在声明设备上执行。

两生产阶段分别封源。Tensor 首批整合在 clean `74e39acd` 上验证；工厂最终阶段记录 dirty-source 三个生产文件与四个测试/工具文件 SHA，后续 Git blobs 与其逐字节相等。独立 review 促使提前记录 overload 类型后，CPU 四文件、CUDA 四文件及 core 均按新生产 SHA 重验，较早日志不改绑。

## 保留的失败与边界

首次 Tensor CUDA 集成的 274 个断言均通过，但门禁错误地要求至少 200 个 accelerator 用例，而实际选中 122 个，故进程 exit 1。保留原脚本和日志，按实际集合设为 122 后原样重跑 exit 0；没有删除用例、修改容差或把失败运行记成通过。四文件新工厂门禁独立要求其实际 152 个 CUDA 节点。

首次完整结构检查为 1621 passed / 6 skipped / 1 failed：旧 `test_factory_install_owners` 假对象要求 constructor 返回后再 cast。更新为验证 dtype 传入 constructor 且同一登记 holder 原样返回；显式 Generator 分支仍不得触碰默认流。原失败日志保留，不把测试期望改回会丢梯度的生产路径。

Tensor mean/std `normal` 的旧位置参数归一化和自有 autograd 契约未在此授予支持；`bernoulli` / `multinomial`、一般 Tensor 变换、所有 out/layout 选项和未测后端也不由本报告覆盖。修复的是创建结果的 Torch 叶语义，不宣称随机字节与 Torch 跨框架互通或完整 Generator ABI。

测试源码分别保存在主树维护节点和仓外字节副本。oracle 使用独立完整官方 Torch wheel 的私有 provider，未修补或改写此前缺文件的共享 Torch 安装；本轮入口验证的是实际二进制身份与契约，不回填历史 oracle 的完整安装来源。

未版本化原始记录位于 `$JITTOR_LAB_ROOT/_state/tensor-main-integration-14a/`：`factory-matrix-{before,oracle,final}.log`、`factory-final-source-freeze.json`、`factory-final-{cpu,cuda}.log`、`oracle-{cpu,cuda}-{before,after}.json`、`oracle-{cpu,cuda}.log`、`core-final.log`，以及分开保留的 Tensor 首批 source freeze、初次配置失败和正确门禁日志。

复现前固定解释器、设备、编译器和独立 unit cache；先 CPU 后 CUDA，不与 benchmark 共用缓存：

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_size_factory_leaf_contract.py \
  compat/tests/torch/test_like_factory_leaf_contract.py \
  compat/tests/torch/test_tensor_factory_leaf.py \
  compat/tests/torch/test_torch_factory_fidelity.py
```

真实 CUDA 对同四文件另起进程，并设置独立 unit cache：

```bash
CUDA_VISIBLE_DEVICES=<allocated-gpu> JT_USE_CUDA=1 JT_BUILD_NVCC_PATH=<nvcc> \
  JITTOR_TEST_REQUIRE_CUDA=1 JITTOR_TEST_REQUIRE_EXECUTION=1 \
  JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=152 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  <candidate-python> -m pytest -q \
  compat/tests/torch/test_size_factory_leaf_contract.py \
  compat/tests/torch/test_like_factory_leaf_contract.py \
  compat/tests/torch/test_tensor_factory_leaf.py \
  compat/tests/torch/test_torch_factory_fidelity.py
```

独立 Torch 使用相同测试字节及 `_rng_init_support.py`，放在仓外目录、清理 shim/PYTHONPATH，运行前后断言真实 binary 和无 Jittor import。用 `--confcutdir=<copied-suite>` 避免加载候选专用 pytest 策略；只有 native-only observer 被排除，CPU 的 CUDA 分支诚实跳过。

最终安装归属定向检查 **5 passed**；更新后的完整 `tests/structure + compat/tests/structure` 为 **1622 passed / 6 skipped**，230.84 秒。六项分别为两项 accelerator、两项 declared，以及当前结构验证解释器未安装 native distribution 的两项解析检查；没有把这些跳过授予安装验收。布局、MANIFEST、空白检查通过，新测试 Ruff 无诊断；修改的工厂文件原三条诊断与基线相同。
