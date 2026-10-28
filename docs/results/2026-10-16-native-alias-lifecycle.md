# 原生别名历史与逻辑 AD 输入

- 状态：仅验证了所列中间契约；KI-COMPAT-004 与 R33 仍未关闭。
- 复查日期：2026-10-16。
- 维护者：原生 autograd / Torch 兼容性维护者。
- 测量：`4f95f71195d60b9c60ca3c1830b823dec552eb3a` 加上 `layout-final-source-v1.json` 中 28 个精确的自有文件 SHA256 值，运行在远端 `00ceb26fd395707fb33649cc43709af045110a9a` 上。
- 整合源码：`0458c28b41643e4b2975845c0a0b366c213e2f8d`，与被测文件逐字节绑定；更早的 e7af/2f/bc/20 文件测量保留其原有身份。
- 复查条件：原生视图历史、边快照、存储重新绑定、布局传播、标志位或叶子梯度布局发生变化。

## 结果与实现

由 no-grad 创建的原生别名现在把可见可训练性、写入版本策略和已记录的反向边分开保存。
源变为可训练时可以改变别名的可见 `requires_grad`，但不会为其臆造累加器。旧的已记录
消费者保留其输入边快照。在冻结写入期间被禁用的源边，在源重新启用后仍保持断开。空写入
与自复制会推进写入历史。公开的历史查询会消耗待处理历史；元数据转换、物理派发观察和
`requires_grad_` setter 不会。

原生别名策略只拥有小型策略对象，绝不拥有 Var/Op/Python Tensor 所有者。其源链和自有/
继承标志建模可见可训练性。写入版本在 detach 间共享，但 data 别名之间各自独立。创建
epoch 是未记录视图的创建版本，或已记录视图最后一次被消耗的历史版本。只有真实的已记录
原生视图生产者才授予已记录视图历史；过期的 no-grad 创建别名无法获得它。源边资格描述
实际的重定基边，仅 epoch 变化不授予任何边。原生永久 stop-grad 得以保留。生命周期测试
验证源 Var 与新增的前向依赖会被释放。

来自 [819182d1b](2026-10-16-recorded-grad-fn-freeze.md) 的原始已记录输入循环逐字节
相同。新增的已记录视图回退路径保留永久 stop 和创建时禁用两项检查。原生目标/遍历资格
遵循实际已记录的消费者，Torch 前端查询原生累加器资格，而非从 Python 张量注册表猜测。

`.data` 此前会用一个已停止的克隆替换持有者的 Var，从而丢失旧的叶子梯度和非叶生产者。
修复会先结束旧读取者，通过既有 detached 存储所有者固定右侧分配，再通过既有的分配器
操作改变同一 Var 的分配/形状/dtype/位置。其生产者、可训练性和输入边快照得以保留。旧的
显式视图保留其旧分配和守卫来源。只有真正被记录的图才保留旧的梯度形状契约；既有的原生
归约会校验兼容的梯度并拒绝不兼容的形状。

`autograd.grad` 返回已记录图的梯度。`.backward` 则另行使用既有的分配/视图/复制原语
施加初始叶子梯度布局契约，包括真实的稠密置换存储和高阶微分。全局原生转置策略保持不变。
Torch 风格 SGD 的既有原生 step 在 no-grad 下执行；closure 与原生/loss 风格优化保持其
原有行为。

布局优化还存在第二个、可独立复现的缺陷：在内部转置或解包已保存操作数会改变它们的
逻辑身份。在 `.data` 替换之后，乘法和依赖输入的一元导数会读到旧分配，或绕过真实的
非叶/视图目标。BinaryOp 与 UnaryOp 现已保留其原始逻辑输入和既有导数公式。显式的
停止依赖用既有内核计算优化后的前向；原算子为这一真实结果建立别名并拥有 AD。已保存
输出的导数仍使用真实保存的输出。no-grad/推理快路径保持不变。这并未全局禁用布局传播、
未保存过期的输入值、未引入参数 ID 表，也未实现替代性的梯度数学。

## 验证

原始证据未纳入版本控制，位于 `$JITTOR_LAB_ROOT/_state/native-alias-lifecycle/20261016/`。
每个命令 JSON 绑定命令、实际环境、返回码、实测 HEAD 和精确的自有文件哈希。CPU 隐藏
CUDA，CUDA 运行要求真实执行，首次 JIT 串行，单测与 benchmark 缓存分离；JIT 文件未被
复制。

| 检查 | CPU | CUDA 可见选择 |
| --- | --- | --- |
| 最终十一个兼容性文件 | 524 passed / 400 个加速器 skip | 924 passed / 0 skipped |
| 独立的真实 Torch 契约副本 | 399 passed / 400 个加速器 skip | 799 passed / 0 skipped |
| 原生邻接，相同的 22 个生产文件 | 100 passed / 15 skips | 89 passed / 1 个保留的 expand 显存失败 / 0 skips |
| Channels-last 布局/梯度/训练邻接 | 未声明 | 3 passed / 0 skips；禁止回退 |
| 规范合并核心 | 322 passed / 42 skips / 1 个既有 xfail | 未声明 |
| 完整结构 | 1630 passed / 4 skips | 仅主机 |

CUDA 可见的参数化选择包含 CPU 行，因此这些总计并不意味每一行都执行了 CUDA 算术。
新的原生/别名语义测试与专用探针会同步张量，并在 `forbid_backend_fallbacks` 下检查
物理位置；原有的 32 用例 CUDA 探针还检查驱动指针属性。这不声明围绕每个不相关遗留测试
的全局回退守卫。三个 channels-last 节点还在整调用级回退守卫内运行。这些守卫在同步后
正常返回；原始 JSON 不含捕获的数值回退计数器字段，也未追溯添加。

三个 oracle 契约/支持文件与维护版源码逐字节相同（`final-oracle-contract-byte-binding-v1.json`）。
独立 oracle 是官方 Torch 2.11.0+cu128：wheel SHA256
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`，实际 `_C`
SHA256 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
候选项使用其独立的 CUDA 12.2 provider。oracle 命令禁用字节码写入，且不带 shim 运行。
真实 Torch 会发出其常规的 `backward(create_graph=True)` 引用环警告；相应用例会清除
累积的梯度。

上级维护者在旧的 4f95 生产字节上独立封存了一个真实的 48 用例 saved-input/saved-output
矩阵：每设备 12 个数值失败，恰好是叶子/非叶/构造器变异后接乘法、sin、cos 或 log。其余
36 行本已与 Torch 一致。修复后的实现在两种设备上与全部 48 行一致。独立的有限性/形状/
数值检查在每设备覆盖 280 个数组：最大绝对误差在 CPU 上为 2.384185791015625e-7，在 CUDA
上为 1.1920928955078125e-7，处于冻结的 FP32 rtol=2e-6 / atol=2e-7 预算内。CPU 原始记录
保留其 26 文件身份，CUDA 保留其最终 28 文件身份；两者都有相同的 22 个生产文件。该矩阵
区分依赖输入的导数与 exp/sqrt/tanh 的保存输出导数，并保留显式的用户复制/视图分配。另一
个惰性的 24 用例矩阵在两种设备上均一致，因此物化一个观察者不会掩盖缺失的目标边。最终
维护版测试包含惰性消费者以及构造器故意报错后的 AD 恢复。

新的秩/布局 48 用例矩阵在 CPU 与 CUDA 上与真实 Torch 一致。它包含非均匀的旧梯度、秩
收缩/扩张、真实的物理转置布局和广播变化。例如，元素数相同的 2x2→4 允许旧形状的
`autograd.grad` 结果，而 `.backward` 确实会拒绝不兼容的复制形状；一个兼容物理跨步的
用例有它自己独立观察到的契约。这些结果逐项测试，而不是用一刀切的 reshape 行为替代。

最终 original32 用例复跑：每设备 32 行与 64 次物理观察，零差异；CUDA 驱动属性在请求的
设备上识别出每个被观察的指针。更早的 20 文件源码原始证据保持独立：extended124、
frozen-write6、empty/self-copy6、observation-order48、data/detach16、rebind4、
stale-escape4，加上 data-orphan guards8/device，均与真实 Torch 一致。维护版最终别名套件
保留这些边界和原有的 constructor41/device 用例。比较 JSON 保留实际错误文本；比较的是
拒绝是否发生，oracle 侧独有的整数版本值保留在原始文件中，而不声明候选项具备版本 API。

相对成功的 26 文件阶段，最终 28 文件冻结只改动测试。
`production-source-reuse-binding-v1.json` 证明全部 22 个生产文件哈希一致，因此上面的原生
邻接与生命周期结果保留其真实的 26 文件源码身份。完整兼容性/核心/结构以及修正后的
channels-last 门禁在最终 28 文件源码上测量。Ruff 比较未发现新诊断；已有的方法/优化器
导入诊断保持不变。仓库布局检查通过。

## 复现

使用分离的 CPU/CUDA 单测 home、用于精确契约副本的真实 Torch 环境，以及命令 JSON 文件
中记录的后端/工具链设置。`logs/compat-layout-final-{cpu,cuda}-v2.command.json` 列出完整
的十一文件选择；`logs/native-layout-full-{cpu,cuda}-v1.command.json` 列出原生邻接。维护
入口包括：

```bash
<Jittor Python> tools/run_test_suite.py --tier core --jobs 0 --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <Jittor Python> -m pytest -q tests/structure compat/tests/structure
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <Jittor Python> -m pytest -q compat/tests/torch/test_alias_lifecycle.py compat/tests/torch/test_tensor_constructor_alias.py
```

对 CUDA，同时设置 `JITTOR_TEST_DEVICE=cuda` 和 `JITTOR_TEST_DEVICES=cuda`，用
`JITTOR_TEST_REQUIRE_CUDA=1` / `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1` 要求真实执行，
启用真实 provider，并使用 `JT_CUDA_KERNEL_MATH=strict`。CPU 显式将
`CUDA_VISIBLE_DEVICES` 置空并禁用 CUDA。上述契约选择是完整十一文件门禁的一部分；它不
替代其余邻接。

## 保留的失败与剩余范围

默认公开转置布局仍有差异：原生 `transpose_storage_view=0` 会物化，而 Torch 暴露视图。
为叶子梯度做的本地物理视图分配并不能关闭默认转置或完整视图/C 扩展 ABI 兼容性。

`test_expand_does_not_materialize_when_it_is_consumed` 保留其原有的 16 MiB 预算，并在
CUDA 上因 68,157,440 字节的分配增长而失败。一个带独立缓存的干净 e7af 全新基线复现了
同样的结果。它仍计入上面原生邻接总计中的失败。

第一次 channels-last 调用在数值上通过了它的三个 CUDA-only 测试，但以 1 退出，因为旧的
类名未被识别为加速器执行。将这个实为 CUDA-only 的类重命名为 `TestChannelsLastCUDA`，
并更新两个既有的结构选择器，保留了全部三个节点、预算和至少执行一个的规则。最终带守卫
的调用以 0 退出。原始的 exit-1 日志予以保留，且不计为成功门禁。

更早保留的失败包括一个既有的重复随机状态 fixture 契约、两个新用户可见形状检查的精确
台账计数，以及被拒绝的中间布局实现尝试。最终 fixture 使用既有辅助函数，台账反映实际
检查，且未新增任何允许列表或测试策略豁免。原始日志保留其源码身份。

本结论仅验证所述的别名/存储/AD 契约。它不关闭 R33 的维护者 PR 处置，不证明完整的默认
视图/ABI 行为，也不声明未测试的加速器或下游库。不可变的原始清单是
`final-layout-evidence-seal-v1.json`；`final-layout-ready-source-binding-v1.json` 将被测
字节绑定到干净的整合提交，`final-layout-ready-source-v1.tar` 保留其完整源码树。


## 整合复核

干净的 `66db6e44af90718fb732faab10c2f832854b9322` 将这同样的 22 个生产文件与已审阅的
AA9 attention 和 Diffusers 来源门禁合并。独立的字节比较保留了全部 22 个别名所有者和
全部四个 AA9 生产所有者。核心通过 322 / 42 skipped / 一个既有 xfail；完整结构通过
1628 / 六个 skipped。十一文件兼容性选择在 CPU 通过 524 / 400 个加速器 skip，在 CUDA
可见下通过 932 / 零 skip，包含八个新的 BF16 attention 测试。原生别名/attention 在 CPU
通过 38；CUDA 可见的原生别名/attention/完整 channels-last 选择通过 49 / 零 skip，并包含
显式的 CPU 数学检查。一个独立的整调用回退守卫让全部十一个 channels-last 节点通过，
实际计数器保持为零不变。主机 source/nightly 测试通过 182。单测缓存与 benchmark 缓存
分离；首次编译在本地执行，未复制 JIT 产物。

未纳入版本控制的整合命令、原始日志、实际在线 CUDA 环境/库观察以及完整的 2923 文件 Git
源码归档位于 `$JITTOR_LAB_ROOT/_state/refactor-remaining-main-00/20261016/`。
这些整合结果保留源码 66db；它们不会给更早的 4f95/0458 证据重新贴标，不修复保留的
expand 显存失败，也不关闭默认转置/视图与完整 ABI 的工作。

## 低精度布局 dtype 后续

由原生/autograd 维护者于 2026-10-16 复查。源码绑定到
`12a9a143d323b41115729d1e8aa1e79028627161`；在标量提升、物理视图、布局传播或二元/一元
梯度存储发生变化后需复查。

在 c470 上，原生 `(root * 3)._storage_permute((1, 0)).sum()` 在启用布局传播时，对惰性和
物化视图都给出错误的 FP16/BF16 梯度。解析梯度恒为 3。FP16 则交替出现 0/2.125；BF16
交替出现 0/3。关闭该优化可恢复全部四种情况；相应的 FP32/FP64 对照通过。独立的真实
Torch 2.11 CPU 检查确认了全部八个 dtype/物化组合的梯度。

一次内部布局转置丢失了操作数的弱标量提升标记。其算术可能分配 FP32，而逻辑梯度却把这些
字节当作 FP16/BF16 暴露。该辅助函数现在会保留该标记。第二个非常量输入的探针暴露了
公开且显式标记的转置在相反方向上的不匹配：逻辑操作数与稠密生产者可以合法地携带不同的
标量元数据。这些标记不同时，该变换现在使用普通内核。二元/一元存储共享还额外要求相同
的 dtype。物理布局优化和逻辑 autograd 所有者继续使用。

新的 `tests/ops/test_storage_layout_dtype.py` 包含 48 个原生用例，覆盖四种浮点 dtype、
惰性/物化视图、梯度/no-grad 路径、广播标量以及经两条转置路径的非常量输入。其最后 16 个
用例分别比较启用/禁用优化、精确前向值和解析梯度；公开转置还保留其普通的 FP32 提升。

| 最终验证 | 结果 |
| --- | --- |
| 原生 dtype、别名、attention 与 channels-last，CPU | 86 passed / 11 个 CUDA-only skips |
| 同样的选择，启用 CUDA | 97 passed / 零 skips |
| 合并核心 | 322 passed / 42 skipped / 一个既有 xfail |
| 完整结构加兼容性结构 | 1628 passed / 六个 skips |
| 布局、生成的 manifest、Ruff 与 diff 检查 | 通过 |

两次完整原生调用都在回退错误守卫下运行，并直接记录前后计数器为 0。新测试会同步并断言
实际的 CPU/设备驻留。CUDA 可见的选择包含既有的显式 CPU 检查；在该调用中全部 48 个新
用例都在 CUDA 上执行。首次编译串行，单测与 benchmark 缓存分离，且没有 JIT 产物被复制
作为输入。

未纳入版本控制的命令、原始失败、独立 Torch 对照、源码字节、已提交源码归档和取证的
原生库保留在 `$JITTOR_LAB_ROOT/_state/native-layout-dtype-review/20261016/`。
`code-binding-v2.json` 证明四个冻结的被测文件作为源码 12a9 原样提交；这些运行最初标识
为 c470 加上那处显式的四文件改动。`evidence-seal-v1.json` 清点了已关闭的测量产物。
仅改辅助函数的中间版本通过了原有的梯度用例，但未通过后来的公开转置 dtype 不变量；该
失败另行保留。本范围是原生 CPU/CUDA 布局正确性；默认 Torch 视图/ABI 与 PR 处置仍未
关闭。
