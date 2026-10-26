# 最近 30 天 PR 审查与当前分支整合

- Status: 已整合；下列分项验收完成，CPU smoke 按用户要求中止，未授予整套 smoke 通过结论。
- Date: 2026-10-25
- Initial synchronized baseline: `ca56aa802c2acf9a22bca32d1bee3276381ed203`
- Integration runtime source reviewed: `0dcd03a9b9e87b8666112e72af405a075c67221b`
- Interrupted smoke source: `90fb7b6423d59b0b64332ddaaea6249cca05d9ed` plus the documented retained WIP and documentation overlay
- Owner: Jittor core and compatibility maintainers
- Review when: 下列 PR head、目标分支、捕获/存储/优化器/数值所有权或所声明设备范围变化。

## 范围和判定口径

本次“最近 30 天”按 `updated_at >= 2026-09-25T00:00:00Z` 搜索，包含 open、closed 和
merged PR，避免遗漏最近更新但较早创建的修复。保存的 GitHub REST 快照有 **25 个 PR**，
其中 13 个 open、12 个 closed；closed 中只有 #18042 有 `merged_at`。若只按同一日期的
`created_at` 截止，则是 **20 个**，额外审查的较早创建项为 #18028、#18030、#18031、
#18033、#18034。这里的 PR 状态绑定本次快照，不推断远端将来的状态。

结束前 REST 列表重查受到 API 限流；Git 协议复核的 294 个 pull head 中，本次 25 项仅 #18052 更新，且没有更高编号的新 PR。其新增两提交已单独审阅并合入；其他 head 未变。PR 状态与 updated_at 范围仍绑定最初保存的 REST 快照，不将 Git 引用复核冒充状态列表刷新。

逐项比较实际 Git head、merge-base、提交组和当前源码 owner。提交不在 ancestry、
patch-ID 不同或旧文件路径不存在，都不能单独证明功能缺失。历史报告仅用于定位和理解
边界；新整合是否通过取决于本轮实际执行的门禁与设备证据。

“完整合入/移植”表示本轮接受该 PR 的完整变更或完整独立修复，不要求保留原提交 SHA；
“部分合入”只授予表内明确列出的内容。其余内容继续保留原 PR，不能把部分修复写成
整个库、整个后端或整个 PR 已验收。`29adfeb69` 只收窄新增测试至已提交优化器路径，`33f640659` 只修收集/CI-cache 结构检查，`90fb7b642` 只更新发布基线及 manifest；它们没有改变 `0dcd03a9b` 的 runtime payload。`509863fed` 随后只整合三个测试/helper 文件中的上游门禁修正。所有整合目标均为当前 `2.0-refactor`，没有借本次
审查改变 GitHub PR 状态或向远端推送。

## 25 个 PR 的处置

| PR；快照 head | 处置 | 当前源码证据与不整包导入的原因 |
| --- | --- | --- |
| [#18054](https://github.com/Jittor/jittor/pull/18054)；`783968e1f` | 完整移植 | `34c8650b8` 引入 DataTrove CPU 数据集/本地管线的三个场景、严格依赖缺席策略、工具与报告；原 head 和整合树的 patch-ID 等价。它没有新增生产 adapter。Linux forkserver、CPU 和历史外部库报告不自动授予 Windows、加速器或当前完整 nightly 覆盖。 |
| [#18053](https://github.com/Jittor/jittor/pull/18053)；`12bcfc60b` | 部分合入 | `cd2471909` 保留 pad 整数宽度/负裁剪与 log1p、softplus、sigmoid 的精度/梯度修复，`e2f6924f7` 修复 const StackVector 访问。没有导入 512 文件的 owned-carrier 根迁移、耦合的符号/嵌套张量、NPU/FSDP/checkpoint/通信栈。原报告本身还保留 PPO 元数据严格检查与重建验收缺口；本轮不宣称完整 verl PPO 支持。 |
| [#18052](https://github.com/Jittor/jittor/pull/18052)；`5c2349d05` → `26b26c5b8` | 完整合入，并补修复 | 原 head 是整合树祖先，由 `65c92ca41` 合入累计的训练、存储、rematerialization、自定义 Function、捕获梯度发布、CUDA 接口及门禁修复。`f18a98d79` 另修 unbatched collate/default_convert 与 sized IterableDataset 分流。保留累计依赖链，未只挑最早的 Swift 接口提交。结束前另合入 `509863fed`，补齐后续两项环境跳过分类及 cache 路径契约；最新 head 是其第二父提交，17 项定向通过。源码完整合入不等于原历史 CUDA/ms-swift 矩阵自动成为当前测量。 |
| [#18051](https://github.com/Jittor/jittor/pull/18051)；`25823c28f` | 由新整合的实现覆盖 | 四处 typing 问题由 #18052 的 `7556ce99c`、`1b25b3d39` 覆盖：callback deque、原生 copy/leaf 声明、nox 参数列表、CUDA wheel tuple 类型。原 patch-ID 不等价，但实际 owner 已保留这些契约；不重复覆盖较新的类型声明。最终 typing 结果单列。 |
| [#18050](https://github.com/Jittor/jittor/pull/18050)；`80903a3f9` | 部分合入 | `cb666a7bc`、`2f9268f79`、`8aa8319ca` 保留 leaf reduction 写回、setup_context、reduce_scatter、record_stream、优化器重新绑定及混合 FSDP/plain 参数保护。拒绝 adapter `_zero.py` 直接调用 Tensor 接收者的 `_storage_is_contiguous`、`_mem_ptr_now`、`_copy_into` 私有接口的性能捷径；native.Var 和公开 placement/device_id 本身不是拒绝理由；继续使用当前公开 bridge/原生 owner，不把 DeepSpeed 私有元数据放进公共 Tensor 语义。 |
| [#18049](https://github.com/Jittor/jittor/pull/18049)；`403b90455` | 完整移植独立修复 | `c1de81a97` 引入 cgroup v1/v2 内存层级预算，保留 affinity、CPU quota、用户上限和动态 native compiler worker 策略。不是用宿主总内存替代容器预算，也不把源码 AST 桩当成唯一验证；有实际 C++ worker-count harness。 |
| [#18048](https://github.com/Jittor/jittor/pull/18048)；`ff5ed494f` | 部分合入 | `d50d0a6f7` 引入精确 index_fill、避免重复工厂 cast 并保留现有 fresh-leaf/三角算子梯度；`f54a96228` 引入外部操作 barrier、Tensor.data_ptr/Triton 边界和外层 CUDA capture 所有权，扩展到 StepCapture。拒绝整包 vLLM/旧 owner 覆盖：原 PR 明示 NPU 数值分歧及实验 CUDA H2D 崩溃，部分 adapter 还访问私有 ProcessGroup API。完整 vLLM、NPU 和未选 kernel 不授予新验收。 |
| [#18047](https://github.com/Jittor/jittor/pull/18047)；`e13959314` | 已有覆盖 | `244ebec9c` 已有无注册 kernel 时的 dispatch membership 快路径。保留当前后续调度优化，不重新导入旧函数体。 |
| [#18046](https://github.com/Jittor/jittor/pull/18046)；`dde61ffbf` | 不合入 | 对应 source-signature memoization 曾由 `2872b4b3c` 引入，随后 `23eb049a3` 因同进程源码失效和 extracted-function 回归撤回；所优化的源码遍历也不在实测热点路径。不能将既有有意回退再次当作漏合并。 |
| [#18045](https://github.com/Jittor/jittor/pull/18045)；`19af33f35` | 已有覆盖 | `92cb97515` 已有 per-dependency 编译 cache key memoization；审查时 `src/utils/cache_compile.cc` 与 PR head 一致。 |
| [#18044](https://github.com/Jittor/jittor/pull/18044)；`ee80c29db` | 已有覆盖且后续增强 | `d5c631a2f` 已扩展 Python 3.9/3.10 launch-origin cache；`3577b70b4` 等后续处理 pre-3.11 `f_lasti` 与 origin 热路径。保留较新实现。 |
| [#18043](https://github.com/Jittor/jittor/pull/18043)；`bde3aeccf` | 部分合入 | `1101b06fa` 为 GraphReplay 和自动 PolicyStep 增加 registered buffer/child/non-persistent buffer 的 holder guard，保留捕获内状态 adoption。310P SoC、NZ packed Linear、PFA prefill、raw ACL free 同步等仍是独立缺口，需要真实 310P/共享后端验收；PR 当时完整 CPU smoke 未收尾，历史 Qwen 性能表不是本轮测量。 |
| [#18042](https://github.com/Jittor/jittor/pull/18042)；`73bffaed5` | 已正式合入 | 官方 merge SHA `659e6a47f42466ba407f2d453c0d1224ba5ca0c6` 是整合树祖先。不能因 GitHub head SHA 不直接处于 ancestry 就重复合并；实际证明对象是正式 merge commit。 |
| [#18041](https://github.com/Jittor/jittor/pull/18041)；`7ccc796ca` | 部分合入 | 既有 `b507d0af9` 已覆盖异常分类/provenance；本轮 `eed787100` 补 SIGINT 留给 Python、初始化后 worker 忽略 SIGINT，并增加异常类型与 traceback 回归；异常传输保留 #18052 较新的 JSON owner。uv/dev 配置已存在。拒绝混合 CUDA12 runtime/双 CCCL/range 打包的整体替换：会删除当前独立 CUDA13.0 固定 toolchain、C++17 和 runtime13 链接实现。 |
| [#18040](https://github.com/Jittor/jittor/pull/18040)；`42dbb52ea` | 与 #18041 重叠，不重复导入 | 其 head 位于 #18041 的历史提交链，按同一功能组审查。uv/报错已有或已选择；CUDA 打包差异保留 #18041 行所述拒绝理由，不把两个入口算成两份新增功能。 |
| [#18039](https://github.com/Jittor/jittor/pull/18039)；`7282e9b59` | 既有部分覆盖，剩余不整包合入 | `d7ef14970` 已选 detached Function 输出梯度边；不兼容的 control-dependency exec-plan 变更由 `ce7e54a16` 明确回退。NPU device/deterministic/HCCL WORLD 基础已有选入；当前真实 backward callbacks、post-accumulate 和原生分布式 owner 不可由旧 no-op callback/环境变量分流覆盖。旧双 NPU DeepSpeed 报告不代替新验收。 |
| [#18038](https://github.com/Jittor/jittor/pull/18038)；`48c5da681` | 保留现有替代方案，剩余暂不导入 | 当前 source-pinned DeepSpeed eager integration 替代旧 finder/activation/scope adapter。Adagrad 数学与部分 GradBucket 能力仍缺，不记为已支持；原 patch 同时引入旧 optimizer_state/Adam 所有权，需要独立迁移及 device/状态验证，不能覆盖当前优化器与用户本地存储身份修改。 |
| [#18037](https://github.com/Jittor/jittor/pull/18037)；`00f81c168` | 保留现有替代方案，剩余暂不导入 | 与 #18038 的旧 adapter/Adagrad 组重叠，另包含旧 NPU、RNG、HCCL 方案和大量历史 probe。保留当前 native RNG/HCCL/梯度 owner；原 no-op queue_callback 与缺失当前 detach_/post-accumulate 的方法表会产生回退，旧二进制 hash 也不能授予当前发布验收。 |
| [#18036](https://github.com/Jittor/jittor/pull/18036)；`a79e57069` | 既有部分覆盖，剩余不整包合入 | `4ae54723f`/`4896af611` 已有 PEFT replay、NumPy 索引、CPU allocator 安全修复；`f1e7d1530`、`4233b3f97`、`f7ab3738b`、`223bbdd3f`、`fa077c459` 已有 ACL 索引、设备、确定性、peer copy、per-device RNG。原 broad-exception swallowing、旧 RNG/threading owner 和历史收敛矩阵不覆盖新实现。 |
| [#18035](https://github.com/Jittor/jittor/pull/18035)；`088593618` | 所需修复已有覆盖 | `60e7f463b` 已有 fused AdamW 分批、meta frontend、同设备生态 runner；`99bb20ddb`/`ae1c65311`、`52edb2b73`、`de2e2f1be` 已有 RMS 舍入、AdamW 默认、launcher 别名和 python-config headers。保留后续 attention/output-backward/storage owner，不导回旧 refactor-wip 结果目录。 |
| [#18034](https://github.com/Jittor/jittor/pull/18034)；`cfe1b415e` | 已有部分覆盖，剩余暂不导入 | `f152bf849` 已有原生可微 STFT，`2acafe9f3` 已有归档延迟模块修复。bool SparseCOOTensor/to_sparse/Whisper gate 尚缺，不能称已支持。原 SparseVar 文件会删除当前验证、coalesce、CSR；应独立添加能力并按现行下游准入重验，而非用旧 CPU L2 定义替代当前标准。 |
| [#18033](https://github.com/Jittor/jittor/pull/18033)；`1cf2a4e19` | 已有覆盖 | `8224841d4` 已在 `python/jittor/contrib/loss_ssl/` 和 `tests/nn/test_loss_ssl.py` 提供 TotalCodingRate；不重建旧顶层 loss_ssl。 |
| [#18031](https://github.com/Jittor/jittor/pull/18031)；`e5cffe1ae` | 部分合入，其余按 owner 保留/拒绝 | 本轮 `1ddc66497` 选择 safetensors 缺失 metadata 返回 None、Conv1d 空间序列参数归一化、Kaiser 负长度拒绝。其 BERT/多模态、原生 distributed/FSDP/checkpoint/RNG 等多数已有较新 owner；旧 safe_globals no-op、生成标量 monkeypatch、重复分片数学不可覆盖它们。pointwise AMP、type_as 跨 device、TensorDataset 批量 gather 等未选差异仍明确保留为未整合范围，不能把 72 个 patch-distinct 提交都称已覆盖或都称缺功能。 |
| [#18030](https://github.com/Jittor/jittor/pull/18030)；`8bdb533c9` | 已有覆盖且使用后续修复 | 当前 `src/ops/composite/getitem_op.cc` 仅在 storage_view 路径折叠 implicit dims，拷贝路径保留独立 stride；`7108fa6c5` 和现有 storage-strides split 回归覆盖多维转置前缀。保留后续高维/strided 寻址处理，不覆盖为旧两行改法。 |
| [#18028](https://github.com/Jittor/jittor/pull/18028)；`e5cffe1ae` | #18031 的同 head 重复入口 | 两个 PR 的完整 head SHA 均为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`；#18028 base 为旧 `2.0`，#18031 base 为 `2.0-refactor`。按 #18031 的提交组审查一次，不重复导入另一份旧布局。 |

## 复现与分项验证证据

下面是明确绑定被测子分支的分项证据，**不是最终整合树完整通过的替代物**。没有将
skip、未收尾进程、只读源码结论或历史 PR 报告计为本轮设备通过。

| 范围 | 最小复现和已执行结果 |
| --- | --- |
| cgroup compiler 预算 | 原 get_total_mem 在 2 GiB cgroup fixture 中仍返回 96 GiB 宿主值；修复后 19 项通过，包含实际 worker-count C++ harness、v1/v2/ancestor/malformed/unlimited、CPU/affinity/quota 和同进程额度变更。 |
| worker 异常与 SIGINT | 原实际 Dataset 方法把 ValueError traceback 包成 RuntimeError；修复后 7 项定向通过，完整 dataset 文件 18 passed/5 skipped（3 network、MPI、accelerator）。SIGINT 用 bounded child helper，worker case 等待两个 worker 完成初始化，避免把 Python fork hook 尚未结束的窗口误作已初始化 worker 契约。 |
| registered buffers | 真 CPU buffer 更新回归在改前失败；改后 TestGraphReplay 29 passed/0 skipped，包括 load/update/assign、嵌套共享 holder 和捕获内累积状态。 |
| opaque operations / nested capture | 在改前真实 CPU/原始 owner 上执行 ctypes 指针写入，explicit replay 返回错误数据；修复后原生 GraphReplay+StepCapture 44 passed/0 skipped，Torch foreign pointer eager/automatic/explicit/step、捕获外不污染与 API 契约 8 passed/0 skipped。CPU mock 只覆盖 native-capture 路由，不能单独证明 CUDA graph 记录。另有真实 CUDA outer-capture/Triton 测试供设备验收。 |
| #18031 小契约 | 改前真实 Conv1d tuple 空间参数抛 TypeError、负 Kaiser 未拒绝、缺 metadata 返回 {}；修复后 native ConvParityCPU 11 passed，compat metadata/Kaiser/Hann 5 passed。 |
| #18048/#18053 数值与 padding | 改前实际 const StackVector 迭代编译失败，负 pad 生成 `i0--5` 被实际 g++ 拒绝，CPU log1p 极小输入丢失、sigmoid(-350) 严重失真；实际原 index_fill/cast 调用也复现大 int64/NaN 与多余转换问题。修复后 66 项 Torch、3 项 native padding 通过，另有 10/1 项加速器 deselect；检查输出与完整输入 VJP。最终实际设备结果另列。 |

本轮使用的维护入口与模式拆分如下；先按环境手册预置工具链，并为各模式指定独立的 lab `JITTOR_HOME`/`TMPDIR`。wheel 的 `$core_wheel`/`$compat_wheel` 是上述 canonical 工具产生的文件；不要拿含用户 WIP 的 wheel 更新发布基线。

```bash
python tools/run_test_suite.py --tier core --backend cpu --jobs 0 --serial-compile
python tools/run_test_suite.py --tier smoke --backend cpu --jobs 0 --serial-compile
JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q tests/structure/backends/acl/test_acl_dtype_preservation.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python:adapters python -m pytest -q \
  tests/structure python/jittor/compat/tests/structure adapters/tests/vllm/test_structure.py \
  tools/release/test_check_wheel_contents.py tools/release/test_check_sdist_contents.py \
  --ignore=tests/structure/backends/acl/test_acl_dtype_preservation.py
python tools/release/check_wheel_contents.py compare "$core_wheel" --baseline tools/release/baselines/wheel-contents-final.txt
python tools/release/check_wheel_contents.py compare "$compat_wheel" --profile compat --baseline tools/release/baselines/wheel-contents-compat-final.txt
python -m mypy
bash tools/check_repo_layout.sh
```

实际 native CUDA 定向集合为 `tests/core/test_graph_replay.py::TestGraphReplayCuda`、
`tests/core/test_alias_autograd_state.py`、`tests/backends/cuda/test_fused_attention_f32.py`、
`tests/backends/cuda/test_parallel_level_zero.py`；Torch CUDA 集合为
`test_activation_checkpoint`、`test_activation_materialization`、`test_activation_rematerialization`、
`test_compile_external_autograd`、`test_raw_cuda_stream`、`test_torch_dataloader_iterable_cuda`、
`test_custom_function_forward_mode`、`test_custom_function_saved_inputs`、
`test_graph_replay_foreign_write`、`test_strided_gradient_writeback`（均位于
`python/jittor/compat/tests/torch/`，文件后缀 `.py`）。设备配置由同一维护 runner 的
`_session_environment(..., backend="cuda")` 建立并通过真实 CUDA warmup，pytest 每例超时 300 秒。

分项日志与 API JSON 保存在未版本化的 `$JITTOR_LAB_ROOT/pr-audit-2026-10-25/`、
`$JITTOR_LAB_ROOT/_state/pr-audit-20261025/` 和
`$JITTOR_LAB_ROOT/_state/pr-audit-numerics/`。审查表源材料为 `prs.json`、
`older-audit.md`、`ecosystem-audit.md`、`perf-typing-disposition.md`、
`pr18031-groups.md`；仓库中不复制原始日志、缓存、编译产物或下游 checkout。

## 整合时发现的存储边界

原主树的三个未提交文件和未跟踪测试已保存于独立 Git 快照
`preserve/local-optimizer-before-pr-audit-20261025`（`deb54e50d07e44406ce288f041b3d8fc94ae40cc`）。
新版 `_optimizer_update` owner 已覆盖原 frontend/base hook 的意图；fused Adam 的三个
`_update_preserve_dtype` 调用仍是用户本地工作，未进入发布 wheel 的源码基线。原测试文件
blob `90b7573f81575f45a12da54381b91dcbfc86ae3f` 保持不变且不跟踪。

用户测试暴露了 eager storage write 的既存 saved-output 缺陷：未观察的 `p.exp().sum()`
在更新参数后反向会错误重算 exp。`deb54e50d` 与未修复集成 `1fa76c8cc` 均有相同四个
CPU 失败，不能归因于 #18052 新引入。`0dcd03a9b` 在 native write owner 保存最近的、
尚未独立分配且反向仍需使用的 reader outputs，沿 storage alias 和未分配的中间图遍历，
保留 kept/replay owner 与 input-saving version guards；不新增全局 flush 或执行无关图。
CPU Torch 定向 85 passed，native alias/autograd 全文件 32 passed；完整 identity/updates/compile 矩阵 140 passed、0 skipped，其中 71 项实际 CUDA、69 项 CPU（含用户 WIP），另有整合树原生 CUDA 92 passed、0 skipped。
新增已跟踪回归仅覆盖已提交的 SGD plain/fused 与 portable AdamW，用户 fused AdamW
工作由原未跟踪测试另行验证，不能将带 WIP 的成功冒充干净发布源码结果。

独立审阅也在未修复集成基线上复现了三个旧的 logical/physical transpose 与 weakref
测试边界，以及 warm replay 之间新建旧图的保存值限制；它们分别记录为
KI-TEST-015 与 KI-EXEC-015。clean fused AdamW holder rebinding 的边界记录为
KI-COMPAT-016。这些不能把本轮 native 修复描述为已解决任意跨 replay 历史图的声明。

## 最终整合验收

运行时门禁使用 Python 3.9.25、NumPy 1.26.4、g++ 11.5 与 CUDA 12.2.140，真实 NVIDIA
设备为 RTX 4090。缓存、Python headers 与 toolkit overlay 在 lab 隔离目录；没有修改
系统 Python 或 CUDA 安装。该环境不是 canonical nox Python 3.11 typing 环境，不能用
mypy 1.8.0 的 68 文件通过冒充后者。wheel 则另用 Python 3.11、setuptools 83.0.0、
wheel 0.45.1 的隔离 canonical 工具构建，保留既有 METADATA/WHEEL/license 布局。

| 门禁 | 对应源码及结果 |
| --- | --- |
| CPU core | `0dcd03a9b` + 保留的 Adam WIP；native 98 passed/44 skipped/1 xfailed，Torch 225 passed/1 skipped，总计 323 passed/45 skipped/1 xfailed，退出 0。 |
| CPU smoke（中止） | `90fb7b642` + 相同 Adam WIP 与三份文档覆盖层；执行 `--tier smoke --backend cpu --jobs 0 --serial-compile`。native 会话推进至 ops 测试，首次 JIT 仍在进行时按用户要求终止；未完成 native/Torch 两段，未获得完整计数或验收退出码，不记为通过。外部 `smoke-user-stop.json` 保存中止时间和受控进程来源。 |
| 原生 CUDA 定向 | 同上；GraphReplayCuda、alias/autograd、fused attention FP32、parallel-level-zero 共 92 passed/0 skipped，退出 0。 |
| Torch CUDA 定向 | `0dcd03a9b` + 保留的 Adam WIP；checkpoint/materialization/rematerialization、外部 autograd、自定义 Function、CUDA stream/DataLoader、foreign writes、strided gradients 共 119 passed/0 skipped，退出 0。 |
| Clean-source structure | 测试树随后原样记录为 `6edcf1ae1`（运行时为 `11888f1ae` + 随后提交的三份基线文件，无用户 WIP）；独立 source-path 环境中 native-only ACL recorder 5 passed，Torch/packaging/adapters 1690 passed/4 skipped；两段均退出 0，严格 execution accounting。 |
| 原工作树最终 structure | `509863fed` + 三份文档覆盖层、Adam 三行 WIP 和未跟踪原测试；完整 Torch/packaging/adapters 1690 passed/4 skipped，退出 0，374.58 秒。严格 execution accounting，前后 HEAD/status/diff/覆盖文件 hash 全部一致；native-only ACL recorder 已另有 5 passed，不在该 Torch 会话重复执行。 |
| 存储完整矩阵 | `881e9ece6` + 相同 Adam WIP + 原测试；identity 128、updates 10、CUDA compile 2，共 140 passed/0 skipped，退出 0；实际 CUDA 71 项、CPU 69 项。 |
| canonical wheels | clean committed `0dcd03a9b`；core 1,141/compat 185 成员资源审查通过，两份精确基线比较均 +0/~0/-0。core 对旧基线仅新增 cgroup Python helper/头文件；compat 另从 `ca56aa802` canonical 构建审阅 +5/~38/-0（37 源码加 RECORD）。所有源码 payload 与 Git blob 逐字节一致，无元数据、license 或删除漂移，不含用户 WIP。 |
| 最新 #18052 门禁 | `509863fed`，完整 gate-scope/cache 集合 17 passed/0 skipped；仅测试相关文件，runtime/wheel payload 与 `0dcd03a9b` 一致。 |
| 报告与布局 | 最终新增报告、索引及 SGD dated follow-up 后，文档/私有路径定向 19 passed/0 skipped，退出 0；布局与文档治理通过（298 active Markdown files），`git diff --check` 通过。 |
| typing | 相同的 `6edcf1ae1` 测试树，68 个被检查源码与 `0dcd03a9b` 一致；mypy 1.8.0、68 source files、退出 0；范围受上述 Python 3.9 工具环境限制。 |


- 用户要求收尾并停止后，本轮 CPU smoke 已中止。已完成的 core、真实 CUDA、结构、typing 与 wheel 证据按上表保留；没有完整 smoke 结果，不据此声明整棵测试树通过。原工作树最终结构检查已完成。
- 本轮未安装、未执行真实 DataTrove，也未重新进行 ms-swift、vLLM、DeepSpeed 或 verl/PPO 端到端验收；#18054 的历史 DataTrove CPU 报告仅作审查材料。本轮有真实单卡 CUDA 定向执行；ROCm、Ascend/NPU/310P 和多机/多卡集合通信未验收。对应上机范围已记入 deferred-hardware；不能由 import、CPU fallback、历史数字或 mock 路由补齐。
- 原工作树已通过 three-way checkout 和带旧值校验的 ref 更新回到 `2.0-refactor`，代码提交 `509863fed`。两处 hook 重叠采用新 owner，fused Adam 三行写回仍未提交；原未跟踪测试 blob 保持 `90b7573f81575f45a12da54381b91dcbfc86ae3f`。完整四文件原始字节保留在 `preserve/local-optimizer-before-pr-audit-20261025`。没有 stash、丢弃本地工作或暂存用户原文件；实际 runtime overlay 与已验收存储矩阵一致。

本报告的结论是按变更内容选择并整合当前分支所需修复，而不是关闭所有未完成的下游
能力。表中未选内容和真实设备边界继续由相应 PR/问题总账承担；不将旧图表、旧适配
等级或旧环境 wheel hash 升格为本次最终发布结论。

## 同日第二轮补整合

- Status: 已整合；CPU core、结构门禁与定向 CUDA 执行通过，未跑完整 smoke/full。
- Baseline: `280d44854b3aeeb0e30b02b7b2e4e14e7a94b1ba` 及其上方逐 PR 处置已提交。
- Final commit of this round: `220c0bc02`（另加本报告提交）。
- Owner: Jittor core and compatibility maintainers。
- Review when: 上表处置、下游 adapter 准入面或所声明设备范围变化。

本轮只补上一节明确记为“尚缺/未整合”、且能在当前环境验证的部分；不重新导入已被
拒绝的旧 owner、私有 Tensor/ProcessGroup 接口、历史 wheel/CI 基线或 NPU 专用实现。

| 组 | 来源 | 合入内容 | 本轮证据 |
| --- | --- | --- | --- |
| 门禁收集 | #18052 新增两提交 | xdist worker 的 collected-file 集合在 session policy 前汇聚到 controller；CUDA 不可用时保留 backend 身份的跳过原因 | gate-scope 18 passed；真实 `-n 2` 24 passed |
| 类型契约 | #18051 残差 | `copy(..., preserve_format=False)` 存根默认值（其余四处已在树内） | 结构测试通过 |
| DeepSpeed | #18050 | `_finite`/`_norm` 公开 API 快路径、`torch._C._current_graph_task_id`、owner-state 弱缓存、NCCL wheel 选择、concat scratch 无梯度边、BF16 GELU 训练核 | adapter 49 passed；concat/grad/NCCL 25 passed；真实 CUDA GELU 14 passed/2 skipped |
| Torch 语义 | #18031 | `Tensor.type_as` 继承 dtype 与 device、反射标量留在张量后端、TensorDataset 单次 gather（并修正双重 collate） | 190 passed；TensorDataset 9 passed；CUDA 定向 2 passed |
| 工厂语义 | #18043 | `device="meta"` 工厂参数的可观测 meta 身份 | 6 passed |
| Adagrad | #18038 | 原生 dense FP32 Adagrad、`torch.optim.Adagrad` 接线与状态迁移；sparse/foreach/fused/differentiable 显式拒绝 | 45 passed/1 skipped（含优化器回归） |
| verl 生态 | #18053 | tensordict、verl、transformers_loading、transformers_qwen35 adapter 及公开 compat 入口 | 门禁测试 27 passed；adapter 自测 283 passed/15 skipped |
| vLLM | #18048 | `Tensor.exponential_`、buffers/model_loader/runner_policy、采样/分页/日志概率/UVA 回归 | vLLM harness 120 passed/72 environment skipped；主树定向 208 passed/75 skipped |
| 门禁与打包 | #18034/#18037 | declared skip bucket 优先于 torch、native-only 文件不进入 Torch 快照、wheel RECORD 审批、ASV matmul 预热、CPU cache 键隔离 | 41 passed |
| 仓库一致性 | — | HEAD 已提交的 results 文档未在生成 `MANIFEST.in` 中登记，结构门禁在基线上即失败；重新生成 | packaging 8 passed |

再次确认不合入：

- **#18046**：进程级 `core_source_signature` 记忆化曾由 `2872b4b3c` 引入、由 `23eb049a3`
  以同进程源码失效与 extracted-function 回归为由回退。本轮不重新引入；它也不是实测
  热点。
- **#18050 的 `_zero.py` 与 `ds_shape`**：前者直接驱动 Tensor 的
  `_storage_is_contiguous`/`_copy_into`/`_mem_ptr_now` 私有接口，后者把 DeepSpeed 私有
  元数据放进公共梯度语义，均按上一节处置移除。
- **#18036/#18039/#18043 的 NPU/310P 专用部分、#18035 的旧 CUDA 内核版本、#18031 的旧
  RNG/分布式 owner**：无 NPU/ROCm 硬件或已被当前实现取代，保持未验证/不导入。
- **#18054/#18049/#18033/#18042/#18030/#18044/#18045/#18047**：已由当前实现覆盖。

验证环境与边界：Python 3.11.16、g++ 11.5.0、CUDA 12.2.140（conda nvcc）与真实
RTX 4090（sm_89）；CPU core 为 native 98 passed/44 skipped/1 xfailed 与 Torch 226
passed/1 skipped，合计 **324 passed/45 skipped/1 xfailed**；`tests/structure`
**1541 passed/6 skipped**。DeepSpeed、vLLM、verl、tensordict 上游包未安装，生态测试
只覆盖 stub/fixture 契约与门禁可见行为，不能记为真机端到端验收；NPU、ROCm、310P 与
多机集合通信未验证。本轮未运行完整 smoke/full，不声明整棵测试树通过。集成期间同一
工作树内另一个会话的 smoke 进程自 `22:0x` 起无 CPU 进展并持续到收尾，本轮没有把它
的任何部分计为结果。

## 2026-10-26 持续修复与测试补全

- Status: 进行中；下表条目已修复并在 CPU 与真实 CUDA 上验证，问题总账已相应关闭或收窄。
- Start baseline: `ffb0f9987`（第二轮整合后）。
- Owner: Jittor core and compatibility maintainers.
- Review when: 下列修复、门禁接入或所声明设备范围变化。

环境：Python 3.11.16、g++ 11.5.0、CUDA 12.2.140（conda nvcc）、RTX 4090（sm_89）。
每个 worktree 使用独立 `JITTOR_HOME`/`TMPDIR`；首次编译串行。上游 DeepSpeed/vLLM/verl
仍未安装，相关结论只覆盖 stub/fixture 契约。

| 缺陷 | 处置 | 证据 |
| --- | --- | --- |
| KI-COMPAT-016 | fused AdamW 写回改走保留存储的 `_update_preserve_dtype` | 存储身份 128 passed；直接与传递旧输出 8 passed；CPU+CUDA |
| KI-COMPILER-008 | `precompile` 保留异常类别；@ 解析、# 指令、花括号、@for/@strcmp/@alias、arity 迁移 `USER_CHECK`；kernel_ir/cache_compile/op_register 的公开路径同样迁移，expr/swap 记录为非公开 | 定向 181 passed/5 skipped/4 xfailed；C++ JIT 93 passed |
| KI-COMPAT-014 | complex128 成为前端 dtype（构造、cast、type、报告），complex32 仍占位 | 83 passed/10 skipped |
| KI-DTYPE-003/004 | Python float 对 float64 保持 53 位；工厂结果不再按弱标量提升，放置由 `_is_constant` 维持 | CPU 77 passed；CUDA 55 passed；strict xfail 标记移除 |
| KI-COMPAT-007 / KI-AUTOGRAD-003 | 基线已修复，补齐 `copy_`-no_grad 与 register_hook 驻留回归 | 7 passed/1 skipped |
| KI-TEST-015 | storage-view 夹具区分逻辑/物理转置并修正生命周期断言 | 文件 14 passed |
| KI-TEST-002 | `run_test_suite` 与 nox 接入 `check_session_completed.py`；xdist 汇聚、setup-skip 计数 | 结构/工具 72 passed |
| KI-TEST-007 | `test_fused_op_relay_matmul` 在三个历史污染文件之后均通过 | 三种顺序各 2/4/5 passed |
| KI-TEST-011 | 生成源码与缓存按 UTF-8 读取（全树同类读取一并修正） | C locale + 关闭 locale coercion/UTF-8 模式，CUDA 1 passed |
| KI-OPS-014 | 全掩码 `finfo.min` 行已由 max 与 `1/l` 分离修复（基线已有） | dense-additive/key-padding CUDA 2 passed |
| KI-OPS-015 | SDPA CUDA 反向已由 `4c722fee0` 修复；本轮记录其依据 | 两个 device-parity 用例在 RTX 4090 通过（2 passed，约 1e-6） |
| KI-TEST-010 | LayerNorm 夹具显式放到设备上，融合路径真正被覆盖 | norm 文件 CPU+CUDA 28 passed |
| KI-TEST-014 | resnet notebook 改断言性质（非单调）并移除已登记的预期失败 | manual smoke 3 passed（use_mkl=0） |
| KI-COMPAT-013 | 模块前向在输入未放置时跟随模块自身已放置的参数 | autocast 套件 CPU+CUDA 12 passed |
| KI-MEM-002 | host 读取返回副本，不迁移源存储 | CUDA 驻留契约 16 passed；CPU 6 passed |
| KI-OPS-012 / KI-BACKEND-009 / KI-BACKEND-014 | 半精度 max/min 恒等值（基线已有）、逻辑/位归约的 16/64 位原子与 bool 结果、cublas/cudnn 自测算子设备化 | 归约矩阵+minmax 24 passed；三个库自测类 3 passed；均在 RTX 4090 |
| KI-COMPAT-017 | 捕获步骤的状态提交在 `keep_graph=2` 下执行，回放保留可重跑的前向 | `test_torch_compile_replay.py` 11 passed |
| KI-COMPAT-018 | portable 快照把非连续张量先 `contiguous()` 再批量 fetch | 两个 portable 节点 passed（safetensors dtype-bytes 节点是环境缺口） |
| KI-COMPAT-019 | autocast 下 LayerNorm 主体按 ATen 策略在 float32 累加、结果一次收窄 | bf16 训练损失 0.128 vs float32 0.122（4.5% < 25%）；half 套件与 CUDA core 361 passed |
| KI-CODEGEN-005 / 006 | 2-D 与 mean/jt.sum 全和融合已修复并扩展门禁（005 关闭）；rank-1 在 reshape 之上的融合仍缺（006 打开） | CUDA full-reduce 12 passed；rank-1 仍分配 68,157,440 bytes |
| 18031 第二批语义 | `Tensor.type_as`、反射标量留在张量后端、TensorDataset 单次 gather、掩码写入梯度、expand 标量维、torchvision resize/pad/color/crop、Conv1d device/dtype、advanced-index/0-dim 设备对齐 | CPU 373 passed/16 skipped；CUDA 定向 14+2 passed |

| 最终门禁（本轮修复合并后） | CPU core 与 structure | core：native 98 passed/44 skipped/1 xfailed + Torch 229 passed/1 skipped（组合 327 passed/45 skipped/1 xfailed，退出 0，session-completion 标记 143/143、230/230）；structure 1548 passed/6 skipped，退出 0 |
| 设备组合门禁 | CUDA core（RTX 4090, sm_89） | native 131 passed/10 skipped/2 xfailed + Torch 230 passed = **361 passed/10 skipped/2 xfailed**，退出 0，session-completion 标记 230/230 |
| PR smoke（CPU，冷缓存） | 本轮修复后的整树选择 | native：4084 passed/2014 skipped/18 failed/239 errors（45m44s）；torch：4775 passed/1878 skipped/16 failed/2 errors。本轮引入 3 个契约回归已修复并复验（op-compiler 旧消息、原生 full-mask 标量 dtype、half 标量物化；7/15/50 与 half sweep+promotion+dtype 70 passed）。其余失败在 280d44854 基线上同样复现或属验证环境缺口（venv 无 scipy、safetensors 不完整、无可用独立 PyTorch、`.pth` 中 modelscope `tests` 遮蔽、真实 triton/vllm）。其中 serialization 批量 fetch 与 compile replay SGD 登记为 KI-COMPAT-018/017，bf16 autocast CUDA 发散（79.5%）登记为 KI-COMPAT-019；三者均已修复并关闭（见上表）。 |

本表不把 partial 文件或单个用例的成功写成整库、整后端验收；NPU/ROCm/310P 与多机集合
通信仍未验证。KI-COMPAT-012 的机制已修复（进程内 shim 发布 torch distribution 元数据，
元数据回归通过），但下游 Transformers/Diffusers 用例仍需在装有这些包的环境验收。
另：一次裸 pytest 宽扫描在验证 venv 中把 `tests._helpers` 解析到 `.pth` 里的 modelscope
`tests` 包而报 ModuleNotFoundError；那是验证环境遮蔽，不是产品缺陷，正式门禁走
`tools/run_test_suite.py` 不受影响。
