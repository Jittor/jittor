# 初始化训练开关与 no-grad retained-view 写回

- Status: CPU / 真实单 CUDA 的限定契约通过；原生 CUDA expand 内存门禁另列。
- Date: 2026-10-15。
- Owner: Torch initializer / native storage maintainers。
- Baseline: 在 `bb910319abe420c93b641bc1ac64c48bfac09efa` 后保留已交付的 Generator/Tensor 历史证据，先同步 `8929633bef1e11294c3f0dfbef01bab9c0e015fe`，测量冻结树为 `b0bd14bccf6d778305a5dd5e0d2716de2659ef09` 加逐文件记录的未提交修改。无在途测试时再整合 `14aeca5df0bf841e99d6241e06b076e34ef08c05` 为 `369c5c255921fc117d216f0f99266246b55f1ed6`，七个测量 owner 文件逐字节不变。
- Code: `cc5fc5eeabcb5dc66e3fcba2e76b7fc014d3d31f`（initializer）；`61c67b67a4d864c39810d47bfddf96d605a2d295`（native / Tensor writeback 和最终测试）。
- Review when: VarHolder 原位写回、view provenance、storage、stream 或 initializer owner 变化。

## 真实断点

先用独立 binary Torch `2.11.0+cu128` CPU 对照，再在候选基线复现：
九个初始化器分别作用于 Tensor / Parameter 的可逆 `requires_grad_(False)`，候选 18/18 错误解冻；真 Torch 保持 False。
对 `root * 3` 的真实非叶 tensor 初始化，候选重新绑定成叶并丢失 root 梯度；真 Torch 保留非叶与完整值为 3 的梯度。
首轮修复的连续非叶与冻结态通过后，三条严格叶 view 测试仍失败。
纯 `with no_grad(): view.copy_(...)` 也独立复现相同断点，定位到共享 storage 写回，未用 xfail 收口。

物理 stride 修复后，CPU 首轮仍有 17 个失败：默认 `transpose_storage_view=0` 的 `.T` 是保留来源的 materialized view。
旧 fallback 会更换 base / view / sibling 的 Var 与图身份，root 梯度为 None。
临时启用物理 transpose flag 的诊断结果只用于归因，正式验收断言 flag 仍为 0。
这些原始失败、分阶段 patch 与源码副本保留，不改写为最终通过。

## 实现

`nn.init._assign` 在 no-grad 下复用 Tensor storage 写回；原生 initializer 先在目标同 device / dtype 的临时值上运行，再写回原目标。
不以 native stopped 标记代替实际公开 requires_grad，也不重新 start_grad 一个原有非叶图。
新测试实际覆盖构造冻结、可逆冻结、重新启用、真实 leaf / nonleaf 与 Parameter。
此前 Generator 报告只覆盖构造态，本页是后续新验收，不回填旧范围。

原生 `_copy_into` 对正 stride、可证明不重叠的目标按连续 runs 调用既有 backend_copy，保留 dtype/device/stream owner。
直接 expanded / overlap 目标在写入前拒绝；同一 layout 自拷贝不动值。
对 address span 相交但 stride 模 residue 可证明元素互不相交的别名，先由既有 device CopyOp 快照。
无法证明安全的稀疏别名明确拒绝，未声称覆盖所有任意 stride 的 Torch copy_。

新增私有 `_copy_view_into` 只处理存在 materialized retained views 的 no-grad 写回。
它复用原 view steps / SetItem 数学，在独立临时 buffer 上准备 root 与全部 materialized sibling 值，完整验证后才写原 buffer。
临时 SetItem 算子只传播数值；原 root / view / sibling Python holder、native graph-node identity、data pointer 与旧图均保留。
跨同步可能释放 GIL 时，原 holder 被明确持有。
物理 sibling 通过共享 buffer 更新；只读 expanded sibling 压缩重复轴后刷新其唯一 backing storage，直接 expanded 写目标仍拒绝。
调用者先完成 pending readers / 旧 clone，CUDA stream 与 CPU→CUDA 传输继续由已有 backend_copy 等待和提交；没有 host 数值回退或另造 AD。

## 验证口径

两个新维护测试文件在独立真 Torch 和候选上运行相同通用节点，完整比较数组与梯度；四个 native-only observer 函数仅候选执行。
九个初始化器为 uniform_ / normal_ / zeros_ / ones_ / constant_ / kaiming_uniform_ / kaiming_normal_ / xavier_uniform_ / xavier_normal_。
copy 测试包括 slice / step / transpose / nested、leaf / nonleaf base、float16 / bfloat16 / float32 / float64、CPU source→CUDA target、source alias、拒绝原子性、materialized siblings、expanded read-only sibling 与原 native graph-node identity。
候选完整测试在 forbid-fallback scope 中运行，记录 actual native allocation / ordinal，正式测量前后重新核七个 owner 文件哈希。
未要求不同 PRNG 的随机样本逐位一致；参考验证初始化与 autograd 行为契约。

| 检查 | 实际结果 |
| --- | --- |
| 独立 Torch CPU，同两个最终测试文件 | 114 passed / 114 CUDA skipped / 8 native-only deselected |
| 独立 Torch CUDA，同两个最终测试文件 | 228 passed / 0 skipped / 8 native-only deselected |
| 候选 CPU，最终扩大初始化 / copy 回归 | 197 passed / 192 CUDA skipped；fallback 0→0 |
| 候选 CUDA，同扩大回归 | 394 passed / 0 skipped；fallback 0→0；默认 transpose flag 0 |
| 原生 CPU 既有 storage / view 回归 | 16 passed / 6 CUDA skipped |
| 原生 CUDA storage / view / graph capture 回归 | 29 passed / 1 expand 内存预算失败；fallback 0→0 |
| 原生 GraphReplay strided-result 私有 copy 消费者 | CPU 1 passed / CUDA 1 passed；fallback 0→0；CUDA 原 kernel-count 断言通过 |
| Canonical CPU core | 322 passed / 42 skipped / 1 existing xfail，native / Torch 两模式 exit 0 |

正式测试选择：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_nn_init_trainability.py \
  compat/tests/torch/test_tensor_copy_strided.py \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_tensor_rng_initializers.py \
  compat/tests/torch/test_torch_init_package.py
```

按环境手册隔离 HOME / JITTOR_HOME / TMPDIR；CPU 显式隐藏 CUDA，再串行首次编译真实 CUDA。
候选 unit cache 与 benchmark 分离，设备已有外部负载，只验证正确性，不授吞吐或峰值性能。
ROCm / NPU 未验证。

原生 CUDA 扩大回归的唯一失败是既有 `test_expand_does_not_materialize_when_it_is_consumed`：
`(ones((4096,1)).expand(4096,4096)*2).sum()` 数值正确，但 allocator 增长 68,157,440 字节，超过原断言 16,777,216 字节。
本次源码单节点重跑同样失败；随后新建干净 detached 远端 `14aeca5df` 工作树、全新私有缓存、相同物理 CUDA / 工具链，
实际重新编译并运行同一维护节点，增长同为 68,157,440 字节，fallback 0→0。
这是未修复的既有 CUDA expand 内存预算问题，不把扩大 suite 记为全绿，也不改原断言。

完整 `tests/structure compat/tests/structure` 实际为 1620 passed / 6 skipped / 2 failed；
仅旧 `tensor-rng-inplace` 报告尚未在本独立树的中央 results index / MANIFEST 登记。
其余结构项通过；中央文件归集成维护者，登记后的完整复验另属主线证据。
仓库布局检查通过；新报告的文档 / 私有路径定向检查 18 passed / 1 orphan-registration node deselected（上述完整运行已保留该节点失败）。新两测试文件 Ruff 通过，生产两文件旧 21 条 Ruff 诊断与 `bb910` 逐项相等。

## 边界与证据

这里修复 no-grad 初始化 / copy 写回，不授予所有 grad-enabled copy_、任意 overlapping view、所有初始化器或完整 R33 Generator ABI。
独立 CPU probe 另确认：grad-enabled 普通 copy_ 的旧非叶重绑定差异在旧 cast 路径与新去除同 dtype 无效 cast 的路径相同，保留为已有独立边界，不改 AD。
materialized view 写回需要 root 与 retained siblings 的临时值，不宣称零临时内存。

未版本化证据：`$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/trainability/`。
`probe.py` / `nonleaf_probe.py` / `view_copy_probe.py` 与 oracle/candidate baseline 日志保留原断点。
`strided-copy/` 保留 physical / materialized / expanded 三阶段原始失败、patch、源码副本与最终 CPU/CUDA metadata。
`source-freeze-expanded.json` 精确绑定历史 dirty-source 的七个 owner 文件；正式结果不能重标为后续 clean commit 测量。
独立 oracle metadata 绑定实际解释器 / Torch binary extension / 测试哈希；未修改共享 Torch 安装，也不以该单个 binary 哈希声明完整安装包来源。
`manifest.json` 封存原始文件相对路径 / 字节数 / SHA256，`strided-copy/source-final-git-proof.json` 逐一验证七个最终文件等于上述实际测量哈希与 `61c67b67a` Git blobs。中央 index / MANIFEST 由集成维护者登记。


## 主线整合复验

同步远端 `bda0f82df2eb4322a3c81eee2c36d31597a9237d` 后，主线分别运行
可逆冻结、非叶初始化、view no-grad copy 三个原始最小复现，均真实失败。
三个交付提交按顺序重放为 `53ae4b44e`、`6d758252a`、`d07310011`，中央索引与
MANIFEST 登记后冻结 clean `0e85bafad1df77d5b3c200b311db8888b256215c`。
七个 owner 文件均与原封存测量及 `61c67b67a` Git blobs 逐字节相等；原有 122 份
原始文件全部重新核哈希，合计 675,935 字节，未改写历史结果或将其改绑到主线。

本轮主线先完成 CPU 新核心编译，再完成真实 CUDA 新核心编译，各运行使用独立 unit cache；
真实 CUDA 使用一张 RTX 4090，候选 GCC 11.5.0 / NVCC 12.2.140 / cu122。
以下结果是本轮 clean 主线测量，与上述子代理 dirty-source 测量分别保留。

| 主线检查 | 实际结果 |
| --- | --- |
| 五文件初始化 / copy CPU | 197 passed / 192 CUDA skipped，34.99 秒 |
| 同五文件真实 CUDA | 394 passed / 0 skipped，182.75 秒 |
| 扩大工厂 / 原位 / autograd 八文件 CPU | 203 passed / 153 CUDA skipped，9.25 秒 |
| 同八文件真实 CUDA | 379 passed / 0 skipped，20.62 秒 |
| 原生 storage / view / graph capture 与 GraphReplay 精确节点 CPU | 17 passed / 14 CUDA skipped，12.13 秒 |
| 同原生集合真实 CUDA | 30 passed / 1 既有 expand 内存预算失败，63.35 秒 |
| Canonical CPU core | 322 passed / 42 skipped / 1 existing xfail，两个模式 exit 0 |
| 完整结构门禁 | 1622 passed / 6 skipped，210.17 秒 |

CPU 原生集合的 14 项跳过包括六个 storage CUDA fixture 与八个 CUDA-only graph capture 节点；
不把它们记为 CPU capture 支持。GraphReplay 的原维护节点
`TestGraphReplay::test_a_strided_result_is_replayed_once` 在两设备均通过，CUDA 原 kernel-count
断言保留。

原生 CUDA 唯一失败仍为
[KI-CODEGEN-005](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/known-issues.md#ki-codegen-005-cuda-full-sum-materializes-a-pending-expanded-elementwise-consumer)：
该节点数值正确，实际增长 68,157,440 字节，原预算 16,777,216 字节；与子代理干净远端基线
重现相同，不把整套原生 CUDA 记为全绿，也不修改或 xfail 该断言。
布局、MANIFEST、空白检查和新增两测试文件 Ruff 通过。

本轮未版本化记录位于 `$JITTOR_LAB_ROOT/_state/native-copy-main-integration-d9/`：
三个 `baseline-*.log`、`source-freeze-main.json`、`agent-evidence-readonly-verification.json`、
`cpu.log` / `cuda.log`、`shared-*.log`、`native-*.log`、`core.log`、`structure.log` 与
精确测试选择 `run-unit.sh` / `run-shared.sh` / `run-native.sh`。
