# 缺硬件时怎么把代码改完，以及硬件到手那天跑什么

看板上有两个归并桶（「并入 硬件验收」「并入 多机硬件验收」）和几条混合任务，共同的意思是
「代码可以现在写，验收要等机器」。桶只说了归并，没说**到时候跑什么**、**绿了之后哪几条算被证明**、
**哪些还根本没有可跑的测试**。这一页只回答这三个问题。

各硬件的操作步骤不在这里：Ascend 看 [`docs/guides/ascend-910b.md`](../../docs/guides/ascend-910b.md)，
HCCL 多卡看 [`hccl-on-device-verification.md`](hccl-on-device-verification.md)，Corex 看
[`docs/guides/corex.md`](../../docs/guides/corex.md)。

## 没有硬件时，什么算「代码改完了」

按有效性从强到弱，**能用强的就不要用弱的**：

1. **拿桩 SDK 编真实翻译单元。** `agent/skills/acl-host-syntax-check` 生成一棵 CANN 桩树，对真实
   `.cc` 跑 `g++ -fsyntax-only`。它抓得到签名不匹配、缺头、类型错——这些占了缺硬件时会犯的大多数
   错。**局限写在 skill 里**：`aclnnXxxGetWorkspaceSize` 的实参无法校验（要真 SDK 头），而且
   `-fsyntax-only` 单独不足以抓「把 workspace query 传给 `launch()`」这类错（可变参桩会愉快地转成
   `std::function`），要另加一条比较裸函数指针类型的检查。**ROCm 与 Corex 目前没有等价物**，谁做那两
   条时值得照着建一个。
2. **行为形状的静态合同。** 断言「每个 family 都不自己发 execute 调用」这类**不变量**，而不是
   「共有 65 处调用 `checkRet`」这类**计数**。计数式合同挡不住合法重构：`test_acl_runner_failure_contract`
   曾断言 65 处，8.06 的第一个提交把样板收进共享 launcher 后它就作废，然后红了约 40 个提交没人看见。
3. **桩后端跑通编译与注册。** `setup_fake_cuda_lib` / `tests/backends/corex/test_corex_discovery.py`
   的离线 fake compiler 属这一类：证明发现与注册路径不依赖真设备。
4. **只写上机文档。** 最弱的一档。可以，但**必须同时给出精确的 nodeid 或命令**，否则硬件到手那天
   要重新读一遍代码才知道该跑什么——这正是本页存在的原因。

**不允许的做法**：把缺硬件的用例写成自我 skip 然后当通过。`nox -s cuda`/`npu` 已经用
`JITTOR_TEST_REQUIRE_CUDA=1`／`JITTOR_TEST_REQUIRE_ACL=1` 加 `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`
把这条堵成 fail-closed（合同在 `tests/structure/test_hardware_gate_matrix.py`）。新开硬件 session
照这个形状建。

## 硬件到手那天跑什么

### Ascend 910B3，单卡（CANN）

| | |
| --- | --- |
| 命令 | `nox -s npu`（需 `CANN_SET_ENV` 指向 `set_env.sh`，否则 session 直接 error） |
| fail-closed | `JITTOR_TEST_REQUIRE_ACL=1`、`JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`；session 先跑 `npu-smi info`，再跑一个带精确期望值的 matmul 探针 |
| 覆盖 | `tests/backends/npu/test_acl.py`、`test_acl_torch_compat.py`、`test_aclop.py`、`test_acl_indexing.py`、`tests/ops/test_ops.py`、`tests/core/test_floor_divide.py::TestFloorDivideNPU` 等 |
| 步骤 | [`docs/guides/ascend-910b.md`](../../docs/guides/ascend-910b.md) |

绿了之后可以判定的看板项：**`6.B02`**（65 处 executeOp 失败都抛；静态侧已由
`test_acl_runner_failure_contract` 与 `test_acl_tensor_workspace_contract` 钉住）、**`6.B16`**
（`sync_run=1/0` 两条路径；精确 nodeid 在 Ascend 指南里）、**`8.06`** 的已迁 family（静态合同
88 passed，逐 family 的设备行为要在这里过一遍）、**`4.11`/`4.12`** 的 ACL 注册路径。

### Ascend 910B3，≥2 卡（HCCL）

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 硬件日之前要建 |
| 现有材料 | [`hccl-on-device-verification.md`](hccl-on-device-verification.md) 的四步：第 0 步先证明没静默回落 CPU、A/B 数值对拍、200 轮压竞态、证明时序真的变了 |
| 主机侧已可跑 | `tests/distributed/test_hccl_check_macros.py`、`test_hccl_collective_sync_switch.py`（对桩编译，任何机器都能跑），但**它们不在任何 nox session 里** |

等它的看板项：**`8.02`** 的 HCCL 那半（4 次全设备同步收进了 `JT_HCCL_COLLECTIVE_SYNC` 开关，
默认 `full` 与改前逐字等价，删除要等实机 A/B）、**`10.19`** 的 HCCL 四项（`HcclAllGatherOp::grad()`
目前直接 `LOGf << "not implemented"`，另三个要在多卡上补真实梯度与 CPU 对照）。

### ROCm

| | |
| --- | --- |
| 命令 | `nox -s rocm` |
| 覆盖 | `tests/backends/rocm/test_rocm.py` 与 `tests/distributed/` 的 MPI 组 |
| 静态侧 | `test_rocm_blob_provenance.py`、`test_rocm_library_provider.py`、`test_rocm_native_provider.py` |

等它的看板项：**`4.12`** 的 ROCm 那半（源码转换已移除，改成独立 HIP 后端 provider，要在真卡上确认）。

### Corex / 天数

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 硬件日之前要建 |
| 现有材料 | [`docs/guides/corex.md`](../../docs/guides/corex.md)（48 行，最薄的一份）；`tests/backends/corex/test_corex_discovery.py` 用离线 fake compiler 验发现路径 |

等它的看板项：**`8.14`**（`check()` 只读、路径可配置；正式前置 `4.12` 未满足）。

### 两台机器

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 这本身就是 `10.22` 的内容 |
| 现有材料 | `8.15` 已合入 TCP/SQLite Store 与 NCCL WORLD bootstrap，可在单机多进程下验 rendezvous、超时与失败传播 |

等它的看板项：**`8.15`**（真实两机 collective 对拍）、**`8.16`**（`torchrun --nnodes=2` 跑通 transformers
训练、两机 loss 轨迹一致）、**`8.17`**（跨机带宽微基准、掉线在超时内退出）、**`8.18`**（2 机保存、
4 机加载的续训）、**`10.22`**（两节点 smoke 进 nightly）。

## 硬件日之前还要补的三件事

上面三处「尚无 nox session」是这一页最该被人接手的部分。它们不是文档问题：**没有 session 就意味着
硬件到手那天没有一条命令可跑**，只能靠人照着散文重新拼命令，而那正是最容易漏掉 fail-closed 检查
的时候。三个都照 `npu` session 的形状建即可——强制环境变量、先跑设备探针、`REQUIRE_*` 加
`MIN_EXECUTED` 堵住自我 skip、然后跑清单。

## 这一页怎么保持不过期

`tests/structure/test_deferred_hardware_manifest.py` 钉住三件事：看板上归入硬件验收桶的每条任务都
在本页出现；本页提到的每个测试路径都真实存在；本页声称有 nox session 的每种硬件，`noxfile.py` 里
确实有那个 session（而声称「尚无 session」的，确实没有——这一条是为了让上面那三件事被补上之后，
本页必须跟着改，而不是留着一句过期的「尚无」）。
