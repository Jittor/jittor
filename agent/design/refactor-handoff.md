# 交接：`2.0-refactor` 的当前状态

2026-09-04。上一轮中断留下的 WIP 分支和工作树残留已经全部收尾。本文写给接手的人，只回答三个问题：
**分支停在哪、交接是否干净、下一步从哪开始**。任务本身在 [整改计划](refactor-plan.md)，进度在
[看板](refactor-board.md)，目录终点在 [目标布局](target-layout.md)，本文不重复它们。

下文用 `$LAB` 指代实验根目录下的 `refactor/`（即 `$JITTOR_LAB_ROOT/refactor`）。所有 worktree、缓存与
共用简报都在那里，不在仓库里。

## 0. 从哪起步

| | |
| --- | --- |
| 分支 | **`2.0-refactor`**，不是 `2.0` |
| 工作目录 | **`$LAB/coord`**——`2.0-refactor` 上一个干净的 checkout，有 `AGENTS.md` 与全套文档 |
| 未提交残留 | **无**；14 个相关 worktree 已逐个审计，索引、工作区与未跟踪文件均为空 |
| 共用简报 | `$LAB/AGENT-BRIEF.md`（在仓库外，因为含机器相关路径） |

**不要用主树。** 它在 `2.0` 上，那里看不到本文与 [怎么派活](refactor-dispatch.md)，而且——

> **`2.0` 分支上的 `refactor-board.md` 是写计划时的初版，265 行全标着「待领」。** 照它干活会把已经
> 合并的 150 条重做一遍。看板只认 `2.0-refactor` 上的那份。

主树里还躺着三个属于别人的未提交文件（见第 9 节），切分支会带上它们。

## 1. 当前状态

| | |
| --- | --- |
| 分支 | `2.0-refactor`；当前状态基线 `6220f55a`，后续状态提交接在其上 |
| 相对 `2.0` 的提交 | 当前 1713 个 |
| 提交里出现过的任务号 | 329 个 |
| 看板 | 已合并 **211** / 进行中 **0** / 待领 **61** / 并入其它任务 **13** |
| 沉淀的 skill | `agent/skills/` 下 **34** 个目录 |

**交接清理完成不等于整改完成。** 看板仍有 61 条待领；当前只是把中断留下的易失状态全部转成了主线提交、
明确待领项或已验证的不采用结论。这个分支不是终态。

看板的「已合并」是权威。提交里的任务号更多，是因为一个任务常有补充提交、改判提交与「更正前一个提交」
的提交；数任务要数看板行，不要数提交。

## 2. 五个 `wip/*` 已验证合入并删除

旧 WIP 没有直接合并；每个都 rebase 到当时最新的 `origin/2.0-refactor`，补足验证后才推送：

| 分支 | 提交 | 内容 |
| --- | --- | --- |
| `wip/bindings` | `a39a2f1c`、`ed12fe21` | 2.16 双标量提升与 2.19 析构半项；CPU、真实 CUDA、生成代码和结构门禁通过 |
| `wip/build` | `9a5f4e7c` | 9.21 前半；后半已在正式第一波以 `237d6460` 完成 |
| `wip/codegen` | `1ea90057` | 3.12；CPU 9 项、CUDA 3 项及结构门禁通过 |
| `wip/gates` | `d957e4aa`、`120b004b`、`9329c4f9`、`9f6a80c7` | 0.15 分层证据与 0.16 实测否决；结构门禁通过 |
| `wip/pyops` | `3d40fa9e` | 5.04 参数模型；CPU 全 `tests/nn`、CUDA、Torch-shim 与独立 PyTorch 对拍通过 |

五个 `origin/wip/*` 精确 ref 已经逐一确认不存在；除 build 外，旧提交与 rebase 后提交的 patch-id 均匹配。
build 的 patch-id 差异来自验证后补入的 `JT_SAVE_MEM` 上游适配，不是内容遗失。

## 3. 未提交残留已清理

| worktree | 文件数 | 是什么 |
| --- | --- | --- |
| `$LAB/dist` | 0 | 8.11 以 `a1e769d5` 合入；三 rank MPI、双卡 NCCL 与结构门禁通过 |
| `$LAB/pyother` | 0 | 5.25 工具迁移部分以 `fdf3b759` 合入并通过实际 wheel 审计；剩余四个硬编码资源仍待 3.18 |
| `$LAB/cudabk` | 0 | 8.04 后半以 `9f2e7b80` 合入；cuDNN 8.9.7/9.10.2 定向与 cuDNN 9 全 CUDA 目录通过 |
| `$LAB/gates` | 0 | 0.15 收尾以 `2fd26522` 合入；实测 smoke 仍为 390 s，未达到五分钟，任务保持待领 |
| `$LAB/pyops` | 0 | 原残留实际属于 5.15，不是 5.19；以 `bf0317af` 补齐实现和修前失败测试 |
| `$LAB/bindings` | 0 | `optim_test.tar` 调试产物已删除 |
| `$LAB/mem` | 0 | 看板证据以 `40544f5e`、`8eb33073` 合入；6.C14 后续以 `b0d90d44` 完成显式 share 判断 |

2026-09-03 的只读复核逐个检查了 14 个相关 worktree：全部干净，HEAD 均为远端主线祖先。审计期间
`gatecheck` 在旧提交 `27c4bdeb` 上运行的 CUDA 全门禁随后正常结束：456 passed、1 skipped、退出码 0；
它不是当前 HEAD 的验收结果，当前也没有残留进程。

## 4. 仍值得接着追的线索

中断时各分区正在做的事，按对接手人的价值排序：

1. **compat：FSDP2 的内存增长找到了复现口径。** 最后一步的结论是「每步加一次 `gc.collect()` 内存就稳住
   了」，正在把它收敛成最小复现。这是一条真线索——说明泄漏是引用环而非分配器，接着往下查引用环的持有者
   就行，不必从头找。
2. **gates：0.15 不是缺接线，而是还差真实性能。** smoke 实测 390 s、模型 446 s；不能靠扩大排除清单伪装成五分钟。
3. **8.15：Store 与 NCCL WORLD bootstrap 已做。** 仍缺 HCCL、NCCL 子组 unique id、真实两机 collective 与跨机失败验收。
4. **3.18/9.01 都已有可复用前半。** 普通 CPU kernel 已绕过 asm_tuner；native import 已不再探测 Torch/改 RTLD。完整终点仍见看板。
5. **7.08/7.09/7.11 都已有可复用前半。** 具体提交与剩余验收已写在看板对应行，继续做而不是重写前半。

## 5. 看板状态已经校正

任务表现在有 **0 条「进行中」和 0 条「部分完成」**。正式第一波新增完成 6.C14、4.08、4.09、4.14、
8.08、9.16、9.18、9.20、9.21、9.22；8.15 的两块可独立成果已合入，但完整两机验收未满足，保持待领。

| 分区 | 本波结果 |
| --- | --- |
| `mem` | 6.C14 显式 share 关系完成，CPU/CUDA 聚焦通过 |
| `device` | 4.14 设备迁移、4.08 拷贝/通信流、4.09 五库 SetStream 完成，均有真实双卡证据 |
| `dist` | 8.08 ProcessGroup 完成；8.15 合入 TCP/SQLite Store 与 NCCL WORLD Store bootstrap，仍待真实两机/HCCL/子组 |
| `build` | 9.20、9.18、9.16、9.22、9.21 完成；depfile 与原子产物均有 CPU/CUDA 聚焦证据 |

第二波新增完成 12 项；build 另合入 2 个保持待领的前置切片：

| 分区 | 第二波结果 |
| --- | --- |
| `gates` | 10.10、10.12、0.14、10.03、10.04 完成；均使用 helper/workflow/规则聚焦验收 |
| `pyops` | 6.P26、5.22、5.23、5.01 完成；MaxPool3d 有真实 CUDA 证据，其余用 namespace/structure 聚焦 |
| `coreops` | 2.08、2.03、2.17 完成；生命周期观察者、可重入 epoch 与 OpId 均有 CPU/CUDA 聚焦证据 |
| `build` | 3.18 前置 `cb853074`、9.01 1/N `361d59b2` 已合入；两项都严格保持待领 |

第三波新增完成 8 项：

| 分区 | 第三波结果 |
| --- | --- |
| `coreops` | 2.18 typed lazy registry、6.C08 局部快照反向、2.05 真 0-D 完成，均有 CPU/CUDA 聚焦证据 |
| `pyops` | 5.16 dtype 保真回填、5.09 按输出梯度语义分派融合 kernel 完成 |
| `gates` | 10.07 全声明 dtype、10.11 窄整数设备轴、10.14 notebook topic 参数化完成；10.08 经审计后未机械改写会杀进程的 skip |
| `build` | 3.18 普通 kernel 前置与 9.01 Torch 惰性导入切片保持待领，本波没有伪报完成 |

第四波新增完成 4 项，并合入 2 个严格保持待领的前置：

| 分区 | 第四波结果 |
| --- | --- |
| `codegen` | 3.16 正常替换去异常、3.17 KernelIR 逃逸与原源码行定位完成，CPU/CUDA 聚焦通过 |
| `cudabk` | 8.13 cuTT plan miss 删除全设备同步，真实 GPU4 timeline 与 cache 聚焦通过 |
| `gates` | 10.15 运行条件合同完成；10.06 严格 helper 切片合入但调用迁移/OpInfo 错误矩阵仍待领；10.08 未机械改写进程级 NPU 缺陷 skip |
| `build` | 3.18 已用 clang intrinsic 删除 asm_tuner 链路，但冷编译仅改善 20-22%，未达 50%，保持待领 |

第五波新增完成 3 项，并修复一项波内发现的 3.17 回归：

| 分区 | 第五波结果 |
| --- | --- |
| `coreops` | 2.09 将 Torch 专属梯度状态上移为通用 autograd 策略对象，CPU、GPU2 与结构聚焦通过 |
| `compat` | 7.04 删除 argv 嗅探和 flags 代理，显式幂等激活可查询并接入 2.09 策略；bootstrap 41 项与部署静态 15 项通过 |
| `gates` | 10.06 完成 34 处旧调用迁移，OpInfo 错误输入覆盖 46/227（20.3%），AST legacy 计数归零 |
| `codegen` | 968ae198 修复重复 `#line` 被登记成同名 KernelIR scope 符号的 3.17 回归；原源码定位能力保留 |

第六波新增完成 3 项：

| 分区 | 第六波结果 |
| --- | --- |
| `mem` | 2.06 用内联容量 2 的 SmallVector 与反向下标实现边表 O(1) 访问；C++ 契约、CPU 生命周期与 GPU1 节点通过 |
| `gates` | 10.08 将 6 个安全复现的 FFT 梯度缺陷转为 strict xfail，并验证修复探针会 XPASS 失败；进程级 NPU skip 保留 |
| `pyother` | 10.16 新增固定输入池、warmup 排除和全输出强同步的 `jt.benchmark`；CPU 3 项通过，真实 CUDA 未跑 |

第七波新增完成 2 项，另合入 1 个严格保持待领的性能实验：

| 分区 | 第七波结果 |
| --- | --- |
| `coreops` | 3.06 改为 per-call future，异常与原 JIT key 回主线程，fork 无幽灵 worker；4 个 CPU 聚焦节点通过 |
| `compat` | 7.09 补齐 AutocastCPU/CUDA dtype-policy 分派，CPU 8 passed/1 skipped，真实 PyTorch oracle 一致 |
| `codegen` | 3.22 两级块归约功能通过，但四形状合计慢 1.64%、单形状慢 16.6%；默认不启用，保持待领 |

第八波新增完成 2 项，另合入 1 个严格保持待领的启动前置：

| 分区 | 第八波结果 |
| --- | --- |
| `coreops` | 2.10 封装三套 liveness 计数、下溢与 release graph-check 不变量；C++/CPU 聚焦 2 项通过 |
| `gates` | 10.09 将计划点名的 12 个高频 API 全部归入 OpInfo 或结构化替代节点；structure 1 项、CPU reference 4 项通过 |
| `build` | 9.01 普通 import 不再 setup NCCL/cuTT/MKL，只读 HOME 可用；热 import 仍 1.332s、冷配置仍编译 174 TU，保持待领 |

第九波新增完成 1 项，并合入 2 个严格保持待领的代码组织前置：

| 分区 | 第九波结果 |
| --- | --- |
| `coreops` | 2.11 将执行触发从 VarHolder 构造移到 Executor 显式提交边界；结构/CPU/GPU1 聚焦 5 项通过 |
| `device` | 6.B16 实现 ACL `sync_run` 同步与错误归因，补 910B3 上机文档；本机无 NPU，静态合同通过但保持待领 |
| `gates` | 1.05 将 20 个 C++ 单元测试原样迁到 `src/tests` 并修活跃引用；其余布局工作未做，保持待领 |

第十波合入 3 个严格保持待领的代码组织前置：

| 分区 | 第十波结果 |
| --- | --- |
| `device` | 6.B02 收紧 ACL tensor 状态与 workspace owner/handle 释放合同，补 910B3 步骤；静态 3 项通过，待实机 |
| `bindings` | 2.19 建立 typed 用户错误/内部不变量入口并迁 7 处 dim 边界；跨 pyjt 与结构聚焦 6 项通过 |
| `compat` | 7.03 为 factories 最终拥有的 20 个 API 建立稳定对象身份与保真度元数据；CPU 聚焦 3 项通过 |

第十一波新增完成 1 项，并合入 2 个严格保持待领的代码组织前置：

| 分区 | 第十一波结果 |
| --- | --- |
| `coreops` | 2.12 移出 VarHolder 头内四个 Executor 相关实现，include 依赖改为单向；结构/头语法/CPU 各 1 项通过 |
| `device` | 6.B02 收口 65 处 ACL 执行失败、launcher 查表与 fused 当前算子归因；代码阶段完成，仍待 910B3 |
| `gates` | 0.20 删除 legacy converter service 与活跃导航，结构/布局通过；其余文档树和系统布局未做 |

第十二波合入 3 个严格保持待领的代码组织前置：

| 分区 | 第十二波结果 |
| --- | --- |
| `device` | 8.14 新增 Corex `discover()` 只读探测、`COREX_HOME` 路径配置和离线 fake compiler 合同；2 项通过。正式依赖 4.12 未满足，本机无 Corex/Iluvatar 硬件 |
| `bindings` | 2.19 再迁 10 处 code/numpy/reindex shape/数量用户边界；累计 17 处，结构、C++、跨 pyjt 聚焦通过，其余调用点待分类 |
| `compat` | 7.03 将 `compile/trace/script` 提升为稳定模块级 callable 并登记 approximate fidelity；身份、metadata、CPU 行为 4 项通过；完整 API 迁移仍待领 |

第十三波合入 2 个严格保持待领的前置；8.06 经只读复核后未在轻量波次冒进：

| 分区 | 第十三波结果 |
| --- | --- |
| `bindings` | 2.19 再迁 transpose/fuse_transpose/reshape 共 9 处视图形状边界；结构、三 TU 语法与 Python 负向节点通过 |
| `compat` | 7.03 将 numerical owner 的 `eye` 提升为稳定模块对象并登记 approximate fidelity；CPU 3 项通过 |
| `device` | 8.06 横跨 65 个 executeOp、胖注册表与 Python 属性通道，未形成足够独立的轻量切片，本波未改 |

第十四波合入 1 个严格保持待领的 ACL 代码组织前置：

| 分区 | 第十四波结果 |
| --- | --- |
| `device` | 8.06 在 BaseOpRunner 建立统一 launcher 尾部，先迁 unary family 并保留异步策略；静态 2 项通过，本机无 NPU |

第十五波合入 3 个严格保持待领的前置：

| 分区 | 第十五波结果 |
| --- | --- |
| `device` | 8.06 复用统一 launcher 迁 binary family，保留原同步策略；静态 3 项通过，仍待其余 family 与 910B3 |
| `bindings` | 2.19 迁 ternary 两处 shape/dim 用户边界，累计 28 处；结构、C++ 语法与 Python 负向节点通过 |
| `compat` | 7.03 将 `empty_like` 收回 factory owner 并登记 approximate fidelity；身份、metadata、CPU shape/dtype 5 项通过 |

第十六波新增 1 个严格保持待领的 ACL 前置，并完成 1 个测试门禁任务：

| 分区 | 第十六波结果 |
| --- | --- |
| `device` | 8.06 将 ternary/SWhere family 接入统一 launcher，保留原异步策略；静态合同 4 项通过，本机无 NPU，仍待其余 family 与 910B3 |
| `gates` | 10.19 建立 26 项 backend `grad()` inventory，并补 cuDNN 3D forward/dx/dw CPU 对拍；复核发现 HCCL 四项缺 CPU reference/多卡实机，状态保持待领 |
| `coreops` | 2.13 只读审计确认 Runtime/Context 与 80 个 flags 的全局状态分散在约 30 个 C++ 文件、309 个 Python 消费者，无法本波安全闭环，未改代码 |

第十七波新增 1 个严格保持待领的 ACL 前置：

| 分区 | 第十七波结果 |
| --- | --- |
| `device` | 8.06 将 ReduceSum/Mean/Amax/Amin 四个单步 family 接入共享 launcher，Prod 多轴路径保持原样；静态合同 5 项通过，本机无 NPU |

第十八波新增 2 个严格保持待领的前置：

| 分区 | 第十八波结果 |
| --- | --- |
| `bindings` | 2.19 再迁 broadcast_to 三处 shape 边界，累计 31 处；结构计数、TU 语法与两个负向节点通过 |
| `compat` | 7.03 将 vstack/row_stack/hstack/dstack/column_stack 提升为 numerical 稳定对象并登记 approximate fidelity；7 项 CPU/身份/metadata 对拍通过 |

第十九波合入 3 个严格保持待领的前置：

| 分区 | 第十九波结果 |
| --- | --- |
| `device` | 8.06 将 Cumsum family 接入共享 launcher，保留原同步策略；静态合同 6 项通过，本机无 CANN/NPU，仍待 910B3 |
| `bindings` | 2.19 将 `reinterpret_view` 的 6 个用户 dtype/shape 边界改为 `USER_CHECK`，累计 37 处；结构、TU 语法和 4 个负向节点通过 |
| `compat` | 7.03 将 `movedim/moveaxis` 提升为 numerical 稳定对象并登记 approximate fidelity；5 个身份/metadata/CPU 节点通过 |

第二十波新增 3 个严格保持待领的前置：

| 分区 | 第二十波结果 |
| --- | --- |
| `device` | 8.06 将 ACL MatMul 接入共享 launcher，保留 `cube_math_type` 与同步策略；静态合同 7 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 迁 binary shape mismatch 为 `USER_CHECKop`，累计 38 处；结构、TU 语法和 Python 负向节点通过 |
| `compat` | 7.03 将 `unflatten/swapaxes/swapdims/ravel` 提升为 numerical 稳定对象并登记 approximate fidelity；定点 3 项通过 |

第二十一波新增 3 个严格保持待领的前置：

| 分区 | 第二十一波结果 |
| --- | --- |
| `device` | 8.06 将 Expand family 接入共享 launcher，保持异步策略；静态合同 8 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 迁 setitem 两处 data dimension/shape 用户边界，累计 40 处；结构、TU 语法和两个负向节点通过 |
| `compat` | 7.03 将 copysign/xlogy/heaviside/signbit 提升为 numerical 稳定对象并登记 approximate fidelity；4 项 CPU/身份/metadata 通过 |

第二十二波新增 3 个严格保持待领的前置：

| 分区 | 第二十二波结果 |
| --- | --- |
| `device` | 8.06 将 Floor family 接入共享 launcher，保留同步策略；静态合同 9 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 迁 getitem 三处公开索引/shape 边界，累计 43 处；结构、TU 语法和三个负向节点通过 |
| `compat` | 7.03 将 trace/diag_embed/diagflat 提升为 numerical 稳定对象并登记 approximate fidelity；4 项 CPU/身份/metadata 通过 |

第二十三波新增 3 个严格保持待领的前置：

| 分区 | 第二十三波结果 |
| --- | --- |
| `device` | 8.06 将 NanToNum family 接入共享 launcher，保留属性参数和同步策略；静态合同 10 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 py_converter 的 bool Var slice 边界改为可捕获用户错误，累计 44 处；结构、TU 语法和负向节点通过 |
| `compat` | 7.03 将 float_power 提升为 numerical 稳定对象并登记 approximate fidelity；3 个 CPU/身份/metadata 节点通过 |

第二十四波新增 3 个严格保持待领的前置：

| 分区 | 第二十四波结果 |
| --- | --- |
| `device` | 8.06 将 Triu family 接入共享 launcher，保留 diagonal 与同步策略；静态合同 11 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 device_copy 非法设备号改为 USER_CHECK，累计 45 处；结构、TU 语法和 CUDA 负向节点通过 |
| `compat` | 7.03 将 isclose/allclose 提升为 numerical 稳定对象并登记 approximate fidelity，保留 equal_nan/rtol/atol；3 项 CPU/身份/metadata 通过 |

第二十五波新增 3 个严格保持待领的前置：

| 分区 | 第二十五波结果 |
| --- | --- |
| `device` | 8.06 将 Sigmoid forward family 接入共享 launcher，backward 保持原路径；静态合同 12 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 NumPy object dtype 不支持检查改为可捕获用户错误，累计 46 处；结构、TU 语法和 CPU 负向节点通过 |
| `compat` | 7.03 将 cdist/bucketize 提升为 numerical 稳定对象并登记 approximate fidelity；4 项 CPU/身份/metadata 通过 |

第二十六波新增 2 个严格保持待领的前置；kron 候选因 canonical owner 已存在而取消：

| 分区 | 第二十六波结果 |
| --- | --- |
| `device` | 8.06 将 Transpose/Permute family 接入共享 launcher，保留 axes 构造、资源销毁与同步策略；静态合同 13 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 fused_adamw 四处 TensorList cardinality 检查改为 `USER_CHECKop`，累计 50 处；结构、TU 语法和构造期负向节点通过，CPU/ACL kernel 未执行 |
| `compat` | 7.03 的 kron 经 owner 审计确认 canonical 实现已在 `nn.functional.tensor`，本波未重复包装，临时改动已清理 |

第二十七波新增 3 个严格保持待领的前置：

| 分区 | 第二十七波结果 |
| --- | --- |
| `device` | 8.06 将 Softmax forward family 接入共享 launcher，backward 保持原路径；静态合同 14 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 var_slices 的字符串索引长度边界改为可捕获用户错误，累计 51 处；结构、TU 语法和负向节点通过 |
| `compat` | 7.03 将 nansum/nanmean 提升为 numerical 稳定对象并登记 approximate fidelity；定点 4 项通过 |

第二十八波新增 3 个严格保持待领的前置：

| 分区 | 第二十八波结果 |
| --- | --- |
| `device` | 8.06 将 Embedding forward family 接入共享 launcher，backward 保持原路径；静态合同 15 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 VarHolder::set_data 两处 dtype/size 合同改为可捕获用户错误，累计 53 处；结构、TU 语法和两个负向节点通过 |
| `compat` | 7.03 将 aminmax 与 namedtuple 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第三十波新增 3 个严格保持待领的前置：

| 分区 | 第三十波结果 |
| --- | --- |
| `device` | 8.06 将 Gather forward family 接入共享 launcher，Scatter 保持原路径；静态合同 17 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 random 的公开 `type` 合同改为可捕获用户错误，累计 56 处；结构、TU 语法和负向节点通过 |
| `compat` | 7.03 将 logcumsumexp 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第三十一波新增 3 个严格保持待领的前置：

| 分区 | 第三十一波结果 |
| --- | --- |
| `device` | 8.06 将 ClampTensor family 接入共享 launcher，保留三输入查询和同步策略；静态合同 18 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 py_caller 非字符串返回值检查改为可捕获用户错误，累计 57 处；结构、TU 语法和负向节点通过 |
| `compat` | 7.03 将 quantile 提升为 numerical 稳定对象并登记 approximate fidelity，明确 NumPy CPU fallback；3 项定点通过 |

第三十二波新增 3 个严格保持待领的前置：

| 分区 | 第三十二波结果 |
| --- | --- |
| `device` | 8.06 将 Stack family 接入共享 launcher，保留 tensor-list 与同步策略；静态合同 19 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 unary 操作语义断言改为可捕获用户错误，累计 58 处；结构、TU 语法和负向节点通过 |
| `compat` | 7.03 将 nanquantile 提升为 numerical 稳定对象并登记 approximate fidelity，明确 NumPy CPU fallback；3 项定点通过 |

第三十三波新增 3 个严格保持待领的前置：

| 分区 | 第三十三波结果 |
| --- | --- |
| `device` | 8.06 将 Flip family 接入共享 launcher，保留 axes 构造与同步策略；静态合同 20 项通过，本机无 CANN/NPU，既有 intarray 生命周期问题未改 |
| `bindings` | 2.19 将 CUDA curand dtype/type 两处公开合同改为 `USER_CHECK`，累计 60 处；结构、nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 std_mean/var_mean 提升为 numerical 稳定对象并登记 approximate fidelity，明确 correction/keepdim 限制；3 项 CPU/身份/metadata 通过 |

第三十四波新增 3 个严格保持待领的前置：

| 分区 | 第三十四波结果 |
| --- | --- |
| `device` | 8.06 将 Scatter forward family 接入共享 launcher，Gather 保持原路径；静态合同 21 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN descriptor dtype 合同改为 `USER_CHECK`，累计 61 处；结构与头语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 mv 提升为 numerical 稳定对象并登记 approximate fidelity，保留 out/错误语义；4 项 CPU/身份/metadata 通过 |

第三十五波新增 3 个严格保持待领的前置：

| 分区 | 第三十五波结果 |
| --- | --- |
| `device` | 8.06 将 Concat forward family 接入共享 launcher，SplitWithSize 保持原路径；静态合同 22 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 输入/权重 dtype 一致性改为 `USER_CHECK`，累计 62 处；结构与 TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 addmm 提升为 numerical 稳定对象并登记 approximate fidelity，保留 alpha/beta；3 项 CPU/身份/metadata 通过 |

第三十六波新增 2 个严格保持待领的前置：

| 分区 | 第三十六波结果 |
| --- | --- |
| `device` | 8.06 将 SplitWithSize 接入共享 launcher，保留 splitSize/tensor-list/dim 与同步策略；静态合同 23 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 Cutt transpose axes 长度/掩码两处检查改为 `USER_CHECK`，累计 64 处；nvcc TU 语法和结构通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 mm 提升为 numerical 稳定对象并登记 approximate fidelity，保留 2-D/out 限制；3 项 CPU/身份/metadata 通过 |

第三十七波新增 3 个严格保持待领的前置：

| 分区 | 第三十七波结果 |
| --- | --- |
| `device` | 8.06 将 Nonzero owner 接入共享 launcher，SWhere 保持原路径；静态合同 24 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 66 处；nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 trapz/trapezoid 提升为 numerical 稳定包装并登记 approximate fidelity；4 项 CPU/身份/metadata 通过 |

第三十八波新增 3 个严格保持待领的前置：

| 分区 | 第三十八波结果 |
| --- | --- |
| `device` | 8.06 将 Range family 接入共享 launcher，保留三个 scalar 构造/销毁与同步策略；静态合同 25 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS batched MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 68 处；nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 masked_select 提升为 numerical 稳定对象并登记 approximate fidelity，保留 1-D/out 限制；3 项 CPU/身份/metadata 通过 |

第三十九波新增 3 个严格保持待领的前置：

| 分区 | 第三十九波结果 |
| --- | --- |
| `device` | 8.06 将 Dropout forward family 接入共享 launcher，backward 保持原路径；静态合同 26 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS acc MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 70 处；nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 narrow 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第四十波新增 3 个严格保持待领的前置：

| 分区 | 第四十波结果 |
| --- | --- |
| `device` | 8.06 将 LeakyReLU forward family 接入共享 launcher，backward 保持原路径；静态合同 27 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuSPARSE CSR 两处公开 dtype 合同改为 `USER_CHECK`，累计 72 处；nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 tile 提升为 numerical 稳定对象并登记 approximate fidelity，保留 tuple/list dims；3 项 CPU/身份/metadata 通过 |

第四十一波新增 3 个严格保持待领的前置：

| 分区 | 第四十一波结果 |
| --- | --- |
| `device` | 8.06 将 ArgReduce max/min 两路径统一接入共享 launcher，保留双输出和同步策略；静态合同 28 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuSPARSE COO 两处公开 dtype 合同改为 `USER_CHECK`，累计 74 处；nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 diff 提升为 numerical 稳定对象并登记 approximate fidelity，保留 n/dim/prepend/append；3 项 CPU/身份/metadata 通过 |

第四十二波新增 3 个严格保持待领的前置：

| 分区 | 第四十二波结果 |
| --- | --- |
| `device` | 8.06 将 Random uniform/normal 两分支统一接入共享 launcher，保留 seed/offset 与同步策略；静态合同 29 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 NCCL reduce_scatter 两处公开 shape 合同改为 `USER_CHECKop`，累计 76 处；nvcc TU 语法通过；本分区只分到一张卡，NCCL 负向仍未运行 |
| `compat` | 7.03 将 square 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第四十三波新增 3 个严格保持待领的前置：

| 分区 | 第四十三波结果 |
| --- | --- |
| `device` | 8.06 为 `aclnn.h` 增加 `#pragma once`，补重复包含静态合同；1 passed，本机无 CANN/NPU，仍待 Ascend 910B3 实机 |
| `bindings` | 2.19 将 CUB argsort/arg_reduce 的 offsets dtype 边界改为 `USER_CHECK`，累计 79 处；结构与双 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `pairwise_distance` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份、metadata、CPU p=2/keepdim 三节点通过 |

第四十四波新增 3 个严格保持待领的前置：

| 分区 | 第四十四波结果 |
| --- | --- |
| `device` | 8.06 将 SiLU forward 接入共享 launcher，backward/Swish/SwiGlu 保持原路径；结构合同 31 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution forward 格式边界改为 `USER_CHECK`，累计 80 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `cosine_similarity` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，三节点动态测试因首次 JIT 编译过久终止，未宣称通过 |

第四十五波新增 3 个严格保持待领的前置：

| 分区 | 第四十五波结果 |
| --- | --- |
| `device` | 8.06 将 BatchMatMul 接入共享 launcher，保留 `cube_math_type` 与同步策略；结构合同 32 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution backward-x 格式边界改为 `USER_CHECK`，累计 81 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `svd` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态三节点因首次编译过久终止，未宣称通过 |

第四十六波新增 3 个严格保持待领的前置：

| 分区 | 第四十六波结果 |
| --- | --- |
| `device` | 8.06 将 RotaryPositionEmbedding forward 接入共享 launcher，保留三输入与同步策略；结构合同 33 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution backward-w 格式边界改为 `USER_CHECK`，累计 82 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `svd_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行 |

第四十七波新增 3 个严格保持待领的前置：

| 分区 | 第四十七波结果 |
| --- | --- |
| `device` | 8.06 将 Maxpool forward 接入共享 launcher，保留 descriptors、`poolCeil`、同步策略及 Avgpool/backward 原路径；静态合同 34 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 输入 rank 边界改为 `USER_CHECKop`，累计 83 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录；`broadcast_to` 计数已同步为实际 5 |
| `compat` | 7.03 将 `pca_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行 |

第四十八波新增 3 个严格保持待领的前置：

| 分区 | 第四十八波结果 |
| --- | --- |
| `device` | 8.06 将 Avgpool forward 接入共享 launcher，保留 descriptors、`poolCeil/divisor`、同步策略及 backward/其他 pool owner；静态合同 35 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 84 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `nan_to_num_` 提升为 numerical 稳定 in-place 对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，因既有 NaN/Inf JIT abort 风险未运行动态测试 |

第四十九波新增 3 个严格保持待领的前置：

| 分区 | 第四十九波结果 |
| --- | --- |
| `device` | 8.06 将 TruthReduce all/any 接入共享 launcher，保留双路径异常处理与同步策略；静态合同 36 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 85 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `sparse_coo_tensor` factory 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态测试未运行 |

第五十波新增 3 个严格保持待领的前置：

| 分区 | 第五十波结果 |
| --- | --- |
| `device` | 8.06 将 Conv2d forward 接入共享 launcher，保留 group/bias/descriptor 与同步策略，backward 不变；静态合同 37 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-x dy rank 边界改为 `USER_CHECKop`，累计 86 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `randint_like` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十一波新增 3 个严格保持待领的前置：

| 分区 | 第五十一波结果 |
| --- | --- |
| `device` | 8.06 将 RmsNorm forward 接入共享 launcher，保留 `eps`、双输出与同步策略，gradient owner 不变；静态合同 38 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-w 的 dy rank 边界改为 `USER_CHECKop`，累计 87 处；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `det` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十二波新增 3 个严格保持待领的前置：

| 分区 | 第五十二波结果 |
| --- | --- |
| `device` | 8.06 将 RmsNormGrad 接入共享 launcher，保留多输入、双输出与同步策略，gradient owner 不变；静态合同 39 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 权重 rank 边界改为 `USER_CHECKop`，累计 88 处、四十一组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `inverse` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十三波新增 3 个严格保持待领的前置：

| 分区 | 第五十三波结果 |
| --- | --- |
| `device` | 8.06 将 Softmax backward 接入共享 launcher，保留 `dim` query 与同步策略；静态合同 40 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort 的 x/indexes rank 边界改为 `USER_CHECK`，累计 89 处、四十二组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `take_along_dim` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十四波新增 3 个严格保持待领的前置：

| 分区 | 第五十四波结果 |
| --- | --- |
| `device` | 8.06 将 Embedding backward 接入共享 launcher，保留 `numEmbeddings`、`paddingIdx`、`scaleGradByFreq` 与同步策略；静态合同 41 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort 循环内 x/indexes shape 边界改为 `USER_CHECK`，累计 90 处、四十三组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `log1p` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

本次看板一致性修复：计划与看板任务 ID 均为 279 个，缺失 0、重复 0；7.14 之后的正式任务行统一为五列，补回 `0.22`、`2.24`、`7.19`、`7.20`，移除 8.06 的两列 `8.06 note` 伪行。主线并发补充 `166010a8`（CUB argsort offsets rank）与 `ccbc6132`（`reciprocal`）已记录；第五十五波已将 Dropout backward 的 WIP `c2821634` 收束为主线 `0b149241`/`a12a2fbe`。新增结构门禁 `tests/structure/test_refactor_board_contract.py`，定向 2 passed。

`broadcast_to_op.cc` 当前源码含 5 个 `USER_CHECK`，但 `MIGRATED_DIMENSION_BOUNDARIES` 仍期望 2（另一个 shape map 期望 5），因此 `test_error_categories.py` 仍为 1 failed；待专门门禁任务统一分类/计数，不能在本波掩盖。

第二十九波新增 3 个严格保持待领的前置：

| 分区 | 第二十九波结果 |
| --- | --- |
| `device` | 8.06 将 Roll family 接入共享 launcher，保留 shifts/dims 构造、销毁与同步策略；静态合同 16 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 `reuse_np_array` 的类型与 C-contiguous 两处输入检查改为可捕获用户错误，累计 55 处；结构、TU 语法和两个负向节点通过 |
| `compat` | 7.03 将 pdist 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

## 6. 下一波起点

第五十五波新增 3 个严格保持待领的前置：

| 分区 | 第五十五波结果 |
| --- | --- |
| `device` | 8.06 将 Dropout backward 接入共享 launcher，保留 `scale` query 与同步策略；静态合同 42 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort offsets rank 边界改为 `USER_CHECK`，累计 91 处、四十四组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `reciprocal` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十六波新增 3 个严格保持待领的前置：

| 分区 | 第五十六波结果 |
| --- | --- |
| `device` | 8.06 将 RotaryPositionEmbedding gradient 接入共享 launcher，保留四输入、三输出 query 与同步策略；静态合同 43 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort offsets 长度边界改为 `USER_CHECKop`，累计 92 处、四十五组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `lerp` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十七波新增 3 个严格保持待领的前置：

| 分区 | 第五十七波结果 |
| --- | --- |
| `device` | 8.06 将 Conv2d backward 接入共享 launcher，保留三输出 gradient query、descriptor cleanup 与同步策略；静态合同 44 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB arg-reduce offsets rank 边界改为 `USER_CHECKop`，累计 93 处、四十六组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十八波新增 3 个严格保持待领的前置：

| 分区 | 第五十八波结果 |
| --- | --- |
| `device` | 8.06 将 UpsampleNearest2d backward 接入共享 launcher，保留 output/input-size RAII descriptor 与同步策略；静态合同 45 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB arg-reduce offsets 长度边界改为 `USER_CHECKop`，累计 94 处、四十七组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `log_softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十九波新增 4 个严格保持待领的前置：

| 分区 | 第五十九波结果 |
| --- | --- |
| `device` | 8.06 将 LeakyReLU backward、SiLU backward 分别接入共享 launcher；静态合同 46/47 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN LSTM mode 边界改为 `USER_CHECKop`，累计 95 处、四十八组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `relu` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十波新增 3 个严格保持待领的前置：

| 分区 | 第六十波结果 |
| --- | --- |
| `device` | 8.06 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 非 LSTM mode 边界改为 `USER_CHECKop`，累计 96 处、四十九组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `torch._shape_as_tensor` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十一波新增 2 个严格保持待领的前置；compat 复核未找到安全的第三项：

| 分区 | 第六十一波结果 |
| --- | --- |
| `device` | 8.06 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN `proj_size==0` 边界改为 `USER_CHECKop`，累计 97 处、五十组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 复核剩余 API 后仅 `vmap` 是复杂闭包，其余为已有原生 owner，未产生安全代码提交 |

第六十二波新增 3 个严格保持待领的前置：

| 分区 | 第六十二波结果 |
| --- | --- |
| `device` | 8.06 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 98 处、五十一组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `outer` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十三波新增 3 个严格保持待领的前置：

| 分区 | 第六十三波结果 |
| --- | --- |
| `device` | 8.06 将 LayerNorm forward 接入共享 launcher，保留 `normalizedShape`、`eps`、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 分组通道 shape 边界改为 `USER_CHECKop`，累计 99 处、五十二组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `isin` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十四波新增 3 个严格保持待领的前置：

| 分区 | 第六十四波结果 |
| --- | --- |
| `device` | 8.06 将 LayerNorm backward 接入共享 launcher，保留 `normalizedShape`、`outMask`、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x LSTM mode 边界改为 `USER_CHECKop`，累计 100 处、五十三组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `tensordot` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十五波新增 3 个严格保持待领的前置：

| 分区 | 第六十五波结果 |
| --- | --- |
| `device` | 8.06 将 GroupNorm forward 接入共享 launcher，保留 group/eps、三输出 query 与同步策略；静态合同 53 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x `proj_size==0` 边界改为 `USER_CHECKop`，累计 101 处、五十四组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 将 `repeat_interleave` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十六波新增 2 个严格保持待领的前置；compat 复核未找到安全小切片：

| 分区 | 第六十六波结果 |
| --- | --- |
| `device` | 8.06 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x 非 LSTM mode 边界改为 `USER_CHECKop`，累计 102 处、五十五组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 复核剩余候选仅有复杂 `vmap` 闭包，未强行拆分 |

第六十七波新增 2 个严格保持待领的前置；compat 复核仍未找到安全小切片：

| 分区 | 第六十七波结果 |
| --- | --- |
| `device` | 8.06 将 Avgpool backward 接入共享 launcher，保留 `countIncludePad/divisorOverride`、descriptor cleanup 与同步策略；静态合同 55 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 103 处、五十六组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 复核剩余 API 后仅 `vmap` 为复杂闭包，本波无安全小切片提交 |

第六十八波新增 3 个严格保持待领的前置：

| 分区 | 第六十八波结果 |
| --- | --- |
| `device` | 8.06 将 Maxpool backward 接入共享 launcher，保留 pool descriptors、`poolCeil`、输出处理、cleanup 与同步策略；静态合同 56 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 输入 rank 边界改为 `USER_CHECKop`，累计 104 处、五十七组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 新增 `vmap` owner 迁移设计前置，记录 Runtime 依赖与后续 CPU 验收；未宣称实现完成 |

第六十九波新增 3 个严格保持待领的前置：

| 分区 | 第六十九波结果 |
| --- | --- |
| `device` | 8.06 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 权重 rank 边界改为 `USER_CHECKop`，累计 105 处、五十八组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 细化 `vmap` owner 的可验证契约与验收节点；仅设计前置，未修改 runtime |

第七十波新增 3 个严格保持待领的前置：

| 分区 | 第七十波结果 |
| --- | --- |
| `device` | 8.06 将 Index 接入共享 launcher，保留 index query 与同步策略，SliceV2 未改；静态合同 58 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 分组通道 shape 边界改为 `USER_CHECKop`，累计 106 处、五十九组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 补充 `vmap` owner 提取协议、AST 完成门禁与 `VmapContext` 约束；仅设计前置，未修改 runtime |

第七十一波新增 3 个严格保持待领的前置：

| 分区 | 第七十一波结果 |
| --- | --- |
| `device` | 8.06 将 SliceV2 接入共享 launcher，保留 begins/ends/steps/axes descriptors 与同步策略，Index/其他 owner 未改；静态合同 59 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 107 处、六十组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 连续补充 vmap 设计契约与 unsupported AST 静态门禁；仅设计/门禁前置，未修改 runtime，未宣称实现完成 |

第七十二波新增 3 个严格保持待领的前置：

| 分区 | 第七十二波结果 |
| --- | --- |
| `device` | 8.06 将 StridedSliceAssignV2 接入共享 launcher，保留 gradient memset 分支与 slice descriptor handling；静态合同 60 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-x dy rank 边界改为 `USER_CHECKop`，累计 108 处、六十一组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 补充 vmap context 夹具契约、提取顺序、绑定与回滚步骤；仅设计前置，未修改 runtime |

第七十三波新增 3 个严格保持待领的前置：

| 分区 | 第七十三波结果 |
| --- | --- |
| `device` | 8.06 将 InplaceMaskedScatter 接入共享 launcher，保留 tracked base-to-output memcpy 依赖与同步策略；静态合同 61 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 109 处、六十二组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 7.03 补充 vmap 评审证据清单，覆盖 AST、closure/global、fidelity、聚焦节点与 skip 归因；仅设计前置，未修改 runtime |

第七十四波新增 3 个严格保持待领的前置：

| 分区 | 第七十四波结果 |
| --- | --- |
| `device` | 8.06 将 IndexPutImpl 接入共享 launcher，保留 index tensor-list handling 与同步策略，IndexPutImplAccumulate 未改；静态合同 62 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-w dy rank 边界改为 `USER_CHECKop`，累计 110 处、六十三组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 补充 vmap unsupported 行为矩阵，覆盖 extent/nested dim/非 bool/depth callback/out_dims；仅设计前置，未修改 runtime |

第七十五波新增 3 个严格保持待领的前置：

| 分区 | 第七十五波结果 |
| --- | --- |
| `device` | 8.06 将 IndexPutImpl accumulate 接入共享 launcher，保留 tracked output memset 与 index tensor-list dependency；静态合同 63 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 推理阶段输入 rank 边界改为 `USER_CHECKop`，累计 111 处、六十四组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 明确 vmap 仅做组织重构，不新增 kernel/设备传输/优化，并定义 CPU/CUDA/ACL 分层验收与 skip 归因；仅设计前置 |

第七十六波新增 3 个严格保持待领的前置：

| 分区 | 第七十六波结果 |
| --- | --- |
| `device` | 8.06 将 AdamWList 各项更新接入共享 launcher，保留 fused D2D copy checks 与唯一同步点；静态合同 64 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 输入通道 shape 边界改为 `USER_CHECKop`，累计 112 处、六十五组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 明确 vmap 稳定签名、内部 callback 注入和 unsupported kwargs 拒绝；仅设计前置，未修改 runtime |

第七十七波新增 2 个严格保持待领的前置；2.19 复核未找到安全用户边界：

| 分区 | 第七十七波结果 |
| --- | --- |
| `device` | 8.06 将 FlashAttention forward 接入共享 launcher，保留 prefix/qstart/kvstart RAII descriptors 与同步策略，backward/KV-cache 未改；静态合同 65 passed，本机无 CANN/NPU |
| `bindings` | 复核剩余 CUDA/CUDNN/CUB/NCCL 断言均属内部不变量或后端运行失败，本波无新增安全用户边界迁移 |
| `compat` | 补充 vmap AST 门禁输出契约，定义计数、禁止捕获、unsupported guard 与 fail-closed 证据格式；仅设计前置，未修改 runtime |

第七十八波新增 3 个严格保持待领的前置：

| 分区 | 第七十八波结果 |
| --- | --- |
| `device` | 8.06 本波未新增 ACL family；上一波 FlashAttention backward 的 canonical 记录保持有效，本机无 CANN/NPU |
| `bindings` | 2.19 新增后端内部断言分类文档与结构门禁，1 passed；用户边界累计保持 112 处、65 组，不改运行时错误语义 |
| `compat` | 补充 vmap context/namespace/行为矩阵/提取流程等设计契约，未修改 runtime，未宣称实现完成 |

第七十九波新增 3 个严格保持待领的前置：

| 分区 | 第七十九波结果 |
| --- | --- |
| `device` | 8.06 将 IncrementalFlashAttention 接入共享 launcher，保留 block-table、actual-sequence、cache-view cleanup 与同步策略；静态合同 67 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuFFT jit_prepare unsupported dtype 边界改为 `USER_CHECK`，累计 113 处、六十六组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 明确 vmap 版本兼容、kwargs 策略与退出标准；仅设计前置，未修改 runtime |

第八十波新增 2 个严格保持待领的前置；device 复核确认 KVCacheMemcpy 不适合通用 launcher：

| 分区 | 第八十波结果 |
| --- | --- |
| `device` | 8.06 复核确认标准 workspace/query/execute/sync owner 已迁移完毕；剩余 KVCacheMemcpy 为逐 token 专用 memcpy 路径，本波不改代码 |
| `bindings` | 2.19 将 CUBLAS matmul 输入 rank 边界改为 `USER_CHECK`，累计 114 处、六十七组证据；结构合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见看板 2.19 行末运行记录 |
| `compat` | 补充 vmap 无可变全局、幂等 install、失败回滚与资源释放门禁；仅设计/门禁前置 |

第八十一波新增 2 个严格保持待领的前置；device 复核无安全标准 owner：

| 分区 | 第八十一波结果 |
| --- | --- |
| `bindings` | 2.19 将 cuDNN RNN 权重查询内部断言纳入分类文档与结构门禁；1 passed，不改变用户错误语义或累计数 |
| `compat` | 7.03 固定 vmap 首门禁夹具（seed=17、简单映射与 nested bool shape）；仅设计前置，未修改 runtime |
| `device` | 8.06 复核确认标准 launcher owner 已穷尽，剩余 KVCacheMemcpy 等专用路径不纳入通用 launcher |

第八十二波新增 2 个严格保持待领的前置；device 继续保持边界说明：

| 分区 | 第八十二波结果 |
| --- | --- |
| `bindings` | 2.19 新增后端内部断言分类文档与 CUDNN RNN bias/descriptor 门禁，结构门禁 1 passed；不改变用户错误语义，用户边界累计保持 114 处 |
| `compat` | 补充 vmap 固定数据、context 夹具、handoff 证据模板与可验证契约；未修改 runtime，未宣称实现完成 |
| `device` | 8.06 明确标准 launcher owner 已迁移完毕，剩余 KVCacheMemcpy 为逐 token 专用 memcpy 路径，本波无代码提交 |

第八十三波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十三波结果 |
| --- | --- |
| `bindings` | 2.19 补充 CUB 状态内部断言分类门禁与说明文档；不改变运行时语义，用户边界累计保持 114 处 |
| `compat` | 7.03 补充 vmap AST 实现草案，定义 module/install/nested 计数与 binding 行号提取伪代码；仅设计前置，未修改 runtime |
| `device` | 8.06 只读确认标准 launcher owner 已穷尽，KVCacheMemcpy 等专用 memcpy 路径不纳入通用 launcher |

第八十四波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十四波结果 |
| --- | --- |
| `bindings` | 2.19 将 CUBLAS 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计 |
| `compat` | 7.03 细化 vmap context 泄漏 AST 门禁，覆盖默认参数、注解、decorator、closure 白名单与模块全局扫描；仅设计/门禁前置 |
| `device` | 8.06 复核确认标准 workspace/query/execute/sync owner 已全部迁移，KVCacheMemcpy 等专用路径不纳入通用 launcher |

第八十五波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十五波结果 |
| --- | --- |
| `bindings` | 2.19 将 CUDNN 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计 |
| `compat` | 7.03 补充 vmap fidelity registry 静态门禁，校验 identity、approximate level、context/backend detail 与重复 install |
| `device` | 8.06 只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher |

第八十六波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十六波结果 |
| --- | --- |
| `bindings` | 2.19 收束后端内部断言分类说明与门禁，覆盖 CUDNN/CUBLAS/CUB 状态路径；内部断言门禁 1 passed，不改变用户边界或运行语义 |
| `compat` | 7.03 补充 vmap 发布检查清单，覆盖 clean import、重复 install identity、回滚与无缓存产物；仅设计/门禁前置 |
| `device` | 8.06 复核确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher |

第八十七波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十七波结果 |
| --- | --- |
| `bindings` | 2.19 精确约束 Cutt wrapper 两处返回码内部断言并纳入分类门禁；结构合同 1 passed，不改变用户边界 |
| `compat` | 7.03 补充 vmap 变更控制与 reviewer sign-off；仅设计前置，未修改 runtime |
| `device` | 8.06 复核确认标准 launcher owner 已穷尽，KVCacheMemcpy 保持专用 `aclrtMemcpyAsync` 路径 |

第八十八波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十八波结果 |
| --- | --- |
| `bindings` | 2.19 将 CUDNN RNN descriptor 内部断言纳入分类门禁；1 passed，不改变运行语义，用户边界累计保持 113 处 |
| `compat` | 7.03 补充 vmap metadata 兼容契约，固定 `_jittor_vmap_base`/`_jittor_vmap_specs` 身份、形状与嵌套层级；仅设计前置 |
| `device` | 8.06 复核确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher |

第八十九波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第八十九波结果 |
| --- | --- |
| `bindings` | 2.19 精确约束 cuDNN plan `ASSERT(ok)` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义 |
| `compat` | 7.03 补充 vmap `in_dims/out_dims` 的 int、None、tuple/list、负轴归一化矩阵；仅设计/门禁前置 |
| `device` | 8.06 复核确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径 |

第九十波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第九十波结果 |
| --- | --- |
| `bindings` | 2.19 补充 CUB 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计 |
| `compat` | 7.03 补充 vmap 嵌套 metadata 深度契约，固定 specs 追加、base identity 与 batch shape 顺序；仅设计前置 |
| `device` | 8.06 复核确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径 |

第九十一波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第九十一波结果 |
| --- | --- |
| `bindings` | 2.19 收紧 CUBLAS/CUDNN 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变用户边界 |
| `compat` | 7.03 补充 vmap kwargs 兼容矩阵，明确归一化、unsupported 与未知 kwargs TypeError；仅设计前置 |
| `device` | 8.06 只读确认标准 ACL launcher owner 已全部处理，KVCacheMemcpy 保持专用路径 |

第九十二波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第九十二波结果 |
| --- | --- |
| `bindings` | 2.19 将 CUB 测试 CUDA 状态断言纳入精确计数门禁；结构门禁 1 passed，不改变用户边界或运行语义 |
| `compat` | 7.03 补充 vmap 标量/zero-dim 输出契约，固定 singleton 归一化、nested batch 轴与 `out_dims` 形状；仅设计前置 |
| `device` | 8.06 只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径 |

第九十三波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第九十三波结果 |
| --- | --- |
| `bindings` | 2.19 精确约束 cuDNN convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义 |
| `compat` | 7.03 补充 vmap autograd 契约，明确 loop/stack 梯度、bool fast path 非微分边界与 CPU gradient 节点草案；仅设计前置 |
| `device` | 8.06 只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径 |

第九十四波新增 2 个严格保持待领的前置；device 继续保持无安全 owner：

| 分区 | 第九十四波结果 |
| --- | --- |
| `bindings` | 2.19 精确约束 cuDNN 3D convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义 |
| `compat` | 7.03 补充 vmap 并发契约，明确 re-entrant 调用、context 生命周期隔离及线程安全 probe；仅设计前置 |
| `device` | 8.06 只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径 |

第九十五波新增 2 个严格保持待领的前置，并完整关闭 10.13：

| 分区 | 第九十五波结果 |
| --- | --- |
| `gates` | 10.13 `-m not slow` 快门禁由 `821bb6ba` 完整关闭；AST 节点 1 passed，native/torch smoke 调用与 `SLOW_FILES` 覆盖均有断言 |
| `bindings` | 2.19 将 CUDNN backward-x `best_algo_idx!=-1` 内部断言纳入精确计数门禁；结构门禁 1 passed，不改变用户边界 |
| `compat` | 7.03 补充 vmap 资源边界/取消契约，未修改 runtime，仍待整体实现 |

第九十六波新增 1 个已完成任务并保留 0.15 专项：

| 分区 | 第九十六波结果 |
| --- | --- |
| `gates` | 10.05 skip reason bucket/CI summary/`other>0` fail-closed 已由 `1a423a16`/`f7f33f5b` 完整关闭；合成结构测试 2 passed |
| `gates` | 0.15 RingBuffer GIL/有界等待完成只读审计，尚未提交代码；需后续专项实现和 smoke 重测 |

0.15 仍未完成：smoke 实测约 390s、预算模型约 446s，计划要求 `<300s`；RingBuffer GIL/timeout 修复及两模式 nodeid/makespan 重测仍待专项处理。

第九十七波完成 0.15 可行性审计但未提交代码：

| 分区 | 第九十七波结果 |
| --- | --- |
| `gates` | 两个长 compat 文件无法在保持 nodeid/import 语义与 `loadfile` 覆盖的前提下安全拆分降时；现有双模式与 worker 配置保持不变 |
| `bindings` | RingBuffer 需要 timed-wait、GIL 安全拆分、pyjt 参数和 Dataset 轮询联动，不能预先加不完整门禁 |
| `device` | 未参与代码修改 |


第九十八波完成 0.15 第一阶段代码，但任务仍待完整验收：

| 分区 | 第九十八波结果 |
| --- | --- |
| `device` | `23814b9a` 增加 Linux `RingBuffer::wait_pop_for`，仅在纯等待阶段释放 GIL，并将 `test_children_died` 改为有界 timeout；worker-death 聚焦节点 1 passed（约 106s） |
| `bindings` | 审查确认共享 condvar/pyjt ABI、GIL 异常边界和 Dataset worker 轮询要求；未新增代码 |
| `gates` | 确认 `SLOW_FILES` 未移除，smoke 仍约 390s/预测 446s，必须重测 `<300s` 和 nodeid 集合后才能关闭 0.15 |

第九十九波完成 0.15 RingBuffer 语义修正，但任务仍待完整验收：

| 分区 | 第九十九波结果 |
| --- | --- |
| `device` | `d3f4853e` 固定 `wait_pop_for` 单次绝对 deadline，处理 EINTR/非零状态，并在恢复 GIL 后保留原异常；`ring_buffer.cc` 与 `py_ring_buffer.cc` TU 语法检查通过 |
| `bindings` | 审查确认固定 timeout 仍需收敛为 Dataset 专用轮询/兼容 API，并补 stop、延迟 producer、正常收发测试 |
| `gates` | `SLOW_FILES`、smoke `<300s` 与 native/torch nodeid 集合尚未重测，0.15 继续待领 |

第九十九波修正版补充：

| 分区 | 修正版结果 |
| --- | --- |
| `device` | `876ec09c` 恢复通用 `pop()` 无限等待/兼容行为，新增 Dataset 专用 `pop_for(timeout_ms)`；单次 deadline、伪唤醒/EINTR/stop/其他异常与 GIL 边界已修正 |
| `gates` | 协议合同与 killed-worker 聚焦节点各 1 passed（killed-worker 19.72s）；`SLOW_FILES` 和 smoke `<300s` 尚未重测，0.15 继续待领 |

第一百波补充关闭 5.21：

| 分区 | 第一百波结果 |
| --- | --- |
| `pyother` | 5.21 已由 `3cd1a614` 完整实现并关闭：`_install_order.SEQUENCE` 声明十步 monkeypatch 安装顺序，`record/verify` 在运行时拒绝乱序、重复和缺失；`jt.sum` 与 `Var.sum`/`mean` 共用 full-reduce 路径。提交内 `tests/core/test_install_order.py` 17 项、`tests/structure/test_install_order.py` 6 项通过。 |

### 2026-09-04 第一百零一波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `ba2f4077` 将 cuBLAS matmul 内维不匹配改为可捕获的 `USER_CHECKop`，新增负向结构合同；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `94df46f7` 将 `complex`、`view_as_complex`、`view_as_real` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；CPU identity/metadata/value 定向 2 passed。7.03 其余 family 仍待领。 |
| `device` | 只读确认 ACL 标准 launcher owner 已穷尽；KVCacheMemcpy 是逐 token `aclrtMemcpyAsync` 专用路径，不安全套用通用 launcher，本波无代码提交。 |
| `gates` | `876ec09c` 后独立 Dataset worker 监管两个 nodeid 在临时缓存下 2 passed/65.68 s；完整 smoke 仍约 390 s，0.15 保持待领。 |

### 2026-09-04 第一百零二波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `cb0a4e77` 将 cuBLAS batched matmul 三项输入 rank 检查改为可捕获的 `USER_CHECKop`，补独立 rank ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `dbe72f0f` 将 `torch.polar` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；identity/metadata 与 CPU NumPy 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `6.B16` 已有 `BaseOpRunner::syncRun`、静态合同与 Ascend 910B3 上机文档；本机无 NPU，实机同步/异步与失败归因仍待验证，本波无新提交。 |
| `gates` | Dataset worker 监管两个 nodeid 独立缓存下 2 passed/65.68 s；完整 smoke 仍约 390 s，0.15 保持待领。 |

### 2026-09-04 第一百零三波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `e38cce97` 将 cuBLAS acc matmul 两项输入 rank-2 检查改为可捕获的 `USER_CHECKop`，补独立 rank ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `6fdb6120` 将 `real`/`imag`/`conj`/`angle`/`is_complex`/`abs` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；complex accessor 定向 2 passed。7.03 其余 family 仍待领。 |
| `device` | `448aa10a` 删除 ACL `utils` 中无消费者的 `op_idx_map` 定义/声明，保留显式 reduce dispatch；ACL 静态合同 68 passed，未做 NPU 实机验证。 |
| `gates` | 10.19 backend grad manifest 静态合同 2 passed；HCCL 四项仍明确为 NPU/unsupported hardware route，缺 Ascend 910B3 多卡实测，任务保持待领。 |

### 2026-09-04 第一百零四波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `0e275f14` 将 cuBLAS batched matmul 内维 `m == m_` 检查改为可捕获的 `USER_CHECKop`，补独立 inner-dim ledger；定向 4 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `84f29d9b` 将 `hann_window`/`stft` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；signal family 定向 3 passed。7.03 其余 family 仍待领。 |
| `device` | `aclnn.h` 的 `#pragma once` 已由 `1e8e90c6` 完成；ACL 标准 launcher 与 `op_idx_map` 子项已有证据，本波无新代码，NPU 实机仍待。 |
| `gates` | 本波复核 0.15 未改 `SLOW_FILES`；Dataset worker 聚焦测试通过但完整 smoke 仍约 390 s，任务保持待领。 |

### 2026-09-04 第一百零五波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `7d366087` 将 cuBLAS batched matmul batch 维逐轴 shape 检查改为可捕获的 `USER_CHECKop`，补 axis/shape 错误信息与独立 ledger；定向 5 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `f848a6a7` 将 `torch.equal` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；shape/value/empty CPU 对拍 3 passed。7.03 其余 family 仍待领。 |
| `device` | ACL `AclOpFunctions` 40 个 typed `std::function` 字段与构造/注册/owner 消费点耦合，类型擦除必须协同迁移；本波无代码提交，避免半改 dispatch。 |
| `gates` | 10.19 manifest 静态合同仍 2 passed；HCCL 四项缺 Ascend 910B3 多卡与 CPU 对照，0.15 smoke 也未达 300 s，相关任务保持待领。 |

### 2026-09-04 第一百零六波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `cc9ed2bc` 将 cuSPARSE CSR `xs==os`、`A_col==xs[0]` 两项 shape 边界改为可捕获 `USER_CHECKop`，补独立 ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `258be343` 将 `kron`/`logsumexp` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；identity/metadata 与 CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `15c86886` 新增 ACL 结构迁移边界文档与静态合同（1 passed），明确类型擦除、属性 data、descriptor cache 的协同范围和 910B3 验收边界；未改运行时。 |
| `gates` | 10.19 backend grad manifest 静态合同 2 passed；HCCL 仍需 Ascend 910B3 多卡 CPU 对照，0.15 smoke 仍约 390 s，任务保持待领。 |

### 2026-09-04 第一百零七波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `a831fbd6` 将 cuSPARSE COO 两项 shape 边界改为可捕获 `USER_CHECKop`，补独立 ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `f20bb4ff` 将 `all`/`any` reduction wrapper 提升为 numerical 模块级稳定对象，固定 builtins 引用并登记 approximate fidelity；axis/keepdims CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `7a35e8e7` 记录 ACL 属性 data 通道候选（`triu.diagonal`、`softmax.dim`、`flip.axes`）的协同迁移边界；静态合同 1 passed，未改运行时，NPU 实机仍待。 |
| `gates` | 本波未扩大 `SLOW_FILES`；0.15 worker 聚焦测试通过但完整 smoke 仍约 390 s，10.19 HCCL route 仍缺 910B3 多卡 CPU 对照。 |

### 2026-09-04 第一百零八波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `55a1f481` 将 cuBLAS matmul 输入 b 的 rank-2 检查改为可捕获 `USER_CHECKop`，补独立 ledger；定向 4 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `9954b4ed` 将 `tensor_split`/`take` 提升为 numerical 模块级稳定对象并登记 approximate fidelity；split/take CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `296e0837` 补充 ACL 属性 data、descriptor cache、type-erasure 候选排序与 910B3/CANN 上机验收合同；静态合同 1 passed，未改运行时。 |
| `gates` | 本波未扩大 `SLOW_FILES`；0.15 smoke 仍约 390 s，10.19 HCCL route 仍缺 910B3 多卡 CPU 对照。 |

### 2026-09-04 第一百零九波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `354ec6f1` 将 cuTT transpose 0-D 输入检查改为可捕获 `USER_CHECK`，补独立 rank ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `aa3ca532` 将 `index_copy` 提升为 numerical 模块级稳定 owner，保留 `Var.index_copy_` 原地语义并登记 approximate fidelity；CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `softmax.dim`/`triu.diagonal` 均依赖统一 data-channel schema、C++ attr 解码和 cache-key 契约，无法安全独立迁移；本波无代码提交。 |
| `gates` | 本波未扩大 `SLOW_FILES`；0.15 完整 smoke 仍约 390 s，10.19 HCCL route 仍缺 Ascend 910B3 多卡 CPU 对照。 |

### 2026-09-04 第一百一十波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `26ee6bee` 将 cuDNN RNN descriptor 两处 GRU mode 检查改为可捕获 `USER_CHECKop`，补独立 mode ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `d3e33b0e` 将非原地 `index_put` 提升为 numerical 模块级稳定 owner，保留 `index_put_` 原地语义并登记 approximate fidelity；CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | ACL 仍缺统一 data-channel C++ 解码入口；`_code.py`、Python attr、C++ `OpAttr` 与 cache-key 需协同迁移，本波无运行时代码提交。 |
| `gates` | 本波未改变 0.15 `SLOW_FILES` 或 10.19 HCCL hardware-only 状态；现有聚焦合同继续通过，完整硬件/性能验收仍待。 |

### 2026-09-04 第一百一十一波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `038819e6` 将 cuBLAS acc matmul 内维 `m == m_` 检查改为可捕获 `USER_CHECKop`，补独立 ledger；定向 4 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `64f4ecfa` 建立 `torch.autocast` numerical installer 模块级 owner alias，登记 approximate fidelity；CPU context enable/restore 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `3cb0da4e` 形成 ACL data-channel schema 草案（版本、operator scope、标量/向量类型、确定性 cache key、统一解码入口目标）和 910B3 验收边界；静态合同 1 passed。 |
| `gates` | 本波未改变 0.15 性能门槛或 10.19 HCCL hardware-only 路线；相关完整验收仍待。 |

### 2026-09-04 第一百一十四波

| 分区 | 结果 |
| --- | --- |
| `gates` | `151c5856` 将 `cpu` 加入 `nox.options.sessions` 默认列表，新增 AST 合同确认默认数值门禁；定向 1 passed。`10.02` 已完整关闭，待领数减少一条。 |

### 2026-09-04 第一百一十五波

| 分区 | 结果 |
| --- | --- |
| `gates` | `5501d0b6` 增加稳定 `nox -s full` 完整 CPU/nightly 入口，CPU workflow 已切换到该入口；AST 调度合同 2 passed，`10.01` 已完整关闭。 |

### 2026-09-04 第一百一十六波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 整卡审计未发现可完整关闭项：2.19 仍缺全量 ASSERT/CHECK/LOGf 分类闭环；8.12 仍缺五处算子缓存键/POD 哈希验收。本波无代码提交。 |
| `compat` | 整卡审计确认 7.03、7.05、7.07、7.08、7.11、7.12、7.13、7.18、7.19、7.20 均有未完成语义、布局或硬件前置；本波无代码提交。 |
| `device` | ACL 8.06 的 data-channel 仍缺统一 C++ decoder；0.15 仍约 390 s，0.22 需 CUDA 多卡，6.B02/6.B16/10.19 需 910B3/HCCL 实机。本波无代码提交。 |
| `gates` | 没有可在当前 CPU/静态环境完整关闭的新卡；看板待领保持 73，未伪造状态。 |

### 2026-09-04 第一百一十七波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `c329a822` 将 cuDNN backward-w format 用户边界统一为可捕获 `USER_CHECKop`，补独立 format ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `24637c96` 将 `vmap` 暴露为 numerical 模块级稳定 wrapper，保留 specialized runtime 转发并登记 approximate fidelity；CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `25c5ffed` 明确 ACL C++ `decode_acl_data` 接口草案、User/Internal 错误映射及 canonical cache key 规则；静态合同 2 passed，未改运行时。 |
| `gates` | 本波没有可完整关闭的新卡；0.15 仍约 390 s，HCCL/ACL 仍待硬件验收。 |

### 2026-09-04 第一百一十八波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 8.12 仍有 cuDNN 2D/3D 全局 `JK` 字符串缓存键，10.17 仍缺低开销最近算子环形缓冲与 Python 行号；两卡均不能整卡关闭。 |
| `compat` | 10.18/10.20/10.21 分别缺核心属性测试、受支持内省 API、import layering lint；7.x 聚合项仍有语义/布局前置。 |
| `device` | 0.20/0.22/9.01/9.07/9.19 分别受跨目录布局、CUDA 多卡、冷启动、2.22 前置和整体打包布局阻塞。 |
| `gates` | 本波无可在 CPU/静态环境完整关闭的新卡；0.15 仍约 390 s，ACL/HCCL 仍待 910B3 实机。 |

### 2026-09-04 第一百二十三波

| 分区 | 结果 |
| --- | --- |
| `compat` | `7.05` 整卡审计确认 namespace 事务仅覆盖部分模块树，仍缺全局安装锁、失败可重试/可查询以及 `os.environ`/flags 全量回滚；本波无代码提交。 |
| `gates` | 其余待领卡未出现可在当前 CPU/静态环境完整关闭的候选，待领保持 73。 |

### 2026-09-04 第一百二十四波

| 分区 | 结果 |
| --- | --- |
| `compat` | `7.05` 复核确认 namespace snapshot 不能替代全局安装锁、失败状态 API 或 `os.environ`/flags 全量回滚；本波无代码提交。 |
| `gates` | 没有新增可完整关闭的任务，待领保持 73；下一步需实现完整事务边界后再验收。 |

### 2026-09-04 第一百二十五波

| 分区 | 结果 |
| --- | --- |
| `compat` | `3c8b46f3` 使 optional install 失败 warn-once、可查询、可重试；`44272e89` 新增带 RLock 的 reversible `InstallTransaction` 和合成 module/env/flags/meta_path 测试 2 passed。尚未接入所有 installers，7.05 仍待领。 |
| `gates` | 本波代码前置已推送，但整卡验收仍缺全量 installer mutation ledger；待领保持 73。 |

### 2026-09-04 第一百二十六波

| 分区 | 结果 |
| --- | --- |
| `compat` | `fa2027e4` 将 `InstallTransaction` 的进程级 RLock 与 namespace undo 接入 `compat.torch.install`；`4f66701e` 明确 flags/env/import/meta_path/module-patcher 尚未纳入回滚，并补边界合同 1 passed。7.05 仍待领。 |
| `gates` | 事务前置测试保持通过；完整 installer mutation ledger 仍缺，未关闭 7.05。 |

### 2026-09-04 第一百二十七波

| 分区 | 结果 |
| --- | --- |
| `compat` | `ec720cd8` 的整表 global snapshot 实现因会覆盖外部 env/finder 且 flags 不完整，已由 `9a674001` 撤回；`1aa640cf` 补充显式 flags/env allowlist、owner-aware restore 要求。7.05 仍待领。 |
| `gates` | 本波没有新增运行时代码；事务回滚仍需逐写 mutation ledger、冲突硬失败和全流程失败注入测试。 |

### 2026-09-04 第一百二十八波

| 分区 | 结果 |
| --- | --- |
| `compat` | `9949fa10` 修复 completed-install namespace 冲突异常路径的 RLock 泄漏；回归 `test_install_context.py` 定向 2 passed。完整 installer mutation ledger 仍未接入，7.05 保持待领。 |
| `gates` | 本波修复只涉及锁生命周期和失败测试，不改变 0.15/硬件验收状态。 |

### 2026-09-04 第一百二十九波

| 分区 | 结果 |
| --- | --- |
| `compat` | `c8b993b4` 让 transaction rollback 校验事务写入的 owner 值；外部改写时抛 `TransactionConflict`，合成事务测试 3 passed。完整 installer mutation ledger 仍待接入，7.05 保持待领。 |
| `gates` | 本波完成 owner-aware 回滚安全子项，未改变整卡状态。 |

### 2026-09-04 第一百三十波

| 分区 | 结果 |
| --- | --- |
| `compat` | `75793c04` 将 distributed installer 的 JT_NCCL_*、`use_nccl/use_mpi` 和 `jt.flags.use_cuda` 写入纳入 mutation ledger；`py_compile`/`diff-check` 通过。其他 installer 写入口仍待迁移，7.05 不关闭。 |
| `gates` | `783699cd` 补 child-env 隔离和 distributed 环境写入静态合同 2 passed；不替代父进程回滚验收。 |

### 2026-09-04 第一百三十一波

| 分区 | 结果 |
| --- | --- |
| `compat` | `f8f838b2` 将 distributed 环境结构合同同步到 `tx.mutate_env/mutate_flag` 新协议，避免 direct-write 旧断言误报；结构 2 passed，事务相关定向 5 passed。7.05 仍待领。 |
| `gates` | 本波仅修正测试合同，未扩大回滚覆盖范围或改变整卡状态。 |

### 2026-09-04 第一百三十二波

| 分区 | 结果 |
| --- | --- |
| `compat` | `1eb7ec07` 修复 `mutate_env` 的规范化值记录：整数/布尔等输入先转字符串再登记，失败回滚不再误报 owner 冲突；事务定向 6 passed。7.05 仍待全部 installer mutation 接入。 |
| `gates` | 本波发现并修复一个真实回滚边界 bug；未改变整卡状态。 |

### 2026-09-04 第一百三十三波

| 分区 | 结果 |
| --- | --- |
| `compat` | `c49efb80` 更新 `test_install_context.py`，移除已被显式 activation API 取代的 `wrap_flags` patch；完整安装上下文测试 `20 passed`。这是测试兼容修复，7.05 仍待全量 mutation 接入。 |
| `gates` | `test_torch_install_state_boundary.py` 等结构合同通过；组合结构长测受 30s 轻量超时截断，不作为成功或失败证据。 |

### 2026-09-04 第一百三十四波

| 分区 | 结果 |
| --- | --- |
| `compat` | `60197b81` 将 factories/tensor installer 的 `jt.flags.use_cuda` 写入接入 transaction helper；`d9d063a5` 将 core installer 安装期 `use_cuda` 写入接入 ledger 并补失败回滚测试（事务定向 5 passed）。nn 的 `.to()` 写入确认是运行时用户语义，不纳入 install transaction。7.05 仍待全部 installer 写入口。 |
| `gates` | 相关 `py_compile`/`diff-check` 通过；本波未改变整卡状态。 |

### 2026-09-04 第一百三十五波

| 分区 | 结果 |
| --- | --- |
| `compat` | `87ca0a82` 新增 owner-aware `InstallTransaction.mutate_attr`，将 transformers runtime guard 与 torchmetrics fastpath 两处 `builtins.__import__` 写入接入 ledger；`7af13605` 补 utilities import-hook 回滚/外部替换冲突测试（6 passed）。无事务时保持直写，7.05 仍待其他 installer mutation。 |
| `device` | `sys.meta_path` permissive finder 需同时恢复 finder 身份、插入索引和 allowlist 增量，暂不做半改；本波无 device 提交。 |
| `gates` | `py_compile`/`diff-check` 通过，未改变整卡状态。 |

### 2026-09-04 第一百四十一波

| 分区 | 结果 |
| --- | --- |
| `compat` | `9f154035` 将 vLLM `_ArmOnFirstImport` finder 插入登记进 transaction，并让 `_install_optional_vllm` 传递 active transaction；vLLM registry callback/extension module 发布和 shim runtime 全局写入仍待。 |
| `device` | external backend source import 有局部 sys.path/sys.modules 回滚，但 registry/hints/entry-point 状态未接入 InstallTransaction；本波无代码。 |
| `gates` | finder/registry 现有结构合同通过，整卡 7.05 仍待完整 mutation 覆盖。 |

### 2026-09-04 第一百四十二波

| 分区 | 结果 |
| --- | --- |
| `compat` | `367716a7` 将 external backend 的 `_BACKENDS`、`_BACKEND_HINTS`、`_ENTRY_POINTS_LOADED` 注册状态接入 transaction，integrations 传递 active tx；source import 的 sys.path/sys.modules 仍由 resolver 局部管理，7.05 不关闭。 |
| `device` | source candidate 失败恢复仍是整 resolver 快照，不能覆盖并发外部增量；需子进程隔离或 owner-aware 增量记录。 |
| `gates` | `py_compile`/`diff-check` 通过，尚未补 backend object 内部 mutation 和外部 registry 失败注入。 |

### 2026-09-04 第一百四十三波

| 分区 | 结果 |
| --- | --- |
| `compat` | `shim.runtime.activate` 的 sys.path、sys.modules、flags、递归 compat install 和 extension build 不在同一 InstallTransaction 生命周期；不能安全半接入，需先设计独立 ActivationTransaction。 |
| `device` | child_env 只隔离扩展子进程，不能回滚父进程路径/模块状态；source candidate 继续使用局部锁和快照。 |
| `gates` | 本波无代码提交，明确 path/module/flag owner token、冲突硬失败和 child/retry 验收要求。 |

### 2026-09-04 第一百四十四波

| 分区 | 结果 |
| --- | --- |
| `compat` | `c204f4e9` 扩展 InstallTransaction 为 ActivationTransaction，加入 owner-aware `mutate_path` 与 `publish_module`，并补 path/module 回滚合同；事务定向 11 passed。尚未接入 `shim.runtime.activate`，7.05 不关闭。 |
| `device` | runtime activation 仍跨扩展构建/递归安装，需外层事务统一生命周期；child build 继续使用显式环境隔离。 |
| `gates` | 本波协议和合同通过，未改变完整 activation 验收状态。 |

### 2026-09-04 第一百四十五波

| 分区 | 结果 |
| --- | --- |
| `compat` | `4d5b8e61` 将 `shim.runtime._activate_once(_composition=True)` 的 Torch install 与 `sys.modules["torch"]` 发布包入 ActivationTransaction，异常 rollback、成功 commit；普通 activation 的 path/build/no_grad 仍未覆盖，7.05 不关闭。 |
| `device` | composition-only 不启动 child build，普通 activation 仍需独立 path/module owner 生命周期测试。 |
| `gates` | 本波 `py_compile`/`diff-check` 通过，未宣称普通 activation 已验证。 |

### 2026-09-04 第一百四十六波

| 分区 | 结果 |
| --- | --- |
| `compat` | `e0885bf7` 为普通 `shim.runtime.activate` 创建外层 ActivationTransaction，将 shim/Jittor/project/import/extension root 的父进程 sys.path 增量改走 `mutate_path`；异常 rollback、成功 commit/release。sys.modules/no_grad/build 未接入，7.05 不关闭。 |
| `device` | child `PYTHONPATH` 是独立字符串快照，不纳入父事务；需后续 child/path failure 专测验证。 |
| `gates` | `py_compile`/`diff-check` 通过，本波无完整 activation 运行测试。 |

### 2026-09-04 第一百四十七波

| 分区 | 结果 |
| --- | --- |
| `compat` | `065b71f9` 将 outer ActivationTransaction 显式传给 integrations，并在 inner `torch_compat.install` commit/rollback 后清理 state，修复 committed inner tx 被后续 mutation 误用；普通 activation 仍需更完整运行验收。 |
| `gates` | 相关事务/安装上下文定向测试 31 passed，未改变 7.05 整卡状态。 |

### 2026-09-04 第一百四十八波

| 分区 | 结果 |
| --- | --- |
| `compat` | `45a1283c` 增加结构合同，锁定 outer transaction 传递与 inner state 清理；合同/事务测试通过，防止 nested activation 回归。 |
| `gates` | 本波仅补回归合同，普通 activation 的 child/build/no_grad 全流程仍待。 |

### 2026-09-04 第一百四十九波

| 分区 | 结果 |
| --- | --- |
| `compat` | `a08d5c15` 更新 activation bootstrap 的旧测试夹具以匹配 `publish_module`/`mutate_path` 协议；完整 `test_torch_bootstrap.py` 42 passed。7.05 仍待扩展构建与全量运行态验收。 |
| `gates` | 本波仅修正回归测试选择/夹具，未放宽冲突保护或宣称硬件验证。 |

### 2026-09-04 第一百五十波

| 分区 | 结果 |
| --- | --- |
| `compat` | `e05064b9` 新增普通 activation 失败注入回归：owner path/module 回滚、failed 状态和 ActivationTransaction 锁释放均验证；完整 `test_torch_bootstrap.py` 43 passed。扩展构建副作用与 child/build 全流程仍待。 |
| `gates` | 本波只增加 mock failure coverage，不宣称真实扩展构建或 child process 已通过。 |

### 2026-09-04 第一百五十一波

| 分区 | 结果 |
| --- | --- |
| `build` | 复核 9.20/9.22 已由 `1919b035`/`c4bbdd72` 合入：asm、`.so`、`.key` 和 wrapper 产物均采用私有临时文件后原子发布；9.22 聚焦测试 3 passed。9.20 的 inode/四 worker 证据已记录在看板。 |
| `bindings` | 复核 9.23 已由 `17e43c9a` 合入：POSIX 进程组终止与有界 drain 覆盖孙进程留存；grandchildren 聚焦测试 1 passed。 |
| `gates` | 本波无新代码提交；只确认已有提交、测试证据与当前主线一致，未宣称 Windows/真实 CUDA/NPU 全平台验证。 |

### 2026-09-04 第一百五十二波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 2.19 首次在真实 CUDA 上跑验收，推翻此前 82 处「本机无 CUDA」的说法（本机 nvcc 12.2.140、`cuda_archs=[89]`；那个结论来自读 `has_cuda` 时带了 `nvcc_path=""`）。**找到一处迁移是错的，也就是第一百五十七波没归因出来的那个 SIGABRT**：`~VarHolder` 让异常逃出析构，而析构隐式 `noexcept`，`std::terminate` 就发生在析构自己的栈帧上，`ed12fe21` 给生成 `tp_dealloc` 包的 catch 永远轮不到——所以 faulthandler 看不到 Python 异常，栈要用 gdb 抓（`__cxa_call_terminate` 上面就是 `~VarHolder`）。真实后果是 `tests/backends/cuda` 跑到 36% 就 SIGABRT、后面 21 个文件一个没跑、pytest 无汇总行；结构门禁看不见它，因为抛出是经由一次函数调用传递的。已修并补 `test_var_holder_teardown.py`（修前 rc=134，修后 2 passed）。新增 `test_cuda_user_error_boundaries.py` 21 passed，覆盖 cuDNN conv/conv3d/RNN、CUB argsort/arg_reduce、curand、cuFFT 此前只有静态证据的边界。修正 `test_backend_teardown.py` 的探针（`Var.sync` 不做 device sync，原断言在真机上一直是假的），2 passed。全目录 `tests/backends/cuda` 180 passed / 23 failed / 37 skipped / 1 xfailed，23 条红与 2.19 无关且已逐条记入看板杂项（21 条是 0 维 `code` 算子代码生成死循环的级联）。方法写成 skill `cuda-negative-path-verification`。2.19 仍待领：cuTT 后端不可达、NCCL 与双卡用例本分区跑不了。 |
| `gates` | **`tests/core` 的原生 CPU 门禁此前一直是 SIGABRT，不是绿也不是红。** 在 `4b5eaaa9` 之前的树上 `JITTOR_TEST_DEVICES=cpu pytest tests/core` 跑到 65%（`test_function.py::TestFunctionWithEagerExecution::test_multi_grads_multi_out_stop_grad_1`）就 EXIT=134、没有汇总行——同一个 backward liveness 账不平，CPU 上也可达，2.10 合入至今没人看出来。修后整轮跑完：**579 passed / 20 failed / 67 skipped / 2 xfailed**。逐条比对过：**修前能跑到的用例，修前修后逐条一致**（`test_core.py` 两边都是同样 3 条红）；20 条里 3 条是那个账不平换了个落点（`test_zmem_leak{,2,3}` 报 `lived_vars 2 != 0`，见看板杂项），其余 17 条修前根本跑不到，需要各归属分区认领。`tests/structure` 的 17 条红全部落在 compat/shim/ACL/`tests/compiler`，与本波无关，已核对本波文件不在其中（rebase 到最新之后重跑一次是 510 passed / 4 failed，4 条都在 compat/shim）。**CPU torch 模式**：`JITTOR_TORCH_SHIM=1 pytest tests/compat/torch` 全目录 **1172 passed / 56 failed / 123 skipped**（1h30m），零 abort。56 条与本波无关——取其中最像生命周期的两个文件（`test_torch_compat_math.py`、`test_torch_compat_grad_management.py`）在**同一棵树**上把本波两个 C++ 文件回退再跑一次，两次都是同样的 5 failed / 9 passed、node id 逐条一致。 |

### 2026-09-04 第一百五十三波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 8.12 修前核对确认 cuDNN 2D/3D forward、backward-x、backward-w 共六条 legacy cache 仍用全局 `JK/to_string()`；需要统一 `LegacyConvAlgoKey`、跨 TU 声明、per-device 生命周期和 CUDA 验收，本机无 CUDA，未做半迁移。 |
| `device` | 4.13 现有 ACL/CUDA/external capability 接口彼此独立，没有统一 `(op, backend, dtype, layout)` 矩阵 owner 或跨后端 runner；ACL/ROCm/Corex 硬件也不可用，未伪造契约完成。 |
| `compat` | 5.24 的十个 `jt._*` 跨模块契约未找到可独立闭环的安全切片；本波无代码提交。 |
| `gates` | 上述三项均保持 `待领`，原因与证据已写入对应看板行；未新增进行中任务。 |

### 2026-09-04 第一百五十四波

| 分区 | 结果 |
| --- | --- |
| `compat` | 7.03 本波未找到可在不拆散 owner/fidelity/Var 委托合同的前提下独立关闭的 tensor API，未提交半成品。 |
| `bindings` | 2.19 剩余 pyjt 断言均为内部不变量（pickle bytes、内部 NumPy copy、ABI 版本），不应改成用户错误；本波无代码提交。 |
| `device` | 8.06 的 `AclOpFunctions`、attribute data channel、descriptor cache 需协同迁移并依赖 910B3/CANN；已有结构合同足够，未重复造 facade。 |
| `gates` | 本波无新代码提交、无新增进行中任务；三项保持 `待领`，硬件限制和下一步边界已记录。 |

### 2026-09-04 第一百五十五波

| 分区 | 结果 |
| --- | --- |
| `gates` | 新增 `agent/results/2026-09-04-cuda-availability-verification.md`：只读核验 `nvidia-smi` 与分区 `probe.json` 均确认开发机有 8 张 RTX 4090、CUDA 12.2.140、sm_89。此前“本机无 CUDA”只能解释为 CPU-only 进程配置，不再作为跳过 CUDA 实机验证的理由。 |
| `bindings` | 2.19 的历史静态证据没有被自动升级为运行时证据；后续应在独立缓存和 GPU 分区重新跑负向用例，再决定是否关闭聚合任务。 |
| `device` | ACL/ROCm/Corex/NPU 与多机限制不变；CUDA 可用性报告不代表这些后端可用。 |

### 2026-09-05 第一百五十六波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 2.19 CUDA cohort A 在独立 GPU/缓存下通过 cuBLAS 3、cuSPARSE 10、CUB cumsum 1 个负向节点；cohort B 的 cuDNN RNN 两个 invalid 节点通过，但 4 个 dtype 正向节点进程 abort，不能计为通过。 |
| `device` | cohort C 的 curand 6、cuFFT 4 个 CUDA 节点通过；cuTT transpose 5 个因 `not use cutt` 跳过，未误报为通过。 |
| `gates` | 本波只补真实 CUDA 证据，无代码提交；2.19 仍需处理 cuDNN abort、未运行的 CUDA family 和剩余用户边界，保持 `待领`。 |

### 2026-09-05 第一百五十七波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 单独运行 `TestCudnnRnnDtype::test_float32_matches_reference` 仍在 `_cudnn_forward` 执行阶段触发 SIGABRT（退出 134），faulthandler 无 Python 异常汇总；不能计为通过，也不能仅凭现象归因到 RNN 断言回归。 |
| `build` | 诊断发现冷缓存并发 nvcc 对临时目录可写性敏感；一次未预创建的 TMPDIR 直接导致 `nvcc fatal: Could not open output file`，后续复现均使用独立、预创建目录。 |
| `gates` | 本波无代码提交；abort 根因待进一步取得 C++/CUDA 运行库日志，2.19 保持 `待领`，不把未归因失败标成环境无关或通过。 |

### 2026-09-05 第一百五十八波

| 分区 | 结果 |
| --- | --- |
| `device` | `8.06` 收口标准 ACL launcher owner：迁 SWhere、SigmoidBackward、BatchNorm 前反向到 `BaseOpRunner::launch`，各自保留同步策略、BatchNorm 属性与 outMask 释放顺序。**此前十波看板记的「标准 owner 已穷尽」不成立**——这四个仍自己驱动 aclnn execute 并保留审计列为「关键」的 `LOG_PRINT` 后 return。同时把 `checkRet == 65` 计数断言（自 `5be5fa15` 被自己的迁移作废、红穿约四十个提交）换成闭集不变量，豁免仅 reduce prod 与 KVCacheMemcpy。ACL 静态合同 81 passed + 1 failed → 88 passed。新增 skill `acl-host-syntax-check`（桩 CANN + `g++ -fsyntax-only`，全树 43 源文件与 68 个 launcher ABI 断言通过，两次反向对照确认能真的报红）。**未做**：data-channel C++ 解码入口、胖 `AclOpFunctions` 类型擦除、属性 data 通道、描述符缓存。**本机无 CANN/NPU，不宣称硬件验证**；910B3 nodeid 与禁止 CPU fallback 检查写在 `docs/guides/ascend-910b.md`。 |
| `gates` | `tests/structure` 单目录 498 passed / 14 failed，14 条全部落在 `test_runtime_composition_structure.py`、`test_torch_compat_structure.py`、`test_torch_shim_structure.py`、`test_vllm_compat_structure.py`，属 `compat`/shim 分区在飞的改动；本波未触碰 `python/jittor/compat/**`，与本波改动无关。 |

### 2026-09-05 第一百五十九波

| 分区 | 结果 |
| --- | --- |
| `gates` | `0.22` 三个提交：`dcc335d6` 修设备对拍的 0 维余切投影——`randn(*shape)` 在 0 维输出上是 `randn()`，返回 python float，`.astype` 抛 `AttributeError`，于是 `sum`/`trace`/全部 loss **从来没走到反向比较**（修后这 10 条第一次真做完 CPU↔CUDA 反向比对，数值一致）；`f9c26111` 落地判据工具 `tools/gate_conclusion_diff.py`（分开记 collected 与 conclusions，差集就是丢结论的形状，skip 原因变化也算变化，墙钟只报告不判据），并给 `verifying-a-gate-actually-ran` 补第九节；本次提交落地方向 (a) 的 CPU 参考值缓存（键含 `python/jittor/**` 内容哈希与物化输入字节，条目自述键、写用 `os.replace`、命中数进摘要）。**A/B（26 个 nodeid、两轮都冷算子缓存）848.5s → 711.1s（−16.2%），`compare` 报 IDENTICAL、26/26 逐条相同。** `0.22` 保持「待领」：原验收未达到。**两处纠正**：计划里「CPU 那半是最直接的一半」不成立（实测 18%/26%）；`0.16` 的「热缓存≈冷缓存所以不是编译瓶颈」复测不成立（同一批 nodeid 背靠背：冷 848.5s、热 23.6s，36 倍）。派生 `0.23`（`cuda.yml` 从没 restore/save JIT 缓存，`cpu.yml` 早就在做）与 `0.24`（没有任何东西检查「CUDA 门禁真跑过 CUDA」，82 处「本机无 CUDA」是错的）。`tests/structure` 单目录 502 passed / 10 failed，10 条全在 compat/shim 分区在飞的文件里，与本波无关。 |

### 2026-09-05 第一百六十波

| 分区 | 结果 |
| --- | --- |
| `build` | `cf3835ee` 把热缓存 `import jittor` 归因到具体一步：看板此前的 1.332 s 是 `nvcc_path=""` 的 CPU-only 配置（复现 1.325/1.335 s），CUDA 配置同树是 2.457 s；「40.015 s 冷编译」复现 39.96 s/176 TU，但触发条件是在 CUDA 配置已热的同一 `JITTOR_HOME` 里切到 CPU-only（`cfg` 指纹不同），不是空缓存。最大一项是核心编译在无事可做时的固定开销 0.906 s / 1.325 s（68%）：176 条编译命令进 16 进程 Pool 做空转缓存校验 0.542 s、`gen_jit_flags` 纯 Python 剥注释扫 176 个 `.cc` 后写出逐字节相同的头 0.212 s、pyjt 0.104 s；`probe.json` 单次读取 <0.2 ms，探测已不是热路径成本项。`51d0439f` 把这一步收进 `compiler.build_core()` 并加构建戳（`src/`+`extern/` 每文件 mtime_ns+size、编译要素、产物 stat、编译顺序，原子写），戳一致整步跳过、任何一项不一致走原完整逐文件校验。热缓存 import CPU-only 1.332→0.413 s（达标）、CUDA 2.457→1.545 s（未达标，剩余归因为 extern 自定义算子 49 条命令空转校验 0.351 s 与无条件 `import cupy` 0.369 s）。冷缓存（空 `JITTOR_HOME` 68.3 s）与换配置两种情形不变，import 仍编译整个核心，所以 9.01 的「移到显式 bootstrap 或首次算子调用」只完成前一半，保持 `待领`。量法沉淀为 `jittor-build-change-verification` §2.5 与 `measure_import_cost.py`。 |
| `gates` | 三套门禁逐条 A/B（基线 `534d375d`，同一 `JITTOR_HOME`，只回退 `compiler.py` 与新增测试文件），结论是**改前改后完全一致，无新增失败**：native 452→459 passed / 21→20 failed（多出的 7 passed 是本波新增测试，少掉的 1 failed 是 `test_jit_search_bounds::test_search_timeout_stops_the_search` 这条本就时序敏感的用例这一轮恰好通过）；torch shim 155 passed / 10 failed 完全相同；`tests/structure` 15 failed / 491 passed 相同（structure 的绝对数已被 158/159 波的 compat 改动降到 10–14 failed，所以本波的比较基线必须记成 `534d375d`）；CUDA `dtype_coverage` 6 passed、`test_torch_compat_cuda_tf32` 2 passed、`network_training_parity` 8 skipped（既有基线）、`tests/backends/cuda` 64 passed / 21 failed、`tests/ops/test_ops.py`（shim）172 passed / 35 failed，均改前改后逐条相同。**本波核实五条既有阻塞，均非本波引入。** 一，`tests/core/test_device_methods.py` 与 `tests/backends/cuda/test_device_methods.py` 同名，pytest 收集期报 import file mismatch 并**整体中止** native 门禁（不加 `--continue-on-collection-errors` 时 native 门禁一个用例都跑不到，这条会让任何人误以为自己的改动搞坏了全树）。二至五，四处进程 abort，改前改后中止在同一 nodeid：native `test_complex64_linalg.py::TestComplex64LinalgCPU::test_svdvals`；torch `test_torch_compat_autograd.py::TestCustomFunctionCompatibility::test_a_second_call_does_not_steal_the_first_calls_context`（该文件单独跑不 abort，是跨文件进程状态泄漏，harness 自己会报「changed process-wide state and did not put it back」）；CUDA `test_cudnn_rnn_dropout.py::TestCudnnRnnReserveSpace::test_reserve_space_is_queried_once_per_configuration`，即 156/157 波记录的 cuDNN RNN abort；shim CUDA `tests/ops/test_ops.py` 在 `test_reference_getitem_step_float64` 之后。另 `tests/backends/parity/test_device_parity.py` 在 `_compare` 内 abort（本波运行的是 615f307a 之前的版本，0.22 随后改过该文件，未做 A/B）。 |

### 2026-09-04 第一百四十波

| 分区 | 结果 |
| --- | --- |
| `compat` | `02b1733b` 将已加载 module 属性前后差异登记进 transaction；`6e0f838a` 对 module-patcher finder/registry/entry-point 回滚增加外部冲突硬失败。`test_compat_mechanisms.py` 20 passed，事务测试 9 passed。可变对象内部 mutation 与全部 installer 汇总仍待，7.05 不关闭。 |
| `gates` | 相关 Python 编译、diff 检查和结构合同通过；未改变整卡状态。 |

### 2026-09-04 第一百三十八波

| 分区 | 结果 |
| --- | --- |
| `compat` | `1a37b895` 让 `module_patcher.install_module_patches` 接收 transaction，记录 `_FINDER` 插入及 `_REGISTRY`/`_ENTRY_POINTS_LOADED` undo，integrations 传递 active transaction；外部 finder/allowlist owner 冲突测试仍缺，7.05 不关闭。 |
| `gates` | `py_compile`/`diff-check` 通过；失败 patch 与已加载模块属性的完整回滚仍待。 |

### 2026-09-04 第一百三十九波

| 分区 | 结果 |
| --- | --- |
| `compat` | `ee1317c2` 让 permissive finder 新增/索引与既有 allowlist 增量回滚做 owner-aware 校验；外部 allowlist/finder 重排负向测试纳入事务合同，事务测试 9 passed。module_patcher/其他 installer 全流程仍待，7.05 不关闭。 |
| `gates` | `py_compile`/`diff-check` 通过，未改变整卡状态。 |

### 2026-09-04 第一百三十六波

| 分区 | 结果 |
| --- | --- |
| `compat` | `88795374` 让 permissive finder 新增和既有 allowlist 增量通过 `InstallTransaction` 记录/回滚，compiler installer 三处调用已接线；未覆盖 module_patcher finder 与并发外部替换，7.05 仍待领。 |
| `device` | child process 不继承父进程内存中的 finder；现有 child-env 合同足够，本波无 device 代码。 |
| `gates` | `py_compile`/`diff-check` 通过；完整 meta_path owner 冲突测试仍待。 |

### 2026-09-04 第一百三十七波

| 分区 | 结果 |
| --- | --- |
| `compat` | `9c6a7e92` 修正 compile refusal 测试中的过时错误文案；transaction/permissive 定向测试共 27 passed。此为测试合同修复，7.05 仍待完整 mutation 接入。 |
| `gates` | `py_compile`/`git diff --check` 通过；未改变整卡状态。 |

### 2026-09-04 第一百一十九波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 8.12/10.17 整卡仍不能关闭：cuDNN cache key 仍用全局 `JK`，异步诊断仍缺 runtime ring/CUDA 行号路径；本波无 bindings 代码。 |
| `compat` | `bb5f72f1` 将 tensor installer `corrcoef` 提升为模块级 numerical owner，登记 approximate fidelity；CPU identity/NumPy 对拍 2 passed。7.03 聚合仍待领。 |
| `device` | `80e0f1b4` 增加 10.17 异步错误诊断契约，规定 per-thread bounded launch ring、allocation-free/not-found 行为及 CUDA 验收命令；静态合同 1 passed，未改 runtime。 |
| `gates` | 本波没有可完整关闭的新卡；0.15 仍约 390 s，HCCL/ACL 仍待实机。 |

### 2026-09-04 第一百二十波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 8.12/10.17 整卡复核未发现可在 CPU/静态环境完整关闭的子卡；cuDNN POD key 与异步 runtime ring 均缺关键实现/实机证据。 |
| `compat` | `a0d3be31` 将 `broadcast_shapes` 提升为 tensor installer 模块级稳定 owner，登记 approximate fidelity；identity/metadata/NumPy shape 对拍 2 passed。10.20 introspection 仍缺完整迁移。 |
| `device` | 10.17 运行时 ring 不能安全独立实现：涉及 executor 并发、fused/stream 归属与节点生命周期；保留 `80e0f1b4` 设计契约，等待 CUDA 验证。 |
| `gates` | 本波未改变 0.15 性能门槛或硬件路线；待领保持 73。 |

### 2026-09-04 第一百二十一波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 8.12 审计确认 cuDNN 2D/3D forward/backward 六条 legacy cache 路径仍需共享 `LegacyConvAlgoKey`、per-device cache 和 CUDA 验收；当前无安全半改。 |
| `compat` | companion 审计确认现有 CUDA cache 测试未禁止 legacy `JK`，必须在主实现后补结构门禁；本波无代码提交。 |
| `device` | cuFFT/cuTT 已有 POD key；cuDNN key 字段/容器/设备隔离不对称，需一次协同迁移，当前无 CUDA 实机。 |
| `gates` | 本波整卡未关闭任何任务，待领保持 73；阻塞证据已记录。 |

### 2026-09-04 第一百二十二波

| 分区 | 结果 |
| --- | --- |
| `bindings` | 2.22/4.06/8.12/9.07 均未满足整卡条件：旧环境变量、fallback CPU 路由、cuDNN 全局 JK cache key 和 import 环境写入仍存在。 |
| `compat` | 7.05/7.07/10.18/10.20/10.21 分别缺安装锁、patch 边界、核心属性测试、统一内省 API 和 import layering lint。 |
| `device` | 0.20/0.22/1.05/2.14/8.06 仍受跨目录布局、CUDA 多卡、Runtime 前置或 ACL 协同大改阻塞。 |
| `gates` | 0.15 仍约 390s/预算 446s，6.B02/6.B16/10.19 仍待 910B3/HCCL 实机；本波无代码提交。 |

### 2026-09-04 第一百一十二波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `d2532ac2` 将 cuDNN convolution format 用户边界统一为可捕获 `USER_CHECKop`，补独立 format ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `dbdfb6d7` 将 `index_copy_` 提升为 numerical 模块级原地 owner，统一 Var/顶层绑定并登记 approximate fidelity；mutation/return-self CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `a768cc9b` 锁定 ACL data schema 静态合同（schema/version/type/cache-key/decoder 必检字段），2 passed；未改运行时，NPU 实机仍待。 |
| `gates` | 本波未改变 0.15 性能门槛或 10.19 HCCL hardware-only 路线；相关完整验收仍待。 |

### 2026-09-04 第一百一十三波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `de5188ab` 将 cuDNN backward-x format 用户边界统一为可捕获 `USER_CHECKop`，补独立 format ledger；定向 3 passed。2.19 聚合任务仍待完整审计。 |
| `compat` | `11e9b456` 将 `index_put_` 原地实现提升为 numerical 模块级 owner，统一 Var/顶层绑定并登记 approximate fidelity；原地/重复索引 CPU 对拍 2 passed。7.03 其余 family 仍待领。 |
| `device` | `4b22f6d9` 增加 ACL data schema Python-only validator，覆盖版本/operator/类型/默认值/cache key 及非法 pointer/object-id；静态合同 6 passed，C++ decoder 和 NPU 实机仍待。 |
| `gates` | 本波未改变 0.15 性能门槛或 10.19 HCCL hardware-only 路线；相关完整验收仍待。 |

按 [派活说明](refactor-dispatch.md) 每波最多四分区、每分区最多五项。第一百波继续优先可独立验证的
family/cohort；8.06 只按 family 迁移，不铺开 65 个尾巴：

- `device`：若续做 8.06，只迁下一个最终 owner 明确的 family，并复用 5be5fa15 的 launcher 合同；无 NPU
  时只做代码组织/静态合同/上机文档，保持待实机。
- `gates`：10.19 继续补 HCCL/其他硬件 backend 的 CPU reference 或明确硬件-only 合同，不能把 unsupported
  误标为 CPU 覆盖。
- `bindings`/`compat`：继续 2.19 用户边界和 7.03 最终 owner family 的窄 cohort；每项先核对最终 owner，
  不跨域抢改。

Corex 8.14 的正式前置 4.12 尚未满足；ACL/NPU、ROCm、Corex 等本机缺硬件的后端允许先完成代码组织、
公共接口和迁移文档，不做性能优化，但看板必须保留待实机状态并写清机型、SDK、命令和禁止 CPU fallback
的检查。设备模型统一后再删除旧 `process_acl` 并完成实机验收。

用户已明确批准：ACL/NPU、ROCm、Corex 等本机缺硬件的后端可以先完成代码组织、公共接口和迁移文档，
不做性能优化；必须在看板写清待跑机型与命令，真实设备验收前不得宣称硬件验证完成。3.18 与 3.22 的
剩余是性能终点，继续留给性能波复查。

### ⚠ `tests/structure` 当前是红的，共 15 条既存失败，跑一遍约 7 分钟

`AGENT-BRIEF.md` 第 11 节说「推之前跑一次 `tests/structure`，只要 3 秒」，并把它当成一道绿的门禁
用来挡跨分区违规。**实测不是这样**：在 `59ee6553` 的干净 `gatecheck` worktree 上
`JITTOR_TORCH_SHIM=1 pytest tests/structure -q` 得到 `15 failed, 491 passed, 2 skipped in 410s`，
在 `coord` 上逐条复现同一份清单。

**后果**：门禁整体为红时，agent 无法把自己新引入的违规从这 15 条噪声里分出来——这正是第 11 节
要防的失败模式（「期间没有任何东西跑过它」）。**在清单归零之前，判据是「你的失败是否在下表之外」，
而不是「是否全绿」。**

已修 5 条：`d1535282`（7.03，`repeat_interleave` 身份，一次修掉 2 条）、`c8ce8760`（10.24，fixture 契约改按真实来源解析——原判据拿参数与一份只含 pytest 内置
fixture 的冻结清单比较，于是 7 处使用自有 fixture 的合法测试全被误判）、`6dd80916`（0.21，
`test_cache_atomic_publish.py` 的 dlink 子进程改走 `child_process` helper，一次修掉 2 条）。

**2026-09-04（compat 分区，第 159 波）实测更新**：在当天的 `2.0-refactor` 上跑同一条命令是 `4 failed, 508 passed, 2 skipped in 136s`——不是 7 分钟，也不再是 15 条。本波修掉的是 `test_misc_structure` 那 2 条；`test_acl_runner_failure_contract`、`test_child_process_contract` 2 条、`test_pytest_contract` 1 条、`test_torch_shim_structure` 3 条、`test_vllm_compat_structure` 1 条在这几小时里被别的分区修掉了。另外 `test_compat_exception_policy::..._catches_exception_at_large` **单独跑是 9 passed**，只有整目录跑时才红，属顺序/状态依赖，不要按「真违规」去追。**判据仍然是「与改前逐条同集合」，不是「全绿」；所以改之前先跑一次留基线。**

**下表按第 155 波的 15 条记录，剩余 12 条，按成因分类：**

| 失败 | 成因 | 归属 |
| --- | --- | --- |
| `test_acl_runner_failure_contract::..._fail_loudly` | `assert 8 == 65`。8.06 把 family 迁进 `BaseOpRunner` 共享 launcher 后，尾巴不再直接出现 `checkRet`，6.B02 的计数法失效 | 契约与实现脱节，8.06 owner |
| `test_cleanup_structure::..._duplicate_implementations_are_reviewed` | `_set_use_cuda` 在 `installers/factories.py` 与 `installers/tensor.py` 各一份（7.05 第 134 波 `60197b81`） | **真重复**，7.05 owner |
| `test_compat_exception_policy::..._catches_exception_at_large` | `compat/transaction.py:155` 宽泛 `except Exception`（7.05 引入） | **真违规**，7.16 要求 compat 内归零 |
| ~~`test_misc_structure::..._use_real_paths_and_legacy_pickle_aliases`~~ | `repeat_interleave` 是两个不同的函数对象 | **已修 `d1535282`**（7.03 把转发 wrapper 改为再导出原生 owner） |
| ~~`test_misc_structure::..._do_not_invent_inplace_aliases`~~ | 同上 | **已修 `d1535282`** |
| `test_runtime_composition_structure::..._only_preflight_and_post_core_composition` | 根部多出 `_publish`、`_make_inplace_alias` | 待判 |
| `test_runtime_composition_structure::..._is_orchestration_only` | 期望 `['enable']`，实际是 7.04 把三条入口收敛成 `activate()` 之后的形状 | **陈旧期望**，改测试 |
| `test_torch_compat_structure::..._has_an_exact_owner_whitelist` | 白名单缺 `runtime.py compose 'torch'` 等 2 项 | **陈旧清单**，属 0.19 精神 |
| `test_torch_shim_structure::..._is_a_runtime_facade` | 期望 `from .runtime import enable`，实际是 `activate, activation_status` 加 `enable = activate` | **陈旧期望**（7.04） |
| `test_torch_shim_structure::..._36_file_manifest_matches_bytes` | 整树字节 manifest 哈希不匹配 | **0.19 明确要求删除的那类精确清单**，本该随 0.19 一起删 |
| `test_torch_shim_structure::..._is_an_identity_only_entrypoint` | 期望 `_torch_compat.install(_jittor)`，实际改成 `shim.activate()` | **陈旧期望**（7.04） |
| `test_vllm_compat_structure::..._through_its_public_entry_points` | `layers.py:19` 导入 `jittor.nn.backends.hooks`，非公开入口 | **真违规** |

即 6 条是陈旧期望（其中 2 条本该在 0.19 里删掉）、4 条是真违规或真重复、2 条待判。**陈旧期望那批
不要靠放宽断言了事**——0.19 的要求是「从精确清单改成规则」，改成规则才算修；`10.24` 是一个示范：
它同时验证了修正后的判据仍能抓住原本要抓的东西（临时反例仍被报出），而不是把门禁改松。

**本波收尾实测：`2 failed, 510 passed, 2 skipped in 327s`。** 剩下的两条就是下表里归属别人的那两条
（`test_cleanup_structure` 的 `_set_use_cuda` 双份、`test_torch_compat_structure` 的 `sys.modules`
白名单缺 2 项）。**清单归零之前，判据仍然是「你的失败是否在下表之外」。** 顺带一条：
`test_state_leak_helper::..._rejects_an_intentional_retained_allocation` 曾在整目录跑时红一次、
单跑连续两次绿，属抖动，不在下表里也不必追。

上表其余各行的落点（按提交）：`6d7df2dd`（7.16）收窄 `compat/transaction.py` 的宽泛 handler；
`71adc134`（7.21）把 vllm 的融合 QK 快路径改走 `jt.nn` 公开入口；`f094dcd3`（0.19）与
`b33e3b3d`（0.19）把 torch shim 三条、runtime composition 两条精确清单改成规则，其中
`b33e3b3d` 还带一个**真违规**：根 `__init__.py` 定义了 `_publish` 与 `_make_inplace_alias`，
违反它自己声明的「根只组合、不定义」，已搬进 `python/jittor/_composition.py`。
剩 `test_cleanup_structure`（`_set_use_cuda` 双份，7.05 owner）与
`test_torch_compat_structure`（`sys.modules` 白名单缺 2 项，陈旧清单）两行。

### `tests/core` 原生 CPU：第一次跑到真实汇总，`17 failed / 570 passed / 106 skipped`

这道门禁此前**既不是绿也不是红**：进程在中途 `EXIT=134` 没有汇总行。`4b5eaaa9`（2.19，
`~VarHolder` 自己接住、liveness 队列排空改 RAII）之后它能跑完了，于是有了第一份真实清单：

```
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core -q      # 5:54
17 failed, 570 passed, 106 skipped, 1 xfailed
```

**`6.C32` 因此改判为已合并**：它把 abort 归因给 `test_complex64_linalg::test_svdvals`，但那条
用例本身是通过的——abort 发生在**进程退出期**，单选时 pytest 已经打完汇总，容易读成"这条失败"。
含 `4b5eaaa9` 的树上该文件 11 passed / 11 skipped、连跑 6 次零 abort。

**两个配置的数字不一样，引用时要带配置**。上面那行是 CPU-only（`nvcc_path=""`）；同一棵树在
有 CUDA 的配置下是 `579 passed / 20 failed / 67 skipped`（见 2.19 的记录）。差的 39 个 skip 就是
CUDA 门控的用例，它们在 CPU-only 下不跑——**所以「17 条」和「20 条」不是矛盾，是两个口径**。

17 条里 **10 条同属「存活 Var 与内存记账」一簇**——`test_function` 的
`test_zmem_leak{,2,3}` ×2 个类共 6 条、`test_misc_issue::test_argmax_memleak`、
`test_core::test_number_of_hold_vars`、`test_core::test_var_holder`、`test_core::test_fuse_memopt`。
pytest 自己的「runtime state left behind」报告也在指同一处（跑完
`test_complex64_linalg` 后 `number_of_lived_vars 0 -> 1`）。

**这一簇里的 `test_zmem_leak{,2,3}` 已经有归因，不要当谜题重查**：2.19 的执行者查明它们报的
`lived_vars 2 != 0` 中泄漏的那 2 个 var，正是那次失败释放留下来的——也就是说这是底下那个
**backward liveness 多释放**的正确落点，而不是新问题。那个账不平属 2.10 的范围（2.10 之前它是
`int` 下溢到 -1、节点永不释放、静默泄漏），已单列在本文档的已知问题表里。**先修账不平，再回来看
这一簇还剩几条**；剩下几条与 `hold_vars`／`fuse_memopt` 相关的再单独归因。

其余 7 条各自独立：`test_core::test_node_order`、`test_grad::test_no_grad`（`assert 5 == 2`）、
`test_grad_missing::test_every_missing_gradient_is_reported`、
`test_complex64_native::test_python_complex_scalar_setitem`、
`test_namespace_exports::..._stub_top_level_names_match_the_export_surface`（`.pyi` 缺
`benchmark`／`BenchmarkResult`，属 5.23）、
`test_rootcause_semantics::..._canonical_installer_owns_the_parameter_marker`、
`test_setitem::test_getitem`（该文件属别人，不要提交）。

**一条方法上的提醒**：我一度以为这些 abort 与机器负载相关（前两次全量跑在 load≈13 时 abort、
后来空闲时不 abort），核对提交祖先后发现是 `4b5eaaa9` 落地与否的差别，负载是巧合。**"环境
还是回归"这个问题要用祖先关系回答，不要用相关性**——尤其在一天里有八个分区在推送的时候。

### 3.23 逐元素带宽：口径已建立，前提已推翻，验收卡在兼容层（`c0f3420a`）

任务描述里的「UNet 61 种融合 kernel 合计 4.47 ms、约 475 GB/s（峰值一半）」在今天的树上
**不成立**。`large_diffusers_unet2d` 一步前向加反向（RTX 4090、TF32、独占卡），nsys 与
Jittor profiler 两法互校：

| 角色 | nsys | profiler | | PyTorch 2.12.1 |
| --- | ---: | ---: | --- | ---: |
| `library:cudnn` | 11.65 ms | 11.86 ms | conv/gemm | 18.17 ms |
| `handwritten:code` | 4.13 ms | 3.58 ms | | |
| **`elementwise`** | **3.37 ms** | **3.29 ms** | elementwise + other | **3.04 ms** |
| `library:cublas` | 3.05 ms | 2.10 ms | | |
| `reduce` | 0.57 ms | 0.59 ms | reduce/norm | 1.20 ms |
| `indexing` | 0.25 ms | 0.62 ms | | |
| 合计 | 23.02 ms | 22.03 ms | | 22.41 ms |

实测可达 copy 带宽 916.7 GB/s（标称 1008 的 91%）。逐元素类跑到 1086 GB/s、屋顶线
ratio 0.84——**整体已经贴着屋顶**，超出部分由 72 MB L2 承担。所以这一类要更快只能
**少搬字节**（更好的融合、不物化中间量），不是把 kernel 写快。49 种融合 kernel 的
正向超出合计只有约 0.59 ms，三个来源没有一个在代码生成里：float64 标量除法 0.55 ms
（兼容层 `_make_truediv` 故意加宽）、裸 `transpose` 0.10 ms、约 60 次几乎不搬数据的
小 kernel 0.23 ms（纯 launch 延迟）。

**验收「≤ PyTorch 的 3.07 ms」未达到（3.37 / 3.29 对 3.04），3.23 保持待领。** 把兼容层那条
加宽临时关掉实测可到 2.73 ms，即这条验收的钥匙在兼容层分区手里，看板上已单列。
顺带两条：shim 的 `Tensor.backward()` 在这张 CUDA 图上直接 abort（速度门禁的 CUDA UNet
一项现在跑不到时间），以及归约类今天已比 PyTorch 快一倍以上、3.22 的验收口径需复核。
量法与四个脚本在 `agent/skills/cuda-elementwise-bandwidth-roofline/`；**先读它的第 3 节**
（profiler 的 rerun 因子按 `-2` 推会让每一个每步数字正好差两倍，而报告内部自洽）。

### 第 159 波（`compat`，7.03 六个 cohort）

| 项 | 结果 |
| --- | --- |
| `16333333` | `amax`/`amin`/`count_nonzero` 收回 `jittor/misc/reductions.py` 原生 owner，提升为薄转发的模块级稳定对象并登记 approximate fidelity；给 `_axis_to_dim` 适配器加 `_torch_accepts_axis` 跳过标记；顺带修 `cosine_similarity` fidelity 文案与断言长期对不上的红。CPU 4 passed |
| `9cba7d68` | `cumsum`/`cumprod`；`out=` 的 retained-view 写回器由 install 通过模块级句柄交接。**CPU 15 passed、CUDA 15 passed** |
| `50876abf` | `sort`/`argsort`/`topk`/`median`。**CPU 13 passed、CUDA 13 passed** |
| `d94c5cbd` | `sign`/`trunc`/`frac`/`exp2`/`log10` 归一到 `installers/core.py` 单一 owner，修掉 `torch.sign` 的静默错 dtype。修前 3 failed → 修后 5 passed |
| `a7dcae1c` | `nan_to_num`/`logaddexp`。CPU 7 passed |
| `d1535282` | `outer`/`tensordot`/`repeat_interleave` 改为再导出原生 owner，`tests/structure/test_misc_structure.py` 2 failed → 9 passed |
| skill | 新增 `agent/skills/torch-api-cohort-promotion/` |
| 7.03 状态 | **仍待领**。AST 实测剩余：`_install_tensor_methods` 76、`_install_reductions` 14、`_install_nn_extras` 135、`_install_module_methods` 40、`_install_cuda` 80、data 的 install 64、`core.install_misc` 34 |

**这一波真正的产出是 CUDA 那一层。** 计划 §0 完成定义第 2 条要求三套门禁，而 7.03 前面约三十个
cohort 的证据全是「CPU N passed」——并行路径从不交叉验证正是审计的核心发现，7.03 自己却一直踩着它。
本波两个 cohort 用 `_helpers.device_types.instantiate_device_type_tests` 做设备参数化（不写
`if jt.has_cuda` 分支，`JITTOR_TEST_DEVICES` 直接驱动两侧），跑出两处真实差异：

- **`cumsum`**：4096 元素 float32、正负交替、部分和量级 ~2e3。CPU 与 float64 参考差 3.7e-03，
  CUDA 与 float64 参考差 2.1e-03，两者互差 2.7e-03（相对 1.4e-06）。**CUDA 更准**——顺序扫描误差
  随 n 线性累积，并行前缀和是 log n。12 元素的小数组两侧逐位相同，整数/bool 路径永远逐位相同。
- **`argsort`**：8×512、取值 `arange % 97`（每键约 5 个重复）。**indices 两侧不同、values 逐位相同。**
  4 元素的小 ties 用例两侧一致，规模上去才分开。

两者都判为后端固有（真 torch 的 sort 默认 `stable=False`，CPU/CUDA 也不同序），因此按要求登记进
fidelity detail 而不是改实现。测试相应写成**有界不一致 + 整数路径逐位相等**、以及
**values 跨设备逐位相等 + indices 能取回自己的 values**，而不是 `assert_array_equal`——
否则换台机器或换个 CUDA 版本就是假红。

**另一条产出是「一个 API 只能有一个对象」。** 核 owner 时撞见两处同一 API 两份实现：

- `sign`/`trunc` 在 `installers/core.py` 与 `installers/tensor.py` 各一份，安装顺序
  （`tensor.install` 在前、`core.install_misc` 在后）让 `Var.sign` 用 tensor 那份、`torch.sign`
  用 core 那份，于是 `torch.sign(int32).dtype == float32` 而 `x.sign().dtype == int32`。
  真 PyTorch 2.12.1（jt312b 实测）两者都是 int32——**值对、dtype 静默错**。
- `outer`/`tensordot`/`repeat_interleave` 被 7.03 早期 cohort 各套了一层转发 wrapper，
  于是 `torch.repeat_interleave is jittor.repeat_interleave` 不成立、也不再 pickle 回 misc owner，
  两条结构门禁长期红。

**给下一个做 7.03 的人**：先读 `agent/skills/torch-api-cohort-promotion/SKILL.md`。三条最省时间的：
(1) 每个 cohort 先跑 owner 探针，`hasattr(jt, name)` 加读原生定义，有原生 owner 就再导出/薄转发，
不要写第二份；(2) `install_methods` 的 `_axis_to_dim` 会重新包 `max/min/argmax/argmin/amax/amin/
cumsum/norm/std/var` 十个 Var 方法，让模块级身份断言失败——稳定对象自己接 `axis=` 并带
`_torch_accepts_axis`；(3) 归约/累加/排序类的 cohort 一定要跑 CUDA，跑之前用三路比对
（CPU vs float64、CUDA vs float64、CPU vs CUDA）判断是舍入还是缺陷。

本波顺手记了两条与 7.03 无关、已写进看板「需要认领的杂项」的既存问题：
`torch.split_with_sizes` + `Var.split` 之后退出时 `node.h:264 backward liveness release without a
matching owner`（会带走整个 pytest 进程，7.03 的 fidelity 测试文件只能 `--deselect` 它，已附最小复现），
以及 opinfo 全量归约参考电池仍把标量提成 `(1,)` 导致 `amax`/`amin`/`count_nonzero` 9 条红
（改前改后同集合，是参考电池自身过期）。

本波 `coreops` 的结果（2026-09-04，派生任务 `2.25`）：

| 分区 | 结果 |
| --- | --- |
| `coreops` | **`2.25`（新登记，`7.11` 与 `7.12` 的共同前置）已合并**：`100c7c5d` 登记，`c6e62ba1` 查询与内核用例，`781d4188` 与真 PyTorch 的逐例对拍。内核给出 `backward_grad_fn(Var*)` 与 `Var.is_backward_leaf`／`grad_fn_node_id`／`grad_fn_op_id`／`grad_fn_name`；requires_grad 与「生产者有一条能带梯度的入边」的合取，四条过滤器与 `grad()` 的 `bfs_backward` 同源，O(生产者入度)、不遍历、不缓存、不引入进程级 id 键字典。修前 20 failed → 修后 20 passed；定向 CPU 208 passed 对基线 188 passed（同 4 条既有失败），CUDA 73 passed，真 torch 2.12.1 的 19 个用例逐例分组一致。**故意不叫 `is_leaf`／`grad_fn`**，否则就等于替兼容层分区把 `7.11` 接了线；`7.11` 只差 `compat/torch/installers/tensor.py:1401-1411` 两个常量属性改成转发。顺手登记 `6.C32` |

**给下一个碰 `tests/core` 的人**，本波量到两件与任务无关但会浪费时间的事：

1. **把整个 `tests/core` 当一条 pytest 命令跑不是有效的门禁口径。** `test_regression.py` 与
   `test_type_system.py` 在 `tests/_helpers/process_modes.py` 的 `TORCH_MODE_PATHS` 里，会把
   **整个进程**翻成 shim 模式，于是同一条命令里的原生用例成片假失败。门禁自己是按
   `gate_scope.native_arguments()`／`torch_arguments()` 分两次跑的，本地要复现就得照做。
2. **CPU-only 下 `tests/core` 跑不到 summary。** 至少四处 abort 会带走整个 session（其中
   `test_complex64_linalg.py::TestComplex64LinalgCPU::test_svdvals` 单选也复现，已登记
   `6.C32`）。**abort 之后 pytest 一行 summary 都不打**，所以「日志没红」在这里不等于通过。
   归因的正确做法是把改动 `git checkout` 回 HEAD、用**同一条命令**再跑一遍比对——本波
   两侧都是 4 failed，只有 passed 数差 20（就是新用例），零回归的结论是这么得出来的。

第一〇〇波完成 8.02 的 NCCL 两截，HCCL 那截保持待实机：

| 分区 | 第一〇〇波结果 |
| --- | --- |
| `dist` | 8.02 前半：五个 NCCL 集合通信「改流」由 4.08 的 `0dfcb3dd` 落地，本波补两卡实测（`bd737c65`）。五个集合通信各一次 rank 相关数值对拍 + event 依赖计数各 +2 + 200 轮「现算输入 → 通信 → 立刻消费」竞态循环；两 rank 各 8 passed。**反证**：临时删掉两条 event（算子仍在侧流）报 `worst=885.0`，两 rank 都红 |
| `dist` | 8.02 后半：新增 `nccl_bucket_begin/end`、`nccl_comm_wait` 与 `cuda_side_stream_defer_join/_hold_block/_resolve_join`，Python 入口 `jittor.distributed.bucket_scope`。一桶合并成一次 `ncclGroupEnd()` 提交（event 依赖 2N → N+1），`defer_join` 给出真实重叠窗口；两 rank 各 7 passed。**反证**：把「扣住块」改成假装成功报 `26310.0`（毒值漏进结果） |
| `dist` | **重叠证据是 nsys timeline，不是墙钟**：同进程同负载 A/B，延迟 join 时集合通信 12.2 ms 窗口内 5 个 matmul kernel 并发、覆盖 57%–63%；立刻 join 时并发数 0。**本机墙钟无收益**——`nvidia-smi topo -p2p r` 全 CNS，NCCL 走共享内存传输、kernel 自旋抢 SM，窗口内 matmul 从 0.31 ms 被拖到 0.83–4.28 ms。取证脚本与判据在 `agent/skills/jittor-distributed-verification/SKILL.md` 加 `nccl_overlap_report.py` |
| `dist` | 开发中自己踩到并修掉一个真 bug：group 打开期间 NCCL 调用尚未提交到流上，此时记 done 事件是空的，于是 `defer_join=False` 也「重叠」了 61%——**那是竞态不是优化**。group 语义下 join 必须放在 `ncclGroupEnd()` 之后。测试里 `test_bucket_submits_once_and_joins_once` 的 `join_pending()` 断言专门挡这个 |
| `dist` | 8.02 的 HCCL 截（「每次集合通信 4 次全设备/流同步」，严重度关键）**没有删除**，只做代码组织：四个算子里重复四遍的同步收进 `hccl_collective_begin/end`，行为由 `JT_HCCL_COLLECTIVE_SYNC` 控制，**默认 `full` 与改动前逐字等价**。本机无 NPU，删除无法跑一次验证，硬删等于把未验证改动送到别人集群上静默算错梯度。上机清单见 `agent/manuals/hccl-on-device-verification.md`（Ascend 910B3、≥2 卡、CANN、env/file rendezvous、禁止 CPU fallback 的四条判据）。清单在真机全绿后才可把默认改成 `stream-order` |

### 2026-09-05 第一百六十一波

| 分区 | 结果 |
| --- | --- |
| `compat` | `2ec34693` 完成 7.11 最后接线：`is_leaf`/`grad_fn` 使用内核真实查询；shim autograd 20 passed，core query 20 passed。 |
| `gates` | `e4682406` 为 CUDA 与 benchmark-CUDA workflow 增加按 CUDA 配置隔离的 JIT cache restore/save；`13d314ec` 让 CUDA 门禁对真实 `has_cuda` 和 accelerator 非 skip 执行数量 fail-closed；相关结构合同 4 passed。 |
| `gates` | 本波没有宣称全 CUDA 门禁已全绿；0.22、9.01 和现有 CUDA abort/跨卡缺口继续保持待领。 |

### 2026-09-05 第一百六十二波

| 分区 | 结果 |
| --- | --- |
| `compat` | `f5461d6f` 完成 quantile/nanquantile Var owner cohort，定向 7 passed；7.03 聚合任务仍待完整 family。 |
| `bindings` | `45d9ed15` 在独立单卡 CUDA 上补 cuBLAS rank/inner-dimension 3 条真实负向，3 passed，异常后 4x4 计算继续通过；2.19 聚合任务仍待。 |
| `device` | `27bbca46` 将 ACL 按 `cu` 前缀删除外部算子的逻辑改为显式 registry，结构合同 2 passed；本机无 CANN/NPU，未宣称硬件验证。 |

### 2026-09-05 第一百六十三波

| 分区 | 结果 |
| --- | --- |
| `bindings` | `8b44ca6d` 补 cuBLAS batched/acc 非法输入后继续计算的真实 CUDA 回归；新增场景 1 passed，`test_cublas_matmul_grad.py` 整文件 10 passed。 |
| `device` | `8c5ef55b` 补 ACL 属性 data-channel 原子迁移门禁、cache-key/失败路径约束与 910B3/CANN 命令，相关静态合同 6 passed；无 NPU 硬件验证。 |
| `compat` | 审计确认简单 numerical owner 已基本完成；剩余 vmap 是复杂 runtime 闭包，sum/mean/std/var/prod 的 owner 在 tensor installer，未做重复迁移。 |

### 2026-09-05 第一百六十四波

| 分区 | 结果 |
| --- | --- |
| `bindings` | cuDNN conv3d rank/group 负向已有覆盖；独立 CUDA 单卡冷缓存运行 `test_input_rank` 1 passed，异常后计算继续通过，无新代码。 |
| `device` | ACL 剩余 data-channel C++ decoder、胖 `AclOpFunctions` 类型擦除、属性通道、descriptor cache 必须按依赖顺序整体迁移；现有 schema/边界/workspace 静态合同 9 passed，本机无 CANN/NPU。 |
| `gates` | 0.22 仍需约两小时级 CUDA 全量性能验收；9.01 热 CPU 达标但 CUDA/冷启动未达标，且只读 HOME import 复现 `PermissionError`，不适合轻量半改。 |

### 2026-09-05 第一百六十五波

| 分区 | 结果 |
| --- | --- |
| `build` | `01090519` 修复默认 HOME 只读时的缓存 fallback，显式 `JITTOR_HOME` 仍保持错误可见；`test_jittor_home.py` + 只读 import 回归 6 passed。9.01 整卡仍受冷启动/惰性导入验收约束。 |
| `compat` | `0100a475` 在 `EXPLICIT_REQUIRES_GRAD` 下让 torch-facing detach 清除返回 Var 的 requires_grad；定向 2 passed、核心策略/对拍 10 passed。NATIVE 策略保持 Jittor 原语义。 |
| `dist` | HCCL 8.02/10.19 只读审计确认无安全静态关闭项；现有合同通过但仍待 910B3 多卡实机。 |

### 2026-09-05 第一百六十六波

| 分区 | 结果 |
| --- | --- |
| `compat` | 7.12 审计确认 torch identity/module graph、`_torch_leaf_params` 19 处引用和 TorchTensorState 缺失构成整卡迁移；无安全窄切片。 |
| `build` | 9.01 冷 CPU import 约 37.6s，其中 compiler 约 35.6s；热缓存约 0.43s。build_core stamp 已解决热缓存，冷启动仍需显式 bootstrap/惰性架构。 |
| `dist` | 8.15–8.18 缺 `--nnodes/--node_rank/--master_addr/--master_port` 等多机 launcher 参数；Store/launch 聚焦 5 passed、1 failed，不能关闭多机任务。 |

### 2026-09-05 第一百六十七波

| 分区 | 结果 |
| --- | --- |
| `compat` | 复核 7.12：`torch_init`/runtime 仍共同承担 torch identity，`_torch_leaf_params` 跨 nested/tensor/nn 19 处引用，没有独立安全切片。 |
| `build` | 复核 9.01：冷启动 import 仍把核心编译放在 import 路径，热缓存优化已落地但冷启动目标未满足。 |
| `dist` | 复核 8.15–8.18：当前 launcher 仍缺多机参数，单机 Store 聚焦仍有 1 个既有失败；本机无两机/HCCL 环境。 |

### 2026-09-05 第一百六十八波

| 分区 | 结果 |
| --- | --- |
| `build` | `d141d8c2` 完成 9.07 一处 import 环境反向写入修复：`cc_path` 保留在模块状态，不污染宿主环境；定向回归 1 passed。 |
| `device` | 4.06 审计确认 backend_fallback 三态不存在，需与 BackendRegistry/OpRegistry 联动；静态 ACL 失败合同 7 passed，不代表三态完成。 |
| `gates` | 10.21 审计确认三条 import/include 真环尚未有 allowlist，直接加 lint 会误报；本波无安全 lint 提交。 |

### 2026-09-05 第一百六十九波

| 分区 | 结果 |
| --- | --- |
| `gates` | `f63856f8` 修 smoke fail-open，`41dfb254` 修 NPU gate fail-open，`c02b1481` 修 NCCL 全 skip 假绿；hardware/gate tier 结构合同 17 passed。 |
| `build` | `515ebf71` 将 compiler.py/pyjt_compiler.py 生成器签名纳入 build_core stamp，laziness 回归 9 passed。 |
| `compat` | `e9b966e7` 修 FSDP2 共享参数重复梯度查找；`b468afcf` 抽 getitem transform context，`5e2667f6` 抽 ACL clamp runtime facade，分别有定向合同通过。 |

### 2026-09-05 第一百七十波

| 分区 | 结果 |
| --- | --- |
| `build` | `6c86fb20` 将可选 CuPy 从 `import jittor` 热路径移出，CPU/导入回归 2 passed，独立探针确认未加载 CuPy。 |
| `compat` | `7eedcbd8` 将 SDPA flash stats 统一到 diagnostics facade，保留根属性兼容别名，定向 2 passed。 |
| `codegen` | `b6de9642` 新增单 kernel 冷编译分段 profiling skill，区分源码生成、cache_compile、编译、链接、加载；clang 全流程实测通过，未宣称 3.18 性能目标完成。 |

### 2026-09-05 第一百七十一波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `83c26d42` 完成 2.24：FusedOp 用显式 `Op*`/`Var*` index map 替代 `Node::custom_data`，update/load/relay 顺序保持不变；结构合同 4 passed，fused 聚焦 2 passed。 |

### 2026-09-05 第一百七十二波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `5248870d` 删除不可达 EventQueue `run_sync`/Worker dead设施、清理 executor 旧注释，NCCL 两处切换为直接 CUDA 检查；结构合同 2 passed，C++14 CUDA syntax check 通过。3.19 已关闭。 |
| `gates` | CPU smoke 未执行，原因是当前环境 CUDA 核心缓存/依赖前置失败；该环境失败未计入本任务通过或失败。 |

### 2026-09-05 第一百七十三波

| 分区 | 结果 |
| --- | --- |
| `coreops` | 2.13 审计确认 flags/Runtime 状态跨约30个 C++ 文件和309个 Python消费者，单状态迁移会破坏 flag_scope/设备 hooks；保持待领。 |
| `cudabk` | 8.12 六条 cuDNN legacy cache 各自使用全局 JK 字符串和 process-global map，需统一 POD key/per-device 生命周期，保持待领。 |
| `build` | 9.01 剩余冷启动主成本仍在 compiler import/核心编译，setup_cutt 无生产调用点，无法安全窄延迟。 |
| `device` | 4.06 缺 BackendRegistry/OpRegistry，ACL fallback 三态不能孤立增加；保持待领。 |

### 2026-09-05 第一百七十四波

| 分区 | 结果 |
| --- | --- |
| `compat` | 复核 5.24、10.20、10.21，剩余状态和 import/include layering 均跨模块，近期 facade 修复未形成新的独立闭环。 |
| `build` | 复核 9.01 冷启动仍在 import 路径编译整核，CUDA/冷启动目标未满足。 |
| `bindings` | 复核 2.19 历史 broadcast_to 计数已修正（结构 57 passed），但 cuTT 不可达和多卡缺口仍阻塞整卡。 |
| `device` | 复核 8.12 六路 legacy cache 仍需统一 POD/per-device 迁移；本波无代码提交。 |

### 2026-09-05 第一百七十五波

| 分区 | 结果 |
| --- | --- |
| `dist` | 8.15 TCPStore 失败在当前环境卡于 Jittor import/编译锁；绕过 import 的双 rank 直接 Store 脚本多次成功，未复现 ConnectionReset。没有提交未经复现的生命周期修复，任务继续待领。 |
| `gates` | 本波没有代码提交，保留现有多机/两节点硬件验收缺口。 |

### 2026-09-05 第一百七十六波

| 分区 | 结果 |
| --- | --- |
| `cudabk` | 8.12 六路 legacy cache（2D/3D forward、backward-x、backward-w）均为字符串 JK key + process-global map；统一迁移需新 POD key、六处 EXTERN_LIB ABI、per-device storage/device-switch cleanup 和 CUDA 回归，不能安全半改。 |
| `gates` | 本波无代码提交；8.12 保持待领，未把静态审计当作完成。 |

### 2026-09-05 第一百七十七波

| 分区 | 结果 |
| --- | --- |
| `device` | `fcce48e3` 建立 Python BackendRegistry/OpRegistry 的注册、查询、分派和冲突合同，6 passed；未接入现有 C++/flags 路由，4.03/4.04 继续待领。 |
| `coreops` | 2.13 Runtime 状态仍跨多模块，单字段迁移会形成双状态；本波无代码提交。 |
| `compat` | 7.12 TorchTensorState、9.01 冷启动和 5.24 剩余状态仍是整卡架构，未提交半成品。 |

### 2026-09-05 第一百七十八波

| 分区 | 结果 |
| --- | --- |
| `device` | 4.06 增加独立 `BackendFallbackPolicy`，统一校验 `error/warn/allow`、默认 `warn`、结构化 fallback 决策与 fail-closed 异常；定向结构测试 3 passed。尚未接入 native flags/BackendRegistry/OpRegistry，4.06 整卡保持待领。 |

### 2026-09-05 第一百七十九波

| 分区 | 结果 |
| --- | --- |
| `gates` | `d6f17450` 增加 10.18 的 native CPU gate 核心属性测试覆盖合同，定向 11 passed；10.18 整卡仍待领。 |
| `cudabk` | `988fd825`、`0f7046c8` 将串行/并行 JIT cache lookup 从线程局部 JK 缓冲改为自有字符串键，静态合同通过；8.12 的 cuDNN POD key/per-device cache 仍待领。 |
| `coreops` | `b1cef650` 将 `VarHolder::item` 多元素边界改为用户错误，`41878a9e` 将 `grad` loss/target dtype 边界改为用户错误，`953462c7` 将 `code` vary-shape 边界分类为用户错误，`e90f6c5d` 将 `reindex` 空 shape 边界分类为用户错误；item 结构/负向 14 passed，grad 定向 11 passed，code 定向 2 passed，reindex 定向 23 passed；2.19 聚合其余调用点仍待领。 |
| `compat` | `ad46690d` 为 7.12 引入显式 `TorchTensorState` owner，保留旧 leaf/retained/optimizer 别名，状态迁移测试 4 passed；7.12 requires_grad/模块边界与独立包验收仍待领。 |

### 2026-09-05 第一百八十波

| 分区 | 结果 |
| --- | --- |
| `compat` | `696e5088` 让 7.12 的 requires_grad 状态由 `TorchTensorState` owner 持有，setter/FSDP peer 更新统一走 owner；生命周期定向 6 passed。独立 torch 包边界与完整 requires_grad 语义仍待领。 |
| `coreops` | `adf96b02` 将 `reindex_reduce` 负 shape 边界改为用户错误，已有负向与结构门禁通过；2.19 聚合仍待领。 |
| `device` | `6e5c2d5c` 将 CPU `clamp` 路径接入 BackendRegistry/OpRegistry 真实分派，registry 合同 5 passed 且 CPU 数值通过；CUDA/ACL 路由与完整 4.03/4.04 仍待领。 |
| `gates` | `faad4898` 将 smoke 独立组切到 xdist loadgroup，11 个 nodeid 与 loadfile 结论逐条 IDENTICAL；完整 smoke 仍未达到 300s 目标。 |

### 2026-09-05 第一百八十一波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `409de4ea` 建立 RuntimeContext owner 与只读 RuntimeState view，`sync_run` owner/snapshot/flag_scope 结构 3 passed；2.13 其余 flags 仍待领。 |
| `compat` | `756a0fb6` 让 `stop_grad` 清理 TorchTensorState requires_grad owner，修复 detached requires_grad 残留；Torch state/autograd 定向 37 passed，7.12 整项仍待领。 |
| `device` | `db0f2a27` 修正真实 CPU `Var.location() == "none"` 的 backend 识别，并补 outer/clamp registry 数值回归；定向 7 passed，4.03/4.04 整项仍待领。 |
| `gates` | `faad4898` 将 smoke 独立组切换到 loadgroup，结论对比 11/11 IDENTICAL；0.15 最终时长目标仍待领。 |

### 2026-09-05 第一百八十二波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `f37da269` 为 RuntimeContext 增加只读 `device_id`，CPU fallback 为 -1；snapshot/pyi/结构测试 4 passed。2.13 其余 flags 仍待领。 |
| `device` | `f37da269` 将 CPU `flatten` 接入 Backend/OpRegistry，补真实 shape/value 与 registry 合同；合同 8 passed。4.03/4.04 完整 native 接线仍待领。 |
| `gates` | 0.15 loadgroup 进一步审计确认共享状态组不可再拆，扩 worker 收益递减；本波无新代码提交，最终 300s 目标仍待领。 |
| `compat` | 本波未形成安全的独立 torch 发布边界提交，保留 7.12 当前状态。 |

### 2026-09-05 第一百八十三波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `1c57b2a0` 将 `use_cuda` 纳入 RuntimeContext owner，结构回归 5 passed；2.13 其余执行状态仍待领。 |
| `device` | `32e8517b` 增加 OpRegistry `has_kernel`/`unregister` 生命周期与 MissingKernel 错误，结构 9 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `41eb41c2` 建立独立 TorchNamespace seam，2 passed；默认 `torch is jittor` 兼容路径保留，7.12 完整模块迁移仍待领。 |
| `gates` | `f239e2ed` 增加可执行 smoke budget/bottleneck 报告，结构 13 passed，预测 445.75/480s；最终真实性能验收仍待领。 |

### 2026-09-05 第一百八十四波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `ac641485` 将 `lazy_execution` 纳入 RuntimeContext owner，嵌套 flag_scope/只读约束结构 6 passed；2.13 其余状态仍待领。 |
| `device` | `86b9c1cd` 支持 `cuda`/`cuda:<id>` location，未知 backend 明确 `UnknownBackend`，结构 10 passed；4.03/4.04 完整 native 接线仍待领。 |
| `compat` | `330d0a4c`/`9bbf87aa` 让显式 activation 发布独立 TorchNamespace 并接通事务替换；namespace 2 passed、bootstrap 1 passed，默认 shim 身份保持不变，完整 7.12 仍待领。 |
| `gates` | `5c6876cd` 将 cgroup v1/v2 CPU quota 纳入 worker budget，结构 22 passed；0.15 最终时长仍需真实全量验收。 |

### 2026-09-05 第一百八十五波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `8b1c2707` 将 `auto_flush_ops` 纳入 RuntimeContext owner，snapshot/类型声明/flag_scope 结构 7 passed；2.13 其余状态仍待领。 |
| `device` | `ba2c88e5` 增加 `unregister_backend` 的 kernel 原子 teardown，重复/未知注销 fail-closed；结构/行为 11 passed。4.03/4.04 native 全量接线仍待领。 |
| `compat` | `d52f02ac` 锁定 activation 后的 native/independent namespace 模式，冲突切换硬失败；bootstrap/namespace 11 passed。7.12 完整模块迁移仍待领。 |
| `gates` | `cab14f53` 将实际 cgroup v1/v2 quota 纳入 xdist worker 上限，结构/环境 24 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百八十六波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `78fec631` 将 `no_grad` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 8 passed；2.13 其余状态仍待领。 |
| `device` | `b83e6889` 支持 provider replacement 并在替换时原子清理旧 kernels，结构/行为 13 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `b84498c4` 将独立 TorchNamespace 子模块绑定纳入事务与失败回滚，定向 9 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `c7bfe24b` 在 pytest 前按 worker/cgroup 预算 fail-closed，结构/环境 24 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百八十七波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `04aae3a5` 将 `gopt_disable` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 9 passed；2.13 其余状态仍待领。 |
| `device` | `6ac1fe8c` 增加 capability dispatch 与 MissingCapability fail-closed，registry 合同 15 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `040b39cf` 修复 TorchNamespace registry root 的事务回滚，定向 namespace/transaction 19 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `de78f9ae` 暴露 runtime worker/quota/线程诊断，结构/环境 25 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百八十八波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `8eb8b7b2` 将 `exec_called` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 10 passed；2.13 其余状态仍待领。 |
| `device` | `a1f5e649` 原子发布/撤销 backend capability，保留 hooks/kernels，registry 合同 17 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `20cea34c` 修复 TorchNamespace 根条目绑定和回滚，namespace 定向 8 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `1ecb35ff` 区分 configured workers 与 quota 截断后的 runtime workers，结构/环境 26 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百八十九波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `61ae6160` 将 `use_threading` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 11 passed；2.13 其余状态仍待领。 |
| `device` | `df1743ff` 增加 capability 原子撤销，dispatch 继续 fail-closed，registry 合同 18 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `fda7501c` 对不完整 TorchNamespace 子模块发布 fail-closed，namespace 定向 9 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `bee0263e` 对齐 configured/runtime workers 并校验预算参数，结构/环境 27 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `7352b82d` 将 `profile_memory_enable` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 12 passed；2.13 其余状态仍待领。 |
| `device` | `6f1a9f35` 深冻结 capability snapshot，增加 BackendRegistry/OpRegistry snapshot，registry 合同 20 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `49906c5b` 预检 TorchNamespace 完整父级闭包，避免无事务半发布，namespace 10 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `895f69a9` 在 pytest 前输出 effective smoke budget，结构/环境 28 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十一波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `ce1b9276` 将 `profiler_warmup` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 13 passed；2.13 其余状态仍待领。 |
| `device` | `18ddbd72` 增加 RegistrySnapshot/snapshot_state 生命周期一致性，registry 合同 21 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `c6409827` 隔离独立 TorchNamespace root alias，namespace 11 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `f3ef7a70` 统一 nox/预算报告/CLI runtime worker policy，结构 18 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十二波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `e8608976` 将 `no_fuse` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 14 passed；2.13 其余状态仍待领。 |
| `device` | `537968fa` 隔离 RegistrySnapshot/provider state 输入与生命周期，registry 合同 23 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `e8cf42d6` 修复 TorchNamespace alias 预检回滚，namespace 12 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `91ca525f` 对齐 budget report 默认配置与 nox runtime workers，结构/环境 32 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十三波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `a2f29c02` 将 `check_graph` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 15 passed；2.13 其余状态仍待领。 |
| `device` | `6db5a349` 增加 RegistrySnapshot 所有权不变量，registry 合同 25 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `b77217e8` 为独立 TorchNamespace 根补标准 ModuleSpec，namespace 13 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `863f013f` 统一 standalone runner 与 nox 执行策略，结构/环境 26 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十四波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `e59adcdf` 将 `profiler_enable` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 16 passed；2.13 其余状态仍待领。 |
| `device` | `f46fa787` 收紧 RegistrySnapshot ownership 输入并补独立快照合同，registry 27 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `3eca5513` 隔离 TorchNamespace `__spec__`/`__file__` 等元数据，namespace 14 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `f7770109` 对齐 standalone runner 的 worker/quota policy，结构 15 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十五波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `a13b9cf6` 将 `profiler_rerun` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 17 passed；2.13 其余状态仍待领。 |
| `device` | `148ea645` 增加 RegistrySnapshot provider 查询与 teardown 隔离，registry 合同 28 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `2831a09e` 对称化 TorchNamespace `__delattr__` ownership，namespace 16 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `4dc47dae` 统一 standalone runner 六类线程池 policy，结构 8 passed；0.15 完整时长仍待最终验收。 |

### 2026-09-05 第一百九十六波

| 分区 | 结果 |
| --- | --- |
| `coreops` | `8aae611d` 将 `profiler_record_peek` 纳入 RuntimeContext owner，snapshot/只读/flag_scope 结构 18 passed；2.13 其余状态仍待领。 |
| `device` | `3add5deb` 收紧 RegistrySnapshot provider query 参数并补 teardown 合同，registry 29 passed；4.03/4.04 native 全量接线仍待领。 |
| `compat` | `8699c440` 修复删除 TorchNamespace 元数据后的 owner 泄漏，namespace 17 passed；7.12 完整模块迁移仍待领。 |
| `gates` | `5d79118f` 让 standalone/nox 共用线程池变量和 budget policy，结构 44 passed；0.15 完整时长仍待最终验收。 |

## 7. 接手怎么开始

0. 派活的话术、验收该问什么、哪些说法会让它跑偏，在 [怎么派活](refactor-dispatch.md)。
1. 按第 0 节确认分支与工作目录，读 `$LAB/AGENT-BRIEF.md`。
2. 不再重复第 2、3 节的 WIP 清理；直接从第 6 节按看板领下一波。
3. 每波结束把看板与本文更新到真实状态。`$LAB/clean-idle-caches.sh` 按 mtime 清空闲缓存，长跑期间挂着跑——**不要按体积
   挑**，体积大的往往正是在跑的那个。

## 8. 这轮的教训已经在 skill 里

29 个 skill 覆盖了这轮踩过的坑。跟并行执行本身最相关的三个，派活前让每个 agent 都读：

- `git-worktree-shared-state`——`git stash` 的栈是所有 worktree 共用的，两个 agent 的 WIP 会整包对调。
- `jittor-worktree-verification`——可编辑安装的 `.pth` 指向主树，worktree 里裸跑 `python -c "import jittor"`
  导入的是主树而不是你改的代码。
- `verifying-a-gate-actually-ran`——绿不等于跑过；先确认用例真的被实例化了。

还有一条不是 skill 而是纪律：**rebase 冲突要逐块解，不要整文件取一侧**。这轮有一次整文件解冲突静默
回退了 40 个已合并的 handler，是靠 `tests/structure` 才发现的。

## 9. 环境残留

- 主树 `/`（`2.0` 分支）上有**三个属于别人的文件**不要提交：`agent/manuals/README.md`、
  `tests/core/test_setitem.py`、`agent/results/2026-08-12-repository-modernization-review.md`。
- 停 agent 时清掉了 19 个超时的 python/pytest 残留进程。若发现构建整体变慢而非失败，先查有没有进程
  卡在编译锁上（ptrace 停住的 gdb、`futex_wait_queue` 里的 pytest 都出现过），症状是「所有人都变慢」。
