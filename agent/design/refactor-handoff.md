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
| 分支 | `2.0-refactor`；当前状态基线 `6d694853`，后续状态提交接在其上 |
| 相对 `2.0` 的提交 | 当前 1067 个 |
| 提交里出现过的任务号 | 329 个 |
| 看板 | 已合并 **199** / 进行中 **0** / 待领 **75** / 并入其它任务 **5** |
| 沉淀的 skill | `agent/skills/` 下 **29** 个 |

**交接清理完成不等于整改完成。** 看板仍有 75 条待领；当前只是把中断留下的易失状态全部转成了主线提交、
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
| `bindings` | 2.19 将 CUDA curand dtype/type 两处公开合同改为 `USER_CHECK`，累计 60 处；结构、nvcc TU 语法通过，本机无 CUDA，负向运行待硬件 |
| `compat` | 7.03 将 std_mean/var_mean 提升为 numerical 稳定对象并登记 approximate fidelity，明确 correction/keepdim 限制；3 项 CPU/身份/metadata 通过 |

第三十四波新增 3 个严格保持待领的前置：

| 分区 | 第三十四波结果 |
| --- | --- |
| `device` | 8.06 将 Scatter forward family 接入共享 launcher，Gather 保持原路径；静态合同 21 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN descriptor dtype 合同改为 `USER_CHECK`，累计 61 处；结构与头语法通过，本机无 CUDA |
| `compat` | 7.03 将 mv 提升为 numerical 稳定对象并登记 approximate fidelity，保留 out/错误语义；4 项 CPU/身份/metadata 通过 |

第三十五波新增 3 个严格保持待领的前置：

| 分区 | 第三十五波结果 |
| --- | --- |
| `device` | 8.06 将 Concat forward family 接入共享 launcher，SplitWithSize 保持原路径；静态合同 22 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 输入/权重 dtype 一致性改为 `USER_CHECK`，累计 62 处；结构与 TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 addmm 提升为 numerical 稳定对象并登记 approximate fidelity，保留 alpha/beta；3 项 CPU/身份/metadata 通过 |

第三十六波新增 2 个严格保持待领的前置：

| 分区 | 第三十六波结果 |
| --- | --- |
| `device` | 8.06 将 SplitWithSize 接入共享 launcher，保留 splitSize/tensor-list/dim 与同步策略；静态合同 23 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 Cutt transpose axes 长度/掩码两处检查改为 `USER_CHECK`，累计 64 处；nvcc TU 语法和结构通过，本机无 CUDA |
| `compat` | 7.03 将 mm 提升为 numerical 稳定对象并登记 approximate fidelity，保留 2-D/out 限制；3 项 CPU/身份/metadata 通过 |

第三十七波新增 3 个严格保持待领的前置：

| 分区 | 第三十七波结果 |
| --- | --- |
| `device` | 8.06 将 Nonzero owner 接入共享 launcher，SWhere 保持原路径；静态合同 24 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 66 处；nvcc TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 trapz/trapezoid 提升为 numerical 稳定包装并登记 approximate fidelity；4 项 CPU/身份/metadata 通过 |

第三十八波新增 3 个严格保持待领的前置：

| 分区 | 第三十八波结果 |
| --- | --- |
| `device` | 8.06 将 Range family 接入共享 launcher，保留三个 scalar 构造/销毁与同步策略；静态合同 25 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS batched MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 68 处；nvcc TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 masked_select 提升为 numerical 稳定对象并登记 approximate fidelity，保留 1-D/out 限制；3 项 CPU/身份/metadata 通过 |

第三十九波新增 3 个严格保持待领的前置：

| 分区 | 第三十九波结果 |
| --- | --- |
| `device` | 8.06 将 Dropout forward family 接入共享 launcher，backward 保持原路径；静态合同 26 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuBLAS acc MatMul 两处公开 dtype 合同改为 `USER_CHECK`，累计 70 处；nvcc TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 narrow 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第四十波新增 3 个严格保持待领的前置：

| 分区 | 第四十波结果 |
| --- | --- |
| `device` | 8.06 将 LeakyReLU forward family 接入共享 launcher，backward 保持原路径；静态合同 27 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuSPARSE CSR 两处公开 dtype 合同改为 `USER_CHECK`，累计 72 处；nvcc TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 tile 提升为 numerical 稳定对象并登记 approximate fidelity，保留 tuple/list dims；3 项 CPU/身份/metadata 通过 |

第四十一波新增 3 个严格保持待领的前置：

| 分区 | 第四十一波结果 |
| --- | --- |
| `device` | 8.06 将 ArgReduce max/min 两路径统一接入共享 launcher，保留双输出和同步策略；静态合同 28 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuSPARSE COO 两处公开 dtype 合同改为 `USER_CHECK`，累计 74 处；nvcc TU 语法通过，本机无 CUDA |
| `compat` | 7.03 将 diff 提升为 numerical 稳定对象并登记 approximate fidelity，保留 n/dim/prepend/append；3 项 CPU/身份/metadata 通过 |

第四十二波新增 3 个严格保持待领的前置：

| 分区 | 第四十二波结果 |
| --- | --- |
| `device` | 8.06 将 Random uniform/normal 两分支统一接入共享 launcher，保留 seed/offset 与同步策略；静态合同 29 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 NCCL reduce_scatter 两处公开 shape 合同改为 `USER_CHECKop`，累计 76 处；nvcc TU 语法通过，本机无 NCCL |
| `compat` | 7.03 将 square 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

第四十三波新增 3 个严格保持待领的前置：

| 分区 | 第四十三波结果 |
| --- | --- |
| `device` | 8.06 为 `aclnn.h` 增加 `#pragma once`，补重复包含静态合同；1 passed，本机无 CANN/NPU，仍待 Ascend 910B3 实机 |
| `bindings` | 2.19 将 CUB argsort/arg_reduce 的 offsets dtype 边界改为 `USER_CHECK`，累计 79 处；结构与双 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `pairwise_distance` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份、metadata、CPU p=2/keepdim 三节点通过 |

第四十四波新增 3 个严格保持待领的前置：

| 分区 | 第四十四波结果 |
| --- | --- |
| `device` | 8.06 将 SiLU forward 接入共享 launcher，backward/Swish/SwiGlu 保持原路径；结构合同 31 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution forward 格式边界改为 `USER_CHECK`，累计 80 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `cosine_similarity` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，三节点动态测试因首次 JIT 编译过久终止，未宣称通过 |

第四十五波新增 3 个严格保持待领的前置：

| 分区 | 第四十五波结果 |
| --- | --- |
| `device` | 8.06 将 BatchMatMul 接入共享 launcher，保留 `cube_math_type` 与同步策略；结构合同 32 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution backward-x 格式边界改为 `USER_CHECK`，累计 81 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `svd` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态三节点因首次编译过久终止，未宣称通过 |

第四十六波新增 3 个严格保持待领的前置：

| 分区 | 第四十六波结果 |
| --- | --- |
| `device` | 8.06 将 RotaryPositionEmbedding forward 接入共享 launcher，保留三输入与同步策略；结构合同 33 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN convolution backward-w 格式边界改为 `USER_CHECK`，累计 82 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `svd_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行 |

第四十七波新增 3 个严格保持待领的前置：

| 分区 | 第四十七波结果 |
| --- | --- |
| `device` | 8.06 将 Maxpool forward 接入共享 launcher，保留 descriptors、`poolCeil`、同步策略及 Avgpool/backward 原路径；静态合同 34 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 输入 rank 边界改为 `USER_CHECKop`，累计 83 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；`broadcast_to` 计数已同步为实际 5 |
| `compat` | 7.03 将 `pca_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行 |

第四十八波新增 3 个严格保持待领的前置：

| 分区 | 第四十八波结果 |
| --- | --- |
| `device` | 8.06 将 Avgpool forward 接入共享 launcher，保留 descriptors、`poolCeil/divisor`、同步策略及 backward/其他 pool owner；静态合同 35 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 84 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `nan_to_num_` 提升为 numerical 稳定 in-place 对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，因既有 NaN/Inf JIT abort 风险未运行动态测试 |

第四十九波新增 3 个严格保持待领的前置：

| 分区 | 第四十九波结果 |
| --- | --- |
| `device` | 8.06 将 TruthReduce all/any 接入共享 launcher，保留双路径异常处理与同步策略；静态合同 36 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 85 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `sparse_coo_tensor` factory 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态测试未运行 |

第五十波新增 3 个严格保持待领的前置：

| 分区 | 第五十波结果 |
| --- | --- |
| `device` | 8.06 将 Conv2d forward 接入共享 launcher，保留 group/bias/descriptor 与同步策略，backward 不变；静态合同 37 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-x dy rank 边界改为 `USER_CHECKop`，累计 86 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `randint_like` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十一波新增 3 个严格保持待领的前置：

| 分区 | 第五十一波结果 |
| --- | --- |
| `device` | 8.06 将 RmsNorm forward 接入共享 launcher，保留 `eps`、双输出与同步策略，gradient owner 不变；静态合同 38 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d backward-w 的 dy rank 边界改为 `USER_CHECKop`，累计 87 处；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `det` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十二波新增 3 个严格保持待领的前置：

| 分区 | 第五十二波结果 |
| --- | --- |
| `device` | 8.06 将 RmsNormGrad 接入共享 launcher，保留多输入、双输出与同步策略，gradient owner 不变；静态合同 39 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 权重 rank 边界改为 `USER_CHECKop`，累计 88 处、四十一组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `inverse` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十三波新增 3 个严格保持待领的前置：

| 分区 | 第五十三波结果 |
| --- | --- |
| `device` | 8.06 将 Softmax backward 接入共享 launcher，保留 `dim` query 与同步策略；静态合同 40 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort 的 x/indexes rank 边界改为 `USER_CHECK`，累计 89 处、四十二组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `take_along_dim` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十四波新增 3 个严格保持待领的前置：

| 分区 | 第五十四波结果 |
| --- | --- |
| `device` | 8.06 将 Embedding backward 接入共享 launcher，保留 `numEmbeddings`、`paddingIdx`、`scaleGradByFreq` 与同步策略；静态合同 41 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort 循环内 x/indexes shape 边界改为 `USER_CHECK`，累计 90 处、四十三组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
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
| `bindings` | 2.19 将 CUB argsort offsets rank 边界改为 `USER_CHECK`，累计 91 处、四十四组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `reciprocal` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十六波新增 3 个严格保持待领的前置：

| 分区 | 第五十六波结果 |
| --- | --- |
| `device` | 8.06 将 RotaryPositionEmbedding gradient 接入共享 launcher，保留四输入、三输出 query 与同步策略；静态合同 43 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB argsort offsets 长度边界改为 `USER_CHECKop`，累计 92 处、四十五组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `lerp` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十七波新增 3 个严格保持待领的前置：

| 分区 | 第五十七波结果 |
| --- | --- |
| `device` | 8.06 将 Conv2d backward 接入共享 launcher，保留三输出 gradient query、descriptor cleanup 与同步策略；静态合同 44 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB arg-reduce offsets rank 边界改为 `USER_CHECKop`，累计 93 处、四十六组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十八波新增 3 个严格保持待领的前置：

| 分区 | 第五十八波结果 |
| --- | --- |
| `device` | 8.06 将 UpsampleNearest2d backward 接入共享 launcher，保留 output/input-size RAII descriptor 与同步策略；静态合同 45 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 CUB arg-reduce offsets 长度边界改为 `USER_CHECKop`，累计 94 处、四十七组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `log_softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第五十九波新增 4 个严格保持待领的前置：

| 分区 | 第五十九波结果 |
| --- | --- |
| `device` | 8.06 将 LeakyReLU backward、SiLU backward 分别接入共享 launcher；静态合同 46/47 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN LSTM mode 边界改为 `USER_CHECKop`，累计 95 处、四十八组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `relu` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十波新增 3 个严格保持待领的前置：

| 分区 | 第六十波结果 |
| --- | --- |
| `device` | 8.06 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 非 LSTM mode 边界改为 `USER_CHECKop`，累计 96 处、四十九组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `torch._shape_as_tensor` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十一波新增 2 个严格保持待领的前置；compat 复核未找到安全的第三项：

| 分区 | 第六十一波结果 |
| --- | --- |
| `device` | 8.06 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN `proj_size==0` 边界改为 `USER_CHECKop`，累计 97 处、五十组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 复核剩余 API 后仅 `vmap` 是复杂闭包，其余为已有原生 owner，未产生安全代码提交 |

第六十二波新增 3 个严格保持待领的前置：

| 分区 | 第六十二波结果 |
| --- | --- |
| `device` | 8.06 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 98 处、五十一组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `outer` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十三波新增 3 个严格保持待领的前置：

| 分区 | 第六十三波结果 |
| --- | --- |
| `device` | 8.06 将 LayerNorm forward 接入共享 launcher，保留 `normalizedShape`、`eps`、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv3d 分组通道 shape 边界改为 `USER_CHECKop`，累计 99 处、五十二组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `isin` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十四波新增 3 个严格保持待领的前置：

| 分区 | 第六十四波结果 |
| --- | --- |
| `device` | 8.06 将 LayerNorm backward 接入共享 launcher，保留 `normalizedShape`、`outMask`、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x LSTM mode 边界改为 `USER_CHECKop`，累计 100 处、五十三组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `tensordot` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十五波新增 3 个严格保持待领的前置：

| 分区 | 第六十五波结果 |
| --- | --- |
| `device` | 8.06 将 GroupNorm forward 接入共享 launcher，保留 group/eps、三输出 query 与同步策略；静态合同 53 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x `proj_size==0` 边界改为 `USER_CHECKop`，累计 101 处、五十四组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 将 `repeat_interleave` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行 |

第六十六波新增 2 个严格保持待领的前置；compat 复核未找到安全小切片：

| 分区 | 第六十六波结果 |
| --- | --- |
| `device` | 8.06 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x 非 LSTM mode 边界改为 `USER_CHECKop`，累计 102 处、五十五组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 复核剩余候选仅有复杂 `vmap` 闭包，未强行拆分 |

第六十七波新增 2 个严格保持待领的前置；compat 复核仍未找到安全小切片：

| 分区 | 第六十七波结果 |
| --- | --- |
| `device` | 8.06 将 Avgpool backward 接入共享 launcher，保留 `countIncludePad/divisorOverride`、descriptor cleanup 与同步策略；静态合同 55 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN backward-x 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 103 处、五十六组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 复核剩余 API 后仅 `vmap` 为复杂闭包，本波无安全小切片提交 |

第六十八波新增 3 个严格保持待领的前置：

| 分区 | 第六十八波结果 |
| --- | --- |
| `device` | 8.06 将 Maxpool backward 接入共享 launcher，保留 pool descriptors、`poolCeil`、输出处理、cleanup 与同步策略；静态合同 56 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 输入 rank 边界改为 `USER_CHECKop`，累计 104 处、五十七组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 新增 `vmap` owner 迁移设计前置，记录 Runtime 依赖与后续 CPU 验收；未宣称实现完成 |

第六十九波新增 3 个严格保持待领的前置：

| 分区 | 第六十九波结果 |
| --- | --- |
| `device` | 8.06 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 权重 rank 边界改为 `USER_CHECKop`，累计 105 处、五十八组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 细化 `vmap` owner 的可验证契约与验收节点；仅设计前置，未修改 runtime |

第七十波新增 3 个严格保持待领的前置：

| 分区 | 第七十波结果 |
| --- | --- |
| `device` | 8.06 将 Index 接入共享 launcher，保留 index query 与同步策略，SliceV2 未改；静态合同 58 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv 分组通道 shape 边界改为 `USER_CHECKop`，累计 106 处、五十九组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 补充 `vmap` owner 提取协议、AST 完成门禁与 `VmapContext` 约束；仅设计前置，未修改 runtime |

第七十一波新增 3 个严格保持待领的前置：

| 分区 | 第七十一波结果 |
| --- | --- |
| `device` | 8.06 将 SliceV2 接入共享 launcher，保留 begins/ends/steps/axes descriptors 与同步策略，Index/其他 owner 未改；静态合同 59 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 107 处、六十组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 连续补充 vmap 设计契约与 unsupported AST 静态门禁；仅设计/门禁前置，未修改 runtime，未宣称实现完成 |

第七十二波新增 3 个严格保持待领的前置：

| 分区 | 第七十二波结果 |
| --- | --- |
| `device` | 8.06 将 StridedSliceAssignV2 接入共享 launcher，保留 gradient memset 分支与 slice descriptor handling；静态合同 60 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-x dy rank 边界改为 `USER_CHECKop`，累计 108 处、六十一组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 补充 vmap context 夹具契约、提取顺序、绑定与回滚步骤；仅设计前置，未修改 runtime |

第七十三波新增 3 个严格保持待领的前置：

| 分区 | 第七十三波结果 |
| --- | --- |
| `device` | 8.06 将 InplaceMaskedScatter 接入共享 launcher，保留 tracked base-to-output memcpy 依赖与同步策略；静态合同 61 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 109 处、六十二组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 7.03 补充 vmap 评审证据清单，覆盖 AST、closure/global、fidelity、聚焦节点与 skip 归因；仅设计前置，未修改 runtime |

第七十四波新增 3 个严格保持待领的前置：

| 分区 | 第七十四波结果 |
| --- | --- |
| `device` | 8.06 将 IndexPutImpl 接入共享 launcher，保留 index tensor-list handling 与同步策略，IndexPutImplAccumulate 未改；静态合同 62 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN conv backward-w dy rank 边界改为 `USER_CHECKop`，累计 110 处、六十三组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 补充 vmap unsupported 行为矩阵，覆盖 extent/nested dim/非 bool/depth callback/out_dims；仅设计前置，未修改 runtime |

第七十五波新增 3 个严格保持待领的前置：

| 分区 | 第七十五波结果 |
| --- | --- |
| `device` | 8.06 将 IndexPutImpl accumulate 接入共享 launcher，保留 tracked output memset 与 index tensor-list dependency；静态合同 63 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 推理阶段输入 rank 边界改为 `USER_CHECKop`，累计 111 处、六十四组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 明确 vmap 仅做组织重构，不新增 kernel/设备传输/优化，并定义 CPU/CUDA/ACL 分层验收与 skip 归因；仅设计前置 |

第七十六波新增 3 个严格保持待领的前置：

| 分区 | 第七十六波结果 |
| --- | --- |
| `device` | 8.06 将 AdamWList 各项更新接入共享 launcher，保留 fused D2D copy checks 与唯一同步点；静态合同 64 passed，本机无 CANN/NPU |
| `bindings` | 2.19 将 cuDNN RNN 输入通道 shape 边界改为 `USER_CHECKop`，累计 112 处、六十五组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
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
| `bindings` | 2.19 将 cuFFT jit_prepare unsupported dtype 边界改为 `USER_CHECK`，累计 113 处、六十六组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
| `compat` | 明确 vmap 版本兼容、kwargs 策略与退出标准；仅设计前置，未修改 runtime |

第八十波新增 2 个严格保持待领的前置；device 复核确认 KVCacheMemcpy 不适合通用 launcher：

| 分区 | 第八十波结果 |
| --- | --- |
| `device` | 8.06 复核确认标准 workspace/query/execute/sync owner 已迁移完毕；剩余 KVCacheMemcpy 为逐 token 专用 memcpy 路径，本波不改代码 |
| `bindings` | 2.19 将 CUBLAS matmul 输入 rank 边界改为 `USER_CHECK`，累计 114 处、六十七组证据；结构合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向 |
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
