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
| 分支 | `2.0-refactor`；本波审计基线 `86733525`，后续状态提交接在其上 |
| 相对 `2.0` 的提交 | 审计基线 731 个 |
| 提交里出现过的任务号 | 233 个 |
| 看板 | 已合并 **196** / 进行中 **0** / 待领 **74** / 并入其它任务 **5** |
| 沉淀的 skill | `agent/skills/` 下 **29** 个 |

**交接清理完成不等于整改完成。** 看板仍有 74 条待领；当前只是把中断留下的易失状态全部转成了主线提交、
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

`broadcast_to_op.cc` 当前源码含 5 个 `USER_CHECK`，但 `MIGRATED_DIMENSION_BOUNDARIES` 仍期望 2（另一个 shape map 期望 5），因此 `test_error_categories.py` 仍为 1 failed；待专门门禁任务统一分类/计数，不能在本波掩盖。

第二十九波新增 3 个严格保持待领的前置：

| 分区 | 第二十九波结果 |
| --- | --- |
| `device` | 8.06 将 Roll family 接入共享 launcher，保留 shifts/dims 构造、销毁与同步策略；静态合同 16 项通过，本机无 CANN/NPU |
| `bindings` | 2.19 将 `reuse_np_array` 的类型与 C-contiguous 两处输入检查改为可捕获用户错误，累计 55 处；结构、TU 语法和两个负向节点通过 |
| `compat` | 7.03 将 pdist 提升为 numerical 稳定对象并登记 approximate fidelity；3 项 CPU/身份/metadata 通过 |

## 6. 下一波起点

按 [派活说明](refactor-dispatch.md) 每波最多四分区、每分区最多五项。第四十九波继续优先可独立验证的
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
