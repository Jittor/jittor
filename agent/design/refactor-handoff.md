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
| 分支 | `2.0-refactor`；本批迁移起点 `2328ce4f`，后续提交见 Git 历史 |
| 相对 `2.0` 的提交 | 迁移起点共 1853 个；提交数不代表任务完成量 |
| 提交里出现过的任务号 | 329 个 |
| 看板 | 已合并 **221** / 进行中 **0** / 待领 **51** / 并入其它任务 **13** |
| 沉淀的 skill | `agent/skills/` 下 **34** 个目录 |

**交接清理完成不等于整改完成。** 看板仍有 51 条待领；当前只是把中断留下的易失状态全部转成了主线提交、
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

### 当前执行优先级（2026-09-06）

优先完成可供 CUDA/NPU 异机测试的模块迁移与调用链接线，不以提交数或新增校验器数量表示完成进度。
并行代理限定文件范围，只修改和验证；主协调者串行审查、暂存、提交、推送，禁止共享暂存区并发提交。
每批一次更新看板与交接，不再为追赶 HEAD 单独追加基准刷新提交。

2.13 已完成：计划点名的七项原生全局状态现归 NativeRuntime，核心/JIT/后端共用导出访问器。
弱同步游标、嵌套遍历恢复、fork 保留遍历编号及退出期 holder 生命周期保留原契约；
旧 `exe`/`tflag_count`/设备 flag 数据符号与 `misc/cuda_flags.*` 均已移除，外部扩展需要重编。
这些是真实原生存储迁移，不是前面旧波次逐字段添加的 Python 只读视图。

本次补齐 config/runtime 分层：启动配置 10 项在后端初始化与兼容层 composition 完成后冻结，
所有 native Flags 实例以及 Python compiler 公开旧字段均拒绝晚写；`jt.config` 深只读，
`jt.runtime` 的 66 项策略可写、5 项计数只读，scope 复用原有恢复机制，snapshot 不创建张量。
`cuda_kernel_math` 作为 NativeRuntime 策略进入普通/融合 JIT key，切换前提交旧策略下的图；
Torch 激活不再改 startup nvcc_flags，启动 preflight 也不再因此构建另一份核心。
分类文件纳入构建指纹；类型声明、旧入口约束和两个看板格式残留均已同步。

验证：CPU/CUDA/结构最终 109 passed（7.35s）；Torch bootstrap 47 passed（8.49s）；
CPU-only 与自定义扩展 23 passed/1 skipped（仅不存在的 cuda_archs 字段）。
真实 CUDA 覆盖默认/严格舍入差异、普通/融合 kernel 缓存分离与策略回退。
没有执行完整后端门禁，NPU/ROCm 仍需异机实测；既有 ACL 主机桩不代替 CANN ABI 验证。

2.14 也已完成：原生 misc 剩余 24 文件归入 debug/runtime/type/utils，目录消失，
核心、生成代码、内嵌 CUDA 与通信后端 include 全部同步，未优化算法。
修复转换缓存只覆盖不清理旧路径的问题：过期原生源移到缓存内的独立备份目录，
不再同时编译新旧实现；原始源码和非原生缓存不删除，迁移回归有修前失败证据。
CPU/CUDA/双卡定向 41 passed；CPU-only 4 passed/1 个 CUDA 节点跳过；
ACL 两 TU 主机语法和负向对照通过，ROCm 两 ABI blob/Corex 目录特判复核通过。
这不代表 2.23 已完成：init/profiler/lock 与 pyjt/pybind 的布局仍未迁移。

4.03 已接通：NativeRuntime 里的版本化 BackendRegistry 是实际 CPU/CUDA 设备与内存操作入口，
provider 位于 `src/runtime/backends/`；公共层保留 SFRL/NFEF/Temp/Stat 组合，原始池由 provider 返回。
数组构造、共享迁移、DeviceCopy、fetch 和 swap 复制走回调，流/同步也已接线；
dual/delay-free 修正真实设备信息，跨卡双向定序、fetch block 持有和 callback 所属设备保留。
通过临时替换真实 callback 表观察张量执行，证明不是元数据旁路；测试 finally 恢复原表。
最终 CPU/CUDA/结构 40 passed，双卡/五库/梯度/共享 39 passed，无 GPU 可见 26 passed，
CPU-only 31 passed/1 个 CUDA 配置字段跳过。未跑完整模型门禁或 NPU/ROCm 实机；
legacy 源转换仍保留，BackendId 枚举已避开旧转换器的 CUDA→ACL 重名问题。

4.04 已将 OpDef/Kernel/Codegen 接入原生执行器、并行编译器和融合 relay，
注册项绑定具体实现，旧虚调用仅保留为源码适配入口。CUDA 库自行登记语义能力，
核心五处可选算子替换与 matmul/conv tuner 不再硬编码库实现名字。
元算子与统一计算图保留，cuBLAS/cuDNN 接管的是图中的实际 relay。
三处真实回归已修：双源码 CodeOp 在选择后端前生成错误缓存键、融合图初次查 CPU 表、
替换生成实现后新旧图共用二进制。初始注册保持稳定缓存键，替换及同名重注册使用独立编译身份。
本轮还修复快速扩展的既有依赖漏记：两个 TU 共用一份 depfile 只保留最后一个源的头文件；
compile_module 现为单 TU，复用旧缓存也会自动重编。独立进程改头回归修前返回旧值、修后返回新值，
连同原生分派、身份生命周期与真实 CUDA 能力/relay 最终 17 passed。
CPU-only 的分派/身份/自定义扩展与相关结构检查 40 passed；ACL 改动 TU 主机语法通过。
完整 structure 在 CPU-only shim 配置下实际跑完：688 passed、9 failed、2 skipped；
本轮新增的两个跨测试文件 import 已移入 `_helpers`，该规则剩余的是此前 backend-registry 测试的 import。
其余失败涉及 compat 重复/吞错/publication、旧 flag-scope 识别、进程模式与 CPU 缺少 CUDA 字段，
未做整目录改前 A/B，不称全门禁通过；也没有执行完整模型门禁或 NPU/ROCm 实机验证。
4.05 已完成真实 Python 分派迁移：`_runtime.dispatch` 消费 native 无同步设备上下文，
按 backend、全部输入 dtype、shape/grad predicate 与优先级选择原始实现；错误不作为转路信号。
矩阵/卷积/RNN、归一化/推理、KV/RoPE、scan/indexing/gamma/FFT/AdamW 全部接表。
`_runtime.registry` 的假后端与 bytearray 原型已删除，27 个仅测原型的节点随原型退役，
真实 outer/clamp/flatten 数值节点保留并改查真正的表，新增真实 CPU/CUDA 分派回归。
库 module、ops 和资源由一个 owner 提供，旧 globals/hook 成为只读查询；ACL 直接发布真实 callback。
MKL 禁用在查询已加载模块及启动 loader 前统一检查，恢复启用可复用模块；不再隐式覆盖显式禁用。
旧 ROCm/Corex 允许的转换实现仍显式登记，不因此声称设备验证。原生算子域不再直接判旧设备 flag，
显式设备迁移/启动配置不混入这个计数，legacy FFT 的 mode 1 限制放入注册项。

最终 CPU/CUDA 专项 54 passed（5.93s），CPU-only 52 passed/8 个 accelerator skip；
原 matmul/推理能力及相关合同 61 passed/8 个未加载 MKL skip，额外卷积/RNN 对拍 2 passed。
补实际 MKL 小形状 batched matmul：选中真实库、profiler 记录该 kernel 且 NumPy 数值通过；
上述 8 skip 来自旧测试在惰性装载前固定判断，不能据此称本机无 MKL。Torch-CUDA RMS 路由 1 passed，
CPU shim 公共签名与模块组合 14 passed；MKL 禁用/恢复的真实输入控制及库合同 11 passed。
双卡 RNN 的自动 hx 和临时权重已由既有 pending scalar 跟随机制正确定位，无需增加 scope/helper。
完整 structure 实跑 678 passed/10 failed/2 skipped；本轮新增 softmax 签名变化已恢复原公开签名，
定向签名/身份节点通过。其余九类与上一轮记录相同，未重跑全量改前 A/B，不称全门禁通过。
只读 HOME 导入通过；同文件旧 TestCoreBuildStamp 仍试图改 immutable jittor_path，属于既有测试适配缺口。
未做完整模型门禁或 NPU/ROCm/Corex 实机。新接口及异机验收入口见 source-architecture 和 Ascend guide。

4.06 已接真实执行：NativeRuntime 的三态旗标默认 warn，非法赋值回滚；
executor 在 CPU kernel/输入迁移前决定，显式拷贝和 Array staging 不算回退。
ACL 完整预检后才执行，只有明确 unsupported 才请求 CPU；SDK/kernel/shape 执行失败清理后传播，
允许回退的 mode、op flags 和 fused context 用 RAII 恢复。修复已实现 runner 的映射键/直接入口，
没有借此复制 kernel 数学。Nox/生态/NPU scope 使用真实策略和尝试计数，删除日志文字验收；
被内部吞掉的拒绝仍会使验证失败，主异常不被 teardown 计数错误替换。
真实 CUDA 三态的 CPU marker、迁移前指针和恢复检查修前失败、修后连原生策略 9 passed；
严格 error 模式的常规 CUDA/BuildConfig/序列化组合 28 passed。
ACL 82 个 host/结构节点、4 个 TU、5 个 launcher ABI 和两次语法反向对照通过，未做 NPU 实机；
各 family 自持局部 SDK 资源的既有异常清理缺口仍归 8.06。
本波完整 structure 为 740 passed/9 failed/2 skipped；新增 setter 探针已补显式恢复，
其余九类与前次记录一致，未重跑完整改前 A/B，不称全门禁通过。

4.07 已完成：三个后端返回冻结 BuildConfig，BuildContext 注入编译/转换/库发布服务；
compiler 只在集中边界发布兼容字段，extra_core_files 是值内 tuple，后端不再修改 compiler。
entry point 发现只导入选中模块，多 SDK 冲突要求显式 JT_BACKEND；普通 CPU/CUDA 缓存键保持，
显式 CPU 在任何 CUDA 探测/安装服务前返回，已用冲突 nvcc_path 和禁止探测探针实际验证。
可选后端成功配置的通用 accelerator 编译标记不依赖本机是否有 NVIDIA 编译器，
Corex 兼容库路径保留；已选可选后端的初始工具链旧耦合仍明确列出，未猜测 vendor SDK 链接目录。
utils 反向 import 清零；真实序列化实现迁 jittor.serialization，旧 utility 模块由已注入服务查询。
jt.save/load、旧 GLOBAL 名与 stride 往返有实际测试；独立 pre-bootstrap 旧 utility API 现在明确报错，
不声称所有未 import Jittor 的独立旧 pickle 场景兼容，详见 backend-build-configuration.md。
构建/扩展依赖/序列化/CPU CUDA 分派组合 50 passed，strict CUDA/显式 CPU 组合 28 passed；
CPU-only 联合 82 passed，offline provider/分支 39 passed。测试文件重名已改为 native 前缀，
序列化新测试恢复 flags，未删除或重写用户缓存；完整 structure 仍为上述九类已记录失败。

4.10 已完成 CUDA 内核实体迁移：七库与核心 GPU 源、NN/misc/math/pool/sparse/CCL/loss3d
归顶层 `backends/cuda`，安装名 `jittor.backends.cuda`；NN 只留领域接口和分派。
共享索引与池化数学没有复制；GPU 前缀/调度和单独内核由真实注册项选择。
复合源码原子发布并保留原文件行号，旧 `type/cuda_atomic.h` 仅作 include 转发。
源码 bridge 与 C++ resource resolver 都优先 checkout，避免旧包残留造成实现混用；
sdist/wheel 包含所有新资源，legacy 转换缓存也纳入迁移，未靠删除用户缓存掩盖问题。
真实 CUDA 分派/库 13 passed，最终内核/推理/池化/深度卷积组合 27 passed；
原子展开、scatter/sparse 梯度与别名 14 passed。修复迁移时发现的原子宏丢失，
以及旧 candidate GPU 的 i/j 语义反转，均有最小修前失败/修后通过。
隔离安装 wheel 的 CPU-only 冷编译、索引/scatter/softmax、C++ schedule helper
及 `python -m jittor.selftest`（前后向、13 子包、三步训练）实际通过。
最终 wheel SHA-256 为 `e810fcb92d94179ef0bd5efc01426f54db393b5817b554a930db0fd32fab14d2`；
986 个生产文件与源码/安装结果逐字节相同，142 个后端文件无缺失或重复。
它与已实测 wheel 仅差 9 个行尾空白整理及 bridge/aliases 两个 Python 修复；
后两项连目录/身份定点 42 passed，未为这些非数学差异重复冷编译。
ACL 两 TU 与 5 个 launcher ABI 主机语法通过，不代替 CANN/NPU 实机。
完整 structure 实跑 765 passed/11 failed/2 skipped；本批新增 nan_checker 路径、
私有 NN alias 导出和跨测试 import 已定点修正，旧九类失败仍在，未再跑完整改前 A/B。
NCCL 通信资源、FlashAttention 外部集成、共享核心搬出包仍属 4.15，不能视为已完成。

4.11 代码阶段已完成，按用户授权把缺失的 NPU 真机/性能留给异机验证，不冒充已验收：
35 个 Python SDK builder 搬到 `backends/acl/kernels/ops`，实际 tensor/neural/normalization
实现模块级注册，旧 module aliases 同对象；native Module/函数/参数 owner 保留。
`change_function/warp`、全局 `do_compile_hook`、CUDA 算子删除清单与源码注释猜测均删除。
`Kernel.compile` 承担真实 ACL 路由，注册期 composer 同时适配现有与晚加载定义；
安装幂等/事务化，稳定启动身份与动态替换代次、旧图 pin 分开，HCCL 显式选择编译。
`jt.code` 的 `backend="acl"` 源标记进入键并由反向继承；第三方旧 `// aclop` 写法必须迁移。
post_process 不再覆盖四个全局，BackendOps ABI 2 的 pinned/并行/归约策略由实际 owner 消费。
typed 基本 Get/Set 新增正步 slice/整数/None/Ellipsis/empty/broadcast/return_x 的纯设备实现，
连续 suffix 合并，实际排队在 `aclstream`，不是默认流；只对核实的相同共享映射 noop。
不安全 overlap、advanced/string/negative-step、native reduction assignment 在写入前拒绝；
Python ACL 路径仍处理其既有变体，scalar 广播尚未优化。十层 cascade 的整体视图重写仍归 5.02。
基本索引反向显式保留赋值 dtype（标量零仍 0-D），advanced indexed-add 累加精度未改。

验证：CodeOp/注册 CPU 与真实 CUDA 30 passed；低精度索引/张量入口/CUDA 小内核 13 passed；
rotary/安装/cascade/策略 24 passed；最后 CPU-only 17 passed/2 个真实 CUDA skip。
Python provider 40、tensor 路由 10、纯 C++ copy-plan NumPy 对拍 38 项通过；
两个 ACL TU 语法与坏符号反向检查通过，不证明 CANN SDK ABI/设备执行。
最终 wheel SHA-256 为 `7556ec1218e38e23b81cb86657ce52f3896181fae9f56d4460d87aa2d7eefb2e`；
992 个生产文件与源码/安装结果逐字节相同，新 SDK 文件齐全、旧 Python 副本为零。
该最终安装包在隔离空缓存下完成 CPU-only 核心编译、前后向、13 子包导入与三步训练自检。
矩阵/归一化/路由组合 60 passed/1 failed；唯一归一化失败在迁移前 `fd4d8820`
独立源码、全新缓存、真实 CUDA 同一 node 复现：`test_norm_unification.py::TestNormParityCUDA::test_module_and_functional_agree`
在 scope 切换 flush 报 `broadcast_to doesn't have cpu version`；只证明既有，不提前认定根因。
完整 structure 793 passed/11 failed/2 skipped；新增 provenance 计数与 pool 分派导入合同
已定点修正（连 install 合同 8 passed），旧九类失败保留，未再重复完整目录。

下一轮以 4.12 移除 legacy SDK 源转换为主线：仍需真正的 ACL 设备/内存/流 provider、
构建服务以及 ROCm/Corex 原生适配，不能只删工具让可选后端断路。
分开“有加速器”与“依赖 CUDA SDK”的编译条件；完整 int64/类型化属性不能塞进
CodeOp 的 double DataMap，8.06 仍需真正 typed 属性消费和描述符缓存。

### 2026-09-06 4.12 WIP 收口

本轮工作树已经收成可交接的 4.12 前置，但**没有把 4.12 标成完成**。当前未提交内容包含：

- `BackendOps` ABI 3 的通用内存、拷贝、stream/event、host callback、架构查询、NaN hook、
  operator bootstrap 和执行策略字段；CPU 后端实现与共享 stream/事件状态已加入，CUDA
  SDK 实现正迁到顶层 `backends/cuda/runtime/`。
- `executor`、`init`、`event_queue`、`profiler`、核心内存/array/fetch/copy/pyjt 消费者
  已移除直接 CUDA SDK include，使用 `HAS_ACCELERATOR` 和 BackendOps；6 个无 CUDA SDK
  C++ TU 语法检查、共享消费者结构合同 2 passed。CUDA NaN TU 使用本机 CUDA 12.2
  host syntax 通过，未做运行时设备验收。
- ACL 侧新增懒初始化的 provider 目录和 workspace/stream 接线前置；ROCm 新增独立
  hipBLAS matmul、rocPRIM cumsum/library build spec，官方 HIP 6.2 头文件下 4 个 TU
  语法通过。MIOpen、RCCL、其余 rocPRIM family、完整低精度/batched GEMM 仍未迁移，
  不能登记为支持；Corex 的 converter 删除和真实 provider 仍在进行。
- 编译侧新增值型 `BuildSource/BuildConfig` backend source/link/language 字段和 native
  JIT compiler 配置入口；目标是选中后端直接编译自己的 TU，不再复制和文本改写核心源码。
  `process_jittor_source`、`process_acl` 尚未删除，ROCm/Corex 的所有旧调用也尚未清零。

本波随后已合入的独立前置：`d861351f0` 为 ACL 绑定自有 Device/Pinned allocator，拒绝
Managed memory（合同 5 passed）；`a672c1997` 为 Corex 保留 CUDA-compatible kernel ABI
标记和已审计 CUDA core kernel source roots（Corex/build 合同 16 passed）；`06190e2b3`
为 ROCm 建立独立 HIP provider、HIP runtime/allocator/stream/event/peer/meminfo，入口不再
解 `rocm_cache.tar.gz` 或调用源码转换，并只登记已有 hipBLAS/rocPRIM 实现；`8d06e85e6`
让共享 pyjt 数组桥接只在真实 CUDA backend 选择 CuPy，ACL/ROCm 不再误走 CUDA 互操作
（共享 build 合同 16 passed）。ROCm 本机无 HIP SDK，未做真实 hipcc/设备验证；MIOpen、RCCL
和其余库族仍明确未实现，不能视为后端完成。随后 `3b081d582` 已移除 ACL
`transform_sources/process_acl/token_replace` 的生产调用链，ACL host syntax
通过 43 个 TU 和 68 个 launcher ABI 检查；`79c6d716` 将 ROCm 旧入口改为原生
provider 转发，删除历史 blob/转换逻辑，ROCm 合同 23 项通过；`8752f15dd`
删除依赖旧 ACL converter 的测试。转换边界合同目前为 27 passed、2 xfailed，
两个 xfail 是 ACL/Corex entrypoint 尚未切到顶层 native provider。

**已失效（4.12 本波已收口）**：下面这段记的是收口前的扫描基线，三处已全部删除，
`test_core_source_is_not_ported.py` 接手钉住。两个 xfail 仍在，它们等的是 4.15/8.14。

当前源码扫描的剩余转换入口是：`jittor_utils.process_jittor_source` 定义及
`compiler.py` 服务字段各 1 处；ACL/ROCm/Corex provider 已不再调用转换。
顶层 ACL/Corex entrypoint 尚未切换到 `jittor.backends.*`，因此边界合同仍保留
两个 xfail。这个扫描结果是
4.12 的收口基线；完成判据是这些调用归零，而不是仅把函数改名或把替换逻辑搬到
另一个 Python 文件。

本 WIP 未执行 Git 提交前的首次 JIT，也未声称 ACL/ROCm/Corex 硬件验证。下一位接手时先检查
`git status` 中上述文件，按 ABI3 合同统一 provider 工厂和 canonical backend 名称
(`acl`/`rocm`/`corex`)，再做 CPU-only 核心语法和一次串行 CUDA 回归；只有三后端均不再
调用源码转换、其设备/编译入口有实际 owner、以及转换负向门禁通过后，才可将 4.12 改为已合并。
独立 torch 包仍要推进实际所有权迁移，不能再回到只写 metadata/validator 的旧波次。
NativeProviderRegistration 的旧元数据不能当作上述执行/构建迁移已完成的证据。
独立 torch 包和后端架构仍是未完成的大需求，不要为追低价值计数改变优先级。
异机 CUDA 先跑 `tests/core/test_startup_config.py`、`tests/backends/cuda/test_cuda_kernel_math_policy.py`
及 `tests/backends/cuda/test_multi_device.py`；NPU 依 `docs/guides/ascend-910b.md` 做真实构建/执行验收。
下文波次表保留为历史证据，不应作为当前已完成范围。

### 2026-09-06 3.02 / 3.03 收口（3.01 解锁的两条内存安全缺陷）

`9cdae20fc`（3.02）与 `0ad6c7504`（3.03）已合并推送，两条都标已合并。它们同时碰
`executor.cc`／`fused_op.cc` 一片代码，所以由一个 agent 串行做完，没有拆分区。

**3.02**：`JitKey` 的 2 MB 无检查缓冲与 mprotect 守护页换成堆分配、按需增长（8 KB 起）、
每个写入口先 `reserve()`、超过 flag `jit_key_max_size`（默认仍 2 MB）抛可捕获 `UserError`；
`utils/log.cc` 的 `protected_page` 与信号处理器里那段一并删除。以 `_` 开头的 loop option
改为也入键。**融合边变长编码那一项核实为 6.C05（`21a4f4fc`）已提前做完，`13ac1d14` 的四条
临时 `ASSERT(<256)` 在那个提交里就已删除**，本波只补键级判据。

**3.03**：`string_view_map` → `jit_cache_map`（键改自有 `string`、容量上限 + LRU、
flag `jit_cache_size` 默认 4096），`jit_fused_ops` 值改 `shared_ptr<FusedOpContext>`
使 context 可释放，`VarRelayManager::fop` 删除（改显式传参），
`execute_fused_prepared` 里每次命中重指向栈上 `FusedOp` 的那句一起删除。
加上淘汰之后原来正确的 `t[a] = t[b] = v` 三处变成 UB，全部拆成两句。

**四条值得记的判据**：

1. **「ASan 无悬垂」跑了真 ASan，办法是单编那个头文件。** `jit_cache_map.h` 除
   `common.h` 与一个 `DECLARE_FLAG` 外无依赖，`-I python/jittor/src -I <python include>`
   两个就能编，不需要建 jittor、不需要生成的 cfg 目录，一次编译加运行不到一秒。
   `tests/compiler/test_jit_cache_map_asan.py` 把它变成常驻用例，并**在同一个文件里再编
   一份旧写法断言 ASan 确实报它**——没有这条牙齿，libasan 缺失或场景没触发缺陷时第一条
   都会漂亮地通过。旧写法不带 sanitizer 也是错的：4096 个短键里 2017 个查不回来。
2. **验收那句「含 300 个算子的融合段与 200 个的键不同」比它看起来弱。** 实测把定长编码
   放回去，这条仍然通过——每个算子都有一条变长的 `«opkey<i>`，算子数不同就必然不同键。
   真正的判据是同算子数同 var 数、边结构不同（链 vs 平衡树）不得同键。**建议以后写这类
   验收句时把「同规模不同结构」写进去**，否则它会被一条空洞地通过的用例满足。
3. **限额做成 runtime flag 时，比较阈值必须在每次操作开头重读。** 第一版只在 `grow()`
   里重算，而缓冲区只增不减，于是在已经跑过大 key 的进程里把 flag 调小完全不生效，
   用例报 `did not raise`。改成 `clear()`（每个 key 一次）重算。
4. **改一个被广泛 include 的头文件会让 `test_cache_dependencies` 在复用缓存里假红。**
   它扫缓存目录里所有 `.key`，要求每个依赖路径只有一个哈希；改前改后的条目同时在那里
   就必然两个。`jittor-core-cpp-edit-loop` §4 记的判据是「失败指向一个你没碰过的头文件」，
   **本例正好相反**（指向 `op.h`，就是我改的），所以判据只能是换空 `JITTOR_HOME` 重跑：
   实测 18 failed / 391 passed，清单与基线逐字相同。

三套门禁（原生 CPU `tests/core`+`tests/ops`+`tests/compiler`、CPU torch shim
`tests/structure`、CUDA `tests/backends/cuda`）**FAILED 集合与本波开始时自测的基线
`f5a0ec5a1` 逐条相同**，passed 只涨不跌；`tests/core` 那三条存活 Var 数断言的数字两轮
不同而用例名集合相同（3.01 记过这个），判据用集合。两个提交各做了推前 CUDA 冒烟。
沉淀 skill：`jittor-core-cpp-edit-loop` 新增 §7ter／§7quater／§7quinquies。

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

> ### ⛔ 上面那个「2 条」已经过期，不要照它判断（2026-09-06 15:40 实测）
>
> 在 `4b9c8b5a2` 上实测是 **`15 failed, 878 passed, 2 skipped, 2 xfailed in 399s`**。套件在两小时内
> 从 512 个用例涨到 895 个（本波八个分区各自加了合同），**而多数失败是本波新引入的**。
>
> **所以判据变了**：不能再用「我的失败是否在下面那张旧表之外」。**每个分区都要检查失败清单里有没有
> 自己刚建的文件**。已经点名的两类新违规，请各自 owner 收掉：
>
> - `test_child_process_contract::test_every_child_launch_pins_this_tree` ←
>   `tests/structure/test_legacy_backend_build_config.py:215` 裸起解释器未钉 `PYTHONPATH`（违反 0.21，
>   改走 `_helpers.child_process`）
> - `test_flag_scope_contract::test_no_test_leaves_a_jittor_flag_changed` ← **七处**裸赋值 `jt.flags.*`
>   且无还原（违反 0.12，改用 `jt.flag_scope`）：`tests/backends/cuda/test_cuda_runtime_device_state.py`
>   的 `:15`、`:32`、`:36`；`tests/compiler/test_jit_math_policy.py` 的 `:12`、`:58`；
>   `tests/core/test_native_backend_registry.py:134`；`tests/structure/test_runtime_sync_state.py:672`
>
> 其余 13 条：`test_backend_execution_policy_contract`（ACL descriptor 未声明 execution 要求）、
> `test_compat_exception_policy::..._no_handler_body_is_only_pass`、`test_native_backend_contract` ×2
> （4.13 的执行者核实过：把本波的 `backend.h`/`backend.cc` 还原后仍以同样的 subprocess 超时失败）、
> `test_native_support_layout`、`test_process_mode_contract`、`test_pytest_contract` ×3、
> `test_runtime_holder_state`、`test_runtime_sync_state`、`test_src_third_party_layout`、
> `test_torch_compat_structure::..._exact_owner_whitelist`（这条是旧表里的）。
>
> **这次失误的责任在我（协调者）**：把一个两小时前的数字写成基线，等于让八个分区都以为自己是干净的。
> 基线只在同一棵树上有效——**改动前自己跑一次留基线**，比引用文档里的数字可靠。

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

### 3.22 归约口径已对齐：两边原来比的是两批几乎不相交的 kernel（量法 `9ab5ea42`）

上一条留下的「归约类 Jittor 0.57 ms 对 PyTorch 1.20 ms、快一倍以上，3.22 看起来已达到」
**不成立**。把两个桶拆开看，它们装的东西几乎没有交集：

- Jittor 的 0.57 ms 只有**代码生成器**的 `reduce` 角色（0.49 ms 通用求和 + 0.08 ms 六个
  注意力 GroupNorm 的回退）。手写 GroupNorm 1.72 ms、手写卷积偏置梯度 0.26 ms、
  手写 softmax 1.59 ms 全都不在里面。
- PyTorch 的 1.20 ms 是按符号名分的 `reduce/norm` 桶：0.65 ms 真归约，**加上** 0.47 ms
  的 GroupNorm **逐元素仿射写回**（三个 `GroupNorm*KernelImplInternal` 的
  `elementwise_kernel`，只因符号名含 `Norm`）；而 PyTorch 真正的 GroupNorm 统计量归约
  0.74 ms（`RowwiseMomentsCUDAKernel` 等四种）落在 `other`，一个都没进去。

对齐做法：`profile_step_torch.py --attribute`（新增）记 CPU 活动，用 `correlation` 把每个
CUDA kernel 接回发起它的 aten 算子栈，再按语义配对，**并用每步调用次数当配对成立的判据**。
三行全部对上：卷积偏置梯度 51:51、其余通用求和 67:67、GroupNorm 41:41。

**每边跑两到三次列范围。** profiler 的逐算子测量对机器负载敏感：一分钟负载 22 时手写
GroupNorm 报 1715 us，负载 9–13 的三次运行是 1533 / 1535 / 1544 us（高估 12%）。
nsys 稳得多，两次运行同一族 kernel 在 0.5% 以内。

| 类别 | Jittor profiler（三次） | Jittor nsys（两次） | PyTorch 2.12.1 CUPTI（三次） |
| --- | ---: | ---: | ---: |
| 通用求和（卷积/线性偏置梯度、广播梯度、全和） | 744.9–753.2 us | ~853 us | 652.6–681.9 us |
| GroupNorm 全部（统计量 + 仿射写回，41 个） | 1532.9–1543.8 us | ~1850 us | 1275.5–1315.6 us |
| **归约类合计** | **2279–2297 us** | **~2705 us** | **1928–1998 us** |
| 同次整步 | 21.19–21.35 ms | 23.09 ms | 21.11–21.95 ms |

**Jittor 慢 15%（profiler 口径）到 36%（nsys 口径），3.22 的验收未达成**；PyTorch 的实测
对应值也不是 1.13/1.20 ms 而是 1.93–2.00 ms。两种量法在手写 kernel 上系统性地差一截，
但方向与归因一致；严格同口径的一对是 **nsys 对 CUPTI**，两边都是真实流水的 kernel 轨迹。
**差距的 75% 在 GroupNorm**（profiler 口径 +241 us、nsys 口径 +555 us，两种量法算出的
占比都是 75%），其中 1.5–1.8 ms 在 `backends/cuda/kernels/nn/group_norm_cuda.py` 的手写
CUDA 里，**不在代码生成里**；通用求和那一栏只差 79–184 us。已在看板单列一行给手写
kernel 的 owner。Jittor 手写的 attention softmax（1.58 / 1.41 ms）在 PyTorch 侧没有对应
的独立 kernel（融进 `fmha_cutlassF/B`，与 QK^T、PV 同一个 kernel），单列不进合计。

**`edf70f52` 的处置：保留为 opt-in，不改默认，但它「慢 1.64%」的判据被推翻了。**
那四个「代表形状」不是 UNet 用的形状。在真实 UNet 上直接量（`--flag para_opt_level=4
--compile-option reduce_lvl4=1`）：`reduce` 角色三次 527.9 / 517.8 / 514.3（均值
**520.0 us**），对默认 warp 的五次独立运行 568.3 / 566.7 / 571.5 / 560.3 / 562.4（均值
**565.8 us**），**快 8.1%**；只看通用求和是 486.0 → 436.4 us（快 10.2%）；逐形状 best-of-30
的 `reduce_ab.py --shapes unet` 是 0.968，同向。不改默认的理由换成：约 45 us 只有整步的
0.2%、320–740 us 差距的 6%–14%，够不到验收；而在输出少每输出长的形状上
（`--shapes representative` 那四个）它仍慢到 1.39 倍；`para_opt_level=4` 又是个同时改
AtomicTunerPass 的粗开关。

顺带三条方法上的坑，都已写进 skill：

1. **`para_opt_level` 不进 jit key**（key 里的 `«choices:` 段只收 `loop_options`），
   只改它的第二轮量到的是第一轮的 kernel，且毫无提示。`profile_step.py` 新增
   `--compile-option name=int` 专治这个。
2. **整网梯度 diff 判不出归约改动**：默认 warp 路径结尾每 warp 一次 `atomicAdd`，
   同一条策略跑两遍，270 个梯度里最坏 `max|Δ|/max|g|` 是 1.8，而两条策略之间是 1.6。
   能判的只有逐形状对更高精度参考（`reduce_ab.py`，十一种形状两条策略都 ≤ 6.9e-7）。
3. **跨进程比 `loss`/`grad_checksum` 也不行**：`profile_step.py` 的 `build()` 原来根本
   没播种（已补 `jt.set_global_seed`），补了也不是位精确——卷积算法选择随负载变，
   而这个 loss 抵消掉约 16 倍量级，输出上 3e-4 就是 loss 上 0.5%。

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

### 2026-09-05 第 197 波整项验收审计

| 分区 | 结果 |
| --- | --- |
| `gates` | 0.15 原计划验收仍为 PR smoke `<300s`；当前 `tiers.py` 使用的 480s 是后续预算校准值。恢复 300s 做定向合同时，预测值 445.75s 明确失败，证明真实性能尚未达标；未提交故意制造红门禁的改动，任务继续待领。 |
| `coreops` | 2.13 仍有约 80 个 C++ flag 及 `exe`/`tflag_count`/`hold_vars`/`sync_ptr` 等全局状态未统一，现有 RuntimeContext 字段迁移不能关闭整项。 |
| `device` | 4.03/4.04 native C++ OpInfo、按 `(op id, backend)` kernel 分派、CUDA/ACL provider 生命周期仍缺，Python registry 只能算前置。 |
| `compat` | 7.12 仍有 `sys.modules[__name__] = _jittor`、`_torch_*` 别名和独立 distribution 缺口，TorchNamespace 阶段不能关闭整项。 |

### 2026-09-05 第一百九十七波：异机测试前置迁移

| 分区 | 结果 |
| --- | --- |
| `coreops` | `8d70ae4a` 增加 profiler_record_shape/hide_relay RuntimeContext owner，结构 20 passed；2.13 仍有大量全局 flags 未迁移。 |
| `device` | `c1a67c91` 集中 native C++ OpRegistry ownership，保留旧 API，C++14 syntax/contract 通过；provider-aware dispatch 仍待。 |
| `compat` | `8fcdc20f` 将独立 Torch publication 拆到显式边界，namespace 17 passed；独立 distribution、默认发布身份和 alias 清理仍待。 |
| `device` | `e2731c0f`/`2dcc3448` 建立无 CANN ACL data schema/normalizer，host-only 合同 11 passed；C++ decoder、属性 owner、descriptor cache、910B3 实机仍待。 |

### 2026-09-05 第一百九十八波：跨 CUDA/NPU 异机测试前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `2d772ef4`/`8d70ae4a` 将 AMP、profiler metadata 等硬件无关状态接入 RuntimeContext，结构 22/20 passed；全局 flags 仍待迁移。 |
| `device` | `c1a67c91`/`b4b7c990` 集中 native OpRegistry ownership 并加入 provider-aware dispatch boundary，主机 C++14 syntax/contract 通过；真实 CUDA/NPU provider 仍待硬件机。 |
| `device` | `e2731c0f`/`2dcc3448`/`7a03fa48` 建立 ACL host-only schema/decoder boundary，15 passed；CANN decoder接线、属性 owner、descriptor cache、910B3 实机仍待。 |
| `compat` | `8fcdc20f` 拆 Torch publication boundary，namespace 17 passed；独立 distribution/alias 清理仍待。 |
| `build/layout` | `09c7665c` 完成 miniz third_party 路径收尾，结构 1 passed。 |

### 2026-09-05 第一百九十九波：跨硬件组织前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `2276d09f` 将 `missing_grad_error` 纳入 RuntimeContext owner，结构 23 passed；全局 flags 仍待迁移。 |
| `device` | `d8edc265` 为 NativeOpDispatchKey 增加 provider generation identity，避免 CUDA/NPU provider 重注册误命中旧 kernel；registry/C++/JIT 定向通过。 |
| `device` | `e00054b0` 增加 ACL `AclDataOwner` host-only C++ seam，5 passed；CANN decoder接线与 910B3 实机仍待。 |
| `compat` | `195544aa` 将 Torch publication aliases 统一走 publication boundary，namespace/alias/bootstrap 合同通过；独立 distribution 仍待。 |

### 2026-09-05 第二百波：异机分发/ABI 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `9d8e2a5a` 迁移 `auto_convert_64_to_32` Runtime owner，结构 24 passed + CPU dtype 回归；全局 flags 仍待迁移。 |
| `device` | `46b1fca7` 增加 NativeProviderRegistration ABI/struct-size/generation contract，相关结构/JIT 合同通过；真实 provider 接线仍待硬件机。 |
| `device` | `62ac7f59` 增加 ACL AclDataView/consume host-only consumer seam，ACL 合同 18 passed；CANN/NPU runner 接线与实机仍待。 |
| `compat` | `94637aa5` 建立 standalone torch distribution import manifest，固定模块图/alias/parent closure 合同；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零一波：跨硬件 ABI/consumer 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `b3034f92` 将 `reuse_array` 纳入 RuntimeContext owner，结构 25 passed；全局状态仍待迁移。 |
| `device` | `27ab42a0` 增加非 owning NativeProviderLifecycleObserver，覆盖 provider/op bind/unbind/teardown，C++/JIT 合同通过。 |
| `device` | `34ab1d99` 增加 ACL AclAttrRunnerContract，host-only ACL 合同 13 passed；CANN/NPU runner 接线与实机仍待。 |
| `compat` | `046d20c4` 完善 standalone torch distribution manifest/alias/structure boundary，namespace 22、组合结构 27 passed；完整 distribution 仍待。 |

### 2026-09-05 第二百零二波：异机边界合同继续

| 分区 | 结果 |
| --- | --- |
| `coreops` | `97d5ce9a` 将 `disable_lock` 纳入 RuntimeContext owner，结构 26 passed；全局状态仍待迁移。 |
| `device` | `3a617079` 增加 Native provider stale dispatch key freshness guard，C++ syntax/结构合同通过；真实 provider 仍待硬件机。 |
| `device` | `d3f95d4a`/`4fc52287` 收紧 ACL invalid schema 类型与 consumer registration fail-closed，ACL 17 passed；CANN/NPU runner 接线仍待。 |
| `compat` | `53e92aba` 增加 standalone torch publication graph validator/alias identity 合同，namespace 24 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零三波：跨硬件 publication/cache 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `32ad37ef` 将 `try_use_32bit_index` 纳入 RuntimeContext owner，结构 27 passed；全局 flags 仍待迁移。 |
| `device` | `a2f44691` 增加非 owning Native observer teardown identity guard，C++/结构合同通过；provider 实机仍待。 |
| `device` | `782c7663` 固定 ACL canonical cache key locale，host-only 13 passed；CANN/NPU runner仍待。 |
| `compat` | `3ac67208` 集中 independent Torch publication 的 child+registry root 原子边界，namespace 25 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零四波：异机 cache/distribution 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `5656f81a` 将 `trace_var_data` 纳入 RuntimeContext owner，结构 28 passed；全局状态仍待迁移。 |
| `device` | `6c073efa` 增加值型 NativeProviderMetadata snapshot，provider replacement/teardown 后旧快照安全，结构 5 passed。 |
| `device` | `d328bb5a` 建立 ACL descriptor key/cache host-only shell，descriptor 合同 26 passed；不创建 CANN 对象，NPU 实机仍待。 |
| `compat` | `165f9a0e` 增加 standalone torch distribution manifest validator，namespace 27 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零五波：异机日志/descriptor/distribution 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `e4399f28` 将 log_silent/log_sync/log_v 纳入 RuntimeContext owner，结构 31 passed；全局状态仍待迁移。 |
| `device` | `838ad234` 增加 NativeProviderConsumerContract，校验 provider/provider_id/ABI 一致性，结构 6 passed；真实 provider 仍待硬件。 |
| `device` | `b37ec5c1` 增加 ACL descriptor 单键失效，ACL 合同 26 passed；CANN/NPU runner仍待。 |
| `compat` | `3e2e9633` 统一 standalone torch distribution facade，namespace 30 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零六波：异机生命周期/Bootstrap 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `5b2c7d08` 将 `node_order` 纳入 RuntimeContext owner，结构 32 passed；全局状态仍待迁移。 |
| `device` | `8ab9591f` 原子发布 Native consumer metadata+dispatch snapshot，结构 7 passed；真实 provider 仍待。 |
| `device` | `ad41509a` 增加 ACL DescriptorCache `erase_device` 批量失效，C++/Python 19 passed；CANN/NPU runner仍待。 |
| `compat` | `23d2f648` 增加 standalone torch bootstrap surface validator，namespace/structure 37 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零七波：异机 lifecycle/alias 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `bb0e9897` 将 `enable_tuner` 纳入 RuntimeContext owner，root/runtime 结构合同通过；全局状态仍待迁移。 |
| `device` | `280762dd` 增加 Native observer scope RAII identity restore，结构 8、JIT 2 passed；真实 provider 仍待。 |
| `device` | `9c7a457a`/`2c2339a5`/`cecfc4f3` 增加 ACL descriptor per-device generation 与 stale-handle guard，host-only 19 passed；CANN/NPU runner仍待。 |
| `compat` | `3079d012` 收紧 standalone torch alias schema，namespace 35 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百零八波：运行时/布局/分发前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `b70afbce` 将 `float32_matmul_precision` 纳入 RuntimeContext owner，结构 34 passed；全局状态仍待迁移。 |
| `device` | `c04941d9` 增加 Native fail-closed consumer dispatch probe，结构 8 passed；真实 provider 仍待。 |
| `device` | ACL descriptor generation/erase 链路已在本波前置提交中完成，未构造 CANN 对象。 |
| `compat` | `7eb5ea90` 拒绝 Torch distribution alias 链/环，namespace 33 passed；独立 distribution 完整发布仍待。 |
| `build/layout` | `b70afbce`/`416a7fe4`/`02a6b5ee` 将 dlink compiler 迁入 build 包并修路径/结构合同，utils 仅剩资源迁移尾项。 |

### 2026-09-05 第二百零九波：布局/Runtime/导入前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `d3c50bdf` 将 `use_parallel_op_compiler` 纳入 RuntimeContext owner，结构 35 passed；全局状态仍待迁移。 |
| `device/build` | `38862b20` 同时完成 NativeProviderRegistrationScope 与 5.25 dumpdef→build 布局迁移，provider 合同 9、布局 11 passed；该提交为并发组合提交，后续恢复单任务提交。 |
| `compat` | `a27e95e4` 为 synthetic torch modules 补 `__package__`/ModuleSpec import metadata，namespace 33 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十波：布局尾项/Runtime/ABI 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `1b2ec5a9` 将 `rewrite_op` 纳入 RuntimeContext owner，结构 36 passed；全局状态仍待迁移。 |
| `device` | `1c042033` 统一 NativeProvider Registration/Metadata/DispatchKey 版本与 struct-size admission，结构 10/JIT 1 passed。 |
| `build/layout` | `01374b45` 完成 tracer/NVTX/MANIFEST 脱离 utils drawer，打包/结构 17 passed；5.25 该尾项已闭环。 |
| `compat` | `afc58e5f` 收紧 Torch distribution manifest canonical order/type，namespace 34 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十一波：Runtime/类型布局/分发前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `85d2279e` 将 cpu/device memory limits 纳入 RuntimeContext owner，结构/CPU 回归通过；全局状态仍待迁移。 |
| `device` | `7b5bda2e` 增加 provider generation-safe unbind，registry 合同 10 passed。 |
| `build/layout` | `e2fbafd6` canonicalize Nano types 到 `src/type`，结构 14 passed；2.14 其余 misc 拆散仍待。 |
| `compat` | `69709bdc` 增加 standalone torch distribution metadata，namespace 36 passed；完整 distribution 仍待。 |

### 2026-09-05 第二百一十二波：Runtime/lease/distribution 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `3d73ab52` 将四项 allocator policy 纳入 RuntimeContext owner，结构 38 passed；全局状态仍待迁移。 |
| `device` | `058a6572` 增加 Native OpId consumer dispatch/index，结构/JIT 合同通过。 |
| `device` | `91147566` 增加 ACL DescriptorCache lease/tombstone/generation 生命周期，host-only 21 passed；CANN/NPU runner仍待。 |
| `compat` | `234d7dc8` 强化 Torch distribution metadata canonical identity，namespace 36 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十三波：Runtime/生命周期/descriptor/distribution 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `1f67af34` 将 `trace_py_var` 纳入 RuntimeContext owner，结构 39 passed；全局状态仍待迁移。 |
| `device` | `8c38c342` 增加 Native provider scopes move-only RAII，结构 10/JIT 合同通过。 |
| `device` | `a8e4b743` 收紧 ACL DescriptorCache malformed canonical key，host-only 22 passed；CANN/NPU runner仍待。 |
| `compat` | `5f97723f` 要求 Torch distribution 显式 aliases，namespace 38 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十四波：Runtime/lease/lifetime/distribution 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `22f0889c`/`a09ca82f` 将 `trace_depth` 纳入 RuntimeContext owner/pyi，结构 40 passed；全局状态仍待迁移。 |
| `device` | `fbbc69d7` 增加 Native generation-checked consumer lease，结构 11/JIT 2 passed。 |
| `device` | `faca96ac` 将 ACL AclDataView 限定为 callback 内借用、不可复制/移动，ACL 20 passed；CANN/NPU runner仍待。 |
| `compat` | `c7393df6` 对 Torch distribution graph 输入 fail-closed，namespace 39 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十五波：Runtime/provider/descriptor/bootstrap 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `2999cce2` 将 `use_cuda_host_allocator` 纳入 RuntimeContext owner，结构 41 passed；全局状态仍待迁移。 |
| `device` | `97224dbb` 增加 Native provider lifecycle event ABI contract，结构 12/JIT 通过。 |
| `device` | `e4393dc3` 增加 ACL descriptor device_size lifecycle count，ACL 10 passed；CANN/NPU runner仍待。 |
| `compat` | `db84f2d0` 强制 Torch bootstrap `__all__` canonical contract，namespace 40 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十六波：Runtime/lifecycle/lease/publication 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `68b4d443` 将 `use_tensorcore` 纳入 RuntimeContext owner，结构通过；全局状态仍待迁移。 |
| `device` | `32f1ea99` 统一 Native provider lifecycle value events，结构 12 passed。 |
| `device` | `56231caa` 增加 ACL DescriptorCache generation-safe release，host-only 25 passed；CANN/NPU runner仍待。 |
| `compat` | `645649b3` 拒绝 Torch publication graph 未声明额外模块，namespace 41 passed；独立 distribution 完整发布仍待。 |

### 2026-09-05 第二百一十七波：TF32/lifecycle/device diagnostics/bootstrap 前置

| 分区 | 结果 |
| --- | --- |
| `coreops` | `98c8ee94` 将 cuda_allow_tf32 纳入 RuntimeContext owner，结构 43 passed；全局状态仍待迁移。 |
| `device` | `b8398291` 修正 provider teardown 同时发布 OP_UNBOUND/PROVIDER_UNREGISTERED events，结构 12 passed。 |
| `device` | `9d49c70c` 对齐 ACL device_size Python/C++ teardown 语义，ACL 14 passed；CANN/NPU runner仍待。 |
| `compat` | `d44782d4` 拒绝 Torch bootstrap 非字符串/非法 `__all__`，负向合同通过；独立 distribution 完整发布仍待。 |

### 2026-09-06 `cudabk`：4.13 跨后端契约矩阵、8.05 MKL 部分交付

| 分区 | 结果 |
| --- | --- |
| `cudabk` | **4.13 已合并**。矩阵两个轴都从 4.03/4.04 的注册表生成：后端轴新增 `@pyjt(known_backends)`（核心声明的全集）∪`registered_backends()`，算子轴 `backend_supported_ops(后端)`。`tests/backends/parity/backend_contract_matrix.py`＋`test_backend_contract_matrix.py`，接进 `noxfile` 的 `cuda` session（`cpu` session 跑整棵 `tests/` 自动包含）。覆盖 47 个算子名×5 个后端行；CUDA 54 passed（cpu 35＋cuda 38＝73 个已验证格子），CPU-only 原生与 torch 模式各 42 passed/12 skipped。缺硬件的后端标 `unverified:not-built`/`unverified:no-device` 而**不是 skip**，25 个 `no-probe` 格子逐格书面理由；判定顺序先「后端能不能跑」再「有没有探针」。两次变异证明有牙：加速器返回值×1.01→40 格红，删一个探针→棘轮单条红。**未覆盖**：只有 (op, backend) 两轴，dtype/layout 不在其中；ACL/ROCm/Corex 无硬件，只有 `not-built` 行，未声称实机验证。 |
| `cudabk` | **4.13 顺带修两条**（都是该门禁发现的、都属于「测试从来没跑过」）：① `setup_cutt()` 是唯一不走 `setup_cuda_lib()` 的库路径，后端搬顶层后既缺 `-I backends/cuda/include`（`stream_compat.h` 在那）也缺 `cuda_sdk_flags`（它 include 的 `cuda_runtime.h`），wrapper 编译不过而**症状伪装成「本机没有 cuTT」**，6 条 cuTT 用例恒 skip 读着是绿的——与看板上「`setup_cutt()` 无调用点」是同一后果的第二个原因（调用点今天在 `compile_extern.py:1305`）。② 惰性库加载让注册表快照缩水：`backend_supported_ops("cpu")` 强制加载前 35 个、后 41 个，少的正是 `mkl_*`。 |
| `cudabk` | **8.05 仍「待领」——部分交付，两条验收只达成「能力表可查」**。① **13 条 MKL 用例原来全是死的**：`jt.compile_extern.mkl_ops` 是「加载了没有」的查询而非取值器，`tests/backends/cpu` 5 条全部 `AttributeError` on None、`test_mkl_batched_matmul` 8 条全部 skip、`test_matmul.py` 结论随命令行顺序变（单独 5 failed/2 passed，先加载过再跑 3 failed）。新增 `tests/_helpers/onednn.py`；改后 `backends/cpu` **13 passed**、`test_mkl_batched_matmul` **7 passed/1 skip**、`test_matmul.py` **3 failed/4 passed** 且不再随顺序变。② **能力表 dtype 声明可查**：`OpCapabilityRegistration.dtypes` ＋ `@pyjt(backend_capability_dtypes)`，`("cpu","matmul")==["float32"]` vs `("cuda","matmul")` 四种 float 宽度；配防漂移测试（逐 dtype 跑 CPU matmul 看实际执行的实现，断言声明集合==真跑起来的集合），把声明改空则三条红。③ 前向 `forward_inference`+`convolution_auto` 与反向 hint 的 `forward`+`convolution_direct` **两处都不一致**，统一为 `forward_training`+`convolution_auto`。④ 库名不再钉死版本：`mkl_library_layout()` 一处探测，v3 只有 `libdnnl.so` 也认，链接名取实际存在的那个。**剩余**：迁 v3 API 未做（**oneDNN 从 v3 起不再发布预编译二进制，已核实 v3.12/v3.13 各版 release assets 为空**，代码改动必须与 v3 库同时到位，要验证得先决定「从源码编 oneDNN 怎么进安装路径」，是设计决定）；**按形状缓存 pd/primitive/reorder 未做，故「CPU 卷积每调用开销下降」未达成**；matmul 仍只 fp32（`dnnl_sgemm` 是 float-only，要宽需换 `dnnl::matmul` primitive）；按 is_train 选 prop_kind 需要核心先有 train flag，今天没有。 |
| `cudabk` | **基线校正**：交接文档此前记「`tests/structure` 2 条失败」，本机实测是 **16 failed / 876 passed / 2 xfailed**（本波开始时 17 条，上游修掉一条）。16 条逐条核对过全部先于本波改动存在，没有一条指向本波新增或修改的文件；其中 `test_native_backend_contract` 的两条把本波的 `backend.h`/`backend.cc` 还原后仍以同样的 subprocess 超时失败。方法沉淀在 `agent/skills/verifying-a-gate-actually-ran` 新增的三节（第十节「门禁的轴也要从代码里生成」、第十一节「编译失败会伪装成这台机器没有这个库」及其子节「`jt.<库>_ops` 是查询不是取值器」）。 |
### 2026-09-06 第二百一十八波：0.15 smoke 时间构成，与判据工具在 xdist 下失效

| 分区 | 结果 |
| --- | --- |
| `gates` | `c31439067` 修 `tools/gate_conclusion_plugin.py`：xdist 下 controller 不做收集，`collected` 恒为空，于是 `compare` 报「丢结论」的两个分支全被禁用——修前实测四条用例、候选 deselect 掉一条，`compare` 同时打印 `passed 4 -> 3` 与 `IDENTICAL` 且退出码 0。改用 `pytest_xdist_node_collection_finished` 取各 node 并集。回归 `tests/structure/test_gate_conclusion_record.py` 修前 2 failed、修后 3 passed。**0.22 之后所有用这个工具背书的分层结论，只在串行下成立。** |
| `gates` | 0.15 量出 smoke 时间构成（热缓存、门禁口径、`-n 4`、16 核、load 13-18）：native 406.3 s + torch 91.8 s = **498.1 s**。两个半边都是 work-bound。`tests/structure` 只占 **9.4%**（约 47 s），且整个在 torch 半边；占 81.6% 墙钟的 native 半边一条结构测试也没有。**「结构套件吃掉整个 smoke 预算」不成立**，全部推迟也只到约 451 s。 |
| `gates` | 同波核实：**行数不是耗时**。ACL 静态合同 3292 行（占 `tests/structure` 16.6% 的行）实测 13.2 s；序贯最贵的三个文件是 116/68/127 行。0.19 的「< 2000 行」与 0.15 的「< 5 分钟」互不相干，已拆出 `0.25` 单独判，`0.19` 看板行标注「验收一项未达」。 |
| `gates` | 方法教训两条，已写进 `agent/skills/gate-tier-budget`：(1) 不按门禁口径量，排名会错 50 倍——同一条用例序贯开 CUDA 是 269.4 s、门禁口径下 4.8 s；(2) 两轮之间 rebase 会让缓存变冷，一次只带进 37 行 `backend.cc/.h` 的 rebase 就重编约 900 个 kernel，native 半边 406 s → 1697 s。 |
| `gates` | **未做**：没有落地任何新的分层推迟项。量完之后 0.15 的 5 分钟验收被判为**分层达不到**——要 300 s 需要 native 半边工作量从 1592.9 s 降到约 550 s（砍 65%），那是「更少或更便宜的比较」（10.18）或更多机器的事，不是排序的事。0.15 保持**待领**。 |

#### ⚠ 结论集合不是逐轮稳定的：实测约 27 个 nodeid 会翻转

用修好的工具在**同一棵树、同一条命令**上比两轮，`collected` 两边都是 2741 / 2201（修前会是
0），但结论有差：native 22 条、torch 5 条状态翻转，两个方向都有。其中
`tests/compiler/test_lock.py::TestBuildLockIsShared`、`tests/core/test_function.py::test_zmem_leak{,2}`、
`tests/distributed/test_process_store.py` 是并行/顺序敏感的既存红（`gate-tier-budget` 第 3.6 节
已编目其中两条）；另有一批 `failed -> passed` 是**冷缓存那一轮才失败**（`test_group_conv_tuner`、
`test_profiler`、`test_conv_parity`、`test_argsort_op`）。

**后果**：「两轮结论集合逐条相同」目前**不能**直接当分层验收用——它有一个非零的抖动地板，
而 `compare` 会把抖动和真正的丢结论一起报成 DIFFERENT。做 before/after 时要先跑两轮同配置
取得抖动名单，再看候选是否引入了名单之外的差异。这条也说明冷缓存那一轮的红有一部分不是真红。

`tests/structure` 的基线与本波 `cudabk` 那条一致：本机实测 **16 failed**（交接文档旧记的
「2 条」不成立）；本波三个提交落地后是 **14 failed / 886 passed / 2 xfailed**（上游修掉两条，
新增本波 3 条），没有一条新红由本波引入。

#### torch 半边的抖动已定位并修掉：`loadgroup` 把有共享状态的文件拆散了

追下去之后，torch 半边那 5 条抖动**全部**来自一个文件
`tests/compat/torch/test_torch_compat_fsdp2.py`，而且不是「并行造出来的红」，是**漏报的红**。

smoke 用 `--dist loadgroup`（`faad4898` 引入，为的是不让 174 s 的别名文件独占一个 worker）。
loadgroup 只把**标了组**的用例绑在一起，其余按用例逐个分发——这个文件的用例共享模块级 FSDP2
状态并依赖文件内顺序（正是第 4 节当初选 `loadfile` 要保住的性质），于是被拆到四个 worker 上，
每个看到不同子集、不同顺序。实测：

| | 两轮之间逐条不同 | 墙钟 |
| --- | --- | --- |
| 改前 | **3 个 nodeid**（两个方向都有），是整个 torch 半边仅有的差异 | 91.5 s / 88.7 s |
| 改后（模块级 `xdist_group`） | **0**，`compare` 报 `IDENTICAL`（2201 对 2201） | 88.5 s / 88.8 s |

方向要紧：改前与改后逐条比，3 处差异**全是 `passed -> failed`**——拆散文件会让本该失败的
用例通过。其中 `test_unresharded_full_grad_is_visible_and_controls_step` 单独串行跑也是 FAILED，
可确认是真红。**这一层此前在漏报失败，而「一轮能报出全部失败」正是 0.15 的第二条验收。**
防退化断言加在 `tests/structure/test_gate_tiers.py`。

顺带一个坑：loadgroup 下 xdist 会把组名附在 collected 的 nodeid 后（`...::test_x@组名`），
两个集合都附所以自洽，但**加/删组标记会让 `compare` 报成 NOT COLLECTED / NEWLY COLLECTED**，
逐条比结论前要先去掉 `@组名`。

native 半边那 22 条尚未逐条追（其中一批是冷缓存才失败，另一批是已编目的并行敏感红），留待下一波。

### 🔴 `2.0-refactor` 现在无法带 CUDA 导入（`e3ad5be9c`）

`import jittor` 在 CUDA 配置下失败；CPU-only 正常（实测 `4.0`）。**bisect 到 `848b98dc0`**——就是
标题以 `WIP：` 开头、今天 12:04 推到共享分支的那个提交。它之后又叠了 13 个提交，期间没有任何
东西带 CUDA 导入过一次。前一个提交 `fd4d8820d^` 上实测正常（`has_cuda=True`、CUDA matmul 8.0）。

`848b98dc0` 做的事本身是对的（4.07 要的「后端不再改写 `compiler` 全局」）：把 CUDA/ACL 的 flag 从
全局 `cc_flags` 抽成按源文件传的 `cuda_sdk_flags`，并接进新的 `BuildSource`。**漏掉的是所有仍由
host 编译器编、却不走那条新管线的后端源码。** 逐层实测到四处，前三处我在本地补过并验证有效
（补完能到 `has_cuda=True`、`registered_backends()==['cpu','cuda']`），**未推送**——它们必然与
4.12／4.15 正在飞的改动重叠，由该 owner 一并落更合适：

| 层 | 症状 | 位置与修法 |
| --- | --- | --- |
| 1 | `fatal error: driver_types.h: No such file` | `compile_extern.py:319` 的 `setup_cuda_extern` 用裸 `cc_flags` 编 `backends/cuda/src/{fp16_emu,helper_cuda}.cc`。加 `cuda_sdk_flags` |
| 2 | `fatal error: cuda_runtime.h: No such file` | `setup_cuda_lib` 的 `compile_custom_ops(extra_flags=...)` 同样缺 SDK include，影响 `backends/cuda/libraries/*/src/*_wrapper.cc` 六个。加 `cuda_sdk_flags` |
| 3 | `'cuda_compute_stream' was not declared` | 声明从 `src/runtime/cuda_streams.h:35` 挪进了**新建的** `backends/cuda/include/stream_compat.h`，而**全树没有任何文件 include 它**。六个 wrapper 都用到它。注意归宿有歧义：它们已经 include 了 `runtime/cuda_streams.h`，所以「把声明放回旧家」和「给六处加新 include」两种修法请由 owner 按新层次决定 |
| 4 | `jit_compiler.cc:88 User check failed: accelerator_compiler.configured  Accelerator compiler is not configured` | 前三层补完后暴露。新 provider 管线里的加速器编译器未被配置，任何 accelerator JIT 算子都编不出来。**这一层属 4.12 本身**，我没有动 |

第 4 层顺带说明 2.19 起了作用：它是一条可捕获的 `USER_CHECK` 而不是 abort。

**这件事同时是 `0.24` 的一次实证。** `0.24` 刚以 `13d314ec` 标为已合并（CUDA session 设
`JITTOR_TEST_REQUIRE_CUDA=1`），但它挡的是「门禁跑了却没真用 CUDA」；这里的形态是**门禁根本
没跑**——13 个提交连续推送、没有一次 CUDA 导入。**建议给 4.12／4.15 这类跨目录搬动加一条最低
门槛：推之前带 CUDA 跑一次 `import jittor` 加一个 matmul（热缓存约 1 分钟）。** 冒烟脚本口径见
`agent/results/2026-09-04-cuda-availability-verification.md` 的命令口径一节。

### 本波结果（`codegen`，3.22 收口）

| 项 | 结果 |
| --- | --- |
| `9ab5ea42` + 本提交 | **3.22 归约口径对齐，验收判定未达成，保持待领。** 对齐后 Jittor 归约类 **2279–2297 us**（nsys 口径 ~2705）对 PyTorch **1928–1998 us**，慢 15%–36%，而不是 3.23 记的「0.57 对 1.20 ms、快一倍以上」——那两个桶几乎不相交，配对与拆解见上文同名小节。**差距 75% 在手写 GroupNorm**（1.5–1.8 ms 在 `backends/cuda/kernels/nn/group_norm_cuda.py`），**不在代码生成里**，已在看板单列一行待派。`edf70f52` 定为 opt-in 不改默认，但其「四形状慢 1.64%」被推翻：真实 UNet 上 level 4 的 `reduce` 角色均值 520.0 us 对 warp 的 565.8 us，**快 8.1%**。顺带修 `test_shared_reduce_helper_is_two_stage`（`fd4d8820d` 移走了 `shared_reduce` 的头文件，断言留在旧路径）：修前 1 failed → 修后 7 passed。无产品代码改动 |
| 测量环境与复测 | RTX 4090 sm_89，卡号从环境读，**非独占**（同卡另有一个 15 天前起的推理服务常驻，占 12 GB、GPU 利用率 0），八个分区并行，`uptime` 一分钟负载在 **7.7–22** 之间。**这一波最该记住的一条：`--mode profiler` 的逐算子数字对负载敏感，第一次发出去的那组来自负载 22 的单次运行，手写 GroupNorm 高估 12%（1715 对复测三次的 1533/1535/1544 us），归约类合计因此从 2545 修正到 2279–2297。**多 kernel 组成的 `reduce` 角色离散约 2%（五次 568.3/566.7/571.5/560.3/562.4），单个大手写 kernel 能到 12%；nsys 两次运行同族 kernel 在 0.5% 以内。凡要写进报告的数字，跑三次列范围 |
| 新增可复用件 | `cuda-reduction-strategy-comparison/reduce_ab.py`（同进程内两条策略的时间 + 对 float64 的误差，两套形状集）；`profile_step_torch.py --attribute`（kernel → aten 算子栈归属）；`profile_step.py --compile-option`（`para_opt_level` 这类不进 jit key 的 flag 必须配它）与 `build()` 的 `set_global_seed` |

### 本波结果（`compat`，7.05 install 事务化）

| 项 | 结果 |
| --- | --- |
| 四个 family 已合入 | `4ecfb14f` 失败路径的 ledger 生命周期；`dcbbedf3` 安装期写入口收归单一 owner；`0e336b58` vLLM arming finder 的 undo 改 owner-aware；`c2fa74d8` 剩余写入口钉成分类闭集。**7.05 仍保持待领**，剩余四处写在 `test_compat_write_entry_points.py` 的 `PENDING` 里（external_backend 整表快照、vllm 两处 post-install `install()`、cpp_extension 构建发布） |
| 最值得记住的一条 | **`install()` 的失败路径没有 `finally`，而 `rollback()` 正是那里最可能抛的一步**（外部改写就抛 `TransactionConflict`）。于是类级 RLock 永久被占、死掉的 ledger 留在 `context.state` 里。既有的 `test_completed_install_conflict_does_not_leak_global_lock` 本来就是为这个写的，**但它从同一个线程 `acquire(blocking=False)`——RLock 对持有者可重入，所以它永远绿，什么也没证明**。新的 `tests/_helpers/install_lock.py` 从另一个线程探，并在获取它的线程里释放 |
| 第二条 | 写入口 helper 不只在安装期跑。`_set_use_cuda` 由 `torch.zeros(device="cuda")` 调到，六份 inline 的事务查找里只有一份检查了 `state == "open"`，所以一次失败安装留下的 ledger 会让此后每一次 CUDA 工厂调用抱 `RuntimeError: transaction is rolled_back` |
| 看板两处描述与实测不符（已改看板，未改计划） | (1) 「`tests/structure` 2 条既存失败」过时：`803e3785` 实测 **15 failed / 872 passed**，多数属 ACL/miniz/rocm/pytest 合同。(2) 第135–139波记的「permissive finder 因 allowlist/身份耦合暂缓」已经在第136/139波做完了，本波复核它已是 owner-aware；真正同形且未做的是 **vLLM 的 arming finder**，已在 `0e336b58` 修掉 |
| 三套门禁 | 原生 CPU：`tests/structure` 需 `JITTOR_TORCH_SHIM=1`（它在 `TORCH_MODE_PATHS` 里，不带就 collect 0 个），**15 failed / 881 passed 与改前逐条同集合**。CPU torch 模式（`JITTOR_TEST_DEVICES=cpu nvcc_path=""`）`tests/compat/torch` 33 failed / 1265 passed / 168 skipped，同集合。CUDA torch 模式定向归因：把本波 10 个源文件 `git checkout 803e37853 --` 回满后跑同一组六个文件，**before 与 after 的 FAILED 集合逐行完全相同（16 failed / 109 passed）** |
| 一个真实的假红源 | 第一次带 CUDA 跑 `tests/compat/torch` 时同时跑着 `tests/structure`，得到 64 failed；**串行重跑后 `test_torch_shim_aliases.py` 全绥**。同一个 `$JITTOR_HOME` 并发两个会编译的 session 还把缓存搞到 `import jittor` 在 `setup_cub` → `compile_custom_ops` 里 `Fatal Python error: Aborted`——删掉整个 `$JITTOR_HOME` 重建后正常（重建约 6.5 分钟）。**同一分区内不要并发两个 pytest session** |
| 新增可复用件 | `tests/_helpers/install_lock.py`（跳线程探锁）；`agent/skills/process-global-state-and-optin` 新增 §6.5（可回滚 ledger 的四条判据）；`agent/skills/structure-rule-has-teeth` 新增「分类闭集」一节（什么时候豁免清单反而是对的） |
| 一个工具坑 | 本波有一次编辑器改写把 `03-compat-shim.md` 的中文全部变成 `?`（UTF-8 → ASCII），提交看起来只是「rewrite (98%)」。已 `git reset` 重做，未推出。**改 CJK 文档后 `file <路径>` 应仍是 UTF-8，且看 `git diff --stat` 的行数是不是与改动量相称** |
### 本波结果：`coreops` / 3.01 执行器拆分（结构完成，性能验收判定不做）

`8e1ad4bd`（executor.h 显式契约 + run_sync 七阶段标注）、`33c10331`（抽出 Planner）、
`fa2d8523`（抽出 Runner）、`029fa9a4`（流水线状态归 SubmissionPipeline）。
`run_sync` 579 → **39 行**，`executor.cc` 945 → 295 行；新增 `exec_plan.{h,cc}`（图 →
值类型 `ExecPlan`）与 `exec_runner.{h,cc}`（计划 → 发射）。验收的两条里
**「`executor.h` 不再有流水线补丁状态」已满足**（并加了结构断言钉住）。

**3.01 保持「待领」，因为性能验收未达；但这条验收的前提被实测推翻了。** 验收原文是
「UNet 每步执行器 CPU 时间由约 16 ms 降到计划命中后的发射成本」。给 `run_sync` 每个阶段
加临时 flag 计数器实测（diffusion UNet，CUDA，`batch=16 res=128`，每步 7 次 `run_sync`，
15 步中位数）：整步 14.21 ms = **等设备 9.78 + 发射 1.94 + 规划 0.173**（其中真正可缓存的
融合划分与两次拓扑排序只有 0.128 ms，phase 2 的 BFS 0.045 ms 无论如何都要走一遍——训练
循环每步都是全新的 `Op*`，缓存里的下标要靠 BFS 的确定性顺序重新对上）。换一档规模
（`batch=8 res=64`，整步 4.41 ms）规划仍是 0.17 ms：**是常数，不是比例**。

所以计划缓存的收益上界是每步 0.128 ms（整步 0.9%），代价是一个必须覆盖 `count_fuse` 会
分支的每一个字段的结构哈希，漏一个就是静默的错误融合，而算哈希本身又是一次 O(算子+边)
的遍历。**判定不做**，数字与推理写进了 `codebase-audit/07-architecture.md`。
**给后来人的方向**：值钱的是 phase 6 那 1.94 ms 的 per-op 发射常数，和 phase 7 那
9.78 ms 里 CPU 的纯等待（正是 `3.07` 要的 GIL 释放，本波顺带量出了它的规模）。

**沉淀**：新 skill `pure-code-motion-refactor`（怎么证明一次纯搬运行为不变：归一化文本
比对、让工作树在长测试跑着时仍可编辑的快照法、「失败集合逐条相同」而不是「全绿」的判据、
以及搬代码会静默掏空按文件名点名的结构门禁）；`jittor-core-planning-cost` 加 §4bis
（run_sync 阶段分解与临时 flag 计数器的量法，含上面那张表）。

**门禁**（与 803e37853 逐条对拍）：原生 CPU `tests/core` **串行跑四轮**（基线 2、改后 2），
四份 FAILED 清单 md5 完全一致（21 failed / 643 passed / 113 skipped / 1 xfailed）；
`tests/ops` 21 failed / 257 passed / 227 skipped 相同；torch shim `tests/structure`
19 failed 清单相同、passed 866→867（多的一条是本任务新增的结构断言）；
CUDA `tests/backends/cuda` 42 failed / 220 passed / 41 skipped / 1 xfailed，42 条逐条相同。

**取 CUDA 基线的办法记一笔，能省下一小时**：本机**冷 CUDA 构建必然失败**——
`dnnl_lnx_2.2.0`、`cub-1.11.0`、`cutt-1.2` 三个三方包要联网，报出来的是 `nan_checker.cu`
的 nvcc `ret 256`，而同一条 nvcc 命令单独手跑是成功的，很容易被当成自己改坏了。
能带 CUDA 构建的只有**预热过的 `$JITTOR_HOME`**。所以基线不要另建缓存目录，
要把工作树临时切回基线提交、复用同一个 `JITTOR_HOME` 跑（还原写进 `trap`，
脚本见本波留下的做法）。

**一条给所有人的坑**：`tests/core` 里 `test_core.py::test_fuse_memopt`、`test_var_holder`
与 `test_grad.py::test_no_grad` 断言的是「进程里还剩几个存活 Var」，**这个数取决于同一进程
里前面跑过什么，不是稳定量**。本波实测：同一份源码（同一个快照）两次并发整目录跑，
`test_no_grad` 一次 9 一次 7，`test_fuse_memopt` 一次红一次绿。**拿它做 A/B 必须串行**，
判据用 FAILED 集合而不是断言里的数字（详见本波在 `pure-code-motion-refactor` §4 的写法）。
### 2026-09-06 9.01 热缓存收口（`build` 分区）

| 分区 | 结果 |
| --- | --- |
| `build` | `d23f9bba` 把核心那份构建戳的做法推广成通用的 `product_build_stamp_path()`/`product_build_is_current()`/`compile_if_stale()`，`compile_custom_ops`（公开 API，**签名未变**）与 `libcuda_extern` 都改为戳一致整步跳过。**热缓存 import CUDA 1.28 → 0.80 s（达标）**、CPU-only 0.52 → 0.38 s，热 import 的编译扇出从 60 条命令降到 **0 条**（改前改后各连测 5 次，当时 `uptime` 负载 12–15，八分区并行）。自定义算子这份戳比核心那份风险大，因为判错方向若是「误判为最新」不会报错而是**继续跑上一次的产物**：所以它记显式列出文件的 stat、调用方各 `-I` 目录（含 `extra_flags` 里手写那些）的递归扫描、以及 `core_source_signature()`；CUDA SDK 与树内目录**故意不扫**（前者由 `cache_path` 的 cuda key 分区，后者已被 core 签名覆盖，照字面扫是每个约 1400 次 stat、六个库合计 90 ms/次 import）。这条按**数值**钉住了：`test_editing_an_unnamed_header_changes_the_answer` 用两个子进程验（同进程内 `__import__` 会拿到已加载的旧模块，所以进程内断言不论戳对错都通过）。另落地 `JITTOR_NO_BUILD=1` + `python -m jittor_utils.bootstrap`（含 `--check`）：import 路径上任何要编译的动作抛 `compiler.BuildNotAllowed` 并指名 bootstrap；`setup_cub` 那个 `except Exception` 显式 re-raise，否则这个开关会变成「import 成功而 cub 悄悄不在」。bootstrap 放在 `jittor_utils` 而**不 import jittor**（起子进程），否则撞 `tests/structure/test_build_config_boundaries.py::test_utils_do_not_import_the_runtime`——第一版就是这么红的，A/B 抓到。离线只读（只读 HOME + 可写 `JITTOR_HOME` + 代理指死端口 + `JITTOR_NO_BUILD=1`）CPU-only 与 CUDA 两配置均 import 成功且算对，这次是**证明了没编译**。三套门禁逐条 A/B（基线 `803e3785`）：native `tests/compiler` 19→**18** failed（355→375 passed，唯一差异是 2.13 冻结 `jittor_path` 后一直红的 `test_source_signature_sees_same_size_edits_and_new_files` 已修，改用 `core_source_signature(root=...)` 传参而不是 patch 冻结属性）、native `tests/core` 21 failed/643 passed 逐条相同、torch shim `tests/structure` 16 failed/869 passed 逐条相同、CUDA `tests/backends/cuda` 42 failed/220 passed 逐条相同、CUDA `tests/compiler` 19→18 failed（397→417 passed）。**零新增失败。** 定向：该测试文件在 native/CUDA/torch 三套各 29 passed；`tests/distributed/test_distributed_init_failure.py` 4 passed（6.B04 显式分布式仍 fail-closed）；CUDA JIT matmul 与 cudnn/cublas 装载正常，`configure_accelerator_compiler` 未被绕过。**9.01 保持 `待领`**：缺口 2（核心编译移到显式 bootstrap 或首次算子调用）仍未完成，冷缓存与换配置下 import 照旧编译整个核心（本波实测换配置/戳失效 49.0 s / 199 TU），因为 `compiler.py` 模块体里 `build_core()` 紧接着就是 `import jittor_core`，而 `flags`/`Var`/全部算子都来自那个模块对象——要真惰性化必须推迟这一句，是整卡改动。核心编译**没有**落到首次算子调用（现首次算子调用 0.037–0.042 s，留作将来搬过去的对照基线）。方法沉淀在 `jittor-build-change-verification` §2.6（构建戳：判错两个方向代价不对称、戳里必须记什么、为什么同进程内验不出来、`custom_ops/` 里的 `.so` 会堆积孤儿不能拿来枚举）与 §2.7（`JITTOR_NO_BUILD=1` 的两个坑：`jit_utils_core` 不在闸门内、`except Exception` 会把它降级）。 |

**给后面人的两条提醒。** 一、**改 `compiler.py`（哪怕只加注释）都会让核心构建戳失效**，因为
戳记了 `compiler.py` 的 sha256（它生成 C++ 源）。症状是改完第一次 import 慢一两秒、或在
`JITTOR_NO_BUILD=1` 下直接被拒；照做 `python -m jittor_utils.bootstrap` 一次即可，不是 bug。
本波的离线只读验收就被这个绊过一次：15:14 跑基线 A/B 时用旧 `compiler.py` 重写了戳，恢复
改动后 CUDA 配置就被拒了。顺手核实过**不是**环境相关的缓存键问题：normal 与离线只读两种
环境下 `core_build_ingredients()` 逐字节相同。二、`_looks_unbuilt()`（把戳挪开）是造
「未构建配置」的**可重复**办法；不要用换 `cc_flags` 指纹那招——它要付一次完整冷编译、留
几百 MB、而且因为指纹固定，**只有第一次是未构建的**，之后测试静默变成空转通过。

### ⚠ `IDENTICAL` 这个判据在 `c31439067` 之前对带 `-n` 的会话是空的

`0.22` 造的 `tools/gate_conclusion_diff.py` 是本轮唯一能证明「一次门禁优化没有丢结论」的工具，
而 `0.15` 的执行者查明它**在 xdist 下不给判据**：`gate_conclusion_plugin.py` 只在
`pytest_collection_modifyitems` 里记 `collected`，而 `-n` 下 controller 不做收集，于是每个带 `-n`
的会话 `collected` 恒为空，`compare` 里报「丢结论」的两个分支**都以它为准、全被禁用**。实测形态：
四条用例、候选 deselect 掉一条，`compare` 在同一段输出里先打 `passed 4 -> 3`、再打 `IDENTICAL`，
退出码 0。已由 `c31439067` 修好（改用 `pytest_xdist_node_collection_finished` 取并集），并补了
`tests/structure/test_gate_conclusion_record.py` 的反向用例。

**影响范围**：交接与看板里共有 9 处拿 `IDENTICAL` 当证据（本文 5 处、看板 4 处）。
**判据是那次运行有没有带 `-n`**：

- **不带 `-n`（串行）的仍然有效** —— `pytest_collection_modifyitems` 在 controller 上正常跑。
  `0.22` 的设备对拍 A/B 属这一类（26 个 nodeid、单卡串行，备注里没有 `-n`），初判完好，
  但请 `0.22` 的下一位接手者按上面的判据自己确认一次，不要因为这行字就当它已确认。
- **带 `-n` 的一律作废**，要用修好的工具重跑。`0.15` 自己就撞上了：smoke 的 torch 半边在
  `loadgroup` 下有 3 个 nodeid 从 `passed` 变成 `failed`——**拆散文件让本该失败的用例通过了**，
  而「一轮能报出全部失败」正是 0.15 的第二条验收。也就是说这一层此前在漏报失败。

一个附带的坑：加或删 `xdist_group` 标记会改 nodeid 字符串，`compare` 会把它报成
NOT COLLECTED / NEWLY COLLECTED。比结论之前要先把组名归一化掉。

### 更正：`tests/structure` 不是 smoke 的瓶颈（我先前的判断是错的）

我曾据「19774 行、跑一遍 5 分半」推断结构套件吃掉了整个 smoke 预算，并据此派活。**实测不成立**：
按门禁自己的口径（热缓存、`gate_scope` 选择集、`-m "not slow"`、`-n 4 --dist loadgroup`），
`tests/structure` 只占 smoke 的 **9.4%（约 47 s / 498 s）**，而且整个落在 torch 半边；占 81.6%
墙钟的 native 半边里一条结构测试都没有。把整个目录推迟也只从 498 s 到约 451 s。

我引用的「5 分半」是**序贯且开着 CUDA** 的数。**不按门禁口径量，排名会错 50 倍**：
`test_process_mode_contract.py::test_naming_a_torch_path...` 序贯 269.4 s、占该目录 44.9%，
门禁口径下是 **4.8 s**。那份序贯数据会把三个「其实是数值测试」的文件排到前三。

**行数也不是耗时**：16 个 ACL 静态合同 3292 行实测合计 **13.2 s**；序贯最贵的三个文件分别只有
116、68、127 行。所以「缺硬件用静态合同顶替」这件事**并没有**在 PR 门禁上收税，我先前担心的那个
冲突不存在——真正的账在 `0.19` 那条「< 2000 行」的验收本身，已由 `0.25` 承接改判。

真正的 smoke 瓶颈是 native 半边，且两个半边都是 **work-bound**（层内工作量 /4 ≈ 墙钟），没有被
某个长文件卡住，所以「再挑几个慢文件推迟」买不到多少。要到 300 s，native 的层内工作量得从
1592.9 s 降到约 550 s（砍 65%），那是 `10.18` 的「更少或更便宜的比较」或更多机器的事。

**其中一个单点值得你知道**：`tests/core/test_setitem.py` 一个文件占 smoke 的 **15%（238.4 s）**，
比整个结构套件的层内成本还高 5 倍。而它在「三个属于别人、任何时候不要提交」的清单里，所以
两位执行者都只报不碰。**要动它需要那个 owner 点头**，这是目前 smoke 预算上最大的一块无主成本。
（第三位执行者复核后同意，同样只报不碰；10.18 的方案不需要动它。）

### 2026-09-06 第二百一十九波：10.18 核心属性测试进门禁，并抓到 2.10 的真缺陷

**`tests/core` 的覆盖构成实测（`tools/measure_core_test_balance.py`，新增）。** 审计那句
「核心 37.5k 行对 tests/core 7355 行」已经漂移，但**更正比值不是重点，更正构成才是**：
核心 C++（去掉 vendored `third_party` 9215 与 `src/tests` 2617）**329 文件 44934 行**，
`tests/core` **81 文件 14374 行**，比值 0.20 → **0.32**。听起来在变好。按文件实际点名的 API 归类：
dtype/数值 **69 文件 12918 行 684 用例**、autograd 41/8933/480、进程与构建 40/7214/375、
绑定 21/5207/284，而这条审计说的那一层——**图与 liveness 只有 10 文件 2328 行 141 用例、
执行器只有 7 文件 1397 行 68 用例**。**翻倍长出来的七千行几乎全在数值面上。**
所以「多写 tests/core 的行」不是解。脚本还报一个下界：179 个核心头文件里 **14 个**在整个
`tests/` 加 `src/tests` 里没被任何文件点到名（反过来读不成立：点名不等于有断言）。

**写了三类属性测试，全部在原生 CPU 门禁里。** 位置选择是本波最该传下去的一条：
**`src/tests/*.cc` 里的 `JIT_TEST` 会被 `compiler.py` 的 `gen_jit_tests()` 自动变成
`jt.tests.*`，再被 `tests/compiler/test_jit_tests.py` 自动变成 pytest 节点**，而 `tests/compiler`
0.04 之后已在门禁里——**不用改任何门禁配置**。新增 `test_exec_plan_properties.cc`（9 条）
与 `test_node_liveness_properties.cc`（5 条），`jt.tests` 由 72 涨到 **86**。
执行器那 9 条断的是「计划自身一致且按它自己声明的顺序可执行」：下标全在范围内、
`range`/`fuse_ops`/`queue` 三者自洽且无空段、每个算子**至少**排进一个段（喂多段的算子会被复制）、
段间与段内都是拓扑序、每个批内 var 恰好一个批内生产者、**建计划不执行任何东西**、
同图两次给出相同计划。**没有录快照**——`count_fuse` 有权改主意什么该融合，录快照会把启发式冻住。
图与 liveness 在 `tests/core/test_core_invariant_properties.py`（11 条），含
`dump_all_graphs` 的边对称/下标范围/无环，以及 `graph_check` 必须**区分「校验通过」与「跳过了」**
（6.C21 的那个坑）：标志打开前返回 **0**、打开后新建节点返回 **> 0**，并钉住
「`swept` 不是进程存活节点数」这条既有限度（实测 10 扫过对 17 存活）。

**抓到了真的不变量违反，就是交接已知问题表里那条 liveness 账不平。** 机制、触发条件与八行
差分对照写在看板 `2.10` 行；一句话：经 `Function` 建的**多输出**算子（`Tapes`）在**有一些但不是
全部**输出被 `stop_grad()` 且整批真的执行时，`LivenessCounter<backward>::release()` 多调一次，
报 `backward liveness release without a matching owner`，异常被 `var_holder.cc` teardown 吞掉，
**每次泄漏恰好 2 个 Var**，各自 `f=0 b=1`（`need_free()==true` 却还活着）。这 2 个就是
`test_zmem_leak{,2,3}` 报的 `2 != 0`。**属性测试比那三条点用例多给了三样东西**：触发条件
（三条必要因素，且**与 `lazy_execution` 无关**——原先「只在 eager」的印象是被 `jt.grad` 变体掩盖的）、
一个对照组（原生多输出 `jt.code` 算子干净，所以问题在 taped 路径而不是多输出本身）、
以及缺陷所在文件正好落在上面那 14 个「没被任何测试点名」的头文件里（`src/ops/tape_op.h`）。
**缺陷本身没修**（属 2.10，且 `var_holder.cc` 正被 `build` 的 5.02 占用）。
落地用**两条测试**而不是一条：`..._leaks_nothing_new` 绿（多一个形状就红，所以没人需要学会忽略它），
`..._leaks_nothing_at_all` 是 **`xfail(strict=True)`**——修好那天 XPASS 变红，强制连那三条点用例
一起更新。再加一条钉住「每次恰好 2 个」，这个数一动说明机制不是原来那个。

**层内成本与分层判定：两个都进 smoke，不进 nightly。** C++ 侧 15 条整文件 **2.49 s**，
`--durations=0` 下每条都 < 5 ms，**边际约 0.05 s**（无设备、无分配、无 kernel）；Python 侧
11 条 **1.88 s**，其中 1.38 s 是那一个子进程。合计约 **1.9 s**，对 native 半边 1592.9 s 的层内
工作量是 **+0.12%**。按 `gate-tier-budget` §2「默认包含，推迟的要写出代价和理由」，
这个量级没有理由去 nightly；**本波没有新增任何推迟项**。
（**扫描必须在一个子进程里跑**，理由有两条：绝对存活计数在并发下不是稳定量，所以断言全部改成
增量；更重要的是泄漏形状会留残留，在本进程跑会让这个文件自己变成下一个跨文件抖动源——
正是它要测的那件事。一个子进程跑完整个矩阵，父进程先断言「矩阵真的跑了」再断言内容。）

**三套门禁。** 原生 CPU（`tests/core`+`tests/compiler`，串行，门禁口径）**51 failed / 1031 passed /
172 skipped / 2 xfailed**（1201 s）；CPU torch（`tests/structure`+`test_type_system`）
**15 failed / 918 passed / 2 skipped / 2 xfailed**（477 s）；CUDA（`tests/backends/cuda`）
**42 failed / 231 passed / 35 skipped / 1 xfailed**（1973 s），42 failed 与 3.01 记录的基线逐数相同、
passed 220 → 231。CUDA 推前烟测通过（`import jittor` + `use_cuda=1` 的 matmul 与 NumPy 对拍，
`jt.tests` 86 条）。torch 与 CUDA 两半的失败清单里**没有一条点到本波三个新文件**；
`tests/structure` 那 5 个扫描类失败逐条查过，点的是
`backends/cuda/test_cuda_backend_registry.py`、`compiler/test_flag_scanner.py`、
`runtime/backends/cuda_streams.cc`（4.12 WIP）等既有项。

**「新增测试没把旧结论挤掉」是机械判定的，不是看摘要行。** 用 `gate_conclusion_diff.py`
在**同一次构建、同一条命令**上做 A/B（baseline 只 `--ignore` 掉新的 py 文件、`--deselect` 掉 14 个
新的 C++ 用例），两轮各 1045.9 / 1038.7 s（1.01x，都是热的）：

    collected 1231 -> 1256    concluded 1231 -> 1256
    failed 51 -> 51    passed 1007 -> 1031    skipped 172 -> 172    xfailed 1 -> 2
    IDENTICAL: every collected nodeid concluded the same way in both runs.
    note: 25 newly collected nodeid(s) accounted for by --expect-new     exit 0

不带 `--expect-new` 时报 25 条差异并非零退出（默认保持严格），那 25 条正好是 14 个 C++ 加
11 个 Python 的新用例。**`failed` 与 `skipped` 逐数不变、没有一条 `STATUS x -> y`、
没有一条 `CONCLUSION LOST`**——这才是「没挤掉」的判据。

**⚠ 一条方法上的自我更正，比上面的数都重要：我第一次取的原生基线是冷缓存的，差点把它当回归读。**
改动前留的基线是 **79 failed / 979 passed**，改动后是 **51 failed / 1031 passed**——
失败少了 28、通过多了 52。**看起来像我修好了 28 个东西，实际上一个都不是。** 逐条差分后
28 条里 27 条集中在 `tests/compiler/test_import_bootstrap_laziness.py`（15 条）、
`test_parallel_compile_attribution.py`（6 条）、`test_probe_cache.py`、`test_lock.py`、
`test_openmp_threads.py`、`test_tracer.py`、`test_signal_handlers.py`——**全是「热缓存下不该重编」
这一类断言**。原因是我在跑基线前只用 `nvcc_path=/usr/local/cuda/bin/nvcc` 预热过，而门禁口径是
`nvcc_path=""`，那是**另一个 cfg 缓存目录**，于是基线那轮是冷的、改动后那轮是热的。
机械证据：基线日志里有一行 `Compiling jittor_core`，改动后那轮 **0 行**。
剩下的唯一同 nodeid 差异是 `tests/core/test_core.py::TestCore::test_relu_memopt`，
两轮都红、只是数字从 75 变成 43——就是 3.01 警告过的那类「存活 Var 数绝对断言会飘」。
**`gate-tier-budget` §1.2 记的是「两轮之间不要 rebase」，这一条要补的是它的同族：
两轮之间也不要换 `nvcc_path`/`JITTOR_TEST_DEVICES`，因为它们换的是 cfg 目录，等于换了缓存。
留基线之前先用门禁那条一模一样的命令把缓存捂热跑一轮。** 已写进
`core-invariant-property-tests` 第 5 节。

**判据工具补了两个缺口，两个都有反向用例。** 第一个：`compare` 默认把新增 nodeid 也当差异报，
于是**加测试的人只能用眼睛扫差异列表**——正是这个工具要替掉的习惯（0.16 那次「数量看着也合理」）。
新增 `--expect-new NODEID`（可重复）：被点名的允许新出现，其它一切照旧检查，
**没点名就出现的仍然红，点了名却没出现的也红**（陈旧的 `--expect-new` 会静默放宽比较）。
修前证据：`compare --expect-new x::y` 在 HEAD 版本上是 `unrecognized arguments`、退出 2。

**第二个是用第一个的时候量出来的，而且它比第一个严重：`collected` 把被 deselect 的 nodeid
也算进去了。** 插件用 `pytest_collection_modifyitems` 回答「这一轮要跑什么」，注释还写着
"After deselection"——**但它不是**。插件在 `pytest_configure` 里注册，因此**先于** pytest 自己对
该钩子的实现跑，而 `--deselect`／`-k`／`-m` 正是在那个实现里剔项的。于是被 deselect 的用例
被记成 collected、又（正确地）永不结论，`compare` 把它们逐条报成
`COLLECTED BUT NO CONCLUSION`。实测（本波 A/B，串行，14 个 `--deselect`）：
`collected=1245 concluded=1231`，14 条差异全是被 deselect 的那些。
**这是这个工具的核心信号在为一件无害的事报警**，比记错数更糟——那个信号的目的是抓「答案丢了」，
而一个每次 `--deselect` 都叫一遍的信号会被它的读者学会跳过，和「永远说 IDENTICAL」是同一类
失效、只是方向相反。它还让两种记录彼此不可比：xdist 那条钩子一直报的是 deselect **之后**的 ids，
所以串行记录和并行记录对「collected」根本不是同一个意思。修法是改用
`pytest_collection_finish` 读 `session.items`（跑在整条 `modifyitems` 链之后；xdist 下 controller
到这里 `items` 为空，所以不能无条件覆盖）。

回归都在 `tests/structure/test_gate_conclusion_record.py`：`TestExpectNew` 4 条 +
`TestDeselectionIsNotCollection` 3 条，**后者修前 3 failed、修后全文件 10 passed / 7.71 s**。
其中一条是「deselect 与真丢结论必须仍然可区分」——**它的第一版把丢结论做成真杀一个 xdist
worker（`os._exit`），xdist 会等那个死掉的 node，花了 300 s 才超时**，对门禁太贵而且测的是
pytest 的崩溃处理而不是这个插件的记账；改成把丢结论注入记录，0.40 s 且确定。

**沉淀 skill `core-invariant-property-tests`**：先量再加测试（行数不是覆盖，是 §2.1「行数不是耗时」的
第二种形态）、`JIT_TEST` 这条最便宜且门禁自动可达的位置、合成图怎么绕开 JIT 编译器
（含 `count_fuse` 会 deref `var->input()` 所以批内每个 var 都要有批内生产者这个坑）、
为什么断性质不录快照、增量扫描与一个子进程、差分形状定位法、以及属性测试抓到真缺陷之后
「两条测试而不是一条」的落地方式。

**剩余**：2.10 那个 backward 多释放没修（属他人范围，且 `var_holder.cc` 被 5.02 占用），
`test_zmem_leak{,2,3}` 三条点用例仍红、修好 2.10 时应连同 strict xfail 一起收口；
本波只加了核心侧的属性测试，**没有削减 `tests/structure`**——前一波已实测那不是 smoke 的杠杆
（9.4%），而 0.19 的「< 2000 行」是另一个目标、由 0.25 承接改判。

### 2026-09-06 `bindings`：2.19 收口（cuTT 可达、LOGf 验收达成、绑定层边界归类）

| 项 | 结果 |
| --- | --- |
| `4bdc7e797` | **cuTT 六条第一次真跑，并因此发现一处静默算错。** `setup_cutt()` 全树无调用点（9.01 把三个 setup 改惰性时给 NCCL/MKL 补了、漏了它），调用点补在 `core_api.transpose` 首次调用（与 MKL 首次 CPU float32 batched matmul 同形，import 期仍不装载）。**静默算错**：`transpose`/`fuse_transpose`/`cutt_transpose` 三处构造函数把「`axes[i]==i` 全部成立」当恒等置换而**不比较秩**，`axes=[0]` 或 `[0,1,2]` 作用在二维输入上直接 `forward(x)` 返回未转置的原张量，`infer_shape` 里的 `USER_CHECK` 不可达；三处统一要求 `axes.size()==x->shape.size()`。修前失败／修后通过：回退三个 `.cc` 实测 3 failed / 23 passed，改回 21 passed 与 cuTT 9 passed（此前 6 skipped）。 |
| `4bdc7e797` | **验收「析构与信号处理器里 grep `LOGf` 为 0」已达成**，核法与结论见下节。同一提交把 `test_destructor_and_handler_contract.py` 的扫描根补上 `backends/` 与 `.cu/.cuh`，并把「总数 > 50」改成「每个根目录都非空」。 |
| `7faf209f3` | **绑定层剩余边界归类**：`py_converter.h` 看着像一大批未迁边界，实际不是（`is_type` 已在更早的地方挡住坏输入）；真正的用户边界是从绑定**返回**的值——`GradCallback` 收到用户 `Function.grad` 的返回值，三处迁 `USER_CHECK`，负向 2 passed。分类理由写进 `docs/testing/error-categories.md` 并加结构门禁。 |
| 三套门禁（都做了同树 HEAD 对照，不引用旧数字） | CUDA `tests/backends/cuda`：改前 42 failed / 220 passed / 41 skipped / 1 xfailed，改后 42 failed / 229 passed / 35 skipped / 1 xfailed，**失败集合逐条相同**（+9/−6 就是 cuTT 六条恒 skip 变成 9 条真跑）。原生 CPU `JITTOR_TEST_DEVICES=cpu nvcc_path="" tests/core`：同树 HEAD 22 failed / 642 passed / 113 skipped，本波 21 failed / 643 passed / 113 skipped，**无新增失败**（HEAD 上红的 `test_complex_svd_batch` 本波未复现，未宣称是本波修好的）。CPU torch 模式 `JITTOR_TORCH_SHIM=1 tests/structure`：rebase 前 15 failed / 872 passed、与同树 HEAD **逐条相同**；rebase 后 14 failed / 881 passed，上游修掉一条。 |
| 基线更正 | 交接文档此前记「`tests/structure` 2 条失败」，`cudabk` 已更正为 16 条，**本机本树实测 14–15 条**（随上游变动）；引用时必须带树与日期。另 `tests/core` 的「17 条」是更早的树，本树 CPU-only 口径是 **HEAD 22 条**。 |
| 顺带发现，未修 | `tests/compiler/test_import_bootstrap_laziness.py::TestCoreBuildStamp::test_source_signature_sees_same_size_edits_and_new_files` 在 HEAD 上就红：它 `mock.patch.object(compiler, "jittor_path", tmp)`，而 `jittor_path` 是 `_FrozenCompilerModule` 的 startup flag，**mock 退出时的还原被拒绝**（`state.py:48 AttributeError: jittor_path is immutable startup configuration`），于是 `compiler.jittor_path` 停在已删除的临时目录上，**随机顺序下把它后面跑的用例一起带红**。属配置冻结（7.05/9.01 一侧），本波未动。 |

### `LOGf` 验收的精确核法（2026-09-06，`bindings`）

验收原文是「析构与信号处理器里 grep `LOGf` 为 0」。**逐条核过，结论是 0，但一次朴素的 grep 得不到这个结论**，
三个地方会骗人：

1. **扫描根**。全树 `LOGf` 实际 **161 处**（含测试与 `log.h` 的宏定义本身），不是 8 处；计划正文的
   「62 处」是旧口径。真正相关的只有析构体与信号处理器里的，所以要先框定范围，而框定范围的目录清单
   **会随重构搬家**——后端搬进 `backends/` 之后，只写 `python/jittor/{src,extern}` 的既有门禁对
   CUDA/ACL/ROCm 的析构一条都没扫到，而总数看着仍然健康。
2. **字面 grep 只是必要条件**。`~VarHolder` 那次就是经 `release_both_liveness` 一跳抛出的，字面扫描全绿而进程 abort。
3. **信号处理器不止一个**，且要跟着它调用的东西走。

实际做法与数字（脚本一次性，不入库）：全树 **612** 个 C++ 文件（`.cc/.h/.cu/.cuh`，三个根）、**110** 个析构体——
字面 `LOGf` **0**，其他抛出宏（`ASSERT`/`CHECK`/`checkCudaErrors`/`throw` 等）**0**；再对每个析构体做一跳
被调函数展开，唯一命中是 `~AsyncQueue → lock()`，**误报**（`std::unique_lock<std::mutex> lock(mutex)` 是变量声明，
不是 `lock.cc:127` 那个带 `LOGf` 的自由函数）。信号处理器侧：`segfault_sigaction` 字面与一跳均为 0，它到达的
`sig_write*`／`print_trace_from_signal` 只做 `write(2)` 与 `_exit`；Windows-only 的 `handle_signal` 也没有 `LOGf`
（它用 `std::cerr` + `abort()`，不是本条验收的对象，但**不是 async-signal-safe**，已写进
`docs/testing/error-categories.md` 单独记账）。

### 「扫描根搬家后变瞎」这个形态，我复核了推广面（`b16773533`）

上面第 1 点值得单独追一次，因为它是「门禁总数看着健康而实际什么都没扫」。**结论：这个形态在别处不成立**，
但顺带撞出一条真的 bug。实测口径与数字：

- `tests/structure` 里有写死扫描根的 **20** 个文件。ACL/ROCm 那五条指的具体文件（`aclops/base_op_acl.cc`、
  `acl_op_exec.cc` 等）**都还在老路径**，实跑 **23 passed**、非 skip，没瞎。
- 后端搬迁是**半途**的：`python/jittor/extern/acl`（104 文件/8606 行）与 `backends/acl`（54/6692）**同时存在**，
  只有 `__init__.py` 同名。裸根 `src`、`extern`（仓库根）不存在，引用它们的测试都是拼在 `python/jittor/` 下面的。
- 「路径不存在就 skip」的模式：`tests/structure` 里 **0 处**。
- 真正危险的是**聚合下界断言**（`assert total >= N`），扫描面变窄时它保持绿。全仓只有 **5 处**，且余量都紧
  （`pass/` 下 `.h` 实测 31 对下界 25）。**新加门禁请写「每个根非空」而不是「总数 > N」**，这正是本波的失效形态。

### cuTT 的构建一直是坏的，而它坏得像「没装」（`b16773533`）

`4bdc7e797` 给 `setup_cutt()` 补上调用点之后我实测了一遍惰性库到底会不会装载，撞见：`install_cutt()` 用宿主
编译器编 `cutt-1.2/src/*.cpp`，那些源码 `include <cuda.h>/<cuda_runtime.h>`，而编译命令里没有 `cuda_sdk_flags`
——`cuda_runtime.h` 找不着、`libcutt` 构建失败。**这是同一 family 的第三例**（`fp16_emu.cc` 与六个 cuda 库
wrapper 在 `803e37853`），都是「宿主编译的 CUDA 源丢了 SDK 头路径」。

坏得看不见有两层：`setup_cutt()` 把构建失败记成「cutt is unavailable」，transpose 回退内建 kernel 只留一条
warning；`tests/_helpers/cutt.py` 又把加载失败一律转成 `SkipTest`。所以有 CUDA 的机器上构建失败 = 全套 skip。
已收紧成「只在 `use_cutt=0` 时 skip，有 CUDA 且启用时加载失败一律 `AssertionError`」。

实测：修前 `libcutt.so` 不存在、`cutt_ops` 为 None、日志 `core_api.py:913 'cuTT is unavailable'`；修后
`cutt_ops` 是真模块、`use_cutt=True`、`transpose(2,0,1)` maxerr 0.0，该组 **11 passed / 0 skipped**（原 9 条）。
门禁有牙的证明：helper 改回 skip 行为后新测试**报红**（1 failed / 1 passed）。测试里捕获 `BaseException`
再显式断言不是 `SkipTest`，就是因为 `SkipTest` 从 `assertRaises` 里逃出去会把这条记成 skipped。

顺带一条实测的惰性库现状（`import` 后全未装载，惰性成立）：`cublas`/`cudnn`/`curand`/`cufft`/`cusparse`/`cub`/
`cutt`/`mpi` 在相应算子首次使用后都会真装载；`nccl`/`hccl` 只在分布式请求时装。注册表
（`register_library_loader`）是全的，洞从来不在注册，而在**有没有东西去请求它**——`get_library` 的 grep 计数
不可信，因为请求走的是 `protect_library_attributes` 的模块属性访问。
### 本波（cudabk 分区）：4.12 代码半已闭合

`process_jittor_source` 的定义、`BuildContext.transform_sources` 字段、`compiler.py` 的赋值三处已删除，`process_acl` 与 `WTF` 补丁此前已 0 处。删除前确认过没有第四个消费者（含 `getattr`/字符串反射/`dataclasses.replace` 关键字/位置构造）。`BuildContext` 属 provider 公开契约，`docs/architecture/backend-build-configuration.md` 原文承诺过 source transformation，已连同 `source-architecture.md` 一并改写。验收原话「核心源码不再是移植的输入」由`tests/structure/test_core_source_is_not_ported.py` 按**形状**钉住（不是按名字），造 4 个反例全部报红。

**本节开头那条「推之前带 CUDA 跑一次 `import jittor` 加一个 matmul」照做了**：`has_cuda=True`、`registered_backends()==['cpu','cuda']`、64×64 matmul 与 numpy 最大差 7.2e-06。

三套门禁与改前逐条同集合（结构 15 failed/880 passed；原生 CPU 两侧同为 88 failed/1731 passed/1200 skipped；CPU torch 两侧日志逐字节相同；CUDA 改后 42 failed 对改前 43 failed，无新增）。**本波没有 CPU 门禁的改前基线可引用，所以是钉在 `HEAD~1` 的只读 worktree 同时刻对跑的**——下一位做删除类改动时建议照办，比事后引用文档里的旧数字可靠。

顺带记两条**分支现状，都不是本波引入、也都不属 4.12**，两棵树表现一致：

1. **原生 CPU 门禁现在收集期直接 error**：`14e5920e5 [4.14]` 同时加了 `tests/core/test_device_methods.py` 与 `tests/backends/cuda/test_device_methods.py`，basename 相同而两个目录都没有 `__init__.py`，pytest 默认 prepend 模式下第二个必然报 `import file mismatch`。整套门禁因此拿不到汇总行（`Interrupted: 1 error during collection`）。修法二选一：给其中一个改名，或给 `tests/` 配 `--import-mode=importlib`。**在修掉之前，原生门禁事实上一条用例都没跑**——这正是第 10 节说的「不在门禁里的测试不是覆盖，是装饰」的又一例，只不过这次整套都没跑。
2. **CPU torch 门禁在 55% 处硬崩**，没有汇总行，两棵树同一位置。

ROCm 那半（计划原文的「需 ROCm 硬件」）本机无卡，四条按序确认项写进 [`../manuals/deferred-hardware.md`](../manuals/deferred-hardware.md) 的 ROCm 一节，**未声称 ROCm 硬件验证完成**。ACL 描述符注册名 `acl_legacy` 与 `BackendId::Acl` 不一致这条改动面跨 4.12，已归 4.15。

**2026-09-06（compat，7.03）：把任务形状从「再挑几个 cohort」换成「清空整个 installer」，两个 installer 归零。**
`796b8e43c` 清空 `_install_reductions`（内嵌 def/class **14→0**、lambda **13→0**），
`d5740ee7d` 清空 `_install_module_methods`（**40→0**、lambda **6→0**）。7.03 此前约 66 波、
每波推进几个 cohort 从没闭合过——这是任务形状的问题：「再挑一个 cohort」没有终点，而
「某个 installer 内嵌 def/class 归零」有，而且随手可测。计数脚本
`agent/skills/torch-api-cohort-promotion/count_installer_closures.py` 随提交落地（`CLEARED`
标记＝nested 与 lambda 同时为 0），两个 installer 各加一条 AST 测试防止下一波往空 installer
里塞新闭包。**下一位如果继续 7.03，建议照这个形状派，不要派开放式的 cohort。**
剩余面同口径实测于 `d5740ee7d`：`_install_nn_extras` 135、`_install_tensor_methods` 83、
`utilities.install` 81、`_install_cuda` 80、`_install_lr_scheduler` 68、data 的 `install` 64、
`_install_distributed` 64、`_install_optimizers` 45、`core.install_misc` 34。
（此前文档记的 `_install_tensor_methods` 76、`_install_cuda` 67 都已漂移，按实测记。）

搬迁手法两类值得记：**install 会覆盖掉的原始方法**（`Module.execute`、`parameters`、
`named_*`、`load_state_dict`）必须在 **import 时**捕获成 `_ORIG_MODULE_*`，晚查会自指递归；
**捕获 `self` 的嵌套 dispatch** 提到模块级显式收 `self`、外层用 `functools.partial` 绑定，
而不是留一层 `def`。清空后 install 里带原生 owner 的 `if not hasattr(...)` 守卫要照抄，
否则会覆盖掉 jittor 原生的 `Module.half` 之类。

**修掉一处静默错**：`zero_grad(set_to_none=False)` 此前一律把 `.grad` 置 None，真 PyTorch
2.12.1 留下的是同 shape 同 dtype 的**全零张量**。之所以静默：梯度裁剪与梯度累积都写成
`if p.grad is not None`，于是每个参数被静静跳过，不报错也不改数，只是训练不再收敛。
修前 1 failed / 修后 7 passed，随同一提交。同提交还把 bridged optimizer 的 `zero_grad()`
移到参数循环**之前**——它会顺带清掉 torch 可见的 `.grad`，放在后面会把刚写好的零张量抹掉。

**一条留给下一位的跨域差异（按「先核最终 owner、不跨域抢改」没有自行改）**：只要进程里
还活着**任何一个** optimizer 对象，`backward()` 就把梯度路由进那个 optimizer，`p.grad`
保持 `None`；把那个 optimizer 释放掉 `.grad` 立刻恢复。真 PyTorch 2.12.1 无论有没有
optimizer 都填 `.grad`。开关是 `jt._active_optimizers`（**清 `jt._current_optimizer` 无效**，
我先试错过一轮）。后果很像第 6bis 节那类问题：任何「先 backward 再读 `.grad`」的用例，会
因为**另一个文件**建过 optimizer 而红，而红的位置和原因毫无关系——`tests/compat/torch`
全量跑里就是这样命中我自己的新用例的（单文件跑 85 passed、全量跑却红一条）。它属
optimizer/autograd 桥，不属 Module installer。

**CUDA 那一层照做了**：`_install_module_methods` 没有归约，但 `to`/`cuda`/`cpu` 就是 residency
迁移面，用 `instantiate_device_type_tests` 在 CPU 与 CUDA 各跑一遍，真实 CUDA
（`CUDA_VISIBLE_DEVICES=2`、nvcc 12.2.140、sm_89）**105 passed**，覆盖跨设备 `state_dict`
往返与 `zero_grad` 两种 `set_to_none` 在两侧的行为。踩到一个坑写进 skill §8：
`instantiate_device_type_tests` 生成的是 unittest 类，**pytest fixture 注入不进去**，多写一个
参数得到的是 `device_types.py` 内部的 `TypeError` 而不是 skip——所以那段状态隔离要写成
contextmanager，普通函数再用 fixture 包一层同一个 contextmanager。
推之前的 CUDA 冒烟照做：`has_cuda=True`、256×256 matmul 与 numpy 最大差 6.9e-05、
`Module.cuda()` 后 forward 形状 dtype 正确。

**三套门禁逐条同集合、零回归**，基线都是本次自己跑的（没有引用文档里的数字，`tests/structure`
文档记的 14 实测已是 15）：结构 15 failed / 887 passed，同一棵树把 `nn.py` 退回 HEAD 再跑同为
15 failed 且**失败集合逐条相同**；CPU torch 模式 `tests/compat/torch` 33 failed / 1390 passed
对 HEAD 基线 33 failed / 1295 passed，失败集合 `diff` 为空、+95 passed 全部来自本波新增文件；
原生 CPU `tests/nn` 3 failed / 204 passed 与基线逐条相同（compat 安装器不参与原生路径），
原生 `tests/core` 21 failed / 647 passed。拿改前基线用的是禁 `git stash` 前提下的
`cp` → `git checkout --` → 跑 → `cp` 回来，并把还原放在 `trap ... EXIT` 里，中途失败也不会把
工作丢在退回状态。

**10.21（`mem`，`a82fd5b9` + `7627bc0ae`）：三个「真环」实测一条都不在了，而 `nox -s typing` 在 base 上一直是红的。**
环这一项闭合。先自己跑环检测（AST 加 Tarjan，扫描根从 `pyproject.toml` 的 `package-dir` 读）再看文档，
结论是审计 §分层与依赖方向 记的三条全已闭合：`jittor_utils` 26 个模块里模块级和函数内都没有 `jittor` import；
`var_holder.h` 不再 include `executor.h`（`318a688e7`）；`node.h` 只剩 `common.h` 与两个 `type/nano_*`，无 pyjt/tracer。
**实测真正还在的**是 Python 模块级 3 个 SCC 共 164 模块（157 是 `jittor` 包门面，约 150 个子模块在模块级
`import jittor as jt`，属 4.07 的面；4 是 `jittor_utils` 内部；3 是 vendored `einops`），外加 1 个 42 个头的
ACL C++ 头环。所以验收按实测改判并同步了计划与看板两侧。

**不用 import-linter，理由是量出来的**：它的图由 grimp 建，grimp 靠 importlib 定位根包再走那一个目录，而
`backends/` 是 `package-dir` 映射加上 `python/jittor/backends/__init__.py` 运行期 `__path__.insert` 拼进来的，
静态解析看不见。实测 `grimp.build_graph("jittor","jittor_utils")` 得 294 个模块、`jittor.backends.*` 只有 1 个，
对比自建图的 384 个模块／86 个 backends 模块——91 个模块不可见且全在最大的那个环里，契约会「零违规」通过。
要让它看见就得把 `jittor.backends.cuda` 当根包，那会先 import jittor 即先编译 C++ 核心。于是 checker 用 stdlib 手写，
无新增依赖，接在 `tests/structure/test_import_layering.py`（每次 structure 门禁都跑，无可选依赖故不会退化成 skip）
与 `nox -s imports` 两个入口。

**反例是在真树上构造的，不是只跑一遍看退出码**：把审计环 1 放回 `jittor_utils/misc.py` → 点名
`misc.py:237 imports jittor.compiler`；给 `jittor/serialization/__init__.py` 加 `import jittor` → 点名
`newly on an import-time cycle: jittor.serialization`；把 `package-dir` 的 `jittor.backends.cuda` 指到不存在的目录
→ 4 条 coverage 报错。**第三条最值得记**：丢掉 overlay 之后 cycle-surface 反而报 `ok`（模块变少是子集），
只有 coverage 的下限挡住了——所以断言写成「每个根非空、模块数 ≥360、backends ≥80、边数 ≥900、契约数 ≥3」。
这正是 `test_destructor_and_handler_contract.py` 那次「后端搬进 backends/ 后一条没扫到而总数看着健康」的同一形状。

mypy 这一项只做了一个子包，剩余面留了数字。7 个文件 → 34。**先修的是门禁本身**：`noxfile.py` 早就在
`[tool.mypy] files` 里，但 `git show HEAD:pyproject.toml` 单独跑 mypy 是 `13 errors in 1 file (checked 7 source files)`
——这条门禁一直不过、一直没人跑。修绿后再扩 `python/jittor_utils` 整包（26 模块，**按目录写进 `files` 而不是逐文件列**，
以后往这个包加模块自动进覆盖）。29 条错误全按真实类型修掉，没有加一个 `# type: ignore`，没有动任何 mypy 开关。
其中值得记的：`get_py3_include_path` 在 macOS 分支对声明为 `Optional[str]` 的全局直接 `.strip()`，改成算在局部再赋回；
`Logwrapper._log` 把 `*msg` 元组就地改写成字符串；`in_ipynb` 依赖 IPython 注入 builtins 的裸 `get_ipython`。
**剩余面**：`python/jittor` 2514 errors／165 files（checked 269），`backends/` 79 errors／32 files（checked 87），
合计约 2593 条／197 文件，下一波按同样「一个子包、一次真干净」的粒度推进。

### 本波（`dist` 分区）：10.19 后端梯度清单，26 → 60，扫描根逐个设下限

`a55d66b6d`、`e5eaacfc2`、`c6b9e967f`（后两个是变基后的号；`15dd50518`／`7c5115df8` 是变基前
的旧号，已失效）。**原清单是本轮「验证手段自己坏在往全过方向」的又一例，
坏法正是 6bis 里点名的那一种**，而且与上一节 `10.21` 撞见的是同一个形状——注意上一节末尾那句
「后端搬进 `backends/` 后一条没扫到而总数看着健康」，这里是它的第三例。原
`test_backend_grad_contract.py` 只扫 `backends/cuda/kernels` 与 `python/jittor/extern`、只匹配
`.cc` 里的 `::grad(`，然后断言**总数 == 26**。26 恰好就是那两个目录里 C++ 梯度的全部，于是漏掉的
34 条**同时不在分子也不在分母**：`backends/rocm/libraries` 的两条在扫描根之外，
`backends/acl/kernels/ops` 的 23 条与 `backends/cuda/kernels/nn` 的 9 条 Python `jt.Function.grad`
在语言之外。

现枚举 **60** 条（C++ 28 + Python 32），与源码树逐条相等。扫描根按后端拆开
（`backends/acl`、`backends/cuda`、`backends/rocm`、`python/jittor/extern`），**每个单独断言
非空**——不写「总数 > N」；`backends/corex` 单列为「应为空」，但同时断言目录存在且有源文件，
免得写错路径的空集合冒充通过；`python/jittor/src` 也扫，断言里面没有后端命名的梯度，堵住
「搬进核心树就不用登记」。

**反例三条都在真树上跑过**，不是只在测试内部造集合：往 `backends/cuda/kernels/cub` 放一个带
`::grad()` 的假 `.cc` → 报红；往 `backends/acl/kernels/ops` 放一个带 `def grad` 的假 `.py` →
报红（这一类原清单完全看不见）；把 `rocprim_cumsum_op.cc` 移出树 → 报红；把扫描根写成
`backends/rocm/kernels` → 「每根非空」那条报红，即原清单当年应该报而没报的那一条。

**24 条本机真跑，36 条硬件延迟，七种 `kind` 全部登记进
`agent/manuals/deferred-hardware.md` 新增一节**（前置、命令、通过判据）。CUDA 侧唯一缺口是
`softmax_cuda.py` 的两条 streaming 反向：既有 `test_misc_op.py::test_code_softmax` 最长一行
2049 列，只走 register 核，`CodeSoftmaxStreaming.grad` 与「register 前向 + streaming 反向」
的混合路线一直没有任何梯度被对过。已补 6 长度 × log/plain 共 12 组。其余 22 条 CUDA 梯度本波
逐条对 CPU 重跑（cumsum/matmul/bmm/transpose/argmax/argsort/conv2d/conv3d），**未发现梯度 bug**。

**两个数值坑，都会让梯度对拍静默失去分辨力，写进了模块 docstring**：
一，**余切必须取正**。用标准正态余切时行归约抵消到远低于其自身舍入误差的量级——131072 列上
`sum(g)` 约 400 而其误差界约 5——于是任何宽到容纳「两个 float32 归约的诚实分歧」的容差，也宽到
容纳「算错百分之一的核」。实测：往 streaming 归约注入 0.1% 误差**仍然全过**。改成正余切后同一
注入立刻报红。二，**长行上不能拿 CPU 当紧容差的参考**。131072 列上 CUDA 反向对 float64 精确解
偏差 3.3e-5，jittor **CPU** 后端是 1.5e-1（CPU 顺序 float32 求和是精度下限，正负相同的项不抵消，
误差按 n·eps 累积；`test_full_reduce.py::test_cpu_results_are_unchanged` 已就同一现象给过误差界）。
我第一版参考写成 CPU 就得出了「CUDA 核不准」的**反向结论**——紧的那条断言必须对 float64，
CPU 那条按顺序求和的 n·eps 预算另立一条，只用来抓公式级错误。

**三套门禁**（都自己重跑，未引用文档数字）：原生 CPU（`JITTOR_TEST_DEVICES=cpu nvcc_path=""`）
19 passed / 3 skipped；CPU torch 模式 `JITTOR_TORCH_SHIM=1 tests/structure` **15 failed /
922 passed**（与基线同 15 条，逐条核对过 offender 列表里没有本波文件；passed 从 887 涨到 922 是
本波 +14 与其他分区新增）；CUDA `tests/backends/cuda` **5 failed / 274 passed**（基线 269，
同 5 条既有失败：`test_cublas_test_op` 3 条、`test_cudnn_op::test_backward_nhwc`、
`test_shared_reduce_helper_is_two_stage`）。推前 CUDA 冒烟：`has_cuda True`、64×64 matmul 求和
467.53、`jittor.__file__` 指向本分区。

**余项属真窟窿而非单纯缺卡**，这是本波把状态从「待领」改成「代码半闭合」时最要紧的区分：七条
即使有卡也测不到反向——`HcclAllGatherOp::grad()` 仍直接 `LOGf << "not implemented"`；
`RocprimCumsumOp` 全树**没有任何用例**碰过（前向反向都没有）；`FloorIntACL`、`IndexACL`、
`NonzeroACL`、`StackACL`、`TriuACL` 只有前向用例。七条逐个列名在手册里，并由
`test_every_gradient_without_a_gradient_test_is_listed_in_the_manual` 钉住：补上用例并改掉
`kind` 之前，谁都删不掉那段文字。**没有用 skip 冒充通过。**

**变基时这两个文档撞了一次冲突，处置记在这里**，因为它正是禁「整文件取一侧」的场景：上游把
`10.18` 改成「已合并」、把 `10.20`/`10.21` 从空行填成长备注，而我的基线里这三行还是旧的（`10.20`
甚至是空的）。取我这一侧会**静默还原另外三条已合并的工作**。实际处置是先机械确认我这个提交对
看板只有 `1 insertion / 1 deletion`（就是 `10.19` 那一行）、对交接是 `53 insertions / 0 deletions`
（纯追加），因此「上游内容 + 重新施加这两处」与逐块解等价且可验证；解完 `git diff` 复核确认
`10.18`/`10.20`/`10.21` 三行与上游逐字相同。

## 6bis. 这一轮的主要失效模式：门禁绿着，但不是因为它通过了

2026-09-06 一天之内，在七个互不相干的分区各撞到一次同一形状的问题：**验证手段自己坏了，而且
坏在往「全过」或「很好」的方向。** 逐例列出，因为形状比结论有用：

| 表象 | 真相 | 怎么照出来的 |
| --- | --- | --- |
| cuTT 六条测试报「过」 | `setup_cutt()` 全树没有调用点，六条恒 skip；补上调用点后立刻暴露三个 transpose 算子的**静默算错**（把「axes 从 0 递增」当恒等置换却不比较秩，`axes=[0]` 作用在二维输入上返回未转置的原张量） | 查「这个 setup 函数有没有人调用」 |
| cuTT 测试全套 skip 而门禁绿 | `tests/_helpers/cutt.py` 把**构建失败**一律转成 `SkipTest`；而构建确实是坏的（`install_cutt()` 用宿主编译器编 cutt 自己的 `.cpp`，缺 `cuda_sdk_flags`），transpose 一直静默走内建慢 kernel | 实测「惰性库到底会不会装载」 |
| 「不泄漏全局锁」测试永远绿 | 它从**同一个线程** `acquire(blocking=False)`，而 RLock 对持有者可重入 | 换到另一个线程去探 |
| `IDENTICAL` 结论永远成立 | `gate_conclusion_plugin` 在 `pytest-xdist` 下收不到 collected 项 | 手工构造一次「结论确实变了」 |
| 析构体 `LOGf` 总数看着健康 | 扫描根只有 `python/jittor/{src,extern}`，后端搬进 `backends/` 后 CUDA/ACL/ROCm **一条没扫到** | 逐根打印命中数 |
| `tests/structure` 报 15 failed 而门禁「跑过了」 | 默认 `pytest`（`testpaths=["tests"]`）撞上三对同名测试文件，`import file mismatch` 是**收集期**错误，`Interrupted` 整个运行、一条不执行、只报 error 而没有汇总行 | 读一份「没有汇总行」的日志 |
| 「300 个算子与 200 个算子键不同」通过 | 把旧的定长编码放回去**仍然全过**——每个算子的 `«opkey<i>` 里 `i` 本身就是变长十六进制，算子个数不同的两段无论边怎么编码都不同键 | 把旧实现放回去，看测试还过不过 |

**共同点**：这七例没有一例是「测试写错了断言」，全都是**测试根本没有观察到它声称观察的东西**。
所以复核的着力点不是读断言，而是三个问题：

1. **它跑了吗**——用例真的被实例化了？(skill `verifying-a-gate-actually-ran`)
2. **它看的是真的东西吗**——扫描根、请求方、探测线程、collected 项，有没有失配？
3. **把旧实现放回去，它会红吗**——这一条最便宜也最有力，`3.02` 与 cuTT 两例都是这么照出来的。

**写新门禁时的两条硬性要求**（都由本轮的实例反推出来）：断言写成**「每个扫描根都非空」**而不是
「总数 > N」；**清单/枚举类门禁两个方向都要造反例**（新增未登记项报红、登记项从树里消失也报红）。

## 6quater. 十二个分区并发编译时，这台机器量不出性能数字（实测 150 倍抖动）

2026-09-07 我在十二个 agent 并发编译期间量 MKL 卷积的每调用开销，同一形状、同一进程、
warmup 之后、三次采样：

| 形状 (n,cin,h,w,cout,k) | 每调用 |
| --- | --- |
| (1, 8, 8, 8, 8, 3) | 9424 – 21285 us |
| (1, 16, 32, 32, 16, 3) | **154 – 23337 us** |
| (8, 64, 56, 56, 64, 3) | 1666 – 1731 us |

中间那行是 **150 倍**抖动。一个 32×32 的卷积真实开销在几十微秒量级，所以这里量到的几乎全是
别人的编译在抢 CPU 与内存带宽。`3.22` 的归约类合计从 2545 修正到 2279–2297us 是同一原因的轻症版。

**结论有两条，第二条是我自己的调度错误：**

1. **验收是数字的任务不能在饱和期做**。剩余待领里这类有：`3.21`（UNet 建图 9ms→≤5ms）、
   `3.22`（归约类对齐 PyTorch）、`3.23`（融合逐元素带宽效率）、`3.18`（冷编译降 ≥50%）、
   `8.05` 的剩余项②（卷积每调用开销下降）、`0.22`（设备对拍时长）、`3.05`/`3.07`/`3.20`
   （建图与流水开销）。这些应当**集中到一个静默窗口串行做**，而不是分散在饱和期。
2. 我在饱和期派出了 `3.21` 与 `3.18`。给它们的指令里要求「三次采样报区间」，但在 150 倍抖动下
   区间本身也不可信。**处置：让它们把代码与判据做完，数字部分由协调者在静默窗口独立复核**——
   这与 `6.C33` 的「42 failed → 5 failed」以同样方式复核过是一致的做法，不需要打断它们。

**静默窗口怎么取**：`uptime` 的 1 分钟负载回到 5 以下（128 核，空载基线约 3–4），且
`ps -eo pcpu,args | rg "cc1plus|nvcc"` 无命中。测量前后各取一次，两次不一致就重测。

## 6ter. 布局收尾那 10 条为什么现在不能派

`0.20`、`1.05`、`2.23`、`3.24`、`4.15`、`5.26`、`7.18`、`8.19`、`9.19`、`10.23` 全是
`target-layout.md` 的机械目录重组，占了剩余待领的五分之一，而且**前置都已满足**——但它们现在
不能派，理由是并发代价而不是依赖：

- `4.15` 要把 `python/jittor/src/` 整体搬到顶层 `src/`，这会改动每个 C++ 文件的 include 路径；
- `0.20` 要把 `agent/design` 并入 `docs/architecture`，而**每个 agent 每波都在写** `refactor-board.md`
  与 `refactor-handoff.md`；
- `10.23` 要让 `tests/` 镜像源码目录，会移动几乎所有测试文件。

在有十来个分区并发改 C++ 与看板的时候做这些，等于让所有人的 rebase 同时进入「整文件取一侧」的
诱惑区——而那个动作 2026-09-03 已经静默回退过一次已合并的 `7.16`（40 个 handler 变回
`except: pass`，提交是绿的，没有任何东西报警）。

**正确的时机是代码级任务收完、树静下来之后，由一个 agent 串行做**，顺序按依赖：

    4.15（枢纽，解开 8.19 / 9.19 / 10.23）
      ├─ 1.05、2.23、3.24、0.20（互不依赖，可依次做）
      ├─ 5.26 → 9.19
      ├─ 7.12 → 7.18 → 10.23
      └─ 8.19

`7.18` 卡在 `7.12`（独立 torch 包）上，`7.12` 还是待领，所以 `7.18`/`10.23` 是这一串里最后的。

### 2026-09-06 `bindings`：上一条「顺带发现，未修」经实测**两个命题都不成立**

上一节最后那条（`test_source_signature_sees_same_size_edits_and_new_files` 在 HEAD 上就红且污染后续用例）
是**误报**，逐条实测如下。留在这里而不是删掉，因为它的三个坑值得下一个人知道。

| 项 | 结果 |
| --- | --- |
| 文件位置写错 | 那条用例在 **`tests/compiler/`** 不在 `tests/structure/`。 |
| 「在 HEAD 上就红」不成立 | HEAD（`4d79ca57d`）上**已绿**：`d23f9bba6`（9.01）早已把它从 patch `jittor_path` 改成传 `core_source_signature(root=)`，该提交说明里就写着「顺带修了 2.13 冻结 jittor_path 之后一直红的」。实测 `TestCoreBuildStamp` 7 passed、整文件 31 passed。上一波大概是在 rebase 到 `d23f9bba6` 之前观察的。 |
| 「污染机制」不成立，且方向相反 | 把旧 mock 写法原样复现出来：`mock.patch.object` 在 **`__enter__`** 就被拒，属性**从未被改过**，受害者随后仍绿。真实情况是 mock 的 `__enter__` 在 setattr 抛出后调自己的 `__exit__` 回滚，而回滚写的是**原值**、也被冻结拒绝——**报出来的异常来自清理路径、消息里带的是原值**，读起来像「还原被禁止」而不是「打补丁被禁止」。所以它**响亮**（用例红）且**不留残留**，不是静默污染。 |
| 四格顺序对照（`-p no:randomly`，nodeid 顺序钉死） | HEAD：单跑受害者 1 passed，先污染源再受害者 2 passed。**打洞之后**（临时让冻结先赋值再抛）：单跑受害者 1 passed（证明受害者不是自己坏的），先污染源再受害者 **2 failed**。即「污染确实会长这样，但今天不发生」。 |
| 本环境没有随机顺序插件 | pytest 7.4.4，插件只有 `pytest-xdist` 与 `pytest-timeout`；`-p randomly --randomly-seed=N` 直接 `ImportError: No module named 'randomly'`。**`-p no:randomly` 不报错不能用来判断插件在不在**（`no:` 不需要 import）。所以「随机顺序下才复现」在本机只能来自 `-n` 的分发。 |
| 同形态全树扫过：**0 处** | 以 `compiler` 模块为 target 的 patch 共 7 处，全是函数或 `has_acl`；唯一点名冻结 flag 的 `tests/structure/test_runtime_sync_state.py:670` 在 `pytest.raises` 里断言拒绝；`tests/compiler/test_preflight.py:24` 改的是 `os.environ` 不是模块。上一波「同一形态很可能不止一处」的猜测不成立。 |
| **真缺口是「没人钉住」，已补两道门禁** | 冻结只保证「写会抛」，不保证「抛完值还是对的」。原有 `test_all_native_flag_instances_reject_late_startup_writes` 写的是 `setattr(obj, name, getattr(obj, name))`——**永远写已经在那儿的值**，对「先赋值再抛」的实现照样通过。变异实测：打洞后经 `mock.patch.object` 的断言 **28 passed 一条不红**（mock 的回滚恰好把值写回去了，把洞盖住），改成直接写一个**不同**的值后 **10 failed**。补的两道：`tests/core/test_startup_config.py` 的 refuse-before-mutate（含 `delattr`），与 `tests/_helpers/state_leaks.py` 把冻结 startup config 纳入每文件快照——后者与机制无关，实测能点名「哪个文件把 `compiler.jittor_path` 改成了什么」，即使那个文件自己是绿的。 |
| 三套门禁 | 原生 CPU `JITTOR_TEST_DEVICES=cpu nvcc_path="" tests/core`：**21 failed / 665 passed / 117 skipped / 1 xfailed**，失败集合与同树 HEAD 的 21 条逐条相同，`+18 passed` 就是本波新增的用例。CPU torch 模式 `JITTOR_TORCH_SHIM=1 tests/structure`：**15 failed / 888 passed / 2 skipped / 2 xfailed**；比上一波记的 14 failed 多的那一条是 `test_runtime_sync_state.py::test_runtime_cuda_allow_tf32_is_a_live_writable_view_on_cpu`，**单独跑也红**（`nvcc_path=""` 的 CPU-only build 上必红，与顺序和本波改动都无关），上一波那次 structure 跑显然带着 CUDA。CUDA 一套本波**未跑完**，原因见下。 |
| 未完成，交下一位 | (1) **CUDA 门禁未跑**：机器 load 13+、另有分区在跑 16 worker 的 pytest，本波末期 shell 反复无响应（连 `echo` 都不返回），不敢在共享 `JITTOR_HOME` 上再起 session。本波改动全是测试与 helper，不含 C++，CUDA 侧风险低但**没有证据，不要当成跑过**。(2) **2.19 的 `op_compiler.cc`(56)/`cache_compile.cc`(40) 切片只完成了静态可达性分析，一行未改**，分析结论见下节，它有一个会让整片迁移变成空操作的前提，必须先解决。 |

### 2.19 下一位必读：`op_compiler.cc` 的 `USER_CHECK` 迁移**当前是空操作**

本波对这两个文件做完了静态可达性分析（未落地任何改动，因为发现了下面这个前提）。

**先更正两个数字。** 看板 2.19 行记 `op_compiler.cc` 51、`cache_compile.cc` 38；本树实测
（含 `USER_CHECK`/`LOGf`）分别是 **56** 与 **40**。更重要的是 `cache_compile.cc` 这 40 处里
**约 28 处在 `#ifdef TEST` 块内**（该文件自带的 `test_main` 单测，由 `gen_jit_tests` 编成独立
可执行文件），天然属内部档、与用户输入无关；真正在产品路径上的只有 **12 处左右**，其中 2 处
还是 `_MSC_VER` only。**按文件总数派工会把这类自测计入工作量**，下一位应先按 `#ifdef TEST` 切分。

**阻塞前提：`precompile()` 整个循环体被一个 `catch (std::exception& e)` 包着
（`op_compiler.cc:808-814`），处理方式是 `LOGf << e.what() << "\nJit compiler error:\n" << this_line`。**
`UserError` 继承自 `JittorError : std::runtime_error`，所以它**会被这个 catch 接住并以 `LOGf`
重新抛出**——而 `LOGf` 抛的是裸 `std::runtime_error`。也就是说：**把 `precompile` 里任何一处
`CHECK`/`ASSERT` 改成 `USER_CHECK`，类型都会在离开这个函数时被抹平**，`catch (const UserError&)`
接不到，`tests/structure/test_error_categories.py` 那种源码计数门禁却照样变绿。
**这正是「让门禁在绿着的时候说谎」的形状，所以必须先改 catch 再谈迁移**：
在 `catch (std::exception&)` 之前加一个 `catch (const UserError&)`，用 `USER_ERROR` 重新抛出
并保留那段「指出源码行」的上下文（那段上下文是这个 catch 真正的价值，不要连它一起删）。

**可达性结论（静态，未实测）。** `precompile` 处理的是**用户自己写的算子源码**——
`jt.code(cpu_src=...)`／`compile_custom_ops` 的内容，所以 `@` 系语法的报错基本全是用户错误：
`@for` 参数不足（`:587`）、`@if`/`@is_def`/`@define` 参数个数（`:617/:628/:678`）、
`@strcmp`/`@alias` 参数个数（`:708/:729`）、`@alias` 指向不存在的量（`:733`）、
`@expand_macro` 找不到宏（`:655`）、`@xxx` 维度不匹配（`:780/:781`）、`@x` 未定义（`:796`）、
花括号不匹配（`:507/:523/:555/:574`）、`@` 后语法非法（`:805`）。已经是用户档措辞的有
`:321`（`expand_op` 找不到 kernel，消息里点名算子与 dtype）。`:607` 的 `total_step < 1000`
**已知用户可达**——看板那条 `jt.bfloat16(math.inf)` 让 `@for` 展开成死循环走的就是它，
所以它也不该带「Could you please report this issue?」。
反过来，`get_name_by_op_var`／`try_get_op_var_by_name`（`:132,:154-160`）、
`fix_op_member`（`:926`）、`__get_fused_src` 里的 `defs` 一致性（`:1144`）、
`:1265` 的 CPU/GPU 不可融合，都是融合器自身的不变量，应留内部档。

**每一条判成 user 档都要有一条负向测试真的走到它**——本波没有跑这些负向测试
（需要编译 C++ 核心，而机器当时已不可用），所以上面这份清单是**待实测的假设，不是结论**。
`jt.code(cpu_src=...)` 加一段坏 `@` 语法就是现成的探针，`tests/compiler/` 是它的落点。

### 本波（pyops 分区）：2.22 闭合；10.20 只复核了数字，仍待领

**首先更正一条会误导下一位的编号。** 本波接手的是上一波被硬取消时留下的 507 行未提交工作
（27 个已跟踪文件 + 5 个新文件，零提交，HEAD 停在 `803e37853`，主线已领先 49 个提交）。
交接给我时它被称作 `10.20`，**实际是 `2.22`**：五个新文件自己的模块标题写的就是 `[2.22]`，
上一波在 `codebase-audit/07-architecture.md` 里也已明写「本波未实现：`2.22` 占满了本波预算，
`10.20` 保持「待领」」。计划里 `10.20` 是「给测试提供受支持的内省 API」，与环境变量无关。
按 `2.22` 收口，看板两行都已更新。

**那 507 行成立多少：机制与门禁自洽且接近完成，方向没有问题，全部保住。** 判断依据是它把
分界线接到了 `2.13` 已有的 `flag_policy.STARTUP_FLAGS` 而不是新发明一份名单，把「哪些旧名可以
留」定成了机械可判的规则（名字里有没有 `_`），并且已经自己踩到并写下了两个扫描坑。
**唯一一处方向问题是门禁的一个空集合漏洞**（下面第三段），已修。另外有意识地**没有**扩大战线：
大写无前缀的一组与 `JITTOR_*` 前缀的约 40 项只进了清单、没有改名，那是开放面，关不掉。

拆成六个提交，一 family 一个：`4751ce240` 机制（`env_config.py` 唯一读取点 + `env_manifest.py`
生成器）、`972adc722` C++ 侧、`66a7727f2` 启动摘要与弃用告警、`1a4c9bebb` 十个调用点、
`7e58c3216` 门禁与反例、`043d19be2` 文档与审计。**顺序是先在旧基线上按 family 提交、然后才
rebase**，所以冲突是在提交之间解的，不是在工作树里丢的。

**rebase 只有一个文件真冲突，不是预告的七个**：`compile_extern.py`、`compiler.py`、`log.cc`、
`jittor_utils/__init__.py`、`04-build-tooling.md`、`07-architecture.md` 六个都自动合上了（双方
改的是不同区域）。**但自动合上不等于没丢东西**，所以按 2026-09-03 那次整文件取一侧的教训，
逐条审了「HEAD 相对主线删掉的每一行」：**全部是我自己有意替换的 `os.environ.get(...)` 调用点**，
外加 `SKILL.md` 的两行（description 与被我改号的 `## 7. 判据` 标题），主线内容一行都没回退。
主线修 cuTT 构建缺 `cuda_sdk_flags` 那处仍在 5 个位置。真冲突的
`agent/skills/process-global-state-and-optin/SKILL.md` 是**双方都在同一位置插新节**：主线插
§6.5「可回滚 ledger 怎么验」（含「同线程探 RLock 什么也证明不了」那段与 `install_lock_is_free`），
我插 §7「环境变量」并把判据从 §7 改号成 §8。解法是两节都留：6 → 6.5（主线原文，一字未改）
→ 7（环境变量）→ 8（判据，两侧的条目都在，判据列表本身在冲突区之外所以已经含两边）。
**主线新加的那节没有被覆盖掉**，`install_lock_is_free` 与「RLock 拒绝跨线程 release」都在。

**门禁的空集合漏洞，本波修的就是这个。** `test_env_var_manifest.py` 的主力规则是参数化扫全树
300 个 `python/**/*.py`，而参数列表来自 `SOURCE.rglob("*.py")`：扫描根一旦与树不再匹配——目录
改名、移动之后 `parents[2]` 差一层——`rglob` 什么都不产出，规则**展开成零个用例**，于是「什么都
没检查」报出和「全都检查过了」一样的绿。这正是清单类门禁最典型的坏法，且坏在往「全过」的方向。
三处防住：两个根各自断言非空且各用一个必须包含的文件（**故意不写成「总数多于 N」**，因为一个
总数在「两个根里一个空了、另一个大得足以兜住」时照样通过）；参数列表为空时代换成哨兵参数，让
规则报**一个失败**而不是消失；这条规则本身的反例进了树，两个方向都断言。为此给
`env_manifest.py` 加了具名的 `default_source_root()`，好让门禁断言「它检查的根就是清单实际读的根」。

**反例结果（两个方向都实测跑过，不是推断）**：往 `python/jittor/dataset/dataset.py` 插一行
`os.environ.get("nvcc_path")` → **1 failed / 297 passed**，失败用例名里带的就是那个文件；
把 `SOURCE` 指到一个不存在的根 → **6 failed / 6 passed**，其中 `test_every_scan_root_matches_the_tree`
与哨兵用例 `[<no python module matched the scan root>]` 各报一次；恢复之后 **307 passed / 2 skipped**。

**三套门禁与基线对拍，零回归**（基线自己跑的，没有引用文档里的数字）：CPU torch 模式
`JITTOR_TORCH_SHIM=1 tests/structure` **15 failed / 1199 passed**，failed 数与基线的 15 相同、
15 条逐条相同且全部与环境变量无关，passed 从 887 涨到 1199（含本任务新增 307 与主线 49 个提交
带来的）；CUDA `tests/backends/cuda` **5 failed / 269 passed / 35 skipped**，与基线 5 failed /
269 passed 相同；定向 `tests/core/test_env_var_namespaces.py` 与 `tests/compiler/test_lock.py`
共 **20 passed**——这一组是本波唯一从未被跑过的部分，它在子进程里实测了 `JT_BUILD_DEBUG=1` 改
缓存目录、裸 `debug=1` 被忽略且被报告为忽略、旧名 `log_v=1` 仍生效且恰好告警一次并写出
`JT_LOG_V`、走错前缀的 `JT_CC_FLAGS` 无效而 `JT_BUILD_CC_FLAGS` 有效。

**CUDA 冒烟照第 6 节的规矩做了**：`import jittor` + matmul，`has_cuda=True`、
`Found device architectures: [89,]`、nvcc 12.2.140、`cuda key:cu12.2.140_..._sm_89`。
顺带这次导入本身就是机制的活证据：因为门禁环境用的是旧名 `nvcc_path=`，启动摘要打出
`environment set 1 setting(s): nvcc_path='/usr/local/cuda/bin/nvcc' from nvcc_path [deprecated name]`
并跟一条 `nvcc_path -> JT_BUILD_NVCC_PATH` 的 `DeprecationWarning`。整套 `tests/structure` 那轮
则打出三个旧名（`nvcc_path`、`use_cutt`、`use_nccl`），即门禁自己一直在用的那些。

**`2.22` 的剩余面（都不属它的验收，是开放面）**：大写无前缀的一组（`CUTT_PATH`、
`DISABLE_MULTIPROCESSING`、`FIX_TORCH_ERROR`、`SKEY`、`JTCUDA*`）与 `JITTOR_*` 前缀的约 40 项
只进了清单、没有改名；`import` 期反写环境变量与 `compiler.py` 的 `cuda_arch` 死代码属 `9.07`。
按交接给我的边界，本波只求把「清单机制本身 + 它的门禁」闭合，没有把所有调用点一次改完。

`10.20` 只做了数字复核（三个数字全漂了，且复核改变了那条任务的形状——`use_cuda` 的多数用法是
**能力查询**而不是策略读取，`2.13` 的 `jt.config`/`jt.runtime` 答不了），**未实现，仍待领**，
详见看板该行与 `codebase-audit/07-architecture.md`。

### 本波（device 分区）：8.06 的 launch 尾部归零

**登记的「65 处」实测是 70 处。** 在引入共享尾部之前的 `5be5fa15~1` 上按「executeOp 体内自己处理
workspace 查询失败 / 自己 `mallocWorkSpace` / 自己发 `(workspaceAddr, workspaceSize, executor, aclstream)`」
清点，71 个 owner 里 70 个带尾巴，唯一例外 KVCacheMemcpy 根本没有 aclnn dispatch。按片段数则是
76（`mallocWorkSpace`）或 69（`syncRun();`）。**归零后 71/71**：本波收完最后 9 个 owner（Upsample
前反向、GroupNorm 前反向、ArgReduce、TruthReduce、reduce prod 的三条路径）。此前被记为「有意保留」
的 reduce prod 其实不需要自己的尾巴——分步只要求异步，`launch(ret, f, false)` 就能表达。

**等价性是这一波最重要的产出，方法值得抄。** `agent/scripts/acl_launch_program.py` 把每个 owner 归约成
(workspace 查询, execute 入口, 同步策略, 失败处理) 的有序 token 流，两个树当参数直接 diff。结论：
迁移前后 **69/71 个 owner 的 execute 入口序列与同步策略逐字相同**，另 2 个（Random、reduce）是同一组
入口的重新分组；唯一系统性差异是失败处理全部收敛到 `LOGf`（`execfail return`×70、`unchecked`×3、
`throw`×2 → `fatal`；`queryfail return`×20、`unchecked`×2、`throw`×1 → `fatal`）。本波自己那 3 处
不一样的：ArgReduce 的查询失败由「打印后 return，留下未初始化的值与下标」变成抛；reduce prod 的
`ret = aclnnProd(...)` **本来赋值了却从不读**（失败就把未动过的输出缓冲当归约结果返回，什么都不打）；
TruthReduce 的查询失败由 `throw std::runtime_error` 统一成 `LOGf`。另 6 处（Upsample×2、GroupNorm×2）
token 流**零差异**——删掉的是 `launch()` 已经做过一遍的死代码。

**没有 Ascend 卡，所以分档说清。** 第 1 档桩 SDK 过 TU：44 个源文件 `-fsyntax-only` 全过、
launcher ABI 断言 70 个（比上一波多 2 个，正是新增的 `aclnnProd`/`aclnnProdDim` 站点）。反向对照做了
5 个，全部报红：把老样板写回 unary（不变量合同报出 `direct_execute`/`query_handler`/`workspace_malloc`
三项）、把 ArgReduce 的静默 return 放回、把 prod 的未检查 execute 放回、把 `backends/acl` 清空
（每根非空断言触发）、把 `launch(ret, aclnnProd, …)` 换成 `aclnnProdGetWorkspaceSize`（`-fsyntax-only`
仍报 `ok`，**只有 `--check-launchers` 挡住**——skill 里那条坑是真的）。第 2 档合同换成不变量式并
要求两个 ACL 根各自非空，机制在 `tests/_helpers/acl_launch_tails.py`。**设备侧一条指令都没跑**：
`tests/backends/npu` 在本机是 `164 skipped, 0 executed -- explained: skipped: no acl found`，
四条上机确认项与精确命令写进 [`../manuals/deferred-hardware.md`](../manuals/deferred-hardware.md)
的 Ascend/CANN 一节，**未声称硬件验证完成**。

三套门禁与改前同集合：CPU torch 模式 `tests/structure` 改前 15 failed / 887 passed，改后
15 failed / 902 passed（新增 15 条全在 ACL 合同里，失败集合逐条相同）；原生 CPU `tests/ops`
21 failed / 268 passed / 221 skipped，其中被点名的五个文件在 `HEAD` 的只读 worktree 与本树上
**同为 9 failed / 62 passed**；CUDA `tests/backends/cuda` 与改前同集合。推之前带 CUDA 跑过
`import jittor` 加 matmul。

**另外注意**：`test_destructor_and_handler_contract.py` 只扫 `python/jittor/{src,extern}` 这条
（后端搬进 `backends/` 之后 ACL/ROCm/CUDA 的析构一条都没扫到）**本波没修**，它不属这一族；
本波只在自己新加的合同里把「每个根非空」这个形状先立起来。剩下两个 family（`AclOpFunctions`
类型擦除、`op_idx_map` 删除）的剩余面见看板 8.06 行。

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
