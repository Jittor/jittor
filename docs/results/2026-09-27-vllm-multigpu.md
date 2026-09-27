# vLLM 双卡验证与 CPU 控制通信

- 状态：Qwen TP=2 的扩展请求矩阵与显式退出清理通过；最终范围和性能见扩展验收小节。
- 日期：2026-09-27。
- 本轮验收基线：`79293b87f8da29a62c4d4301a196fc96abf01174` 加本报告同提交的修复；
  开工时已 fetch 并确认 `origin/chk` 合入。此前通信与重放定位记录保留于历史提交。
- Owner：vLLM adapter / Torch compatibility maintainers。
- 复查条件：通信组、设备放置、worker 启动、vLLM 或 NCCL 版本变化。

## 范围与环境

单机两张 RTX 4090，vLLM 0.24.0、Transformers 5.5.3、Python 3.10.21；
Jittor 使用 CUDA 12.2，独立 oracle 为 PyTorch 2.11.0+cu130。
模型为 Qwen3-0.6B，FP16、TP=2、eager、FlashAttention、关闭 custom all-reduce，
初始小样例为上下文 512、单次生成 32 token；扩展矩阵为上下文 4096、最多 4 个
并行请求。保留 Jittor 既有 CUDA 默认设备。

初始定位时卡上有其他任务，其生成耗时不作为双卡性能。扩展轮的性能单独选择
同一对空闲物理卡，记录整个进程期间的 GPU 占用，数值与口径见下文。
原始日志、JSON、源码清单、独立缓存位于未版本化目录
`$JITTOR_LAB_ROOT/_state/vllm-multigpu/20260927/`。

## 已确认的修复与验证

1. 请求 `gloo` 的 CPU 通信组此前被错误处理为 MPI/NCCL。保留 singleton 与真实
   FileStore/TCPStore 双进程失败证据后，在原生 ProcessGroup 下补充 Store 驱动的
   CPU 控制通信，Torch 层负责公开 API 路由。
2. 销毁 CPU 子组不再关闭 WORLD 的 Store。实际 TCP 测试覆盖正常/反序组、服务器
   位于不同 rank、延迟清理、重复交换、调用顺序不一致、关闭后的访问和有界键数量。
3. CPU 控制通信没有反向传播实现，`distributed.nn.all_reduce` 明确拒绝此路径，
   避免静默丢失梯度；GPU 原生 collective 不被替换成 CPU 计算。
4. 补齐 `get_group_rank`：真实 vLLM 暴露缺失后，公开 API 测例先 RED，再补齐
   WORLD/非连续子组映射与无效组错误。独立 PyTorch 三个 CPU rank 对照通过。
5. 默认 `env://` 动态初始化保留同一个 Store 到 WORLD 销毁；不再让 bootstrap
   临时创建、关闭后由 CPU 子组重建通信服务。先保留生命周期 RED，再验证身份和清理。
6. PyNccl 使用 `ByteTensor` 交换通信 ID，暴露负数字节转换和旧式 CPU 构造器
   放置错误。CPU/真实 CUDA 测例先失败，再修复旧式构造器；同时保留并修复 CUDA
   构造器委托产生的回归。现代 factory 和全局默认设备不变。
7. 模型预热的 `_apply_write_kernel` 暴露第二个 worker 的请求缓冲区落在 GPU 0。
   `get_cuda_view_from_cpu_tensor` 适配此前硬编码复制到设备 0，现改为当前设备。
   新增实际 StagedWriteTensor 测例先得到 1 failed / 3 passed，修复后四项通过。
   原生 vLLM 的 UVA 可以标为 cuda:0，却实际指向所有卡可访问的映射主机内存；
   Jittor 的显存副本不能沿用这一标签约定，因此必须核查真实指针和执行结果。

最初 11 项 host/owner 测试通过；加入 autograd 拒绝测例后 13 项通过；加入 rank
映射和默认 Store 生命周期后相关组合 17 项通过。原生 host transport 与 ProcessGroup
测试共 14 项通过。两套真实 CPU FileStore/TCPStore 数据交换结果一致。
最终版本的 StagedWriteTensor 两套环境均 4 passed；同份代码单卡 Qwen 与 OPT
各生成 32 token，分别与独立 PyTorch 对照完全一致。这是小样例回归，不代表重新
完成此前全部单卡验收矩阵。
旧式类型构造的独立对照：两套环境均为 CPU 30 passed / 7 GPU skipped，CUDA
37 passed。覆盖有符号字节、溢出拒绝、NumPy 转换、设备、别名及梯度。

`multigpu_collectives.py` 两侧真实双卡均通过：NCCL 求和 `[3,6]` → CPU 求和、
gather、广播、对象交换 → 销毁 CPU 子组 → NCCL 求和 `[21,42]`。CPU 指针为 host
memory，GPU 指针分别属于逻辑设备 0/1；Jittor 默认 CUDA 与 `use_cuda=1` 全程保持。
该测试直接复现了默认 Store 重建导致的挂起/端口冲突，修复后不再重建服务。

独立双 GPU NCCL all-reduce 在两套环境均通过：rank 0 输入 `[1,2]`，rank 1 输入
`[2,4]`，两端均得到 `[3,6]`。通过 CUDA driver 指针属性确认实际设备为逻辑 0/1。
PyTorch Qwen TP=2 已生成 32 token；每个 worker 的 226 个参数、28 个 KV cache
及 32 次 forward 设备检查通过。Jittor 在修复 metadata 放置后也完成 32 token，
两 rank 的参数、KV cache、forward 和实际加载二进制检查通过，但第 4 个 token 起
与 oracle 不同，出现重复词；因此不是正确性验收通过。同份代码的 Qwen TP=1
32 token 与 oracle 完全一致。另行测试真实 vLLM PyNccl 的 all-reduce、all-gather、
broadcast、新建/复用输出、FP16/FP32、输入变化及 rank 延迟，共 256 项数值检查通过。
此前通过的 Jittor 原生 NCCL 求和不能替代这条 ctypes 通信路径的验证。

排查过 Python Stream 的句柄 0 与原生 PTDS 句柄 2 不同这一假设：实际 CUDA 事件
测例中，current/default/new 三种公开 stream 都等待 PTDS 的受阻写入；显式
nonblocking stream 负对照能提前完成，因此测例有效，但没有复现顺序错误。
未凭句柄数值不同修改生产代码。另一次实际 TP=2 诊断同时设置
`JITTOR_TRITON_FAST_SYNC=0`、`JITTOR_TRITON_SYNC_AFTER_LAUNCH=1`，仍从第 4 个
token 起发生同样的重复词；这项强同步没有解决错误，也不能单凭它排除所有异步问题。
该诊断正常生成 32 token，但收尾记录 worker 异常退出与共享内存清理警告，
不将退出码 0 等同于完整生命周期验收。后续同前缀逐步对照与修复见下一节。

## 第四步偏差：自动重放漏掉外部操作

两套环境各抓取前六步，两 rank 的有效输入、位置、完整 logits、采样与请求状态
在各自环境内完全一致。每份 trace 的 61 项状态连续性检查通过。跨环境到第四步
仍是同一输入前缀；前三步 logits 最大差异约 0.025–0.035，第四步骤增为
14.5078125，hidden 最大差异 43.328125。首个 token 分歧是 576 对 6722。
第五步起前缀已不同，不再作同输入模型数值对照。

完整逐层读取会让错误消失；只读取 o_proj/down_proj 则仍失败。关闭
`auto_graph_replay` 也恢复完整 32 token。这些都是诊断干预，不能作为默认配置
通过证据。问题与第三次相同形状 decode 开始重放的时点一致。

保留的最小 RED 证据：

- CPU Torch Module 用 `Tensor.data_ptr()` 和 ctypes 写入输出，输入
  10、20、30、40、50 对应错误输出 11、21、21、21、21；关闭自动重放正确。
- 真实 CUDA Triton Module 的显式/自动重放均失败：新输入本应产生 41/61，
  实际复用了 21/41。失败日志为 `triton-replay-red2.log`；前一次脚本 API
  拼写错误单独保留，不作为缺陷证据。

根因是保留图只能重新执行 Jittor 算子，无法重复经裸指针执行的外部 NCCL/Triton
操作。旧检查仅覆盖随机算子、输出读回等情况，没有识别外部写入；额外同步不能
补上图中缺失的操作。新增公开 `jt.graph_replay_barrier(reason)`，捕获遇到不在图中
的操作后拒绝重放并清理保留图，回到正常执行。Torch `Tensor.data_ptr()` 和
Triton bridge 自动声明这一边界；原生裸指针扩展需主动声明。普通纯 Jittor 图仍可重放。
这不等于支持 vLLM CUDA Graph，也没有全局关闭自动重放或更改默认设备。

实际捕获日志在第三步记录拒绝 `lm_head` 的外部指针访问。该 Qwen 配置开启
`tie_word_embeddings`，`lm_head` 与 `model.embed_tokens` 是同一模块；输入嵌入的
TP 路径需要跨卡求和。因而此次失败集中在嵌入层的外部通信未被保留图表示，
不是两 rank 采用了不同输入，也不是采样器故意选错 token。

修复后 CPU 外部写入/纯图对照 4 passed；原生重放、嵌套标记、异常清理和线程隔离
15 passed、11 CUDA skipped；另行实际 CUDA 重放套件 22 passed，真实 Triton 两项从 RED 变为 GREEN。
默认配置、没有逐层读取、没有关闭自动重放的 Qwen TP=2 正常生成 32 token，
与原生 oracle 完全一致，两个 rank 的模型/KV/forward 设备证据通过。
独立的修复后逐步追踪也输出相同 32 token；前六步采样均一致，第四步 logits 最大
差异从 14.5078125 降到 0.02197265625，其余步最大差异约 0.025–0.043。
这证明原来的大幅偏差已消失，不宣称所有浮点数逐位一致。
修复后的单卡 Qwen 与 OPT 各 32 token 也与对应原生对照完全一致。
该轮仍在进程收尾出现 worker 异常退出和共享内存清理警告；后续显式关闭与扩展复验见下节。

未版本化新增证据位于同一结果根目录的 `trace-diagnosis/`、`replay-repro/`、
`replay-fixed-normal/` 等目录；完整日志与前后源码清单一起保存。

## 扩展验收发现的初始化与事件边界问题

扩展轮基线为 `79293b87f8da29a62c4d4301a196fc96abf01174` 加本报告同提交修复。
保留默认设备与自动重放，使用 Qwen FP16、TP=2、eager、异步调度；上下文扩到
4096、同时活动请求上限 4。先运行新增失败测例，再修改生产实现。

1. WORLD 初始化后的 `set + 所有 rank wait` 存在另一处 Store 回包竞态。rank 0
   已返回并进入持 GIL 的 NCCL 子组初始化，rank 1 仍等待 rank 0 的 Python 服务线程
   回复。真实双进程 TCPStore 测例在下一次持 GIL 调用持续 2 秒时，测得 rank 1
   恰好多等 2.001 秒。将完成通知也改为已有的 `arrive` 协议、仅服务端 rank 等待，
   保证离开前其他 rank 的回复已发出；定向 6 项通过，实际模型也越过了该挂起点。
2. 扩展矩阵的首个混合批次中，一个请求结束后 rank 0 在
   `count_fuse → run_sync → sync_all → Python thread_run` 崩溃。
   带 forward hook 与不带 hook 两轮都复现。vLLM 后台输出线程通过
   `AsyncOutput.get_output → copy_event.synchronize()` 等待本轮拷贝；shim 却执行
   全局惰性图，把主线程尚在构建的下一轮也拉进执行器。事件的 `record()` 本来已
   同步完成此前工作，后续等待不应再次提交新图。新增同线程/后台线程、已记录/未记录
   四种事件边界测例，真实 CPU 与 CUDA 均先得到 4 failed，修复后均 4 passed。
   同步/计时相关 CPU 组合 16 passed。实际无 hook 矩阵越过原崩溃点，六轮混合批次
   与三轮随机批次都完成；未关闭异步调度，也未声称修好了所有多线程核心访问。
3. 重复惩罚样例生成 token 92999 后，rank 1 的 embedding 报索引 92807 越界；
   正确局部索引应为 `92999 - 75968 = 17031`。`75968 % 256 = 192` 恰好解释了
   错误偏移。单元素 bool 张量与 Python int 运算先被原生路径截成 uint8，之后
   转 int64 无法恢复。最小 CPU 测例先 8 failed / 12 passed，完整分片 embedding
   在 CUDA 也先失败。Torch 层现在先确定结果类型，仅在需要时转换操作数再计算；
   张量之间的运算与真除法路径不变。新增与既有 promotion/dtype 组合
   85 passed、2 skipped；真实 CUDA 的全部 36 项通过，原生 PyTorch CPU 独立对照通过。
   后续 penalties 写入栈是 CUDA 上下文失败后的次生报错，不是最早故障位置。

最终无 hook 请求矩阵两套环境均完成 **70 请求、1208 输出 token**，跨 backend
贪心 61/61 请求、1028/1028 token 一致，随机 9/9 请求、180/180 token 一致。
随机一致仅限这些固定样例，不代表所有种子或采样分布完全等价。两侧六轮重复批次、
三轮随机重放、单请求/批量/反序内部比较全部通过。缓存计数均为 **0 → 624 → 0**，
冷计算、命中与清空后的生成完全相同。高词表 token 的重复惩罚样例也完成。
两个 rank 的参数/KV 指针分别属于真实 CUDA 0/1。两个 worker 退出码均为 0，
三个记录的共享内存名称全部消失；完整日志不再出现 worker 意外退出或 shared_memory
泄漏警告。仍有未使用的可选扩展导入与 host-empty-cache 提示，不宣称日志零警告。

长文本两侧也完成 **10 请求、224 输出 token**：1024/3072-token 输入、长短混合批次
及清空缓存后的重复批次全部通过，生成 token 全部与 oracle 一致，退出清理通过。
请求矩阵与长文本合计 **80 请求、1432 输出 token**。这是有界正确性验收，不覆盖
长时间服务、请求取消、KV 紧张下抢占恢复、四卡/多机或任意模型。

关闭工具现在显式调用固定版本的 `llm.llm_engine.engine_core.shutdown()`，记录
两个 worker 的退出码与三个队列共享内存名称，确认进程退出后名称消失；不手动 unlink，
不屏蔽警告。生成成功与生命周期成功分别记录。默认正确性矩阵和性能工具均不安装
forward hook；观察模式仅用于诊断，因为 Jittor Module 的 hook 会改变自动重放入口。

## 双卡热态吞吐与延迟

两套独立环境先后使用同一对 RTX 4090，外部监控全程未发现其他 GPU 进程；
每种批量预热 3 轮、测量 21 轮。固定输入 128、输出 32 token，FP16、TP=2、
eager、上下文 512、关闭 prefix cache 和 custom all-reduce，不安装 forward hook。
吞吐为总输出 token / 总测量时间；延迟为每请求首 token 和后续 token 交付间隔的中位数。
这是离线同步 `engine.step` 口径，包含推理通信，不包含引擎初始化、预热、放置检查、
边界同步 RPC 和退出，也不是 HTTP 或纯 GPU kernel 耗时。

| 批量 | PyTorch 吞吐 token/s | Jittor 吞吐 token/s | Jittor/PT | 首 token PT/JT ms | 后续 token PT/JT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.45 | 15.15 | 36.55% | 24.78 / 104.08 | 23.92 / 64.49 |
| 4 | 162.02 | 56.09 | 34.62% | 25.81 / 109.83 | 24.42 / 69.74 |

预热与测量共 120 请求、3840 输出 token，两侧逐请求全部一致；各自重复输出也一致。
首轮性能测量后，两个 worker 最终退出码为 0、共享内存也释放，但有一个 worker
超过 vLLM 默认 5 秒关闭宽限期，日志记录发送 SIGTERM。因此不能仅凭退出码宣称
未受强制终止。保留该轮全部日志，使用官方
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=15` 再跑完整同配置测量；第二轮
14.93 / 57.04 token/s，全部 3840 token 再次与 oracle 一致，完整关闭约 5 秒，
日志没有 SIGTERM/SIGKILL、worker 意外退出或 shared_memory 警告。
这只是给正常释放资源更充分的有界时间，不改变模型执行，关闭时间不计入性能。

表格保留首轮计时；Jittor 共两个独立会话，每个批量各测量 21 轮，oracle 为一个
会话内各 21 轮。记录 median/p95 和原始样本，不能据此声称所有负载的稳定速度。
未在本轮重测 TP=1，因而不是单卡→双卡扩展效率或相对旧代码的提速结论。
当前双卡功能可用，性能仍明显落后于相同 TP=2 配置的 PyTorch。

最终证据：`final-jittor-{matrix,long,latency}/`；oracle 为
`acceptance-oracle-nohook/`、`acceptance-oracle-long-nohook/`、`latency-oracle-r1/`。
退出等待复验为 `final-jittor-latency-grace15/`。
对应 `process-result.json` 保留父进程退出码和占用判断；`occupancy.jsonl` 保留采样。
本地 `acceptance-results/` 保留矩阵、长文本和延迟对照，源码以
`acceptance-final-manifest.json` 的 1165 个生产文件 SHA256 绑定；两端核对无差异。
所有路径均位于前述未版本化结果根目录。

## 运行隔离与尚未承诺的范围

- CPU `gloo` 名称映射到本仓 Store 控制传输，不是 libgloo 二进制绑定；不声称支持
  混合 PyTorch/Jittor rank 互通、全部 Gloo API 或训练。
- 外部启动器使用每轮唯一的 `JT_NCCL_ROOTINFO_FILE`。历史默认路径的 `.pgN`
  文件可能被后续运行读取；本轮一次子组初始化失败，换独立路径后通过该阶段。
  通用旧文件隔离机制仍需专门修复，不能以启动器隔离宣称已解决。
- spawn 会继承父进程缓存搜索路径。验证工具在启动 worker 时移除这些继承路径，
  并检查实际导入二进制所在目录；通用编译器搜索顺序仍需另行回归。
- 使用独立工作目录，防止 shim 自动扫描公共临时目录中的无关扩展。
- 尚未验收四卡、多机、PP、Graph/compile、量化或多卡 HTTP 服务。

可复用入口、环境和设备证据要求见
[vLLM 验证 skill](../../agent/skills/vllm-torch-compat/SKILL.md) 中的双卡小节。

## 仓库检查

CPU `tools/run_test_suite.py --tier core`：310 passed、44 skipped、1 xfailed；
跳过项不计入 CUDA 覆盖。适配层生命周期 11 passed，真实命名空间结构检查 12 passed。
最终结构全集为 1365 passed、8 skipped、4 failed；失败均为原有的 child-process
两项、pytest collection 两项（未改动的 executor thread / H3 decode 测试）。新增报告
最初触发索引行数约束，已压缩并通过文档治理复验。布局与生成 manifest 检查通过。
远端 1165 个生产文件 SHA256 与本地候选清单一致；保留清单与测试原始日志。
旧 adapter 注册 fixture 与真实 vLLM 导入不能在同一进程重复注册；将两组测试隔离后，
既有 backend 26 项通过。原始混跑错误保留，不把它记为通过。

扩展轮同份最终生产代码的 CPU core 仍为 310 passed、44 skipped、1 xfailed。
另一次人为关闭 MKL 的 core 运行出现调度顺序断言及其后两项存活变量计数失败；
保留该日志，恢复默认构建设置后通过，没有修改这些无关测试。事件较宽 CPU 组合中
另有无 CUDA 环境非法设备编号校验失败，定向事件组合与真实 CUDA 边界结果另列。
