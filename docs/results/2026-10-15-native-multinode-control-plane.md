# 原生 jtrun 固定节点控制面：单机双 agent 与真实 CUDA 验证

- 状态：Linux/NCCL 固定节点软件控制面已实现；实际多主机验收待硬件。
- 复查日期：2026-10-15。
- Owner：Native distributed maintainers。
- 测量源码：`bc07df0c079fcaa6ffcefce7ede1423ca599305e`；上游基线 `4ccd72fafe51496190671dc93548c852be2974c1`。
- 后续同步：官方远端 `bd17d080ee31c8cb84f3ac2617662f423c343fec`，本树合并 `ee7a5ce6d697ca4a39d37e3d672cc85f7d16b925`；生产 owner 与测量源码逐字节一致。
- 独立重算工具：`abb4f302c434a7ea3251bfdc6949ae1dfe22e442`；仅测试组织修复 `5844a4581`。工具/报告不改变历史测量身份。
- 复查条件：launcher、Store、NCCL bootstrap/subgroup/watchdog、FSDP fixture 或实际网络/设备拓扑变化。

## 已确认的问题与实现

原基线 `4ce19215cba8fb15d7b347f53fda012ced9b4143` 的实际 host 最小复现：
`--nnodes/--node-rank` 未识别；非 master agent 会尝试 bind 远端 master 地址；两个 agent
各自分配 rank 0/1。旧 subgroup 与 watchdog 还依赖同一 root-info 路径下的文件。
原始复现保存在未版本化 `baseline-control-plane-probes.json`。

公开 `jtrun` 与等价模块入口 `python -m jittor.distributed.launch` 新增固定节点参数。
全局 rank 由 node rank 和本地进程序号派生；默认单节点行为及每 worker 一卡掩码保留。
多节点仅启用 Linux/NCCL；HCCL 在设备探测与 worker 启动前显式拒绝。

节点 0 的 launcher 持有既有原生 TCPStore 服务；其他 agent/worker 连接同一服务。
job id 隔离键空间，节点数/进程数/backend 必须一致，重复节点 rank 拒绝。
worker 不持有服务端，避免 GPU 集合通信持有 GIL 时阻塞其他 worker 的 Store 请求。
WORLD 与 subgroup 交换 NCCL opaque unique id；subgroup 检查所有 WORLD 参与者的 rank
顺序，再复用原 C++ communicator owner 创建独立 communicator。

watchdog 只读本节点 worker 的心跳文件；agent 用 Store 序号和本机 monotonic 时间监视
其他 agent，转发 worker 失败。Linux parent-death signal 约束 worker 的 agent 生命周期。
没有复制 FSDP/TP/ZeRO/optimizer/checkpoint 数学；这批仅扩展启动与通信控制面。

## 实际执行

使用两个 RTX 4090、真实 Cu122 wheel stack、NVCC 12.2.140；每个 worker 的 CUDA mask
只暴露一张卡，实际设备 ordinal 为 0。保留外部使用者进程，不作吞吐、L4 或峰值内存结论。
每 rank 使用任务私有 HOME/JITTOR_HOME/TMPDIR；单测与先前 benchmark 缓存分开。
首次核心、扩展和维护 fixture 的预热逐 rank 串行；真实 NCCL 直接调用避免 singleton
公共恒等路径掩盖未加载的库。候选始终为 native 模式，禁止 fallback，计数增量 0。

参考解释器实际加载 binary Torch `2.11.0+cu128`，拒绝 shim，禁 TF32，使用 highest precision。
候选与参考复用维护 fixture 的固定权重与输入；独立执行五步 dense 参考。

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| CPU core | 322 passed，42 skipped，1 xfailed | native 98 + Torch 224；clean `abb4f302c` |
| 新增与既有 host 邻接测试 | 46 passed，0 skipped | loopback agent、Store、legacy ownership、证据负例 |
| 双 agent / 两张 CUDA 卡 | 每 rank 3 passed，0 skipped | 逆序 root、两个独立 subgroup、直接 singleton NCCL、WORLD、75 次混合集合、错序拒绝 |
| 原单 agent / 两张 CUDA 卡 | 每 rank 2 passed，0 skipped | 同一数值节点与原 file subgroup 路径；额外 Store 错序节点显式 deselect |
| 独立 Torch / native FSDP parity | 每 rank 4 passed，0 skipped | SGD/AdamW × FP32/BF16，五步全梯度、参数、moment、原预算 |
| 生产端 fresh launcher | 每 rank 4 passed，0 skipped | 第三步保存、dropout 0.25、继续两步 |
| 恢复端另一个 fresh launcher | 每 rank 4 passed，0 skipped | 新 PID、新模型/optimizer，加载后两步完整数组 exact |
| 实际 CUDA worker 故障 | agent 返回 `[1,137]` | 双方 STEP 5 后只终止本任务 rank 1；约 0.717 秒传播退出，无遗留 worker |

候选生产端 PID 为 `594850/594848`，恢复端为 `604357/604349`，launcher PID 也不同。
worker identity 原始记录还包含测量 SHA、world/local-world、external Store 标志与独立缓存路径。
这次 fresh-process 是原生 FSDP/checkpoint fixture，不能据此重标历史 Accelerate 的
same-prepared-object 记录。

FP32 逐元素预算保持 `atol=2e-6, rtol=2e-5`；BF16 保持 `atol=4e-4, rtol=3e-2`。
每参数 update L2 预算分别为 `2e-6 + 2e-5 * oracle_norm` 与
`1e-6 + 0.03 * oracle_norm`，不根据本轮误差放宽。每个真实梯度与参数更新都有 zero
负例；实际 Linear 输入/权重 BF16、FP32 master/梯度/moments、BF16 all-gather 和
FP32 reduce-scatter 均由原 fixture 检查。每 rank 持有 35 个存储元素，对应全模型 66
元素及 padding；该 owned storage 事实不是峰值内存节约证明。

## 原始证据与独立重算

未版本化根目录：`$JITTOR_LAB_ROOT/_state/native-multihost/20261015/`。
`raw-bindings-bc07.json` 绑定 299 个原始文件（2,166,662 bytes），SHA256 为
`9ee73c750a46ddf6194328ef1c38f67dd37e57d57ed46edd203114dcba95d1cb`。
`source-bc07.json` 记录 2,804 个常规文件 SHA256，未将目录 symlink 的目标文本纳入该扫描；
`source-bc07.bundle` 保留完整历史 Git source，因此不把常规文件清单称为完整 Git tree 证明。

实际 argv/env 在 `cuda-unit/commands-{warm,oracle,multi,legacy,parity,baseline,resume,kill}.json`；
逐 rank 编译/执行日志、原始 NPZ/JSON、四个 checkpoint、实际 PID 与脚本都在同目录。
正式结果由公共模块 launcher 执行，不将私有双进程或 MPI 结果重标为 jtrun。
原始执行器为 `cuda-unit/run.py`，worker 环境入口为 `cuda-unit/rank_entry.py`；其当前
文件哈希在原始绑定中，历史 warm 失败只保留诊断身份、不授予通过。

维护入口 [独立重算工具](../../agent/skills/jittor-distributed-verification/recheck_native_fsdp.py)
使用 NumPy 校验所有绑定、完整参数清单与 metadata，再复算 680 组 oracle 数组、
368 组 fresh-resume 数组、160 项 update 预算和 160 项 zero-gradient 拒绝。
四类 RNG 是 Python/NumPy/native CPU/native CUDA 的后续实际抽样，恢复后与生产端逐位一致。
工具无 Torch/Jittor import，也没有以重算代替原始 GPU 执行。

复算用法见 [distributed verification skill](../../agent/skills/jittor-distributed-verification/SKILL.md)。
本轮结果为 `recheck-bc07.json`；维护者还用独立自写 NumPy 审计重新得到同样的 680/368/160
结果，并将测量快照中 2,804 个常规文件与对应历史 Git blobs 逐一对照匹配。独立审计记录在
`$JITTOR_LAB_ROOT/_state/jtrun-control-integration-bd17/independent-source-raw-audit.json`
与同目录 `independent-array-audit.json`。

此外，拷贝真实 raw 后分别移除完整参数梯度、将真实梯度归零、把 BF16 运行的实际 compute
metadata 改成 FP32、让恢复端复用 producer PID；四个负例均被拒绝。拷贝的修改单独重绑定，
原 299 文件没有改写。记录为 `negative-recheck-abb4.json`。

安装环境最初的 Torch-first legacy-pickle 失败、错误 eager sitecustomize 造成的编译锁等待，
以及 world-one 不自动加载 NCCL 的 warm 失败均保留。最终 CPU 私有 prefix 只通过仓库
`compat/shim/deploy.py --target` 部署普通包入口，不保留自造 finder 或 eager startup hook，
不修改共享环境和 genuine Torch 参考环境。

## 后续门禁与边界

clean `abb4f302c` 的 CPU core 结果为 322 passed、42 skipped、1 xfailed。完整结构检查
为 1,507 passed、6 skipped、1 failed：新增工具测试在收集阶段动态加载模块，违反既有
collection side-effect 契约。`5844a4581` 将加载延后到 fixture 执行期，工具与 pytest/child
结构契约合计 46 passed、0 skipped；保留原失败，不将其改写为完整结构全过。
整合方在工具、报告与中央 index/MANIFEST 接入后统一复验完整结构。
日志为 `unit-cpu/core-final.log`、`unit-cpu/structure-final.log`、
`unit-cpu/evidence-fixture-followup.log`；其他哈希和布局记录在 `final-verification-manifest.json`。
CPU core runner 分别使用 native/Torch 会话缓存；runner 会覆盖外层 `JITTOR_HOME`，
实际路径已保留在日志中，与本轮 CUDA rank 缓存分开，不能把它声称为外层指定的 unit home。
报告加入后的 docs/private-path 定向检查为 18 passed、1 failed，唯一失败是新结果页
尚未接入中央 toctree；按文件所有权交由整合方补 results index/MANIFEST 后复验。
布局、改动 Python 文件 Ruff 与 diff whitespace 检查通过。
首次完整结构检查实际发现新 `JT_NCCL_LOCAL_WORLD_RANKS` 漏登记，已登记并定向复验通过。

本轮只有一台物理主机。host 的四 global ranks 只证明 rank 算术和协议；实际 CUDA 为两个
agent 各一张卡。真实跨主机 TCP/NCCL 路由、防火墙、不同文件系统与时钟、网络分区、
多 host 训练及恢复仍需真实多机验收。控制面不传输 checkpoint；现有共享文件路径要求保留。
多节点 HCCL、弹性成员/重启及任意 collective 的端到端 deadline 不在本轮已验收范围。
R17 不能因这批单 host 软件与设备证据整体关闭。
