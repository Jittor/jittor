# 原生双节点验收入口：维护 runner 与单机真实 CUDA 复验

- 状态：维护入口、host 契约和单机双 agent CUDA 协议通过；真实多主机验收未完成。
- 日期：2026-10-15；Owner：Native distributed maintainers。
- 远端同步基线：`7a1e66b3858279b40eaec0c4895c6059ce8eb8db`；保留本地工作后的合并为 `5e4ca616584885a5553be025e3bb9df5b81a4f7f`。
- 最终测量源码：`0c0e711c71bf05f2e7ff54f51b50fbbff562813f`，测量时工作树干净。
- 复查条件：launcher、runner、源码/环境选择、NCCL、维护 FSDP fixture、独立 oracle 或实际主机拓扑改变。

## 缺口与入口

此前 skill 给出维护 fixture 与手工环境设置，实际编排脚本在仓外并写死本机路径/设备。
它自行启动两个本地 agent，不能作为两台真实主机各自执行的维护命令。

新增 [run_native_multinode.py](../../agent/skills/jittor-distributed-verification/run_native_multinode.py)
和 worker，另提供 `nox -s native_multinode -- <runner arguments>` 薄入口。
完整命令见 [distributed verification skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md#two-host-maintained-entry)。
显式参数绑定源码路径/SHA、候选与独立参考解释器、node rank、master 地址/端口、job id、
本地分配设备、私有缓存与共享结果目录。当前协议限定两个节点、每节点一张 CUDA 卡。

两台主机分别运行相同入口，按 `warm → collectives → parity → baseline → resume` 分阶段执行，
每阶段等待两端完成；node 0 在 parity 前另执行 `oracle`，最后执行 `seal`。
不发送 SSH，不配置远端，不自动传输 checkpoint；共享文件系统是明确前置条件。
每端 warm 使用公开 jtrun 的真实 rank 0/world 1，缓存索引不冒充通信 rank。

三个训练阶段各自创建新的公开 launcher/worker，调用原有维护 pytest 节点。
未修改 FSDP、优化器、RNG、checkpoint 数学或任何数值预算。
source snapshot 包含全部 Git 普通文件及符号链接目标文本；真实 host identity、PID、
解释器/prefix、运行环境允许字段、argv、返回码和日志哈希在实际进程中记录。
严格环境允许列表避免复制完整进程环境。失败保留 claim 和部分日志，禁止覆写重跑。

seal 要求两端所有阶段与四份独立参考都实际成功，然后绑定 raw 并调用 NumPy rechecker。
只有重算成功才写 `verification.json`。不同主机可以使用相同本地路径与设备 ordinal，
隔离按 `(实际 host identity, 路径/设备)` 判断；无 host identity 的历史记录保留严格隔离规则。
host identity 不等于外部物理拓扑证明，工具不据此授予真实多主机验收。

## 实际验证

真实 CUDA 使用同一物理主机的两张 RTX 4090、Cu122/NVCC 12.2.140，候选为原生模式。
独立解释器加载 binary Torch `2.11.0+cu128`，禁 TF32、使用 highest precision。
每节点使用独立 HOME/JITTOR_HOME/TMPDIR；首次核心、扩展和模型 JIT 依次串行完成。
最终复验沿用本任务各节点私有 unit cache；生产/数学源码没有变化，缓存不与 benchmark
或另一并发会话共享。耗时不作为性能或峰值内存结论。

| 检查 | 结果 | 实际范围 |
| --- | --- | --- |
| Host / 工具 / pytest / child / nox 邻接 | 78 passed，0 skipped | 包含实际 plan、子进程记录、解释器别名、源码与缓存隔离负例 |
| 完整结构 | 1508 passed，6 skipped | clean `0c0e711c7`；既有 skip 不计 CUDA 执行 |
| Public singleton warm | 两端均成功，fallback 增量 0 | 每端真实 rank 0/world 1；独立缓存 |
| Public 双 agent collectives | 每 rank 3 passed，0 skipped | 逆序/独立/singleton subgroup、WORLD 与混合集合 |
| Parity | 每 rank 4 passed，0 skipped | SGD/AdamW × FP32/BF16，五步独立 dense oracle、全梯度/参数/moments |
| Baseline | 每 rank 4 passed，0 skipped | dropout 0.25，第三步保存后继续两步 |
| Fresh resume | 每 rank 4 passed，0 skipped | 新 launcher/PID/model/optimizer，两步所有数组与四种后续 RNG 抽样 exact |
| Seal / 独立 NumPy 重算 | 680 / 368 / 160 / 160 | oracle 数组 / exact resume 数组 / update 预算 / zero-gradient 拒绝 |

最终整套运行刻意继承错误的源码路径与 CPU 测试路由，runner 仍正确绑定显式源码和 CUDA。
源审计另将两份 snapshot 的 2,818 个 Git blobs 逐一匹配，包含一个目录符号链接的目标文本。
历史 `bc07df0c` 的 299 文件未改写；新 rechecker 的历史复算 JSON 与旧结果逐字段相同。

首次 host 检查实际发现测试将完整环境交给 `child_env` 却缺 `inherit=False`；修复后通过。
首次实际 warm 另复现 `Path.resolve()` 把虚拟环境解释器替换为 base Python，报 tqdm 导入失败；
改为保留显式可执行路径，新增符号链接回归，并在 worker 读取真实解释器/prefix。
`796c0903` 的完整成功矩阵单独保留；随后补显式源码/设备路由 pin，最终以 `0c0e711c7`
在新结果目录重跑整套协议。原失败、原测量 SHA 和原始数组没有重标或覆写。

## 证据与边界

未版本化根目录：`$JITTOR_LAB_ROOT/_state/native-multihost-runner/20261015/`。
最终结果位于 `attempt3/shared/`，维护入口生成 `commands/`、逐节点/阶段日志、source snapshot、
完整 NPZ/checkpoint、worker identities、`raw-bindings.json` 和 `verification.json`。
282 份 raw 的绑定 SHA256 为 `6ece2b422165093e0a93a4523fd3b40f777b188c41b72cd986da879352a8cf5e`。
`independent-source-audit.json` 保存独立 Git/blob/源码与设备 pin 审计；
`structure-final.log`、`source-pin-fixed.log` 与最终验证 manifest 保存门禁及附加证据哈希。

这证明维护命令可在本机双 agent 上实际执行完整协议，不证明两台物理主机的网络、
防火墙、文件系统、网络分区处理或多机训练/恢复。该入口也不扩展到多节点 HCCL、
任意 world size、弹性成员或 checkpoint 传输。R17 的实际多主机验收继续保留。
