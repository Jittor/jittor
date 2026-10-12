# jtrun P2P 路径验证（2026-10-02）

## 状态与基线

状态：**P2P 后端、原生多卡 smoke 和 launcher 的显式传输策略均已通过。**

本次检查基于 `integrate/cgq-transformers-native-multicard-target@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，同步时记录的 `origin/cgq_transformers` 为同一 SHA，`origin/2.0-refactor` 为 `077e07aaea4d5fb1c511971cb35328b982f45e9e`。

## 硬件与驱动信号

- 主机有 10 张 NVIDIA A800 80GB PCIe；驱动为 `550.144.03`。
- NCCL 为 `2.21.5+cuda12.4`，CUDA Runtime 报告 10 个设备。
- 选取四张同一 NUMA 节点的 A800 PCIe 做四 rank 验证，其中两张已有外部训练进程，smoke 只使用小张量和小模型。
- `nvidia-smi topo -p2p r`、`-p2p w` 和 `-p2p p` 对设备矩阵均报告 `OK`。
- CUDA Runtime 的 `cudaDeviceCanAccessPeer` 对选定设备的所有设备对均为可访问；双向 1 MiB `cudaMemcpyPeer` 对全部 12 个方向均通过。重复调用 `cudaDeviceEnablePeerAccess` 返回的 `cudaErrorPeerAccessAlreadyEnabled`（704）按幂等成功处理。

## P2P-enabled jtrun smoke

launcher 现在只继承调用方显式提供的 `NCCL_P2P_DISABLE`，不再无条件写入 `1`；因此显式设置为 `1` 仍可统一选择共享内存路径，未设置或设置为 `0` 则交给 NCCL 在所有 rank 上共同选择。修改后的源码入口分别在显式 `NCCL_P2P_DISABLE=0` 和完全未设置该变量的父环境中直接运行 `tests/distributed/native_nccl_smoke.py`，不再需要包装 `_rank_environment`。

验证结果：

- 两次 4 rank launcher 均返回 `rc=0`，所有 rank 正常退出，rendezvous 文件和进程均清理。
- rank 0 报告 `NATIVE_NCCL_SMOKE_OK checks=20`。
- 覆盖 NCCL sum/broadcast/all-gather、BF16 归约、Dataset 分片、SyncBN、参数与 buffer 广播、DDP 梯度平均、`no_sync` 和最终副本一致性。
- 两次运行的 NCCL 日志均出现 28 条 `Channel ... via P2P/CUMEM`，四个 rank 均有 `ncclCommInitRank ... Init COMPLETE`；没有出现 `via SHM` 或 `via NET` 的 collective transport 记录。

原始 rank 日志和隔离缓存位于 `$JITTOR_LAB_ROOT/_state/jtrun-p2p-20261002/`，不进入版本库。运行时需要预加载系统 `libstdc++`，以避开 Conda 库缺少 `GLIBCXX_3.4.30` 的环境问题；这不是 NCCL/P2P 错误。启动器环境回归同时确认显式 `NCCL_P2P_DISABLE=1/0` 会传到每个 rank，未设置时不会被 launcher 注入。

`tests/distributed/test_launch.py` 在相同的系统运行库条件下 **16 passed**；其中显式 `1`、显式 `0` 和未设置三种环境分别覆盖了所有 rank 的继承与不注入行为。

## 结论与后续边界

这台服务器已经可以验证开启 P2P 后的 NCCL 与原生 DDP，当前四卡拓扑实际选择了 P2P/CUMEM，不能再把本机归类为“无法运行 P2P 的机器”。源码修改后，正式 launcher 也已能保留调用方的 P2P 设置；不设置变量时由 NCCL 自主选择，需要回退时仍可显式使用 `NCCL_P2P_DISABLE=1`。

当前没有做 P2P 与共享内存的吞吐对照，也没有宣称多机或 FSDP 支持；后续性能结论仍需在负载稳定的主机上单独测量。
