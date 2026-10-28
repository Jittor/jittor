# Accelerate 维护版 CPU 全新进程 nightly

- 状态：完整本地 CPU nightly 与严格 L3 协议已接受；托管调度未验证。
- 复查日期：2026-10-16。
- 测量提交：`0dc8fdbd863c784dbe6564ebfeb9efb764b7b272`。
- 维护者：下游验证维护者。
- 复查条件：上游/wheel 固定版本、installed-source 证明、Module/优化器/RNG
  恢复、阶段 launcher、平台捕获、provisioning 或源码闭包变化。

新的 `accelerate` profile 会 provision 三个独立环境，并以完整的原始 world-one FP32 全新进程协议运行 `ecosystem_accelerate`。两个输入种子 0/7 保持相同的初始权重；每个 producer 执行三次 AdamW/StepLR 更新、保存、计算两个预期续训更新然后终止。新的 consumer 进程构造全新对象、恢复并执行同样的两次更新。没有削减任何阶段、字段、梯度或预算。

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py --library accelerate \
  --state-root "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/accelerate-nightly/fresh-run" --run
```

| 平台 | 卡 / 数量 | 精度 / 规格 | Oracle | 实际 flag |
| --- | --- | --- | --- | --- |
| Linux 6.1.188, AMD EPYC 9554, GCC 11.5 | CPU / 零张可见卡 | FP32 训练, Linear(3,2), batch 4, 两个运行时线程 | 官方 Torch 2.11.0+cpu, 已校验真实二进制 | CUDA/ACL 0, eager, TF32/HF32 false |

CUDA/CANN/驱动/nvcc 记录为不适用。worker 显式捕获空的 `CUDA_VISIBLE_DEVICES`；candidate 的 CPU 观测与 native flag 排除加速器执行。两侧都使用离线输入与 HF 离线模式。这是正确性证据，不含速度或峰值内存主张。

官方 Accelerate 1.10.1 源码仍为
`623cc0ba588d192c77758939a6e340adf07edd4f`；其官方 wheel SHA256 为
`3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11`。
正常 pip 会把完整的运行时 Requires-Dist 闭包分别解析进 candidate/oracle 前缀；nox controller 拥有自己的工具闭包。三者都通过 `pip check`。源码 editable 安装复用这个已解析闭包。没有复制任何安装前缀或 JIT 缓存。保留实际 pip RECORD，而不是用 archive RECORD 替换；仅凭 receipt 捕获并不主张每个传递依赖生成的 pyc 字节相等。

独立的只读审计把每个 Torch 安装与其官方归档的 11,945 个载荷文件比较，把每个 Accelerate 安装与其 88 个载荷文件比较，排除 archive RECORD。
每个实际的带哈希 RECORD 行都匹配：每个 Torch 安装 11,949 行，每个 Accelerate 安装 96 行。Torch 的 11,781 个包文件范围窄于其完整 wheel。candidate 上安装的 Torch 提供依赖 metadata；实际数值执行使用已断言的 native shim。oracle 的真实二进制单独哈希。全部 2,934 个框架 Git blob、可执行模式与符号链接目标文本都与测量归档匹配；全部八个 worker 映射都与其 14 个选定的已提交源码匹配，包括 Module loader 与被调用的控制项。

| 模型族 / 方法 / 精确规格 | CPU① 等级 | Device③ 等级 | 速度比 / 一致性 | 状态 / 证据 |
| --- | --- | --- | --- | --- |
| 仅协议：Linear(3,2)、batch 4、AdamW/StepLR、3+2 步、全新进程 | L3 | 未重跑；仅有独立 Module 单测 | 速度未测量；原数值预算 | 1 接受 / 0 阻塞 / 0 拒绝；`arrays-v2.json` |

维护 collector 恰好接受一条 L3 CPU 记录，零阻塞、零拒绝。独立 NumPy 复算验证了全部 438 对数组、28 对完整参数梯度与 14 对输入梯度、固定的完整初始权重、输入扰动、两侧/种子/阶段合计 56 次实际逐参数更新，以及精确的全新对象 model/optimizer/scheduler/RNG 与下一次随机抽样恢复。跨运行时最大绝对误差为 `5.960464477539063e-8`；原 `atol=2e-6, rtol=2e-5` 逐元素预算不变。已验证八个不同的 worker PID。16,616 项递归精确检查包含 RNG 标量，与早前报告计数方式不同的检查不可互换。

在 `7cba3f5462f0ba08f3c0c56053611640bdf8b588` 的一次初始完整运行保留了两个真实的 candidate consumer 失败：把加载的值复制进可训练 leaf 的操作跑在了 no-grad 之外。既有的连续复制路径现在通过公共 `copy_` 归属方，在内部原生 no-grad 作用域内执行复制，保留调用方的 grad 模式、分配、别名、leaf 与 frozen 标志。`assign=True` 路径未改动。
它不增加任何优化器或复制数学。独立的真实/candidate CPU 与 CUDA 各通过七项 FP32/FP64 契约，并校验了实际 CUDA 指针；既有 candidate 存储回归在每个设备上通过三项，包括 CUDA BF16。这些单测在 `23913169dde347754a8ee4d8817085196d0002da` 上测量。

另有一个可复现的原生 autograd 缺口仍未关闭：公共 `copy_` 之后经由更早的平方 consumer 反向，未能拒绝过期的已保存输入。Module 改动既不关闭它，也不主张完整的 mutation/version ABI 等价。一个旧的常量梯度乘法 consumer 在两个运行时上都仍然有效。修前的原始诊断与失败的完整运行保持不变。

完整 CPU 模式通过 `JITTOR_ACCELERATE_COMPLETE_CPU=1` 与 collector 默认关闭的 `--require-complete-cpu` 显式开启。既有的普通 CPU/CUDA 调用仍然可用。持久化产物使用 `JITTOR_ACCELERATE_EVIDENCE_DIR`；单测与 benchmark 缓存相互分离。基线 Git 仪器化声明捕获到的控制源码；缺失/被修改的当前源码会失败。历史 c7 的 48 个源码映射与四条记录保留其字段，并重放为两个接受/两个未配对阻塞/零拒绝。没有虚构注册表分母或旧的原始字段。

强制校验：core 322 passed / 42 skipped / 1 xfailed；
`tests/structure` 1,516 passed / 4 skipped；host 147 passed。Canonical Ruff、
布局、manifest 与 diff 检查通过。证据未版本化，
位于 `$JITTOR_LAB_ROOT/_state/accelerate-nightly/20261016/`：`fresh-v2`、
`arrays-v2.json`、`git-v2.json`、`installations-v2.json` 及其 `independent_*`
复核器。`measurement-seal-v2.json` 绑定 590 个常规文件 / 65,569,144 字节，外加
一个符号链接目标文本（591 条目）；SHA256
`0958756b13f54b073c532d7ae0fa4b2d3bd104da392a3f1b8c45d6ca1bcc1c46`。
更早的失败运行有单独的不可变封条；验证/交付 receipt
另行封存，不改动测量提交。

这个本地 nightly 不授予 CPU world-two FSDP2、不替代 DDP、不重跑
CUDA 全新协议，也不证明多主机/分片 checkpoint 或整库支持。
[历史 CPU/CUDA 源绑定结果](2026-10-16-accelerate-installed-source-protocols.md)
保留其原始提交与范围。工作流接线与本地执行都不能证明 GitHub 的托管调度实际运行过。
