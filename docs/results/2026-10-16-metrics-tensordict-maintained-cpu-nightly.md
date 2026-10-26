# Metrics 与 TensorDict 维护版 CPU nightly

- 状态：完整全新本地 CPU nightly 通过；托管调度执行仍未观测。
- 复查日期：2026-10-16；维护者：下游验证维护者。
- 测量：`c47033ed0a6bb692922efa98ecbbf7896bfe867f`，整个执行过程保持干净。
- 复查条件：源码/wheel 固定版本、安装证明、builder、梯度、精度、
  provisioning、运行时隔离或源码闭包校验变化。

维护的 `metric_tensordict` profile 现在运行两个完整原始组件
协议。它不重标历史 `912` CPU/CUDA 测量或
更早的 `90ee` 记录。[安装报告](2026-10-16-metrics-tensordict-installed-source.md)
保留那些原始身份与范围。

| 实际 CPU 组件 | 完整协议 | 结果 | 公共 API 覆盖 |
| --- | --- | --- | --- |
| TorchMetrics 1.7.4 | MeanAbsoluteError + MeanSquaredError；FP32 prediction/target `(4,8)`；种子 0 与固定权重种子 7；两个输入梯度；零参数 | 四个 worker，六对数组，精确相等，L2 | 2/105 |
| TensorDict 0.10.0 | TensorDict + 实际 lazy_stack；FP32 sample `(2,3,4)`，batch `[2,3]`，Linear(4,3)；两个参数梯度与 sample 梯度；种子 0/7 与 lr .001 的三次 SGD 步 | 六个 worker，29 对，最大绝对 9.536743e-7，L3 | 2/60 |

未改动的预算为 **.002 绝对 或 .01 逐阶段全局缩放**。
共同审计器接受两条 CPU 记录，零阻塞/拒绝行。
独立 NumPy 审计重算了全部 35 对、38 个非零梯度数组与
两个运行时中的 12 次真实参数更新。最大全局缩放误差为
6.690514e-8。更新算术满足独立的两倍 epsilon float32
舍入界限；其最大偏差为 1.323223e-8。每个阶段都使用相同的
完整初始权重，每个运行时的输出在种子 7 下改变。
真实干净上游清单保留了全部 105/60 个公共导出。

## 实际环境与源码权威

三个私有环境都是全新 provision 的，走完整正常 pip
解析并通过 `pip check`：candidate、真正二进制 Torch oracle 与
nox controller。NumPy/SciPy 为 1.26.4/1.13.1。oracle 为 Torch 2.11.0+cpu；
candidate 报告 Jittor Torch frontend 2.0.0 并执行原生 CPU kernel。
candidate 中的真实 Torch distribution metadata 不是其数值 provider。
未使用任何依赖闭包绕过；本地 editable 源码安装复用了
已检查的闭包。没有复制任何安装或 JIT 缓存。

全部十个 worker 都直接捕获 `CUDA_VISIBLE_DEVICES=''`、CUDA 可用性
false 与零张已分配卡。其 74 条张量观测确认 CPU；
原生观测还要求 CPU 驻留/dispatch 与 fallback
error/count0。主机为 Linux 6.1.188/glibc 2.34、AMD EPYC 9554、g++ 11.5.0。
两个运行时都使用两个线程、highest 精度、TF32/benchmark 禁用。
原生图 replay 为 1，保留预算 268435456 字节。
HOME/JITTOR_HOME/TMPDIR/XDG 根按运行时分离；单测缓存
独立于 benchmark，首次编译串行执行。

| 固定官方构件 | SHA256 | 排除 RECORD 的整归档文件 / 包文件 / 实际带哈希 RECORD 行（各计算侧） |
| --- | --- | --- |
| Torch 2.11.0+cpu | `8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` | 11945 / 11781 / 11949 |
| TorchMetrics 1.7.4 | `9298ad0e893b0cf2956bee95b0f7eecdc65205ab84ddb4f6762eff157c240518` | 350 / 346 / 353 |
| TensorDict 0.10.0 | `bccb6a7d7f02a8c83ca41815943bfa284da7d28019ab4276135809bda5ba05a6` | 46 / 42 / 49 |

实际上游 commit 为 `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` 与
`c34264a58b12471dc3e2913ea586b32431872d4c`。provisioning 获取真实
TensorDict `v0.10.0` tag 并要求其 peeled commit；无 tag 的 checkout
在该修复之前确实被拒绝。TensorDict 生成的 `_version.py` 与
原生扩展由其固定整 wheel 哈希保护。
实际 pip METADATA/RECORD 与完整目标包载荷被重新校验；
本 profile 从不用归档 RECORD 覆盖已安装 receipt。独立
Torch 证明包括 functorch/torchgen/dist-info，超出仅 torch 的范围。
oracle 二进制哈希被捕获；candidate 的 JIT 二进制 SHA 未被
本协议捕获，因此原生源码/dispatch 证据不被称为二进制哈希证明。
全部 2929 个 Git blob、模式与符号链接目标文本都与归档树匹配；
每个 worker 的 1613 个选定源码哈希都匹配同一 commit。

三个前缀中全部 85 个已安装 distribution 的原始 metadata/receipt
被保留。一个辅助的全依赖 receipt 检查器暴露了 NumPy 生成的
`conv_template.pyc` 重复行与陈旧的原始字节码哈希；其两个失败
诊断被保留。这不主张所有传递
RECORD 行字面相等，也不削弱目标包校验。没有改动任何已安装字节。

## 复现与校验边界

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py --library metric_tensordict \
  --state-root "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/metric-td-nightly/fresh-run" --run
```

可选的只读 `--source-cache`/`--wheel-cache` 仅复用下载输入。
专用 nox session 要求两个组件与全部十个 worker，并以既有 producer、verifier、training probe 与
collector 调用 `run_metric_td_gate.py`。它不允许削减 case/step/预算。既有六个 profile 保留
原固定版本、根与 session 行为。工作流把本 profile 加入
CPU 矩阵；接线与本次本地执行都不能证明托管调度实际运行过。
历史 `912` 重放仍以精确的原始 1609 字段
源码映射接受四条记录；新 runner 只有出现在测量 Git 树中时才被捕获。
一次单独的根集成复查成功地把全部三项完整数组/Git/目标安装
审计对照原始 c470 产物重放。

校验：core 322 passed / 42 skipped / 一个预期失败；完整 `tests/structure`
1516 passed / 四个 skipped；定向 host 契约 135 passed。最终文档、
布局、生成 manifest、Ruff 与空白检查通过。原始数组、命令、锁、原始 receipt、
独立复核器、失败诊断与源码归档位于
`$JITTOR_LAB_ROOT/_state/metric-td-nightly/20261016/`；`measurement-evidence-seal-v1.json`（761 个文件，351659740 字节；SHA256
`0ef2211694cc8dd22e48ef45de6627e96c5f8d4630706894c89545d5d6a66fd8`）绑定这些证据。
本批不授予新的 CUDA 测量、整库、其他 dtype/后端、
分布式、L4 或 L5 结论。
