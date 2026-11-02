---
name: ms-swift-torch-compat
description: 定位 ms-swift 独立仓库的固定版本、官方测试与完整训练验收；主仓仅维护公共 Torch 语义和通用回归。
---

# ms-swift × Jittor

下游专项实现、完整训练门禁、来源证明、负向测试与 nightly 维护在
[JittorEcosystem/jittor-ms-swift](https://github.com/JittorEcosystem/jittor-ms-swift)，
入口见其[测试说明](https://github.com/JittorEcosystem/jittor-ms-swift/blob/main/docs/testing.md)。
主仓不再提供对应专项 nox session 或收集下游仓库的测试。

在独立仓库的干净 checkout 中执行：

```bash
python tools/provision_training.py --state-root "$JITTOR_LAB_ROOT/_state/ms-swift/<run>" --run
```

已配置双解释器时使用独立仓库的 `tools/run_training_protocol.py`，显式给出
`--oracle-python`、`--candidate-python`、`--upstream`、`--wheel`、`--device` 和新外部 `--output`。
完整来源 pin 见独立仓库 `support/training-protocol.json`；先 CPU，再真实加速设备。
迁移后的 host 契约通过不等于重新跑过数值或硬件验收，不能沿用旧数字升级支持状态。

按[下游适配流程](../downstream-library-adaptation/SKILL.md)分流：核心能力修核心，
公共 Torch 签名和语义修 compat，库私有细节修独立仓库。主仓保留的通用生态 tiny smoke
不能替代完整专项验收。

## 历史证据

完整 Swift SFT 协议固定 `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` / `4.5.2`；
Swift 自带 LoRA tuner 与 Trainer LoRA 是不同场景。

- [2026-10-15-swift-trainer-installed-source.md](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-15-swift-trainer-installed-source.md)：保留原测量提交、包版本、设备和范围；不是迁移后的重新验收。
- [2026-10-14-downstream-oracle-scorecards.md](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-14-downstream-oracle-scorecards.md)：保留原测量提交、包版本、设备和范围；不是迁移后的重新验收。
