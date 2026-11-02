---
name: verl-torch-compat
description: 定位 verl 独立仓库的固定版本、官方测试与完整训练验收；主仓仅维护公共 Torch 语义和通用回归。
---

# verl × Jittor

下游专项实现、完整训练门禁、来源证明、负向测试与 nightly 维护在
[JittorEcosystem/jittor-verl](https://github.com/JittorEcosystem/jittor-verl)，
入口见其[测试说明](https://github.com/JittorEcosystem/jittor-verl/blob/main/docs/testing.md)。
主仓不再提供对应专项 nox session 或收集下游仓库的测试。

在独立仓库的干净 checkout 中执行：

```bash
python tools/provision_training.py --state-root "$JITTOR_LAB_ROOT/_state/verl/<run>" --run
```

已配置双解释器时使用独立仓库的 `tools/run_training_protocol.py`，显式给出
`--oracle-python`、`--candidate-python`、`--upstream`、`--wheel`、`--device` 和新外部 `--output`。
完整来源 pin 见独立仓库 `support/training-protocol.json`；先 CPU，再真实加速设备。
迁移后的 host 契约通过不等于重新跑过数值或硬件验收，不能沿用旧数字升级支持状态。

按[下游适配流程](../downstream-library-adaptation/SKILL.md)分流：核心能力修核心，
公共 Torch 签名和语义修 compat，库私有细节修独立仓库。主仓保留的通用生态 tiny smoke
不能替代完整专项验收。

## 历史证据

完整组件协议固定 `e5a627cf55c94d85406556854c2c1a22db4ff5e7` / `0.10.0.dev`，
与独立 adapter 的 `0.9.1` 版本不同；迁移后单独维护，不互相授予支持。

- [2026-10-15-verl-installed-source-training.md](../../../docs/results/2026-10-15-verl-installed-source-training.md)：保留原测量提交、包版本、设备和范围；不是迁移后的重新验收。
- [2026-10-15-verl-registered-component-oracles.md](../../../docs/results/2026-10-15-verl-registered-component-oracles.md)：保留原测量提交、包版本、设备和范围；不是迁移后的重新验收。
