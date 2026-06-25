# Jittor Agent 工作规范

本目录是 AI agent 在 jittor 仓库协作开发的工作空间。

## 开工流程

1. **先读上下文**：阅读 `skills/jittor-dev-context/SKILL.md` → 它会指引你读 `CONTEXT.md` 了解项目全貌（目标、进度、环境、已知 bug）。
2. **确认自己的工作文档**：开始任务前，**主动询问用户**你的工作细节应该记录在哪个文档里（可以是已有文档的某个章节，也可以是新建的子文档）。
3. **开始工作**，遵循下面的协作规范。

## 协作规范

### 文档更新（核心纪律）

- **`skills/jittor-dev-context/CONTEXT.md`** 是全局进度文档，所有人共享。当你完成了有意义的进展（修了 bug、跑通了验证、发现了新问题），**必须更新 CONTEXT.md 中对应的章节**，保持它是最新的全局状态。
- **你自己的工作文档**：记录具体的调试过程、实验细节、中间结论等。这样不同 agent 的工作内容可以清晰地合并，不会互相覆盖。
- 更新文档时注明状态（✅ 已完成 / 🟡 进行中 / 🔴 有问题），便于他人接手。

### Skill 沉淀

工作过程中写出的**可复用工具**（对拍脚本、验证 harness、调试探针等），沉淀为 skill：

- 在 `skills/` 下创建新目录，包含 `SKILL.md`（说明用途和用法）+ 工具文件。
- 已有 skill 直接复用，别重新造轮子。

### 效率原则

- **GPU 先行**：先在 GPU 上跑通全流程，再做 NPU 验证。
- **计算在 device 上**：torch_compat 里新加的计算必须能跑在 GPU/NPU 上，不能只支持 CPU。
- **多用 subagent**：互不依赖的验证任务并行展开（分卡并行、多模型并行、三后端并行）。
- **verify-then-fix**：~75% 的审计是误报，先复现再修。

## 目录结构

```
agent/
├── README.md                           ← 本文件（工作规范）
└── skills/
    ├── jittor-dev-context/             ← 项目上下文（开工必读）
    │   ├── SKILL.md                    ← 入口指引
    │   └── CONTEXT.md                  ← 全局进度文档
    └── jittor-torch-diff/              ← 对拍/调试工具集
        ├── SKILL.md
        ├── parity.py
        ├── run_parity.sh
        ├── op_parity.py
        ├── grad_probe.py
        ├── grad_ops.py
        └── linalg_grad_check.py
```
