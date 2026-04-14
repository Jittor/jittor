# Jittor Governance / Jittor 治理文档

[English](#jittor-governance) | [中文](#jittor-治理文档)

## Overview

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators, maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/) (Computer Science and Computer Graphics Group, Tsinghua University). This document describes the governance structure of the Jittor project, including roles, responsibilities, decision-making processes, and how contributors can grow within the community.

Our governance philosophy is inspired by successful open-source projects, and is designed to be transparent, merit-based, and welcoming to all contributors.

## Governance Structure

Jittor adopts a layered governance structure with the following roles:

```
┌─────────────────────────────┐
│     Project Lead (BDFL)     │  Ultimate decision maker
├─────────────────────────────┤
│      Core Maintainers       │  Overall project direction
├─────────────────────────────┤
│     Module Maintainers      │  Specific module ownership
├─────────────────────────────┤
│        Committers           │  Trusted regular contributors
├─────────────────────────────┤
│        Contributors         │  Anyone who contributes
└─────────────────────────────┘
```

## Roles and Responsibilities

### Contributors

Anyone who contributes to the Jittor project in any form is a Contributor. Contributions include but are not limited to:

- Submitting pull requests (code, documentation, tests)
- Filing bug reports or feature requests
- Reviewing pull requests
- Participating in discussions on issues or forums
- Helping other users in the community

**How to become a Contributor**: Submit your first pull request or meaningful issue report.

### Committers

Committers are contributors who have demonstrated sustained, high-quality contributions and a solid understanding of the Jittor codebase.

**Responsibilities:**
- Review and approve pull requests in their area of expertise
- Mentor new contributors
- Help triage issues
- Ensure code quality standards are maintained

**Privileges:**
- Write access to the repository
- Ability to merge approved pull requests
- Listed in the project's Committers list

**How to become a Committer:**
1. **Sustained contributions**: Demonstrate a track record of multiple high-quality PRs over a period of at least 3 months.
2. **Code review participation**: Actively participate in reviewing others' pull requests.
3. **Community engagement**: Help other contributors and participate in technical discussions.
4. **Nomination**: Existing Committers or Maintainers may nominate a Contributor. The nomination is reviewed by Core Maintainers.

### Module Maintainers

Module Maintainers are responsible for specific modules or components of the Jittor project.

**Modules include (but are not limited to):**
- Core JIT Engine (`python/jittor/src/`)
- Neural Network Modules (`python/jittor/nn.py`)
- Operators (`python/jittor/src/ops/`)
- Model Library (`python/jittor/models/`)
- Documentation (`doc/`)
- Testing Infrastructure (`python/jittor/test/`)
- Build System and CI (`.github/workflows/`, `setup.py`)

**Responsibilities:**
- Drive the technical direction of their module
- Review and approve changes to their module
- Ensure their module maintains high quality and compatibility
- Respond to issues and PRs related to their module in a timely manner
- Write and maintain documentation for their module
- Coordinate with other Module Maintainers on cross-module changes

**How to become a Module Maintainer:**
1. **Deep expertise**: Demonstrate deep knowledge of the specific module through sustained, high-quality contributions.
2. **Ownership**: Show initiative in maintaining and improving the module (bug fixes, feature additions, documentation).
3. **Nomination**: Core Maintainers may nominate a Committer to become a Module Maintainer based on their contributions and expertise.

### Core Maintainers

Core Maintainers have overall responsibility for the Jittor project's direction, quality, and sustainability.

**Responsibilities:**
- Set the long-term vision and roadmap for Jittor
- Make final decisions on disputed or cross-module changes
- Evaluate and approve major feature additions or breaking changes
- Nominate and confirm new Committers and Module Maintainers
- Ensure the project maintains its open-source ethos
- Represent the project in external communications

**Current Core Maintainers:**
- [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/)

### Project Lead (BDFL)

The Project Lead serves as the ultimate decision-maker when consensus cannot be reached among Core Maintainers.

**Responsibilities:**
- Make final decisions on contentious issues
- Ensure the project stays true to its mission
- Confirm or remove Core Maintainers
- All decisions must be made publicly with clear reasoning

## Decision Making Process

### Routine Changes

For routine changes (bug fixes, small features, documentation improvements):

1. A Contributor submits a Pull Request.
2. A Committer or Maintainer reviews and approves the PR.
3. The PR is merged after CI checks pass.

### Significant Changes

For significant changes (new features, API changes, architectural decisions):

1. Open a GitHub Issue describing the proposed change.
2. Discuss the design with the community and relevant Maintainers.
3. After reaching consensus, implement the change and submit a PR.
4. The PR must be reviewed by at least one Module Maintainer of the affected area.
5. Core Maintainers have veto power over significant changes.

### Dispute Resolution

1. First, try to resolve the disagreement through discussion on the relevant issue or PR.
2. If unresolved, escalate to the relevant Module Maintainer(s).
3. If still unresolved, escalate to Core Maintainers for a final decision.
4. As a last resort, the Project Lead makes the final call.

All decisions must be documented publicly with clear reasoning.

## Nomination and Removal Process

### Nomination

- Any existing Committer or Maintainer may nominate a Contributor for promotion.
- Nominations should include evidence of the nominee's contributions, expertise, and community engagement.
- Core Maintainers review nominations quarterly.
- Decisions are based on **individual merit**, not organizational affiliation.

### Removal

- Removal can be initiated by any community member (including self-nomination).
- Reasons for removal include: extended inactivity (> 12 months), code of conduct violations, or irreconcilable conflicts of interest.
- Core Maintainers evaluate the situation and make a decision.
- Removed members may be granted "Emeritus" status in recognition of past contributions.

## Communication Channels

All governance-related discussions and decisions are conducted through public channels:

- **GitHub Issues**: For technical discussions and proposals
- **GitHub Pull Requests**: For code changes and reviews
- **Jittor Forum**: https://discuss.jittor.org/ - For general discussions
- **Email**: jittor@qq.com - For private matters (e.g., Code of Conduct reports)
- **QQ Group**: 761222083 - For real-time communication

## Changes to Governance

This governance document may be amended through the following process:

1. A proposal is submitted as a GitHub Issue.
2. The community discusses the proposed change for at least 2 weeks.
3. Core Maintainers vote on the proposal (majority required).
4. The Project Lead has final approval.

---

---

# Jittor 治理文档

## 概述

Jittor 是一个基于即时编译和元算子的高性能深度学习框架，由[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)维护。本文档描述了 Jittor 项目的治理结构，包括角色、职责、决策流程，以及贡献者如何在社区中成长。

我们的治理理念参考了成功的开源项目实践，旨在保持透明、基于功绩，并欢迎所有贡献者。

## 治理结构

Jittor 采用分层治理结构，包含以下角色：

```
┌─────────────────────────────┐
│     项目负责人 (BDFL)        │  最终决策者
├─────────────────────────────┤
│        核心维护者            │  项目整体方向
├─────────────────────────────┤
│        模块维护者            │  特定模块负责
├─────────────────────────────┤
│        提交者               │  受信任的常规贡献者
├─────────────────────────────┤
│        贡献者               │  任何做出贡献的人
└─────────────────────────────┘
```

## 角色与职责

### 贡献者 (Contributor)

任何以任何形式为 Jittor 项目做出贡献的人都是贡献者。贡献包括但不限于：

- 提交 Pull Request（代码、文档、测试）
- 提交 Bug 报告或功能建议
- 审查 Pull Request
- 参与 Issue 或论坛讨论
- 在社区中帮助其他用户

**如何成为贡献者**：提交你的第一个 Pull Request 或有意义的 Issue 报告。

### 提交者 (Committer)

提交者是已经展现了持续的、高质量贡献，并对 Jittor 代码库有扎实理解的贡献者。

**职责：**
- 在其专业领域审查和批准 Pull Request
- 指导新贡献者
- 协助分类处理 Issue
- 确保代码质量标准得到维护

**权限：**
- 仓库写入权限
- 合并已批准的 Pull Request
- 列入项目提交者名单

**如何成为提交者：**
1. **持续贡献**：展示在至少 3 个月内多次高质量 PR 的记录。
2. **参与代码审查**：积极参与审查他人的 Pull Request。
3. **社区参与**：帮助其他贡献者，参与技术讨论。
4. **提名**：现有的提交者或维护者可以提名一位贡献者，提名由核心维护者审查。

### 模块维护者 (Module Maintainer)

模块维护者负责 Jittor 项目的特定模块或组件。

**模块包括（但不限于）：**
- 核心 JIT 引擎（`python/jittor/src/`）
- 神经网络模块（`python/jittor/nn.py`）
- 算子（`python/jittor/src/ops/`）
- 模型库（`python/jittor/models/`）
- 文档（`doc/`）
- 测试基础设施（`python/jittor/test/`）
- 构建系统和 CI（`.github/workflows/`、`setup.py`）

**职责：**
- 推动其模块的技术方向
- 审查和批准对其模块的修改
- 确保模块保持高质量和兼容性
- 及时响应与其模块相关的 Issue 和 PR
- 编写和维护模块文档
- 与其他模块维护者协调跨模块变更

**如何成为模块维护者：**
1. **深入专业知识**：通过持续的高质量贡献展示对特定模块的深入了解。
2. **主动维护**：在维护和改进模块方面展现主动性（Bug 修复、功能添加、文档）。
3. **提名**：核心维护者可根据贡献和专业知识提名提交者成为模块维护者。

### 核心维护者 (Core Maintainer)

核心维护者对 Jittor 项目的方向、质量和可持续性负有整体责任。

**职责：**
- 设定 Jittor 的长期愿景和路线图
- 对有争议的或跨模块变更做出最终决策
- 评估和批准重大功能添加或破坏性变更
- 提名和确认新的提交者和模块维护者
- 确保项目维持开源精神
- 在外部沟通中代表项目

**当前核心维护者：**
- [清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)

### 项目负责人 (BDFL)

项目负责人在核心维护者之间无法达成共识时，作为最终决策者。

**职责：**
- 对争议问题做出最终决策
- 确保项目忠于其使命
- 确认或罢免核心维护者
- 所有决策必须公开并给出清晰理由

## 决策流程

### 常规变更

对于常规变更（Bug 修复、小功能、文档改进）：

1. 贡献者提交 Pull Request。
2. 提交者或维护者审查并批准 PR。
3. CI 检查通过后合并 PR。

### 重大变更

对于重大变更（新功能、API 变更、架构决策）：

1. 在 GitHub 上创建 Issue 描述提议的变更。
2. 与社区和相关维护者讨论设计方案。
3. 达成共识后，实现变更并提交 PR。
4. PR 必须至少由一位受影响领域的模块维护者审查。
5. 核心维护者对重大变更拥有否决权。

### 争议解决

1. 首先，尝试通过相关 Issue 或 PR 上的讨论解决分歧。
2. 如果未解决，升级至相关模块维护者。
3. 如果仍未解决，升级至核心维护者做出最终决策。
4. 作为最后手段，由项目负责人做出最终裁决。

所有决策必须公开记录并给出清晰理由。

## 提名与罢免流程

### 提名

- 任何现有的提交者或维护者均可提名贡献者晋升。
- 提名应包含被提名人的贡献、专业知识和社区参与的证据。
- 核心维护者每季度审查提名。
- 决策基于**个人功绩**，而非组织关系。

### 罢免

- 任何社区成员均可发起罢免（包括自我罢免）。
- 罢免原因包括：长期不活跃（> 12 个月）、违反行为准则、或不可调和的利益冲突。
- 核心维护者评估情况并做出决定。
- 被移除的成员可获得"荣誉退休"状态，以表彰其过去的贡献。

## 沟通渠道

所有与治理相关的讨论和决策均通过公开渠道进行：

- **GitHub Issues**：技术讨论和提案
- **GitHub Pull Requests**：代码变更和审查
- **Jittor 论坛**：https://discuss.jittor.org/ - 一般性讨论
- **邮箱**：jittor@qq.com - 私密事务（如行为准则举报）
- **QQ 群**：761222083 - 实时沟通

## 治理文档修改

本治理文档可通过以下流程修改：

1. 以 GitHub Issue 形式提交提案。
2. 社区对提议的变更讨论至少 2 周。
3. 核心维护者对提案投票（需多数通过）。
4. 项目负责人最终批准。
