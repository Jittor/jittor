# Jittor Governance

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
- Core JIT Engine (`src/`)
- Operators (`src/ops/`)
- Backends (`backends/`: `cpu/ cuda/ acl/ rocm/ corex/ comm/`)
- Neural Network Modules (`python/jittor/nn/`)
- Model Library (`python/jittor/models/`)
- Torch Compatibility (`compat/`) and optional adapters (`adapters/`)
- Documentation (`docs/`)
- Testing Infrastructure (`tests/`)
- Build System and CI (`.github/workflows/`, `noxfile.py`, `pyproject.toml`)

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
