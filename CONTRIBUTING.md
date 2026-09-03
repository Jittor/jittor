# Contributing to Jittor / 为 Jittor 做贡献

Jittor accepts bug fixes, features, operators, tests, documentation, and
performance work. By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and the project's
[governance rules](GOVERNANCE.md).

Jittor 欢迎缺陷修复、新功能、算子、测试、文档和性能优化。参与项目即表示同意遵守
[行为准则](CODE_OF_CONDUCT.md)与[治理规则](GOVERNANCE.md)。

## Before you start / 开始之前

- Search [existing issues](https://github.com/Jittor/jittor/issues) before filing
  a new report. For behavior changes, describe the intended contract before
  implementation.
- Keep one pull request focused on one coherent change. Avoid generated files,
  runtime caches, model weights, and unrelated formatting churn.
- Reproduce a bug before changing code. Record the smallest command, device,
  input, expected result, and actual result.
- Do not claim accelerator support from import success. Exercise the operation
  on every backend the change advertises.

- 提交新 issue 前先搜索[已有问题](https://github.com/Jittor/jittor/issues)；行为变更应先说明预期契约。
- 一个合并请求只处理一组相关改动，不提交生成文件、运行缓存、模型权重或无关格式化。
- 修复前先复现，并记录最小命令、设备、输入、期望结果和实际结果。
- 仅能导入不代表支持加速设备；必须在所声明的每个后端执行目标操作。

## Development setup / 开发环境

Runtime compatibility is declared in [`pyproject.toml`](pyproject.toml). The
package supports Python 3.7 or newer; repository tooling uses pinned Python 3.11
environments and also compiles the tree with a real Python 3.7 interpreter.

运行时兼容范围以 [`pyproject.toml`](pyproject.toml) 为准。包支持 Python 3.7
及以上版本；仓库工具使用固定的 Python 3.11 环境，并用真实 Python 3.7 解释器编译检查全树。

```bash
git clone https://github.com/<your-account>/jittor.git
cd jittor
python -m pip install -e .
python -m pip install -r requirements/dev-tools.txt
python -m jittor.selftest
```

Jittor compiles C++ and CUDA code on demand. Use an isolated cache for each
checkout or concurrent job:

Jittor 会按需编译 C++ 与 CUDA。每个工作树或并行任务应使用独立缓存：

```bash
export JITTOR_HOME="${JITTOR_HOME:-$HOME/.cache/jittor-dev}"
export cache_name="${cache_name:-local-dev}"
```

For CUDA, set `nvcc_path` when `nvcc` is not already on `PATH`. Backend-specific
CI requirements are documented in [`agent/manuals/environment.md`](agent/manuals/environment.md).

## Repository map / 仓库结构

```text
.
├── python/jittor/       # runtime package and JIT C++/CUDA sources
├── python/jittor_utils/ # installation and compiler utilities
├── tests/               # repository pytest suite; not shipped in wheels
├── examples/            # runnable examples and MyST notebook sources
├── benchmarks/          # ASV performance suite
├── tools/               # build, release, install, and maintenance commands
├── docs/                # durable architecture, development, and release docs
├── agent/               # maintainer/agent workflow and verification reports
├── pyproject.toml       # authoritative package, tool, and pytest configuration
└── noxfile.py           # reproducible local and CI sessions
```

Runtime files loaded by path, especially `python/jittor/src/` and
`python/jittor/extern/`, have packaging and compiler contracts. Read the
[repository layout decision](docs/architecture/repository-layout.md) before
moving them.

## Making a change / 修改代码

Follow the style already used by the subsystem you edit. New Python code must
remain Python 3.7 compatible unless project metadata changes in the same
reviewed change.

修改代码时遵循对应子系统已有风格。除非同一变更明确调整项目元数据，否则新增 Python
代码必须兼容 Python 3.7。

### Python

- Use four spaces and descriptive names.
- Keep public APIs documented and typed where the surrounding module is typed.
- Do not hide import or installation failures with broad exception handlers.
- Add a focused regression test for every corrected behavior.

### C++ and CUDA

- Follow nearby naming, formatting, and generated-code conventions.
- Preserve CPU behavior when changing a CUDA path and vice versa.
- Treat dtype inference, broadcasting, empty tensors, non-contiguous layouts,
  gradients, and device dispatch as part of an operator's contract.
- First-time extension/JIT builds sharing a cache must run serially.

## GitHub collaboration workflow / GitHub 协作流程

This section is the repository's default workflow for day-to-day GitHub
development. GitHub branch-protection settings are part of the repository
configuration and must enforce the same rules on the configured default branch.
The repository may maintain multiple lines (for example `master` and `2.0`);
the GitHub repository setting and the Issue determine the correct target for a
PR. Do not assume that the checked-out branch is the default branch.

本节是日常 GitHub 开发的默认流程。GitHub 的分支保护设置属于仓库配置，必须在当前默认
分支上落实相同规则。仓库可能同时维护多条分支线（例如 `master` 和 `2.0`）；PR 的目标由
GitHub 仓库设置和 Issue 决定，不要假定当前检出分支就是默认分支。

### 1. Issue and scope / Issue 与范围

- Search existing issues first. A routine, narrowly scoped fix may open a PR
  directly, but a new public API, behavior change, backend change, packaging
  change, performance claim, or architectural change starts with an Issue.
- Give the Issue one owner, a concrete acceptance condition, and the affected
  area/backend. Link follow-up work instead of expanding one Issue indefinitely.
- One PR should implement one coherent intent. Split independent changes even
  when they touch the same files.

- 先搜索已有 Issue。日常且范围很小的修复可以直接发起 PR；新公开 API、行为变化、后端、
  打包、性能声明或架构变化必须先建立 Issue。
- Issue 应明确一个负责人、可验证的验收条件，以及受影响的模块和后端；后续工作用关联
  Issue 管理，不要无限扩大单个 Issue 的范围。
- 一个 PR 只实现一个完整意图；即使修改同一批文件，彼此独立的改动也应拆开。

### 2. Branches and remotes / 分支与远程仓库

- Treat the default branch as protected: do not push to it or commit directly
  on it. Release branches and tags are maintainer-owned.
- External contributors work from a fork. Committers may use a branch in the
  canonical repository, but the PR and review rules are identical.
- Each developer maintains one long-lived personal branch, named
  `dev/<name>` or `<team>/<name>` (for example `dev/zhangyi`). The branch is
  owned by one person and is the normal place for that person's daily commits.
  A short-lived task branch is optional when a change must be isolated from the
  personal branch; it is not required for every Issue.
- Open a PR from the personal branch only at a coherent, independently
  reviewable milestone. A milestone may contain several related commits and
  be a large update, but it must not combine unrelated work.
- Initialize the personal branch once from a clean, up-to-date target branch.
  Replace `<target-branch>` with the branch named by the Issue or maintainer:

  ```bash
  git fetch origin
  git switch <target-branch>
  git pull --ff-only origin <target-branch>
  git switch -c dev/<your-name>
  git push -u origin dev/<your-name>
  ```

  If the personal branch already exists on the remote, check it out with
  `git switch --track origin/dev/<your-name>` instead of creating it. If it is
  already present locally, update it with `git switch dev/<your-name>` followed
  by `git pull --ff-only origin dev/<your-name>`.

- Before opening or updating a PR, synchronize the personal branch with the
  target branch and resolve conflicts locally. Use a regular merge to preserve
  the long-lived branch history:

  ```bash
  git fetch origin
  git switch dev/<your-name>
  git merge --no-ff origin/<target-branch>
  git push origin dev/<your-name>
  ```

- Keep generated files, JIT/build caches, model weights, credentials, and
  unrelated formatting out of the branch. Use an isolated `JITTOR_HOME` or
  `cache_name` for each checkout or concurrent run.

- 默认分支必须受保护：禁止直接 push 或直接在其上提交；发布分支和 tag 由维护者管理。
- 外部贡献者从 fork 开发；Committer 可以在主仓库建立分支，但 PR 和评审规则完全相同。
- 每位开发者维护一个长期个人分支，命名为 `dev/<name>` 或 `<team>/<name>`（例如
  `dev/zhangyi`）。该分支由一人负责，是其日常提交的默认位置。只有在需要将某项改动与
  个人分支隔离时才创建短生命周期任务分支，不要求每个 Issue 都建新分支。
- 个人分支只在形成可独立评审和验收的里程碑时发起 PR。一个里程碑可以包含多个相关提交，
  也可以是较大的完整更新，但不能混入无关工作。
- 首次创建个人分支时，从干净且最新的目标分支开始；将 `<target-branch>` 替换为 Issue 或
  维护者指定的分支：

  ```bash
  git fetch origin
  git switch <target-branch>
  git pull --ff-only origin <target-branch>
  git switch -c dev/<your-name>
  git push -u origin dev/<your-name>
  ```

  如果远程已经有个人分支但本地尚未检出，使用 `git switch --track origin/dev/<your-name>`
  而不是创建新分支；如果本地已有该分支，则执行 `git switch dev/<your-name>` 和
  `git pull --ff-only origin dev/<your-name>`。

- 发起或更新 PR 前，先将个人分支同步到目标分支并在本地解决冲突。使用普通 merge 保留
  长期个人分支的历史：

  ```bash
  git fetch origin
  git switch dev/<your-name>
  git merge --no-ff origin/<target-branch>
  git push origin dev/<your-name>
  ```

- 生成文件、JIT/构建缓存、模型权重、凭据和无关格式化不得进入分支；每个工作树或并行
  运行都要使用隔离的 `JITTOR_HOME` 或 `cache_name`。

### 3. Commits / 提交信息

Keep each commit logically clear and reviewable. A long-lived personal branch
may accumulate multiple commits between PRs, and one milestone PR may contain
several related commits. The preferred subject format is:

```text
<type>(<scope>): <imperative summary>
```

Use one of `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `build`, `ci`,
`chore`, or `revert`; omit `(scope)` when it adds no information. Keep the
subject concise, explain non-obvious trade-offs in the body, and include issue
references when useful. Chinese summaries are fine, for example
`fix(reduce): 修正负整数 floor divide`.

Merge commits are allowed on a long-lived personal branch when synchronizing it
with the target branch; use the regular merge shown above and resolve conflicts
locally. Do not require squashing the personal branch before every PR. During
review, keep corrective commits when they make the discussion clearer, or
rewrite the branch only after coordinating with anyone who depends on it. Never
force-push without that coordination.

每个提交都应逻辑清晰、便于评审。长期个人分支可以在两次 PR 之间持续积累多个提交，
一个里程碑 PR 也可以包含多个相关提交。推荐主题格式为：

```text
<type>(<scope>): <祈使式摘要>
```

`type` 使用 `feat`、`fix`、`docs`、`test`、`perf`、`refactor`、`build`、`ci`、
`chore` 或 `revert`；括号中的 scope 没有信息量时可以省略。主题保持简洁，非显然的取舍
写在正文中，并在需要时关联 Issue。长期个人分支与目标分支同步时允许产生 merge commit；
评审期间可以保留修复提交，也可以在与依赖该分支的协作者沟通后整理历史。没有完成沟通时，
不要 force-push 或重写其他人正在使用的个人分支。

### 4. Local verification / 本地验证

Run the smallest relevant check first, then expand it according to the change:

| Change | Required evidence before review |
| --- | --- |
| Repository or documentation only | `python -m nox -s structure docs_links`; run `docs`, `docs_zh`, or `tutorials` when their sources are affected |
| Python API or refactor | focused pytest, import/identity checks where public APIs are involved, and `structure` |
| Core operator or autograd | independent forward reference, gradient check, and CPU regression |
| CUDA/NPU/ROCm behavior | real-device execution plus CPU/device parity on every advertised backend |
| Packaging or runtime resource | `packaging`/`structure`, wheel or sdist audit, and installed `python -m jittor.selftest` |
| Distributed behavior | focused multi-process test and the relevant MPI/NCCL/FSDP2 gate |
| Performance claim | correctness gate plus a reproducible benchmark comparison with the baseline commit |

The normal pull-request baseline is the `lint`, `format`, `typing`, `structure`,
`packaging`, and supported-Python checks exposed by `noxfile.py`. A local full
run is:

```bash
python -m nox -s lint format typing structure packaging py37 py312 py313
```

Hardware workflows use protected runners and are intentionally triggered on
trusted pushes rather than arbitrary fork pull requests. A contributor must
still attach the exact command, device, software versions, cache isolation,
pass/skip counts, and limitations in the PR. A maintainer reruns the privileged
CUDA/NPU gate before merge when the change affects that backend. An unavailable
runner is an environment limitation; it must not be converted into an
unconditional test skip or presented as backend support.

先运行最小相关检查，再按改动范围扩大验证。PR 的常规基线是 `noxfile.py` 暴露的
`lint`、`format`、`typing`、`structure`、`packaging` 和受支持 Python 版本检查；本地完整
运行命令为：

```bash
python -m nox -s lint format typing structure packaging py37 py312 py313
```

硬件工作流使用受保护 runner，有意只在可信 push 上触发，不对任意 fork PR 执行。贡献者仍
必须在 PR 中记录精确命令、设备、软件版本、缓存隔离方式、通过/跳过数量和限制；涉及后端
时，维护者在合并前重新运行有权限的 CUDA/NPU 门禁。runner 不可用属于环境限制，不能因此
改成无条件 skip，也不能据此宣称支持该后端。

### 5. Pull request lifecycle / PR 生命周期

1. Open a PR from the long-lived personal branch when a coherent milestone is
   ready for independent review and acceptance. Use draft status when early
   design or implementation feedback is useful. Set the target to the branch
   named by the Issue or maintainer, link the Issue (`Fixes #123` when the PR
   fully resolves it), and declare dependencies or stacked PRs.
2. Fill every section of the PR template: problem and intended behavior,
   implementation boundary, compatibility impact, tests with environment and
   backend, limitations, and documentation/release-note impact. Mark a draft
   **Ready for review** only after the checklist is truthful.
3. The author performs a self-review of the rendered diff, test output, public
   API changes, and staged files. Do not use `git add -A`; stage only files in
   scope.
4. Reviewers discuss correctness and maintainability in the PR. The author
   resolves conversations with a new commit or a clear reply; do not silently
   force-push away an unresolved decision.
5. A routine PR needs one approval from a relevant Committer or Module
   Maintainer and all required checks green. Core JIT, backend kernels, public
   API/compatibility, packaging, distributed, security, and other high-risk or
   breaking changes need two approvals, including the affected Module
   Maintainer, plus an Issue/design decision when applicable.
6. The merge owner confirms no unresolved conversations, a current base branch,
   passing required checks, and documented follow-up issues. Merge with
   **Create a merge commit** and keep the personal branch for later work. Before
   the next milestone, merge the latest target branch back into that personal
   branch. Only maintainers create release tags or merge release branches.

1. 个人分支形成可独立评审和验收的完整里程碑后再发起 PR；如果需要提前获得设计或实现反馈，
   可以先使用 Draft 状态。目标设为 Issue 或维护者指定的分支，关联 Issue（完全解决时使用
   `Fixes #123`），并声明依赖或堆叠 PR。
2. 填写 PR 模板的每一项：问题和预期行为、实现边界、兼容性影响、带环境和后端信息的测试、
   限制，以及文档/发布说明影响。只有清单内容真实完整时才标记 **Ready for review**。
3. 提交者自行检查渲染后的 diff、测试输出、公开 API 变化和暂存文件；不要使用 `git add -A`，
   只暂存属于本次改动的文件。
4. 评审在 PR 中讨论正确性和可维护性。提交者用新提交或明确回复解决对话，不要通过无提示
   force-push 隐去尚未解决的决定。
5. 常规 PR 需要相关 Committer 或模块维护者至少一项批准，并通过所有必需检查。核心 JIT、
   后端 kernel、公开 API/兼容层、打包、分布式、安全及其他高风险或破坏性变化，需要包括
   受影响模块维护者在内的两项批准，并在适用时先有 Issue/设计决策。
6. 合并者确认没有未解决对话、目标分支已更新、必需检查通过且后续工作已建 Issue；使用
   **Create a merge commit** 合并，并保留个人分支供后续开发。下一次里程碑开始前，先将最新
   目标分支 merge 回个人分支。发布 tag 或合并发布分支只能由维护者执行。

### Maintainer repository settings / 维护者仓库设置

Apply these rules to the configured default branch in **Settings > Branches >
Branch protection rules** (or the equivalent ruleset):

- require a pull request before merging; require at least one approval and
  dismiss stale approvals after new commits;
- require conversation resolution, successful required status checks, and the
  branch to be up to date before merging;
- disable force-push and deletion for the protected branch; allow
  **Create a merge commit** for normal changes and do not require a linear
  history;
- **Squash and merge** and **Rebase and merge** are not the default for this
  workflow; use them only when the merge owner explicitly documents an
  exceptional reason.
- do not enable automatic deletion of pull-request head branches, because
  long-lived personal branches must be retained. A personal branch may be
  force-pushed only by its owner, only when no collaborator depends on it, and
  only after the change is disclosed in the PR;
- require the `nox / lint`, `nox / format`, `nox / typing`, `nox / structure`,
  `nox / packaging`, `nox / py37`, `nox / py312`, and `nox / py313` checks for
  code PRs. Add `nox / docs*`, `nox / cpu`, or other checks to a path-specific
  ruleset when the repository's GitHub configuration exposes them as required;
- keep self-hosted CUDA/NPU checks out of the untrusted fork-PR required set.
  A maintainer must run the privileged gate on a trusted push and record the
  result before merging a backend change;
- when module ownership is formalized, add a `.github/CODEOWNERS` file and
  require the affected owner review. Until then, the merge owner explicitly
  selects a relevant Module Maintainer and records that decision in the PR.

在当前默认分支的 **Settings → Branches → Branch protection rules**（或等效 ruleset）中落实：

- 合并前必须通过 PR；至少一项批准，新提交后自动使旧批准失效；
- 必须解决所有评审对话、通过必需状态检查，并要求分支与目标分支保持最新；
- 禁止保护分支 force-push 和删除；普通改动允许使用 **Create a merge commit**，不要要求线性
  历史；
- **Squash and merge** 与 **Rebase and merge** 不是本流程的默认方式；只有合并者明确记录特殊
  原因时才使用；
- 不要启用自动删除 PR head branch 的设置，因为长期个人分支需要保留。个人分支只有在没有
  其他协作者依赖、且已在 PR 中说明时，才允许由其负责人 force-push；
- 代码 PR 必须通过 `nox / lint`、`nox / format`、`nox / typing`、`nox / structure`、
  `nox / packaging`、`nox / py37`、`nox / py312` 和 `nox / py313`。当 GitHub 配置将其暴露
  为必需检查时，再按路径规则加入 `nox / docs*`、`nox / cpu` 或其他门禁；
- 不要把自托管 CUDA/NPU 检查加入不可信 fork PR 的必需集合。后端改动合并前，维护者必须
  在可信 push 上运行有权限的门禁，并在 PR 中记录结果；
- 模块归属稳定后增加 `.github/CODEOWNERS` 并要求对应 owner 评审；在此之前，由合并者在
  PR 中明确选择相关模块维护者并记录这一决定。

### 6. Failures, exceptions, and releases / 失败、例外与发布

Classify a failing check before changing the code: (a) a reproducible product
failure belongs in the PR and usually needs a regression test, (b) a flaky or
infrastructure failure is rerun and recorded by a maintainer, or (c) an
unsupported backend is documented with an exact contract, owner, workaround,
and exit condition in `agent/manuals/known-issues.md`. Never hide installation,
registration, compilation, or backend errors with a broad exception or an
unexplained skip.

Release changes also update the relevant `docs/releases/` page and user-facing
documentation. Only a maintainer creates a `v*` tag after the release PR is
merged; the tag workflow builds and audits the sdist/wheel and performs the
platform validation described in the release documentation.

遇到失败时先判断类别再改代码：(a) 可复现的产品失败应在 PR 中说明并通常补回归测试；
(b) flaky 或基础设施失败由维护者重新运行并记录；(c) 不支持的后端必须在
`agent/manuals/known-issues.md` 中记录精确契约、负责人、workaround 和退出条件。不得用
宽泛异常或无解释的 skip 隐藏安装、注册、编译或后端错误。

发布相关改动同时更新 `docs/releases/` 对应页面和面向用户的文档。发布 PR 合并后只能由
维护者创建 `v*` tag；tag 工作流会构建并审计 sdist/wheel，并执行发布文档中说明的平台验证。

## Tests / 测试

The repository suite lives at root [`tests/`](tests/) and is collected by
pytest. Existing `unittest.TestCase` tests are valid pytest tests. Test
configuration and registered markers are in [`pyproject.toml`](pyproject.toml).

仓库测试统一位于根目录 [`tests/`](tests/)，由 pytest 收集；已有
`unittest.TestCase` 无需改写。测试配置和合法 marker 均在
[`pyproject.toml`](pyproject.toml) 中定义。

Put tests in the matching domain and name modules `test_*.py`. Apply the narrowest
backend or lifecycle marker (`structure`, `cpu`, `cuda`, `rocm`, `npu`, `mpi`,
`slow`, `network`, or `manual`). Skips and expected failures must state the exact
unsupported contract; do not turn failures into unconditional skips.

```bash
# Focused test
python -m pytest tests/nn/test_nn_capabilities.py -v

# One test case
python -m pytest \
  tests/nn/test_nn_capabilities.py::TestAttentionCapabilities::test_layout_lengths_and_cumulative_cache \
  -v

# Collection contract
python -m pytest --collect-only -q tests
```

### Nox gates / Nox 门禁

[`noxfile.py`](noxfile.py) is the canonical command surface. Its default local
sessions are `lint`, `format`, `typing`, `structure`, `packaging`, `py37`,
`py312`, and `py313`; hardware and documentation sessions are selected when
their surfaces are affected.

[`noxfile.py`](noxfile.py) 是统一命令入口；默认门禁包括 `lint`、`format`、
`typing`、`structure`、`packaging`、`py37`、`py312` 与 `py313`；涉及硬件或文档时再选择相应
session。

```bash
python -m nox
python -m nox -s structure
python -m nox -s cpu
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s benchmark
python -m nox -s docs
python -m nox -s docs_zh
python -m nox -s docs_links
python -m nox -s tutorials
```

The `structure` session checks repository layout, source and wheel contents, a
wheel built from the sdist, and the installed `jittor.selftest`. Hardware
sessions require a provisioned backend and accept pytest targets after `--`.
`docs` and `docs_zh` build the English and Chinese documentation with warnings
treated as errors, `docs_links` validates documentation links, and `tutorials`
materializes MyST sources in temporary storage and executes the maintained
notebook smoke tests.

`structure` 会检查仓库布局、源码包和 wheel 内容、从源码包生成的 wheel，以及安装后的
`jittor.selftest`。硬件 session 需要预先配置对应后端，并可在 `--` 后传入 pytest 目标；
`docs` 与 `docs_zh` 分别严格构建英文和中文文档，`docs_links` 验证文档链接，
`tutorials` 在临时目录生成 notebook 并执行维护的教程冒烟测试。

```bash
python -m nox -s cuda -- tests/nn/test_nn_capabilities.py
```

For changes spanning shared semantics, run the focused test first, then the
relevant CPU and accelerator gates. Use separate `JITTOR_HOME` or `cache_name`
values for concurrent jobs; do not run benchmarks and tests against one cache.

## Documentation / 文档

- Update [`README.md`](README.md) for installation or first-use changes. It is
  the single authoritative root README and keeps English and Chinese together.
- Put durable decisions under `docs/architecture/`, testing contracts under
  `docs/testing/`, development guidance under `docs/development/`, and research
  proposals under `docs/research/`.
- Put reproducible maintainer evidence under `agent/results/`; keep raw logs and
  generated artifacts outside the source checkout.
- Use relative links for repository files and run the structure gate after moves.

- 安装和首次使用发生变化时更新 [`README.md`](README.md)；它是唯一权威的根 README，
  中英文共同维护。
- 长期决策、测试契约、开发指南和研究提案分别放入 `docs/architecture/`、
  `docs/testing/`、`docs/development/` 与 `docs/research/`。
- 可复现的维护者结论放入 `agent/results/`；原始日志与生成产物不进入源码仓库。
- 仓库内文件使用相对链接，移动后执行结构门禁。

## Pull requests / 合并请求

A pull request should include:

- the problem and intended behavior;
- the implementation boundary and compatibility impact;
- exact tests run and their backend/environment;
- known limitations, skips, or deferred follow-up;
- documentation and release-note changes when users are affected.

Before requesting review, inspect the staged diff, avoid `git add -A`, and ensure
that only files belonging to the change are staged. Reviewers prioritize
correctness, public compatibility, backend behavior, packaging, and regression
coverage over cosmetic consistency.

提交评审前检查暂存区，只加入本次改动涉及的文件。评审会优先关注正确性、公开兼容性、
后端行为、打包边界和回归覆盖。
