## Description / 描述

<!-- Please provide a clear and concise description of your changes. -->
<!-- 请提供清晰简洁的修改描述。 -->

### Problem and intended behavior / 问题与预期行为

<!-- What problem does this solve? State the observable behavior before and after. -->
<!-- 解决什么问题？请说明修改前后的可观察行为。 -->


### Implementation boundary / 实现边界

<!-- Which modules, backends, public APIs, or packaging surfaces are affected? -->
<!-- 涉及哪些模块、后端、公开 API 或打包边界？明确未涉及的范围。 -->



## Related Issue / 关联 Issue

<!-- Link the related issue(s). Use "Fixes #<number>" to auto-close the issue when merged. -->
<!-- 关联相关 Issue。使用 "Fixes #<编号>" 可以在合并时自动关闭 Issue。 -->

Fixes #

## Personal Branch and Milestone / 个人分支与里程碑

<!-- Which long-lived personal branch is this PR from, and what independently
reviewable milestone does it complete? 一个 PR 应对应一个完整里程碑，不要求只包含一个提交。 -->

Personal branch / 个人分支：
Milestone / 里程碑：

## Type of Change / 修改类型

<!-- Check the relevant option(s). -->
<!-- 勾选相关选项。 -->

- [ ] Bug fix / Bug 修复
- [ ] New feature / 新功能
- [ ] Performance improvement / 性能优化
- [ ] Documentation update / 文档更新
- [ ] Test addition / 添加测试
- [ ] Refactoring (no functional changes) / 重构（无功能性变更）
- [ ] Other (please describe) / 其他（请描述）

## Changes Summary / 修改摘要

<!-- Describe the key changes made in this PR. -->
<!-- 描述此 PR 中的关键修改。 -->

-
-
-

## Testing / 测试

<!-- Describe the tests you ran to verify your changes. -->
<!-- 描述你运行了哪些测试来验证修改。 -->

<!-- Include exact commands, pass/skip counts, Python/dependency versions, device, -->
<!-- cache isolation, and known limitations. Do not report import success as -->
<!-- accelerator support. 请填写精确命令、通过/跳过数量、版本、设备、缓存隔离和限制； -->
<!-- 仅导入成功不能证明支持加速后端。 -->

```text
Command(s):
Environment / backend / device:
Result (pass/skip/fail):
Limitations or unavailable runners:
```

- [ ] I have run the narrowest relevant pytest/nox checks and added regression tests where behavior was corrected.
      我已运行最小相关 pytest/nox 检查，并在修复行为时添加回归测试。
- [ ] I have run every advertised backend on a real device, or documented the unavailable backend and maintainer follow-up.
      我已在真实设备运行所有声明支持的后端，或已记录不可用后端及维护者后续验证。
- [ ] I have run the relevant packaging, documentation, distributed, or benchmark gate when the change touches that surface.
      改动涉及打包、文档、分布式或性能时，我已运行相应门禁。

## Checklist / 检查清单

- [ ] My code follows the project's code style guidelines.
      我的代码遵循项目的代码风格指南。
- [ ] I have commented my code where necessary.
      我已在必要处添加了代码注释。
- [ ] I have updated the documentation accordingly.
      我已相应地更新了文档。
- [ ] My changes do not introduce new warnings or errors.
      我的修改没有引入新的警告或错误。
- [ ] I have read the [CONTRIBUTING](../CONTRIBUTING.md) guide.
      我已阅读[贡献指南](../CONTRIBUTING.md)。
- [ ] I have self-reviewed the rendered diff and staged only files in scope.
      我已自行检查渲染后的 diff，且只暂存了本次改动涉及的文件。
- [ ] I have synchronized my personal branch with the target branch before review.
      我已在评审前将个人分支同步到目标分支。
- [ ] I have checked whether this change needs a documentation, release-note, or follow-up Issue update.
      我已确认是否需要更新文档、发布说明或后续 Issue。

## Breaking Changes / 破坏性变更

<!-- Does this PR introduce any backward-incompatible changes? If yes, describe the impact and migration path. -->
<!-- 此 PR 是否引入了向后不兼容的变更？如果是，请描述影响和迁移方法。 -->

- [ ] No breaking change / 无破坏性变更
- [ ] Breaking change described below with migration path / 已在下方说明破坏性变更及迁移方法

Impact and migration / 影响与迁移：

## Additional Notes / 补充说明

<!-- Any additional context, screenshots, or information. -->
<!-- 任何额外的上下文、截图或信息。 -->
