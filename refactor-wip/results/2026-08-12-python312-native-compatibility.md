# Python 3.12 与旧接口兼容验证

日期：2026-08-12

代码基线：`dccca5b2`，验证改动位于独立工作树
`codex/repository-modernization-final-fixes`

## 结果

Jittor 现有公开 Python 接口在 Python 3.12 上可以从 wheel 冷启动、执行前向并计算梯度。
Python 3.7 仍是语法下界，Python 3.12 成为 CI 中真实解释器、真实构建和真实运行的上界门禁。

旧 Jittor 用户代码的兼容边界是公开 import、函数签名、对象 identity 和 pickle 路径；
`jittor_fsdp2/`、`torch_fsdp2_compat.py` 等曾经混入仓库的临时物理路径不是公开实现边界，
也不会为了“兼容”重新引入。规范实现迁移后，确有历史公开价值的名称由中央 alias registry
解析到同一个对象。

## 修正

- 新增 `py312` Nox 与 CI job：在 Python 3.12 中编译全仓 Python 文件，构建 wheel，
  安装到源树外目录，并运行 CPU `jittor.selftest`。
- 修复 Python 3.12 会报告的无效转义序列；门禁以 `-W error::SyntaxWarning` 执行。
- Torch 兼容层安装后，`jt.seed(value)` 继续调用 Jittor 原生种子接口；无参数
  `torch.seed()` 与 `torch.random.seed()` 保持 Torch 语义。
- `uniform_`、`constant_` 同时接受旧 Jittor 和 Torch 关键字，不再破坏原生初始化代码。
- transform 内部不再依赖被 Torch 兼容层赋予不同语义的 `Var.data`；einops 的 min/max
  reduction 显式选择 Jittor 的 values-only 语义。
- wheel 当前发布基线与 2026-08-12 最终整理历史快照分离。历史迁移仍逐文件审计，当前
  wheel 的合法内容变化也不再篡改历史结论。

## 验证

| 门禁 | 最终结果 |
| --- | --- |
| `nox -s py312` | Python 3.12.13；620 个 Python 文件无 `SyntaxWarning`；wheel 构建、源树外安装和 `jittor.selftest` 通过 |
| Python 3.12 CPU 数值 | forward 为 `(1, 4, 9)`，梯度为 `(2, 4, 6)` |
| 结构门禁 | 219 passed，2 skipped；无失败 |
| 旧接口结构回归 | 95 passed；覆盖 legacy import、identity、pickle、nn、misc、pool、optim |
| 扩展功能回归 | 249 passed，11 skipped；覆盖初始化、transform、einops 与 Torch 兼容冲突 |
| 随机数/初始化定向回归 | 6 passed，2 skipped |
| wheel 内容 | 786 个成员，0 新增、0 变化、0 删除；SHA-256 `fafbb9627d4275dded5a959657c4dad6054a1196b44119f5ec5bbdb5d114d2c1` |
| 静态检查 | `git diff --check`、仓库布局、Python 3.7 grammar 和 Python 3.12 compile 通过 |

本轮未重新执行真实 Python 3.7 全仓门禁；上一轮最终验收已在真实 Python 3.7 上编译
620 个文件，本轮修改文件另经 Python 3.7 grammar 检查。真实 Python 3.7 与 Python 3.12
CI job 会分别守住两端。

原始 wheel、Nox 环境、JIT 缓存和临时日志位于仓库外：

```text
${JITTOR_LAB_ROOT}/_state/verify/jittor-todo/nox-py312/
${JITTOR_LAB_ROOT}/_state/nox/
```

主工作树中的用户改动未参与本轮修改；验证前后其 diff SHA-256 均为
`29b5a063f975d0897dbfd5c5730762833323b513d833ae1f3e9af51411e81b2f`。
