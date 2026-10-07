# 贡献与文档工作流

**贡献流程的权威文档是仓库根目录的
[`CONTRIBUTING.md`](https://github.com/Jittor/jittor/blob/master/CONTRIBUTING.md)**：
开发环境、仓库结构、代码风格、测试分层、nox 门禁和合并请求要求都写在那里，本页不复制
一份。行为准则见
[`CODE_OF_CONDUCT.md`](https://github.com/Jittor/jittor/blob/master/CODE_OF_CONDUCT.md)，
角色与决策规则见
[`GOVERNANCE.md`](https://github.com/Jittor/jittor/blob/master/GOVERNANCE.md)。

本页只补一件根目录指南没有展开的事：**文档本身怎么改、怎么构建、怎么检查。**

## 文档树的规矩

`docs/` 下的中文 MyST Markdown 是 Sphinx 的唯一文档源。**不要新增按语言复制的第二棵
文档树**；仓库根目录的 `README.md` 是唯一的双语文件。

每一页只属于一个栏目：使用指南 [`guides/`](guides/index.md)、机制说明
[`notes/`](notes/index.md)、Torch 兼容 [`compatibility/`](compatibility/index.md)、
开发文档 [`development/`](development/index.md)、研究提案
[`research/`](research/index.md)、验证结论 [`results/`](results/index.md)、发布说明
[`releases/`](releases/index.md)。新增页面必须进对应栏目的 toctree，否则站点里没有
入口（结构门禁会报孤立页面）。

任务状态与交接记在 GitHub issue 与 PR 中，**不在仓库里维护看板**；PR 描述携带验证证据。
可复现的维护者验证结论写进 `docs/results/`，每份报告都要带状态、对应提交、验证范围、
维护者和复查条件，格式与必填元数据见
[NPU 缺陷与性能记录模板](development/npu-validation-templates.md)。

链接的两种写法：文档之间用相对路径，指向 `docs/` 之外的仓库文件用
`https://github.com/Jittor/jittor/blob/master/<路径>`。两种都会被检查，**相对路径不能
跨出 `docs/`**——Sphinx 构建不收录 `docs/` 之外的文件，那样的链接在站点上是坏的。

## 本地构建

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s docs
python -m nox -s docs_links
python -m nox -s tutorials
```

`docs` 会先把当前工作树构建成 wheel，在隔离的 nox 环境中安装该 wheel 再构建文档，
并拒绝 autodoc 从源码树导入——这样文档反映的是**发行物**的真实公开接口，而不是
源码树里恰好能 import 到的东西。

HTML、doctree 和生成的 notebook 一律写在 `$JITTOR_LAB_ROOT/_state` 之下，不落进
工作树。

## 检查项

| 会话 | 作用 |
| --- | --- |
| `docs` | 严格构建（`-W -n`，警告即错误），并校验 API 锚点与 `docs/api/inventory.json` 一致 |
| `docs_links` | 离线校验仓库内部链接、图片、MyST 角色与 toctree |
| `tutorials` | 生成 notebook 并执行离线 CPU 冒烟教程 |

`docs_links` 是 PR 上确定性的离线检查，只管仓库内部链接；它不去爬外部 URL。

## 外部引用

严格构建会用 Python、NumPy 和 PyTorch 的官方 intersphinx 清单解析交叉引用。
文档 CI 允许访问这三个来源，设有超时；清单不可用时按构建失败处理，不会降级放行。
**离线环境里这三份清单取不到，因此会多出若干 `ref.class` 警告**，那不是文档缺陷。

## 关于翻译

文档源已改为中文，此前的 gettext 翻译目录（`docs/locales/zh_CN`）随之退役——它们
翻译的是不再存在的英文原文。如果之后要提供英文版，应新建 `docs/locales/en` 目录，
把中文作为源语言重新抽取，历史中文译文可从 Git 历史中取回作为素材。
