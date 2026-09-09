# 文档工作流

`docs/` 下的中文 MyST Markdown 是 Sphinx 的唯一文档源。**不要新增按语言复制的
第二棵文档树**；仓库根目录的 `README.md` 是唯一的双语文件。

整改期的过程文档（任务看板、迁移期契约、验证报告）不在 `docs/` 里，而在仓库顶层的
`refactor-wip/`，整改收口后整个目录删除。

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
| `docs` | 严格构建（警告即错误），并校验 API 锚点与 `docs/api/inventory.json` 一致 |
| `docs_links` | 离线校验仓库内部链接、图片、MyST 角色与 toctree |
| `tutorials` | 生成 notebook 并执行离线 CPU 冒烟教程 |

`docs_links` 是 PR 上确定性的离线检查，只管仓库内部链接；它不去爬外部 URL。

## 外部引用

严格构建会用 Python、NumPy 和 PyTorch 的官方 intersphinx 清单解析交叉引用。
文档 CI 允许访问这三个来源，设有超时；清单不可用时按构建失败处理，不会降级放行。

## 关于翻译

文档源已改为中文，此前的 gettext 翻译目录（`docs/locales/zh_CN`）随之退役——它们
翻译的是不再存在的英文原文。如果之后要提供英文版，应新建 `docs/locales/en` 目录，
把中文作为源语言重新抽取，历史中文译文可从 Git 历史中取回作为素材。
