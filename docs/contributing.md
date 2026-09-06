# GitHub and documentation workflow

The complete GitHub collaboration policy is maintained in the root
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Read it before opening an Issue or
pull request; it defines the branch, commit, review, merge, CI-evidence, and
release rules. This page records the additional workflow for documentation
authors.

完整的 GitHub 协作规范维护在根目录的
[`CONTRIBUTING.md`](../CONTRIBUTING.md)。提交 Issue 或 PR 前请先阅读该文件，其中规定了
分支、提交信息、评审、合并、CI 证据和发布规则。本页补充文档作者需要遵守的工作流程。

## Documentation workflow / 文档工作流

English MyST Markdown under `docs/` is the only Sphinx source. Simplified
Chinese translations live in gettext catalogs under
`docs/locales/zh_CN/LC_MESSAGES`; do not add a parallel Chinese source tree.

Install the pinned Python 3.11 documentation environment and run the public
gates through nox:

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s docs
python -m nox -s docs_zh
python -m nox -s docs_links
python -m nox -s tutorials
```

`docs` and `docs_zh` build the current checkout as a wheel, install that wheel
in an isolated nox environment, and reject an autodoc import from the source
tree. HTML, gettext templates, compiled catalogs, doctrees, and generated
notebooks are written below `JITTOR_LAB_ROOT/_state`, never into the checkout.

## Update a translation

After editing English source, extract messages into external state and update a
temporary catalog copy:

```bash
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/docs-i18n"
python -m sphinx -W --keep-going -n -b gettext docs "$STATE/gettext"
cp -R docs/locales "$STATE/locales"
sphinx-intl update -p "$STATE/gettext" -d "$STATE/locales" -l zh_CN
```

Review the catalog diff, apply the intended `msgstr` updates to tracked `.po`
files, then run `python -m nox -s docs_zh`. The session repeats extraction on a
copy and fails if catalogs are stale, fuzzy, or obsolete. `.pot`, `.mo`, and
HTML files are generated products and must not be committed.

## External inventories

Strict Sphinx builds resolve Python, NumPy, and PyTorch references against the
projects' official intersphinx inventories. Documentation CI is network-enabled
for these three authoritative files, uses a bounded timeout, and treats an
unavailable inventory as a failed strict build. `docs_links` is the deterministic,
offline pull-request check for repository-internal Markdown, images, MyST roles,
and toctrees; arbitrary external URL crawling is not part of that gate.
