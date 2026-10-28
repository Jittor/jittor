# Contributing and documentation workflow

**The authoritative document for the contribution process is
[`CONTRIBUTING.md`](https://github.com/Jittor/jittor/blob/master/CONTRIBUTING.md) at the repository root**:
the development environment, repository structure, code style, test tiers, nox gates, and pull request
requirements are all written there, and this page does not duplicate them. For the code of conduct see
[`CODE_OF_CONDUCT.md`](https://github.com/Jittor/jittor/blob/master/CODE_OF_CONDUCT.md), and for roles and
decision rules see
[`GOVERNANCE.md`](https://github.com/Jittor/jittor/blob/master/GOVERNANCE.md).

This page fills in just one thing the root guide does not expand on: **how the documentation itself is changed,
built, and checked.**

## Rules for the documentation tree

The Chinese MyST Markdown under `docs/` is the default Sphinx source. Every page also has
an English page in the same directory (`X.en.md`), and every index comes in a pair
(`index.md` / `index.en.md`). The root uses `README.md` (Chinese) and `README.en.md`
(English). A change to one side must be mirrored in the other; do not add a second,
language-copied documentation tree.

Each page belongs to exactly one section: usage guides [`guides/`](guides/index.en.md), mechanism notes
[`notes/`](notes/index.en.md), Torch compatibility [`compatibility/`](compatibility/index.en.md), development
documentation [`development/`](development/index.en.md), research proposals [`research/`](research/index.en.md),
validation conclusions [`results/`](results/index.en.md), release notes
[`releases/`](releases/index.en.md). A new page must enter its section's toctree, or the site has no entry for
it (the structure gate reports orphan pages).

Task status and handoffs are recorded in GitHub issues and PRs, **not in a board maintained in the repository**;
PR descriptions carry the validation evidence. Reproducible maintainer validation conclusions go into
`docs/results/`, and every report must carry status, corresponding commit, validation scope,
maintainer, and review conditions; the format and required metadata are in
[NPU defect and performance record templates](development/npu-validation-templates.en.md).

Two ways to write links: use relative paths between documents, and
`https://github.com/Jittor/jittor/blob/master/<path>` for repository files outside `docs/`. Both are checked,
and **a relative path must not escape `docs/`** -- the Sphinx build does not include files outside `docs/`,
so such a link is broken on the site.

## Local build

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s docs
python -m nox -s docs_links
python -m nox -s tutorials
```

`docs` first builds the current working tree into a wheel, installs that wheel in an isolated nox environment
before building the documentation, and refuses autodoc imports from the source tree -- so the documentation
reflects the **distribution's** real public interface, not whatever happens to be importable from the source
tree.

HTML, doctrees, and generated notebooks are always written under `$JITTOR_LAB_ROOT/_state`, never into the
working tree.

## Checks

| Session | Purpose |
| --- | --- |
| `docs` | Strict build (`-W -n`, warnings are errors), plus validation that API anchors match `docs/api/inventory.json` |
| `docs_links` | Offline validation of in-repo links, images, MyST roles, and toctrees |
| `tutorials` | Generates notebooks and runs the offline CPU tutorial smoke set |

`docs_links` is a deterministic offline check on PRs; it only covers in-repo links and does not crawl external
URLs.

## External references

The strict build resolves cross-references using the official intersphinx inventories for Python, NumPy, and
PyTorch. Documentation CI is allowed to reach these three sources, with timeouts; when an inventory is
unavailable the build is treated as failed rather than let through in a degraded state.
**In an offline environment the three inventories cannot be fetched, so several `ref.class` warnings appear**;
those are not documentation defects.

## About translation

The default source language is Chinese, and English pages are maintained beside the Chinese
pages (`X.en.md`). The previous gettext catalogs (`docs/locales/zh_CN`) are retired; English
pages are no longer carried by `.po` directories.
