---
name: torch-compat-library-docs
description: Create, revise, or audit concise personal development records for one third-party library being adapted through Jittor's torch compat layer. Use when raw notes, commits, tests, bugs, or performance results must become a Feishu-ready per-library collaboration document. Do not use for the project homepage, ecosystem-wide tracker, or shared C5 issue registry.
---

# Torch Compat Library Docs

## Scope

Own only the personal development subdocument for one third-party library. The document should let another developer quickly determine:

- what is in scope and what has been implemented;
- what remains unsupported or unverified;
- which problems occurred and how they were handled;
- what evidence supports the stated compatibility and performance.

Do not create or rewrite the project homepage, the ecosystem-wide ownership tracker, or the shared serious-problem document. When a C5 issue is present, flag the required handoff, but do not update the shared C5 registry unless the user separately requests that task.

## Required Resources

Before creating, updating, or auditing a library document, read [references/document-standard.md](references/document-standard.md) completely. It defines the C1-C5 problem taxonomy, L0-L5 maturity criteria, evidence rules, and Feishu formatting constraints.

For a new document or a full structural rewrite, use [assets/library-development-record.md](assets/library-development-record.md) as the starting structure. Remove irrelevant optional fields instead of filling the document with empty sections.

## Modes

Choose the narrowest mode that matches the request:

- **Create**: turn the user's notes and available project evidence into a new single-library record.
- **Update**: read the existing record first, preserve still-valid facts and links, then merge only the new progress, issues, evidence, and status changes.
- **Audit**: check a supplied record for unsupported claims, missing reproduction context, incorrect C/L classification, stale status, and Feishu paste hazards. Do not edit unless the user asks for changes.

## Workflow

1. Identify the library name, versions, owner if known, compatibility scope, and requested output location. Infer these from supplied notes or local artifacts when reliable; leave unknown fields as `待补充` rather than inventing them.
2. Inspect only relevant user-provided notes, existing documents, tests, logs, benchmark results, and commits. Treat a claimed result as verified only when the supplied or inspected evidence supports it.
3. Classify each material problem independently as C1-C5. A library does not have one global C level; place the class on each problem record.
4. Determine the current L level from completed acceptance evidence. Record the highest fully passed level and show the next partial level separately, for example `L2（L3 进行中）`. A target level is not a current level.
5. Write or patch the document using the template's information order. Prefer concise conclusions and reproduction pointers over chronological debugging narration.
6. For every C5 issue, add `协作动作：需同步至“严重问题”文档，并经全体开发者确认后再修改相关公共实现。` Also surface this handoff in the final response.
7. Update the absolute `最后更新时间` when materially changing a record. Preserve prior evidence and decisions unless newer evidence explicitly supersedes them; mark superseded conclusions rather than silently erasing useful history.
8. Validate the result against the completion checks below before delivery.

## Writing Rules

- Default to Chinese unless the user requests another language.
- Default to the Feishu direct-paste format: headings, short paragraphs, numbered steps, and field-style bullets.
- Do not use Markdown pipe tables or fenced code blocks in direct-paste output. Put short commands and identifiers in inline code; link to long logs or artifacts instead of embedding them.
- Keep the record simple and accurate. Omit raw trial-and-error logs, repetitive stack traces, abandoned experiments without decision value, and implementation details already clear from code or commits.
- State supported devices, dtype values, backends, task types, and relevant version bounds. Do not generalize beyond the tested scope.
- Distinguish `已完成`, `进行中`, `待验证`, `受阻`, and `暂不支持`. Do not convert absence of a failure report into a pass.
- Give exact test or benchmark conditions when reporting numbers. Comparisons without matching device, dtype, shape, backend, warmup, and measurement method must be labeled non-equivalent.
- Preserve the user's terminology and existing document links unless they conflict with the taxonomy. Explain any necessary reclassification briefly.

## Completion Checks

Before returning a created or updated document, confirm that:

- the document covers exactly one third-party library;
- scope and explicit exclusions are visible near the top;
- each issue has its own C classification, reproduction condition, resolution or limitation, and status;
- the current L level is supported by evidence and partial work is not reported as passed;
- test and performance conclusions include enough environment context to reproduce them;
- every C5 issue contains the required shared-review handoff;
- direct-paste output contains no lines beginning with the Markdown table delimiter `|` and no fenced code blocks;
- the record is concise enough to scan and does not duplicate full logs.

When writing a local file, report its absolute path and summarize any evidence gaps or C5 handoffs. Do not claim that Feishu itself was updated unless an actual Feishu integration performed that action.
