# Personal Library Document Standard

Use this standard for one developer-owned document about one third-party library adapted through Jittor's torch compatibility layer.

## 1. Document Contract

The record is a maintained technical summary, not a diary. It must make the current state auditable without reproducing every implementation detail.

Required information:

- library name, owner when known, target versions or dependency bounds, branch or commit when available, current verified maturity, target maturity, supported runtime scope, and last update date;
- compatibility goal and explicit exclusions;
- implemented modules, APIs, protocols, or native capabilities;
- material problems, each with a C class, reproduction conditions, impact, resolution or current limitation, status, and evidence pointer;
- L0-L5 verification outcomes relevant to the current target;
- performance results when measured, including reproducibility conditions;
- actionable remaining work.

Use `待补充` for metadata that the user is expected to supply and `待验证` for technical claims that require evidence. These labels are different: missing metadata must not be presented as a failed technical check.

## 2. Problem Classes: C1-C5

Classify a problem by the capability and repair boundary, not merely by how difficult the debugging felt. Apply the class to each issue rather than to the library as a whole.

### C1: Existing Implementation Already Satisfies the Contract

Jittor has a single existing implementation that completely satisfies the relevant PyTorch contract.

Default scope: bind or publish the same object directly.

### C2: Protocol-Level Thin Wrapper

The computation exists, but parameters, defaults, return shape or structure, in-place behavior, or another public protocol detail differs.

Default scope: implement a thin wrapper in `compat.torch`; keep project-private protocols in `integration`.

### C3: Compatible Composition

A small number of Jittor primitives can be composed correctly and differentiably without fallback.

Default scope: compose in the compatibility layer; consider promotion to a native helper only after the behavior is shared and validated.

### C4: Missing Real Capability

Composition would be incorrect, invoke host fallback, cause unacceptable graph expansion, or create an unacceptable hot performance hotspot.

Default scope: implement in native Jittor, autograd, or the target backend.

### C5: Cannot Be Credibly Supported in the Current Boundary

The current ABI, dtype, quantization capability, or backend scope cannot credibly support the behavior.

Default disposition: define an explicit failure boundary and document the impact, candidate approaches, and exit conditions.

Every C5 entry must include this collaboration action:

**协作动作**：需同步至“严重问题”文档，并经全体开发者确认后再修改相关公共实现。

Do not soften a C5 into an undocumented fallback. Do not state that collective approval exists unless it is evidenced in the supplied material.

## 3. Maturity Levels: L0-L5

These are six maturity levels. Report the highest fully passed level as current. Levels are cumulative for maturity reporting: if a lower required level is unverified, a later isolated result may be recorded as evidence but must not raise the current level. Show partial work separately, such as `当前：L1；L2：进行中`.

### L0: Import and Construction

Acceptance scope: public imports, Config, AutoClass registration, tiny config, and parameter names, shapes, and dtype values.

Evidence: PyTorch and Jittor succeed in independent processes; model types and parameter inventories agree.

### L1: Forward Semantics

Acceptance scope: outputs under identical weights and inputs, shape, dtype, tuple/`return_dict`, masks, and error contracts.

Evidence: comparisons on CPU and real CUDA; the target path has no host fallback.

### L2: Training Semantics

Acceptance scope: task loss, gradients for all trainable parameters, train/eval, dropout, zero-grad, and at least one optimizer step.

Evidence: matching loss, gradient, and parameter-update trajectories; explicit checks for missing or abnormal gradients and mode switches.

### L3: Weights and Tasks

Acceptance scope: `state_dict`, safetensors, `save_pretrained`/`from_pretrained`, at least one formal task head, and tokenizer/processor behavior.

Evidence: zero missing/unexpected keys or an explicit whitelist; save/load round-trip comparisons pass.

### L4: End-to-End Capability

Acceptance scope: a real checkpoint; for generation models, prefill/decode, KV cache, greedy/beam/sample; for other modalities, real preprocessing.

Evidence: fixed-input end-to-end results, cache structures, and public API behavior match the baseline.

### L5: Performance

Acceptance scope: post-warmup forward, forward+backward or prefill/decode, full train step, throughput, and peak memory.

Evidence: reproducible statistics and hotspot localization under the same device, dtype, shape, and backend.

Performance data can be recorded before L4 is complete, but it does not establish L5 maturity until the preceding semantic and end-to-end requirements are met.

## 4. Evidence and Status Rules

- `已完成`: implementation exists and the stated acceptance evidence passes.
- `进行中`: implementation or verification is actively incomplete.
- `待验证`: an implementation or claim exists, but required evidence is absent or insufficient.
- `受阻`: progress depends on an identified unresolved condition; name that condition.
- `暂不支持`: the boundary is deliberate and explicit; state scope and user-visible behavior.

For numerical comparisons, record the metric and tolerance actually used. When only qualitative evidence is available, say so. Never invent commands, commit hashes, versions, measurements, tolerances, approvals, or pass results.

For performance claims, record at least device, dtype, backend, input shape, batch size when applicable, warmup count, measurement count, statistic, throughput or latency, and peak memory when available. If any condition differs between the two systems, label the comparison as non-equivalent and avoid a headline speedup claim.

## 5. Update Semantics

When updating an existing record:

1. Preserve still-valid scope, historical decisions, evidence links, and explicit limitations.
2. Merge new evidence into the relevant section instead of appending a chronological journal entry.
3. Recompute current maturity only when the new evidence closes every required acceptance item.
4. Mark a replaced conclusion as superseded when its history matters; remove only noise that has no collaboration value.
5. Use an absolute date in `YYYY-MM-DD` format for the last material update.

## 6. Feishu Direct-Paste Format

Use headings, paragraphs, numbered lists, and field-style bullets. Do not use Markdown pipe tables because Feishu may convert only the header row and leave subsequent rows as literal text. Avoid fenced code blocks in the direct-paste version; use inline code for short commands and file links for long logs.

Keep heading depth shallow. Prefer one issue subsection per material issue. Do not turn each sentence into a separate heading or bullet, and do not embed large logs or stack traces.
