# Archived ACL manual model probes

- Status: historical observations, not current hardware acceptance
- Review date: 2026-11-02 (file relocation only; no hardware rerun)
- Source baseline: `140dab6693972a01b31f21101879962d20db1763`; the original
  measurement commits and complete environments were not recorded
- Scope: former `tests/backends/acl/manual/` capacity and GPT-2/Qwen diagnostics
- Owner: Ascend backend maintainers
- Review when: reusing any number as a support claim; rerun with pinned versions,
  same-device independent oracle, full tensor comparisons and fallback accounting

The former manual directory mixed scripts with observations. Maintained native
probes now live in [`tools/probes/acl/`](../../tools/probes/acl/README.md).
Transformers-specific validation belongs to
[Jittor Transformers](https://github.com/JittorEcosystem/jittor-transformers).
The original documents remain accessible in Git history; the observations below
are preserved for context and are not newly reproduced results.

## Capacity observations

The former `MAX_MODEL_RESULT.md` reported a GPT-style decoder with RMSNorm,
vocabulary 50257, batch 1, sequence length 512 and one Adam step on a 64 GB 910B3:

| Dtype | Parameters | Recorded outcome |
| --- | --- | --- |
| fp32 | 1.6B | completed, approximately 39 GB HBM |
| fp32 | 2.5B | completed, near capacity |
| fp32 | 3.6B | out of memory |
| bf16 | 2.5B | optimizer operation error, not established as an OOM |

The report estimated a 2.5–3B full-finetuning ceiling per card, reported roughly
22 GB for Qwen3-8B bf16 LoRA, and extrapolated a roughly 20B LoRA ceiling and
8× data-parallel throughput. These extrapolations are **not measured limits**.
The toy capacity probe's unused LoRA branch did not connect adapters to its
forward pass, so it could not substantiate LoRA capacity; that branch is removed.
Data parallelism replicates the model and does not pool card memory. Neither
current bf16 support nor multi-card throughput can be inferred from this record.

## Numerical observations

The original GPT-2 report was dated 2026-06-22 (2 layers, 4 heads, width 128,
fp32, common generated weights and inputs):

| Backend | Loss | Total gradient L2 | Relative loss / norm / worst parameter norm |
| --- | --- | --- | --- |
| PyTorch CPU | 6.237304 | 0.045244 | reference |
| Jittor CUDA (4090) | 6.237304 | 0.045244 | 7.6e-8 / 1.0e-7 / 2.8e-7 |
| Jittor ACL (910B3) | 6.237304 | 0.045244 | 7.6e-8 / 4.7e-6 / 4.4e-5 |

These are **gradient norm** comparisons against a CPU reference, not elementwise
same-device forward/backward acceptance. Equal norms cannot exclude different
gradient directions. The original comparator printed differences without
failing on missing parameter names; the relocated comparator rejects incomplete
or out-of-tolerance measurements, with host-only regression tests.

The Qwen3-0.6B note recorded the same argmax (12095) and top five tokens
`[12095, 7407, 279, 1112, 30743]`, with logit norms 1493.90 (PyTorch CPU) and
1494.59 (Jittor NPU). It also quoted max logit error 0.0012, a CUDA GPT-2 loss
trajectory 6.39 → 1.41 over eight steps, and `.pt` loading on both backends.
The stored probe emitted only summaries, so those statements cannot be promoted
to current full-logit, training or serialization acceptance without the original
artifacts and a new reproducible run. Historical `import jittor as torch`
spelling is an implementation identity, not the recommended current shim API.
