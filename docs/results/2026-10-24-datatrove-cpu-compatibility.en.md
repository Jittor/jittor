# DataTrove CPU dataset and local pipeline compatibility verification

- Status: functional parity for three fixed Linux CPU scenarios passed
- Date: 2026-10-24
- Baseline commit: `ca56aa802c2acf9a22bca32d1bee3276381ed203` plus this PR's tests and docs
- Owner: yzhouli
- Review when: DataTrove dataset/executor, Torch shim release, or Jittor build entry changes

## Submission scope

DataTrove `v0.10.0`, upstream `https://github.com/huggingface/datatrove.git`, commit `7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`. This PR adds regression tests and reproduction notes for the CPU dataset, local JSONL pipeline and local multi-worker executor. The ordinary pipeline runs directly; the dataset uses the existing Torch compatibility layer; no dedicated adapter, core or backend changes.

## Environment and identity verification

This run executed through Slurm on a Linux x86_64 compute node, job `119913`, CPU only. Python 3.11.13, GCC 11.5.0, NumPy 2.4.6, DataTrove 0.10.0, multiprocess 0.70.19. Independent oracle is PyTorch `2.8.0+cu128`, verified the actual binary `torch._C`, no Jittor shim installation marker. Jittor side checked the independent Torch frontend is activated, `flags.use_cuda == 0`.

Both runtimes use the same installed DataTrove; all **126 Python source files** are identical to the fixed upstream commit's source on comparison. Environment is in the job's independent directory; when sharing existing dependency files, the old editable `.pth` is not executed.

## Fixed input and measured results

| Scenario | Input spec and assertions | Jittor | Independent PyTorch |
| --- | --- | --- | --- |
| `DatatroveFileDataset` | uint16 `[10,11,2,20,21,30,2,40,41,42]`, seq_len=4, EOS=2; the two windows' tokens and positions exactly equal, outputs all int64 CPU Tensor | passed | passed |
| JSONL reader/filter/writer | 5 documents, keep even IDs; output exactly `0,2,4` | passed | passed |
| LocalPipelineExecutor | 8 documents, tasks=4, workers=2, forkserver; output `0,2,4,6`, total=8, forwarded=4, dropped=4, 4 completion files | passed | passed |
| Total | same test file and fixed expected values | **3/3** | **3/3** |

Both `input_ids` are `[[10,11,2,20,21],[30,2,40,41,42]]`, positions `[[0,1,2,0,1],[0,1,0,1,2]]`, element-wise exactly equal. The JSONL path itself does not call torch, so it also verifies the token dataset's real Tensor output. These are data-processing scenarios; model parameters, gradients and training levels are not used to replace their acceptance assertions.

Test file SHA256: `969f0cd671035f9b0891e9d40a4a359c403759b29f36d670d4a9dff6b794498a`. Jittor one full run 328.13 s, including fresh CPU core and op compilation; independent PyTorch one run 4.11 s. These are functional-test wall times, not throughput or steady-state performance comparisons.

## Gates and reproduction

This round's repository layout, Markdown link hygiene and release manifest checks passed. Subsequent job `119936`'s missing-dependency negative gate **9/9 passed**, related structure checks **43/43 passed**.

Full structure first round was 1513 passed, 6 skipped, 1 failed; the only failure is the verification environment lacking ASV. After completing the repository-pinned `asv==0.6.6`, the above 43 re-checks include all ASV entry, gate scope, private path and documentation structure checks. The first-round failure record is retained, not rewritten as a full pass.

Test entry is [`test_datatrove_compat.py`](../../compat/tests/torch/test_datatrove_compat.py), dual-interpreter commands in [`datatrove-torch-compat`](../../agent/skills/datatrove-torch-compat/SKILL.md). Raw logs, environment identity, installed source verification and Slurm results are in `$JITTOR_LAB_ROOT/_state/datatrove/publish-20261024/`, unversioned.

| Unversioned artifact | SHA256 |
| --- | --- |
| `results/119913/04-oracle.log` | `a2ea7f9e9409379061c7281a4018c0664402fb3c5e04535121d46ff39b1619e9` |
| `results/119913/05-jittor.log` | `803a51696952f217f9f8c95072f287eaef84274925443387343092b873206eb2` |
| `results/119936/report.json` | `e33a9c88226564f2a6277b4027821cead0c8957206a78a01d93a7b80f44822ff` |
