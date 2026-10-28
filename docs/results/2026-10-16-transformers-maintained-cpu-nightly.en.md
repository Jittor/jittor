# Transformers: fresh maintained CPU nightly validation

- Status: complete local execution of the maintained CPU nightly profile; independently audited.
- Reviewed: 2026-10-16.
- Owner: Torch compatibility and downstream verification maintainers.
- Measurement source: `2f8895f1a656bcdd66c280dbdbe222eb8268f56e`.
- Environment setup source: `13b05c8b1951622ab12511e2936243086d5408ba`.
- Independent audit checkout: `bc4dc0bc01a3930d3fad22326da0f9b8c70d444f`.
- Review when: dependency pins, provisioning, case scope, source/installation validation, precision, numerical budgets or training builders change.

This is a new CPU execution of the complete `ecosystem_transformers_source` session. The original nox process exits zero after all ten cases, four phases and 80 workers finish. Its controller verifies the measured checkout before and after execution. An independent read-only audit repeats every original numerical assertion, complete requested gradient set, initial-weight binding and three-step trajectory. It changes neither the measurement tree nor the original arrays and does not relabel them to a later source revision.

## Actual scope and results

Every case runs seed 0, fixed-weight input perturbations at seeds 7 and 19, and three SGD steps at learning rate .001 with input seeds 1/2/3. Complete reference parameters and buffers are supplied to both runtimes. Base and training seed-0 initial arrays also agree exactly. The original TF phase-global budgets remain absolute .002 or scaled-relative .01; Longformer retains its original per-field/gradient-block normalization, forward .002 and backward .01.

| Maintained case | Parameters / trainable | Actual input scope | Array pairs | Largest scaled error |
| --- | --- | --- | --- | --- |
| BERT | 39 / 39 | token IDs `(2,8)` | 360 | 3.06e-7 |
| CLIP multimodal | 78 / 78 | token IDs `(2,8)` and pixels `(2,3,16,16)` | 717 | 7.27e-7 |
| GPT-2 | 28 / 28 | token IDs `(2,8)` | 261 | 3.20e-7 |
| Llama | 21 / 21 | token IDs `(2,8)` | 198 | 5.63e-7 |
| Llama RoPE | 12 / 12 | token IDs `(1,2048)` | 117 | 1.80e-6 |
| Mixtral MoE | 41 / 41 | token IDs `(2,32)` | 378 | 2.56e-7 |
| T5 | 47 / 47 | encoder/decoder token IDs `(2,8)` | 432 | 7.13e-7 |
| ViT | 40 / 40 | pixels `(2,3,32,32)` | 375 | 5.17e-7 |
| Whisper | 89 / 88 | features `(2,16,64)` and decoder IDs `(2,8)` | 813 | 3.83e-7 |
| Longformer | 29 / 29 | IDs `(1,32768)` and embedding noise `(1,32768,32)` | 276 | 9.85e-6 |

The independent NumPy calculation and frozen maintained comparator agree field for field on all **3,927 array pairs**. These include 2,541 parameter-gradient and 24 input-gradient pairs; all requested gradients exist, are finite and are nonzero in both runtimes. Shapes, dtypes and complete array-name sets agree. Both perturbations change outputs in each runtime. All 2,538 trainable parameter updates are checked against each runtime's saved SGD gradients and previous values; the largest update-replay discrepancy is 1.49e-8. Whisper's one frozen parameter remains bitwise unchanged in all six saved training snapshots.

The largest absolute cross-runtime discrepancy is .0166015625 in Longformer. Its original normalized rule passes; applying a different absolute-only .002 rule would misdescribe that contract. All phase-specific errors are retained in the independent audit. These ten CPU scenarios cover nine selected declared families out of the automatically re-derived **374** `MODEL_MAPPING_NAMES` families at the pinned source. Two Llama scenarios count as one family. This is bounded L3 evidence, without a full-library claim or a new paired CPU/CUDA scorecard.

## Actual platform and source authority

| Field | Recorded CPU execution |
| --- | --- |
| OS / CPU | Linux 6.1.188, glibc 2.34; AMD EPYC 9554 |
| Accelerator / card count | CPU tensor residency; zero allocated cards |
| Driver / CUDA / nvcc / CANN | Not applicable for this CPU execution |
| Compiler / precision | g++ 11.5.0; FP32; TF32 and cuDNN benchmark off; highest |
| Threads / graph | Runtime and OMP/MKL/OpenBLAS threads 2; candidate default graph replay |
| Independent oracle | Genuine binary Torch 2.11.0+cpu in a separate private prefix |
| Runtime boundaries | Candidate fallback policy error, observed count zero; separate runtime HOME/JIT/temp/cache paths |
| Speed / whole-process memory | Not measured for L4 |

All 80 actual worker PIDs are distinct. Their 36,810 tensor observations are CPU-resident, with the original input/output/parameter/gradient shapes and FP32 computation. The controller records empty `CUDA_VISIBLE_DEVICES`, and the Git-bound CPU launcher explicitly sets empty visibility and both CUDA flags to zero. Individual worker metadata does not separately record `cuda.is_available()`; that readback is not inferred or backfilled. The oracle provisioning health check separately asserts CUDA unavailable.

The controller's complete 2,905 tracked Git blob hashes, including symlink target text, agree with the measured commit and unchanged checkout. Every worker's 1,575 selected framework/probe source hashes agree; the selected map is not an exhaustive Python import inventory. A complete historical Git archive is retained separately.

The clean official Transformers 4.56.2 checkout is `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`. Its fixed official wheel SHA256 is `79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`. The actual imported package's complete 2,173 files agree with the wheel and tracked source in both prefixes. Only the two already-declared non-runtime omissions, GPT-2 `CONVERSION.md` and T5 `download_from_gcp.sh`, are excluded from the wheel/source mapping.

The existing TF profile deliberately restores the pinned purelib wheel's METADATA/RECORD bytes after real full dependency installation. Both actual TF RECORD files equal the archive receipt; they are **not untouched pip-generated RECORD files**. Separately, the official Torch CPU wheel SHA256 `8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` is rechecked against all 11,945 original installed archive files except self-unhashed RECORD: 700,036,604 bytes, including `torch/`, `functorch/`, `torchgen/` and dist-info payload. Torch's actual pip RECORD hashes and sizes agree with every original payload file. All three private environments independently pass `pip check`; retained locks and actual distribution METADATA hashes describe other dependencies without claiming complete wheel-payload revalidation of each one.

## Replay and evidence

From a clean checkout, use a healthy Python 3.11 and a new external state directory:

```bash
python tools/provision_ecosystem_nightly.py --library transformers \
  --python <healthy-python-3.11> \
  --state-root "$JITTOR_LAB_ROOT/_state/transformers-nightly/<new-run>" --run
```

This provisions separate controller/candidate/oracle environments, resolves full dependencies, checks them, and runs the complete maintained session. Read-only source/wheel caches are optional inputs; JIT caches must remain private and are never copied. The session permits output/cache paths only, so the ten cases or four phases cannot be selected away. Existing [source-bound training evidence](2026-10-15-transformers-installed-source-training.en.md) retains its own historical identities.

Original unversioned evidence is under `$JITTOR_LAB_ROOT/_state/transformers-nightly/20261016/`, including `run-formal-v1.py`, `formal-controller-v1.json`, `formal-terminal-v1.json` and `fresh-v2/logs/027.log`. Independent outputs are under `$JITTOR_LAB_ROOT/_state/transformers-nightly-independent/20261016/`: `full-v2/numeric-audit.json`, per-field comparisons, source archive, full Torch installation audit, platform/installation summary and `evidence-seal.json`. The audit's first attempt incorrectly treated the existing backend dictionary as a string; that auditor failure is retained, and its corrected v2 passes without changing original data or budgets.

This execution proves the maintained CPU profile locally. It does not prove that GitHub's hosted schedule executed, add new CUDA measurements, upgrade historical CUDA records, or establish L4/L5, full-model-family or unseen-backend coverage.

The independent evidence seal, rehashed again during integration, binds 572 files
and 919,783,355 bytes, SHA256
`bcc385d75c48872244df28ac5f7f0008bdbfaea1d8bfc85c792016ad19606112`.
