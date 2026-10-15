# Diffusers: official installation and maintained component replay

- Status: bounded CPU/CUDA L3 measurements accepted by the current shared auditor; broader library and hosted-nightly acceptance remain open.
- Reviewed: 2026-10-16.
- Measurement baseline: `b2127fd6a4a0c080ea19c45fbeae4b0b3251b74f`.
- Implementation remote base: `00ceb26fd395707fb33649cc43709af045110a9a`.
- Owner: Torch compatibility and downstream verification maintainers.
- Review when: official source/wheel pins, source/receipt guards, builder, precision/device controls, requested gradients or training/comparison protocol change.

The current strict auditor actually rejects both original `1a92aae04f4c70fa44e251f8fafdc7e2cd77beda` CPU/CUDA records. Their 697 package-file hashes match the retained local source-built wheel, but the workers did not capture its wheel SHA or actual METADATA/RECORD. This batch preserves those numerical observations and JSON bytes; it obtains new evidence by real execution.

The new profile uses official PyPI Diffusers **0.35.1**, wheel SHA256 `fe29ff10200970c7c5934c6488c213e2a77a03dad5e6fa00bbd8e1d04234cb0e`, and clean official source `0f252be0ed42006c125ef4429156cb13ae6c1d60`. This differs from the historical local source-built artifact `fe763b8bb4381794f64c7a33ffdd29b3e63266c0013194317521331b3a15bdeb`.

All 697 official package payload files match tracked Git contents. The only source omissions permitted are six README pages under `experimental/`, `models/`, `pipelines/`, `pipelines/deprecated/`, `pipelines/stable_diffusion/` and `schedulers/`. No generated or unbound Python/resource payload is allowed. Each actual worker binds its loaded package, complete payload, original METADATA and **actual pip RECORD**, including package hashes and sizes. No archive RECORD is restored.

Both devices execute the original tiny FP32 `UNet2DModel`: sample `[2,3,16,16]`, timestep `[2]`, block channels `(16,32)`, one layer per block, group count eight and dropout zero. The complete 144 parameter gradients and floating sample gradient are required. Base seed 0 and input perturbation seed 7 use the same complete initial weights; three SGD updates use learning rate .001. Per device, six distinct real workers produce 146 base arrays, 146 perturbation arrays and **873 training fields**, totaling **1,165 pairs**. The builder, runner, training math and existing `.002` absolute / `.01` globally scaled comparison budgets are unchanged.

| Device / cards | Oracle | Native CUDA / ACL / replay | TF32 / HF32 | Pairs / original agreement | Largest absolute / scaled error | Time / ratio / process memory | Level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU / 0 | genuine Torch 2.11.0+cpu | 0 / 0 / default, actual replay=1 | off / off | 1,165 pass, .002 / .01 | 4.95911e-5 / 9.40814e-7 | not benchmarked / not sampled | L3 |
| CUDA / 1 | genuine Torch 2.11.0+cu128 | 1 / 0 / default, actual replay=1 | off / off | 1,165 pass, .002 / .01 | 3.81470e-5 / 9.80451e-7 | not benchmarked / not sampled | L3 |

Actual platform: Linux 6.1.188/glibc 2.34, AMD EPYC 9554, GCC 11.5.0; CUDA uses one RTX 4090, driver 580.178.04 and NVCC 12.2.140. Candidate providers are separately installed CUDA 12.2.140, cuDNN 8.9.7.29 and cuBLAS 12.2.5.6; oracle Torch uses its full normal CUDA 12.8 dependency resolution, including cuDNN 9.19.0.56 and cuBLAS 12.8.4.1. Full provider versions and the actual physical UUID/visible mapping are retained in each worker. CPU explicitly hides CUDA; unavailable accelerator fields are `not-applicable`. Both sides use one thread, TF32/benchmark off, highest matmul precision, fixed offline inputs and separate runtime paths. Candidate observations require native dispatch/residency, and CUDA pointers must belong to logical device zero on the allocated physical card. Every candidate reports `fallback_policy=error,count=0`.

Independent NumPy replay checks all **2,330 pairs**, **2,900 nonzero gradient arrays** across the two runtimes/devices and **1,728 SGD updates**. Largest update reconstruction error is 3.72529e-9 on CPU and zero on CUDA. All twelve workers consume the same initial-weight archive SHA256 `b303c894e752f54c296757bd0b56fc5c2cc7e29848923ad267873a918b93c7a0`. Each device has 2,618 actual tensor observations, including initialization and training. These are training/correctness results, without performance or whole-process memory acceptance.

The genuine oracle `_C` binaries are hashed in the actual workers and match their official artifacts. Independent audits additionally verify every original official wheel file except RECORD across the complete installed prefix: **11,945** files for each CPU Torch installation and the CUDA candidate's metadata-only CPU Torch distribution, **11,841** for the CUDA oracle, and **702** for each Diffusers installation, including its 697 package files and five distribution files. Every original payload row matches the untouched actual pip RECORD. Installation-generated scripts/metadata/bytecode are not called original archive bytes. Post-execution rehashes remain identical. The candidate runs the shim/native framework; its installed genuine Torch distribution supplies dependency metadata and is not its numerical provider. Candidate identity uses the actual native import path, source and dispatch/device observations; **no candidate JIT binary hash was captured inside these workers**.

The complete measured Git archive has **2,915 tracked files**, with modes and symlink target text. All **1,612 selected worker source hashes** match the measured committed bytes; that selected map is not an exhaustive import trace. Both full archive and Git bundle retain the original measurement identity. Neither results nor source hashes are relabelled to later integration commits.

Fresh clean upstream enumeration is unchanged: `model_api=87`, pipeline-family axis `model_family=79`, `pipeline_class=249`, `auto_family=37`, `task=3`. Current common admission is **two accepted / zero rejected / zero blocked**. Each device counts only `UNet2DModel`, **1/87 on model_api**; all other axes remain zero. The export denominator includes Flax, helpers, mixins, enums and functions, rather than 87 model families. No complete generation pipeline follows from this UNet component.

Reproduce from clean committed source with complete separately installed runtimes and the pinned official checkout/wheel:

```bash
python tools/run_diffusers_gate.py --candidate-python <private-candidate-python> \
  --oracle-python <private-binary-torch-python> --device cpu \
  --checkout <clean-official-diffusers-checkout> --wheel <fixed-official-wheel> \
  --output "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/cpu" \
  --cache-root "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/benchmark-cpu"
```

Run CPU before CUDA. For CUDA allocate one healthy card explicitly, select each private runtime's actual provider libraries, set `CUDA_VISIBLE_DEVICES=<allocated-card>` and a real `JT_BUILD_NVCC_PATH`, then rerun the same command with `--device cuda` and fresh output/cache roots. The measured invocation's executable wrappers only select the actual per-prefix shared libraries; process metadata still records the underlying real interpreter and providers. Initial JIT is serialized; unit and benchmark caches are distinct. The gate exposes no reduced-case/gradient/step option. It reuses the maintained collector and actual common auditor, and emits records only after source-clean checks and complete admission.

Nine forged copies of real packets are rejected: missing official wheel SHA, changed distribution version, changed METADATA/RECORD digest, unbound non-Python declaration, changed committed source hash, missing HOME, missing parameter gradient and missing third-step parameter. Original packets/installations remain read-only. The first negative-audit attempt incorrectly expected within-process HOME/TMPDIR inequality from the single-packet validator; that diagnostic failure is retained, and the corrected test checks the required HOME field. No guard was weakened. Host regressions also cover coherent wheel/resource/metadata tampering and default callback behavior. The existing Metrics/TensorDict `912` four rows still admit under the default validator; the original MM `7bb` 231 pairs, initial weights, 84 updates and 84 momentum states replay unchanged, including its `mmcv-lite` naming/receipt contract.

Normal complete pip resolution and pip checks pass in fresh candidate/oracle/controller environments. An earlier online CPU setup was explicitly terminated during dependency acquisition, before numerical execution; its partial prefix and logs remain. A first offline resolution lacked `importlib_metadata`; official dependency wheels were verified and complete offline resolution succeeded. No `--no-deps`, installed-site copy or JIT-cache copy was used. CPU core passes **322 / 42 skipped / one existing xfail**. Focused host checks and final structure results are recorded in the external delivery manifest; central toctree/generated-manifest ownership is separate from the measured source.

Unversioned evidence is under `$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/20261016/`: `formal-{cpu,cuda}-v1/`, `formal-common-records-v1.json`, `formal-common-scorecard-v1.json`, independent array/installation audits, source archive/bundle and preserved attempts. Common scorecard SHA256: `a779a561e74ac1e8dd366cea4720b44807b5232ea61663192f77c3bf32763aa0`. Measurement seal `measurement-evidence-seal-v1.json` binds **188 files / 179,082,533 bytes**, SHA256 `d26745297c9c5c141ca2d6462b0ab2c86bd6bd559700326d555eeccf9e320f18`. Future validation/delivery logs have their own manifest; this measurement seal is not rewritten.

Dedicated CPU nightly wiring is a separate follow-up. This batch proves no hosted schedule execution, complete pipeline, other dtype, distributed training, NPU/ROCm, L4 speed, whole-process memory or L5 grade.
