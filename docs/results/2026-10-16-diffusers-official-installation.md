# Diffusers: official installation and maintained component replay

- Status: source/receipt capture and bounded replay gate implemented; fresh numerical validation pending.
- Reviewed: 2026-10-16.
- Owner: Torch compatibility and downstream verification maintainers.
- Implementation remote base: `00ceb26fd395707fb33649cc43709af045110a9a`.
- Review when: official source/wheel pins, source/receipt guards, component builder, precision/device controls, requested gradients or training/comparison protocol change.

This follows an actual strict-admission failure. Under auditor `67e166018cd5ec1c05413d2daf1c7302b2461970`, both original `1a92aae04f4c70fa44e251f8fafdc7e2cd77beda` CPU/CUDA records are rejected. Their 697 package-file hashes match the retained source-built wheel, but their worker bindings did not capture its wheel SHA or actual METADATA/RECORD fields. The old mathematical observations remain historical evidence; this batch does not add installation fields to those JSON files or relabel their source identity.

The new profile uses the actual official PyPI Diffusers **0.35.1** wheel, SHA256 `fe29ff10200970c7c5934c6488c213e2a77a03dad5e6fa00bbd8e1d04234cb0e`, and clean official source `0f252be0ed42006c125ef4429156cb13ae6c1d60`. It is distinct from the historical local source-built artifact `fe763b8bb4381794f64c7a33ffdd29b3e63266c0013194317521331b3a15bdeb`.

The complete official package has 697 payload files, all matching tracked Git contents. Git additionally tracks six README pages omitted by this wheel: `experimental/`, `models/`, `pipelines/`, `pipelines/deprecated/`, `pipelines/stable_diffusion/` and `schedulers/`. Only those six documentation omissions are explicit exceptions. No generated or otherwise unbound Python/resource payload is accepted. Every actual worker binds the selected package root/version, full payload, original METADATA and real pip RECORD package hashes/sizes before the record can be admitted. The pip receipt remains distinct from the archive receipt.

The maintained gate keeps the original `UNet2DModel` component: 144 full parameters, its sample input gradient, seed 0, fixed-weight seed-7 input perturbation and three SGD steps at learning rate .001. Each device requires six independent binary-oracle/candidate processes, 146 base arrays, 146 perturbation arrays and all 873 training fields: 1,165 pairs. It reuses the existing builders, runner, training probe, collector and numerical budgets .002/.01. It never claims a diffusion pipeline or whole library from this model component.

The full actual-process platform capture, tensor residency/pointer observations, private runtime paths, zero-fallback guard and immutable Git source closure are attached during execution. Metrics/TensorDict keep their existing default source validator; the new profile explicitly supplies its own committed-source validator to the existing internal runtime contract. This does not add new fields to the historical Metrics `912` closure.

From a clean committed checkout, with two separately installed complete runtime environments and an actual pinned upstream checkout/wheel:

```bash
python tools/run_diffusers_gate.py --candidate-python <private-candidate-python> \
  --oracle-python <private-binary-torch-python> --device cpu \
  --checkout <clean-official-diffusers-checkout> --wheel <fixed-official-wheel> \
  --output "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/cpu" \
  --cache-root "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/benchmark-cpu"
```

Run CPU first with CUDA explicitly hidden, then use the same frozen source for CUDA after allocating and verifying the actual device and private CUDA providers. Unit and benchmark caches remain separate; initial compilation is serialized. Both runtimes must use the same initial weights and actual precision/thread conditions. A failed worker, missing gradient, changed source/payload, reused output, incomplete protocol or rejected common scorecard prevents the success marker.

Fresh measurement results, fixed platform/specification tables, current admission, negative controls and exact external evidence hashes will be recorded after actual execution terminates. This implementation commit itself grants no new L grade, CUDA coverage, L4 speed, L5 or hosted-nightly execution claim. Nightly/nox/provisioner wiring is outside this batch.
