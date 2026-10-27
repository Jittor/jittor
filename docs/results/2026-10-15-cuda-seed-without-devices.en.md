# Seeding APIs without visible CUDA devices

- Status: the bounded CPU and actual CUDA seed contracts are verified; full downstream training support is not granted.
- Date: 2026-10-15.
- Baseline: modifications were frozen after syncing `7a1e66b3858279b40eaec0c4895c6059ce8eb8db`; the actual measurement used the same frozen files, retaining the then-dirty state, and commit `4a941dbe28fd93895c40590d4048094b333e9946` maps these two files. Integration baseline `6f8e160a4cc7807f89e2d24b4c41349405df7720`, integration commit `97d35128f`.
- Owner: Torch CUDA frontend maintainers.
- Review when: CUDA device discovery, lazy initialization, seed conversion or the Transformers/Swift initialization paths change.

A real Swift Trainer's CPU initialization calls `torch.cuda.manual_seed_all` through Transformers.
The original implementation initialized cuRAND unconditionally, so CPU-only environments failed. A
20-line API probe against independent binary Torch 2.11 confirmed that with no visible device the
integer conversion still runs first, but CUDA is not initialized and the CPU RNG is not changed;
a convertible integer beyond 64 bits does not trigger a device seed range error in that state.
The original candidate matched only 4/20 lines.

The fix converts the input first, then queries the visible device count and returns immediately when
there are no devices. With real CUDA it continues with the original full-width range validation,
synchronization and native cuRAND restoration. No random kernel or cross-framework aligned sample
stream was changed.

## Original actual verification

| Check | Result | Scope |
| --- | --- | --- |
| Independent Torch / candidate API probe on CPU | 20/20 identical after the fix | Argument conversion, no device, no initialization and CPU state protection |
| CPU unit tests | 20 passed / 16 skipped | Hardware nodes do not count as executed |
| Targeted actual CUDA regression | 36 passed / 0 skipped | New contracts, seeding wraparound and existing state restoration |
| Independent Torch CUDA | 8 lines passed | Full bit width, negative seeds, reseeding and CPU stream isolation |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | Native and Torch modes |
| Complete structure | 1508 passed / 6 skipped | The original agent's source baseline; not relabelled as an integration measurement |

Actual CUDA used the allocated real GPUs; fallback was refused, real device samples were read and the
complete arrays after reseeding were compared item by item. An initial oracle configuration that
mistakenly exposed the cu122 libraries caused an undefined-symbol installation error, and the original
log is retained; the formal oracle uses a separate cu128 environment. Both the exit of the unit cache
relocation and the normal restart are retained separately.

## Evidence and boundaries

Unversioned evidence is under `$JITTOR_LAB_ROOT/_state/downstream-training-evidence/20261015/`.
`seed-api-sealed-manifest-v1.json` binds 31 raw files, both source hashes and the original run identity;
its SHA256 is `14630026c81b40a80cacfde7f0e680a3a575bf8e69ab4df2af73083507910b0e`.
The original failures do not count as final passes, and old results are not relabelled with a new
version number or new dependencies.

The integrated tools, CPU/CUDA targeted, core and structure regressions are stored separately in
`$JITTOR_LAB_ROOT/_state/native-runner-seed-integration-6f8/integration-manifest.json`, recorded with
the actual frozen files and commit mapping while preserving the historical measurement identities above.
This item does not mean that complete Swift training, CPU jtrun, MPI or real multi-host has been accepted.
