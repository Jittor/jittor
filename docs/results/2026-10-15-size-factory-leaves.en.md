# Leaf-tensor identity of size factories and scalar normal

- Status: the following factory contracts, scoped to CPU / a real single CUDA device, pass; no speed claims or claims for untested backends.
- Date: 2026-10-15.
- Baseline: synchronized remote `14aeca5df0bf841e99d6241e06b076e34ef08c05`; after the Tensor random-fill integration it is `74e39acd1fd4b211c34cc5460e623f4499760f6e`.
- Code: `d51c2a48cc4ea2367e41d0dca84e2051795eefb6`. At actual verification time the source had not yet been committed; an external source freeze records the production and test SHA256; after the commit each file was checked to be equal, without relabeling them as clean-source measurements.
- Owner: Torch factory / autograd maintainers.
- Review when: factory dtype/device wrapping, leaf registration, native detach/backward, or the Generator sampler owner changes.

## Real problem and fix

The standalone binary Torch `2.11.0+cu128` returned `is_leaf=True` and `grad_fn=None` for all 32 combinations of 11 factories, FP32/FP64, and the applicable default/explicit Generator, and could obtain a complete `.grad` after backward. The candidate baseline had nine nonconforming combinations: FP32 `zeros` / `ones` / `full`, `linspace` in both dtypes, and the scalar `normal` of both dtypes on both Generator paths. They exposed the internal broadcast/cast/affine graph, with `.grad` being None.

Earlier, the short probe of the Tensor random-fill integration used `torch.ones(..., requires_grad=True)` as the write target without first checking `is_leaf`. That target is in fact not a leaf tensor on the candidate side, so its write result cannot prove that the leaf check fails. Additional observations confirm that a real leaf tensor created explicitly with `torch.tensor` is correctly rejected by the random-fill fix; the baseline evidence that a private Generator was not advanced remains valid. Both the original probe and the additional observations are retained.

The factory wrappers, after the final dtype conversion and before setting requires-grad and registering the backward holder, use the existing native `detach` to cut the internal history of the creation process. The explicit Generator path likewise only processes the sampled output and adds no random numbers, does not modify the PRNG, and does not implement another set of math. The scalar `normal` decision is recorded before the old shape normalization; the Tensor mean/std overload does not enter the newly added scalar handling branch.

`linspace` has one more external dtype wrapper: previously the cast happened after the leaf holder was registered, so FP64 formed a new graph node, and a same-dtype cast could also separate the returned holder from the registered holder. Now the original constructor performs the dtype conversion before detach/registration, and the endpoint and step conversions keep their original owner.

## Measured scope

The maintenance node is `compat/tests/torch/test_size_factory_leaf_contract.py`. The same source file executes under standalone Torch and the candidate; cross-framework samples with different PRNGs are not required to be bitwise identical.

- `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `rand`, `randn`, scalar `normal`, and the float-output `randint` / `randperm`: FP32/FP64, the explicit requires-grad switch, and plain and `no_grad` creation.
- Enabling gradients afterwards on default no-grad outputs; checking the true leaf identity, absence of grad-fn, independent storage, complete gradients, and accumulation over two backwards.
- Five explicit Generator factories: full state replay, private stream advance, and global CPU/CUDA stream isolation.
- FP16/BF16/FP32/FP64 true leaf targets for `zeros` / `ones` / `full`: `uniform_` / `normal_` reject before sampling, and the target values and the private/global states are unchanged array by array.

All new candidate nodes execute under a forbid-fallback scope, checking the requested device against the real location; CUDA uses the driver pointer attribute to check the allocation ordinal before host readback. The standalone oracle rejects the shim and Jittor imports before and after, and checks the real `_C` extension and the test file SHAs.

| Verification | Result | Notes |
| --- | --- | --- |
| Standalone Torch short matrix | 32/32 leaves and gradients pass | 9 combinations nonconforming before the candidate fix, all passing after |
| Final candidate CPU | 161 passed / 152 CUDA skipped | The new file plus the existing like / tensor / factory fidelity four files |
| Final candidate CUDA | 318 passed / 0 skipped | Same four files, requiring real CUDA and at least 152 accelerator nodes |
| Standalone Torch CPU | 222 passed / 222 CUDA skipped / 8 native-only deselected | New factory file plus nn.init / Tensor random-fill two files |
| Standalone Torch CUDA | 444 passed / 0 skipped / 8 native-only deselected | Same source and same set as CPU |
| Existing random/in-place eight files after the factory fix | CPU 147 passed / 105 CUDA skipped; CUDA 274 passed / 1 both-device skipped | before-review source freeze; the subsequent review only moved the overload decision earlier, and the final four files and core were re-verified again |
| Final CPU core | 322 passed / 42 skipped / 1 existing xfail | Both native and Torch modes exit 0 |

The final four files took 5.21 / 12.54 seconds on CPU / CUDA respectively. Compilation caches, other processes, and specs differ, so the durations are reproduction information only and do not constitute L4. CPU explicitly hides CUDA; CUDA allocates one RTX 4090, driver 580.178.04. The candidate is GCC 11.5.0 / NVCC 12.2.140 with the cu122 provider; the oracle is the actual cu128 Torch. The outputs and gradients of all new tests execute on the declared device.

The two production stages were source-frozen separately. The first Tensor integration was verified on clean `74e39acd`; the final factory stage recorded dirty-source SHAs for three production files and four test/tool files, and the subsequent Git blobs are byte-for-byte equal to them. After independent review prompted recording the overload type earlier, the CPU four files, CUDA four files, and core were all re-verified against the new production SHA, and the earlier logs are not rebound.

## Retained failures and boundaries

All 274 assertions of the first Tensor CUDA integration passed, but the gate wrongly required at least 200 accelerator cases while 122 were actually selected, so the process exited 1. The original script and log were retained, the gate was set to the actual set of 122, and the rerun as-is exited 0; no cases were deleted, no tolerances were changed, and no failed run was recorded as passing. The new factory gate for the four files independently requires its actual 152 CUDA nodes.

The first full structure check was 1621 passed / 6 skipped / 1 failed: the old `test_factory_install_owners` fake object required the constructor to return and then be cast. It was updated to verify that the dtype is passed into the constructor and the same registered holder is returned as-is; the explicit Generator branch still must not touch the default stream. The original failure log is retained, and the test expectation is not reverted to the production path that would lose gradients.

Support is not granted here for the old positional-argument normalization and own autograd contract of Tensor mean/std `normal`; `bernoulli` / `multinomial`, general Tensor transforms, all out/layout options, and untested backends are also not covered by this report. What is fixed is the Torch leaf semantics of creation results; it does not claim that random bytes interoperate cross-framework with Torch or that the Generator ABI is complete.

The test sources are kept in the main-tree maintenance node and as out-of-repo byte copies. The oracle uses the private provider of a separate complete official Torch wheel and neither patched nor rewrote the previously file-incomplete shared Torch installation; this round's entry point verifies the actual binary identity and contracts and does not backfill the full installation provenance of historical oracles.

The unversioned raw records are in `$JITTOR_LAB_ROOT/_state/tensor-main-integration-14a/`: `factory-matrix-{before,oracle,final}.log`, `factory-final-source-freeze.json`, `factory-final-{cpu,cuda}.log`, `oracle-{cpu,cuda}-{before,after}.json`, `oracle-{cpu,cuda}.log`, `core-final.log`, together with the separately retained Tensor first-batch source freeze, initial configuration failures, and correct-gate logs.

Before reproducing, pin the interpreter, device, compiler, and an independent unit cache; run CPU before CUDA and share no cache with benchmarks:

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_size_factory_leaf_contract.py \
  compat/tests/torch/test_like_factory_leaf_contract.py \
  compat/tests/torch/test_tensor_factory_leaf.py \
  compat/tests/torch/test_torch_factory_fidelity.py
```

Real CUDA starts another process for the same four files and sets an independent unit cache:

```bash
CUDA_VISIBLE_DEVICES=<allocated-gpu> JT_USE_CUDA=1 JT_BUILD_NVCC_PATH=<nvcc> \
  JITTOR_TEST_REQUIRE_CUDA=1 JITTOR_TEST_REQUIRE_EXECUTION=1 \
  JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=152 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  <candidate-python> -m pytest -q \
  compat/tests/torch/test_size_factory_leaf_contract.py \
  compat/tests/torch/test_like_factory_leaf_contract.py \
  compat/tests/torch/test_tensor_factory_leaf.py \
  compat/tests/torch/test_torch_factory_fidelity.py
```

Standalone Torch uses the same test bytes and `_rng_init_support.py`, placed in an out-of-repo directory with the shim/PYTHONPATH cleared, asserting the real binary and no Jittor import before and after the run. `--confcutdir=<copied-suite>` avoids loading candidate-specific pytest policies; only the native-only observer is excluded, and the CPU CUDA branch skips honestly.

The final installation ownership targeted check is **5 passed**; the updated complete `tests/structure + compat/tests/structure` is **1622 passed / 6 skipped**, 230.84 seconds. The six are two accelerator, two declared, and two resolution checks for the native distribution that the current structure-verification interpreter has not installed; these skips are not granted as installation acceptance. Layout, MANIFEST, and whitespace checks pass, the new tests have no Ruff diagnostics; the modified factory files keep the same three original diagnostics as the baseline.


## KI-COMPAT-004 factory sub-item wrap-up

On the synchronization baseline `d9b2ae9c0b9f0a92102a619940273b3efd645be1`, the existing
`test_leaf_and_intermediate_report_torch_shape` still asserted that factory results are
non-leaves, and the minimal re-verification indeed failed.
The test now asserts the correct leaf identity, absence of grad-fn, and the complete
gradient `[2, 2, 2]`, retains the producer identity check on the intermediate result, and
selects the device explicitly according to the running backend.

Measured after modifying only that test: the CPU autograd semantics full file and the
standalone Tensor constructor node are **23 passed / 0 skipped** (10.35 seconds); real CUDA
autograd semantics full file **22 passed / 0 skipped** (23.85 seconds); native-mode JVP/VJP
seed full file **6 passed / 0 skipped** (7.25 seconds), where the multi-input JVP is still 348.
These logs correspond to `d9b2ae9c0` plus the test change and were not relabeled as
measurements of the earlier factory production fix.

At that historical stage KI-COMPAT-004 still had the lost graph connection of
`torch.Tensor(other)`; that stage neither fixed the constructor nor turned its existing
defect assertion into a passing declaration. The later fix and the remaining boundaries are
in the next section. The unversioned raw logs are in
`$JITTOR_LAB_ROOT/_state/factory-ledger-followup-d9/`, including the pre-fix failure, the
two-device re-verification, and the native JVP/VJP.


## Tensor source constructor follow-up

This follow-up synchronized published `0fb3920d4a44f37fa6cea7fa856374f0bacf73b4`.
The constructor previously overwrote its native clone's requires-grad state with
false. `Tensor(source)` now inherits the source state through the native view
owner, preserving shared storage and the source graph when grad mode is enabled.
The independent-frontend regression expects this connected Torch answer.

Native autograd separately represents a no-grad-created view's inherited public
requires-grad flag and its lack of a backward edge/gradient accumulator. Operator
input snapshots preserve this distinction when a later explicit
`requires_grad_(True)` activates the alias: earlier graphs remain disconnected,
while newly constructed graphs accumulate its gradient. Clone, detach, nested
alias, retain-grad and invalid backward-root cases use the same native semantics;
no compatibility registry suppression or fabricated gradients is used.

The new file has 41 contracts per device: 32 combinations of float32/float64,
trainable/frozen sources, leaves/intermediates, contiguous/transposed storage and
grad/no-grad creation, plus seven lifecycle variants, activation timing and invalid
backward roots. Checks include exact values/derivatives, strides/storage sharing,
physical CUDA residency and CUDA driver pointer-device identity. Fallbacks are
forbidden and pending execution is synchronized.

| Verification | Actual result |
| --- | --- |
| Expanded five-file Torch-mode CPU regression | 235 passed / 193 CUDA skips, 9.29 s |
| Expanded five-file CUDA-enabled regression | 428 passed / 0 skipped, 462.82 s including cold compilation |
| Independent official binary Torch, identical 41 contracts | 41 passed on CPU and 41 on CUDA |
| Shared native autograd five files, CPU | 30 passed / 1 CUDA skip, 22.34 s |
| Shared native autograd five files, CUDA-enabled | 31 passed / 0 skipped, 129.40 s |
| Canonical core, native and Torch modes | 322 passed / 42 skipped / 1 existing xfail; both mode exits zero |
| Complete structure and compatibility structure | 1622 passed / 6 skipped, 277.77 s |

CUDA-enabled runs include explicit CPU controls; their totals are not exclusively
CUDA cases. Compilation-inclusive durations are validation evidence, not speed
comparisons. The five Torch files are `test_tensor_constructor_alias.py`,
`test_independent_frontend.py`, `test_torch_compat_autograd_semantics.py`,
`test_size_factory_leaf_contract.py` and `test_tensor_state.py`. The five native
files cover the engine, policy, backward-twice, missing gradients and functional
seeds. Raw commands, source hashes, probes and logs are unversioned under
`$JITTOR_LAB_ROOT/_state/tensor-alias-main-0fb/`.

Historical stages remain distinct. The original grad-enabled-only candidate had
226 CPU passes / 184 skips and 410 CUDA-enabled passes; its oracle covered 32
contracts per device. The first oracle rejected the incorrect assumption that a
no-grad alias should accumulate its own gradient. A subsequent real candidate
probe exposed retroactive connection of old graphs after alias activation; native
operator-input snapshots fixed that failure. These failed attempts and corrected
contracts remain available under their original filenames and source snapshots.

KI-COMPAT-004 remains open. Independent review additionally found Torch's
creation-mode/mutation guard: modifying a no-grad-created alias or its base makes
later grad-mode `is_leaf`, `grad_fn` and arithmetic reject the view. It also found
that enabling a previously frozen source dynamically changes its alias's visible
requires-grad flag without activating an accumulator. Candidate CPU probes confirm all 12 mutation variants accept the forbidden use
and all eight unfreeze variants leave the alias flag false. These reproductions
and remaining differences are recorded separately; the 41 passing contracts above do
not establish either additional behavior. Full subclass/C-extension ABI and
unexecuted consumers are outside this result and R33 remains open.

The six structure skips comprise two accelerator cases, two declared skips and
two installed-distribution resolution checks unavailable in this interpreter.
They are not evidence of installation support. Layout, Ruff and whitespace
checks also pass. The initial mutation probe completed its observations but failed
while recording the shim’s nonexistent `torch._C.__file__`; its log is retained,
and the corrected v2 probe records the actual frontend path without claiming an
independent binary oracle.
