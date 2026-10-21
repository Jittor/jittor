# Custom pytree nodes and Transformers 5.13.0

- Status: CPU/CUDA correctness verified for the bounded contracts below.
- Date: 2026-10-15
- Baseline commit: `c92ae5cecca90801804e677b7200b2ef3b30b27c` (clean source for the final GPT2 replay).
- Owner: Torch compatibility maintainers.
- Review when: pytree installation, custom-node traversal, adapter version guards,
  the shared ecosystem/training runner, or Transformers output structures change.

## Failure and ownership

The genuine Transformers 5.13.0 distribution initially stopped at the adapter's
version guard. With provisional admission, importing its `ModelOutput` then
failed because `torch.utils._pytree._dict_flatten_with_keys` was absent. A separate
custom `dict` subclass also showed that the existing `register_pytree_node`
returned without registering anything: flatten/unflatten reconstructed `dict`
instead of the registered class. Independent PyTorch 2.13.0 reconstructed the
custom class and supplied actual mapping/sequence key entries.

This is a shared Torch API contract. The fix belongs to
`compat/torch/installers/utilities.py`: store registered callbacks; use them for
flatten, unflatten, mapping and keyed traversal; preserve registered subclasses;
reject duplicate registration, missing keyed callbacks and malformed leaf counts.
Built-in `OrderedDict` and `defaultdict` keep their type and default factory.
Both public and legacy registration use the same registry. TreeSpec JSON
serialization is outside this verified scope.

The adapter now admits the exact version **5.13.0**, alongside previously admitted
4.56.2 and 5.5.3. The host contracts still reject an unknown version; no broad 5.x
range is enabled. There is no ModelOutput-specific patch.

## Verification

The oracle is genuine **PyTorch 2.13.0+cu130** in a separate interpreter. Candidate
execution uses this checkout and the Jittor Torch frontend, whose reported Torch
API version remains **2.11.0**. Neither framework version was rewritten. Both
sides load the same genuine Transformers **5.13.0** files. Python is 3.11;
NumPy is 1.26.4. Candidate CUDA uses the existing CUDA 12.2 toolchain; the oracle
uses its genuine CUDA 13.0 wheel libraries. CPU runs explicitly hide CUDA and set
Jittor's backend/use_cuda to CPU/zero.

| Check | Result |
| --- | --- |
| New custom-pytree tests, genuine PyTorch CPU / CUDA | 9 / 9 passed |
| Candidate CPU custom-pytree plus existing tree-map/owner neighbors | 17 passed |
| Candidate CUDA custom-pytree tests, including tensor input gradients | 9 passed, no skips |
| Adapter host version/probe contracts | 7 passed |
| CPU core suite | 322 passed, 42 skipped, 1 expected failure |
| Full `tests/structure` in Torch mode | 1488 passed, 4 skipped, no failures |
| Repository layout, new-test Ruff, no added implementation lint diagnostics, diff whitespace | Passed |

Pytree tests cover custom dict and non-dict nodes, nested key paths and `get`,
registered roundtrip/map, real tensor input gradients, duplicate registration,
missing path callbacks, too few/many leaves, unregistered subclass behavior,
legacy registration, and the two built-in mapping subclasses. The same new test
file runs against the independent PyTorch oracle.

The final GPT2 replay uses the existing tiny GPT2 builder (two layers, width 64,
two heads, vocabulary 128, length 8, batch 2, dropout zero). A small wrapper uses
the public `get_input_embeddings()` entry point plus `0.05 * embedding_noise`, so
there is a floating input gradient as well as the complete 28-parameter gradient
set. It saves outputs, loss, every parameter gradient, input gradient and updated
parameters for each of three SGD steps (`lr=0.001`, seed 7). No parameter is
frozen. Both devices/runtimes load the same initial-state artifact:
`a82b8e3e9c20f57ed6da16920a87042ec0a2626f651a9913412ea44db96ea639`.

Thresholds were frozen before the clean replay: maximum absolute error divided
by `max(max(abs(reference)), 1e-8)`, limit 0.002 for outputs/loss/updated parameters
and 0.01 for parameter/input gradients. CUDA retains these CPU limits. Each
array also requires equal keys, exact shape and finite values. TF32 is off,
matmul precision is `highest`, native CUDA kernel math is `strict`, and runtime
thread counts are one. Candidate fallback policy is `error` and observed
fallback count is zero on both devices.

| Clean-source replay | Arrays compared | Largest normalized error | Result |
| --- | ---: | ---: | --- |
| CPU | 177 | 0.00000346416447 | All passed |
| CUDA | 177 | 0.00090132669048 | All passed |

The CUDA maximum is step-zero loss (reference 0.00048499566037, absolute error
0.00000043713953). The next largest normalized error is 0.00060066114 for
step-two `c_attn.weight` gradient. These are the final replay measurements;
earlier provisional dirty-source outputs remain separate and are not substituted
for these numbers. This evidence supports the tested tiny GPT2/ModelOutput
training path, not every Transformers model, vLLM engine, Omni model, or a speed
claim. GPU sharing and the tiny workload preclude a performance conclusion.

## Reproduce and inspect

The committed API tests can be run unchanged against each interpreter:

```bash
CUDA_VISIBLE_DEVICES='' JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  JT_BACKEND=cpu JT_USE_CUDA=0 use_cuda=0 \
  <jittor-python> -m pytest -q compat/tests/torch/test_pytree_custom_nodes.py
CUDA_VISIBLE_DEVICES=<gpu> JITTOR_TEST_DEVICE=cuda \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python JT_BACKEND=cuda JT_USE_CUDA=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_pytree_custom_nodes.py
env -u JITTOR_TORCH_SHIM -u PYTHONPATH CUDA_VISIBLE_DEVICES=<gpu> \
  JITTOR_TEST_DEVICE=cuda <real-torch-python> \
  compat/tests/torch/test_pytree_custom_nodes.py -v
# For the CPU oracle, hide CUDA and set JITTOR_TEST_DEVICE=cpu.
```

The unversioned raw artifacts are under
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`:

- `logs/pytree-*-before.log` preserves the original independent API failure.
- `logs/pytree-*.log` and `logs/adapter-513-*.log` contain the directed/core/
  structure/layout results.
- `tf513_training_check.py` is the small public-embedding wrapper around the
  versioned `training_probe.py`; `replay_tf513_clean.py` launches the four isolated
  runs and checks the fixed thresholds. They are external experiment fixtures,
  not new registered ecosystem coverage.
- `tf513-clean-c92ae5cec/` contains both runtimes' CPU/CUDA NPZ, metadata and logs,
  plus per-array comparison JSON and its manifest. Each metadata record binds
  the clean source SHA, probe SHA, initial weights, dependency origin/version,
  precision settings, actual backend and fallback count.
- `pytree-final-manifest.json` binds **31 artifacts and six source files**, including
  both external fixture scripts, their initial weights and every final output.
  SHA256: `e48eb9d3e5bcb98427ec80ed535a229bd3fdbb1f002e1e8742ee4ac1222be4af`.
- The inner clean-replay manifest SHA256 is
  `1e706c2d679231ef2f96f27348ea29595a788562e848a89ef9a71faea74b4001`.

To reproduce the wrapper with the existing versioned runner, register an external
case whose builder calls the registered `transformers_gpt2` builder from `CASES`
with the supplied Torch namespace, wraps its model as
above, adds input spec `embedding_noise=('float32', (2, 8, 64), None)`, and then
calls `training_probe.main()`. Invoke the retained external wrapper with:

```bash
<python> <external-wrapper> --root <checkout> --case <external-case> \
  --runtime <torch-or-jittor> --device <cpu-or-cuda> --steps 3 --seed 7 \
  --weights <same-initial-state.npz> --output <outside-checkout.npz>
```

The retained wrapper and replay script provide the exact fixture and environment
used for this report.

Integration onto `a5932248fdb06fb52293f5ddd8a357f4920195ff` independently
rehashed all 31 artifacts and six matching source files and recomputed all 177
array comparisons on each device. The measured replay baseline above is
unchanged. The integrated tree passed CPU core (322 passed, 42 skipped, one
existing xfail), CPU owner/adapter checks (24 passed), CUDA owner checks
(17 passed, fallback delta zero), and combined structure/tools checks
(1686 passed, six environment/capability skips). Existing implementation lint
diagnostics were compared with the parent; no new diagnostics were introduced.
Unversioned integration logs and independent numerical audit are under
`$JITTOR_LAB_ROOT/_state/pytree-scm-integration/`.
