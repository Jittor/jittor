# vLLM extraction into the repository's adapters distribution

Status: integrated locally; not pushed, as requested.
Baseline: `b2e03162b`; owner: compatibility integration maintainers.
Recheck when the module-patch registrar, runtime ledger, or vLLM APIs change.

Following the user's revised ownership requirement, the five `compat/vllm`
implementations and their dedicated tests moved to the main repository's
`adapters/jittor_adapters/vllm` and `adapters/tests/vllm`. The existing
`jittor-torch-adapters==1.3.11.0` distribution owns them and depends on the
matching `jittor-torch`; it does not install vLLM/PyTorch itself. The earlier
external extraction directory is left untouched and is not a delivery or
publication prerequisite. The four
math/layer modules have identical ASTs after normalizing import paths.
The former initializer is now `bootstrap.py`, with lazy implementation imports
and a separate entry-point registrar. Package import alone activates nothing.

The Torch optional step selects only the `jittor_vllm` entry point in
`jittor.module_patches`, targeting `jittor_adapters.vllm:register`.
The registrar receives the public registration callback
and returns `None`. Before-import extension publication remains separate from
after-import layer patching. Both use the shared transaction/runtime-hook ledger;
foreign replacements cause conflicts instead of silent cleanup. Missing plugin
installation reports unavailable without failing the frontend. Named optional
steps no longer import unrelated adapter entry points prematurely.

Focused evidence:

- 20 host lifecycle/structure tests passed; one existing runtime public-API
  check that reads real `jittor.nn` was explicitly not selected in the fake
  backend harness. No numerical or hardware claim is derived from this harness.
- The main repository's actual optional-vLLM function passed its stdlib-only
  missing-plugin/named-selection test.
- After integration, the previously unselected public-API node ran against the
  real Jittor namespace and passed in 0.75 s. This checks public primitive
  availability, not vLLM model execution; raw output is `real-public-api.log`
  under the final `vllm-adapters-2kqwo8v1` artifact directory.
- Controlled external package import observed `before → body → after`, and
  rollback restored extension modules, module attributes, and both finders.
  Registration failure and foreign-owner conflict probes passed.
- Core, compatibility, and plugin wheel/sdist builds succeeded. Core/compat
  wheel audits passed, all three wheel path intersections were empty, and
  neither framework wheel contained the old or new vLLM implementation.
- Rebuilding the adapters wheel from its sdist preserved all eleven production
  files, including the seven vLLM owners, byte-for-byte. Actual wheel metadata discovery loaded the registrar
  without eager native/Torch/vLLM import, then rolled back registry/finders.

Raw build logs, artifacts and SHA-256 report are unversioned beneath
`$JITTOR_LAB_ROOT/_state/vllm-adapters-*`. No Git remote was created or
published. No separate repository is required by this delivery.

The historical external Ascend platform/worker package is not available in this
checkout. This extraction therefore declares no `vllm.platform_plugins` entry
point and does not claim NPU serving support. Historical CUDA checkout evidence
and the distinct NPU version reference are recorded in `adapters/README.md`;
neither is represented as a new hardware run. Numerical layer/attention tests
are preserved in the adapters package for the next target-machine run.

Integration: apply the two removed vLLM arming tests and two runtime-hook tests
to their moved source owners if the test-layout batch lands first. Preserve
the new optional-plugin test in the generic runtime-hook test file. Remove the
old vLLM process-mode selector and point the generic process-mode contract at
an existing Torch test. Together with the previously accepted core/compat
packaging boundary, this completes the revised 7.18 layout requirement locally.
Legacy frontend removal remains 7.12 and is not claimed complete by the extraction.
