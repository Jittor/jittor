# Independent Torch default entry

- Status: public activation and deployed torch default to independent types
- Baseline: `79966ef2f`
- Date: 2026-09-08
- Owner: coord
- Review when: deployment templates, activation mode or child environments change

Public `activate()` (and its same-object enable alias) now defaults to the
independent namespace. The deployed torch package selects independent import
composition before loading Jittor and publishes the returned Torch namespace,
not the Jittor module. Its Tensor, Parameter, Module and NN types therefore use
the independent installation implemented in the preceding migrations.

Activation records `JITTOR_TORCH_INDEPENDENT` for child interpreters. Native
composition reads that explicit mode. Existing `JITTOR_TORCH_SHIM=1` without
the independent flag remains the legacy alias path, also available through
`activate(independent_namespace=False)`. Changing mode after installation still
fails rather than mixing class identities. Native-only test children clear the
new flag along with the existing activation environment.

Real failed-install rollback revealed that the transaction ledger handled
plain dictionaries but not os.environ's MutableMapping implementation: it read
environment keys as attributes and falsely reported ownership loss. Mapping
read/restore now accepts MutableMapping, preserving additions and previous
values. A real process-environment regression covers both cases. Legacy tests
request their mode explicitly, and direct private-activation fixtures restore
their environment instead of leaking mode into later tests.

Verification: 117 CPU state/bootstrap/transaction/context/entry/template checks
passed in 5.08 s, no skips. Two real CUDA integration cases passed in 9.51 s:
explicit independent installation with failure/retry and a freshly deployed
torch-first import, model forward/backward and a child importing Jittor first.
The existing integration asserts CUDA availability/use and tensor residency.
The deployment helper also staged the actual resource package in an external
temporary site-packages directory; a direct torch-first CPU model backward
probe passed there. CPU/CUDA caches were separate. No full suite, wheel rebuild
or NPU execution was performed for this entry-point change.

7.12 remains open for full independent API/model coverage, serialization
storage/stride guarantees, shared native children and mixed-thread behavior.
Default publication is now independent; remaining work must not describe it as
an opt-in-only prototype. Public compatibility and architecture docs describe
the new entry and the explicit legacy route.
