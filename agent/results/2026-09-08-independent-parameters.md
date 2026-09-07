# Independent Parameter types and containers

- Status: real Parameter hierarchy implemented in explicit independent mode
- Baseline: `316698abd`
- Date: 2026-09-08
- Owner: coord
- Review when: parameter construction, serialization or layer adapters change

Independent `nn.Parameter` now really inherits the installation's Tensor type.
Its constructor allocates a separate native holder on a detached graph leaf,
sets the requested gradient flag, and runs the normal Python subclass
construction protocol. It does not mark the source Tensor as a parameter.
The binding's inferred `_frontend_result_type` selects Tensor for Parameter
operations; explicit construction scopes still allocate the requested
Parameter subclass. Native Var and the unified graph representation are unchanged.

Layer adapters replace constructor-owned parameter references by identity,
preserving tied aliases and excluding buffers, caller-owned objects and shared
native child parameters. The FSDP parameter helper resolves the independent
Parameter owner. ParameterList and ParameterDict are now distinct Module
subclasses in independent mode. They convert incoming native or frontend
tensors to new Parameters, retain existing Parameter identity, and expose
parameter traversal, ordering, indexing/mapping operations and get_parameter.
Their public module spellings share the same class objects. Legacy mode is
unchanged.

Independent Tensor pickle and deepcopy reconstruct the actual type, dtype,
gradient flag and Python state. Reconstruction initializes the native holder
without rerunning user subclass constructors. Deepcopy memoizes the object
before copying its state. This fixes Parameter deepcopy falling through the
native tensor-copy protocol and returning an ordinary Tensor. The historical
NumPy reconstruction helper also resolves the active frontend owner.

Verification: CPU state/bootstrap/transaction/context checks and the real
independent integration case: 109 passed, no skips, 4.07 s. Real CUDA 12.2/sm89
integration: 1 passed, no skips, 4.42 s after incremental compilation. The
integration includes failed installation and retry, native type/policy
preservation, Parameter inheritance/leaf identity, source marker preservation,
Tensor results from Parameter and custom-subclass arithmetic, Parameter pickle
and deepcopy, custom subclass metadata, Tensor pickle, parameter-container
conversion/access/state_dict, and Linear/Sequential parameter gradients.
CPU/CUDA caches were separate. No full suite, wheel rebuild or NPU execution.

7.12 stays open. Default activation/deployment still uses legacy mode. General
serialization storage-alias/stride guarantees, all container edge semantics,
shared native-child conversion and full independent API coverage are not
established by this focused integration case. No empty uninitialized-parameter
types were added as substitutes for those remaining behaviors.
