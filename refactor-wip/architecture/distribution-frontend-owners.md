# Distribution frontend implementation owners

Status: 7.03 distribution type-family migration, 2026-09-08.
Owner: distribution compatibility maintainers. Recheck when adding a native
distribution, constructor parameter or sampling policy.

`distribution_frontend.make_distribution_frontend` configures an independent
module and its class hierarchy. It creates no method, initializer or property
implementation closures. Native distribution mathematics is unchanged.

`distribution_adapters.py` owns the actual behavior:

- `DistributionConstructor` implements Normal keyword aliases, supported
  validation options, Uniform scalar bounds, parameter tensorization and
  frontend initialization scope.
- `DistributionMethod` wraps native method/property execution in the captured
  frontend scope and preserves the existing `rsample` trainability policy.
- Module-level loc/scale getters and setters keep Normal's native mu/sigma
  fields as their single values.
- `DistributionAdapterState` owns one frontend's native/target references and
  class cache. Its `adapt` method only constructs the hierarchy and binds
  descriptors, properties and copied mutable class attributes.

Descriptors keep real implementation metadata in the compatibility adapter
module. Their signatures reflect the native delegate, but they do not pretend
their code is defined in the native module through `wraps` or rewritten
qualnames. Class access returns the descriptor; instance access uses ordinary
bound-method semantics. Pickling an active descriptor resolves that exact
published class member, while an unpublished owner is rejected rather than
silently restored as a different frontend's member.

Each generated distribution remains a subclass of its native type and the
corresponding frontend Distribution bases. Independent frontends have separate
types, descriptors, policy state and mutable class attributes. Native classes
are not patched. Returned tensors still share the same native Var/Op graph.

The installer records conservative fidelity for the supported public classes
and their existing constructor, sampling, probability, moment and shape APIs.
Missing native methods are not fabricated. Native validation, parameter-shape
and backend restrictions remain; registration does not establish full Torch
distribution equivalence.

Focused coverage checks owner/scope identity, constructor aliases and refusal,
constant versus trainable sampling, native gradients, class/instance/member
pickles, frontend hierarchy isolation and unchanged native class dictionaries.
