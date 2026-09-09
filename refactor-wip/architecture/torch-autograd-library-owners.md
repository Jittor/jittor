# Autograd and custom-operator owners

- Status: Implemented ownership boundary; semantic limitations remain
- Reviewed: 2026-09-08
- Owner: compatibility maintainers
- Review when: Function context, installation ownership or operator registration changes

`compat/torch/autograd.py` owns the independent `Function`, gradient entry points,
saved-tensor context helpers and explicitly classified placeholder APIs.
`compat/torch/installers/autograd.py` publishes these objects. Its parent package
imports the installer as `autograd_installer`, leaving the `autograd` module name
for the implementation owner. Public functions/classes have real module-level
definitions and can be pickled by their implementation names.

The independent `Function` subclasses the native Function. It creates no separate
graph: `_new_call_context`, `_run_call`, native Var tapes and optional gradients
remain the execution/differentiation owners. Input shapes, saved tensors and output
metadata live on the one-shot context, not on a reusable Function instance.
`InstallContext.state["autograd_api"]` captures the original native Function call
delegate and any optional TensorDict indexing delegates before installation.
The native-as-Torch activation path and native Function monkeypatch helper have
been removed. Installation only publishes the independent module-level Function.

`compat/torch/library.py` owns `Library`, registration functions, schema inference,
operator/namespace objects and the native `_LibraryAutograd` bridge.
`InstallContext.state["library_api"]` owns the dispatcher and active type namespace.
Rebinding publishes the same API objects and retains this dispatcher. A Library
instance retains the dispatcher from its creation context. Registration decorators
may create call-time callbacks; installation does not create public implementations.

Registry map writes and operator metadata writes participate in the current
InstallTransaction or RuntimeHook. Namespace creation, implementations, fake kernels
and backward/setup callbacks are restored on enclosing failure. `Library` advertises
`_transactional_registry = True`: adapters must not add a second snapshot undo for
the same slots. External Torch implementations without this marker retain their
existing adapter cleanup path. Outside a transaction, registration remains persistent.

For a registered backward with differentiable inputs, floating/complex results are
enabled before native output tape construction, including when the kernel internally
detached its result. This makes the registered backward reachable under the explicit
requires-grad policy. Integer/bool results are not enabled. Device dispatch still
uses actual tensor residency; Meta kernels cannot serve real tensors.

Fidelity records distinguish executable approximate APIs from placeholders.
Anomaly/profiler scopes, saved-tensor offload/hooks, once-differentiable enforcement,
the legacy engine callback, higher-order dispatch and vmap annotations are not
implemented by promoting their Python objects. Existing gradient option limitations,
single-default-overload semantics and schema annotation limits remain. This boundary
does not introduce new backend support or close the aggregate 7.03/7.12 tasks.

See [focused evidence](../../docs/results/2026-09-08-autograd-library-owners.md).
