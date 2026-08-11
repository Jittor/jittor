# jittor-hf-compat

Narrow Transformers version-drift compatibility for Jittor's torch shim.
Importing the package has no Jittor or Transformers side effects. Jittor loads
the registrar through the `jittor.module_patches` entry-point group, or an
application can call `jittor_hf_compat.install()` explicitly.

The package registers exactly three module patches:

- `transformers.utils.import_utils`: restore legacy torch availability helpers
  needed by trust-remote-code models when `torch` is the Jittor shim.
- `transformers.modeling_utils`: convert legacy list-form `_tied_weights_keys`
  declarations before newer Transformers code expands them.
- `transformers.configuration_utils`: normalize MiniCPM's modern no-op default
  RoPE placeholder back to `None` for older model implementations.

This package does not own TRELLIS, PEFT, TRL, ms-swift, FSDP, or general torch
semantic emulation. Install its `transformers` extra only when the application
wants this distribution to install Transformers as well:

```sh
python -m pip install 'jittor-hf-compat[transformers]'
```
