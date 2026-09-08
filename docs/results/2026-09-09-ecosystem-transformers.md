# Ecosystem import smoke (2026-09-09)

The isolated overlay `/home/zy/jittor-lab/_state/torchlatest-extra` was
checked with the independent Python 3.12 PyTorch environment:

```text
torch 2.12.1+cu126
swift 3.8.0
verl 0.9.0
```

`import torch`, `import swift`, and `import verl` all succeed when launched
with `PYTHONPATH` set to that overlay. No model was downloaded and no model
forward/backward claim is made. The full ecosystem parity gate still requires
random tiny-model forward/backward comparisons under the Jittor shim and a
matching dependency lock.

The first local random-model attempt was also recorded. With the incompatible
torchvision wheel temporarily hidden, `Swift.prepare_model` reached the LoRA
wrapper but a tiny GPT-2 forward failed with
`TypeError: Linear.__init__() missing 1 required positional argument: config`.
This is the installed ms-swift 3.8.0 / Transformers 4.56.2 compatibility
boundary. `verl` imports successfully (0.9.0), but trainer construction was
not attempted because its distributed runtime requires a complete Ray setup.
