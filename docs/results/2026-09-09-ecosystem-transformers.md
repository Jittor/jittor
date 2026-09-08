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
