# Get started with Jittor

[Documentation](../index.en.md) · [中文](index.md)

Start with [installation](installation.en.md), then [quickstart](quickstart.en.md). This path uses only a CPU and needs no model weights or dataset downloads.

| Your task | Read next |
| --- | --- |
| Learn tensors, gradients and training | [Complete introductory tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/getting_started.md) |
| Run an existing PyTorch script | [Torch compatibility guide](../compatibility/torch.en.md) |
| Continue with a course | [Tutorial directory](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/README.md) |
| Look up function arguments | [API reference](../api/index.en.md) |

## Run the full tutorials

The repository stores Markdown tutorials in `examples/notebooks`. Convert them into notebooks from the repository root:

```bash
python -m pip install -r requirements/examples.txt
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
python -m notebook --ServerApp.root_dir="$STATE"
```

Complete installation before running the cells. The tutorial directory lists prerequisites and device requirements; chapters that download datasets need network access. Generated notebooks and your outputs stay in `STATE`.

<!-- sphinx-only
```{toctree}
:hidden:

installation.en
quickstart.en
```
-->