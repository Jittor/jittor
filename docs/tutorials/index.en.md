# Tutorials

Executable tutorials are maintained as MyST Markdown under
`examples/notebooks`.

**The authoritative learning path is
[`examples/notebooks/README.md`](https://github.com/Jittor/jittor/blob/master/examples/notebooks/README.md)**.
It contains the complete ordered course and states what each tutorial teaches,
what it requires, and whether a GPU is needed.

This page does not repeat that list. Previously this page, the root README, and
the notebooks directory each kept a copy; the three disagreed and none was
complete. `tests/integration/test_notebooks.py` now checks that the
authoritative path lists every tutorial, so a tutorial that exists but is not
scheduled turns the gate red.

## Where to start

The **single entry point** is
[Getting started: install, run, and train a first model](https://github.com/Jittor/jittor/blob/master/examples/notebooks/getting_started.md).
It covers installation through training and runs entirely on CPU. After that,
choose a track by background:

* **Native Jittor**: continue with
  [Operators and Var](https://github.com/Jittor/jittor/blob/master/examples/notebooks/basics.md),
  [Devices and residency](https://github.com/Jittor/jittor/blob/master/examples/notebooks/device_placement.md),
  and [Defining and training a model](https://github.com/Jittor/jittor/blob/master/examples/notebooks/example.md).
* **PyTorch compatibility**: if you already have `torch` code, start with
  [Writing Jittor with the PyTorch API](https://github.com/Jittor/jittor/blob/master/examples/notebooks/torch_compat.md),
  then read
  [Migrating an existing PyTorch script to Jittor](https://github.com/Jittor/jittor/blob/master/examples/notebooks/torch_compat_migration.md).

"Devices and residency" comes before training on purpose: Jittor moves the
whole graph with one global flag instead of giving every tensor its own device.
Without that difference, later performance problems get blamed on the wrong
place.

## Open them locally

```bash
python -m pip install -r requirements/examples.txt
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
python -m notebook --ServerApp.root_dir="$STATE"
```

Notebooks are generated from Markdown sources; **generated files never enter
the repository** and always live under `$JITTOR_LAB_ROOT/_state`. For
reproducibility and the offline CPU check use:

```bash
python -m nox -s tutorials
```
