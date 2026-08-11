# Tutorials

The executable tutorials are maintained only as MyST Markdown under
`examples/notebooks`. The `tutorials` nox session materializes notebooks in
external state with Jupytext and executes five offline CPU smoke tutorials.

## Quick start

- [Model definition and training](https://github.com/Jittor/jittor/blob/master/examples/notebooks/example.md)
- [Operators and variables](https://github.com/Jittor/jittor/blob/master/examples/notebooks/basics.md)
- [Meta-operators](https://github.com/Jittor/jittor/blob/master/examples/notebooks/meta_op.md)
- [Custom C++ and CUDA operators](https://github.com/Jittor/jittor/blob/master/examples/notebooks/custom_op.md)
- [Profiler](https://github.com/Jittor/jittor/blob/master/examples/notebooks/profiler.md)

## Open the notebooks

```bash
python -m pip install -r requirements/examples.txt
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
python -m notebook --ServerApp.root_dir="$STATE"
```

Run the reproducibility and offline CPU checks with:

```bash
python -m nox -s tutorials
```
