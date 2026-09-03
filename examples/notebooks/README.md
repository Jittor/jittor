# Jittor notebooks

The Markdown files in this directory are the only canonical, reviewable
notebook sources. Jupytext materializes `.ipynb` files in a temporary or
distribution-specific output directory; generated notebooks are not committed.

Validate every source and execute the offline CPU smoke tutorials with:

```bash
python -m nox -s tutorials
```

To open one tutorial without writing a product into the checkout:

```bash
mkdir -p "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
python -m jupytext --to ipynb \
  --output "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks/basics.ipynb" \
  examples/notebooks/basics.md
python -m notebook \
  --ServerApp.root_dir="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
```

Notebook cells use these execution tags:

- `network`: downloads data or other remote assets.
- `cuda`: requires or explicitly enables CUDA.
- `gan`: belongs to a GAN workflow.
- `long-running`: performs training or another expensive operation.
- `interactive`: produces interactive or help output unsuitable for a smoke run.
- `skip-execution`: excluded from the repository's curated CPU smoke test.

Generated notebooks must have no saved outputs or execution counts. Keep
machine-specific cache, home, and environment paths out of the MyST sources.
