# Jittor notebooks

The Markdown files in this directory are the canonical, reviewable notebook
sources. Each one is paired with a generated `.ipynb` file through Jupytext's
MyST format.

To refresh the pairs after editing a Markdown source, run:

```bash
find examples/notebooks -name '*.md' ! -name README.md -print0 \
  | xargs -0 jupytext --sync
```

Notebook cells use these execution tags:

- `network`: downloads data or other remote assets.
- `cuda`: requires or explicitly enables CUDA.
- `gan`: belongs to a GAN workflow.
- `long-running`: performs training or another expensive operation.
- `skip-execution`: excluded from the repository's curated CPU smoke test.

Committed notebooks must have no saved outputs or execution counts. Keep
machine-specific cache, home, and environment paths out of both pair files.
