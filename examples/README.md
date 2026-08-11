# Jittor Examples

Examples are source-distribution assets and are intentionally excluded from the
runtime wheel. They are not a Python package and importing an example must not
download data, initialize a device, open a port, or create a Jittor cache.

Use a Python 3.11 environment for the complete notebook and web-example toolset:

```bash
python -m pip install -r requirements/examples.txt
```

Tutorial notebooks live in `notebooks/`; runnable applications live in their
domain directory, such as `gan/`.
