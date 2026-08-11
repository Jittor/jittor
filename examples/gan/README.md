# Conditional GAN Web Example

`simple_cgan.py` downloads the historical pretrained weights only after its
command-line entry point is invoked. PyWebIO is optional and is installed by
`requirements/examples.txt`.

Start a local-only server:

```bash
python examples/gan/simple_cgan.py --host 127.0.0.1 --port 8123
```

Generate one image without opening a listening socket:

```bash
python examples/gan/simple_cgan.py --digits 0123456789 \
  --output "$JITTOR_LAB_ROOT/_state/examples/cgan.png" --no-server
```

The `--generator-weights` and `--discriminator-weights` options accept local
paths when network access is unavailable.
