# Attention API

[Section contents](index.en.md) · [中文](nn-attention.md)

`jittor.nn.attention` provides attention operators, including scaled dot-product
attention and its fused implementations.

When choosing an attention interface, check input shapes, masks, dtypes and device requirements. Fused implementations depend on the backend; an exported name does not guarantee execution on every device.

:::{autopublicmodule} jittor.nn.attention
:::

## Related pages

- [Neural Network API](nn.en.md)
- [Core API](core.en.md)
- [Quickstart](../getting-started/quickstart.en.md)
