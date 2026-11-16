# Attention API

[Section contents](index.en.md) · [中文](nn-attention.md)

`jittor.nn.attention` provides attention operators, including scaled dot-product
attention and its fused implementations.

When choosing an attention interface, check input shapes, masks, dtypes and device requirements. Fused implementations depend on the backend; an exported name does not guarantee execution on every device.

Set `JITTOR_FLASH_ATTN_JITTOR_SRC` to directly compile official FlashAttention
CUDA sources. This backend uses FP16/BF16; FP32 is not implicitly downcast.
Ordinary execution supports forward and implemented custom backward paths.
CUDA graphs currently support dense calls with no gradients or dropout and
aligned head dimensions, including GQA and rectangular causal attention. Graph
nodes own their inputs and outputs. Other extension calls explicitly refuse
whole-step capture and execute through Python, preventing replay from reusing
outputs unrelated to new inputs. Availability still depends on the compiled
heads/dtypes, upstream source version and target device; a successful import
does not establish validation for every feature.

:::{autopublicmodule} jittor.nn.attention
:::

## Related pages

- [Neural Network API](nn.en.md)
- [Core API](core.en.md)
- [Quickstart](../getting-started/quickstart.en.md)
