# Core API

[Section contents](index.en.md) · [中文](core.md)

Import the core API with `import jittor as jt`. Dynamic operators are exposed
through `jt.ops`, and common operators are available on both `jt` and `jt.Var`
(that is, in functional and method form).

Start here for tensor creation, computation graphs and gradients. For your first training loop, read the [quickstart](../getting-started/quickstart.en.md).

:::{autopublicmodule} jittor
:::

The complete list of dynamic operators is available at runtime with
`help(jt.ops)`. Operator metadata is generated from the installed Jittor
package; the documentation build does not rewrite runtime objects.

## Related pages

- [Neural Network API](nn.en.md)
- [Optimizer API](optim.en.md)
- [Quickstart](../getting-started/quickstart.en.md)
