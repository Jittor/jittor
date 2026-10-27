# Core API

Import the core API with `import jittor as jt`. Dynamic operators are exposed
through `jt.ops`, and common operators are available on both `jt` and `jt.Var`
(that is, in functional and method form).

:::{autopublicmodule} jittor
:::

The complete list of dynamic operators is available at runtime with
`help(jt.ops)`. Operator metadata is generated from the installed Jittor
package; the documentation build does not rewrite runtime objects.
