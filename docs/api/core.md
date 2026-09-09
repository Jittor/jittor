# 核心 API

用 `import jittor as jt` 导入核心 API。动态算子通过 `jt.ops` 暴露，常用算子同时
挂在 `jt` 和 `jt.Var` 上（即函数式与方法式两种写法）。

:::{autopublicmodule} jittor
:::

完整的动态算子列表在运行时用 `help(jt.ops)` 查看。算子元数据由已安装的 Jittor
包生成，文档构建不会改写运行时对象。
