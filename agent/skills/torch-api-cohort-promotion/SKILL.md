---
name: torch-api-cohort-promotion
description: 把 installer 内动态定义的 Torch API 提升为模块级对象时，维护身份、状态捕获、设备参数与注册语义。
---

# 整理一族 compat API 的对象归属

## 操作

1. 以同一语义族为边界，列出公开名称、唯一实现所有者、导出/安装位置和真实状态依赖。
2. 模块级函数或类不能丢失原闭包捕获的状态，也不能把每次安装的私有状态变成进程全局共享。
3. installer 只负责绑定已有对象；验证多次安装、多个模块入口的 identity、名称和 pickle 路径。
4. device/dtype 等参数要传到实际行为，不能仅保持签名。涉及张量时运行 compat 行为回归。
5. 更新维护中的保真度/归属声明；结构测试与数值测试各守自己的边界。

[scripts/count_installer_closures.py](scripts/count_installer_closures.py) 用 AST 统计
installer 内的 def/class/lambda。默认扫描 `compat/`，`--only` 限定文件。
`--min` 只过滤 def/class 数量，会隐藏纯 lambda 的项；完整审计使用默认值。
计数为零不证明对象语义正确，也不能将输出当成 API 支持表。
