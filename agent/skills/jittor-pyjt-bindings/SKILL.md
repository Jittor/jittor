---
name: jittor-pyjt-bindings
description: 修改 pyjt 生成器、Python/C++ 转换或公开绑定时，检查真实生成结果及引用、视图和对象初始化语义。
---

# 验证 pyjt 绑定与对象生命周期

## 操作

- [render_bindings.py](scripts/render_bindings.py) 从指定源码生成绑定；先读参数和当前 generator
  接口，再在 lab 输出。实际运行产物从当前 `jittor.compiler.cache_path` 获取，不能用
  `find | head` 或最近修改时间猜测。
- 对比实际生成代码中的签名、默认值、重载分派、异常翻译和返回策略，随后触发真实编译和调用。
- 手写 `_PyObject_New` 分配路径不会自动清零自定义字段；逐个确认初始化、析构和部分失败的清理。
- 同时维护 Python 对象与底层 view 内存的生命周期。父对象释放、临时返回、循环引用、别名读写
  和重复销毁要有针对性用例，不能只以“import 成功”验收。
- 公开对象 identity、模块归属和序列化路径改变时验证调用者；生成器修复应覆盖多个签名类别。

危险 crash 使用 `tests/_helpers/child_process.py` 隔离。默认超时以 helper 当前实现为准，
不要把固定机器上的编译时长当合同；首次编译按隔离规则串行准备。
