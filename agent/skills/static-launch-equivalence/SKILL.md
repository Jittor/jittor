---
name: static-launch-equivalence
description: 重构 ACL 等设备调用包装层时，比较调用参数与生命周期，并明确静态证明的边界。
---

# 检查设备调用的静态等价

## 操作

1. 在独立基线快照与候选中定位同一组 launcher、workspace query 和资源释放路径。
2. 为每个调用记录函数名、实参顺序、dtype/shape、指针来源、stream、workspace 与错误处理。
   token 序列相同只证明提取部分相同；不能忽略宏展开、求值顺序、所有权及条件分支。
3. 检查临时 storage、executor 与 workspace 的存活期覆盖异步执行。移出作用域不应提前释放。
4. 对真实 TU 做宿主语法与 launcher ABI 检查；用真实 SDK 编译验证 workspace 参数。
5. 用一处故意错位参数或错误 launcher 的隔离负例确认检查能失败，再运行受影响设备用例。

维护工具可比较两个 checkout 中提取的调用 token：

```bash
python tools/build/acl_launch_program.py "$BASELINE_TREE" "$PWD"
```

它比较 query/execute、失败处理和同步策略，不比较全部实参及对象生命周期；
不能用相同 token 替代上面的参数核对和真实 SDK 编译。

没有硬件时只报告静态检查范围，不宣布数值、竞态或性能等价。
ACL 桩工具见[宿主语法检查](../acl-host-syntax-check/SKILL.md)；
HCCL 同步变化另按[设备验收](../hccl-on-device-verification/SKILL.md)。
