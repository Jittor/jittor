# ACL 注册表与类型擦除

状态：2026-09-08，维护者 coord；基线 `86c93c0e4`，本批本地提交包含本文。
8.06 的注册表边界已实现，整个任务仍未完成；未做 CANN/NPU 实机验证。

## 实现与行为

`backends/acl/include/acl_op_registry.h` 用一个统一 workspace 查询回调与一个
有类型的 execute 指针替代原来的 77 个函数槽和 39 个构造重载。
Unary、Cast、Binary、Add 四类实际使用的查询保留显式参数适配；已有直接调用
SDK 查询的 runner 继续保留其查询，不再保存无调用者的查询指针。
`backends/acl/src/acl_jittor.cc` 持有惰性只读注册表，所有 TU 共用一个实例。
全部 103 条原始名称/execute 绑定及保留的查询绑定逐条相同；两个既有重复名称
保留，最终表仍是 101 个不同名称。Unary/Binary 的同步策略未改变。

检查工具不再按文件名过滤整个 `acl_jittor.h` 的错误，任何编译器非零退出都失败。
修正队列结构检查：约束所有输入/输出访问属于 `current_op`，替代已被连续化路径
作废的固定输入调用次数。

## 验证范围

- 桩 SDK 加真实核心头：47/47 个 C++ TU 通过，70 个 launcher ABI 站点通过。
- 编译链接真实注册表和两个独立 TU：实例地址相同；四类查询的参数、状态、
  workspace 大小及 executor 转发正确；拒绝 direct 条目的误查询和空 launcher。
  对象大小不超过 64 字节。该主机测试 1 passed。
- 主机 ACL 结构集合 222 passed / 1 failed；唯一失败为上述旧队列次数合同，
  改为访问 owner 不变量后该节点 1 passed。未重复整套运行。
- 71/71 个 runner 无手写 launch 尾部；程序 token 对比只有 Unary/Binary 的
  查询字段和 launcher 访问拼写改变，其余 owner 不变，同步/失败策略不变。
- 两个反向对照均退出 1：同名 `acl_jittor.h` 内未声明符号不再被过滤；把 workspace
  查询函数作为 launcher 时 ABI 检查报错，即使普通语法检查可以接受它。

完整 ACL 目录另有两个依赖运行时 import 的文件因 Python 构建配置缺失而收集失败；
上述主机集合排除了它们，不能描述为全目录通过。原始日志位于实验根目录下
`_state/acl-registry-4fSFld/`，包括 `syntax.log`、`registry.log`、
`acl-host-structure.log`、`launch-program.diff` 与反向探针。

## 剩余与上机交接

旧看板“只剩类型擦除”的结论不成立。`acl_data.py` 和 `acl_data_channel.h`
已有主机 schema/decoder/cache 壳，但生产 `_code.py` 仍生成属性赋值源码，
runner 尚未消费该 data 通道，也没有使用描述符缓存。属性通道仍有少量 owner 待迁；SwiGlu.dim 已迁为 int64 typed data，并由结构合同覆盖；
描述符缓存属于用户允许后移的优化。详见[迁移边界](../architecture/acl-structure-boundary.md)。

在 CANN 机器上必须先用真实 SDK 编译全部 ACL TU，再按
[Ascend 指南](../../docs/guides/ascend-910b.md)证明 NPU 实际驻留与禁止 CPU fallback。
本批重点覆盖 Unary/Cast/Binary/Add 查询，以及正反向 runner 的同步和失败传播。
桩查询声明为 variadic，主机测试的四类签名是自有 mock；两者均不能证明所安装
CANN 版本的实际查询 ABI。SDK 升级或注册表/查询适配器改变时重新验证。
