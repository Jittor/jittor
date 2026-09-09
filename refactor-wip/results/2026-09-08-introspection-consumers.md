# Runtime 内省调用者整合

状态：2026-09-08，coord整合；10.20调用者与共享pytest策略已收齐。
基线为`86c93c0e4`的内省接口，三方合入当前源码/测试布局，保留同期缓存和精度修改。

## 行为

普通测试的设备/库可用性、运行策略、构建路径、执行/分配器/liveness计数改用
`jt.introspection`。库的只读查询保持UNPROBED；必要初始化是执行期的独立前置步骤，
FAILED不会变成零设备或skip。通用加速器轴保留ACL/ROCm，CUDA专属用例仍明确查CUDA。

修改策略使用Runtime scope，生命周期覆盖原setUp/test/tearDown以及失败后的cleanup。
子进程、手工探针和跨单元notebook的scope覆盖完整进程，并在退出时恢复。
生态测试的设备scope由调用者持有到工作负载结束，不能在选择设备函数返回时就退出。
仅比较图存活量的测试使用三个liveness字段，避免把执行次数变化误判为泄漏。

原生setter、绑定生成、启动封印、安装事务等白盒测试保留所测试的实际实现读取，
原因在`tests/_helpers/introspection_exceptions.json`逐项说明。
用户保护的`tests/core/test_setitem.py`没有修改，也不计入归零范围。

## 证据与限制

- 720个文件的AST审计包含别名、动态getattr/hasattr/setattr和嵌入脚本；列明例外外
  没有原生flags读写。审计源和结果留在实验目录，不能把该结论解释为任意Python
  反射方式的完整证明。
- 冻结CPU正式入口集中24 passed，2.70秒；前后完整policy snapshot相等。
  覆盖helpers、CPU allocator/where、生态选择、Runtime NPU stand-in、错误、统计、
  序列化owner、FSDP store绑定及激活状态。NPU stand-in不是NPU执行证据。
- 主树三方合入292个tracked Python文件，无冲突；全部语法解析通过，12个helper
  契约复验通过。同期cuTT真实入口/流API及Torch双精度隔离语义保留。
- 受保护文件SHA-256仍为
  `00e347ab6c505aba2643d5997a3c3538f74d894c985c9f88605293be326d2cdf`。

原始日志、JUnit与审计JSON位于实验根目录
`_state/single-frontend-cpu-entry-f2dluU/introspection-consumers-{final.log,final.xml,audit.json}`。
运行重用了冻结CPU源/缓存，仍有约4.5秒核心缓存检查，不声称完全跳过build_core。
没有全数值套件、模型性能、NPU或多机验证结论。

共享pytest策略最后四文件也已整合，正式pytest入口27个host契约通过：CUDA/ACL
硬件门禁读取明确的backend状态与失败原因；ROCm scope覆盖yield和异常退出；
状态调查用公共计数/策略观测，保留报告键，查询失败不再被吞成空报告。
启动封印的白盒检查继续直接观察compiler发布值，新增绕过发布但正确config不变的
反例，证明不会用正确配置掩盖被污染的接口。例外清单已删除整文件共享策略豁免，
仅保留startup bypass、有界缓存库存和原生autograd服务的具体契约说明。
