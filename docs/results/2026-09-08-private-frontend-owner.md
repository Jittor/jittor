# 独立前端状态私有化

2026-09-08，协调分区；基线69c533404并整合本轮3.04/5.03。状态：此边界已实现，
7.12整体未完成。复查触发：安装事务或owner路由变化。

owner映射成为compat私有弱module binding，独立激活不再向native module发布
leaf/retained/optimizer别名；仅显式legacy初始化保留旧属性。绑定与回滚在同一事务
中，owner查找不延长已卸载前端生命周期。独立tensor工厂不再写仅legacy需要的plain标记。

15项无JIT状态合同通过；当前核心上的CPU状态/前端/生命周期23项通过，CUDA前端
2项通过，包含新native module无状态别名断言。未重建wheel、未做NPU验证。
日志（未版本化）：`/home/zy/jittor-lab/_state/compat-dist-OIR3DI/owner-integration.log`
与`owner-cuda.log`。后者实际319.18秒：完整前端脚本虽只有2个node，仍覆盖大量计算，
常规状态变更不要重复跑它；使用无JIT合同与具体短链路，把完整整合留到核心批次。

最近optimizer也改从弱注册表解析，删除强持最后一个optimizer的全局字段；单步Adam
之后optimizer和Parameter可释放，见owner-small.log。存储模型在同批另行整合。
剩余：legacy activation/原生类型修改路径收敛及API对象边界的整体复核。
