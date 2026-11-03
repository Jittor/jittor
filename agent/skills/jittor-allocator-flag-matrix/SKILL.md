---
name: jittor-allocator-flag-matrix
description: 定位切片、别名、重绑定或多线程引发的存储错误时，用 allocator 配置矩阵区分路径并形成回归。
---

# 诊断分配器与生命周期问题

## 诊断顺序

- 固定数据和最小读写序列，分别比较默认、SFRL、NFEF、temporary、stat allocator 配置。
  开关只帮助归因，关闭某个分配器后不再报错不等于修复。
- 输入应包括真正的算子输出：直接 `jt.array` 可能始终走 CPU allocator，没覆盖执行器分配路径。
- 检查 basic view、高级索引 copy、输入先释放、输出先释放和共享 holder 重绑定。
  用独立 NumPy 数组保存期望值，不从可能已污染的 Var 构造答案。
- 异步执行时 storage 的存活必须覆盖消费者；同步、GC 和 `.numpy()` 都会改变竞态窗口。

## 附带诊断探针

- [probe_allocator_matrix.py](scripts/probe_allocator_matrix.py) 输出各配置的 getitem/setitem 错误计数。
  它会按可用性选择 CPU/CUDA，退出码不能作验收；逐行检查实际设备及 `bad` 数量。
- [probe_shared_param_threads.py](scripts/probe_shared_param_threads.py) 是 CUDA 并发崩溃复现，
  只收集线程异常，没有最终数值 oracle；`OK` 只表示本轮未观察到异常。

探针在本任务隔离环境中运行并限制外部超时。确认缺陷后在维护中的测试增加确定性数据、
最终值检查和有界线程退出，再把它作为修复验收，不拿诊断输出替代回归。
