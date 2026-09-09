# 2.19 用户错误边界收尾

状态：功能边界已实现，待协调者合入看板；不包含 ACL 与 oneDNN 专属改动。

本批将公开参数错误从内部断言中分离：reindex/numpy_code/code/binary/reduce/
setitem/getitem/fused_adamw、梯度入口、VarHolder 接口、NanoString/NanoVector、
profiler/memory-profiler、device setter，以及 CUDA cuDNN 卷积和 cuSPARSE CSR/COO
元数据检查均使用可捕获的 `USER_CHECK`/`USER_ERROR`。CUDA cuDNN 六种卷积构造器检查
stride/padding/dilation/groups/format/尺寸；cuSPARSE 的 rank、dtype、shape、索引长度
和矩阵维度检查只在 JIT 执行阶段读取已解析 shape，避免动态 shape 在构造期被误拒绝。
内部 producer/fuser/graph/backend status、析构、CUDA 计划和注册表不变量保留断言。

针对 CodeOp 的空 gradient source、multi-grad 输出/输入索引和计数也加入了边界检查；
ACL 分区另有 data-channel 错误分类与负向节点，合入时保留其 data 复制改动。

## 定向证据

本地 `git diff --check` 和 C++ 源结构核对通过。已有 2.19 CUDA 用户边界验收报告记录
真实 CUDA 结果；本批新增代码应在协调者合并 ACL/oneDNN 后，以一次 CPU 优先构建执行
负向输入及“抛错后继续计算”节点，再补 CUDA 构造器/CSR-COO 负向节点。未在此树重复
启动 JIT，避免与其它后端 ABI 批次争用缓存。

## bool Var getitem 既有失败

协调者在同一错误门禁中复现了唯一失败：bool Var 作为 getitem 索引的旧负向测试期望
抛出 `RuntimeError`，当前实现实际支持该索引而不抛。对照 2.19 diff，getitem 仅新增了
ellipsis 维度边界的 `USER_CHECK`；bool-index converter、getitem 的 bool Var 路径以及
该测试期望均未被本批改动。该失败在本批基线已存在，因此不能通过改变期望或添加重复
检查来伪造验收。它属于独立的旧语义测试/兼容决策，保留在已知失败清单，不阻塞本批
用户边界实现。

关闭建议：将 2.19 功能边界标记为完成，验收记录保留 bool Var getitem 的基线失败和
ACL/oneDNN/真实硬件节点的独立状态；不要把该旧失败计入本批回归。
