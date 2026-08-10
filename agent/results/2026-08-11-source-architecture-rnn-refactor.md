# 2026-08-11 源码架构重构第四批：RNN

## 结论

完成源码现代化第四批：从 `python/jittor/nn.py` 迁出 7 个 RNN 类，按 cell、共享
参数/递推/cuDNN backend、序列层拆成 3 个私有模块。`nn.py` 从 4,306 行降到
3,747 行，净减少 559 行（13.0%）；相对第二批开始前的 5,220 行已累计减少
1,473 行（28.2%）。

本批没有修改 RNN 数值算法、公开构造参数、权重创建顺序、state_dict 名称、递推
门序、投影、双向/多层状态布局或 cuDNN 调用参数，也没有新增依赖。

## 实现结构

```text
python/jittor/
  nn.py                         # 稳定公开 facade 与同对象重导出
  _nn/
    recurrent_cells.py          # LSTMCell / RNNCell / GRUCell
    recurrent_base.py           # 参数布局、递推调度、完整 cuDNN 路径
    recurrent_layers.py         # RNN / LSTM / GRU
    runtime.py                  # 运行时代理与类/方法公开元数据恢复
```

三个新模块分别为 185、220、183 行；`_nn` 现在共 11 个文件、1,684 行，最大模块
285 行。`nn.py` 的结构预算从 4,400 行收紧到 3,800 行，私有实现模块继续不超过
350 行。

## 兼容边界

- `LSTMCell/RNNCell/GRUCell/RNNBase/RNN/LSTM/GRU` 由 facade 直接重导出实现类，
  没有 wrapper 或兼容子类，公开对象身份不变。
- 七类的 `__module__`、`__qualname__` 以及类方法来源保持 `jittor.nn`；类和六类
  可实例化模块均完成 pickle 往返，state_dict key 顺序不变。
- `RNN/LSTM/GRU` 仍直接继承公开 `RNNBase`；Torch shim 的
  `torch.nn.modules.*` 快照继续引用同一类对象。
- 私有实现通过 `jt.nn.init`、`jt.nn.matmul_transpose`、`jt.nn.relu` 和
  `jt.nn.dropout` 动态解析 facade，避免静态捕获绕过后端或 shim 后置重绑。
- 参数创建、cuDNN weight offset 缓存、gate flatten 顺序、training/dropout、
  projection fallback、batch_first 复原和 hidden/cell 返回结构整体迁移。
- MPI 没有 RNN 专用路径；保留根 `Module`、参数属性名及插入顺序后，通用参数广播
  和序列化协议不变。

7 个迁移类与未拆分提交 `a6eaa7bb` 完成 AST 对照。归一化 `jt.nn.*` facade 名称、
`RNNBase` 基类引用和 docstring 行尾空格后，7/7 AST 全等。

## 验证

| 验证 | 结果 |
| --- | --- |
| CPU 结构、独立递推 oracle、compat RNN | 32/32 通过 |
| CUDA 结构、独立递推 oracle、cuDNN 前向/有限反向 | 32/32 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| 公开类与实例 pickle/state_dict | 7 类、6 类实例全部通过 |
| wheel 内容 | 1,017 个条目；11 个 `_nn` 文件齐全；0 个禁止项 |
| wheel 隔离安装 | 32 项：31 通过、1 项源码条件跳过 |

CPU/CUDA 主矩阵覆盖 RNN tanh/relu、LSTM、GRU、cell 单步、batch_first、多层、双向、
独立 NumPy recurrence 前向、CPU finite-difference input backward，以及 CUDA cuDNN
LSTM 参数梯度有限性。CUDA Torch shim 总入口还验证 batch_first 等价、GRU shape 和
与真实 PyTorch 相同权重下的 LSTM 输出。

## 基线同样存在的旧测试问题

`test_lstm.py` 的 2 项和 `test_rnn.py` 的 21 项在当前树与未拆分基线
`a6eaa7bb` 上得到相同错误：Torch compatibility 安装后 `jt.float32` 是 dtype 对象，
旧测试仍调用 `jt.float32(data)`。`test_rnn.py` CPU 结果在两边均为 36 项中 3 通过、
12 项 CUDA 跳过、21 项同类错误。

另外选择 5 个旧 cuDNN/PyTorch 对比用例时，当前树与基线均为 1 项前向通过、4 项
训练错误；错误发生在 optimizer 汇总参数梯度时把 `None` 与 Var 相加。主 CUDA
矩阵的 cuDNN 前向和有限反向均通过，因此这些结果登记为既有测试/优化器问题，不在
纯结构提交中改变算子或 optimizer 语义。

## 打包与环境

wheel 检查确认没有 `.pyc`、`__pycache__`、`jittor/projects` 或
`jittor_fsdp2`。隔离测试的 `jittor.__file__` 指向 wheel 安装目录，RNN 类与方法
仍显示 `jittor.nn`。所有构建、JIT 缓存和临时 worktree 均位于
`/home/zy/projects/jittor-lab/_state/verify/source-refactor`；临时 worktree 已删除。

本批没有执行真实 NPU/ACL RNN。源码审计未发现 RNN 专用 ACL wrapper；动态 facade
解析仍按后端兼容要求保留，因此不把源码审计宣称为 NPU 执行验证。

## 后续

下一批处理 convolution。该区域必须先保护 ACL 对 `Conv/Conv2d/conv2d` 的后置
重绑、DepthwiseConv 循环依赖、Torch shim 类快照、cuDNN 2D/3D/transpose 条件和
别名身份，再分 functional/backend 与公开类两个阶段迁移。
