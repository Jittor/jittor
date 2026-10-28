# 输入临时冻结不应抹掉已记录的 grad_fn

- 状态：已验证所述 CPU/CUDA 契约；KI-COMPAT-004 仍未关闭。
- 日期：2026-10-16
- 生产源码：`819182d1b333b4f300f8c530ecb97c3a96287084`.
- 修复前基线：`dfe462aa10bb3abc6aedce30c11e54a5b42867e7`.
- 维护者：原生 autograd / Torch 兼容性维护者。
- 复查条件：生产者查询、边快照、可逆 requires-grad 策略或别名创建发生变化。

## 问题与修复

先创建需要梯度的 `result = torch.Tensor(source)`、`source[:]` 或 `source * 2`，再对
`source.requires_grad_(False)`，独立 Torch 仍把 result 视作带有原 grad_fn 的非叶节点。
旧候选却把 result 改判为叶节点并返回空 grad_fn。FP32/FP64 的六个 CPU 定向用例全部失败；
独立二进制 Torch 2.11 的同六个断言在 CPU 和真实 CUDA 上分别通过。

原生 `backward_grad_fn` 查询读取了输入当前的临时冻结标志，混淆了已记录的生产者和当前
可接收梯度的目标。修复继续检查输出自身状态、永久 stop-grad、控制边和操作创建时的禁用边
快照，但输入之后的可逆冻结不再删除已有生产者。没有在兼容层伪造 grad_fn、叶身份或梯度。
原生 JIT test 同时覆盖后冻结的生产者保持不变，以及创建前已禁用的边仍被排除。

新回归还实际求出 result 的梯度，并在源重新解冻后沿旧图求源梯度，检查完整数值和设备。

## 验证结果

| 范围 | CPU | 真实 CUDA |
| --- | --- | --- |
| 四文件共享回归：constructor alias / tensor state / autograd / autograd semantics | 118 passed / 47 accelerator skips | 165 passed / 0 skipped |
| 原生 `test_backward_leaf_query.py`（含 C++ JIT tests） | 20 passed | 20 passed |
| 独立 Torch 同六个新断言 | 6 passed，2.11.0+cpu | 6 passed，2.11.0+cu128 |

CUDA 共享回归耗时 152.79 秒，原生查询 4.65 秒；这些是验证耗时，不授予性能结论。
候选的 shim 专属 fixture 使用 fallback=error 并同步实际张量；每个新梯度检查实际设备。
单测缓存正常复用，未复制编译二进制；与下游 benchmark 的缓存分离。

Canonical core：**322 passed / 42 skipped / 1 existing xfail**，两个模式退出均为零。
完整 `tests/structure + compat/tests/structure`：**1622 passed / 6 skipped**，224.75 秒。
单测针对冻结的五文件工作区运行，其字节在提交前逐文件核实与 819182d1b 一致。

## 复现与原始证据

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TORCH_SHIM=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_tensor_constructor_alias.py \
  compat/tests/torch/test_tensor_state.py compat/tests/torch/test_torch_compat_autograd.py \
  compat/tests/torch/test_torch_compat_autograd_semantics.py
CUDA_VISIBLE_DEVICES=<gpu> JT_USE_CUDA=1 JITTOR_TORCH_SHIM=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_tensor_constructor_alias.py \
  compat/tests/torch/test_tensor_state.py compat/tests/torch/test_torch_compat_autograd.py \
  compat/tests/torch/test_torch_compat_autograd_semantics.py
JITTOR_TORCH_SHIM=0 <jittor-python> -m pytest -q tests/core/test_backward_leaf_query.py
```

独立 oracle 直接调用相同 `test_recorded_grad_fn_survives_source_freeze` 测试函数的六个
参数组合，并断言真实 Torch 的二进制身份和设备；不加载 Jittor 或 shim 专属 fixture。
最初三个 pytest oracle 启动尝试分别被动态库 ABI、shim 冲突、进程模式策略拒绝，均未当作
数值验证通过；最终直接调用方法与既有独立构造器 oracle 一致。所有原日志保留。

未版本化证据在 `$JITTOR_LAB_ROOT/_state/tensor-alias-guards-dfe/20261016/`：
`directed-before.log`、`oracle-recorded-{cpu,cuda}-v1.log`、`shared-{cpu,cuda}-v1.log`、
`native-{cpu,cuda}-v1.log`、`core-v1.log`、`structure-v1.log`、完整源指纹及原始诊断脚本。
复查 32 个真实生命周期组合，候选差异由 18 个降为 17 个；修复的源冻结场景与独立 Torch
完全一致。

## 仍未关闭

no-grad 别名在源或别名写入后的拒绝保护、从冻结源继承的动态解冻标志，以及非叶 requires-grad
修改规则仍有差异，KI-COMPAT-004 保持开放。本次不授予全部 view 语义、完整 C-extension ABI、
其他加速后端或未执行下游消费者支持，也不代表 R33 的 PR 维护者处置已经完成。
