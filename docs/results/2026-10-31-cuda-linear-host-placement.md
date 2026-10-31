# CUDA `nn.Linear` 的 cuBLASLt 快路径：按操作数位置而不是按进程 flag 决定

- 状态：已验证（CUDA 实机）。这是对 `lt_linear_cuda` 的准入条件修正，不改变它算出的值。
- 日期：2026-10-31
- 基线提交：`3dc5c7ef3`（复现）；修复见本页末的提交。
- 验证范围：RTX 4090 24 GB，CUDA 12.2.140 + `jittor[cuda12]` 组件轮子
  （`pipcu122`），对照 PyTorch 2.11.0+cu128；两侧同一张卡。
- 维护者：Jittor 核心维护者
- 复查条件：`lt_linear_cuda`/`lt_linear_train_cuda` 的准入条件、`Var.placement_backend`
  的语义、或 Torch 前端对 host 张量的 placement 记录方式变化

## 现象

在 `compat/tests/torch/test_torch_hf_models.py` 的 ConvNeXt 子用例上，一步前向在
`modeling_convnext.py` 的 `pwconv1(features)` 处抛出：

```
JittorError: code requires source for the selected backend (code_op.cc:408)
op: code
  in:  float32[1,8,8,32,], float32[128,32,], float32[128,]
  out: float32[1,8,8,128,]
```

同一条用例原先更早地死在另一个缺陷上（`var_relay` 的内部不变量，记在
KI-COMPAT-022），那个失败留下的 Var 会毒化后续每个 `sync`，所以这个错误一直没被
看见；把模型按架构逐个前向才暴露出来。

## 根因

`lt_linear_cuda` 只带 `cuda_src`，是纯 CUDA 的 `jt.code` 算子；它的准入条件
`_supports` 只看 `jt.flags.use_cuda`。但那个 flag 说的是"新张量会去哪儿"，不是
"这些张量现在在哪儿"：Torch 前端把 `torch.tensor(...)` 留在 host 上，直到调用者
自己搬走，而 shim 在进程启动时就把 `use_cuda` 置为 1。于是这个算子被按 CUDA 建出
来，执行时却按输入的 placement 落到 CPU backend——那里没有它的 source。

## 修复

准入条件改为读取每个操作数的**显式 placement**（`Var.placement_backend`：
`0` 是 `BackendId::Cpu`，`-1` 表示没有显式 placement，由环境决定，而环境已经是
CUDA）：

```python
def _on_the_device(*values):
    for value in values:
        placement = int(value.placement_backend)
        if placement >= 0 and placement != 1:   # 1 是 BackendId::Cuda
            return False
    return True
```

`_supports`（前向）与 `_supports_training`（训练）都加上了这一条。读法沿用
`ops/concatenation.py` 的既有写法。代价是每个 linear 调用 0.04 µs/操作数；最初用
`dispatch_context` 实现是 3 µs/层（每次都要构造元组和 backend 名字），在这个模块
专门省主机时间的语境下不可接受，因此换成了直接读属性。

## 结果

- ConvNeXt 前向（同配置，`torch.no_grad()`）不再报错，输出 `(1, 64, 4, 4)`。
  注意：ConvNeXt 在 shim 下**仍**跑不通，因为它的 depthwise 卷积还会撞上
  KI-COMPAT-022 的 `var_relay` 不变量；本页修的是它后面的第二个缺陷。
- `tests/nn/test_lt_linear_cuda.py`：17 passed，含新增
  `test_a_host_placed_operand_declines_even_when_use_cuda_is_on`（用
  `_set_construction_placement(0, 0)` 给操作数打上显式 CPU placement，断言快路径
  拒绝、可移植路径给出同样的值）。
- `compat/tests/torch/test_bfloat16_linear_rounding.py`、
  `test_torch_amp_boundaries.py`、`test_nn_layer_subclassing.py`：40 passed。
- `tools/run_test_suite.py --tier core`：327 passed / 1 xfailed。
- 配对基准（同一张 4090，`benchmarks/torch_compat`，TF32 两侧都开）：`vit_b16_train`
  191.5 / 169.4 ms（0.89x）、`qwen3_train` 300.0 / 177.6 ms（0.59x），与本改动之前
  的配对结果一致，没有可测的主机开销。
