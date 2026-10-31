# relay 组成员落在 fused op 外时放弃 relay，而不是中断

- 状态：已验证（CUDA 实机）。改的是 codegen 的准入：不能表示的 relay 不做，模型照常
  用 fused op 里的通用实现算出结果。
- 日期：2026-10-31
- 基线提交：`3dc5c7ef3`（复现）
- 验证范围：RTX 4090 24 GB，CUDA 12.2.140 + `jittor[cuda12]` 组件轮子，Torch 前端
  （`JITTOR_TORCH_SHIM=1`，`enable_tuner=1`），同样三个模型在 `enable_tuner=0` 下对拍。
- 维护者：Jittor 核心维护者（与 codegen 维护者共同确认 relay 的边界）
- 复查条件：`add_relay_group` 的成员约束、`count_fuse` 对 broadcast view 的处理、
  或 relay 开始支持 fused op 之外的成员时

## 现象

Torch 前端下 ConvBERT 的一步前向在 `modeling_convbert.py` 的 span conv 处抛出
`JittorError: var_relay.cc:107: [check failed: fnodes.count(v)]`；失败算子是两个卷积
（depthwise 3×3 与 1×1）组成的 `fused_op(reindex, reindex, binary.multiply,
reindex_reduce.add)`。ConvNeXt 的 depthwise 卷积撞同一处（此前被 ConvBERT 那个失败
留下的 Var 毒化，看不出来）。

最小复现（设备上、shim 下，`enable_tuner` 由 shim 代开）：

```python
import torch, torch.nn as nn
d = nn.Conv1d(64, 64, 3, groups=64, padding=1, bias=False)
p = nn.Conv1d(64, 32, 1, bias=False)
v = torch.randn(1, 8, 64).transpose(1, 2)   # 在 no_grad 之外建出的转置视图
with torch.no_grad():
    p(d(v)).numpy()                          # var_relay.cc:107
```

同一张卡上：`p(d(v.contiguous()))` 通过；只做一次卷积通过；`v` 改在 `no_grad`
里建也通过；同样的图在 CPU（`jt.flags.use_cuda = 0`）上通过。

## 根因

`VarRelayManager::add_relay_group` 在构造 relay 组时要求：被 relay 的算子除了被替换
的输出之外，其余成员都必须是 fused op 的节点（执行时按 fused op 的节点号取）。带
broadcast view 的卷积在这里有一个成员在 fused op 之外，于是断言中断。这一步发生
在组的输入已经从各算子摘掉之后，所以也谈不上“跳过继续”。

## 修复

在动任何状态之前先做同样的检查，检查不过就放弃这个 relay 组：

```cpp
for (const auto& p : group) {
    Op* op = p.first->input();
    const auto& op_info = op->codegen();
    for (uint i=0; i<op_info.var_members.size(); i++) {
        Var* v = GET_VAR_MEMBER(op, op_info.var_members[i].second);
        if (v && v != p.first && !fnodes.count(v)) {
            relay_groups.pop_back();
            return -1;
        }
    }
}
```

放弃之后算子回到 fused op 里，由 jittor 自己的融合实现计算——比库内核慢，但结果
正确。这不是 KI-TUNER-001 里说的“放宽成员检查”（那是陷阱）；要让这两条路都拿到
库内核，仍得让那个成员加入 fused op，那是另一件事。

## 结果

- ConvBERT、ConvNeXt 在 `enable_tuner=1` 下都能跑；与 `enable_tuner=0` 的输出
  **逐位相同**（`max abs diff 0.0`，batch 1、同一随机种子）。
- `python/jittor/compat/tests/torch/test_torch_hf_models.py::TestTorchHFModels::test_forward_and_eval_determinism`：
  1 passed（33 个架构 × 两次前向，97 s）。
- `tests/codegen`：169 passed / 2 failed / 38 skipped / 5 xfailed；两条失败是
  KI-TEST-017 记录的既有用例，与本改动无关。
- `tests/backends/cuda` 的 relay 相关用例（`test_cublas_matmul_grad`、
  `test_cublas_test_op`、`test_cuda_op_capabilities`、`test_cudnn_conv_amp_dtype`）：
  20 passed / 2 xfailed / 2 failed，两条失败都是本机缺 CUB 头
  （`cub/device/device_segmented_radix_sort.cuh`），与本改动无关。
- `tools/run_test_suite.py --tier core`：327 passed / 1 xfailed。
