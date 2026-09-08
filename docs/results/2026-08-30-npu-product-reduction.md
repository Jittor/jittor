# Ascend product reduction 原生执行修复

- Status: Verified for full, single-axis, and multi-axis float32/integer product on one Ascend 910B3
- Last reviewed: 2026-08-30
- Source baseline: `a3ae89f0` plus the changes in this report's commit
- Owner: ACL backend and reduction maintainers
- Review when: CANN, reduce lowering, dtype inference, shape, or product performance changes

## 结论

修复前，ACL executor 把所有 `reduce.multiply` 映射到无效的 `op_idx=999`；真实
NPU 上的 `jt.prod(x, dim=1)` 输出 `no such reduce` 后以 exit code 134 abort。

修复后，全量 product 使用 CANN 9 `aclnnProd`，单轴 product 使用
`aclnnProdDim`，多轴 product 按轴连续执行 `aclnnProdDim`，中间值保留在 NPU。
在一张 Ascend 910B3 上，float32 full/dim/multi-axis/keepdims 前向、非零输入的
解析反向，以及 uint8/int8/int16/int32/int64 full/dim/multi-axis 前向均与独立
NumPy 参考一致。捕获日志包含 ACL `reduce.multiply`，且没有 `compile cpu` 或
`fallback cpu`。Reduce runner 还在析构时释放每次调用创建的 CANN axis
descriptor，避免包括 product 在内的归约持续泄漏 host descriptor。

对应的 float product 与 integer product NPU OpInfo 从 skip 变为通过。sum、max、
min 的窄整数 NPU skip 保留。

## 验证范围

- Device: one Ascend 910B3
- Driver / CANN: 25.5.1 / 9.0.0
- Python: 3.9.25
- Dtypes: float32, uint8, int8, int16, int32, int64
- Shapes: 2-D full and single-axis; 3-D multi-axis; keepdims true/false
- Gradient: float32 single-axis and multi-axis product with nonzero inputs and independent formula

核心真实设备结果：

```text
tests/backends/npu/test_acl.py -k product_reduction_forward_backward
1 passed, 32 deselected

tests/ops/test_ops.py -k test_reference_prod  # JITTOR_TEST_DEVICES=npu
2 passed, 225 deselected

tests/backends/npu/test_acl.py
33 passed

tests/ops/test_ops.py  # JITTOR_TEST_DEVICES=npu
220 passed, 7 skipped

maintained NPU inventory
373 passed, 9 skipped

tests/ops/test_ops.py -k test_reference_prod  # JITTOR_TEST_DEVICES=cpu
2 passed, 679 deselected
```

## 边界

- CANN `aclnnProdDim` 一次只表达一个轴；多轴路径按降序轴连续归约，并在释放
  中间 device buffer 前同步 ACL stream。本轮验证了正确性，尚未单独声明多轴性能。
- product 对含零输入的数学梯度需要单独处理；Jittor 现有通用梯度使用
  `output / input`，本次没有扩大该既有语义。
- float16/bfloat16 和 bool 未纳入本次维护声明。
