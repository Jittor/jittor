# 分配器拒绝共享时 getitem 不回退为拷贝

- Status: 已复现，未修复；测试已由死路径复活为真实失败
- Date: 2026-09-09
- Baseline: `e88009b14`
- Owner: 内存与算子维护者
- Review when: `getitem` 的 inplace/共享决策改变，或 `Var::share_with` 的
  storage-view 守卫改变

## 结论

`tests/mem/test_allocator_contracts.py` 长期被记为「8 条既存失败」并整体略过。
实测其中**只有 2 条是死测试**：`test_inplace_ops_use_the_explicit_share_relation`
读 `src/ops/getitem_op.cc`，而布局迁移后该文件在 `src/ops/composite/` 下，于是它以
`FileNotFoundError` 失败，从未验证过任何运行时行为。改指真实路径后两条通过。

剩下 **6 条是真实缺陷**（`test_gc_all_is_reentrant`、
`test_getitem_copies_under_every_allocator`、
`test_setitem_copies_under_every_allocator`，各在 CPU 与 CUDA 两个类下）。
当前口径为 `6 failed / 6 passed`。

## 复现

```
JITTOR_TORCH_SHIM=0 PYTHONPATH=<repo>/python \
  python -m pytest -q tests/mem/test_allocator_contracts.py
```

```
[Reason]: var.cc:218 User check failed:
  size == 0 || (is_contiguous() && (!input() || !input()->is_storage_view()))
  Allocator cannot represent shared strided storage
```

触发点是 `_getitem_roundtrip` 中的 `b = a[0]`（`a` 为 `(4, 8)`），在
`no_sfrl` / `nfef` / `no_temp` 三组分配器 flag 下。

## 机制

`getitem_inplace` 把算子标记为 `_inplace` 并调用 `Var::share_with`。当所选分配器
无法表达该共享关系时，`share_with` 清掉 `share_src`/`share_offset`，随后为该 var
重新分配一块**连续**内存（`src/core/var.cc` 的 `allocator->alloc(storage_span_bytes())`）。

`var.cc:218` 的 `USER_CHECK` 挡在这条回退路径前面，且它是对的：一个带非平凡 stride
的 storage view 无法用一块新的连续内存表示——那需要按索引搬运数据，不是换个块。
守卫在这里响亮失败，好过静默给出布局错误的结果。

因此缺陷不在守卫，而在上游：**分配器无法共享时，`getitem` 不应该产出 storage view，
而应直接发出拷贝。** 测试注释写的正是这个契约——「when the underlying allocator
refuses to share, the kernel must still copy」——实现没有做到。

## 影响范围

默认分配器组合未在本轮复现该失败；触发需要显式关闭 SFRL、启用 NFEF 或关闭
temp 分配器。所以这不是默认路径上的静默错误，而是**非默认分配器配置下的硬失败**，
并连带使 `gc_all` 的可重入契约无法验证。

## 未做

未改 `getitem`/执行器的 inplace 决策。该修复需要在稳定的工作树上按
verify-then-fix 做定向回归；本轮树上有多个并发 agent 在改 `src/` 与算子，
盲改的风险高于收益。修复落地后应补一条「分配器拒绝共享时产出的是拷贝而非视图」
的定向断言，并逐一回退确认其变红。
