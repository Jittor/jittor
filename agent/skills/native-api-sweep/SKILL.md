---
name: native-api-sweep
description: 全面驱动原生 Jittor 公开接口面，检测入口存续、数值精度和设备显存。用于回归排查、新增算子验收、以及确认某次「清理」是否删掉了只能通过间接引用抵达的东西。三个 harness 分别覆盖可调用面、对 NumPy 的数值参考、按算子类别的显存泄漏。
---

# 原生接口全面 sweep

OpInfo 数据库覆盖 179 个算子的真实数值。**公开可调用面约 1294 个名字**，差集正是
删除和静默错误能藏身的地方。这三个 harness 补的就是这段差集。

## 三个 harness

| 文件 | 回答什么 | 代价 |
| --- | --- | --- |
| `drive_public_surface.py` | 每个公开可调用入口能否被调用；谁抛无消息异常；谁漏出 C++ 编译错误 | 高（会编译大量核） |
| `precision_vs_numpy.py` | 逐算子数值是否对得上独立 NumPy 参考 | 中 |
| `memory_categories.py` | 按算子类别重复执行是否涨显存；显式释放是否真的还回去 | 低（需真实 CUDA） |

## 运行

**必须用 PYTHONPATH，不要 editable 安装 core**：

```bash
export PYTHONPATH=<repo>/python
export JITTOR_TORCH_SHIM=0                 # 原生语义
export JITTOR_HOME=$JITTOR_LAB_ROOT/_state/<topic>/<run>/jittor-home
python drive_public_surface.py   --device cpu  --out cpu.json
python precision_vs_numpy.py     --device cuda --out prec_cuda.json
CUDA_VISIBLE_DEVICES=7 python memory_categories.py --smi-index 7 --out mem.json
```

`--smi-index` 是给 `nvidia-smi` 的**物理**卡号；进程自己跑在
`CUDA_VISIBLE_DEVICES` 暴露出的 0 号上。两者写混就会在一张卡上分配、去另一张卡上测量。

## 踩过的坑（照做能省很多时间）

1. **`HOME` 隔离会触发 oneDNN 3.9.1 从源码重建**（十几分钟）。第三方归档跟着
   `JITTOR_HOME` 走而不是 `HOME`，所以隔离缓存要么先把
   `oneDNN-v3.9.1.tar.gz` 拷进 `$JITTOR_HOME/.cache/jittor/mkl/`，要么接受这次构建。
2. **驱动器要增量落盘。** 只在末尾写结果的版本被超时杀掉后什么都不剩；现在每
   25 条写一次 `.partial`。
3. **无消息异常会让 harness 自己崩。** `str(exc).splitlines()[0]` 对空消息抛
   IndexError——`_first_line()` 把它变成一条记录。这个副产品正是发现 11 个用裸
   `assert` 做用户侧校验的公开入口的方式。
4. **别用分配器计数断言显存。** 源张量的块先回池，重复分配可直接复用，池不增长，
   于是未修复的实现照样通过。要用尺寸契约（对 `total_cuda_ram` 取比例）。
5. **测量前必须强制实体化。** Jittor 工厂是惰性的，同步只冲刷被索取的工作；先读
   数会把源张量的分配算到下一步头上。

## 归因纪律

sweep 报出来的多数不是缺陷，先证再改：

- **float64/int64 误差约 1e-8**：几乎一定是 `auto_convert_64_to_32=1` 的构造期
  窄化（KI-DTYPE-002）。在 `jt.flag_scope(auto_convert_64_to_32=0)` 下复跑，
  逐位相等就说明运算本身没问题。
- **形状对不上**：先查签名默认值。`jt.norm` 默认 `dim=-1`（按行），NumPy 默认整体
  Frobenius；这是 API 约定差异，不是缺陷。
- **bitwise 系列吃到浮点**：会漏出原始 C++ 编译错误。属报错质量问题（2.19），
  不是静默错误。
- **logical 归约吃到浮点是合法的**（非零即真），公开的 `jt.all`/`jt.any` 正确。
