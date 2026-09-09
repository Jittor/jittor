# 公开接口覆盖面的首次测量

- Status: 已测量；棘轮基线待建
- Date: 2026-09-09
- Baseline: `f2692a2eb`
- Owner: 测试基础设施维护者
- Review when: OpInfo 数据库、公开面清单或覆盖工具的口径变化

## 起因

一次随手的 REPL 会话就撞出 `Var.device` 不存在、惰性 `x.cpu().cuda()` 静默留在
主机两个缺陷。问题不是这两个，而是**为什么整套测试都没发现**。

## 三个测量

### 1. 算子面 vs 数据库

公开可调用面 **1294** 个名字（`tests/structure/public_api_manifest.json`），
OpInfo 数据库 **231** 个条目。差集不是"还没来得及"。

### 2. 真实调用覆盖

`tests/_helpers/api_coverage.py`，`JITTOR_API_COVERAGE=1` 启用。包住每个公开入口，
跑测试，看哪些包装器触发过。

清单的 1294 个名字里 **1099 个可包装**，195 个不可（类、非可调用属性）——
不可包装的**单独记账而不是丢弃**，分母里少算一个入口会美化结果。

已用开/关对照验证不改变被测系统：`tests/ops/test_where_op.py` 两侧同为 18 passed。
该文件的 18 条测试触及 1099 个入口中的 **6** 个。

全量套件的数字尚未取得（需要在 JIT 构建锁空闲时单独跑一轮），它是棘轮的基线。

### 3. dtype 缺口

`tools/opinfo_dtype_gaps.py`，逐个 (算子, dtype) 直接构造输入探测。

```
OpInfo 条目 231 个
跑得通却不在声明集合里的 (算子, dtype) 组合   1019 对
未能探测                                        0
```

按 dtype：

| dtype | 未测组合 |
| --- | --- |
| bool | 157 |
| float16 | 144 |
| int32 / int64 | 136 / 136 |
| uint8 / int8 / int16 | 135 / 134 / 134 |
| float64 / float32 | 23 / 20 |

`unprobed=0` 意味着每个条目都答出了 arity，分母没有悄悄丢掉难例。

## 这个数字是什么，不是什么

**1019 是覆盖缺口的上界，不是缺陷数。** 抽查最高风险的 bool 列——
`bitwise_and/or/xor`、`logical_and/or/xor/not`、`all`、`any`、`abs`、`add`
共 11 项对 NumPy 参考——**全部正确**。

`bitwise_not` 之所以独中（对 bool 恒返回 `True`），是因为 C++ 的 `~` 对提升后的
整数产生非零值（`~true = -2`、`~false = -1`，两者转回 bool 都是 True），而
`&`、`|`、`^` 保持 0/1。同族其余算子不受影响。

所以这个数的用法是：**它是应当单调收缩的分母，以及"该往哪看"的指路牌**，
不是一张缺陷清单。

## 已证实的三个形态

**算子面远大于数据库。** 1294 对 231。

**"被覆盖"不是二值的。** `bitwise_not` 全程有 OpInfo 条目，条目声明
`dtypes=_INT`（`integral_types()` 不含 bool），样本生成器又写着
`dt = _int_dtype(dtype)` 把传入 dtype 强制转成整型——于是没有任何测试**能**
喂给它 bool。另：`OpInfo.dtypes` 缺省为 `floating_types()`，忘记声明的条目
静默只测浮点。

**按文本统计覆盖率是自欺。** 1294 个名字每一个都在 `tests/` 下某处出现过，
那个口径给出约 96%。

## 未做

- 全量套件的真实调用覆盖数字，以及基于它的棘轮（未覆盖集只许缩小）；
- dtype 缺口的分诊：1019 对里哪些应当补测、哪些应当在条目里写明不支持的理由。
  照 `gate_scope.EXCLUDED` 的成例，缩窄应当带上说明而不是默认沉默。
- dtype 探测按 (算子, dtype) 逐个试编译，属 nightly 级诊断，不适合放进 PR 门禁。
