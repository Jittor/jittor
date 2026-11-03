---
name: torch-shim-noop-audit
description: 核查 Torch API 是否实际完成承诺的计算或状态改变，区分合法 identity、限制和误导性 noop。
---

# 查找 compat 中的空操作

## 操作

- 从当前 `compat/` 的导出与实现定位入口，不只扫已迁走的旧 shim 路径。
  `pass`、返回 self 或常量只是候选，合法 identity 与 unsupported stub 需要按 API 合同判断。
- 为候选构造能够观察差异的输入：非默认 device/dtype、非平凡 shape、状态前后变化、
  参数更新或异常条件。输入本来无需改变时，noop 也会通过。
- 与独立真实 Torch 比较行为、返回值和对象身份；环境/权限类 API 明确各自平台条件。
- 不支持的功能应给出可辨识错误或公开限制，不能保持完整签名却默默不做事。
- 将通用缺陷修在 compat 或核心；第三方专属路径按[下游分流](../downstream-library-adaptation/SKILL.md)。

确认缺陷后增加最小行为回归和必要的结构归属检查。不用名字存在或导入成功推断支持完成。
