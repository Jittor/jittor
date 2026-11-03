---
name: acl-host-syntax-check
description: 使用生成的 CANN 桩检查 ACL C++ 语法和 launcher ABI；用于缺少目标 SDK 时的静态验证，不证明真实 SDK 签名或设备执行。
---

# 在无 CANN 主机检查 ACL 翻译单元

## 使用

[scripts/make_cann_stub.py](scripts/make_cann_stub.py) 从当前 ACL 源码生成最小桩，
[scripts/syntax_check.py](scripts/syntax_check.py) 编译真实翻译单元并可检查 launcher ABI。
从当前解释器取得 `jittor.compiler.cache_path` 和 Python include，不能选缓存中的“最新目录”。

```bash
python agent/skills/acl-host-syntax-check/scripts/syntax_check.py \
  --repo "$PWD" --jittor-cache "$JITTOR_CACHE" --python-include "$PYTHON_INCLUDE" \
  --stub "$RUN_DIR/cann-stub" --cuda-include "$CUDA_INCLUDE" --check-launchers \
  backends/acl/kernels/native/where_op_acl.cc
```

`RUN_DIR` 在 lab，`CUDA_INCLUDE` 为此次配置实际使用的 CUDA 兼容头文件目录，
`--cxx` 可选择宿主编译器。先用 `--help` 查看参数；编译失败保留完整诊断，不过滤成成功。

## 证明范围

- 桩的 `aclnn*GetWorkspaceSize` 是 variadic：可检查 Jittor 侧符号、语法及类型连接，
  **不能**验证 SDK workspace 查询参数数量和类型。
- execute launcher 的固定函数指针 ABI 需要额外 TU 检查。仅 `std::function` 可转换性会
  接受 variadic 桩，不能识别把 query 误传为 execute。
  需要看到 `ok launcher ABI` 及非零匹配数量；未匹配站点时只完成了 TU 语法检查。
- x86 宿主可能需要将 `__fp16` 映射到 `_Float16` 以解析存储类型；CANN 主机不限定单一架构。
- 目标 SDK 的真实编译、链接与设备数值仍需另行验收。静态调用等价见
  [static-launch-equivalence](../static-launch-equivalence/SKILL.md)。
