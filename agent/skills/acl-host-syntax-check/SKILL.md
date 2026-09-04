---
name: acl-host-syntax-check
description: 在没有 CANN/NPU 的开发机上对 python/jittor/extern/acl/** 的 C++ 改动做真实的 TU 语法检查。改了任何 ACL 后端代码（executeOp 尾部迁移、BaseOpRunner、aclops/*_acl.cc、acl_jittor.h）在提交前都要用它；也用于判断「这个 ACL 改动在本机能验到什么、验不到什么」，避免把只跑过静态字符串合同的改动说成验证过。
---

# 在无 CANN 的机器上语法检查 ACL 后端

ACL 后端在开发机上**编译不到**：没有 CANN，`acl/acl.h` 与 180 多个 `aclnnop/*.h` 都不存在。
后果是 `python/jittor/extern/acl/**` 的 C++ 改动一直**没有任何东西解析过它**——静态合同只是对源码做
字符串匹配，一个拼错的标识符、一个少写的括号、一个传错的 launcher 都能一路绿到实机。

本 skill 用**生成的 CANN 桩头文件**加真实的 Jittor 核心头文件跑 `g++ -fsyntax-only`，把这一层补上。

## 怎么跑

```bash
# CFG 是 jittor.compiler 的 cache_path（放着生成的头文件），取法见下
CFG=$(PYTHONPATH=<worktree>/python JITTOR_HOME=<你的 JITTOR_HOME> \
      python -c "import jittor.compiler as c; print(c.cache_path)")

cd <worktree>/agent/skills/acl-host-syntax-check
python syntax_check.py \
  --repo <worktree> \
  --jittor-cache "$CFG" \
  --python-include <env>/include/python3.11 \
  --stub <你的 TMPDIR>/cann-stub \
  --check-launchers \
  python/jittor/extern/acl/aclops/*_acl.cc python/jittor/extern/acl/*.cc
```

退出码 0 表示全过。每个源文件打一行 `ok` 或 `FAIL` 加具体诊断。
全树（43 个源文件）约 40 秒。

`cache_path` 只在 `import jittor` 成功后才有；**第一次会重编核心**，而且 jittor 常会要求
「rerun the same command」——照做，第二次就有了。手写命令记得带 `PYTHONPATH=<worktree>/python`，
否则取到的是主树的 cache_path。

## 它能证明什么，不能证明什么

**能**：解析错误、未声明的标识符、Jittor 侧 helper 的参数个数/类型错误、`op_attr` 的
`dynamic_cast` 目标类型写错、以及（配 `--check-launchers`）**把 workspace 查询函数当 launcher
传给 `BaseOpRunner::launch`**。

**不能**：`aclnnXxxGetWorkspaceSize` 的实参对不对。每个算子的查询签名没有 SDK 就无从得知，
桩里声明成 variadic，所以传错张量个数、顺序、类型**照样绿**。这一层只有 910B3 实机编译能挡。

**不要**把本检查通过说成硬件验证。它是源码级检查，实机验收另算。

## 两个必须知道的坑

**坑 1：`-fsyntax-only` 单独跑挡不住传错 launcher。**
`launch()` 收的是 `std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>`。
桩里的查询函数是 variadic，而 variadic 函数**可以**转换成这个 `std::function`——于是
`launch(ret, aclnnSWhereGetWorkspaceSize, true)` 这种真错误编译通过。实测确认过。
`--check-launchers` 另起一个 TU，对每个 `launch(ret, X, ...)` 站点断言
`std::is_same<decltype(&X), AclExecuteAbi>`；比较**裸函数指针类型**才能拒绝 variadic。
**改了 launcher 相关的东西一定要带 `--check-launchers`。**

**坑 2：`acl_jittor.h` 在桩下必然报错，这是预期的。**
`AclOpFunctions` 有 40 个 `std::function` 构造重载（正是 8.06 要做类型擦除的那个胖结构）。
variadic 桩对每个重载都可转换，于是 `aclOpFuncMap` 那张表每一行都报 ambiguous。
脚本按**文件名**过滤掉 `acl_jittor.h` 的诊断，其它任何文件的诊断都算失败。
副作用：g++ 报错后继续做语义分析，但错误恢复**可能吞掉后面的个别诊断**。所以本检查是下界。

## 判据：怎么确认这次检查真的在起作用

跑绿了不算数，先做一次**反向对照**——故意改坏再跑，必须变红：

```bash
# 对照 1：把查询函数传给 launch（--check-launchers 才挡得住）
sed -i 's/launch(ret, aclnnSWhere, true);/launch(ret, aclnnSWhereGetWorkspaceSize, true);/' \
  python/jittor/extern/acl/aclops/where_op_acl.cc
# 期望：FAIL launcher ABI check，退出码 1

# 对照 2：文件里加一个不存在的符号
# 期望：FAIL <该文件>，退出码 1
```

改完记得还原（用 `git diff` 确认工作区干净，别靠记忆）。

## 主机差异

- `__fp16` 是 aarch64 的内建类型，CANN 机器都是 aarch64。x86_64 复核机上脚本自动加
  `-D__fp16=_Float16`，否则 `binary_op_acl.cc` 的 fp16 分支根本解析不了。
- `extern/cuda/inc/helper_cuda.h` 里 `findCudaDevice` 调的 `checkCmdLineFlag` 只在 `IS_CUDA`
  定义时才随 `helper_string.h` 引入，所以脚本固定加 `-DIS_CUDA`（连同 `-DHAS_CUDA -DIS_ACL`）。
- 需要 `/usr/local/cuda/include` 存在（`executor.h` 直接 include `cuda_runtime.h`）。

## 桩是怎么来的

`make_cann_stub.py` 扫 `python/jittor/extern/acl/**` 自动生成，不需要手工维护清单：

- 所有 `aclnn*` 标识符（**不只是调用点**——注册表里 `AclOpFunctions(aclnnAbsGetWorkspaceSize, aclnnAbs)`
  是当值用的，只匹配 `aclnnX(` 会漏掉一半）；`*GetWorkspaceSize` 声明成 variadic，
  其余按真实的四参数 execute ABI 声明；
- 所有 `ACL_ERROR_*` 名字（`acl_error_code.cc` 用到几百个），值取互不相同的负数；
- 每个被 include 的 `aclnnop/*.h` 生成一个只 include 总表的转发头。

**加了新算子之后不用改桩**，重跑即可。桩生成在 `--stub` 指定的目录（放 `$TMPDIR` 下，别放仓库里）。
