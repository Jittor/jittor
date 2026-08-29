# Python 3.13 与 NumPy 2 wheel 兼容验证

- Status: Accepted for the maintained Python 3.13 CPU delivery gate
- Last reviewed: 2026-08-29
- Source baseline: `08caacc8` plus the changes documented here
- Owner: build, packaging, and core-runtime maintainers
- Review when: Python, NumPy C ABI, packaging dependencies, or upper-version CI changes

## 结论

Jittor 2.0 当前源码可在真实 Python 3.13.13 与 NumPy 2.4.6 上编译核心、执行
CPU forward/backward、传输连续与非连续 NumPy 数组，并启用 `trace_py_var`。软件包
NumPy 上界由 `<2` 放宽为 `<3`，Python 3.13 加入发布 classifier、Nox 和 CI 门禁。

首次最小复现并未通过：核心编译到 `py_var_tracer.cc` 时，Python 3.13 不再声明
私有 `_PyObject_LookupAttr`。修复按版本使用 3.13 的公开
`PyObject_GetOptionalAttr`，旧 Python 继续使用原接口，并保留非 `AttributeError`
失败的显式报错。

## NumPy 2 复核

旧兼容文档仍声称 NumPy 2 存在偶发堆损坏，但该结论早于已合入的三项 ABI 修复：
版本化 `PyArray_CopyInto` 槽位、移除伪造的旧 dtype descriptor，以及从 Jittor
dtype 推导复制字节数。本轮在当前提交重新执行 5000 次压力循环，覆盖 13 个 dtype、
C 连续、Fortran 连续和步长切片输入、`jt.array`、`.numpy()`、行索引、同步和回收，
结果全部通过。正式 3.13 门禁固定使用 `numpy>=2.1,<3.0`。

## 验证结果

| Gate | Result |
| --- | ---: |
| Python 3.13.13 repository syntax scan | passed |
| Source CPU core JIT | 153/153 compiled |
| Source `jittor.selftest` | passed; forward `(1, 4, 9)`, gradient `(2, 4, 6)` |
| `tests/compiler/test_trace_var.py -m 'not network'` | 5 passed, 1 deselected |
| NumPy 2.4.6 transfer stress | 5000/5000 passed |
| Installed-wheel `py313` Nox gate | passed on Python 3.13.13 / NumPy 2.5.2 |
| Existing Python 3.12 Nox gate | passed on Python 3.12.14 / NumPy 1.26.4 |
| Repository structure suite | 219 passed, 2 skipped |
| Focused delivery/environment structure checks | 10 passed |
| Clean-worktree layout and documentation governance | passed |

所有 JIT 缓存、wheel、中间源码副本和原始运行文件均位于
`$JITTOR_LAB_ROOT/_state/python313/` 或 `$JITTOR_LAB_ROOT/_state/nox/`，未写入主仓库。
本报告只声明 Python 3.13 的 CPU 基础交付兼容，不替代 CUDA、ROCm 或 NPU 在相应
Python 版本上的真实设备门禁，也不扩大厂商运行时自身的 Python 支持范围。
