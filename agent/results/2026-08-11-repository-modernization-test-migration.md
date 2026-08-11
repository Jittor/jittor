# Stage 5 测试外移与制品验收

日期：2026-08-11

## 验收边界

Stage 5 将仓库测试从安装包 `jittor.test` 外移到根 `tests/`，并以 Stage 4 final
wheel 作为 accepted baseline。构建使用 Python 3.11.15、build 1.3.0、setuptools
83.0.0 和 wheel 0.45.1，所有源码快照、构建目录、安装目录、缓存与日志均位于
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/`。

Stage 4 accepted wheel 的 SHA-256 为
`b42ff0b1dacf81244280dfca3e855f86ad4d9955dacbd18f3da1c8a44aa1d1a2`，含
1,057 个成员。其精确成员与哈希已固化为
`agent/baselines/wheel-contents-stage4.txt`。

工作树中 `python/jittor/src/var_holder.cc` 与迁移后的 `tests/core/test_setitem.py`
只有用户已有的 EOF 换行差异；final 构建快照使用两者迁移前的 accepted blob，
没有把这些未暂存差异计入 Stage 5 制品。

## 三类制品

| 制品 | SHA-256 |
| --- | --- |
| direct wheel | `534259e1f32c07f5d9f0ef085a8d5c5ffb26ad62c46a52c5fba9d381f1657d3c` |
| sdist | `6bd4ba1df5a14df9af7c89232d8e17438ae539fa6c6c32ac2618fca51239d26c` |
| sdist-derived wheel | `0c215622cfbce5a8af61af1bebd39f22b10911fd6312ac439a8aa9d62224e92b` |

direct wheel 与 sdist-derived wheel 的 zip 容器哈希因归档元数据不同，但二者均有
762 个成员，精确路径集合和每个成员的 SHA-256 完全一致：0 additions、0 content
changes、0 removals。

wheel 构建使用 Stage 4 提交时间作为固定 `SOURCE_DATE_EPOCH`；从同一最终索引二次
重建后，direct wheel 与 sdist-derived wheel 的容器 SHA-256 均保持不变。setuptools
sdist 的 tar 成员时间仍取实际构建时间，表中记录的是最终验收构建的容器哈希。

sdist、direct wheel 和 sdist-derived wheel 均未包含根 `tests/`、旧
`python/jittor/test` / `jittor/test`、`__pycache__`、`.pyc` 或
`jittor_fsdp2`。

## 精确差异策略

相对 Stage 4 accepted baseline，Stage 5 final wheel 的差异为：

- 2 additions，精确哈希记录于 `wheel-additions-stage5.txt`：
  `jittor/selftest.py` 和 `jittor/src/utils/test.h`。
- 9 content changes，精确候选哈希记录于
  `wheel-content-changes-stage5.txt`。
- 297 removals，全部严格位于 `jittor/test/`，逐路径记录于
  `wheel-removals-stage5.txt`。

`agent/scripts/check_wheel_contents.py` 的默认 reference 已更新为 Stage 4
baseline，默认 addition/content-change policy 已更新为 Stage 5。removal 继续按门禁
原有设计显式传入，避免大规模删除被隐式接受。direct wheel 和 sdist-derived wheel
均完整消费 `2 + 9 + 297` 项 allowance，unexpected 与未消费 allowance 均为 0。

## 隔离安装

direct wheel 通过 `pip --target --no-deps` 安装到仓库外目录，并从非源码目录以
`PYTHONNOUSERSITE=1` 和独立 HOME/Jittor cache 运行：

```text
Jittor self-test passed (cpu): forward=(1.0, 4.0, 9.0), gradient=(2.0, 4.0, 6.0)
```

导入来源确认为隔离安装目录下的 `jittor/__init__.py`；
`jittor.selftest` 可发现，`jittor.test` 不可发现，安装树中也不存在 `jittor/test/`。

## 验证记录

测试系统本身的最终验证结果：

- `pytest --collect-only -q`：3094 项、237 个产生 nodeid 的模块、0 collection error；
- 完整 `tests/structure`：107 passed、2 skipped；与 checker 单测合并运行时为
  116 passed、2 skipped；
- CPU 迁移代表组：20 passed、15 个硬件条件 skip；
- 自定义算子现行 dtype/data 契约：CPU 2 passed；RTX 4090（GPU 1）CUDA
  flags 与自定义算子 2 passed；
- `tests/compiler/test_utils.py`：TEST、TEST_LOG+LOG_ASYNC、mwsr LOG_ASYNC
  三种 C++ 组合全部通过，异步日志用两个独立 cache 额外稳定性重跑两次；
- source checkout 与隔离 wheel 的 `python -m jittor.selftest` 均通过；
- Python 3.7 AST：300 个测试 Python 文件、0 语法错误；
- AST contract：零 test-to-test import，并禁止 collection 阶段导入 Torch/Triton、
  安装 shim、创建缓存、执行编译/子进程/动态代码或修改 Jittor backend flags。

pytest 现在由 `pyproject.toml` 统一配置 strict markers、strict config 和 strict xfail；
nox 在 CPU 会话先执行全树 collect gate，每个执行单元有 600 秒 timeout，并提供
CPU、CUDA、NPU、ROCm 和 MPI 独立入口。

- 暂存索引导出的 direct wheel + sdist build：`stage5-index-reviewed/build.log`
- sdist-derived wheel build：`stage5-index-reviewed/sdist-wheel.log`
- exact direct/sdist wheel comparison：`stage5-index-reviewed/direct-vs-sdist.log`
- Stage 4 policy checks：`stage5-index-reviewed/direct-check.log`、
  `stage5-index-reviewed/sdist-check.log`
- artifact hashes：`stage5-index-reviewed/sha256.txt`
- exact-index wheel 隔离安装/selftest：`stage5-index-reviewed/install.log`、
  `stage5-index-reviewed/selftest.log`
- checker unit tests、syntax、diff check：`stage5-final-checker-tests.log`、
  `stage5-final-checker-syntax.log`、`stage5-final-diff-check.log`
- final collection/structure：`stage5-final-collect.log`、
  `stage5-final-structure.log`
- compiler/CPU/source selftest：`stage5-compiler-utils-final.log`、
  `stage5-cpu-smoke-retry.log`、`stage5-selftest-run.log`

以上日志均位于
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/`。wheel checker 的 9 项
单元测试、Python syntax 与 `git diff --check` 均通过。
