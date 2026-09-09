# Python 3.12 wheel 与 Nox 解释器隔离验证

- Status: Accepted for the maintained Python 3.12 gate; repository-structure
  source tests pass apart from external workspace contents recorded below
- Last reviewed: 2026-08-31
- Source baseline: `91e64815`
- Owner: Build and test infrastructure maintainers
- Review when: supported Python versions, Nox environments, wheel self-test, or
  external hardware interpreters change

## 结论

Jittor 2.0 当前源码已在 Python 3.12.14 下完成语法检查、wheel 构建安装和真实
CPU JIT 自测。自测从独立缓存编译 154 个核心单元，最终前向结果为
`(1.0, 4.0, 9.0)`，梯度为 `(2.0, 4.0, 6.0)`。

验证同时修复了 Nox 宿主解释器与 session 解释器不一致时的 ABI 配置污染。
修复前，由 Python 3.12 启动的 Nox 会给 Python 3.11 `structure` session 注入
Python 3.12 的 config helper 和头文件，生成 `cpython-312` 扩展并导致 10 个
测试模块收集失败。现在虚拟环境和外部硬件环境都从实际执行解释器解析
`python-config`；跨版本结构门禁生成 `cpython-311` 扩展并完整通过。

## 实现边界

- 普通 Nox session 从其虚拟环境中的 `python` 解析 config helper，不再读取
  Nox 宿主进程的 `sys.version_info`。
- CUDA、ROCm、NPU、MPI、NCCL 和 benchmark session 从
  `JITTOR_CI_PYTHON` 指向的外部解释器解析并校验 config helper。
- 不需要 Jittor 编译的 session 在找不到 helper 时清除继承值，避免把宿主 ABI
  泄漏到子进程。
- 新增结构测试覆盖虚拟环境覆盖错误继承值、缺失 helper 时清理继承值，以及
  外部硬件解释器覆盖宿主配置。

## 验证结果

| Gate | Result |
| --- | ---: |
| `nox -s py312` on Python 3.12.14 | passed in 8 minutes |
| Python 3.12.14 syntax scan | 701 files passed |
| Python 3.12 wheel build and isolated install | passed |
| Installed-wheel CPU forward and gradient self-test | passed; 154 core units compiled |
| NumPy/Fortran array and `trace_py_var` probe | passed with NumPy 1.26.4 |
| Built wheel SHA-256 | `4f77f3f70cc52cfa72440d19354181d1eaf4b030a82601397c47483f8f045fb2` |
| Direct `tests/structure` source coverage | 226 passed, 2 skipped |
| Current-checkout structure failures | 2 external-worktree documentation scans |

首次 JIT 和完整结构门禁均使用独立 `JITTOR_HOME` 串行执行。缓存、虚拟环境和
临时 wheel 安装位于 `$JITTOR_LAB_ROOT/_state/nox/`，没有写入主仓库。

当前 checkout 的结构门禁还会扫描仓库内 ignored 的 `.claude/worktrees`，因此
`test_documentation_governance_checker` 和
`test_one_canonical_documentation_tree_remains` 失败；布局脚本还报告用户的
`TODO.md` 和旧 `python/jittor.egg-info`。这些文件不属于当前 Git 提交，本轮没有
删除或修改。该工作区状态不影响从受控源码副本构建、安装和执行的 Python 3.12
门禁，但也没有被本报告声明为布局通过。

## 复查范围

本报告只证明 Python 3.12 基础交付门禁和 Nox 解释器隔离，不替代 CUDA、ROCm、
NPU 的真实设备执行，也不声明 Transformers、TRELLIS、MMCV、verl、vLLM、
Diffusers 等下游项目的完整 Python 3.12 兼容性或性能结论。
