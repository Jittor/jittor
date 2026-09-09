# 当前 AArch64 CPU native/Torch 全量门禁

- Status: Accepted for the complete maintained CPU native and Torch sessions
- Date: 2026-09-01
- Baseline: `e076a9e5`; the native session used runtime-equivalent predecessor
  `36d8cda1` as detailed below
- Owner: Jittor core and test-infrastructure maintainers
- Review when: complete-suite ownership, process-mode isolation, CPU JIT,
  notebook execution, or the runner summary format changes

## 结论

当前维护的 CPU 两进程 inventory 为零失败：native `768 passed, 738 skipped`，
Torch `1591 passed, 536 skipped`，合计 `2359 passed, 1274 skipped`。两次运行均在
AArch64、Python 3.11.15、GCC 10.3.1、NumPy 2.2.6、pytest 7.4.4 下执行，明确
禁用 CUDA 与 ACL，并分别使用独立 JIT cache。

native session 在 `36d8cda1` 上完成；随后发现并修复的 `e076a9e5` 只修改
`tools/run_test_suite.py` 及其 structure 回归，没有修改 runtime 源码，也不属于
native collection。Torch session、standalone structure 和布局门禁均在精确
`e076a9e5` 上完成。

## 结果

| Gate | Result |
| --- | ---: |
| Complete native CPU session | `768 passed, 738 skipped` in `1:45:28` |
| Complete Torch CPU session | `1591 passed, 536 skipped` in `2:21:51` |
| Combined maintained CPU inventory | `2359 passed, 1274 skipped`, zero failures |
| Clean-worktree structure | `232 passed, 2 skipped` in `2:31` |
| Repository layout and documentation governance | passed; 92 Markdown files |
| Runner contract regressions | `21 passed` |
| Changed-file Python 3.7 grammar and Ruff | passed |

两次完整 session 都从对应 detached-worktree cache 名称开始冷编译。native 包含
离线执行的 12 个原生 Jittor notebook topic；Torch 包含聚合 compatibility
子进程、vLLM compatibility、严格 CPU OpInfo、core type/regression、完整 structure
和按 CPU 环境明确 skip 的硬件用例。缓存位于
`$JITTOR_LAB_ROOT/_state/test-suite/{native,torch}/`，clean worktree 位于
`$JITTOR_LAB_ROOT/worktrees/`，均未进入主仓库。

## Runner 修复

维护 runner 以 `pytest -q` 执行，但旧 `_parse_counts` 只接受以 `=` 开头的 summary。
因此 native 实际通过后，runner 仍打印空的 `native`/`combined` 计数。当前解析器
从最后一条包含 pytest count token 的 summary 提取 `passed`、`failed`、`skipped`、
`error`、`xfailed` 和 `xpassed`，同时支持 quiet plain summary 与 decorated summary。
回归覆盖实际 native 输出、带失败/error 的 decorated 输出和最后 summary 优先。
修复后 Torch session 正确打印 `passed=1591 skipped=536`。

## 命令口径

```bash
python tools/run_test_suite.py --session native
python tools/run_test_suite.py --session torch
python -m pytest -q tests/structure
bash agent/scripts/check_repo_layout.sh
```

运行时额外将 clean worktree 的 `python/` 加入 `PYTHONPATH`，并把
`python_config_path` 指向执行解释器自己的 Python 3.11 config helper。runner 自身
设置 `JITTOR_TEST_DEVICES=cpu`、空 `nvcc_path`、独立 `JITTOR_HOME` 与 `TMPDIR`、
`JITTOR_TORCH_SHIM` mode 以及串行首次 JIT。

## 边界

本报告证明当前维护的 CPU native/Torch 全量 inventory，不把 skip 解释为能力通过，
也不替代 CUDA、ROCm、NPU、MPI/NCCL 多进程和可选下游依赖的真实设备门禁。输出中
仍有 NumPy/Pillow/Python deprecation warning；它们未造成当前失败，但需要在对应
依赖升级前单独处理。
