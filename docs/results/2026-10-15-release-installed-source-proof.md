# Release scorecard 的实际安装来源校验

- 状态：工具层缺口已修复；不授予新的模型或设备覆盖。
- 日期：2026-10-15。
- 基线：`97d7d478a48ecc80f13f2858f0bc8fa78e377480`；本报告同提交包含修复。
- 维护者：Jittor core maintainers。
- 复查条件：库存生成、wheel 绑定、测量元数据或统一审计准入发生变化。

旧审计对普通 release 只比较版本字符串，未校验测量中的实际安装源码。
最小 host 反例中，缺失绑定和源码不符的两条记录仍获 L2。修复后两条均拒绝；
原始反例与修复结果分别保存，不重写旧结果。

全部注册表库存现记录完整 Python 源码及全部 tracked 包文件，轴与分母保持原义。
统一审计要求 base、perturb、training 两端的源码绑定覆盖自身及被引用库。
非 Python 源码可以作为更强的来源清单；声明的额外源码必须匹配库存。
有 wheel 时独立核对元数据及完整安装内容，能拒绝 Python 全齐但漏 CUDA/data 的记录。
既有 SCM 的专用严格绑定与 protocol 校验继续保留。

只读复验真实 Swift 原始 CPU/CUDA、三 seed、两 runtime 共 12 份元数据，
517 Python 源码与 wheel 全部 532 包文件通过。此 wheel 由固定上游源码自行构建，
并非声称官方 PyPI wheel；原测量源码仍为 `606b602e7e79b5bfa5573b9693144f8d224c6a07`，
本次读取不重新授予该训练的数值或设备结论。

原始反例、只读复验清单及测试日志位于未版本化的
`$JITTOR_LAB_ROOT/_state/release-source-proof-97d/`。测试入口：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  tests/tools/test_downstream_coverage.py tests/tools/test_installed_source_proof.py \
  tests/tools/test_component_scorecard.py tests/tools/test_accelerate_protocol_scorecard.py
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

历史记录缺少实际安装来源时不能在当前严格审计下进入分子。旧数值证据保留原始
基线，必须重新运行采集完整来源，不能从当前安装目录补填。
