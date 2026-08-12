# 仓库结构现代化最终验收

日期：2026-08-12

## 结果

阶段 0 到阶段 8 的仓库内改造已经完成。仓库现在以 `pyproject.toml`、Nox、
分层 CI、根层 `tests/`、`benchmarks/`、`examples/`、`tools/` 和唯一 `docs/`
为工程边界；运行时领域实现收敛到 `jittor.nn`、`jittor.misc`、`jittor.pool`、
`jittor.optim` 与 `jittor.compat` 的规范包结构。

这轮终验继续删除了根层 `python/jittor/attention.py`。它的实现和旧公开符号现在都由
`jittor.nn.attention` 唯一拥有，`jittor.attention` 仅由中央 alias registry 发布为同一
module object，旧 pickle 路径仍可解析。以下名称在 checkout、sdist 和 wheel 中都没有
同名物理文件或目录：

- 根目录 `jittor_fsdp2/`；
- `python/jittor/torch_fsdp2_compat.py` 与 `torch_fsdp2_compat/`；
- `python/jittor/attention.py`；
- `_nn/`、`_misc/`、`_pool/`、`_torch_compat/`、`_torch_fsdp2/`；
- `torch_shim/`、`triton_shim/`、`monkeypatch_ops.py`、`optim.py`；
- 运行时中的 `script/`、`demo/`、`notebook/`、`vcompiler/`、`version` 与
  `extern/llvm/`。

旧 import spelling 只允许作为 `python/jittor/compat/_aliases.py` 中的兼容字符串，
不允许重新成为物理实现。结构测试还锁定了 `python/jittor/` 的全部直接子项；以后新增
任何未经审阅的根层文件或目录都会失败，而不是只检查一组已知坏名字。

根层保留的 Python 文件已逐个审计：`compiler.py`、`compile_extern.py`、
`pyjt_compiler.py` 和 `init_cupy.py` 是运行时编译/引导边界；`contrib.py`、
`distributions.py`、`init.py`、`linalg.py`、`lr_scheduler.py`、`sparse.py` 与
`weightnorm.py` 是实际公开实现；`selftest.py` 是安装后自检。它们不是重复 facade。
其中 `lr_scheduler.py` 保留五个历史调度器，`optim/schedulers.py` 则拥有新的
`LRScheduler`/`LambdaLR`，两者公开能力不同。

## 终验修正

- sdist inventory 不再把 Git 索引中已删除、工作树中不存在的文件误判为必需成员；
  对应 deleted-worktree 回归测试已加入。
- wheel 基线固定为 786 个逐成员 SHA-256；相对 Stage 7 的 40 项新增、27 项内容变化和
  3 项删除均有精确 allowlist，最终默认 allowlist 为空。
- CPU oracle 必须来自 `REAL_TORCH_SITE` 下的真实 `torch._C` 二进制；部署的 Jittor
  Torch shim 不再能被误认成 PyTorch。CPU CI 显式安装 PyTorch 2.7.1 CPU wheel 并以
  `JITTOR_REQUIRE_REAL_TORCH=1` fail closed。
- 六条 cuDNN 2D/3D forward/backward 路径的 math mode 纳入 JIT cache key；严格 FMA
  只在 cuDNN 8+ 启用，并用 `IS_ROCM` guard 保持 MIOpen/旧 cuDNN 可编译。

## 验证

| 门禁 | 结果 |
| --- | --- |
| Ruff / format / Mypy | 通过；55 个 format ratchet 文件，7 个 mypy 源文件 |
| Python 3.7 | 真实 Python 3.7 compile 通过，620 个 Python 文件 |
| 结构门禁 | 219 passed，2 个无 CUDA 环境 skip；包含根层精确清单、物理路径、alias identity、pickle、sdist/wheel 契约 |
| packaging | sdist 1084 个成员；direct 与 sdist-derived wheel 各 786 个成员；安装后 `jittor.selftest` 通过 |
| CPU | 维护子集 124 passed / 14 skipped；独立 PyTorch 2.12.1 oracle 18 passed / 2 skipped |
| CUDA | 全量门禁 220 passed / 3 skipped / 4 xfailed；最终 cuDNN FP16/BF16/TF32、2D/3D、forward/backward 定向组 9 passed |
| 文档与教程 | 严格英文/中文 Sphinx、gettext freshness、链接和离线 notebook 门禁通过 |
| ASV | matmul 7.36 ms、softmax 8.77 ms、LayerNorm 2.93 ms、GELU 1.60 ms；optimizer 扩展性结果写入 commit 结果集 |
| Gaussian Splatting | train/render/metrics exit 0；SSIM 0.0095421、PSNR 6.7422075、LPIPS 0.6262948 |
| TRELLIS.2 | cold/hot/aligned 均 exit 0；PBR GLB 38,638,084 bytes；aligned 7.4999 s，对 Torch 7.4661 s，ratio 1.0045；无 fast-math key |

原始证据保存在仓库外：

- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/final-66bc9a14/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/gs-final-66bc9a14-20260812/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/trellis-final-e2e-66bc9a14-d10ed75-20260812/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/final-98951d54/`

## 外部边界

本机没有 CANN/Ascend 设备，当前用户也不能访问 Docker daemon，因此 NPU 与两种容器
镜像只完成了 fail-closed CI、配置和静态契约，没有伪报本机运行成功。

ms-swift 两处 gap 已分别整理为干净提交 `cc127f467f70` 和 `7e254520bfd0`，补丁位于
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/ms-swift-patches/`。当前环境没有
`gh`、GitHub token 或上游写权限，无法代替维护者发布外部 PR；这是计划中唯一仍需外部
凭据完成的动作，不影响 Jittor 仓库内的结构与运行时交付。
