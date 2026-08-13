# 仓库结构现代化最终验收

日期：2026-08-13（最终工作树复核）

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

根层 Python 文件的后续复审发现初次结论不完整：`contrib.py` 含重复定义和多个领域，
`weightnorm.py` 与 Torch 兼容实现重复，`other/` 与 `gradfunctional/` 的物理归属含糊。
这些项目在后续根目录收敛阶段迁入 `jittor.misc`、`jittor.pool`、`jittor.nn` 和
`jittor.autograd`，旧 import 由中央 alias registry 保持同对象兼容。
其中五个历史调度器迁入 `optim/legacy_schedulers.py`，新的
`LRScheduler`/`LambdaLR` 继续由 `optim/schedulers.py` 拥有；旧
`jittor.lr_scheduler` 是前者的同对象别名。COO 稀疏张量与稀疏卷积也收敛到
`jittor.sparse` 包，旧 `jittor.nn.sparse` 指向规范卷积模块。

根层剩余的九个 Python 文件也逐项锁定归属：`__init__.py` 负责运行时组合；
`compiler.py`、`compile_extern.py`、`pyjt_compiler.py`、`init_cupy.py` 是编译器或
设备引导边界；`distributions.py`、`init.py`、`linalg.py` 是原生公开领域；
`selftest.py` 是安装后自检入口。结构门禁会拒绝任何未经复审的新根层文件。

全树 AST 复审同时消除了兼容 entry-point 发现、编译 flags 解析和 Torch 命名空间
快照的同职责重复。保留的相同代码只限于有独立加载边界的部署命令、ACL 代码生成模板、
模型局部构件以及新旧 PyTorch checkpoint 读取器，并由显式结构白名单锁定；出现新的
跨文件整段复制会失败。

## 终验修正

- sdist inventory 不再把 Git 索引中已删除、工作树中不存在的文件误判为必需成员；
  对应 deleted-worktree 回归测试已加入。
- 初次终验 wheel 基线为 786 个逐成员 SHA-256；根目录复审删除 9 个旧物理入口、
  新增 14 个规范实现后，稳态基线更新为 791 项。直接构建和 sdist 派生 wheel 的
  每项哈希一致，最终默认 allowlist 仍为空。
- CPU oracle 必须来自 `REAL_TORCH_SITE` 下的真实 `torch._C` 二进制；部署的 Jittor
  Torch shim 不再能被误认成 PyTorch。CPU CI 显式安装 PyTorch 2.7.1 CPU wheel 并以
  `JITTOR_REQUIRE_REAL_TORCH=1` fail closed。
- 六条 cuDNN 2D/3D forward/backward 路径的 math mode 纳入 JIT cache key；严格 FMA
  只在 cuDNN 8+ 启用，并用 `IS_ROCM` guard 保持 MIOpen/旧 cuDNN 可编译。

## 最新工作树验证（2026-08-13）

本节覆盖本次继续整理后的 checkout，而不是沿用上一轮报告中的旧条目数。

| 门禁 | 最新结果 |
| --- | --- |
| 结构/布局与 checker 定向组 | 42 passed；根层生成的 `build/`、`dist/` 已清理 |
| 全量测试 collection | 3268 tests collected |
| CPU 维护门禁（`noxfile.py` 的 `CPU_TESTS`） | 108 passed / 13 skipped |
| 独立真实 PyTorch oracle | 18 passed / 1 skipped；`REAL_TORCH_SITE` 验证了 `torch._C` 来源 |
| FFT/complex/einsum + diffusers 视频兼容定向组 | 43 passed / 6 skipped |
| GPT2 tiny forward/backward 对拍 | forward rel 2.33e-7；loss rel 3.59e-7；最坏梯度 rel 1.15e-6 |
| ResNet tiny forward/backward 对拍 | forward rel 1.07e-7；最坏梯度 rel 6.62e-8 |
| ViT tiny forward/backward 对拍 | forward rel 2.85e-7；最坏梯度 rel 7.15e-7 |
| sdist 内容审计 | 1093 members，`check_sdist_contents.py` 通过 |
| direct wheel 内容审计 | 793 members；SHA-256 `7a8e9b4c1ac7b15108e67e2efdf4b8c305026b70b21eaff68016b1769910fde2` |
| sdist-derived wheel 内容审计 | 793 members，和终态 manifest 逐项一致 |

`examples/notebooks/` 中的 Markdown notebook 代码块已经扫描确认使用原生
`jittor` API；没有发现需要从 Torch 改写的代码块。新增 `jittor.fft` 和
`jittor.misc.reductions` 已进入类型桩、结构门禁和 wheel baseline。

## 验证

| 门禁 | 结果 |
| --- | --- |
| Ruff / format / Mypy | 通过；55 个 format ratchet 文件，7 个 mypy 源文件 |
| Python 3.7 | 真实 Python 3.7 compile 通过，620 个 Python 文件 |
| 结构门禁 | 历史基线 219 passed；本次继续复核的完整结构组为 207 passed / 2 skipped（生成缓存清除后） |
| packaging | 历史基线记录保留；本次最新 1093 sdist members、793 wheel members，两个 wheel 均通过默认审计 |
| CPU | 维护子集 124 passed / 14 skipped；独立 PyTorch 2.12.1 oracle 18 passed / 2 skipped |
| CUDA | 全量门禁 220 passed / 3 skipped / 4 xfailed；最终 cuDNN FP16/BF16/TF32、2D/3D、forward/backward 定向组 9 passed |
| 文档与教程 | 严格英文/中文 Sphinx、gettext freshness、链接和离线 notebook 门禁通过 |
| ASV | matmul 7.36 ms、softmax 8.77 ms、LayerNorm 2.93 ms、GELU 1.60 ms；optimizer 扩展性结果写入 commit 结果集 |
| Gaussian Splatting | train/render/metrics exit 0；SSIM 0.0095421、PSNR 6.7422075、LPIPS 0.6262948 |
| TRELLIS.2 | cold/hot/aligned 均 exit 0；PBR GLB 38,638,084 bytes；aligned 7.4999 s，对 Torch 7.4661 s，ratio 1.0045；无 fast-math key |

### 根目录复审增量验证

| 门禁 | 结果 |
| --- | --- |
| 定向接口/结构 | 编译 flags、兼容注册、根领域与清理契约合计 46 passed |
| 完整结构门禁 | 208 passed / 2 skipped；包含精确根目录、alias identity、旧 pickle、类型桩、九个旧模块冷启动表面与全树重复实现审计 |
| 原生 CPU 回归 | 60 passed / 4 skipped；另有 transform 47 passed、sparse 4 passed、weight norm 4 passed / 1 skipped |
| CUDA 定向回归 | `.data` device 写入 1 passed；稀疏卷积 forward/feature/weight backward 1 passed，均在真实 NVIDIA GPU 上执行 |
| packaging | sdist 1090 项；直接与 sdist 派生 wheel 各 791 项且逐项哈希一致；源树外 `jittor.selftest` 及九个旧模块的属性/星号导入通过 |

本次增量证明旧公开模块入口、模块身份、pickle 解析、根类型桩和核心原生 `.data`
语义在本次物理迁移后保持兼容。它不等同于“任意历史私有实现路径”或全部第三方项目
已经完成验收；后者仍按下游对拍和完整测试阶段分别给出证据。当前 CPU/CUDA
对拍覆盖了 tiny GPT2、ResNet、ViT，尚未把所有第三方仓库的完整训练矩阵或每个模型的
性能“不慢于 PyTorch”变成无条件结论。

## 当前边界

- `ms-swift` 自身的 bug 按用户要求不纳入 Jittor 核心修复；主仓库不再携带其专属
  monkeypatch。上游修复和本地验证材料仍保留在仓库外审计目录。
- 当前主机没有 CANN/Ascend，不能宣称 NPU/HCCL 的真实执行通过；Docker daemon 也不可用。
- CUDA FFT 仍保留 [KI-FFT-001](../../agent/manuals/known-issues.md) 的序列敏感风险：
  当前 CPU/compat 组通过，但尚未关闭聚合 CUDA `rfft/irfft` 复现。
- 既有 Transformers 性能记录显示部分小模型/LayerNorm 场景 Jittor 慢于 PyTorch；
  因此本报告只给出已测的数值对拍，不把“速度不会更慢”写成已证明事实。

原始证据保存在仓库外：

- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/final-66bc9a14/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/gs-final-66bc9a14-20260812/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/trellis-final-e2e-66bc9a14-d10ed75-20260812/`
- `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/final-98951d54/`
- `${JITTOR_LAB_ROOT}/_state/verify/jittor-todo/root-cleanup-packaging-final/`
- `${JITTOR_LAB_ROOT}/_state/verify/jittor-todo/root-cleanup-structure/`

## 外部边界

本机没有 CANN/Ascend 设备，当前用户也不能访问 Docker daemon，因此 NPU 与两种容器
镜像只完成了 fail-closed CI、配置和静态契约，没有伪报本机运行成功。

ms-swift 的 GKD native rollout gap 已由上游
[#9818](https://github.com/modelscope/ms-swift/pull/9818) 在提交 `4805d7f` 合并解决，旧本地
补丁已标记为 superseded，不再重复提交。PPO reward/value task type 修复已在 2026-08-12
上游 `main` `d17f031` 上重放为干净提交 `6b1f0b2`；真实 PyTorch 2.12.1、Transformers
5.12.1、PEFT 0.19.1、TRL 0.29.1 环境的新增测试为 `2 passed`，YAPF 0.30、Flake8、
diff check 与从基线 `git am` 后的 tree identity 均通过。补丁位于
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/ms-swift-patches/`，SHA-256 为
`c7c8ecbda91f275ebdf4ce2c35e29ca73d9ef083ccbc7db4650d778141bb9f02`。

ms-swift 自身缺陷的修复和上游 PR 不属于本次 Jittor 仓库整理范围。阶段 4d 的验收边界
是主仓库不再拥有或运行 ms-swift 专属 monkeypatch；该边界已经完成。上述本地 PPO
提交和补丁仅作为未版本化审计证据保留，不再作为计划阻塞项。
