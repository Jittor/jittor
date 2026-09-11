# 仓库结构现代化：交付验收报告

> 验收对象：`docs/architecture/repository-layout.md` 所定义的八阶段仓库现代化改造
> 基准：HEAD `dccca5b2`，分支 `2.0`（工作区另有 3 个无关未提交改动）
> 验收方式：静态核查 + 实际构建 wheel/sdist + 实际运行 ruff/mypy + 实际模拟 Docker 构建上下文
> 结论：**主体通过**。八个阶段的实质工作都完成了，其中打包、架构收敛、根因修复三项质量很高。
> 但有 3 项"配置了但没生效"和 1 项当前即失败的门禁需要处理。

---

## 一、总体结论

| 阶段 | 结论 |
| --- | --- |
| 0 目标架构 RFC | 通过 |
| 1 打包与元数据 | 通过 |
| 2 工具链 | **有问题**：配置齐全但覆盖率 8.9%，等于没生效 |
| 2 CI | 部分完成：分层名不副实，有永不运行的 job |
| 2 发布链 / 容器基线 | 通过 |
| 2 ASV 性能追踪 | **有问题**：Jittor 特有的 CSE 陷阱未处理，会产出错误结论 |
| 3 领域包形态收敛 | 通过（`misc/` 例外，见 3.2） |
| 4 兼容层四层分离 | 通过（质量最高的一项） |
| 5 测试外移与 pytest 化 | 通过 |
| 6 杂物清理 | 通过 |
| 7 模块拆分收尾 | 通过 |
| 8 文档工具链与治理 | 通过 |
| 全程验收口径 | **check_repo_layout.sh 当前失败** |

---

## 二、值得肯定的部分

### 2.1 打包：实测无一文件丢失

实际构建 wheel 验证，不是看配置：

- `python/` 下 git 跟踪的 **780 个文件全部进入 wheel**（786 个成员 = 780 + 6 个 dist-info），零缺失
- 含 `__init__.py` 的包源码 40 个、wheel 里 40 个
- 原来掉在 wheel 外的 7 层深文件已回归：
  `jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh`
- 六层通配符 `package_data` 已删除，改为 `MANIFEST.in` 显式清单 + `include-package-data`
- `setup.py` 退化为 6 行纯 shim，与 `pyproject.toml` 无重复定义
- 结构测试 `tests/structure/test_packaging_structure.py:16` 断言 `find_packages` 结果与文件系统上所有 `__init__.py` 目录集合相等
- `agent/scripts/check_wheel_contents.py:34` 把三个深层文件写成硬性 `REQUIRED_MEMBERS`，并用逐文件 SHA-256 卡回归；实测 `compare` 输出 `wheel contents OK`

按 `.dockerignore` 规则复现构建上下文后重新构建，wheel 成员仍是 786，**Docker 上下文零缺失**。

### 2.2 架构收敛：不是目录搬迁

四个下划线私有包（`_nn`/`_misc`/`_pool`/`_torch_compat`/`_torch_fsdp2`）全部消失，四份逐字重复的 `_JittorRuntimeProxy` 与 `preserve_facade_origins` 元数据伪装全部删除，源码零命中，只剩测试中的负向断言。

实际运行验证元数据是真实路径而非伪装：

```
Linear      jittor.nn.modules.linear
Conv2d      jittor.nn.modules.convolution
LayerNorm   jittor.nn.modules.normalization
relu        jittor.nn.functional.activation
MaxPool2d   jittor.pool.layers
```

`nn/` 从原来的 19 文件扩展为 58 文件的三层结构（`modules/` 23、`functional/` 23、`backends/` 3、能力模块 8），facade 91 行且无任何定义，所有实现文件被测试强制 ≤350 行。旧 pickle artifact 靠 unpickling 映射保兼容，而不是靠改 `__module__`。

### 2.3 根因修复：这是全部改动里质量最高的一项

阶段 4c 声称的三个语义问题**确实在框架层面修了，不是删掉下游补丁了事**：

**`stop_grad` 与 `requires_grad_` 语义分离**，在 C++ 层新增 `NodeFlags::_requires_grad_disabled`（`python/jittor/src/node.h:55`），使 `requires_grad_(False)` 可逆而 `stop_grad()` 保持永久：

```156:156:python/jittor/src/var_holder.cc
void VarHolder::set_requires_grad(bool flag) {
```

新标志位在整条反向链路上都有消费方：`op.cc:43` 快照被禁用的输入边、`graph.h:41-155` 四处遍历跳过、`grad.cc:110` 梯度传播判定、`var_holder.cc:119` 跨赋值传递。

**Parameter 元类改为查显式标记位**（`compat/torch/installers/nn.py:31-36`），不再把所有 Var 都报成 Parameter。

**`Var.shape` 可哈希**（`src/misc/nano_vector.h:293` 新增 `__hash__`，保证 `hash(shape) == hash(tuple(shape))`）。

回归锁定在 `tests/core/test_rootcause_semantics.py`（237 行 13 个用例），**实跑全绿**（22.07s）。其中 `test_temporary_freeze_preserves_preexisting_policy_graph` 的注释明写复刻 PEFT `disable_adapter()` 场景。

三个被删除的下游补丁与三项根因修复一一对应，docstring 自述的根因都已消除。

### 2.4 粘合移出：真的移出了仓库

`jittor-trellis`、`jittor-gs`、`jittor-hf-compat` 三个独立发行物在仓库外真实存在，各带 `pyproject.toml`/`src/`/`tests/`/`README.md`，通过 `jittor.module_patches` 与 `jittor.external_backends` 两个 entry-point 组注册。

能力是**下沉而非复制**——适配包反向调用主仓库新 API，例如 `jittor-trellis/src/jittor_trellis/runtime.py:1100` 调 `jittor.nn.rms_norm_cuda.multihead_rms_norm_cuda`。

`flex_gemm` 硬编码已清理：`compat/shim/extensions/readonly.py:22-25` 四个默认字典全空，策略由调用方传入；主仓库对 `FLEX_GEMM_AUTOTUNE` 零命中。残留的 8 处仅为解释性注释。

三个 CUDA kernel 是**真参数化**而非原样搬运：`nn/rms_norm_cuda.py:42` 从形状推导 `num_heads/head_dim`，magic number 11.313708498984761 换成 `math.sqrt(head_dim)`；测试用 heads=3、head_dim=96 验证，直接证伪"仍是 12/128"。submanifold sparse conv3d 真重写为 C++/CUDA 邻居发现 + 单次批量 matmul，AST 测试断言函数体内无循环、无 `.item()`、`matmul` 恰好一次。

### 2.5 测试、清理、文档

- `tests/` 16 个主题分组，250 个测试文件；`python/jittor/test/` 已清空，测试不再进 wheel
- 自制 `_runner.py` 与 `test_skip_l/test_skip_r` 索引语义已消除，`tests/conftest.py` 建立
- `script/`、`demo/`、`notebook/`、`vcompiler/`、`version`、`extern/llvm/` 全部移除；`other/code_softmax.py` 正确保留（CUDA softmax 快路径活代码，`nn/functional/softmax.py:20` 仍在引用）
- 根 `__init__.py` 从 2851 行降到 **196 行**，最大 Python 文件从 8683 行降到 2822 行
- 文档：`doc/` 与 `docs/` 已合并；`recommonmark`/`AutoStructify` 仅剩测试中的禁止性断言；sphinx-intl 接入，40 个 `.po` 文件；`translator.py` 与 `.src.md` 机制废除（残留 0）；jupytext 接入，`md_to_ipynb.py` 删除；三份 README 合一；`CONTRIBUTING.md` 无旧路径引用；`AGENTS.md` 的 cc-connect 段落已清；`project-context.md` 从 769 行降到 132 行

---

## 三、需要处理的问题

### 3.1 P0：`check_repo_layout.sh` 当前即失败

这是 plan 的全程验收口径中明确要求的一条，现在退出码为 **1**：

```
Python bytecode caches must stay outside the checkout.
/home/zy/projects/jittor/tests/compiler/__pycache__
...
```

触发原因是本地 49 个 `__pycache__` 目录。这些文件**已被 `.gitignore:9` 忽略、未被 git 跟踪**，但门禁检查的是文件系统状态而非 git 状态。

更麻烦的是它挂在 pre-commit 且 `always_run: true`（`.pre-commit-config.yaml:51-54`），意味着**任何人在本地跑过一次测试后就无法 commit**，除非先清理全部 `__pycache__`。

建议：该检查改为只看 git 跟踪状态（`git ls-files`），或在 CI 干净 checkout 时才启用，从 pre-commit 的 `always_run` 中摘除。

### 3.2 P0：ASV 的 CSE 陷阱未处理，会产出错误的性能结论

这是唯一会直接导致**错误结论**的问题。仓库自己的方法学规则写在
`agent/skills/jittor-transformers-perf/SKILL.md:24-26`：预分配多个不同输入 slot，
且 `slots >= repeats`，避免 lazy 图把同输入算子 CSE 掉造成假快。三个 benchmark 都没落实：

- `benchmarks/tiny_llama.py:64-75` 老实预分配了 4 个 slot，但计时函数 `:106-113` 永远只用 `self.slots[0]`，`slots[1..3]` 是死代码；且 `:25` 的 `repeat` 上限为 5，即便实现轮换也不满足 `slots >= repeats`
- `benchmarks/operators.py:96-98` 完全没有 slot 机制，每轮在同一对张量上重跑
- `benchmarks/optimizer_step.py:67-71` 写了 `_reset_gradients()`，但只在 `setup` 末尾调用一次（`:61`），计时循环之间从不调用

`docs/performance/benchmarking.md` 通篇未提这条规则，读者无从知道怎么写新 benchmark；`benchmarks/` 也不在 lint 白名单内，死代码不会被发现。

在修复前，ASV 产出的数字不可用于任何结论。

### 3.3 P1：工具链配置齐全但覆盖率不足一成

ruff 与 mypy 都配置了，但真正被检查的文件是硬编码白名单：

- `noxfile.py:92-114` 的 `RATCHET_FILES` 实测 **55 个文件**，仓库有 **620 个跟踪的 `.py`**，覆盖率 **8.9%**
- 实测对比：`ruff check <55 个白名单文件>` 输出 All checks passed；同一套规则跑全仓库 `ruff check .` 得到 **4109 errors**（F405 ×1470、F401 ×785、E702 ×555、F821 undefined-name ×22、invalid-syntax ×2）
- mypy 的 `files` 只列 7 个文件（1.1%），且 `pyproject.toml:82` 开了 `follow_imports = "skip"`，跨模块引用全部退化为 `Any`
- pre-commit 的 ruff `files:` 正则实测只匹配 **11 个文件**，与 nox 的 55 个不一致，35 个 NN 迁移文件 pre-commit 完全不查

结果是 `structure.yml` 的 lint/typing job 基本没有信息量。ratchet（棘轮）本身是合理的推进策略，但需要明确的扩量计划，且至少先让 pre-commit 与 nox 的白名单统一。

### 3.4 P1：声称支持 3.7 但已有 3.9+ 语法进入仓库

```385:391:tests/structure/test_pool_structure.py
        with (
            mock.patch.object(pool_facade, "Pool", FakeCore),
```

括号形式的多 context manager 是 Python 3.9+ 语法。在 3.7 上会被解析成一个 tuple，因此 `compile()` 能通过、运行时才抛 `TypeError`。`:409` 同样。

两道防线同时漏掉：该文件不在 lint 白名单，ruff 看不到；`noxfile.py:938-945` 的 `py37` session 是 compile-only，也看不到。

### 3.5 P1：CI 分层名不副实，且有永不运行的 job

- **"structure 秒级"不成立**：`tests/structure/` 的 19 个文件里有 **10 个在模块级 `import jittor`**（如 `test_optim_structure.py:9`），会触发完整 JIT/C++ 编译，是分钟级；且该 job 未安装 `build-essential`/`libomp-dev`（同文件的 `packaging` job 装了）
- **CUDA/NPU job 绑定不存在的 runner**：`.github/ci-baseline.env:6-7` 的 self-hosted 标签配合 `cuda.yml:4`、`npu.yml:4-5` 的**无过滤 `on: push`**，在未注册 runner 的环境里每次 push 都产生排队至超时的 job。`cpu.yml`、`structure.yml`、`docs.yml` 同样是裸 `push:`
- **benchmark job 永远不会因回归失败**：`noxfile.py:834-853` 调 `asv compare --factor 1.10`，但 asv 的 `compare` 仅在找不到结果时报错，检测到回归时正常返回 0；`asv.conf.json:17-21` 的 `regressions_thresholds` 只影响 `asv publish` 的 HTML 高亮

### 3.6 P2：`misc/` 只完成形态收敛，未完成职责拆分

`misc/tensor_ops.py` **2822 行**，而重构前 `misc.py` 是 2805 行——git diff 显示这是一次纯重命名，`misc/` 下无 `modules/`、`functional/` 子目录。

更需要注意的是 `tests/structure/test_misc_structure.py:265` 把 **2850 行**写成了该文件的行数预算，等于把现状固化为契约，后续想拆得先改测试。相比 `nn/`（facade 91 行 + 每个实现 ≤350 行）差距明显。

### 3.7 P2：`.gitignore` 仍有会静默吞文件的规则

- `agent/` 整块仍是反向白名单且放行只有一层：实测 `git check-ignore -v agent/scripts/sub/tool.py` 命中 `.gitignore:58` 的 `/agent/scripts/**`，子目录脚本会被静默忽略
- 搬迁遗留的死规则：`.gitignore:44` 的 `extern/mkl/mkldnn_lnx*/*` 锚定仓库根，而顶层 `extern/` 已不存在；实测对 `python/jittor/extern/mkl/...` 不匹配，该规则完全失效

### 3.8 P2：两个 ms-swift 补丁被删除且无迁移说明

`_patch_ppo_reward_value_seq_cls`（trl PPO 需要 reward/value 走 seq_cls，否则首轮 rollout 报缺 `.score`）与 `_patch_gkd_native_rollout_engine`（`--use_vllm false` + `lmbda > 0` 时 `GKDTrainer` 缺 `engine`）在三个适配包里均搜不到，只存在于 `jittor-lab/_state/` 的旧构建快照。

架构上讲得通（主仓库不该扛下游框架的 bug），但客观上是能力丢失。建议补一个适配包，或在 `docs/compatibility/torch-shim.md` 明确记一笔"已知不再覆盖"。

### 3.9 说明：pytest 未在默认环境安装

当前 `jt311` 环境跑 `python -m pytest` 报 `No module named pytest`。这不是交付缺陷——`requirements/dev-tools.txt:10` 与 `noxfile.py:43` 都 pin 了 `pytest==7.4.4`，走 `nox -s cpu` 会自动安装，且 `tests/compiler/__pycache__/` 下存在 `*-pytest-7.4.4.pyc` 说明确实跑过。但值得在开发文档中提示：直接跑 pytest 前需先安装 dev-tools。

### 3.10 其他小瑕疵

- 老 `setup.py` 的两处运行时守卫未迁移：非 Linux/Darwin 需 `FORCE_INSTALL=1` 的断言、Windows 要求 Python ≥3.8；现在只有统一的 `requires-python = ">=3.7"`
- 构建时产生 62 条 setuptools `Package would be ignored` 警告（`jittor.compat.shim.cpp_extension.include.ATen` 等目录靠 `include_package_data` 作为父包数据打入）。实测文件都在 wheel 里，但这是 setuptools 明确警告的用法
- cibuildwheel 无配置文件，`release.yml:132-141` 用它做"断言产不出平台 wheel"的反向校验，依赖其内部退出码 5 与精确错误文案，升版可能静默失效
- `nn/` 内部仍有 191 处 `jt.nn.xxx` 动态解析。这是被 6 个测试刻意锁定的 monkeypatch 可观测性契约，不算清理漏项，但仍是运行时耦合

---

## 四、建议的处理顺序

1. **修 `check_repo_layout.sh`**（3.1）。当前状态下没人能在跑过测试后正常 commit。
2. **修 ASV 的 slot 轮换**（3.2）。在此之前 benchmark 数字不可用；顺带把方法学规则写进 `docs/performance/benchmarking.md`。
3. **修 `test_pool_structure.py` 的 3.9+ 语法**（3.4）。在声称支持 3.7 的仓库里是真 bug。
4. **统一 pre-commit 与 nox 的 lint 白名单，并制定扩量计划**（3.3）。
5. **给 CI 的 `push:` 加分支过滤**（3.5），避免每次推送排队等不存在的 runner。
6. 其余按 P2 处理。

---

## 五、验收方法备注

本次验收实际执行了以下操作以避免"看配置即通过"：在 `/tmp` 下构建 wheel 与 sdist 并逐成员比对、按 `.dockerignore` 规则复现构建上下文后重新构建比对、在临时 venv 中实跑 ruff 全仓库与白名单两种范围、实跑 mypy、实跑 `tests/core/test_rootcause_semantics.py`、实际运行 Python 验证 `__module__` 返回值、实跑 `check_repo_layout.sh` 取退出码、查阅 asv 0.6.6 源码确认 `compare` 的返回值语义。未修改仓库任何文件。
