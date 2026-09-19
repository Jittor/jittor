# 下游仓库 torch-compat runbook 与四轴实测

- Status: Accepted for the measured scope; several repos remain report-derived
- Date: 2026-09-19
- Baseline commit: `90fe0b9d`
- Owner: Jittor maintainers
- Review when: 任一 `agent/skills/<lib>-torch-compat/` 的目标库换大版本、`_ecosystem_runner`
  的 device 分支再改动、或 `verify_repo.py` 的口径变化

## 这次做了什么

为 13 个下游仓库各写了一份"在 jittor shim 上跑 + 与原生 torch 对比"的 runbook
（`agent/skills/<lib>-torch-compat/`），并把**模板与四轴验收协议**独立成
[`torch-compat-repo-runbook`](../../agent/skills/torch-compat-repo-runbook/SKILL.md)。
runbook 不是文档草稿：每条都跑过，结果写回各自的 `## 实测（2026-09-19）` 一节。

四轴 = **支持清单 / 精度 / 显存 / 速度**。执行工具
`agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py`，单 case 执行委托给
项目自己的 `compat/tests/torch/_ecosystem_runner.py`，所以数字与生态门禁同源。

## 修掉的一个门禁缺陷

**现象**：`_ecosystem_runner.py --device cuda --runtime jittor` 必挂在第一个
numpy 支撑的张量上：

```
RuntimeError: dispatch_context.cc:52: Expected all tensor inputs on the same backend
and device ... first input backend=0 index=0 but another input has backend=1 ...
```

**根因**：`_select_device` 的 jittor 分支返回的是恒等函数，注释理由是"Jittor 移动的是
图，不是张量"。但这句话只对 Jittor **自己创建**的张量成立：`use_cuda=1` 让参数落在
设备上，而 `from_numpy` 出来的张量仍驻留 host，两者进同一个算子就触发同后端检查。
真 PyTorch 的 `from_numpy` 同样返回 CPU 张量，caller 必须自己搬——harness 在 torch 分支
搬了，在 jittor 分支没搬。

**修复**：`compat/tests/torch/_ecosystem_runner.py` 的 cuda 分支改为显式把 host 支撑的
张量搬到设备。这不是绕开 Jittor：torch 侧行为完全一致，Jittor 没有问题。NPU 分支保持
原样，因为本机没有 ACL 设备可验，未经验证的改动不该进树。

**范围**：生态门禁默认跑 CPU（`JITTOR_TEST_DEVICES=cpu`），所以这次修复不影响现有
nightly；它解锁的是 CUDA 上的四轴实测。

## 顺带修掉的第二个门禁缺陷：结构门禁会被构建残渣判红

`tests/structure` 是这批改动的回归门禁。改动后本地它多报 2 条失败，都在
`test_import_layering.py`；同一份代码在一个 pristine HEAD worktree 上只有 3 条（3 条都与
本次改动无关）。逐条定位，2 条都不是代码问题。

**现象**：`python tools/lint/check_import_layering.py` 报
`cycle-surface FAIL (1) — 4 import-time cycles, up from 3`，而 diff 里没有任何东西能解释它。

**根因**：`compat/` 下留着一份被 gitignore 的 setuptools 构建产物
`compat/build/lib/jittor/compat/...`（148 个 `.py`，`setup.py build` 的整棵拷贝）。
`scan_roots` 按 `package-dir` 把 `compat/ -> jittor.compat` **整棵**递归，于是这份拷贝被当成
源码读进来：模块数 539 → 687，拷贝自身成对的 `__init__` 构成了源码里并不存在的环。
顺带澄清一个一度被怀疑的对象：本次新增的 `python/jittor/nn/functional/_amp.py` 只是加入了
本来就存在的那个大环（146 → 147 个模块），`MAX_CYCLIC_MODULES = 164` 远未触及，
`CYCLIC_SUBPACKAGES` 里也早有 `jittor.nn`——它本身不触发任何契约，两条失败全部来自残渣。

**修复**：`discover_modules` 现在跳过**不是包的** `build/` 与 `dist/` 目录——判据是"该目录
自己没有 `__init__.py`"，因为 `jittor.build` 是真实包，只按目录名判断会误伤。
这两个名字不是新口径：`compat/pyproject.toml` 的 `[tool.jittor.sdist]` 早就写着
`exclude = ["build/**", "dist/**", "tests/**"]`。回归测试
`test_build_output_inside_a_package_root_is_not_source` 用临时目录同时锁住两侧
（产物被跳过、`jittor.build` 保留）。

**没有删那份残渣**：它是别人环境里已有的东西，删掉只是把同一个问题推迟到下一次构建。
修完之后残渣仍在树里，门禁是绿的，且数字与没有它时逐字相同（539 模块 / 1424 边 / 3 环）。

**结构门禁现状**：`tests/structure/test_import_layering.py` 16 passed。其余 3 条失败
（`test_packaging_structure.py` 的 find_packages 与 MANIFEST 两条、`test_pytest_contract.py`
的 collection 副作用一条）在 pristine `90fe0b9d` 上以同样方式失败，与本次改动无关，未处理。

**复验（2026-09-19 晚，grad 身份修复那次）**：同一份 `compat/build/` 残渣还会判红另外两个
扫描器——`tests/structure/test_cleanup_structure.py::test_cross_file_duplicate_implementations_are_reviewed`
（把 `compat/` 的每个实现镜像成一份"未审查重复实现"，多出 1262 条，逐条都是
`compat/x` 对 `compat/build/lib/jittor/compat/x`）与
`compat/tests/structure/test_compat_write_entry_points.py::test_every_process_global_write_in_compat_is_classified`
（54 条未分类的进程级写入全部来自残渣；非 `compat/build/` 的条目只有 pytest 自己的节点名）。
连同 `python/jittor/extern` 造成的
`tests/structure/backends/comm/test_comm_resource_layout.py::test_legacy_runtime_resource_trees_are_absent`
（本机保留的目录，fresh checkout 通过，见 `2026-09-14-vllm-omni-h3-enablement.md` §Gates）。
这三处红项都由 **git 不跟踪**的本地产物引起（`build/` 在 `.gitignore`，`extern/` 被忽略），
在 pristine checkout 上不存在，故为绿；与源码改动无关。

## 四轴实测汇总

CUDA，两侧同 device，速度取重复最小值，`fallback_count` 见各 runbook。

| 仓库 | 跑通的 case | 精度（最差，对全场量级） | 速度 jt/torch | 备注 |
| --- | --- | --- | --- | --- |
| transformers | tiny 5/6，large 5/5 | ≤3.6e-4 / ≤1.51e-3 | 1.71–3.21x / 1.03–1.19x | `transformers_t5` CUDA 挂在 **oracle 侧** apex |
| diffusers | 3/3 | 4.93e-5–6.77e-4 | 1.30–2.89x | 大 case 显存 jt 3.78 GiB > torch 1.10 GiB（**旧口径**，见下） |
| peft | 1/1 | 2.745e-4（abs 1.209e-2） | 2.04x | 需 transformers 4.x 包站 |
| ms-swift | 1/1 | 2.745e-4 | 2.05x | 同上 |
| mmcv | 1/1 | 2.514e-7 | 0.998x | 用 mmcv-lite（完整 mmcv 需 CUDA 编译） |
| mmengine | 1/1 | 3.983e-7 | 0.984x | |
| torchmetrics | 0（无 case） | — | — | 见下"真实缺陷" |
| tensordict | 0（无 case） | — | — | shim 缺 `torch.distributions.constraints` |
| vllm | 0（无 case） | — | — | adapter 生命周期测试 9 passed |
| vllm-omni | 无 case | — | — | 小 VAE 解码端到端跑通（含冷 JIT 167.5 s） |
| flash-attention | 无 case | — | — | 首次观测到 fused hit |
| verl | — | — | — | 本机无环境，报告推导 |
| trellis | — | — | — | 本机无环境，报告推导 |

小 case 的 speed ratio（2–3x）是 dispatch-bound，不代表真实尺寸；transformers 的 large
tier（1.03–1.19x）才是可引用的形态。

## 显存：口径错了两轮，剩下的差额根因是 shim 的 grad 对象身份

工具原本两侧各问各的，问的不是同一个量：oracle 侧 `torch.cuda.max_memory_allocated`
（**活跃**字节，单卡），Jittor 侧 `get_mem_info().total_cuda_used`——它是**活跃+缓存空闲**，
且 `mem_info.cc:316-322` 对**所有**设备求和。reserved 对 allocated。

工具已改为两侧都报 live 与 pool 两个数：torch 用 `max_memory_allocated` /
`max_memory_reserved`，Jittor 用 `device_memory_used(N)` / `device_memory_reserved(N)`。
被测 jittor 太老、没有这两个 API 时，报告 `jittor_peak_bytes = -1` 并附 `memory_note`
说明原因，而**不是**拿 reserved 顶上——deployed 那份 2026-09-11 拷贝就落在这一支。

换口径后差额仍在（2.88x），于是分相查根因。这期间又踩到第二个口径陷阱，一并记下：

`jt.core.get_peak_allocator_used_memory()` 形状像 `max_memory_allocated` 的对应物，
但 `MemoryProfiler::get_memory_info` 是**不过滤 CUDA** 地累加 `SFRLAllocator::sfrl_allocators`
（`memory_profiler.cc:58`），也就是**主机+设备**。实测：512 MiB 纯 CPU 的 Var 把它推了
512 MiB，再加 256 MiB CUDA 的 Var 推到 768 MiB；它也没有复位接口。**不能与 torch 的
`max_memory_allocated` 并列**，能并列的只有 `device_memory_used` 与 CUDA-only 的
`total_cuda_used`。

根因在 shim：`p.grad` 每一步都是新对象。torch 的 `zero_grad(set_to_none=False)` 就地
清零已有张量、`AccumulateGrad` 也就地累加，所以一个训练循环里 `p.grad` 始终是同一个对象；
shim 的 `_zero_grad` 用 `zeros_like` 重建、optimizer-free 的累加写成 `prev + gr`，两者都换
对象。于是任何跨步持有 `[p.grad for p in model.parameters()]` 的代码（梯度裁剪、四轴的
显存 pass）在 jittor 上每步钉住一整套梯度。`large_transformers_bert` 实测每步多 198 个
Var、约 415 MiB——198 ≈ 参数个数，415 MiB ≈ 参数字节数。两处改为就地操作后，live 在步间
回落到约 840 MiB（阶段边界实测 839–845 MiB），Var 数恒定在 403–411，不再随步数增长；同一次运行里 198 个输出与梯度
对 torch 的相对误差与修复前逐项一致（仍在上表 ≤3.6e-4 一档），所以修的是对象身份，没动数值。

修完重测同一个 case（仓库 checkout，只取 device-only 计数）：

| 阶段 | torch live 峰值 | jittor live 峰值 |
| --- | --- | --- |
| 第一个 backward | 1640.3 MiB | 4048.2 MiB |
| 第一个 warmup step | 2122.5 MiB | 4469.7 MiB |
| warmup / timed 稳态每步 | 2083.6–2122.5 MiB | **2366.4 MiB** |

所以整个 run 的 2.11x（live）/ 2.01x（pool）**几乎全来自第一步的瞬态**：jittor 首次
backward 要 4048 MiB、首个训练步 4470 MiB，之后稳定在 2366 MiB，torch 没有这个瞬态；
稳态每步只差 **1.11x**。jittor 的 pool 是 high-water 且从不归还 CUDA，瞬态被永久记进
pool，四轴工具报的 whole-run 峰值也就一直带着它。

做显存预算时两个数都要说：稳态约 **1.11x**，但**第一步峰值约 2.1x** 才是必须留得下的那个。
上表 diffusers 的 1.10 -> 3.78 GiB 仍是**旧口径**，尚未按新口径重测。

（外部采样仍然不可用：`nvidia-smi --query-compute-apps` 看不到进程（容器 pid 映射），
按 GPU 的 `memory.used` 又含同租户。所以只能问运行时自己，才更要保证问的是同一个量。）

## 顺带发现的真实缺陷

这些不是 runbook 的问题，是 shim/兼容层的问题，值得单独立项：

1. `torchmetrics`：`Reduce dim out of range`（`src/op/reduce_op.cc:291`），classification
   用例打不出来；另有 nvcc 对 `setitem` 的 return-512 报错。
2. `tensordict`：shim 的 `torch.distributions.constraints` 是空的（`real` 等属性缺失），
   `import tensordict` 直接失败。
3. `peft 0.20.0`：shim 缺 `torch.distributions.wishart`，导入失败。
4. `ms-swift 4.5.3 + peft 0.20.0`：`Linear` 缺 `config` 参数，TypeError。
5. `torchmetrics` 辅助函数标记名不一致：`_jittor_fast_*` vs `_jittor_orig_*`。
6. `transformers_t5` 在 **oracle** 侧（真 torch 2.13.0+cu129）挂 apex
   `fused_rms_norm_affine_fwd: input must be contiguous`——不是 shim 的锅，但会挡住
   后续所有用这个 oracle venv 的 t5 CUDA 对拍。

## 复现

```bash
cd <repo>
source /root/jittor-lab/minimax-h3/env-jittor.sh
export JITTOR_HOME=<本任务独占的缓存目录>
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
"$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo <lib> --device cuda --repeats 5 --out <lab>/_state/<topic>/verify/<lib>
```

`--list-only` 先看该库注册了哪些 case，不起进程。缺库时用
`pip install --target <site>` 建包站，再把 `JITTOR_ECOSYSTEM_PACKAGE_SITE` /
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` 指过去；**不要装进 shim 解释器本身**。

本次用到的包站：`/root/jittor-lab/_state/verify-ml/site`（peft 0.20.0、ms-swift 4.5.3、
mmcv-lite 2.2.0、mmengine 0.10.7）与 `site-peft17`（transformers 4.56.2、peft 0.17.1、
ms-swift 4.5.2）；`/root/jittor-lab/_state/verify-misc/site`（torchmetrics、tensordict）。
原始 `verify-report.json` 在各 lab 输出目录，不进主仓库。

## 合并远端后的复验

写完这份报告后分支和远端分叉了：本地 10 个提交对远端 59 个（merge-base
`4bd42443`），而远端那条线上有整套 autocast / GradScaler / 半精度重做，正好压在本文
量化的对象上。按协作规则先提交本地、再 `git merge origin/2.0-refactor`，合并提交
`5fca26ff`。重叠 4 个文件，只有 `src/core/var_holder.h` 冲突——两边加的是同一个
`#include "runtime/device.h"`，只是注释措辞不同，按"重叠处保留远端"取了远端；其余
自动合并的文件逐个核对过两边内容都在。

**因此上面所有数字的基线仍是 `90fe0b9d`，不是合并后的树。** 合并后只复验了这些：

| 项 | 结果 |
| --- | --- |
| 导入方向门禁（3 条契约） | 全过：543 模块 / 1441 边 / 3 环，无新增环 |
| `tools/check_repo_layout.sh` | OK，221 个 Markdown |
| `tests/structure` 三个受影响文件 | **55 passed, 8 subtests** |
| bias 的 dtype 行为 | 5/5 与 torch 一致 |

那三个文件合并前有 3 条失败，远端 `f0bd4279` 已经全部修掉；我新增这份报告时漏了重新
生成 `MANIFEST.in`（`test_manifest_covers_runtime_trees_without_cache_payloads` 正是
量这个），已按 `tools/build/generate_manifest.py` 补上，整份文件只多一行。

bias 那一条单独复验，因为它对应的三个用例住在 `compat/tests/torch/`，在本 lab 收不起来
（见「限制」）。用 `probe_bias_dtype.py` 直接跑：autocast 下
`conv3d(f32, f32, f32)`、`Conv2d` 模块、`linear` 都返回 **float16**，无 amp 区域返回
**float32**，conv3d 不带 bias 的对照组同样是 float16——与真 torch 2.13 的 dtype 逐条一致。

**还没做**：四轴没有在合并后的树上重跑。远端改的是 dtype/半精度/autocast 与 generator
的流，正是 transformers/diffusers/peft/ms-swift 那几个 case 量的东西，所以上表的精度与
速度数字在合并后需要重测。另一个障碍是四轴走 lab 里那个部署树（2026-09-11 的副本，见
copy-deploy 陷阱），要量合并后的树得先把仓库重新部署进 `venv-jittor`——那会动到 H3 lab，
要先确认没人在用。

## 限制

- `compat/tests/torch/*.py` 在本机 lab venv 里无法用 pytest 收集
  （`compat/__init__.py` → `_aliases.py:8` 相对导入越界）；torchmetrics/tensordict 的
  结论来自独立 runner，不是门禁原样复现。
- peft/ms-swift/mmcv/mmengine 只有 tiny case，没有 speed tier。
- flash-attention 没有原生基线（oracle 的真 2.8.1 导入失败），所以只有 dispatch 事实，
  没有速度对比。shim 侧解析到的是部署树里的 stub。
- verl / trellis 本机无环境，结论全部报告推导，runbook 里已标注。
