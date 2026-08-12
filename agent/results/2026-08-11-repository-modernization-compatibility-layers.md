# 仓库结构现代化阶段 4：兼容层四层分离

## 状态

阶段 4 已完成框架能力、通用机制、根因修复和下游粘合四层拆分。主仓库不再发布
`monkeypatch_ops.py`、TRELLIS runtime、Gaussian Splatting runtime 及其专属脚本；
对应适配分别由三个独立可选包通过 entry point 注册。

## 框架内改动

- `jittor.nn` 新增 varlen SDPA、累计长度缓存、参数化多头 RMSNorm、部分 RoPE、
  dual-grid mesh 和设备端 submanifold sparse conv3d。稀疏卷积不再按 tap 做 Python
  matmul，也没有 `.item()` 强制同步；layer norm CUDA 后端取消项目特定形状和
  `eps` 白名单。
- `jittor.compat.module_patcher` 提供单一进程级 import finder、注册表、
  entry point 加载、逐项错误报告以及不会覆盖后来补丁的 method restore。
- `jittor.compat.external_backend` 参数化模块、源码根、manifest、build script、
  capability cache 和环境提示。嵌套构建目录会归属于最近的已注册源码根，因而
  `flash-attention/hopper/setup.py` 不再被通用扫描器误执行。
- `requires_grad_(False)` 改为可逆的叶子策略，不再等价于永久切图的
  `stop_grad()`；图遍历按创建时快照跳过冻结边。Parameter 标记只识别真实参数，
  `NanoVector` 的哈希与同值 Python tuple 一致。
- 根模块 bootstrap 改成逐项加载和可读报告，不再用一个 `try/except pass`
  静默吞掉所有补丁失败。

## 外置适配包

| 包 | 提交 | 单元/CUDA | wheel SHA-256 |
| --- | --- | --- | --- |
| `jittor-trellis` | `d10ed75` | 8 pass，4 个显式 CUDA skip；既有 CUDA 能力 4/4 | `fcd5a9fbf2489864476918a4c973b309bb63306dfc7c688d3a86492334b3585d` |
| `jittor-gs` | `88e3a53` | 4/4 | `5cbbee86b0902392db8670e9d8940a278fd6d39ed5391f98751917faa90eef52` |
| `jittor-hf-compat` | `f8cdedb` | 5/5 | `9a70443209fdafbeb25f4a702f2213c3087354b0e43c0f4f455842a14e278339` |

2026-08-12 在上述当前提交重新执行单元测试和 wheel 构建。三个 wheel 在不安装 Jittor 的隔离环境中均能被
`jittor.module_patches` / `jittor.external_backends` 发现并调用；总注册量分别为
GS 5、HF 3、TRELLIS 21。构建产生的 ignored cache、`build/` 和 `*.egg-info` 已删除，
三个独立仓库均保持 clean。

## ms-swift 修复

- PPO reward/value task type：在 2026-08-12 的上游 `main` `d17f031` 上重放为分支
  `codex/fix-ppo-regression-heads-main`、提交 `6b1f0b2`。真实 PyTorch 2.12.1、
  Transformers 5.12.1、PEFT 0.19.1、TRL 0.29.1 组合环境完成模块导入，新增测试
  `2 passed`（7 条分支断言）；YAPF 0.30、Flake8 和 diff check 通过。
- PPO `git format-patch` 保存在
  `${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/ms-swift-patches/`，SHA-256 为
  `c7c8ecbda91f275ebdf4ce2c35e29ca73d9ef083ccbc7db4650d778141bb9f02`；已从
  `d17f031` 实际 `git am`，应用后的 tree 与 `6b1f0b2` 完全一致。
- GKD native rollout gap 已由 ms-swift 上游
  [#9818](https://github.com/modelscope/ms-swift/pull/9818) 在提交 `4805d7f` 合并解决；
  上游把 native `TransformersEngine` 初始化放入共享 `RolloutTrainerMixin`，覆盖原补丁的
  核心行为，因此不再提交重复 PR。旧 `7e25452` 补丁仅保存在 `superseded/` 供追溯。

以上 ms-swift 审计用于证明项目专属补丁已经离开 Jittor 核心。按本次仓库整理的最终
范围，修复和发布 ms-swift 自身缺陷不属于 Jittor 的交付条件；本地 PPO 补丁仅作为
未版本化研究证据保留，不要求创建上游 PR。

## Wheel 边界

Stage 3 的 1,052 成员基线已固化为
`agent/baselines/wheel-contents-stage3.txt`。Stage 4 wheel 有 1,057 个成员：

- 10 个批准新增：两个通用兼容机制、五个 `nn` 能力模块、三个回归测试；
- 22 个批准内容变化，全部固定候选 SHA-256；
- 5 个批准删除：`monkeypatch_ops.py`、两个项目 runtime 和两个 GS 脚本。

direct wheel 与 sdist 重建 wheel 的 1,057 个成员逐文件哈希完全一致。最终产物：

| 产物 | SHA-256 |
| --- | --- |
| direct wheel | `b42ff0b1dacf81244280dfca3e855f86ad4d9955dacbd18f3da1c8a44aa1d1a2` |
| sdist wheel | `7c42e3587f419ef9a0a44c2318fc72b412085a633da2c5036d13448e5304d215` |
| sdist | `94ef40cb56923059b80a7acb3552118b105793a72cd32bdd91582542889a3817` |

## 验证

| 验证 | 结果 |
| --- | --- |
| root cause / capability / mechanism / bootstrap / attention 最终矩阵 | 85 项：82 pass，3 个未配置原生 flash 源的显式 skip |
| compat rollback / resource loader 定向复验 | 17/17 |
| autograd engine / silent regression | 9/9；11/11 |
| optimizer 核心模块 | 21/21 |
| Python 3.7 grammar | Stage 4 索引中 `python/jittor` 485 个 Python 文件通过 |
| wheel gate / packaging / deploy / CUDA wheel / runner 结构测试 | 9/9；4/4；12/12；6/6；7/7 |
| 隔离 wheel CPU | 来源为隔离安装目录；可逆 grad、Parameter、shape hash 均通过 |
| 隔离 deploy | 7 项；`--check`、`torch` 和 `flash_attn` 导入通过 |
| 隔离 wheel CUDA | RTX 4090 / JTCUDA 12.2 / cuDNN 8；loss `14.0`，grad `[2,4,6]` |
| Gaussian Splatting | 扩展 smoke、3 iteration 训练、4 train + 2 test render、metrics 全通过 |
| TRELLIS | 冷启动和独立热启动均生成 mesh + PBR GLB；热启动未再扫描 `hopper/setup.py` |

GS 三步 smoke 的测试集指标为 SSIM `0.0095421`、PSNR `6.7422075`、LPIPS
`0.6262895`。这组数值只证明三次迭代的 train/render/metrics 链路闭合，不作为质量
基线。完整产物、日志和缓存均在
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/`，不进入主仓库。

本阶段没有执行 NPU 回归；不把 CPU/CUDA 结果扩写成 NPU 支持证据。FSDP2、统一 shim
installer 和剩余模块物理路径将在阶段 7 按既定兼容别名方案继续收敛。
