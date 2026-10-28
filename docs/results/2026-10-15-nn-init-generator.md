# Generator 驱动的浮点原位初始化

- Status: 限定 CPU / 真实单 CUDA 的两个 nn.init 消费者通过；其他初始化器未授予支持。
- Date: 2026-10-15。
- Baseline: `origin/2.0-refactor` 的 `8bb59861de5bbf98954728a331a4efe42ea8a389`，保留既有 runner refs 后整合为 `b196ab01a6d9a0bdef1e9143205983c2048f8b66`。
- Code: `222c46662fba3b4fa7550ccfef66401cde3f4787`（测量文件与该提交逐字节核对）。
- Owner: Torch frontend / native RNG maintainers。
- Review when: factory 随机流、Generator、初始化 dtype、原位写回或设备发布变化。

## 实际断点与修复范围

独立 binary Torch `2.11.0+cu128` CPU 先执行两个初始化的最小状态对拍。
显式 Generator 会推进自己的完整状态，不改变全局 CPU 状态，恢复其状态后重放完整数组。
冻结基线上的候选实际调用则两者都表现为私有状态不推进、全局状态改变、重放失败。
原始 `candidate-cpu-baseline.log` 保留，不将该失败改记为后续修复源码的结果。

`torch.nn.init.uniform_` / `normal_` 现在经已安装的 `rand` / `randn` factory
调用原有 native Generator owner。工厂负责 Generator 类型、设备和 CUDA ordinal 校验，
没有增加 PRNG、全局 reseed、NumPy 抽样或跨设备 fallback。
输出跟随目标 tensor 的实际 dtype/device；float64 用 float64 抽样，
float16/bfloat16 的尺度与偏移先在 float32 计算，再写回目标 dtype。
本批验证同一 Python 对象、构造时可训练/停梯度状态与既有 retained-view 写回。
可逆 `requires_grad_(False)` 和非叶张量旧图不在本批通过范围：后续独立探针发现共享
写回仍可能解除可逆冻结、把非叶变为叶；该既有缺口由后续共享写回修复单独验收。
无效 std、非浮点目标、反向/非有限/超 dtype 范围的 uniform bounds 在抽样前拒绝。

## 验证方法

新测试的相同通用函数体由独立解释器的真 Torch 与候选执行。
参考解释器记录自己的 `torch._C` binary 哈希、包版本、实际解释器、GPU 可用性和测试文件哈希，
并在测试前后拒绝 `jittor` import。候选特有的 native draw observer 单独执行；
它拒绝 NumPy 抽样，核对传递的 native Generator handle，读取 random 输出实际 dtype，
在 CUDA 主机读回前用 driver pointer attribute 检查真实分配 ordinal。
每个候选节点在 `forbid_backend_fallbacks()` scope 中执行。

所有样本/梯度/状态比较使用完整数组，不只比 seed 或统计量。
跨框架验证的是状态/设备/原位行为契约，不声称两种原有 PRNG 的样本逐位相等。
有界 uniform 与非默认 normal 均实际取样检查完整范围/均值/std，
并覆盖 std=0 与相等 bounds 仍推进随机流的行为。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 独立 Torch CPU | 29 passed / 29 CUDA skipped / 4 native observer deselected | 相同通用测试 |
| 独立 Torch CUDA | 58 passed / 0 skipped / 4 native observer deselected | 相同通用测试，实际 CUDA |
| 候选 CPU | 84 passed / 61 skipped | 新 init、旧 Kaiming、native Generator、default Generator |
| 候选 CUDA | 149 passed / 1 skipped | 同四文件；仅两可见 CUDA 设备节点跳过 |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | native 98 + Torch 224；两个模式 exit 0 |
| 布局 / 完整 structure | 布局通过；1622 passed / 6 skipped | tests/structure + compat/tests/structure；代码提交前冻结源 |

最初新增测试在 CPU-only build 中无条件读取 CUDA 状态，且 observer 将公开 dtype
误按 native 字符串比较；分别修为可用设备条件和 dtype 对象比较后重新执行。
这些测试 fixture 失败原日志保留；没有修改 CUDA API 来迁就新测试。
参考第一次 retained-view normal 用 contiguous target 作期望值失败，
原因是 Torch CPU normal 的 contiguous/noncontiguous 路径消费不同序列；
最终回归重放相同 retained-view 布局，而不要求不同布局消费相同采样序列。

## 复现与证据

未版本化目录：`$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/`。
其中 `probe.py`、oracle metadata、CPU/CUDA 环境白名单脚本、原始失败与最终日志、
source freeze 与最终 manifest 绑定实际测量源码和原始文件 SHA256。
测量时的两个修改文件尚未提交；提交后逐文件核对 Git blobs，不重标为历史 clean-source 测量。
缓存均为本任务独立的 HOME/JITTOR_HOME/TMPDIR，CPU 与 CUDA 分开，首次编译串行。core runner 的 native/Torch 会话各使用独立目录，
以同树、同配置、本任务已完成的 CPU 编译缓存为起点；之后独立写入。
GPU 上存在其他进程；本轮只声明正确性，不比较速度或整进程峰值。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_torch_init_package.py \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_default_generator_contract.py
```

实际执行还必须按环境手册设置独立状态目录、显式 CPU 隐藏 CUDA 或启用真实 CUDA，
设置 strict fallback 与已安装工具链；上式只列维护的测试选择。

本批不授予所有初始化器的 Generator 支持：native Kaiming helper 仍明确拒绝，
trunc_normal_ 等既有 NumPy/近似路径未修改；也不授予完整 C-extension Generator ABI、
整数范围精度、跨框架 RNG state 互换或未实测的 ROCm/NPU。
本轮冻结基线的 CPU-only build 中 `torch.cuda.get_rng_state_all()` 会拒绝，真 Torch 此时返回空列表；
新 fixture 只采集实际可用设备的状态，不授予该 CUDA API 等价。后续主线 `97d7d478a48ecc80f13f2858f0bc8fa78e377480` 已独立修复该缺口；
本表与原始结果继续保留 b196 测量身份，不改记为 97d 的结果。
Tensor.uniform_ / normal_ 的同类独立消费者将另行修复与验收，不能以这两个 nn.init
入口的通过替代；R33 原始 PR rebase/作者处置仍未执行。

完整结构检查针对代码和测试冻结状态执行；本页随后独立提交。

主树在同步 `1ecea7d5879761ec487d2497a25ef07bad47ae17` 后独立复现上述缺口，
整合代码为 `661e1851e`，报告为 `d8038c44b`；中央索引与 MANIFEST 随后登记。
该冻结状态下 CPU 84 passed / 61 skipped，真实 CUDA（两卡可见）145 passed / 0 skipped，
core 322 passed / 42 skipped / 1 existing xfail，完整 tests/structure 与 compat/tests/structure
1622 passed / 6 skipped。结构跳过含两项缺目标加速器、两项声明跳过及两项该解释器未安装
原生包而无法检查安装路径归属。主树已有后续默认 Generator 测试变化，节点数不与 b196
历史结果合并。本表保留原测量身份；主树日志及独立重算 29 个原始证据文件的清单位于
未版本化的 `$JITTOR_LAB_ROOT/_state/nn-init-main-integration-1ece/`。
