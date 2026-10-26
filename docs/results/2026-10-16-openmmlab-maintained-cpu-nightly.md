# OpenMMLab：全新的维护版 CPU nightly 验证

- 状态：完整的本地 CPU profile，经独立审计；不声明托管计划已执行。
- 复查日期：2026-10-16。
- 维护者：Torch 兼容性与下游验证维护者。
- 测量源码：`7bb144e984097b8cf14242818cb7fd4c9a58faa6`。
- 复查条件：上游固定版本、构建、安装身份、源码采集、数值预算或训练所有者发生变化。

完整的 `ecosystem_openmmlab_installed` CPU 会话在 163.15 秒内通过两个原始节点，零跳过。
nox 执行包含其首次私有 JIT 编译，共耗时七分钟。每个节点都运行种子 0、种子 7 与 19 的
固定权重输入扰动，以及三步真实的 MMEngine 训练。没有预检、用例过滤或更小的输入替代该
会话。

## 实际范围与结果

| 原始用例 | 输入 / 参数 | 数组对 | 最大绝对误差 | 最大归一化误差 |
| --- | --- | --- | --- | --- |
| MMEngine BaseModel | FP32 样本 `(4,6)`；六个可训练参数 | 99 | 2.39e-7 | 5.22e-7 |
| MMCV ConvModule | FP32 样本 `(2,3,8,8)`；八个可训练参数和三个 BatchNorm 缓冲区 | 132 | 2.39e-7 | 1.55e-6 |

独立的 NumPy 重算验证了全部 **231 个数组对**、精确的字段/形状/dtype 集合、有限数值、
所有请求的参数与样本梯度、两种扰动下输出与梯度的变化，以及所有阶段之间完全一致的初始
参数/缓冲区。未改动的逐字段归一化预算为前向 .002、反向 .01，并保留原有的梯度块下限。
固定输入、目标和整型计数完全一致。

训练使用未改动的上游 `BaseModel.train_step`、`OPTIM_WRAPPERS.build` 和 `OptimWrapper`：
SGD，学习率 .01，动量 .9，梯度裁剪最大范数 1。保存的梯度阶段仍位于裁剪之后、优化器
step 之前。独立审计针对每个运行时实际保存的梯度和前值，检查全部 84 次参数更新和 84 次
动量更新，误差在更新尺度四个 float32 ULP 以内。每个步骤所有参数都发生变化。训练包含
75 个 MMEngine 字段和 102 个 MMCV 字段，无缺失梯度或缩短的轨迹。

全部 16 个实际 worker PID 各不相同，全部 478 次快照观察都驻留在 CPU。两个运行时都使用
两个 runtime/OMP/MKL/OpenBLAS 线程、FP32 计算、禁用 TF32/cuDNN benchmark 和最高 matmul
精度。控制器显式隐藏 CUDA；候选元数据记录 `use_cuda=False`、`use_acl=False`、回退策略
error 且计数为零。oracle 从其自己的前缀加载真实的 Torch 2.11.0+cpu。解释器使用独立的
安装前缀。会话具有外部 HOME/临时/缓存边界和一个私有的候选 JIT 缓存；它不声明单独捕获
了每个运行时的 HOME/临时路径。benchmark 与单测编译缓存保持分离，首次编译串行执行。
单个 MM worker 数据包不提供较新的共享完整平台头部，也不提供单独的 CUDA 可用性回读；
两者都没有被回填。

## 实际源码与安装

固定的干净上游仍为 MMEngine 0.10.7，位于 `390ba2fbb272816adfd2883642326d0fd0ca6049`，
以及 MMCV 2.2.0，位于 `3a5fb4b96ab1b241e7379adf7c326ef6edef6515`。provisioner 从这些
检出构建实际 wheel，使用兼容的固定运行时构建工具，并保留上游符号链接。MMCV 构建为
`mmcv-lite`，不含编译算子；因此不支持 `mmcv.ops`。

| 构建输入 | 每个前缀的完整包文件数 | 实际 wheel SHA256 |
| --- | --- | --- |
| MMEngine | 157 | `d9691e3a6bc32141d9c6e13d9f8dce1234b4be3716024bf3f6b9dbdd32064da8` |
| MMCV-lite | 531 | `0a9658f06bd79a10343621834cbfad7a4dbf57f7bd41d8e7e483468939ea09a8` |

两个独立安装的前缀都与每个源码构建 wheel 包逐字节一致，并匹配其原始 METADATA。它们
实际的 pip RECORD 哈希/大小绑定了每个包成员；这些收据与归档 RECORD 不同，且从未被
后者替换。运行后的四份安装证明与测量前的证明完全相等。官方 CPU Torch wheel SHA256
`8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` 另行绑定了除自
未哈希 RECORD 之外的全部 11,945 个原始已安装归档文件，包括 `functorch/`、`torchgen/`
和 dist-info 载荷。其实际 pip RECORD 行与每个原始文件匹配。三个私有环境都在运行后通过
`pip check`。

在准备新的 profile 期间，provisioning 最初在脏的 `2f8895f1a` 上运行。其 manifest 如实
记录 `source_clean=False` 和 provisioner SHA256
`bce3d824bd7c5cb82ce52baaf9441a70bda455198c4f15ea9d6adf0cdd57db55`；这些字节与已提交
的 profile 一致。而被测运行使用的是干净的 `7bb144e98` 检出。控制器的完整 2,905 个受
跟踪文件哈希、当前字节和原始 Git blob 一致，包括符号链接目标文本。完整的源码归档予以
保留。每个 worker 还绑定其选定的原始用例/runner/测量文件；该选定映射不是完整的导入
清单。

## 已复现的失败与修复

原始的干净 `33e69e84b` 运行确实在两个节点上于基础比较处失败：两个合法的独立安装具有
不同的绝对 `origin` 和 `package` 路径，而比较器错误地把它们当作不同的身份。其原始输出
和失败日志保持不变。修复首先针对该前缀的包初始化器、当前完整包哈希、版本和原始源码/
wheel 证明来检查每个真实 origin。只有那两个绝对位置字段被排除在跨运行时身份比较之外；
其他所有已声明字段都必须一致。遗留的训练辅助函数在复制的报告中接收经验证的依赖身份，
原始证据包保持不变。Origin、载荷、资源、版本、缺失/多余身份和训练整合的反例覆盖了该
行为。

该修复通过 54 项定向主机测试，CPU 核心 322 passed / 42 skipped / 一个既有 xfail，扩展
结构 1,628 passed / 六个 skipped，布局、生成的 manifest、Ruff 和 diff 检查。全部 74 个
既有的 nox 函数 AST 保持不变。后来的 Metrics/TensorDict 共享证明整合也在其 260 个通过
测试中重跑 MM 主机契约；原始 `7bb144e98` 的数值测量未被改贴到该整合上。

## 重放与保留的证据

```bash
python tools/provision_ecosystem_nightly.py --library openmmlab \
  --python <healthy-python-3.11> \
  --state-root "$JITTOR_LAB_ROOT/_state/openmmlab-nightly/<new-run>" --run
```

CPU 工作流会安装其 Ubuntu/OpenCV 运行时库。一次本地无 root 容器的预检在用户命名空间
设置处失败，此时尚未运行 profile；不声明 CI 容器通过。

未纳入版本控制的证据位于 `$JITTOR_LAB_ROOT/_state/openmmlab-nightly/20261016/`：
`formal-controller-v2.json`、`formal-terminal-v2.json`、`fresh-v3/logs/038.log`、两个
保留的 `maintained-output` 尝试、`complete-independent-audit-v2.json`、
`complete-source-audit-v2.json`、实际安装收据和完整的 Torch 证明。经过两次重新哈希的
`complete-evidence-seal-v2.json` 绑定 **275 个文件 / 412,397,540 字节**，SHA256
`62a8d264feafcf836712b5c238ccd53d3e56d3857cf92a6a807d1b47f37d463c`；已安装环境、
上游克隆和编译缓存不包括在内。

这些是维护版 profile 的全新有界 CPU 训练结果。它们不新增 CUDA 运行或成对的公共评分卡，
不证明 GitHub 的计划已执行，不确立性能结论，也不暗示整库支持。[更早的 CPU/CUDA 源码
绑定测量](2026-10-15-openmmlab-installed-source-training.md)保留其自身的源码身份与范围。
