# Diffusers：官方安装与维护组件重放

- 状态：受限 CPU/CUDA L3 测量已由当前共享审计器接受；更广的库覆盖与托管 nightly 验收仍未关闭。
- 复查日期：2026-10-16。
- 测量基线：`b2127fd6a4a0c080ea19c45fbeae4b0b3251b74f`。
- 实现远端基线：`00ceb26fd395707fb33649cc43709af045110a9a`。
- 维护者：Torch 兼容与下游验证维护者。
- 复查条件：官方源码/wheel 固定版本、源码/receipt 守卫、builder、精度/设备控制、请求的梯度或训练/比较协议变化。

当前严格审计器实际上拒绝了原先的 `1a92aae04f4c70fa44e251f8fafdc7e2cd77beda` CPU/CUDA 两条记录。它们的 697 个包文件哈希与保留的本地源码构建 wheel 匹配，但 worker 没有捕获其 wheel SHA 或实际 METADATA/RECORD。本批保留这些数值观测与 JSON 字节；它通过真实执行获取新证据。

新 profile 使用官方 PyPI Diffusers **0.35.1**，wheel SHA256 `fe29ff10200970c7c5934c6488c213e2a77a03dad5e6fa00bbd8e1d04234cb0e`，以及干净官方源码 `0f252be0ed42006c125ef4429156cb13ae6c1d60`。这与历史的本地源码构建构件 `fe763b8bb4381794f64c7a33ffdd29b3e63266c0013194317521331b3a15bdeb` 不同。

全部 697 个官方包载荷文件都与受跟踪的 Git 内容匹配。唯一允许的源码省略是 `experimental/`、`models/`、`pipelines/`、`pipelines/deprecated/`、`pipelines/stable_diffusion/` 与 `schedulers/` 下的六个 README 页面。不允许任何生成的或未绑定的 Python/资源载荷。每个实际 worker 都绑定其加载的包、完整载荷、原始 METADATA 与**实际 pip RECORD**，包括包哈希与大小。没有恢复任何 archive RECORD。

两个设备都执行原来的 tiny FP32 `UNet2DModel`：sample `[2,3,16,16]`，timestep `[2]`，block channels `(16,32)`，每 block 一层，group 数为八，dropout 为零。要求完整的 144 个参数梯度与浮点 sample 梯度。base 种子 0 与输入扰动种子 7 使用同一份完整初始权重；三次 SGD 更新使用学习率 .001。每设备上，六个不同的真实 worker 产生 146 个 base 数组、146 个扰动数组与 **873 个训练字段**，合计 **1,165 对**。builder、runner、训练数学与既有 `.002 absolute OR .01 per-phase globally scaled` 比较预算不变；这不是 atol 加 rtol 的叠加规则。

| 设备 / 卡数 | Oracle | 原生 CUDA / ACL / replay | TF32 / HF32 | 对数 / 原一致性 | 最大绝对 / 缩放误差 | 时间 / 比值 / 进程内存 | 等级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU / 0 | 真实 Torch 2.11.0+cpu | 0 / 0 / 默认, 实际 replay=1 | off / off | 1,165 通过, .002 / .01 | 4.95911e-5 / 9.40814e-7 | 未 benchmark / 未采样 | L3 |
| CUDA / 1 | 真实 Torch 2.11.0+cu128 | 1 / 0 / 默认, 实际 replay=1 | off / off | 1,165 通过, .002 / .01 | 3.81470e-5 / 9.80451e-7 | 未 benchmark / 未采样 | L3 |

实际平台：Linux 6.1.188/glibc 2.34，AMD EPYC 9554，GCC 11.5.0；CUDA 使用一张 RTX 4090，驱动 580.178.04，NVCC 12.2.140。candidate provider 为单独安装的 CUDA 12.2.140、cuDNN 8.9.7.29 与 cuBLAS 12.2.5.6；oracle Torch 使用其完整的常规 CUDA 12.8 依赖解析，包括 cuDNN 9.19.0.56 与 cuBLAS 12.8.4.1。完整 provider 版本与实际物理 UUID/可见映射保留在每个 worker 中。CPU 显式隐藏 CUDA；不可用的加速器字段为 `not-applicable`。两侧都使用单线程、TF32/benchmark 关闭、最高 matmul 精度、固定离线输入与分离的运行时路径。candidate 观测要求原生 dispatch/驻留，且 CUDA 指针必须属于已分配物理卡上的逻辑设备零。每个 candidate 都报告 `fallback_policy=error,count=0`。

独立 NumPy 重放校验了全部 **2,330 对**、两个运行时/设备上的 **2,900 个非零梯度数组**与 **1,728 次 SGD 更新**。最大更新重建误差在 CPU 上为 3.72529e-9，在 CUDA 上为零。全部十二个 worker 消费同一个初始权重归档 SHA256 `b303c894e752f54c296757bd0b56fc5c2cc7e29848923ad267873a918b93c7a0`。每个设备有 2,618 条实际张量观测，包括初始化与训练。这些是训练/正确性结果，不含性能或整进程内存验收。

真实 oracle 的 `_C` 二进制在实际 worker 中哈希，并与其官方构件匹配。独立审计另外在完整安装前缀上校验除 RECORD 以外的每个原始官方 wheel 文件：每个 CPU Torch 安装与 CUDA candidate 的仅 metadata CPU Torch distribution 各 **11,945** 个文件，CUDA oracle **11,841** 个，每个 Diffusers 安装 **702** 个，包括其 697 个包文件与五个 distribution 文件。每个原始载荷行都与未改动的实际 pip RECORD 匹配。安装生成的脚本/metadata/字节码不被称为原始归档字节。执行后的重新哈希保持相同。candidate 运行 shim/native 框架；其已安装的真实 Torch distribution 提供依赖 metadata，而非其数值 provider。candidate 身份使用实际原生导入路径、源码与 dispatch/device 观测；**这些 worker 内没有捕获任何 candidate JIT 二进制哈希**。

完整测量 Git 归档有 **2,915 个受跟踪文件**，含模式与符号链接目标文本。全部 **1,612 个选定 worker 源码哈希**都与测量的已提交字节匹配；该选定映射不是穷尽的导入追踪。完整归档与 Git bundle 都保留原始测量身份。结果与源码哈希都没有被重新标记到后续集成提交。

全新干净上游枚举不变：`model_api=87`，pipeline 族轴 `model_family=79`、`pipeline_class=249`、`auto_family=37`、`task=3`。当前共同准入为 **两个接受 / 零拒绝 / 零阻塞**。每个设备只计入 `UNet2DModel`，`model_api` 上 **1/87**；其他所有轴仍为零。导出分母包含 Flax、helper、mixin、enum 与函数，而非 87 个模型族。从这个 UNet 组件不能推出完整的生成 pipeline。

使用干净已提交源码、完整单独安装的运行时与固定的官方 checkout/wheel 复现：

```bash
python tools/run_diffusers_gate.py --candidate-python <private-candidate-python> \
  --oracle-python <private-binary-torch-python> --device cpu \
  --checkout <clean-official-diffusers-checkout> --wheel <fixed-official-wheel> \
  --output "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/cpu" \
  --cache-root "$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/<run>/benchmark-cpu"
```

先跑 CPU 再跑 CUDA。对 CUDA，显式分配一张健康卡，选择每个私有运行时的实际 provider 库，设置 `CUDA_VISIBLE_DEVICES=<allocated-card>` 与真实的 `JT_BUILD_NVCC_PATH`，然后用 `--device cuda` 与全新的输出/缓存根重跑同一命令。被测量调用的可执行包装只选择各前缀实际的共享库；进程 metadata 仍记录底层真实解释器与 provider。首次 JIT 串行执行；单测与 benchmark 缓存相互独立。门禁不提供缩小 case/梯度/步数的选项。它复用维护 collector 与实际共同审计器，并且只在源码清洁检查与完整准入之后才输出记录。

九个真实 packet 的伪造副本被拒绝：缺少官方 wheel SHA、distribution 版本被改、METADATA/RECORD 摘要被改、未绑定的非 Python 声明、已提交源码哈希被改、缺少 HOME、缺少参数梯度、缺少第三步参数。原始 packet/安装保持只读。第一次负例审计尝试错误地期望从单 packet 校验器得到进程内 HOME/TMPDIR 不等式；该诊断失败被保留，修正后的测试检查所需的 HOME 字段。没有削弱任何守卫。host 回归还覆盖了自洽的 wheel/资源/metadata 篡改与默认回调行为。既有的 Metrics/TensorDict `912` 四条在默认校验器下仍然准入；原 MM `7bb` 的 231 对、初始权重、84 次更新与 84 个动量状态原样重放，包括其 `mmcv-lite` 命名/receipt 契约。

全新的 candidate/oracle/controller 环境中，正常完整 pip 解析与 pip 检查通过。早前一次在线 CPU 装配在依赖获取阶段被显式终止，早于数值执行；其部分前缀与日志保留。首次离线解析缺少 `importlib_metadata`；官方依赖 wheel 经校验后完整离线解析成功。没有用 `--no-deps` 绕开下游/运行时 Requires-Dist 闭包；本地 editable 安装复用了该已检查闭包。没有使用任何安装站点或 JIT 缓存副本。CPU core 通过 **322 / 42 skipped / 一个既有 xfail**，扩展的定向 host 检查通过 **238 / 零 skip**。第一次扩展 host 运行保留了一次仅源码正例 fixture 的陈旧失败；其替代检查同时校验完整 wheel 接受与嵌套源码篡改拒绝。强制 `tests/structure` 起初记录 **1,514 passed / 四个 skip / 两个失败**，均为缺少 toctree/生成 manifest 登记。后来的窄集成补丁提供授权的三行；定向验证与交付 manifest 记录其结果，不改变测量源码身份。

未版本化证据位于 `$JITTOR_LAB_ROOT/_state/diffusers-installed-binding/20261016/`：`formal-{cpu,cuda}-v1/`、`formal-common-records-v1.json`、`formal-common-scorecard-v1.json`、独立的数组/安装审计、源码归档/bundle 与被保留的尝试。共同 scorecard SHA256：`a779a561e74ac1e8dd366cea4720b44807b5232ea61663192f77c3bf32763aa0`。测量封条 `measurement-evidence-seal-v1.json` 绑定 **188 个文件 / 179,082,533 字节**，SHA256 `d26745297c9c5c141ca2d6462b0ab2c86bd6bd559700326d555eeccf9e320f18`。后续验证/交付日志有自己的 manifest；本测量封条不被重写。

专用 CPU nightly 接线是单独的后续工作。本批不证明托管调度执行、完整 pipeline、其他 dtype、分布式训练、NPU/ROCm、L4 速度、整进程内存或 L5 等级。
