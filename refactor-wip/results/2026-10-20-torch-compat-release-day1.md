# Torch 兼容发布首日候选与常用接口验收

- Status: 首日功能与维护门禁通过，候选正在按字节冻结；本报告不构成完整发布通过。
- Date: 2026-10-20
- Baseline: `integrate/cgq-transformers-native-multicard-target@2690b03e3ff489134c093eaf5c4a8fd2af53fd5c` 加保留工作树。
- Remote sync: 已 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为 HEAD 祖先，无需 merge；未整合 `origin/2.0-refactor`。
- Candidate: 待冻结；主工作树和原始未提交改动保持原位。
- Owner: cgq / Jittor Torch 兼容层维护者。
- Review when: 候选源码、依赖、device/RNG、Tensor assignment、加载或 optimizer 实现变化；扩展为真实训练、公开 HF DDP、性能或安装后的声明。

## 首日目标与输入

本轮执行[一周发布计划](../testing/2026-10-20-torch-compat-release-gate.md)的第一天：盘点保留的本地修复、形成可复现源码候选，按 verify-then-fix 收口日常接口及现有 core/structure 失败。开始时的 70 个未提交路径全部保留，已保存原 binary diff、空 index、路径状态和 SHA256；没有 stash、clean 或删除旧工作。本次新增修复不能被解释为原有所有改动均获得重新验收。

候选输入保留已有 native DDP readiness/NCCL/launcher 与 optimizer 依赖、低精度优化器、CUDA embedding/linear-bias kernel 及注册、AMP/GradScaler、HF 生命周期与 trajectory harness、BF16 参数序列化、Dataset 分片和 worker 异常传播等工作。保留的 `tests/optim/test_optimizer_save_load.py` 容差修改不是本轮新增；本轮没有用它证明旧实现与 PyTorch 的完整训练轨迹一致。原有四份根目录库开发文档和未跟踪的 library-docs skill 保留在主树，不属于运行候选输入。

## 环境与证据位置

两侧均 Python 3.11.3，Transformers 4.56.2、Accelerate 1.10.1、PEFT 0.17.1、torchdata 0.11.0、NumPy 1.26.4、pytest 7.4.4、safetensors 0.6.2、tokenizers 0.22.1；独立 oracle 为真实 PyTorch 2.6.0+cu124，检查其没有 shim install context。完整包版本和两份 pip freeze 的 SHA256 见[环境快照](../testing/2026-10-20-torch-compat-release-environment.json)。现有虚拟环境的精确版本已记录；clean install、完整可下载 wheel hash 锁和安装包验收仍属后续门槛。

CUDA 为共享 A800、sm_80、nvcc 12.2.140、GCC 11.4.0，显式选择一个 GPU UUID。正确性测试启动前复查 GPU/进程；空闲显存约 59 GiB，其他用户的推理任务仍在运行。本轮使用 strict CUDA math、串行算子编译、MKL/MPI/CUTT 关闭、NCCL 开启。共享负载下的墙钟仅是验证耗时，没有速度或显存竞争力结论。

未版本化证据统一位于 `$JITTOR_LAB_ROOT/_state/release-day1/20261020-r1/`，包括 preflight、原工作树 patch、两解释器 freeze/version、修前修后 JSON、原始测试日志、GPU 快照、运行脚本与候选文件 hash。CPU correctness、suite 两种模式和 CUDA 使用各自已隔离缓存，所有 Jittor 首次编译串行。初次 core 因两套缓存冷编译触及 15 分钟边界而退出 124，作为 setup 超时保留，不能计作行为失败。

## 已复现与修复的行为

| 问题 | 修前证据 | 本轮实现与验收意图 |
| --- | --- | --- |
| CUDA `load_state_dict(assign=False)` 令两个模型共用可写存储 | 修前 CUDA probe：仅更新 source 的 fused AdamW 同时改动 target；PyTorch 两设备独立 | 按目标 dtype/device 转换后，使用真正复制存储的 frontend clone；保留 Parameter/buffer 对象身份，测试特意到 source 更新后才首次读取 target |
| `no_grad/copy_` 丢失目标叶子梯度 | CPU/CUDA probe 均复现；独立 PyTorch 通过 | no_grad 时隔离 source graph，复制到独立存储，再恢复目标原有 trainable/冻结状态；覆盖两侧 requires_grad 四种组合及 source 梯度隔离 |
| SGD 普通 Torch 参数不兼容 | `foreach=False/fused=False` 修前两设备均 TypeError | 对齐默认 lr 与 positional dampening/weight_decay 次序，映射 native fused，保留 group metadata；常用 momentum 两步轨迹由双解释器检验 |
| 非零 dampening 和参数组 maximize 静默错算 | JT 首步分别为 0.9/0.8，PT 为 0.8/1.2 | 非零 dampening、foreach=True、maximize=True、differentiable=True 明确拒绝；构造与每次 step 校验有效 group，防加载或修改 group 后静默执行另一种更新 |
| Linear 显式 super anchor、boolean RHS 梯度、arange dtype/精度 | 当前输入上旧四个 core node 最小复验 4 failed | initializer bridge 保留 native execute/super 链；浮点索引目标连接可微 RHS，异常后回滚；arange 保留浮点 Tensor/NumPy step，整数默认 int64，计算前选精度，native 默认 dtype 保留 |
| deepcopy 仍保留旧 `.data` owner、`+=` 等绕过兼容原位逻辑 | 完整 core 到达原 owner 断言失败；已有 deepcopy/RHS 梯度回归复现 | 仅清副本的 view write-through metadata，不创建额外 state；四种 augmented assignment 统一共享写入口，检查 leaf/dtype/shape，再保留 holder 与 RHS graph |
| 初始化意外开启冻结梯度、empty 快路径默认可微 | normal_/view 原断言失败；empty 最小 probe 显示默认 requires_grad=True | 隔离原生常量初始化 graph，恢复目标状态；empty 快路径显式应用 Torch 默认 requires_grad=False，保留显式 True |
| Tensor initializer 忽略显式 generator | 原 uniform 同 seed 回归失败；浮点 factory 路径的 restore probe 也暴露边界 | CPU uniform 复用 native seed/counter owner并验证 round-trip；normal 与 CUDA uniform 的显式 generator 在写入前明确拒绝。旧 uniform 测试的两整数编码断言更新为当前版本化 payload 对应的 native counter，同时保留数值断言并新增 replay |

`types.py` 的设备参数校验按职责移到独立 `device_validation.py`，optimizer 默认初始化与 SGD 参数校验归到 `optim_frontend.py`；没有提高 800 行结构预算。onboarding 文档补可达入口，生成 MANIFEST 补相应文档及现有 Stateful Trainer 测试，USER_CHECK 台账与 dirty GradReadyCallback 返回值约束对齐，负向 callback 测试仍验证同进程恢复。

## 验收结果

| 门禁 | 最终结果 | 原始日志 |
| --- | --- | --- |
| CPU core，native + Torch 两种模式 | **328 passed、46 skipped、1 xfailed、0 failed**；native 101 passed，Torch 227 passed | `core-final.log` |
| 完整 structure | **1398 passed、2 skipped、0 failed**，163.58 s；skip 为 accelerator 类 | `structure-final.log` |
| CPU 日常状态 + sparse/deepcopy + optim | **57 passed、0 skipped** | `daily-cpu-final.log` |
| 独立 PyTorch 日常状态 | CPU **24 passed**，CUDA **24 passed** | `daily-oracle-{cpu,cuda}-final2.log` |
| Jittor CUDA 日常状态与扩展选择集 | **100 passed、1 failed**；失败为旧 optim helper 的默认 CPU placement。helper 修正后 optim 全文件 **27 passed**，没有普通失败残留 | `daily-cuda-final.log`、`optim-cuda-placement-final.log` |
| CUDA mixed RNG / legacy 原子性 | 两个 node 通过；与首轮 17 项接口合跑 **19 passed** | `daily-rng-cuda-r1.log` |
| Callback 用户错误与同进程恢复 | CPU 定向 **3 passed、7 deselected**；native CUDA probe 的负向与恢复均通过，实际 cuda:0、fallback=0 | `callback-cpu.log`、`callback-cuda.log` |
| 日常三项修后最小 probe | CPU/CUDA 均 **3/3 pass**，实际 device 与声明一致，fallback=0；PyTorch 修前即均通过 | `after-jittor-{cpu,cuda}.json` |
| 布局、MANIFEST 生成一致性、diff 格式与受影响相对链接 | 通过 | 最终静态检查记录 |

core 的 46 项 skip 中，native 为 2 insufficient-devices、34 accelerator、7 opt-in；Torch 为 3 accelerator。跳过项与既有 xfail 不计作通过。CUDA 扩展的 101 个 node 含仅在 CPU 执行的既有 autograd/serialization 用例，不将整个选择集冒称全部在 CUDA 运算；其中新 daily 24 项有显式设备与 fallback 守卫，arange 新用例也断言目标 device。各集合重叠，不把上述数字相加为唯一用例数。

优化器旧 `both_devices` helper 仅切换 native flag，Torch 工厂仍默认 CPU，导致旧 CUDA placement 断言失败；本轮显式设置并恢复 Torch default device 后重新验证。没有删 placement 断言。

新 `test_daily_state_contracts.py` 在两个解释器运行同一文件。Jittor fixture 禁止 backend fallback，核对张量 native dispatch placement；CUDA 结束同步并要求 fallback 增量为 0。两项显式 generator 组合的 fail-closed 检查只证明拒绝时不改变目标、generator 和默认 RNG，不计作支持该组合。其余 legacy 测试的 skip 按实际环境分类报告。

## 复现入口

使用环境快照中的独立解释器与仓库环境隔离规则，先配置 `JITTOR_LAB_ROOT`、`JITTOR_HOME`、`TMPDIR`、源码与 shim 路径，离线固定依赖。CPU 必须显式禁用 CUDA；CUDA 必须指定当前可用设备并确认真实执行。原始 `run.sh` 保存本机确切设置，源码回归入口如下：

```bash
python tools/run_test_suite.py --tier core --serial-compile
python -m pytest -q compat/tests/torch/test_daily_state_contracts.py
python -m pytest -q compat/tests/torch/test_sparse_dense_add.py
python -m pytest -q compat/tests/torch/test_torch_compat_optim.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
bash tools/check_repo_layout.sh
python tools/build/generate_manifest.py --check
```

独立 PyTorch 执行 daily 文件时使用 `--confcutdir=compat/tests/torch`，避免收集 shim 专有 conftest；设置 `JITTOR_TEST_DEVICES=cpu` 或 `cuda`。CUDA 定向还包括 independent frontend、autograd semantics、serialization owners、constructor dtype 与 fractional Tensor arange；RNG mixed history/legacy atomicity 用当前两个明确 node 验证。

## 发布边界与下一天输入

本轮不关闭默认 CUDA 并行编译、历史 random SIGSEGV、卷积 liveness 或长期稳定性问题，不把串行短跑等同默认安装通过。显式 initializer generator 的支持仅 CPU uniform；非零 SGD dampening 与高级更新选项仍不支持。generic floating factories 的 NumPy generator stream 与 native seed/counter state 的完整保存协议也需要另行收口，不能用本轮 CPU uniform 结果外推。

后续按计划完成真实分类/LoRA/Seq2Seq/generation、多 epoch/尾批/AMP/普通恢复，公开 Torch/HF 单机 DDP 2/4 rank，与 PyTorch 配对的速度/峰值显存，以及 clean wheel、默认配置和稳定性验收。历史原生 DDP 或显式 Stateful Trainer 的通过范围仍各自绑定原提交，不能替代这些门槛。本轮没有做完整 smoke/full、发布包安装、性能测量或 push。
